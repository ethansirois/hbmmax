#!/usr/bin/env python3
"""Build CUDA → PTX → cubin/SASS, inspect assembly, and sweep SM budgets."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from collapse_sass import collapse
from godbolt import render_snippet

ROOT = Path(__file__).resolve().parents[1]


def integer_expression(value: str | int) -> int:
    """Evaluate bounded integer arithmetic without eval(), names, or function calls."""
    text = str(value)
    if len(text) > 512:
        raise argparse.ArgumentTypeError("expression is too long")
    try:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("expression is too complex")

        def visit(node: ast.AST) -> int:
            if isinstance(node, ast.Constant) and type(node.value) is int:
                result = node.value
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): result = left + right
                elif isinstance(node.op, ast.Sub): result = left - right
                elif isinstance(node.op, ast.Mult): result = left * right
                elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
                    result, remainder = divmod(left, right)
                    if isinstance(node.op, ast.Div) and remainder:
                        raise ValueError("/ must produce an integer; use // for floor division")
                elif isinstance(node.op, ast.Mod): result = left % right
                elif isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift)):
                    if not 0 <= right <= 63:
                        raise ValueError("exponents and shifts must be between 0 and 63")
                    if isinstance(node.op, ast.Pow): result = left ** right
                    elif isinstance(node.op, ast.LShift): result = left << right
                    else: result = left >> right
                else: raise ValueError("unsupported arithmetic operator")
            else:
                raise ValueError("use integer arithmetic only")
            if not -(1 << 63) < result < (1 << 63):
                raise ValueError("value exceeds signed 64-bit range")
            return result

        return visit(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError) as error:
        raise argparse.ArgumentTypeError(f"invalid integer expression: {error}") from error


def positive(value: str | int) -> int:
    result = integer_expression(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value: str | int) -> int:
    result = integer_expression(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def sm_counts(value: str) -> list[int]:
    """Comma-separated counts and inclusive start:stop[:step] ranges."""
    counts = set()
    for part in value.split(","):
        pieces = part.strip().split(":")
        if len(pieces) == 1:
            counts.add(positive(pieces[0]))
        elif len(pieces) in (2, 3):
            start, stop = positive(pieces[0]), positive(pieces[1])
            step = positive(pieces[2]) if len(pieces) == 3 else 1
            if stop < start or (stop - start) // step > 1024:
                raise argparse.ArgumentTypeError("SM range must be ascending and at most 1024 points")
            counts.update(range(start, stop + 1, step))
        else:
            raise argparse.ArgumentTypeError("use counts like 1,2,4,8 or ranges like 1:32")
    if len(counts) > 1024:
        raise argparse.ArgumentTypeError("too many SM counts")
    return sorted(counts)


def architecture(value: str) -> str:
    value = value.removeprefix("sm_").removeprefix("compute_")
    if not re.fullmatch(r"[0-9]+[a-z]?", value):
        raise argparse.ArgumentTypeError("use an architecture such as sm_80, 80, or sm_90a")
    return value


def available(problem: str) -> dict[str, Path]:
    return {p.stem: p for p in sorted((ROOT / "kernels" / problem).glob("*/*.cu"))}


def source_hashes() -> dict[str, str]:
    paths = sorted(p for directory in ("bench", "kernels", "tools")
                   for p in (ROOT / directory).rglob("*")
                   if p.suffix in (".cu", ".cuh", ".hpp", ".py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def tool_info(command: str) -> dict[str, str]:
    executable = shutil.which(command)
    if executable is None:
        raise ValueError(f"tool not found: {command}")
    return {"path": executable, "version": subprocess.check_output(
        [executable, "--version"], text=True).strip()}


def stage(command: list[str], log, verbose: bool) -> None:
    display = shlex.join(command)
    log.write(display + "\n")
    log.flush()
    if verbose:
        print(display, file=sys.stderr, flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(result.stdout)
    log.flush()
    if result.returncode:
        raise RuntimeError(f"{result.stdout.strip()}\nBuild stage failed; see {log.name}")
    if result.stdout and verbose:
        print(result.stdout, file=sys.stderr, end="")


def build(args: argparse.Namespace, kernel: Path, sms: int) -> tuple[Path, dict]:
    tools = {name: tool_info(getattr(args, name)) for name in ("nvcc", "ptxas", "cuobjdump")}
    problem = ROOT / "kernels" / args.problem
    # Kernels include their family configuration first; share it with the host driver.
    config_header = re.search(r'^#include "([^"]+_config.cuh)"', kernel.read_text(), re.M)
    if config_header is None:
        raise ValueError(f"kernel must include a family configuration: {kernel}")
    definitions = [f'-DHBM_KERNEL_HEADER="{kernel.relative_to(ROOT)}"',
                   f'-DHBM_CONFIG_HEADER="{config_header[1]}"',
                   f'-DHBM_BASELINE_HEADER="{problem.relative_to(ROOT)}/baseline.cuh"',
                   f'-DHBM_PROBLEM_HEADER="{problem.relative_to(ROOT)}/problem.cuh"',
                   f"-DHBM_SMS={sms}", f"-DHBM_PEAK_SMS={args.peak_sms}",
                   f"-DHBM_THREADS={args.threads}"]
    if args.numel is not None:
        definitions.append(f"-DHBM_NUMEL={args.numel}LL")
    if args.main_loops is not None:
        definitions.append(f"-DHBM_MAIN_LOOPS={args.main_loops}")
    metadata = {
        "problem": args.problem, "kernel": str(kernel.relative_to(ROOT)), "tools": tools,
        "parameters": compile_parameters(args, sms),
        "definitions": definitions, "source_sha256": source_hashes(),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    output = ROOT / "build" / f"{args.problem}.{kernel.stem}" / digest
    required = ("bench", "kernel.ptx", "kernel.cubin", "kernel.sass", "kernel.gist.sass",
                "kernel.godbolt.cu", "manifest.json")
    if all((output / name).is_file() for name in required) and not args.rebuild:
        if args.verbose:
            print(f"Cached: {output.relative_to(ROOT)}", file=sys.stderr)
        return output, json.loads((output / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    (output / "kernel.godbolt.cu").write_text(render_snippet(ROOT, kernel, metadata["parameters"], tools))
    common = ["-std=c++17", f"-O{args.opt}", "-I", str(ROOT), *definitions]
    commands = [
        [tools["nvcc"]["path"], *common, f"-arch=compute_{args.arch}", "-lineinfo", "-src-in-ptx",
         "--ptx", str(ROOT / "bench/device.cu"), "-o", str(output / "kernel.ptx")],
        [tools["ptxas"]["path"], f"-arch=sm_{args.arch}", f"-O{args.opt}", "-lineinfo", "-v",
         str(output / "kernel.ptx"), "-o", str(output / "kernel.cubin")],
        [tools["nvcc"]["path"], *common, f"-arch=sm_{args.arch}",
         str(ROOT / "bench/main.cu"), "-lcuda", "-o", str(output / "bench")],
    ]
    with (output / "build.log").open("w") as log:
        for command in commands:
            stage(command, log, args.verbose)
    raw = subprocess.check_output([tools["cuobjdump"]["path"], "--dump-sass",
                                   str(output / "kernel.cubin")], text=True)
    (output / "kernel.sass").write_text(raw)
    (output / "kernel.gist.sass").write_text(collapse(raw))
    for name in metadata["source_sha256"]:
        snapshot = output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, snapshot)
    metadata["commands"] = commands
    metadata["artifacts_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                    for name in required if name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata


def compile_parameters(args: argparse.Namespace, sms: int) -> dict:
    return {"arch": f"sm_{args.arch}", "opt": args.opt, "sms": sms,
            "peak_sms": args.peak_sms, "numel": args.numel,
            "threads": args.threads, "main_loops": args.main_loops}


def settings_name(settings: dict) -> str:
    count = settings["numel"] if settings["numel"] is not None else "auto"
    return (f"{settings['arch'].replace('sm_', 'sm')}-O{settings['opt']}"
            f"-sms{settings['sms']}-peak{settings['peak_sms']}-n{count}"
            f"-t{settings['threads']}-loops{settings['main_loops']}")


def link(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(os.path.relpath(target, destination.parent))


def inspect(args: argparse.Namespace, kernel: Path, output: Path, metadata: dict) -> Path:
    settings = metadata["parameters"]
    view = ROOT / "build/inspect" / settings_name(settings)
    name = f"{args.problem}.{kernel.stem}"
    for directory, suffix, target in (
        ("source", ".cu", kernel),
        ("built-source", ".cu", output / "source" / kernel.relative_to(ROOT)),
        ("godbolt", ".cu", output / "kernel.godbolt.cu"),
        ("ptx", ".ptx", output / "kernel.ptx"),
        ("sass", ".sass", output / "kernel.sass"),
        ("sass", ".gist.sass", output / "kernel.gist.sass"),
        ("manifests", ".json", output / "manifest.json"),
    ):
        link(target, view / directory / (name + suffix))
    rows = ["# Kernel comparisons", "", "Source opens the editable file; built source records the inspected build.",
            "PTX/SASS contain small_SMs and the large_SMs baseline. The benchmark loads this exact cubin.", "",
            "| Kernel | Source | Built source | Godbolt snippet | PTX | SASS | Collapsed SASS | Build info |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in sorted((view / "manifests").glob("*.json")):
        key = entry.stem
        snippet = f"[snippet](godbolt/{key}.cu)" if (view / "godbolt" / f"{key}.cu").exists() else "—"
        rows.append(f"| {key} | [source](source/{key}.cu) | [built](built-source/{key}.cu) | {snippet} | "
                    f"[PTX](ptx/{key}.ptx) | [SASS](sass/{key}.sass) | "
                    f"[collapsed](sass/{key}.gist.sass) | [manifest](manifests/{key}.json) |")
    (view / "README.md").write_text("\n".join(rows) + "\n")
    return view / "README.md"


def measure(args: argparse.Namespace, output: Path, mode: str | None = None) -> dict:
    report = output / "run.json"
    report.unlink(missing_ok=True)
    command = [str(output / "bench"), "--device", str(args.device), "--warmup", str(args.warmup),
               "--iterations", str(args.iterations), "--json", str(report)]
    if mode:
        command.append(f"--{mode}-only")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "run.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    data = json.loads(report.read_text())
    if not data["valid"]:
        raise RuntimeError("Kernel failed validation")
    if args.verbose:
        print(result.stdout, file=sys.stderr, end="")
    return data


def sweep(args: argparse.Namespace, kernel: Path) -> dict:
    points = []
    # A baseline run at the first point establishes one reference for the whole sweep.
    first_output, _ = build(args, kernel, args.sweep_sms[0])
    baseline = measure(args, first_output, "peak")
    peak = baseline["peak_gbps"]
    if not math.isfinite(peak) or peak <= 0:
        raise RuntimeError("Invalid peak bandwidth measurement")
    minimum = None
    for sms in args.sweep_sms:
        if sms > baseline["available_sms"]:
            raise ValueError(f"SM budget {sms} exceeds the device's {baseline['available_sms']} SMs")
        output, metadata = build(args, kernel, sms)
        inspect(args, kernel, output, metadata)
        data = measure(args, output, "kernel")
        if data["numel"] != baseline["numel"]:
            raise RuntimeError("Sweep workload changed; specify a fixed --numel")
        bandwidth = data["kernel_gbps"]
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise RuntimeError("Invalid kernel bandwidth measurement")
        points.append({"sms": sms, "gbps": bandwidth, "fraction_of_peak": bandwidth / peak,
                       "build": str(output.relative_to(ROOT)), "measurement": data})
        if bandwidth >= peak * (1 - args.within_percent / 100):
            minimum = sms
            break
    return {"kernel": kernel.stem, "minimum_sms": minimum, "peak_gbps": peak,
            "within_percent": args.within_percent, "baseline": baseline, "points": points}


def save_sweeps(args: argparse.Namespace, results: list[dict]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.output or ROOT / "results" / f"sweep-{timestamp}.json"
    if path.suffix != ".json":
        raise ValueError("--output must be a .json path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"timestamp": timestamp, "problem": args.problem,
                               "config": {"arch": f"sm_{args.arch}", "opt": args.opt,
                                          "numel": args.numel, "peak_sms": args.peak_sms,
                                          "threads": args.threads, "tested_candidates": args.sweep_sms},
                               "results": results}, indent=2) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kernel", "sms", "GB/s", "peak_GB/s", "fraction_of_peak", "minimum_sms"))
        for result in results:
            for point in result["points"]:
                writer.writerow((result["kernel"], point["sms"], point["gbps"], result["peak_gbps"],
                                 point["fraction_of_peak"], result["minimum_sms"]))
    return path


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "config.json")
    early, _ = pre.parse_known_args()
    try:
        defaults = json.loads(early.config.read_text()) if early.config.exists() else {}
        if not early.config.exists() and early.config != ROOT / "config.json":
            raise ValueError(f"configuration file does not exist: {early.config}")
        if not isinstance(defaults, dict):
            raise ValueError("configuration must be a JSON object")
    except (OSError, ValueError) as error:
        pre.error(str(error))
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("action", choices=("list", "build", "run", "inspect", "sweep", "snippet"))
    parser.add_argument("--problem", default="copy")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--kernel", help="kernel name, e.g. k0 or k3_1")
    selection.add_argument("--kernels", help="comma-separated names; default: all for build/inspect/sweep/snippet")
    parser.add_argument("--arch", "-arch", type=architecture, default="80")
    parser.add_argument("-O", "--opt", type=int, choices=range(4), default=3)
    parser.add_argument("--sms", type=positive, default=4)
    parser.add_argument("--peak-sms", type=positive, default=32)
    parser.add_argument("--sweep-sms", type=sm_counts, default="1:32")
    parser.add_argument("--within-percent", type=float, default=3.0)
    parser.add_argument("--numel", type=positive, help='integer or arithmetic, e.g. "1024 * 1024 * 2 + 1"')
    parser.add_argument("--threads", type=positive, default=256)
    parser.add_argument("--main-loops", type=nonnegative, help="derive k3 numel from a pipeline loop count")
    parser.add_argument("--device", type=nonnegative, default=0)
    parser.add_argument("--warmup", type=nonnegative, default=5)
    parser.add_argument("--iterations", type=positive, default=100)
    for tool, variable in (("nvcc", "NVCC"), ("ptxas", "PTXAS"), ("cuobjdump", "CUOBJDUMP")):
        parser.add_argument(f"--{tool}", default=os.environ.get(variable, tool))
    parser.add_argument("--output", type=Path, help="sweep JSON path; also writes CSV")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    supported = {action.dest for action in parser._actions}
    if set(defaults) - supported:
        parser.error(f"unknown config keys: {sorted(set(defaults) - supported)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    try:
        args.arch = architecture(args.arch)
        for key in ("sms", "peak_sms", "threads", "iterations"):
            setattr(args, key, positive(getattr(args, key)))
        for key in ("device", "warmup"):
            setattr(args, key, nonnegative(getattr(args, key)))
        if args.numel is not None: args.numel = positive(args.numel)
        if args.main_loops is not None: args.main_loops = nonnegative(args.main_loops)
        if isinstance(args.sweep_sms, str): args.sweep_sms = sm_counts(args.sweep_sms)
        args.within_percent = float(args.within_percent)
        if args.opt not in range(4) or not math.isfinite(args.within_percent) or not 0 <= args.within_percent < 100:
            raise ValueError("optimization must be 0–3 and within-percent must be 0–99.999")
        if not re.fullmatch(r"[A-Za-z0-9_]+", args.problem): raise ValueError("invalid problem name")
        if args.main_loops is not None and args.numel is not None:
            # --main-loops deliberately overrides the file's default numel; explicit CLI conflict is an error.
            if "--numel" in sys.argv:
                raise ValueError("choose --numel or --main-loops")
            args.numel = None
        if args.action == "sweep" and args.main_loops is not None:
            raise ValueError("sweep requires a fixed --numel, not --main-loops")
        if args.action != "list" and args.numel is None and args.main_loops is None:
            args.numel = 1 << 24
        kernels = available(args.problem)
        if args.action == "list":
            for name, path in kernels.items(): print(f"{name:12} {path.relative_to(ROOT)}")
            return 0
        if not kernels: raise ValueError(f"no implemented kernels for {args.problem}")
        if args.action == "run" and not (args.kernel or args.kernels):
            raise ValueError("run requires --kernel or --kernels")
        names = (args.kernel or args.kernels or ",".join(kernels)).split(",")
        for name in names:
            if name not in kernels: raise ValueError(f"unknown kernel {name!r}; use list")
        if args.action == "snippet":
            settings = compile_parameters(args, args.sms)
            directory = ROOT / "build/snippets" / settings_name(settings)
            directory.mkdir(parents=True, exist_ok=True)
            tools = {name: tool_info(getattr(args, name)) for name in ("nvcc", "ptxas")
                     if shutil.which(getattr(args, name))}
            for name in names:
                path = directory / f"{args.problem}.{name}.cu"
                path.write_text(render_snippet(ROOT, kernels[name], settings, tools))
                print(f"{name}: {path.relative_to(ROOT)}")
        elif args.action == "sweep":
            results = []
            if args.output is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                args.output = ROOT / "results" / f"sweep-{timestamp}.json"
            if args.output.suffix != ".json":
                raise ValueError("--output must be a .json path")
            for name in names:
                result = sweep(args, kernels[name])
                results.append(result)
                save_sweeps(args, results)
                minimum = result["minimum_sms"]
                print(f"{name}: {minimum} SMs" if minimum is not None else f"{name}: target not reached")
                for point in result["points"]:
                    if minimum is None or point["sms"] < minimum:
                        print(f"  {point['sms']} SMs: {point['gbps']:.3f} GB/s")
            print(f"Saved: {save_sweeps(args, results)}")
        else:
            for name in names:
                output, metadata = build(args, kernels[name], args.sms)
                view = inspect(args, kernels[name], output, metadata)
                if args.action == "run":
                    print(f"Godbolt: {(view.parent / 'godbolt' / f'{args.problem}.{name}.cu').relative_to(ROOT)}",
                          flush=True)
                    data = measure(args, output)
                    print(f"{name}: {data['kernel_gbps']:.3f} GB/s; peak: {data['peak_gbps']:.3f} GB/s")
                else:
                    print(f"{name}: {view.relative_to(ROOT)}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, argparse.ArgumentTypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
