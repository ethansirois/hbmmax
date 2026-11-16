"""Create standalone Compiler Explorer inputs from kernels and their local headers."""
from pathlib import Path
import re


def render_snippet(root: Path, kernel: Path, settings: dict, tools: dict | None = None) -> str:
    root, kernel = root.resolve(), kernel.resolve()
    visited: set[Path] = set()

    def inline(path: Path) -> str:
        path = path.resolve()
        if path in visited:
            return ""
        visited.add(path)
        lines = [f"// From {path.relative_to(root)}"]
        for line in path.read_text().splitlines():
            if line.strip() == "#pragma once":
                continue
            include = re.fullmatch(r'\s*#\s*include\s+"([^"]+)"\s*', line)
            if include:
                candidates = (path.parent / include[1], root / include[1])
                dependency = next((p for p in candidates if p.is_file()), None)
                if dependency is None:
                    raise ValueError(f"Cannot make a standalone snippet: missing {include[1]} in {path}")
                if not dependency.resolve().is_relative_to(root):
                    raise ValueError(f"Snippet dependency is outside the project: {dependency}")
                lines.append(inline(dependency))
            else:
                lines.append(line)
        return "\n".join(lines).strip() + "\n"

    flags = (f"-std=c++17 -O{settings['opt']} -arch={settings['arch']} "
             f"-Xptxas=-O{settings['opt']} -lineinfo")
    header = ["// Compiler Explorer (godbolt.org): select CUDA C++ and NVCC.",
              f"// Compiler options: {flags}",
              "// Open the Device pane and select SASS; PTX is also available.",
              "// Compilation only: no main() or execution is needed.",
              "// Match local NVCC and ptxas versions for assembly comparisons."]
    for name in ("nvcc", "ptxas"):
        version = (tools or {}).get(name, {}).get("version", "")
        if version:
            summary = next((line.strip() for line in version.splitlines() if "release" in line),
                           version.splitlines()[-1].strip())
            header.append(f"// Local {name}: {summary}")
    header.extend([f"// Kernel: {kernel.relative_to(root)}", "",
                   "// Settings for this build; edit here to explore other configurations.",
                   f"#define HBM_SMS {settings['sms']}",
                   f"#define HBM_PEAK_SMS {settings['peak_sms']}",
                   f"#define HBM_THREADS {settings['threads']}"])
    if settings.get("numel") is not None:
        header.append(f"#define HBM_NUMEL {settings['numel']}LL")
    if settings.get("main_loops") is not None:
        header.append(f"#define HBM_MAIN_LOOPS {settings['main_loops']}")
    return "\n".join(header) + "\n\n" + inline(kernel)
