#!/usr/bin/env python3
"""Collapse consecutive SASS instructions of the same opcode.

Keeps the first and last instruction of each run, and replaces the
middle copies with a one-line ellipsis. Also strips the trailing
encoding comments (/* 0x... */).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENCODING_COMMENT_RE = re.compile(r"/\*\s*0x[0-9a-fA-F]+\s*\*/")
INSTR_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"/\*(?P<addr>[0-9a-fA-F]+)\*/"
    r"(?P<gap>\s+)"
    r"(?P<body>"
    r"(?P<pred>@!?P\S+\s+)?"
    r"(?P<opcode>[A-Z][A-Z0-9_.]*)"
    r".*)"
    r"$"
)


def strip_encoding(line: str) -> str:
    return ENCODING_COMMENT_RE.sub("", line).rstrip()


def parse_lines(text: str) -> list[tuple[str | None, str]]:
    """Return (opcode or None, cleaned line) for each kept source line."""
    parsed: list[tuple[str | None, str]] = []
    for raw in text.splitlines():
        had_encoding = bool(ENCODING_COMMENT_RE.search(raw))
        cleaned = strip_encoding(raw)
        if not cleaned.strip():
            if had_encoding:
                continue
            parsed.append((None, ""))
            continue
        match = INSTR_RE.match(cleaned)
        if match:
            parsed.append((match.group("opcode"), cleaned))
        else:
            parsed.append((None, cleaned))
    return parsed


def emit_run(lines: list[str], opcode: str) -> list[str]:
    if len(lines) <= 2:
        return list(lines)

    omitted = len(lines) - 2
    first = INSTR_RE.match(lines[0])
    if first:
        prefix_len = (
            len(first.group("indent"))
            + len(first.group("addr"))
            + 4  # /* */
            + len(first.group("gap"))
        )
    else:
        prefix_len = 8
    ellipsis = f"{' ' * prefix_len}... another {omitted} {opcode} ..."
    return [lines[0], ellipsis, lines[-1]]


def collapse(text: str) -> str:
    parsed = parse_lines(text)
    out: list[str] = []
    run_opcode: str | None = None
    run_lines: list[str] = []

    def flush() -> None:
        nonlocal run_opcode, run_lines
        if run_opcode is None:
            return
        out.extend(emit_run(run_lines, run_opcode))
        run_opcode = None
        run_lines = []

    for opcode, line in parsed:
        if opcode is None:
            flush()
            out.append(line)
            continue
        if opcode == run_opcode:
            run_lines.append(line)
            continue
        flush()
        run_opcode = opcode
        run_lines = [line]
    flush()
    return "\n".join(out) + ("\n" if out else "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collapse repeated SASS opcodes, keeping the first and last of each run."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="SASS dump to read (default: stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write collapsed SASS here (default: stdout)",
    )
    args = parser.parse_args()

    text = args.input.read_text() if args.input else sys.stdin.read()
    collapsed = collapse(text)

    if args.output:
        args.output.write_text(collapsed)
        return 0
    try:
        sys.stdout.write(collapsed)
    except BrokenPipeError:
        sys.stdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
