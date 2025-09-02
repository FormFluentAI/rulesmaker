#!/usr/bin/env python3
"""
Simple Markdown fixer for:
- MD009: Trailing spaces (preserve intentional double-space line breaks)
- MD032: Blank lines around lists (ensure blank line before/after list blocks)

Usage:
  python scripts/fix_markdown.py <file1.md> [file2.md ...]
"""

from __future__ import annotations

import sys
import re
from pathlib import Path


def fix_md009(line: str) -> str:
    # Preserve exactly two spaces at EOL (hard line break), but strip any other trailing whitespace
    if line.endswith("  \n"):
        # Keep the two spaces and newline
        return line
    # Strip all trailing whitespace, preserve newline
    if line.endswith("\n"):
        core = line[:-1]
        core = re.sub(r"[\t ]+$", "", core)
        return core + "\n"
    # No newline: just strip trailing whitespace
    return re.sub(r"[\t ]+$", "", line)


list_item_re = re.compile(r"^(\s*)([-+*] |\d+\. )")
heading_re = re.compile(r"^\s*#{1,6} \S")
blockquote_re = re.compile(r"^\s*> ")
code_fence_re = re.compile(r"^\s*```")


def is_list_line(s: str) -> bool:
    return bool(list_item_re.match(s))


def ensure_blank_lines_around_lists(lines: list[str]) -> list[str]:
    # Skip inside fenced code blocks
    fixed: list[str] = []
    in_code = False
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if code_fence_re.match(line.strip()):
            in_code = not in_code
            fixed.append(line)
            i += 1
            continue

        if not in_code and is_list_line(line):
            # Identify contiguous list block
            start = i
            j = i
            while j < n and (is_list_line(lines[j]) or lines[j].strip() == ""):
                # include blank lines within list block (e.g., wrapped items)
                # stop if encountering a heading before items? keep simple.
                if lines[j].strip() != "" and not is_list_line(lines[j]):
                    break
                j += 1
            end = j  # exclusive

            # Ensure blank line before block unless at file start or preceded by heading/blockquote/code fence
            if len(fixed) > 0:
                prev = fixed[-1]
                if prev.strip() != "" and not heading_re.match(prev) and not blockquote_re.match(prev):
                    fixed.append("\n")

            # Append the block as-is
            fixed.extend(lines[start:end])

            # Ensure blank line after block unless EOF or next is heading/blockquote
            if end < n:
                nxt = lines[end]
                if nxt.strip() != "" and not heading_re.match(nxt) and not blockquote_re.match(nxt):
                    fixed.append("\n")
            i = end
            continue

        # default
        fixed.append(line)
        i += 1

    return fixed


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    # MD009
    lines = [fix_md009(l) for l in original]
    # MD032
    lines = ensure_blank_lines_around_lists(lines)

    if lines != original:
        path.write_text("".join(lines), encoding="utf-8")
        return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/fix_markdown.py <file1.md> [file2.md ...]")
        return 2
    changed = 0
    for arg in argv[1:]:
        p = Path(arg)
        if not p.exists() or not p.is_file():
            continue
        if process_file(p):
            changed += 1
            print(f"fixed: {p}")
    print(f"Total files fixed: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

