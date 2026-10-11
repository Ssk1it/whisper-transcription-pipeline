"""
Glossary-driven term repair for machine transcripts.

ASR models spell proper nouns phonetically: product names, people, acronyms and
technical jargon are the first things they get wrong, and the error is usually
consistent across a whole recording. Fixing them with a small curated list is
far cheaper than re-running with a bigger model.

Glossary file: TSV, `#` comments allowed, one rule per line

    wrong<TAB>right            exact, case-sensitive
    wrong<TAB>right<TAB>ci     case-insensitive
    \\bN\\s*8\\s*n\\b<TAB>n8n<TAB>re   regular expression

CLI
---
    python termfix.py input.txt output.txt glossary.tsv
    python termfix.py input.txt output.txt glossary.tsv --to-simplified

The `--to-simplified` / `--to-traditional` flags use OpenCC (optional dependency)
to normalise Chinese output: Whisper frequently mixes Traditional characters
into Simplified transcripts of Mandarin audio.
"""
from __future__ import annotations

import argparse
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

#: Matches the `[00:01:23] text` lines produced by transcribe.py so timestamps
#: are never touched by a replacement rule.
TIMED_LINE = re.compile(r"^(\[\d\d:\d\d:\d\d(?:\.\d+)?\]\s*)(.*)$")


def load_glossary(path: str) -> list[tuple]:
    rules: list[tuple] = []
    # utf-8-sig: tolerates a BOM, which editors on Windows add by default.
    with open(path, encoding="utf-8-sig") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[glossary] line {lineno} skipped (needs TAB): {line!r}", file=sys.stderr)
                continue
            wrong, right = parts[0], parts[1]
            mode = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else "exact"
            if wrong == right or not wrong:
                continue
            if mode.startswith("re"):
                rules.append(("re", re.compile(wrong), right))
            elif mode.startswith("ci"):
                rules.append(("ci", re.compile(re.escape(wrong), re.IGNORECASE), right))
            else:
                rules.append(("exact", wrong, right))
    return rules


def fix(text: str, rules) -> tuple[str, int]:
    count = 0
    for kind, pattern, right in rules:
        if kind == "exact":
            if pattern in text:
                count += text.count(pattern)
                text = text.replace(pattern, right)
        else:
            text, n = pattern.subn(right, text)
            count += n
    return text, count


def normalise_chinese(text: str, direction: str) -> str:
    try:
        from opencc import OpenCC
    except ImportError:
        raise SystemExit("opencc is required for --to-simplified / --to-traditional:\n"
                         "  pip install opencc-python-reimplemented")
    return OpenCC("t2s" if direction == "simplified" else "s2t").convert(text)


def apply_file(src: str, out: str, glossary: str, direction: str | None = None) -> int:
    """Repair `src` into `out`; returns the number of replacements made."""
    rules = load_glossary(glossary)
    with open(src, encoding="utf-8") as fh:
        content = fh.read()

    if direction:
        content = normalise_chinese(content, direction)

    total = 0
    lines = []
    for line in content.split("\n"):
        m = TIMED_LINE.match(line.strip())
        if m:
            fixed, n = fix(m.group(2), rules)
            total += n
            lines.append(m.group(1) + fixed)
        else:
            fixed, n = fix(line, rules)
            total += n
            lines.append(fixed)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply a glossary to a transcript.")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("glossary")
    ap.add_argument("--to-simplified", action="store_true")
    ap.add_argument("--to-traditional", action="store_true")
    args = ap.parse_args()

    direction = None
    if args.to_simplified:
        direction = "simplified"
    elif args.to_traditional:
        direction = "traditional"

    n = apply_file(args.src, args.out, args.glossary, direction)
    print(f"rules applied: {n}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
