"""
Rebuild readable prose from a raw transcript.

A transcript straight out of the model is one line per 3-8 second fragment,
which is painful to read or hand to a client. This tool:

  * merges fragments into paragraphs of a target length (breaking on sentence
    enders so sentences are never split),
  * optionally inserts chapter headings from a timestamp file,
  * emits plain text, Markdown, or a flattened single-block text file.

Input is the `[HH:MM:SS] text` format written by transcribe.py, or the `.json`
file it produces.

Usage
-----
    python segment.py transcripts/talk.txt --out talk.readable.md --markdown
    python segment.py transcripts/talk.json --chapters chapters.txt --out talk.md --markdown
    python segment.py transcripts/talk.txt --flatten --out talk.plain.txt

chapters.txt:

    00:33 Opening
    10:59 Two real cases
    21:35 Finding the first paid job
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TIMED_LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)(?:\.\d+)?\]\s*(.*)$")

#: Sentence terminators for CJK and Latin text. A paragraph only breaks after
#: one of these, so a sentence is never split across paragraphs.
SENTENCE_END = "。！？!?…；;."
CLOSING = "\"'”’）)]】"  # a terminator may sit inside a closing quote or bracket


def ends_sentence(text: str) -> bool:
    stripped = text.rstrip(CLOSING)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def parse_seconds(hh: str, mm: str, ss: str) -> int:
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def load_segments(path: str) -> list[tuple[int, str]]:
    """Return [(start_seconds, text), ...] from .json or timestamped .txt."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [(int(round(s["start"])), s["text"].strip()) for s in data.get("segments", [])]

    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = TIMED_LINE.match(line.strip())
            if m:
                out.append((parse_seconds(*m.groups()[:3]), m.group(4).strip()))
    return out


def load_chapters(path: str) -> list[tuple[int, str]]:
    chapters = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d{1,2}):(\d\d)(?::(\d\d))?\s+(.*)$", line)
            if not m:
                continue
            h, mnt, sec, title = m.groups()
            total = int(h) * 3600 + int(mnt) * 60 + int(sec or 0)
            chapters.append((total, title.strip()))
    return sorted(chapters)


def build_paragraphs(segments, target_chars: int):
    """Merge fragments into paragraphs; never split mid-sentence."""
    paragraphs = []
    buf, start = "", None
    for ts, text in segments:
        if start is None:
            start = ts
        buf += text
        if len(buf) >= target_chars and ends_sentence(buf):
            paragraphs.append((start, buf))
            buf, start = "", None
    if buf:
        paragraphs.append((start if start is not None else 0, buf))
    return paragraphs


def render(paragraphs, chapters, markdown: bool) -> str:
    lines: list[str] = []
    used: set[int] = set()
    for start, body in paragraphs:
        if chapters:
            for i, (cs, title) in enumerate(chapters):
                if start >= cs and i not in used:
                    for j in range(i):
                        used.add(j)
                    used.add(i)
                    lines.append("")
                    lines.append(f"## {clock(cs)} — {title}" if markdown else
                                 f"\n{'=' * 70}\n{clock(cs)} — {title}\n{'=' * 70}")
                    lines.append("")
                    break
        lines.append(f"[{clock(start)}] {body}" if markdown else f"[{clock(start)}] {body}")
        if markdown:
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn a raw transcript into readable prose.")
    ap.add_argument("src", help=".txt (timestamped) or .json produced by transcribe.py")
    ap.add_argument("--out", default="", help="output path (default: <src>.readable.<ext>)")
    ap.add_argument("--chapters", default="", help="optional chapter file: 'MM:SS Title' per line")
    ap.add_argument("--paragraph-chars", type=int, default=220,
                    help="approximate paragraph length before a break (default 220)")
    ap.add_argument("--markdown", action="store_true", help="emit Markdown with ## headings")
    ap.add_argument("--flatten", action="store_true",
                    help="emit one continuous text block with timestamps stripped")
    args = ap.parse_args()

    segments = load_segments(args.src)
    if not segments:
        print("No timestamped segments found — is this a transcribe.py output?", file=sys.stderr)
        return 1

    if args.flatten:
        text = "".join(t for _, t in segments)
        out = args.out or os.path.splitext(args.src)[0] + ".plain.txt"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"flattened {len(segments)} segments -> {out} ({len(text)} chars)")
        return 0

    chapters = load_chapters(args.chapters) if args.chapters else []
    paragraphs = build_paragraphs(segments, args.paragraph_chars)
    default_ext = ".md" if args.markdown else ".readable.txt"
    out = args.out or os.path.splitext(args.src)[0] + default_ext

    header = ""
    if args.markdown:
        title = os.path.splitext(os.path.basename(args.src))[0]
        total = clock(segments[-1][0])
        header = f"# {title}\n\n_Transcribed automatically, {len(segments)} segments, {total}._\n\n"

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(header + render(paragraphs, chapters, args.markdown))

    print(f"{len(segments)} segments -> {len(paragraphs)} paragraphs"
          f"{f', {len(chapters)} chapters' if chapters else ''} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
