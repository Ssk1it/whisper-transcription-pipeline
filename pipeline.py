"""
End-to-end driver: audio/video in, client-ready transcript out.

    transcribe.py  ->  termfix.py (glossary + Traditional->Simplified)  ->  segment.py

Equivalent to running the three tools by hand; kept as the copy-pasteable
recipe in the README's quick-start, so a client can reproduce a delivery.

Usage
-----
    python pipeline.py interview.mp4 --language en --model medium \
        --glossary glossary.sample.tsv --chapters chapters.txt --workdir out
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"step failed with exit code {result.returncode}: {cmd[0]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full transcription pipeline.")
    ap.add_argument("input", help="audio/video file or directory")
    ap.add_argument("--workdir", default="out", help="output directory (default: out)")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--model-dir", default=os.environ.get("WHISPER_MODEL_DIR", ""))
    ap.add_argument("--language", default="auto")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--glossary", default="", help="TSV glossary for term repair")
    ap.add_argument("--simplified", action="store_true",
                    help="convert Chinese output to Simplified (needs opencc)")
    ap.add_argument("--chapters", default="")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    py = sys.executable
    work = args.workdir
    os.makedirs(work, exist_ok=True)

    # 1. transcribe -> txt/srt/vtt/json/clean.txt
    cmd = [py, "transcribe.py", args.input, "--outdir", work,
           "--model", args.model, "--language", args.language, "--device", args.device]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.prompt_file:
        cmd += ["--prompt-file", args.prompt_file]
    run(cmd)

    # 2. glossary repair (optional)
    stem = os.path.splitext(os.path.basename(args.input.rstrip("/\\")))[0]
    src_txt = os.path.join(work, stem + ".txt")
    if not os.path.exists(src_txt):
        print(f"[skip] {src_txt} not found — glossary step needs a single file input")
        return 0

    final_txt = src_txt
    if args.glossary:
        fixed = os.path.join(work, stem + ".fixed.txt")
        cmd = [py, "termfix.py", src_txt, fixed, args.glossary]
        if args.simplified:
            cmd.append("--to-simplified")
        run(cmd)
        final_txt = fixed

    # 3. readable output
    cmd = [py, "segment.py", final_txt, "--out", os.path.join(work, stem + ".readable.md"),
           "--markdown"]
    if args.chapters:
        cmd += ["--chapters", args.chapters]
    run(cmd)

    print("\nDelivered:")
    for name in (".txt", ".srt", ".vtt", ".json", ".readable.md"):
        p = os.path.join(work, stem + name)
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
