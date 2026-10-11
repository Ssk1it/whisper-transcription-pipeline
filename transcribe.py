"""
Offline batch transcription with faster-whisper (CTranslate2) on CUDA.

Turns audio/video files into timestamped TXT / SRT / VTT / JSON, with an
optional domain glossary fed to the model as `initial_prompt` so proper nouns
(products, people, technical terms) come out right.

Usage
-----
    python transcribe.py INPUT [INPUT ...] --model medium --language en
    python transcribe.py ./audio --model large-v3 --language zh --prompt-file prompt_zh.txt
    python transcribe.py talk.mp4 --glossary glossary.tsv      # post-process terms

INPUT can be a file or a directory (searched with --ext).

Why offline: the audio never leaves the machine, there is no per-minute cost,
no rate limit, and a 100-minute recording is a single call instead of hundreds
of chunked API requests.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# Windows / CUDA: pip-installed `nvidia-*` wheels ship the CUDA runtime in
# `site-packages/nvidia/*/bin`, which is NOT on the DLL search path. Without
# this, importing faster_whisper raises "cublas64_12.dll not found" even though
# the packages are installed. The directories must be registered BEFORE
# faster_whisper is imported.
# --------------------------------------------------------------------------
def add_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    try:
        import glob as _glob
        import site

        roots = set(site.getsitepackages() + [site.getusersitepackages()])
        for root in roots:
            for d in _glob.glob(os.path.join(root, "nvidia", "*", "bin")):
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except OSError:
                        pass
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # A missing CUDA install is reported later by faster_whisper itself.
        pass


add_cuda_dll_dirs()

DEFAULT_EXTS = ("mp4", "mkv", "mov", "webm", "mp3", "m4a", "wav", "flac", "ogg", "aac", "ts")


def fmt_ts(seconds: float, sep: str = ",") -> str:
    """Timestamp for SRT (comma) or VTT (dot)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s - int(s)) * 1000):03d}"


def collect_inputs(paths, exts, recursive: bool) -> list[str]:
    found: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in exts:
                pattern = os.path.join(p, "**", f"*.{ext}")
                found += glob.glob(pattern, recursive=recursive)
        else:
            found.append(p)
    # de-duplicate, stable order
    seen = set()
    out = []
    for f in sorted(found):
        key = os.path.abspath(f).lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def read_prompt(args) -> str | None:
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            return fh.read().strip()
    return args.prompt or None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline batch transcription (faster-whisper + CTranslate2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", nargs="+", help="audio/video file(s) or directory")
    ap.add_argument("--model", default="medium",
                    help="model name (small|medium|large-v3, downloaded on first use) "
                         "or a path to a local CTranslate2 model directory")
    ap.add_argument("--model-dir", default=os.environ.get("WHISPER_MODEL_DIR", ""),
                    help="directory holding local models; --model is resolved inside it first")
    ap.add_argument("--language", default="auto", help="language code, or 'auto' to detect")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                    help="'translate' renders any language into English")
    ap.add_argument("--outdir", default="transcripts")
    ap.add_argument("--prompt", default="", help="domain glossary / context for initial_prompt")
    ap.add_argument("--prompt-file", default="", help="file with the initial_prompt text")
    ap.add_argument("--glossary", default="", help="TSV glossary applied after transcription")
    ap.add_argument("--ext", default=",".join(DEFAULT_EXTS), help="extensions for directory input")
    ap.add_argument("--recursive", action="store_true", help="search directories recursively")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default="", help="float16 (cuda) / int8 (cpu); auto if empty")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="disable the voice-activity filter")
    ap.add_argument("--threads", type=int, default=0, help="cpu threads (0 = library default)")
    ap.add_argument("--no-skip", action="store_true", help="re-transcribe files that already have output")
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed.\n  pip install -r requirements.txt", file=sys.stderr)
        return 2

    # Resolve the model: a local directory wins, otherwise let CTranslate2/HF resolve the name.
    model_ref = args.model
    if args.model_dir:
        candidate = os.path.join(args.model_dir, args.model)
        if os.path.isdir(candidate):
            model_ref = candidate
    if os.path.isdir(model_ref):
        model_ref = os.path.abspath(model_ref)

    compute = args.compute or ("float16" if args.device == "cuda" else "int8")
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    exts = tuple(e.strip().lstrip(".") for e in args.ext.split(",") if e.strip())
    files = collect_inputs(args.input, exts, args.recursive)
    if not files:
        print("No input files matched.", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    prompt = read_prompt(args)

    t_load = time.time()
    if not args.quiet:
        print(f"[model] {model_ref}  device={args.device}  compute={compute}", flush=True)
    model = WhisperModel(model_ref, device=args.device, compute_type=compute,
                         cpu_threads=args.threads or 0)
    load_s = time.time() - t_load
    if not args.quiet:
        print(f"[model] ready in {load_s:.1f}s", flush=True)

    results = []
    for idx, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(args.outdir, stem)

        if not args.no_skip and os.path.exists(base + ".json"):
            if not args.quiet:
                print(f"[{idx}/{len(files)}] {stem} — skipped (output exists)", flush=True)
            results.append({"file": stem, "skipped": True})
            continue

        size_mb = os.path.getsize(path) / 1e6
        if not args.quiet:
            print(f"\n[{idx}/{len(files)}] {stem}  ({size_mb:.1f} MB)", flush=True)
        t0 = time.time()

        segments, info = model.transcribe(
            path,
            language=None if args.language == "auto" else args.language,
            task=args.task,
            beam_size=args.beam,
            vad_filter=not args.no_vad,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )

        rows = []
        with open(base + ".srt", "w", encoding="utf-8") as f_srt, \
             open(base + ".vtt", "w", encoding="utf-8") as f_vtt, \
             open(base + ".txt", "w", encoding="utf-8") as f_txt:
            f_vtt.write("WEBVTT\n\n")
            for n, seg in enumerate(segments, 1):
                text = seg.text.strip()
                if not text:
                    continue
                rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
                f_srt.write(f"{n}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n")
                f_vtt.write(f"{fmt_ts(seg.start, '.')} --> {fmt_ts(seg.end, '.')}\n{text}\n\n")
                f_txt.write(f"[{fmt_ts(seg.start, '.')[:8]}] {text}\n")

                if not args.quiet and n % 50 == 0:
                    el = max(time.time() - t0, 0.001)
                    print(f"  .. {n} segments | {seg.end / 60:.1f}/{info.duration / 60:.1f} min "
                          f"| {el:.0f}s elapsed | {seg.end / el:.1f}x realtime", flush=True)

        payload = {
            "source_file": os.path.basename(path),
            "model": args.model,
            "device": args.device,
            "compute_type": compute,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": rows,
        }
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)

        with open(base + ".clean.txt", "w", encoding="utf-8") as fh:
            fh.write("".join(r["text"] for r in rows))

        if args.glossary:
            from termfix import apply_file

            fixed = apply_file(base + ".txt", base + ".fixed.txt", args.glossary)
            if not args.quiet:
                print(f"  glossary: {fixed} replacement(s) -> {os.path.basename(base)}.fixed.txt",
                      flush=True)

        elapsed = time.time() - t0
        speed = info.duration / max(elapsed, 0.001)
        results.append({
            "file": stem,
            "language": info.language,
            "duration": round(info.duration, 1),
            "segments": len(rows),
            "seconds": round(elapsed, 1),
            "speed_x": round(speed, 1),
        })
        print(f"[done] {stem}: {len(rows)} segments, {elapsed:.1f}s, {speed:.1f}x realtime",
              flush=True)

    # Summary table — useful when batching a whole archive.
    print("\n" + "=" * 78)
    print(f"model      : {args.model} ({args.device}/{compute}), load {load_s:.1f}s")
    print(f"output dir : {os.path.abspath(args.outdir)}")
    total_audio = sum(r.get("duration", 0) for r in results)
    total_wall = sum(r.get("seconds", 0) for r in results)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['file']:<40} skipped")
        else:
            print(f"  {r['file']:<40} {r['language']:<4} {r['duration']:>7.1f}s audio "
                  f"| {r['seconds']:>6.1f}s wall | {r['speed_x']:>6.1f}x")
    if total_wall:
        print(f"  {'TOTAL':<40} {'':<4} {total_audio:>7.1f}s audio "
              f"| {total_wall:>6.1f}s wall | {total_audio / total_wall:>6.1f}x")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
