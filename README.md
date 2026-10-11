# whisper-transcription-pipeline

Offline batch transcription for long audio and video, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) with CUDA
acceleration.

Give it a folder of recordings, get back timestamped **TXT / SRT / VTT / JSON** plus a
readable Markdown transcript with paragraphs — and proper nouns spelled correctly.

---

## Why run this locally instead of calling a transcription API

| | Local (this repo) | Cloud API |
|---|---|---|
| Per-minute cost | 0 | metered, per file |
| File length limit | none — a 100-minute recording is one call | usually chunked uploads + polling |
| Audio leaves the machine | never | always |
| Rate limits / retries | none | to design around |
| Speaker privacy (interviews, medical, legal) | keeps | depends on vendor policy |

The trade-off is real: you need a GPU (or patience — CPU works, roughly an order of
magnitude slower), and you own the accuracy-tuning work. That is what the glossary and
prompt features below are for.

## Measured performance

Reproducible numbers from `transcribe.py` on this machine — 212.6 s of English speech,
`beam_size=5`, VAD on, `float16`:

| Model | Wall time | Speed | Model load | Segments |
|---|---|---|---|---|
| `small` | 5.8 s | **36.6x** realtime | 0.6 s | 45 |
| `medium` | 18.7 s | **11.4x** realtime | 1.9 s | 64 |

Hardware: RTX 4090 Laptop 16 GB, 16-core CPU, Windows 11, models stored locally in
CTranslate2 format (no download at runtime). Wall time excludes model load, which is
reported separately. Throughput scales close to linearly with audio length once the
model is resident, so a 100-minute recording at `medium` lands around 9 minutes.

Model choice is a straight accuracy/speed dial — `small` is fine for clear single-speaker
audio, `medium` is the default here, `large-v3` is worth it for accents, crosstalk and
heavy jargon.

## Quick start

```bash
pip install -r requirements.txt

# one file
python transcribe.py interview.mp4 --language en --model medium

# a whole archive, skipping anything already done
python transcribe.py ./recordings --recursive --model large-v3 --language zh

# with domain context and post-hoc term repair
python transcribe.py call.wav --language en \
    --prompt-file prompt_en.sample.txt --glossary glossary.sample.tsv
```

Outputs land in `transcripts/` (override with `--outdir`):

```
call.txt         [00:00:04] timestamped fragments, one line per segment
call.srt         subtitles
call.vtt         WebVTT
call.json        start/end/text + detected language + confidence
call.clean.txt   continuous text, no timestamps
call.fixed.txt   after --glossary
```

### Full pipeline

`pipeline.py` chains the three tools and prints the delivered file list:

```bash
python pipeline.py interview.mp4 --language en --model medium \
    --glossary glossary.sample.tsv --chapters chapters.sample.txt --workdir out
```

```
transcribe.py  ->  termfix.py (glossary, optional Traditional->Simplified)  ->  segment.py
```

Run the steps individually when you want to inspect or hand-edit the intermediate
transcript — which is what actually happens on a paid job.

## Getting proper nouns right

Transcription accuracy on ordinary conversation is a solved problem; names, brands and
jargon are not. Two cheap stages fix most of it.

**1. `--prompt-file` / `--prompt` — context before the fact.** Whisper accepts an
`initial_prompt` that biases the decoder. A short paragraph listing the terms that will
recur (see [`prompt_en.sample.txt`](prompt_en.sample.txt),
[`prompt_zh.sample.txt`](prompt_zh.sample.txt)) measurably reduces homophone errors.
Keep it to 2–4 sentences; it only conditions the opening window.

**2. `--glossary` — corrections after the fact.** ASR mistakes on a given proper noun are
usually *consistent* across the whole recording, so one replacement rule fixes every
occurrence. The glossary is a plain TSV, ordered, with `exact` / `ci` / `re` modes:

```
N8N	n8n
open ai	OpenAI	ci
\bN\s*8\s*N\b	n8n	re
```

Real before/after from [`samples/termfix_demo_before.txt`](samples/termfix_demo_before.txt)
→ [`samples/termfix_demo_after.txt`](samples/termfix_demo_after.txt), 7 replacements,
timestamps untouched:

| | |
|---|---|
| **before** | `…move everything off spreadsheets into N8N.`<br>`The intake form posts jason to a webhook, and then open ai classifies it.`<br>`We ran a subtraction experiment…`<br>`Every step has a retry with a close loop…` |
| **after** | `…move everything off spreadsheets into n8n.`<br>`The intake form posts JSON to a webhook, and then OpenAI classifies it.`<br>`We ran a ablation study…`<br>`Every step has a retry with a closed loop…` |

The rule of thumb: only add a rule when the replacement is the *only* reasonable reading.
A wrong glossary entry silently corrupts a whole transcript, which is worse than one
visible error.

For Chinese output, Whisper regularly mixes Traditional characters into Simplified text.
`termfix.py --to-simplified` (via OpenCC) normalises it as part of the same pass.

## Readable output

`segment.py` turns the fragment-per-line output into prose. Fragments are merged until a
paragraph reaches a target length, but a break only happens *after* a sentence
terminator, so sentences are never split:

```bash
python segment.py transcripts/call.txt --out call.readable.md --markdown \
    --chapters chapters.sample.txt
```

With `--chapters` (a file of `MM:SS Title` lines) it inserts section headings at the
right timestamps — useful for long interviews and podcasts.
`samples/sample_en.readable.md` is a real generated example.

## Three pitfalls this repo exists to save you from

These cost real debugging time and are documented in
[`docs/cuda-and-windows.md`](docs/cuda-and-windows.md):

1. **`cublas64_12.dll not found` on Windows with CUDA installed.** The `nvidia-*` pip
   wheels put the runtime in `site-packages/nvidia/*/bin`, which is *not* on the DLL
   search path. `transcribe.py` registers those directories with
   `os.add_dll_directory()` **before** importing `faster_whisper` — order matters, and
   adding it to `PATH` alone is not enough on its own.
2. **Traditional characters leaking into Simplified Chinese output.** Solved with
   OpenCC, not with prompt hacking.
3. **Proper nouns are wrong in a *consistent* way.** Which is good news: it means a
   glossary is a complete fix rather than a partial one.

## Project layout

```
transcribe.py   batch transcription -> txt / srt / vtt / json
termfix.py      glossary-driven term repair + optional Traditional->Simplified
segment.py      paragraph and chapter assembly, plain text or Markdown
pipeline.py     all three in one command
glossary.sample.tsv      annotated rules file
chapters.sample.txt      chapter markers format
prompt_en/zh.sample.txt  initial_prompt examples
samples/                 real outputs, including a term-repair before/after
docs/                    CUDA / Windows troubleshooting
```

## Limitations — worth knowing before paying for this

* **No diarization.** Multi-speaker recordings are transcribed but not labelled per
  speaker. If you need "who said what", that is a separate component (`pyannote` etc.)
  and extra work — ask for it explicitly.
* **Timestamps are segment-level, not word-level.** Good enough for subtitles; if you
  need word-level alignment for karaoke-style captions, enable
  `word_timestamps=True` and expect slower runs.
* **`initial_prompt` is not a hard constraint.** The model can still override it; the
  glossary pass is what makes the guarantee.
* **Long silences and music** are skipped by VAD by default (`--no-vad` to disable).
* **CPU mode works** (`--device cpu --compute int8`) but is roughly an order of magnitude
  slower — fine for a few files, not for an archive.

## License

MIT — see [LICENSE](LICENSE).
