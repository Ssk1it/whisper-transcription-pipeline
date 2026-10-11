# CUDA and Windows notes

Everything here was learned the hard way on a Windows 11 machine with a laptop GPU.
None of it is exotic, but every item below produces an error message that points
somewhere other than the real cause.

## `cublas64_12.dll not found` after installing `nvidia-cublas-cu12`

**Symptom**

```
Could not locate cublas64_12.dll
```

or a bare `ImportError` / `OSError` from `faster_whisper` / `ctranslate2` — even though
`pip list` shows `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` installed, and even though
`import torch; torch.cuda.is_available()` (if you have torch) returns `True`.

**Cause**

Since Python 3.8, Windows no longer searches the process `PATH` for DLL dependencies of
an extension module. It uses directories registered with `os.add_dll_directory()`. The
`nvidia-*` wheels install their runtime into:

```
<site-packages>/nvidia/cublas/bin/cublas64_12.dll
<site-packages>/nvidia/cudnn/bin/cudnn_*.dll
```

…which nobody registers.

**Fix** — register the directories *before* the first CTranslate2 import. `transcribe.py`
does this at the top of the module:

```python
import glob, os, site

for root in set(site.getsitepackages() + [site.getusersitepackages()]):
    for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
```

Two details that matter:

* **Order.** Putting this after `from faster_whisper import WhisperModel` has no effect —
  by then the DLL is already resolved (and already failed).
* Do not rely on setting `PATH` alone. It is not the mechanism Windows uses here.

## Verifying the GPU is actually being used

```python
from faster_whisper import WhisperModel
m = WhisperModel("medium", device="cuda", compute_type="float16")
print(m.model.device)   # cuda
```

If CUDA is unavailable, CTranslate2 silently has nowhere to go and raises at model
construction, not at the first transcript — which is a mercy. `--device cpu --compute int8`
is the fallback and needs no NVIDIA runtime at all.

## Local models avoid a first-run download

`--model medium` downloads from Hugging Face on first use and caches it. In environments
without reliable access to Hugging Face, convert or copy the model once and point at it:

```bash
python transcribe.py in.mp3 --model medium --model-dir D:/models/whisper
# or
set WHISPER_MODEL_DIR=D:/models/whisper
```

`--model-dir` is checked first; if `<model-dir>/<name>` exists it is used, otherwise the
value is passed through to CTranslate2 as a name. A directory containing
`model.bin`, `config.json`, `tokenizer.json` and `vocabulary.*` is a valid CTranslate2
model directory.

## Long files, memory and interrupts

* A 100-minute recording runs as a single `transcribe()` call. VRAM use is driven by
  model size, not audio length — `medium` + `float16` sits comfortably in 16 GB.
* VAD (`--no-vad` to disable) is what keeps long silences from costing time and
  producing hallucinated text over music or noise.
* Killing the run mid-file leaves a partial `.srt` but no `.json`, and `--no-skip` is
  off by default, so simply re-running finishes the file cleanly instead of skipping it.
  That is deliberate: `.json` is the completion marker.

## Encoding on Windows

Console output is UTF-8. If you see mojibake in `cmd.exe`, either use Windows Terminal /
PowerShell, or pass `PYTHONUTF8=1`. The scripts call
`sys.stdout.reconfigure(encoding="utf-8")` themselves, so piping to a file is safe.
