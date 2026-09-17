# audio-qc: did the audiobook say what the script said?

On-demand, run on a laptop, not in the cluster. Whisper's `base.en` model transcribes
110 minutes of audio in about 2.5 minutes on an M1 Max; on the homelab's CPU-only nodes
it would be several times slower than real time and would compete with Ollama and
Kokoro, so this stays a tool you reach for when someone reports an oddity.

What it found the first time (scribe#13, 2026-09-17): two minutes of a chapter spoken
twice, because the OCR model transcribed one page twice under its token cap. That led
to the repeated-paragraph lint and the OCR repeat collapse now in the pipeline.

## Run

```bash
# 1. a venv with faster-whisper (not a project dependency on purpose)
uv venv .qc && uv pip install --python .qc/bin/python faster-whisper

# 2. fetch the m4b from Audiobookshelf (item id from the shelf URL; the file's ino from
#    GET /api/items/<id>, media.audioFiles[0].ino). Token: the one scribe uses.
curl -sSL -H "Authorization: Bearer $ABS_TOKEN" -o item.m4b \
  "https://shelf.meklab.net/api/items/<item-id>/file/<ino>"

# 3. transcribe -> JSONL of {start, end, text} segments (model: base.en, small.en, ...)
.qc/bin/python scripts/audio-qc/transcribe.py item.m4b item.jsonl base.en

# 4. repeats: sentences of 8+ words spoken more than once, and runs of them
.qc/bin/python scripts/audio-qc/repeats.py item.jsonl
```

`repeats.py` prints approximate timestamps for each repeat so you can jump to it in the
player. Compare against `ffprobe -show_chapters item.m4b` to see whether a repeat sits
inside one chapter (script or OCR side) or straddles a boundary (the summary quoting
the source).

## Not done here

Aligning the transcript against the original script to find omissions. The script is
not stored after delivery; rebuild it with `scribe listen --dry-run <source>` and diff
normalized sentences if that question comes up.
