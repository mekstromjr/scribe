"""Transcribe an audio file with faster-whisper; write one JSON object per segment.

Usage: transcribe.py <audio> <out.jsonl> [model]   (model default: base.en)
Progress goes to stderr. Laptop tool: see README.md for why it is not a pipeline stage.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    size = sys.argv[3] if len(sys.argv) > 3 else "base.en"

    from faster_whisper import WhisperModel

    t0 = time.time()
    model = WhisperModel(size, device="cpu", compute_type="int8")
    # condition_on_previous_text=False: a repeat in the audio must not be smoothed away
    # by the decoder's own context, which is exactly what this tool looks for.
    segments, info = model.transcribe(
        src, beam_size=1, vad_filter=True, condition_on_previous_text=False
    )
    n = 0
    with open(out, "w") as fh:
        for s in segments:
            rec = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"# {n} segments, at {s.end / 60:.1f} min, {time.time() - t0:.0f}s",
                      file=sys.stderr, flush=True)
    print(f"done: {n} segments, audio {info.duration / 60:.1f} min, "
          f"wall {(time.time() - t0) / 60:.1f} min, model {size}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
