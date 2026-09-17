"""Find passages spoken more than once in a transcript produced by transcribe.py.

Usage: repeats.py <transcript.jsonl> [min_words]

Reports sentences of at least ``min_words`` (default 8) that occur more than once, and
runs of consecutive repeated sentences, which is what a repeated paragraph looks like.
Timestamps are approximate (nearest whisper segment start) but close enough to jump to
in a player.
"""

from __future__ import annotations

import collections
import json
import re
import sys


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    min_words = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with open(sys.argv[1]) as fh:
        segs = [json.loads(line) for line in fh if line.strip()]

    # Sentences over the joined transcript, each mapped back to a start time by
    # character offset into the same joined text.
    text = " ".join(s["text"] for s in segs)
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    keys = [_norm(s) for s in sents]

    sent_offsets: list[int] = []
    pos = 0
    for s in sents:
        sent_offsets.append(pos)
        pos += len(s) + 1
    seg_starts: list[tuple[int, float]] = []
    pos = 0
    for s in segs:
        seg_starts.append((pos, s["start"]))
        pos += len(s["text"]) + 1

    def time_at(offset: int) -> str:
        t = 0.0
        for seg_off, start in seg_starts:
            if seg_off > offset:
                break
            t = start
        return _clock(t)

    counts = collections.Counter(k for k in keys if len(k.split()) >= min_words)
    dupes = {k for k, n in counts.items() if n > 1}
    print(f"{len(sents)} sentences; {len(dupes)} distinct sentences "
          f"(>= {min_words} words) spoken more than once")

    runs: list[list[int]] = []
    cur: list[int] = []
    for i, k in enumerate(keys):
        if k in dupes:
            cur.append(i)
            continue
        if len(cur) >= 2:
            runs.append(cur)
        cur = []
    if len(cur) >= 2:
        runs.append(cur)
    print(f"{len(runs)} run(s) of 2+ consecutive repeated sentences (paragraph-level)")

    in_runs = {i for r in runs for i in r}
    for r in runs[:10]:
        first = r[0]
        others = [i for i, k in enumerate(keys) if k == keys[first] and i != first]
        print(f"  at {time_at(sent_offsets[first])} ({len(r)} sentences), also at "
              f"{[time_at(sent_offsets[i]) for i in others]}: {sents[first][:110]}...")
    shown = 0
    for k in dupes:
        idx = [i for i, kk in enumerate(keys) if kk == k]
        if any(i in in_runs for i in idx):
            continue
        print(f"  single sentence at {[time_at(sent_offsets[i]) for i in idx]}: "
              f"{sents[idx[0]][:100]}...")
        shown += 1
        if shown >= 8:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
