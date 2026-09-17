"""Stitch WAV segments into a single .m4b with chapter markers.

Chapter offsets come from the WAV headers themselves (stdlib ``wave``) rather than
ffprobe: Kokoro emits plain PCM, so frames/framerate is the exact duration and the
container image needs one less tool. ffmpeg does the concat + single AAC encode.

Metadata lives in the m4b TAGS, not the upload request: Audiobookshelf's upload API
ignores most request metadata server-side (verified upstream issue), so the tags are
the source of truth the scanner actually reads.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


class PackageError(RuntimeError):
    pass


@dataclass
class ChapterAudio:
    title: str
    files: list[Path]


def wav_seconds(path: Path) -> float:
    """Duration from the actual bytes on disk, NOT the header's frame count.

    Kokoro STREAMS its WAV responses, so the length fields in the header are
    placeholders written before the audio existed — trusting getnframes() yielded a
    "5965-minute" article and chapter offsets to match. The sample format in the
    header is real; the amount of audio is whatever follows the data chunk.
    """
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        frame_bytes = w.getsampwidth() * w.getnchannels()
        if not rate or not frame_bytes:
            raise PackageError(f"{path.name}: malformed WAV header")

    raw = path.read_bytes()
    at = raw.find(b"data")
    if at < 0:
        raise PackageError(f"{path.name}: no data chunk")
    return (len(raw) - at - 8) / (rate * frame_bytes)


def _ffmetadata(chapters: list[ChapterAudio]) -> str:
    """FFMETADATA1 chapter block. Times in ms (TIMEBASE 1/1000)."""
    lines = [";FFMETADATA1"]
    cursor = 0
    for ch in chapters:
        length_ms = round(sum(wav_seconds(f) for f in ch.files) * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={cursor}",
            f"END={cursor + length_ms}",
            # '=' and newlines would corrupt the metadata syntax; titles are
            # ours ("Summary", "Full article") but escape defensively anyway.
            f"title={ch.title.replace(chr(10), ' ').replace('=', '-')}",
        ]
        cursor += length_ms
    return "\n".join(lines) + "\n"


def build_m4b(
    chapters: list[ChapterAudio],
    dest: Path,
    *,
    title: str,
    author: str,
    workdir: Path,
    narrator: str | None = None,
    grouping: str | None = None,
    comment: str | None = None,
) -> float:
    """Concat + encode. Returns total audio seconds.

    ``narrator`` lands in the composer tag, which Audiobookshelf reads as the
    narrator; ``grouping`` (the person) in grouping and ``comment`` in comment. The API patch in
    abs.py is the authoritative metadata; these make the file self-describing if it
    is ever rescanned or moved (scribe#12).

    Mono 64k AAC: Kokoro output is 24 kHz mono speech, where 64 kbps is transparent —
    a 45-minute article lands around 22 MB instead of the ~120 MB stereo-bitrate
    default.
    """
    files = [f for ch in chapters for f in ch.files]
    if not files:
        raise PackageError("no audio segments to package")

    concat = workdir / "concat.txt"
    # The concat demuxer unquotes with its own rules; single-quote and escape.
    concat.write_text(
        "".join(f"file '{str(f).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                for f in files)
    )
    meta = workdir / "ffmeta.txt"
    meta.write_text(_ffmetadata(chapters))

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-i", str(meta),
        "-map_metadata", "1", "-map_chapters", "1",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        # album = title, deliberately: Audiobookshelf's audiobook scanner takes the
        # ALBUM tag as the book title (verified 2026-08-27 — a fixed album string
        # became the shelf title of the first test upload). Genre carries the
        # article-ness instead.
        "-metadata", f"title={title}",
        "-metadata", f"album={title}",
        "-metadata", f"artist={author}",
        "-metadata", "genre=Article",
        *(["-metadata", f"composer={narrator}"] if narrator else []),
        *(["-metadata", f"grouping={grouping}"] if grouping else []),
        *(["-metadata", f"comment={comment}"] if comment else []),
        "-movflags", "+faststart",
        "-f", "mp4",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        raise PackageError("ffmpeg is not installed in this environment") from exc
    except subprocess.TimeoutExpired as exc:
        raise PackageError("ffmpeg timed out encoding the audiobook") from exc
    except subprocess.CalledProcessError as exc:
        raise PackageError(f"ffmpeg failed: {exc.stderr[-500:]}") from exc

    return sum(wav_seconds(f) for f in files)
