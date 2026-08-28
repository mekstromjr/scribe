"""Orchestrate the audio stage: listening script -> Kokoro -> m4b -> Audiobookshelf.

Runs AFTER the note is published, and best-effort throughout: the note is the product,
the audio is a bonus. Synthesis of a long article takes tens of minutes at Kokoro's
measured 1.6x realtime, and delaying the note (or failing the job) for it would make
the fast path hostage to the slow one.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scribe.config import Settings
from scribe.document import Document
from scribe.listening import build_script, lint_script
from scribe.package import ChapterAudio, build_m4b
from scribe.summarize import Summary
from scribe.tts import synthesize_segment

log = logging.getLogger("scribe.audio")


@dataclass
class AudioResult:
    m4b: Path
    # Deleting this deletes m4b; kept so callers control when the file dies (the Slack
    # upload happens after this returns).
    workdir: tempfile.TemporaryDirectory
    audio_seconds: float
    synth_seconds: float
    segments: int


def produce_audio(
    settings: Settings, doc: Document, summary: Summary, *, title: str, author: str
) -> AudioResult:
    """Synthesize and package. Raises on failure — the caller decides how quiet to be."""
    chapters = build_script(doc, summary, max_chars=settings.tts_max_chars)
    # Non-fatal: a slightly noisy audiobook beats no audiobook. Each finding names the
    # cleaning rule that is missing.
    for finding in lint_script(chapters):
        log.warning("listening-script artifact (will be vocalized): %s", finding)
    workdir = tempfile.TemporaryDirectory(prefix="scribe-audio-")
    work = Path(workdir.name)

    synth_seconds = 0.0
    n = 0
    audio_chapters: list[ChapterAudio] = []
    for ci, chapter in enumerate(chapters):
        files: list[Path] = []
        for si, segment in enumerate(chapter.segments):
            dest = work / f"c{ci:02d}s{si:03d}.wav"
            secs = synthesize_segment(settings, segment, dest)
            synth_seconds += secs
            files.append(dest)
            n += 1
            log.info("synthesized %s segment %d/%d in %.1fs",
                     chapter.title, si + 1, len(chapter.segments), secs)
        audio_chapters.append(ChapterAudio(chapter.title, files))

    m4b = work / "audiobook.m4b"
    audio_seconds = build_m4b(
        audio_chapters, m4b, title=title, author=author, workdir=work
    )
    return AudioResult(
        m4b=m4b,
        workdir=workdir,
        audio_seconds=audio_seconds,
        synth_seconds=synth_seconds,
        segments=n,
    )
