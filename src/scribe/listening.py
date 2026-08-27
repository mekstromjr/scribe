"""Build the "listening script" — the text that actually gets spoken.

Deliberately rule-based, not an LLM pass. Rewriting a full article "for listening" with
the CPU model would roughly double every job's wall clock (generation is the expensive
direction at ~15 tok/s, and a rewrite generates the WHOLE document, not a summary) and
would risk silent paraphrasing — the reader should hear the article, not the model's
memory of it. Deterministic stripping removes the things that are genuinely painful in
audio (URLs, citation markers, figure captions, markdown syntax) and nothing else. An
LLM polish pass can slot in here if the pipeline moves to a cloud model.

The script is split into chapters (Summary first, then the article) so the m4b gets
chapter markers — one tap in the Audiobookshelf player replays or skips the summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scribe.document import Document
from scribe.summarize import Summary

# Markdown constructs, in stripping order (images before links — an image IS a link
# with a bang, and the link rule alone would leave its alt text plus a stray '!').
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1")
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

# Citation/footnote markers: [1], [12], [^3] — but not [text], which the link rule
# already reduced to its label.
_CITATION = re.compile(r"\[\^?\d+\]")

# Caption-ish lines. Anchored to line start and short lines only: an article ABOUT
# photography legitimately starts sentences with "Photo" mid-paragraph; a caption is a
# short standalone line.
_CAPTION = re.compile(
    r"^(?:figure|fig\.|photo|image|photograph|illustration|source|credit|caption)\b[^\n]{0,200}$",
    re.IGNORECASE | re.MULTILINE,
)

_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_for_listening(text: str) -> str:
    """Strip what is painful to hear; keep every sentence the author wrote."""
    text = _FENCE.sub(" Code example omitted. ", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    text = _CITATION.sub("", text)
    text = _TABLE_ROW.sub("", text)
    text = _CAPTION.sub("", text)
    # Headings become spoken sentences: a pause-inducing period, not a hash.
    text = _HEADING.sub(lambda m: f"{m.group(1).strip().rstrip('.:')}." , text)
    text = _INLINE_CODE.sub(r"\1", text)
    # Emphasis twice: bold-italic nests (*** outside, * inside after one pass).
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def split_segments(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into segments of at most ``max_chars``.

    A paragraph longer than the cap is split at sentence boundaries as a fallback;
    only a pathological single sentence ever gets hard-cut.
    """
    segments: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            segments.append("\n\n".join(current))
            current, size = [], 0

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) + 1 > max_chars:
                    segments.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
                # A single sentence over the cap: hard-cut rather than send an
                # unbounded request.
                while len(buf) > max_chars:
                    segments.append(buf[:max_chars])
                    buf = buf[max_chars:]
            if buf:
                segments.append(buf)
            continue
        if size and size + len(para) + 2 > max_chars:
            flush()
        current.append(para)
        size += len(para) + 2
    flush()
    return segments


@dataclass
class Chapter:
    title: str
    segments: list[str]


def build_script(doc: Document, summary: Summary, *, max_chars: int) -> list[Chapter]:
    """Summary chapter first, then the article — the agreed listening order."""
    title = (doc.title or summary.title or "Untitled").strip()
    summary_text = clean_for_listening(
        f"{title}.\n\nSummary.\n\n{summary.tldr}\n\n{summary.summary}"
    )
    article_text = clean_for_listening(doc.text)
    chapters = [Chapter("Summary", split_segments(summary_text, max_chars))]
    if article_text:
        chapters.append(
            Chapter(
                "Full article",
                split_segments(
                    f"End of summary. The full article begins now.\n\n{article_text}",
                    max_chars,
                ),
            )
        )
    return chapters
