"""Split an over-long document into summarizable sections.

Only used when a document would otherwise be TRUNCATED. Truncation drops the tail, which
for an essay is usually the conclusion — often the most valuable part. A lower-resolution
summary of the whole beats a sharp summary of the first three quarters.

Splitting prefers semantic boundaries. trafilatura returns markdown, so markdown headings
are the best seams available; paragraph breaks are the fallback. A chunk that begins
mid-sentence produces a summary that begins mid-thought.
"""

from __future__ import annotations

import re

# A markdown heading at line start. Headings are the strongest seam in extracted articles.
_HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)


def _split_on(text: str, positions: list[int]) -> list[str]:
    parts = []
    for start, end in zip([0, *positions], [*positions, len(text)], strict=True):
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
    return parts


def _segments(text: str) -> list[str]:
    """Break text at the strongest available seams, finest granularity last."""
    heads = [m.start() for m in _HEADING.finditer(text)][1:]  # keep the first heading
    if heads:
        return _split_on(text, heads)
    paras = [m.start() for m in re.finditer(r"\n\s*\n", text)]
    if paras:
        return _split_on(text, paras)
    return [text]


def chunk(text: str, max_chars: int) -> list[str]:
    """Group the document into chunks of at most ``max_chars``, on semantic seams.

    Segments are packed greedily rather than split evenly: a chunk that ends at a heading
    is worth more than one of uniform size. A single segment longer than the limit (a wall
    of unbroken prose) is hard-split as a last resort — better a mid-sentence break than
    silently exceeding the context window, which is the failure this exists to avoid.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for seg in _segments(text):
        while len(seg) > max_chars:
            # Oversized single segment: emit what we have, then hard-split.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(seg[:max_chars])
            seg = seg[max_chars:]
        if not current:
            current = seg
        elif len(current) + 2 + len(seg) <= max_chars:
            current = f"{current}\n\n{seg}"
        else:
            chunks.append(current)
            current = seg
    if current:
        chunks.append(current)
    return chunks
