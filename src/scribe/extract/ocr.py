"""Vision-model OCR — the fallback for pages with no usable text layer."""

from __future__ import annotations

import io
import logging
import re

from PIL import Image

from scribe.config import Settings
from scribe.document import Method, Page
from scribe.ollama import OllamaTimeout, VisionResult, generate_with_image

log = logging.getLogger("scribe.ocr")

PROMPT = (
    "Transcribe all text in this image exactly as written. Preserve line breaks and "
    "reading order. Output only the transcription, with no commentary."
)

# glm-ocr emits LaTeX for math glyphs: a "1/2" on the page comes back as "$\frac{1}{2}$".
# In Obsidian that renders as MathJax rather than a fraction, so normalize to plain text.
_FRAC = re.compile(r"\$?\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}\$?")
_BARE_MATH = re.compile(r"\$([^$\n]{1,40})\$")


# A repeated block this long is the model looping, not the page saying something
# twice: a running header repeats, a paragraph of body text does not.
_REPEAT_WINDOW = 160


def collapse_repetition(text: str) -> tuple[str, bool]:
    """Cut an OCR transcription at the point where it starts repeating itself.

    Vision models loop: on 2026-09-16 glm-ocr transcribed a textbook page, then its
    running header, then the whole page again, all under the token cap, and two
    minutes of audio played twice (scribe#13). The generation-cap guard only catches
    loops that overrun the cap. Here, if a 160-char window starting at some line
    recurs later at a line start, everything from that second occurrence on is
    dropped, plus one short line just before it (the running header that introduced
    the loop). Returns (text, cut).
    """
    lines = text.split("\n")
    norms = [" ".join(ln.split()) for ln in lines]
    # Flat text and the flat offset at which each non-empty line begins.
    offsets: dict[int, int] = {}
    parts: list[str] = []
    pos = 0
    for i, n in enumerate(norms):
        if not n:
            continue
        offsets[i] = pos
        parts.append(n)
        pos += len(n) + 1
    flat = " ".join(parts)
    if len(flat) < 2 * _REPEAT_WINDOW:
        return text, False
    starts = {off: i for i, off in offsets.items()}
    for i, off in offsets.items():
        probe = flat[off: off + _REPEAT_WINDOW]
        if len(probe) < _REPEAT_WINDOW:
            break
        again = flat.find(probe, off + 1)
        if again < 0:
            continue
        # The repeat begins at (or just inside) a later line: cut at that line.
        cut = starts.get(again)
        if cut is None:
            later = [k for o, k in starts.items() if o <= again and k > i]
            if not later:
                continue
            cut = max(later)
        if cut > 0 and 0 < len(norms[cut - 1]) <= 60:
            cut -= 1  # the running header that led the loop
        return "\n".join(lines[:cut]).rstrip(), True
    return text, False


def normalize_latex(text: str) -> str:
    """Undo the LaTeX-isms glm-ocr introduces for ordinary typographic glyphs."""
    text = _FRAC.sub(r"\1/\2", text)
    # Strip $...$ wrappers left around plain content (e.g. "$2$" -> "2"). Only short spans,
    # so genuine inline math in a lecture slide is left intact.
    return _BARE_MATH.sub(r"\1", text)


def downscale(image: Image.Image, max_edge: int) -> Image.Image:
    """Cap the longest edge. Full 300-DPI renders cost real CPU in the vision encoder for
    no accuracy gain at this model size."""
    if max(image.size) <= max_edge:
        return image
    ratio = max_edge / max(image.size)
    new = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(new, Image.LANCZOS)


def ocr_image(settings: Settings, image: Image.Image) -> VisionResult:
    """OCR one page image. The returned text is normalized; `complete` is passed through."""
    prepared = downscale(image.convert("RGB"), settings.ocr_max_edge)
    buf = io.BytesIO()
    prepared.save(buf, "PNG")
    result = generate_with_image(settings, PROMPT, buf.getvalue())
    text, cut = collapse_repetition(result.text)
    if cut:
        log.warning("OCR output repeated itself; kept the first %d of %d chars",
                    len(text), len(result.text))
    return VisionResult(
        text=normalize_latex(text).strip(),
        seconds=result.seconds,
        complete=result.complete,
    )


def ocr_page(settings: Settings, image: Image.Image, number: int) -> Page:
    """OCR one page into a Page, turning the two runaway signals into SKIPPED pages.

    A page that hits the generation cap or the OCR timeout is a page the model could not
    finish -- a repetition loop on sparse content (scribe#4). Both are deterministic at
    temperature 0, so neither is allowed to raise: raising would requeue the whole job and
    redo every good page only to hang on the same one again. The page is recorded as
    SKIPPED with a reason, the rest of the document proceeds, and the note says so.

    A genuinely unreachable server (connection refused, 5xx) still raises OllamaError and
    still requeues -- that IS transient.
    """
    try:
        result = ocr_image(settings, image)
    except OllamaTimeout as exc:
        log.warning("page %d: %s — skipping", number, exc)
        return Page(
            number=number, text="", method=Method.SKIPPED,
            seconds=settings.ocr_timeout_seconds,
            reason=f"OCR timed out after {settings.ocr_timeout_seconds:.0f}s",
        )
    if not result.complete:
        # The truncated text is discarded, not kept: what the model produced before the
        # cap is repetition junk, and junk in the summary is worse than a declared gap.
        log.warning(
            "page %d: OCR hit the %d-token generation cap after %.0fs — skipping",
            number, settings.ocr_num_predict, result.seconds,
        )
        return Page(
            number=number, text="", method=Method.SKIPPED, seconds=result.seconds,
            reason=f"OCR runaway, cut at {settings.ocr_num_predict} tokens",
        )
    return Page(number=number, text=result.text, method=Method.OCR, seconds=result.seconds)
