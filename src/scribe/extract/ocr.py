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
    return VisionResult(
        text=normalize_latex(result.text).strip(),
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
