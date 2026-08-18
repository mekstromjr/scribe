"""Vision-model OCR — the fallback for pages with no usable text layer."""

from __future__ import annotations

import io
import re

from PIL import Image

from scribe.config import Settings
from scribe.ollama import generate_with_image

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


def ocr_image(settings: Settings, image: Image.Image) -> tuple[str, float]:
    """OCR one page image. Returns (normalized_text, wall_clock_seconds)."""
    prepared = downscale(image.convert("RGB"), settings.ocr_max_edge)
    buf = io.BytesIO()
    prepared.save(buf, "PNG")
    raw, seconds = generate_with_image(settings, PROMPT, buf.getvalue())
    return normalize_latex(raw).strip(), seconds
