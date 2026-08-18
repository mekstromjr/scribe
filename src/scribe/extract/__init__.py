"""Input dispatch: figure out what the caller handed us, route to the right extractor."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from scribe.config import Settings
from scribe.document import Document, Method, Page
from scribe.extract.ocr import ocr_image
from scribe.extract.pdf import extract_pdf
from scribe.extract.web import ExtractionError, extract_url

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

__all__ = ["ExtractionError", "extract", "extract_pdf", "extract_url"]


def extract_image(settings: Settings, path: Path) -> Document:
    t0 = time.monotonic()
    with Image.open(path) as img:
        text, seconds = ocr_image(settings, img)
    return Document(
        source=path.name,
        kind="image",
        title=path.stem,
        pages=[
            Page(
                number=1,
                text=text,
                method=Method.OCR,
                seconds=seconds or time.monotonic() - t0,
            )
        ],
    )


def extract(settings: Settings, target: str) -> Document:
    """Dispatch on the target: URL, PDF, or image."""
    if target.startswith(("http://", "https://")):
        return extract_url(settings, target)

    path = Path(target).expanduser()
    if not path.is_file():
        raise ExtractionError(f"not a file or URL: {target}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(settings, path)
    if suffix in IMAGE_SUFFIXES:
        return extract_image(settings, path)
    raise ExtractionError(f"unsupported file type '{suffix}' — expected a PDF, image, or URL")
