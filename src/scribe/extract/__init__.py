"""Input dispatch: figure out what the caller handed us, route to the right extractor."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from scribe.config import Settings
from scribe.document import Document
from scribe.extract.ocr import ocr_page
from scribe.extract.pdf import PageCache, extract_pdf
from scribe.extract.web import ExtractionError, extract_url

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

__all__ = ["ExtractionError", "PageCache", "extract", "extract_pdf", "extract_url"]


def extract_image(settings: Settings, path: Path) -> Document:
    with Image.open(path) as img:
        page = ocr_page(settings, img, 1)
    return Document(source=path.name, kind="image", title=path.stem, pages=[page])


def extract(settings: Settings, target: str, cache: PageCache | None = None) -> Document:
    """Dispatch on the target: URL, PDF, or image.

    `cache` remembers finished OCR pages across a requeue; only the PDF path uses it, since
    a single image is one page and a URL never OCRs.
    """
    if target.startswith(("http://", "https://")):
        return extract_url(settings, target)

    path = Path(target).expanduser()
    if not path.is_file():
        raise ExtractionError(f"not a file or URL: {target}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(settings, path, cache=cache)
    if suffix in IMAGE_SUFFIXES:
        return extract_image(settings, path)
    raise ExtractionError(f"unsupported file type '{suffix}' — expected a PDF, image, or URL")
