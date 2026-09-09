"""PDF extraction: read the text layer first, fall back to OCR only where there isn't one.

This ordering is the single most important performance decision in scribe. Lecture slides
exported from PowerPoint/Keynote/LaTeX carry selectable text, and reading it is both
lossless and effectively instant. Running the vision model over those pages instead costs
~20s each AND is worse — vision models paraphrase and drop content, while PDFium returns
exactly what is embedded.

OCR is therefore the exception (scans, image-only slides), not the default.

Page rendering follows recipe-pipeline's `worker/src/recipe_pipeline/pdf.py`, which uses
pypdfium2 so there is no system dependency on Poppler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pypdfium2 as pdfium

from scribe.config import Settings
from scribe.document import Document, Method, Page
from scribe.extract.ocr import ocr_page


def _page_text(page: pdfium.PdfPage) -> str:
    """Extract the embedded text layer for one page, or '' if there is none."""
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range() or ""
    finally:
        textpage.close()


class PageCache(Protocol):
    """Where finished OCR pages are remembered across a requeue. See scribe.queue."""

    def get(self, number: int) -> Page | None: ...
    def put(self, page: Page) -> None: ...


def extract_pdf(settings: Settings, path: Path, cache: PageCache | None = None) -> Document:
    doc = Document(source=path.name, kind="pdf", title=path.stem)
    pdf = pdfium.PdfDocument(str(path))
    ocr_used = 0
    try:
        for index, page in enumerate(pdf, start=1):
            text = _page_text(page)

            # Whitespace is not content: a page of blank lines has a "text layer" that
            # tells us nothing, so count only non-whitespace characters.
            if len("".join(text.split())) >= settings.min_page_chars:
                doc.pages.append(Page(number=index, text=text.strip(), method=Method.TEXT_LAYER))
                continue

            if settings.max_ocr_pages and ocr_used >= settings.max_ocr_pages:
                # Record the gap rather than silently truncating — a note that quietly
                # omits half a document is worse than one that says it did.
                doc.pages.append(
                    Page(number=index, text="", method=Method.SKIPPED, reason="OCR page cap")
                )
                continue

            # Only OCR'd pages are cached: the text layer is free to re-read, OCR is
            # minutes per page and deterministic, so a requeued job must not pay for
            # pages it already finished (scribe#4).
            if cache is not None and (cached := cache.get(index)) is not None:
                doc.pages.append(cached)
                ocr_used += 1
                continue

            bitmap = page.render(scale=settings.ocr_render_dpi / 72)
            result = ocr_page(settings, bitmap.to_pil(), index)
            ocr_used += 1
            doc.pages.append(result)
            if cache is not None:
                cache.put(result)
    finally:
        pdf.close()
    return doc


def has_text_layer(path: Path, min_chars: int = 1) -> bool:
    """Cheap probe used by tests and the CLI's --dry-run to classify a PDF without OCR."""
    pdf = pdfium.PdfDocument(str(path))
    try:
        total = 0
        for page in pdf:
            total += len("".join(_page_text(page).split()))
            if total >= min_chars:
                return True
        return False
    finally:
        pdf.close()
