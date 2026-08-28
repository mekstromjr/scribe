"""Estimate how long a document will take, before any model work starts.

The ack is the only feedback a phone user gets for the next N minutes, so it should say
when to check back. The estimate is a piecewise-linear fit over the two things that are
knowable at enqueue time without paying for them: extracted character count and how many
pages will need OCR.

Everything here must stay CHEAP. Reading a PDF's text layer is milliseconds; OCR is
minutes. The sizing pass mirrors extract_pdf's per-page min_page_chars decision without
ever rendering a page, so the estimate covers exactly the pages the extractor will later
send to the vision model. URLs are the one input we refuse to size (fetching the page to
estimate it would do the extraction's network work twice) — they get the single-call
constant, which is right for nearly every article.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pypdfium2 as pdfium

from scribe.config import Settings
from scribe.extract import IMAGE_SUFFIXES

# Rough text yield of one OCR'd page, for sizing the summarize step that follows the
# OCR. Only the summarize cost scales with this, so precision barely matters: at the
# measured per-char rate an error of 1,000 chars moves the estimate by ~20s.
_OCR_CHARS_PER_PAGE = 1800


def _budget_chars(settings: Settings) -> int:
    # Mirrors summarize.py: the threshold where a document stops fitting one call and
    # goes map-reduce. Keep the formulas identical or the estimate will pick the wrong
    # branch for documents near the boundary.
    return int((settings.context_tokens - settings.response_reserve_tokens) * 4)


def _model_seconds(settings: Settings, chars: int) -> float:
    """Cost of the summarize step alone for an extracted text of ``chars``."""
    if chars <= _budget_chars(settings):
        return settings.eta_single_base_seconds + chars * settings.eta_single_seconds_per_char
    chunks = math.ceil(chars / settings.chunk_chars)
    # +1 is the reduce call, which is a chunk-sized request in its own right.
    return (chunks + 1) * settings.eta_chunk_seconds


def _scan_pdf(settings: Settings, path: Path) -> tuple[int, int]:
    """(text_layer_chars, pages_needing_ocr) — text layer only, never renders a page."""
    chars = 0
    ocr_pages = 0
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in pdf:
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range() or ""
            finally:
                textpage.close()
            if len("".join(text.split())) >= settings.min_page_chars:
                chars += len(text)
            else:
                ocr_pages += 1
    finally:
        pdf.close()
    if settings.max_ocr_pages:
        ocr_pages = min(ocr_pages, settings.max_ocr_pages)
    return chars, ocr_pages


def estimate_seconds(settings: Settings, target: str) -> int:
    """Estimated processing seconds for one job, excluding queue wait.

    Never raises: an unreadable file is the pipeline's error to report, not the ack's —
    fall back to the single-call constant rather than blocking the acknowledgement.
    """
    try:
        if target.startswith(("http://", "https://")):
            chars = _budget_chars(settings)  # size unknown; assume a budget-full article
            return round(_model_seconds(settings, chars))

        path = Path(target).expanduser()
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return round(
                settings.eta_ocr_page_seconds
                + _model_seconds(settings, _OCR_CHARS_PER_PAGE)
            )

        chars, ocr_pages = _scan_pdf(settings, path)
        chars += ocr_pages * _OCR_CHARS_PER_PAGE
        return round(ocr_pages * settings.eta_ocr_page_seconds + _model_seconds(settings, chars))
    except Exception:
        return round(_model_seconds(settings, _budget_chars(settings)))


def eta_line(settings: Settings, total_seconds: float, tz: str | None = None) -> str:
    """One sentence with a 24-hour clock.

    Rendered in ``tz`` — normally the Slack profile timezone of whoever sent the
    message, which Slack keeps current as they travel — falling back to
    settings.timezone when the profile has none or names a zone this host does not
    know. Slack's <!date^...^{time}> token renders viewer-local but only in 12-hour
    format, which is why the clock is server-side at all. An ETA that lands on a
    different calendar day says so, or "23:58" quoted at 23:50 would read as fourteen
    hours away.
    """
    try:
        zone = ZoneInfo(tz) if tz else ZoneInfo(settings.timezone)
    except KeyError:
        zone = ZoneInfo(settings.timezone)
    now = datetime.now(tz=zone)
    done = now + timedelta(seconds=total_seconds)
    day = " tomorrow" if done.date() != now.date() else ""
    return f"Estimated completion: {done:%H:%M}{day}."
