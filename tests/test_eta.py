"""The estimator's job is picking the right branch and the right arithmetic; the
constants themselves are measured host facts, so tests pin the FORMULA, not the values.
"""

from __future__ import annotations

import re

import pypdfium2 as pdfium

from scribe.config import Settings
from scribe.eta import _budget_chars, _model_seconds, estimate_seconds, eta_line

SETTINGS = Settings(slack_bot_token="", slack_app_token="", gitlab_token="")

# The deployment overrides context_tokens down to 14000 (insp1's 16k window minus the
# 4-chars/token overshoot margin — see k8s apps/scribe). The chunk-math tests use the
# same shape so they exercise the map-reduce branch the way production reaches it.
DEPLOYED = Settings(
    slack_bot_token="", slack_app_token="", gitlab_token="", context_tokens=14000
)


def test_small_doc_is_not_quoted_the_full_budget_price():
    small = _model_seconds(SETTINGS, 2000)
    full = _model_seconds(SETTINGS, _budget_chars(SETTINGS))
    assert small < full / 3


def test_map_reduce_scales_by_chunk_count_plus_reduce():
    chars = DEPLOYED.chunk_chars * 4  # exactly 4 chunks, and over the 32k-char budget
    assert chars > _budget_chars(DEPLOYED)
    assert _model_seconds(DEPLOYED, chars) == 5 * DEPLOYED.eta_chunk_seconds


def test_boundary_prefers_single_call():
    # At exactly the budget the summarizer still makes one call; the estimate must agree.
    budget = _budget_chars(DEPLOYED)
    assert _model_seconds(DEPLOYED, budget) < 2 * DEPLOYED.eta_chunk_seconds


def test_url_assumes_budget_full_single_call():
    assert estimate_seconds(SETTINGS, "https://example.com/article") == round(
        _model_seconds(SETTINGS, _budget_chars(SETTINGS))
    )


def test_image_charges_one_ocr_page(tmp_path):
    # Suffix routing only — the file is never opened by the estimator.
    est = estimate_seconds(SETTINGS, str(tmp_path / "photo.png"))
    assert est >= SETTINGS.eta_ocr_page_seconds


def test_unreadable_pdf_falls_back_instead_of_raising(tmp_path):
    bad = tmp_path / "not-really.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert estimate_seconds(SETTINGS, str(bad)) > 0


def test_blank_pdf_pages_count_as_ocr(tmp_path):
    # A PDF whose pages carry no text layer: every page should be priced as OCR.
    path = tmp_path / "scanned.pdf"
    pdf = pdfium.PdfDocument.new()
    for _ in range(3):
        pdf.new_page(612, 792)
    pdf.save(str(path))
    pdf.close()
    est = estimate_seconds(SETTINGS, str(path))
    assert est >= 3 * SETTINGS.eta_ocr_page_seconds


def test_eta_line_renders_slack_date_token():
    line = eta_line(600)
    assert re.fullmatch(
        r"Estimated completion: <!date\^\d+\^\{time\}\|in about 10 min>\.", line
    )
