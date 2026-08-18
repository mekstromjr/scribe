"""Unit tests for the extraction layer. No network, no Ollama, no cluster."""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.config import Settings
from scribe.document import Document, Method, Page
from scribe.extract import IMAGE_SUFFIXES, extract
from scribe.extract.ocr import normalize_latex
from scribe.extract.web import ExtractionError

SAMPLES = Path.home() / "Dev/meklab/recipe-pipeline/samples"


class TestNormalizeLatex:
    """glm-ocr emits LaTeX for ordinary typographic glyphs; Obsidian would render it as
    MathJax rather than the fraction the page actually shows."""

    def test_frac_becomes_plain(self):
        assert normalize_latex(r"$\frac{1}{2}$ cup basil") == "1/2 cup basil"

    def test_dfrac_variant(self):
        assert normalize_latex(r"\dfrac{3}{4}") == "3/4"

    def test_strips_trivial_math_wrapper(self):
        assert normalize_latex("about $350$ degrees") == "about 350 degrees"

    def test_leaves_plain_text_untouched(self):
        text = "2 tablespoons white balsamic vinegar"
        assert normalize_latex(text) == text

    def test_leaves_long_math_spans_alone(self):
        # A genuine equation on a lecture slide should survive; only short wrappers are
        # unwrapped, so this stays as-is.
        eq = "$x = \\sum_{i=0}^{n} a_i b_i \\text{ for all } i \\in S \\text{ where } n > 100$"
        assert normalize_latex(eq) == eq


class TestDispatch:
    def test_rejects_unsupported_suffix(self, tmp_path):
        f = tmp_path / "notes.docx"
        f.write_text("x")
        with pytest.raises(ExtractionError, match="unsupported file type"):
            extract(Settings(), str(f))

    def test_rejects_missing_file(self):
        with pytest.raises(ExtractionError, match="not a file or URL"):
            extract(Settings(), "/nonexistent/nope.pdf")

    def test_image_suffixes_cover_common_slack_uploads(self):
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            assert suffix in IMAGE_SUFFIXES


class TestDocument:
    def test_text_joins_only_nonempty_pages(self):
        doc = Document(
            source="x.pdf",
            kind="pdf",
            pages=[
                Page(number=1, text="alpha", method=Method.TEXT_LAYER),
                Page(number=2, text="   ", method=Method.SKIPPED),
                Page(number=3, text="beta", method=Method.OCR, seconds=20.0),
            ],
        )
        assert doc.text == "alpha\n\nbeta"
        assert doc.ocr_pages == 1
        assert doc.seconds == 20.0

    def test_summary_reports_mixed_extraction(self):
        doc = Document(
            source="x.pdf",
            kind="pdf",
            pages=[
                Page(number=1, text="a", method=Method.TEXT_LAYER),
                Page(number=2, text="b", method=Method.OCR),
            ],
        )
        assert "mixed" in doc.summary_line()


@pytest.mark.skipif(not SAMPLES.exists(), reason="recipe-pipeline samples not present")
class TestTextLayerDetection:
    """The text-layer probe is what keeps OCR off the common path, so it is worth a test
    against a real image-only PDF rather than a synthetic one."""

    def test_scanned_pdf_has_no_text_layer(self):
        from scribe.extract.pdf import has_text_layer

        assert has_text_layer(SAMPLES / "mediterranean-potato-salad.pdf") is False
