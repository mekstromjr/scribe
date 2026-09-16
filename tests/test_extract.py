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


class TestDemoteHeadings:
    """The model writes its own heading hierarchy, which would otherwise collide with the
    note's `# Title` / `## Summary` structure and flatten Obsidian's outline."""

    def test_shifts_block_so_shallowest_becomes_h3(self):
        from scribe.note import demote_headings

        md = "# Top\n\ntext\n\n## Sub\n\n### Deep\n"
        assert demote_headings(md) == "### Top\n\ntext\n\n#### Sub\n\n##### Deep\n"

    def test_preserves_relative_nesting(self):
        from scribe.note import demote_headings

        md = "## A\n\n#### B\n"
        # A shifts 2->3, so B shifts by the same offset (4->5) rather than being clamped.
        assert demote_headings(md) == "### A\n\n##### B\n"

    def test_noop_when_already_deep_enough(self):
        from scribe.note import demote_headings

        md = "### Already fine\n"
        assert demote_headings(md) == md

    def test_clamps_at_h6(self):
        from scribe.note import demote_headings

        assert demote_headings("# a\n\n###### deep\n") == "### a\n\n###### deep\n"

    def test_ignores_hash_not_at_line_start(self):
        from scribe.note import demote_headings

        md = "# Real\n\nsee issue #42 inline\n"
        assert "#42" in demote_headings(md)


class TestNoteTitle:
    """The source's own title wins: the name you go looking for later is the one the
    article actually had, not the model's paraphrase of it."""

    def test_prefers_the_source_title(self):
        from scribe.note import note_title
        from scribe.summarize import Summary

        doc = Document(source="https://x", kind="link", title="Machines of Loving Grace")
        s = Summary(title="A Vision for AI's Positive Impact", tldr="", summary="")
        assert note_title(doc, s) == "Machines of Loving Grace"

    def test_falls_back_to_the_model_when_source_has_none(self):
        from scribe.note import note_title
        from scribe.summarize import Summary

        doc = Document(source="https://x", kind="link", title=None)
        s = Summary(title="Model Title", tldr="", summary="")
        assert note_title(doc, s) == "Model Title"

    def test_treats_a_blank_source_title_as_absent(self):
        from scribe.note import note_title
        from scribe.summarize import Summary

        doc = Document(source="https://x", kind="link", title="   ")
        s = Summary(title="Model Title", tldr="", summary="")
        assert note_title(doc, s) == "Model Title"


class TestObsidianUri:
    def test_builds_a_deep_link(self):
        from scribe.note import obsidian_uri

        uri = obsidian_uri("MekVault", "+/My Note.md")
        assert uri == "obsidian://open?vault=MekVault&file=%2B%2FMy%20Note"

    def test_drops_the_md_suffix(self):
        """Obsidian resolves by note name; leaving .md on makes the link miss."""
        from scribe.note import obsidian_uri

        assert obsidian_uri("V", "+/N.md").endswith("%2FN")

    def test_encodes_vault_names_with_spaces(self):
        from scribe.note import obsidian_uri

        assert "vault=My%20Vault" in obsidian_uri("My Vault", "+/N.md")


class TestTitleLadder:
    """scribe#9: metadata title, then the model's, then the filename stem. '03-dynprog'
    is not a title."""

    def _doc(self, source, title, kind="pdf"):
        from scribe.document import Document
        return Document(source=source, kind=kind, title=title)

    def _sum(self, title):
        from scribe.summarize import Summary
        return Summary(title=title, tldr="", summary="")

    def test_junk_filter(self):
        from scribe.note import is_junk_title

        for junk in ("03-dynprog", "monetary20250618a1", "Microsoft Word - final.docx",
                     "untitled", "IMG_4821", "", "  ", "x", "report_v2.pdf", "2024-01-15"):
            assert is_junk_title(junk), junk
        for real in ("Federal Reserve issues FOMC statement", "Liber Abaci", "Dune",
                     "Attention Is All You Need", "Algorithms, Chapter 3: Dynamic Programming"):
            assert not is_junk_title(real), real

    def test_pdf_metadata_title_wins(self):
        from scribe.note import note_title
        doc = self._doc("monetary20250618a1.pdf", "Federal Reserve issues FOMC statement")
        assert note_title(doc, self._sum("FOMC Holds Rates")) == \
            "Federal Reserve issues FOMC statement"

    def test_no_metadata_falls_to_model_not_filename(self):
        from scribe.note import note_title
        doc = self._doc("03-dynprog.pdf", None)
        assert note_title(doc, self._sum("Algorithms, Chapter 3: Dynamic Programming")) == \
            "Algorithms, Chapter 3: Dynamic Programming"

    def test_filename_stem_is_the_last_resort(self):
        from scribe.note import note_title
        doc = self._doc("03-dynprog.pdf", None)
        assert note_title(doc, self._sum("")) == "03-dynprog"

    def test_junk_model_title_falls_to_stem(self):
        from scribe.note import note_title
        doc = self._doc("Quarterly Letter.pdf", None)
        assert note_title(doc, self._sum("untitled")) == "Quarterly Letter"

    @pytest.mark.skipif(not SAMPLES.exists(), reason="recipe-pipeline samples not present")
    def test_metadata_title_reader_on_a_real_file(self):
        """The sample recipe PDF carries no Title; the reader yields None so the model's
        title takes over. (Not via extract_pdf: that sample has no text layer and would
        try to OCR.)"""
        import pypdfium2 as pdfium

        from scribe.extract.pdf import _metadata_title
        pdf = pdfium.PdfDocument(str(SAMPLES / "mediterranean-potato-salad.pdf"))
        try:
            got = _metadata_title(pdf)
        finally:
            pdf.close()
        assert got is None or not got.lower().startswith("mediterranean-potato")

    def test_image_has_no_filename_title(self, tmp_path):
        from scribe.document import Document
        from scribe.note import note_title
        doc = Document(source="IMG_4821.jpg", kind="image", title=None)
        assert note_title(doc, self._sum("Whiteboard: Sprint Plan")) == "Whiteboard: Sprint Plan"
