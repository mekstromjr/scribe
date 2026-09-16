"""The full note is delivered as a file in the thread (scribe#5). `md` must be the raw
note so it drops into a vault unchanged; `pdf`/`docx` strip the Obsidian-only syntax
because those readers will never open Obsidian."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from scribe.note_export import FORMATS, ExportError, export_note, portable_markdown

NOTE = """---
tags: [aigen, scribe]
created: 2026-09-16
---
# Title

> [!note]- Full extracted text
> line one
>
> line two

## Summary

Body.
"""


class TestPortableMarkdown:
    def test_frontmatter_is_stripped(self):
        out = portable_markdown(NOTE)
        assert not out.startswith("---")
        assert "tags:" not in out
        assert out.startswith("# Title")

    def test_callout_header_becomes_a_bold_lead_in_the_same_blockquote(self):
        out = portable_markdown(NOTE)
        assert "[!note]" not in out
        assert "> **Full extracted text**\n>\n> line one" in out

    def test_body_without_frontmatter_is_untouched(self):
        assert portable_markdown("# Plain\n\ntext\n") == "# Plain\n\ntext\n"


class TestExportNote:
    def test_md_is_verbatim(self, tmp_path):
        path = export_note(NOTE, "md", stem="t", out_dir=tmp_path)
        assert path.name == "t.md"
        assert path.read_text() == NOTE

    def test_none_and_unknown_are_refused(self, tmp_path):
        for fmt in ("none", "html", ""):
            with pytest.raises(ExportError):
                export_note(NOTE, fmt, stem="t", out_dir=tmp_path)

    def test_formats_are_the_documented_four(self):
        assert FORMATS == ("pdf", "md", "docx", "none")

    def test_missing_pandoc_is_an_export_error(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError("pandoc")

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(ExportError, match="pandoc is not installed"):
            export_note(NOTE, "docx", stem="t", out_dir=tmp_path)

    def test_pandoc_failure_carries_stderr(self, tmp_path, monkeypatch):
        def fail(*a, **kw):
            return subprocess.CompletedProcess(a, 1, stdout="", stderr="bad input")

        monkeypatch.setattr(subprocess, "run", fail)
        with pytest.raises(ExportError, match="bad input"):
            export_note(NOTE, "docx", stem="t", out_dir=tmp_path)

    @pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
    def test_docx_really_renders(self, tmp_path):
        path = export_note(NOTE, "docx", stem="t", out_dir=tmp_path)
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:2] == b"PK"  # a zip container

    @pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
    def test_pdf_really_renders(self, tmp_path):
        try:
            path = export_note(NOTE, "pdf", stem="t", out_dir=tmp_path)
        except ExportError as exc:
            # weasyprint imports, but its pango/cairo stack is missing on this machine
            # (macOS: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib). The image has them.
            if "not installed" in str(exc) or "native libraries" in str(exc):
                pytest.skip(f"weasyprint native libs unavailable: {exc}")
            raise
        assert path.read_bytes()[:5] == b"%PDF-"
