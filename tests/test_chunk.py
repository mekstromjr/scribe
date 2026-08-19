"""Chunking for the map-reduce fallback. Pure text handling — no model calls."""

from __future__ import annotations

from scribe.chunk import chunk

HEADED = "# One\n\nalpha alpha\n\n## Two\n\nbeta beta\n\n## Three\n\ngamma gamma\n"


class TestNoChunkingWhenItFits:
    """Single pass is both faster AND better — the model sees the whole argument. Chunking
    is only for the case where the alternative is losing the tail."""

    def test_short_text_is_one_chunk(self):
        assert chunk("short", 1000) == ["short"]

    def test_exactly_at_the_limit_is_one_chunk(self):
        text = "x" * 100
        assert chunk(text, 100) == [text]


class TestSemanticSeams:
    def test_splits_on_headings_not_mid_sentence(self):
        chunks = chunk(HEADED, 30)
        # Every chunk after the first should begin at a heading, never mid-sentence.
        assert all(c.startswith("#") for c in chunks[1:]), chunks

    def test_falls_back_to_paragraphs_without_headings(self):
        text = "para one here\n\npara two here\n\npara three here"
        chunks = chunk(text, 20)
        assert len(chunks) > 1
        assert all("para" in c for c in chunks)

    def test_packs_greedily_rather_than_evenly(self):
        """A chunk ending at a heading is worth more than one of uniform size."""
        chunks = chunk(HEADED, 60)
        assert len(chunks) < len(HEADED.split("\n\n"))


class TestCoverage:
    """The whole point: nothing may be dropped. Truncation loses the conclusion, which is
    the failure this replaces."""

    def test_no_content_is_lost(self):
        joined = "".join(chunk(HEADED, 30))
        for token in ("alpha", "beta", "gamma"):
            assert token in joined

    def test_every_chunk_is_within_the_limit(self):
        for c in chunk(HEADED, 30):
            assert len(c) <= 30, c

    def test_oversized_single_segment_is_hard_split(self):
        """A wall of unbroken prose has no seam. A mid-sentence break beats silently
        exceeding the context window."""
        wall = "x" * 250
        chunks = chunk(wall, 100)
        assert len(chunks) == 3
        assert "".join(chunks) == wall

    def test_handles_text_with_no_seams_at_all(self):
        assert chunk("abc", 100) == ["abc"]
