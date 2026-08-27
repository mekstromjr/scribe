"""Listening-script construction. Pure text transforms — nothing to stub."""

from __future__ import annotations

from scribe.document import Document, Method, Page
from scribe.listening import build_script, clean_for_listening, split_segments
from scribe.summarize import Summary


def _doc(text: str) -> Document:
    return Document(
        source="https://example.com/a", kind="link", title="Test Article",
        pages=[Page(number=1, text=text, method=Method.WEB)],
    )


def _summary() -> Summary:
    return Summary(title="Test Article", tldr="Short version.", summary="## Key\n- point")


class TestCleanForListening:
    def test_links_keep_their_label_and_lose_their_url(self):
        assert clean_for_listening("see [the docs](https://x.y/z) here") == "see the docs here"

    def test_images_vanish_entirely(self):
        cleaned = clean_for_listening("a ![alt text](img.png) b")
        assert "img.png" not in cleaned
        assert "alt text" not in cleaned

    def test_bare_urls_are_removed(self):
        assert "https" not in clean_for_listening("go to https://example.com/x now")

    def test_citation_markers_are_removed_but_bracketed_words_survive(self):
        cleaned = clean_for_listening("known fact [12] and [^3] but [sic] stays")
        assert "[12]" not in cleaned
        assert "[^3]" not in cleaned
        assert "[sic]" in cleaned

    def test_code_fences_are_announced_not_read(self):
        text = "intro\n```python\nx = 1\n```\noutro"
        cleaned = clean_for_listening(text)
        assert "x = 1" not in cleaned
        assert "Code example omitted" in cleaned

    def test_headings_become_spoken_sentences(self):
        assert clean_for_listening("## The Middle Section") == "The Middle Section."

    def test_caption_lines_are_dropped(self):
        text = "Real paragraph one.\n\nPhoto: a cat on a keyboard\n\nReal paragraph two."
        cleaned = clean_for_listening(text)
        assert "cat on a keyboard" not in cleaned
        assert "Real paragraph one." in cleaned
        assert "Real paragraph two." in cleaned

    def test_photography_sentence_mid_paragraph_survives(self):
        # The caption rule is line-anchored: prose ABOUT photos is not a caption.
        long_line = ("Photography changed the war because " + "x" * 220)
        assert "Photography changed" in clean_for_listening(long_line)

    def test_emphasis_and_lists_read_as_plain_prose(self):
        cleaned = clean_for_listening("- **bold** item\n- *soft* item")
        assert "*" not in cleaned
        assert "bold item" in cleaned


class TestSplitSegments:
    def test_short_text_is_one_segment(self):
        assert split_segments("hello world", 100) == ["hello world"]

    def test_paragraphs_pack_up_to_the_cap(self):
        paras = [f"para {i} " + "x" * 40 for i in range(6)]
        segments = split_segments("\n\n".join(paras), 120)
        assert len(segments) > 1
        assert all(len(s) <= 120 for s in segments)
        # Nothing lost.
        joined = " ".join(segments)
        for i in range(6):
            assert f"para {i}" in joined

    def test_oversized_paragraph_splits_at_sentences(self):
        para = " ".join(f"Sentence number {i}." for i in range(50))
        segments = split_segments(para, 200)
        assert all(len(s) <= 200 for s in segments)
        assert " ".join(segments).count("Sentence number") == 50

    def test_single_monster_sentence_is_hard_cut_not_unbounded(self):
        segments = split_segments("x" * 950, 300)
        assert all(len(s) <= 300 for s in segments)
        assert sum(len(s) for s in segments) == 950


class TestBuildScript:
    def test_summary_chapter_comes_first(self):
        chapters = build_script(_doc("Body text here."), _summary(), max_chars=4000)
        assert [c.title for c in chapters] == ["Summary", "Full article"]
        assert "Short version." in chapters[0].segments[0]
        # The document title opens the audio — the listener should know what this is.
        assert chapters[0].segments[0].startswith("Test Article.")

    def test_article_chapter_announces_the_transition(self):
        chapters = build_script(_doc("Body text here."), _summary(), max_chars=4000)
        assert chapters[1].segments[0].startswith("End of summary.")

    def test_empty_article_text_yields_summary_only(self):
        chapters = build_script(_doc("   "), _summary(), max_chars=4000)
        assert [c.title for c in chapters] == ["Summary"]
