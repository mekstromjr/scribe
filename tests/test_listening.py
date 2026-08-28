"""Listening-script construction. Pure text transforms — nothing to stub."""

from __future__ import annotations

from scribe.document import Document, Method, Page
from scribe.listening import (
    Chapter,
    build_script,
    clean_for_listening,
    lint_script,
    split_segments,
)
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

    def test_sup_blocks_vanish_with_their_contents(self):
        # The first listening test vocalized "sup one" every few words — <sup> blocks
        # are citation markers and must disappear entirely, not just lose their tags.
        raw = 'stupid")<sup>[\\[1\\]](#cite_note-BRich-1)</sup> is a principle'
        cleaned = clean_for_listening(raw)
        assert "sup" not in cleaned
        assert "1" not in cleaned
        assert 'stupid") is a principle' in cleaned

    def test_escaped_citation_links_are_removed(self):
        raw = "in 1960.[\\[2\\]](#cite_note-TDal-2) First seen"
        cleaned = clean_for_listening(raw)
        assert "2" not in cleaned
        assert "cite" not in cleaned
        assert "in 1960. First seen" in cleaned

    def test_angle_bracket_urls_are_removed(self):
        cleaned = clean_for_listening("Johnson (<https://en.wikipedia.org/wiki/Kelly>).")
        assert "http" not in cleaned
        assert "Johnson" in cleaned

    def test_letter_and_note_citations_are_removed_but_sic_stays(self):
        cleaned = clean_for_listening("claim[a] and[note 3] but [sic] stays")
        assert "[a]" not in cleaned
        assert "[note 3]" not in cleaned
        assert "[sic]" in cleaned

    def test_inline_html_tags_lose_brackets_keep_text(self):
        cleaned = clean_for_listening("a <em>stressed</em> word")
        assert cleaned == "a stressed word"

    def test_end_matter_sections_are_dropped(self):
        text = "Real content.\n\n## See also\n\n- Related thing\n\n## References\n\n1. citation"
        cleaned = clean_for_listening(text)
        assert "Real content." in cleaned
        assert "Related thing" not in cleaned
        assert "citation" not in cleaned

    def test_edit_markers_and_anchor_husks_are_removed(self):
        cleaned = clean_for_listening("Heading\n\n[edit]\n\n(#citeref-BRich1-0)text")
        assert "edit" not in cleaned
        assert "cite" not in cleaned
        assert "text" in cleaned


class TestDocumentHygiene:
    """Rules ported from the pre-scribe tts-pipeline (MekVault/Misc/Scripts)."""

    def test_repeated_page_headers_are_removed(self):
        page = "The Journal of Important Things\n\nReal paragraph {n} content here."
        text = "\n\n".join(page.format(n=n) for n in range(4))
        cleaned = clean_for_listening(text)
        assert "Journal of Important Things" not in cleaned
        assert "Real paragraph 2 content here." in cleaned

    def test_page_furniture_lines_vanish(self):
        text = "Real sentence one.\n\n3/54\n\nPage 12 of 54\n\n- 7 -\n\n42\n\nReal sentence two."
        cleaned = clean_for_listening(text)
        for junk in ("3/54", "Page 12", "- 7 -"):
            assert junk not in cleaned
        assert "42" not in cleaned
        assert "Real sentence one." in cleaned
        assert "Real sentence two." in cleaned

    def test_journal_boilerplate_vanishes(self):
        text = ("Findings follow.\n\nDownloaded from science.org at MIT\n\n"
                "Copyright 2024 AAAS\n\nDOI: 10.1126/science.abc123 continues")
        cleaned = clean_for_listening(text)
        assert "Downloaded from" not in cleaned
        assert "Copyright" not in cleaned
        assert "10.1126" not in cleaned
        assert "Findings follow." in cleaned

    def test_photo_credits_and_email_lines_vanish(self):
        text = ("A real thought.\n\nPHOTOGRAPH BY ANNIE LEIBOVITZ\n\n"
                "author@university.edu\n\nAnother real thought.")
        cleaned = clean_for_listening(text)
        assert "LEIBOVITZ" not in cleaned
        assert "@university" not in cleaned
        assert "Another real thought." in cleaned

    def test_hyphenation_across_line_breaks_is_healed(self):
        assert "understanding" in clean_for_listening("deep under-\nstanding of the topic")

    def test_cid_refs_and_timestamps_vanish_inline(self):
        cleaned = clean_for_listening("It was(cid:31) seen 1/30/26, 6:57 PM by all.")
        assert "cid" not in cleaned
        assert "6:57" not in cleaned
        assert "It was seen" in cleaned


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


class TestLintScript:
    def test_clean_script_reports_nothing(self):
        chapters = build_script(_doc("Plain prose, nothing fancy."), _summary(), max_chars=4000)
        assert lint_script(chapters) == []

    def test_finds_residue_the_cleaner_missed(self):
        # Bypasses the cleaner deliberately — the lint is the safety net UNDER it.
        chapters = [
            Chapter("Summary", ["ok <sup>1</sup> and [broken markup( and https://x.y"])
        ]
        findings = lint_script(chapters)
        kinds = " ".join(findings)
        assert "html tag" in kinds
        assert "bracket residue" in kinds
        assert "url" in kinds

    def test_author_brackets_are_not_flagged(self):
        # [sic]/[if]/IPA survive cleaning ON PURPOSE; the lint must not cry wolf.
        chapters = [Chapter("Summary", ["he said [sic] and nature employs [if] one"])]
        assert lint_script(chapters) == []

    def test_findings_carry_counts_and_context(self):
        chapters = [Chapter("Summary", ["a [broken one( b [broken two( c"])]
        findings = lint_script(chapters)
        assert any("x2" in f and "e.g." in f for f in findings)

    def test_real_cleaned_wikipedia_style_text_is_clean(self):
        raw = (
            'KISS ("Keep it simple")<sup>[\\[1\\]](#cite_note-BRich-1)</sup> is a '
            "[design](https://en.wikipedia.org/wiki/Design) principle.[\\[2\\]]"
            "(#cite_note-TDal-2)\n\n## See also\n\n[edit]\n\nRelated."
        )
        chapters = [Chapter("Full article", [clean_for_listening(raw)])]
        assert lint_script(chapters) == []
