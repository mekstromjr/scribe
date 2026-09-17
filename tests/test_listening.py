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


class TestStructureAwareChapters:
    """One chapter per source section (scribe#3)."""

    def _sections(self, body: str):
        return build_script(_doc(body), _summary(), max_chars=4000)

    def test_markdown_headings_become_chapters(self):
        body = (
            "# Doc Title\n\nLead paragraph. " + "x" * 720 + "\n\n"
            "## Origin\n\n" + "o" * 900 + "\n\n"
            "## Variants\n\n" + "v" * 900 + "\n\n"
            "## Usage\n\n" + "u" * 900
        )
        titles = [c.title for c in self._sections(body)]
        assert titles == ["Summary", "Introduction", "Origin", "Variants", "Usage"]

    def test_lead_before_the_first_heading_is_never_lost(self):
        body = ("# Title\n\nThe definition sentence. " + "x" * 720
                + "\n\n## One\n\n" + "a" * 900 + "\n\n## Two\n\n" + "b" * 900)
        spoken = " ".join(s for c in self._sections(body) for s in c.segments)
        assert "The definition sentence." in spoken

    def test_heading_is_spoken_at_the_top_of_its_chapter(self):
        body = ("## Alpha\n\n" + "a" * 900 + "\n\n## Beta\n\n" + "b" * 900)
        chapters = self._sections(body)
        beta = next(c for c in chapters if c.title == "Beta")
        assert beta.segments[0].startswith("Beta.")

    def test_single_heading_document_stays_flat(self):
        # One heading is not a structure; today's behavior is correct.
        chapters = self._sections("## Only\n\n" + "a" * 900)
        assert [c.title for c in chapters] == ["Summary", "Full article"]

    def test_unstructured_text_stays_flat(self):
        chapters = self._sections("Just prose. " * 200)
        assert [c.title for c in chapters] == ["Summary", "Full article"]

    def test_tiny_sections_merge_into_their_predecessor(self):
        body = ("## Big\n\n" + "a" * 900 + "\n\n## Tiny\n\nshort\n\n"
                "## AlsoBig\n\n" + "b" * 900)
        titles = [c.title for c in self._sections(body)]
        assert "Tiny" not in titles
        assert titles == ["Summary", "Big", "AlsoBig"]
        # Merged, not dropped: the text still gets spoken.
        spoken = " ".join(s for c in self._sections(body) for s in c.segments)
        assert "short" in spoken

    def test_chapter_count_is_capped(self):
        body = "".join(f"## Section {i}\n\n" + "x" * 900 + "\n\n" for i in range(40))
        chapters = self._sections(body)
        assert len(chapters) <= 21  # 20 article chapters + Summary

    def test_long_headings_are_truncated_for_the_player(self):
        body = ("## " + "Very Long Heading " * 8 + "\n\n" + "a" * 900
                + "\n\n## Short\n\n" + "b" * 900)
        title = self._sections(body)[1].title
        assert len(title) <= 63
        assert title.endswith("...")

    def test_pdf_style_numbered_headings_are_detected(self):
        body = ("3.1 Turing Machines\n\n" + "a" * 900
                + "\n\n3.2 Decidability\n\n" + "b" * 900)
        titles = [c.title for c in self._sections(body)]
        assert titles == ["Summary", "3.1 Turing Machines", "3.2 Decidability"]

    def test_sentences_are_not_mistaken_for_pdf_headings(self):
        # Ends with punctuation and reads as prose — must not split here.
        body = ("The Machine Was Running Well.\n\n" + "a" * 900
                + "\n\nAnother Sentence That Ends.\n\n" + "b" * 900)
        assert [c.title for c in self._sections(body)] == ["Summary", "Full article"]


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


class TestPdfReflow:
    """scribe#10: PDFium gives lines, not paragraphs; segments used to end at page
    breaks mid-sentence. Reflow rebuilds paragraphs from line shape."""

    def test_soft_wraps_join_and_ragged_last_line_ends_paragraph(self):
        from scribe.listening import reflow_pdf_text
        page = (
            "This is the first line of a paragraph that wraps across several printed\n"
            "lines because the column is narrow and the sentence is long enough to\n"
            "need it. Here it ends.\n"
            "A new paragraph starts here and also wraps onto the following printed\n"
            "line before it is done."
        )
        out = reflow_pdf_text(page)
        paras = out.split("\n\n")
        assert len(paras) == 2
        assert paras[0].startswith("This is the first") and paras[0].endswith("Here it ends.")
        assert "\n" not in paras[0]

    def test_page_join_inside_a_sentence_is_healed(self):
        from scribe.listening import reflow_pdf_text
        text = (
            "The algorithm proceeds by filling the table row by row, and each entry\n"
            "depends only on entries computed earlier in the same row or the previous\n"
            "\n"
            "row, which is exactly what makes the memoized version fast. That is all."
        )
        out = reflow_pdf_text(text)
        assert "previous row, which" in out
        assert out.count("\n\n") == 0

    def test_hyphenated_wrap_is_rejoined(self):
        from scribe.listening import reflow_pdf_text
        out = reflow_pdf_text("We analyse the dyn-\namic programming table carefully here.")
        assert "dynamic programming" in out

    def test_pseudocode_and_figure_label_runs_are_dropped(self):
        from scribe.listening import reflow_pdf_text
        text = (
            "Unfortunately, this naive recursive algorithm is horribly slow, as we\n"
            "will now see in some detail.\n"
            "F5\nF3 F4\nF2 F1\nF1 F0\nreturn 0\n"
            "Except for the recursive calls, the entire algorithm requires only a\n"
            "constant number of steps to run."
        )
        out = reflow_pdf_text(text)
        assert "F3 F4" not in out and "return 0" not in out
        assert "horribly slow" in out and "constant number" in out

    def test_numbered_heading_gets_its_own_line(self):
        from scribe.listening import reflow_pdf_text
        text = (
            "and so the previous section ends with this sentence.\n"
            "3.2 Aside: Even Faster Fibonacci Numbers\n"
            "The recurrence can be solved faster still, as the following argument\n"
            "shows in some detail."
        )
        paras = reflow_pdf_text(text).split("\n\n")
        assert paras[1] == "3.2 Aside: Even Faster Fibonacci Numbers"


class TestMissingGlyphRepair:
    def test_ligatures_inside_words(self):
        from scribe.listening import _repair_missing_glyphs as fix
        assert fix("their e￾orts at the o￾ce") == "their efforts at the office"
        assert fix("we de￾ne it") == "we define it"

    def test_word_initial_ligature(self):
        from scribe.listening import _repair_missing_glyphs as fix
        assert fix("the ￾￾￾ow of ￾￾￾￾￾￾￾￾ rst") \
            .startswith("the flow of")

    def test_standalone_runs_vanish(self):
        from scribe.listening import _repair_missing_glyphs as fix
        assert fix("n ￾ 1") == "n  1"
        assert fix("Each￾￾See, I told you") == "Each See, I told you"

    def test_unknown_word_still_gets_a_plausible_guess(self):
        from scribe.listening import _repair_missing_glyphs as fix
        assert "￾" not in fix("zorbl￾ng")


class TestSentenceSplit:
    def test_abbreviations_do_not_split(self):
        from scribe.listening import _split_sentences
        s = _split_sentences("As shown by Smith et al. 2020, see Fig. 3 and e.g. Eq. 4. Done here.")
        assert len(s) == 2 and s[0].endswith("Eq. 4.")

    def test_bare_list_marker_stays_with_its_item(self):
        from scribe.listening import _split_sentences
        got = _split_sentences("3. Third item here. Next one.")
        assert got == ["3. Third item here.", "Next one."]

    def test_a_year_ends_a_sentence(self):
        from scribe.listening import _split_sentences
        assert len(_split_sentences("It happened in 2020. Then more.")) == 2


class TestHeadingHeuristicTightening:
    def test_diacritic_residue_is_tolerated(self):
        from scribe.listening import _looks_like_pdf_heading as h
        assert h("3.1. Matr ¯ avr ¯ .tta")

    def test_formula_author_and_table_lines_are_not_headings(self):
        from scribe.listening import _looks_like_pdf_heading as h
        assert not h("0 if j > n")
        assert not h("Aidan N. Gomez∗ †")
        assert not h("EN-DE EN-FR EN-DE EN-FR")

    def test_real_headings_still_pass(self):
        from scribe.listening import _looks_like_pdf_heading as h
        for line in ("3.2 Aside: Even Faster Fibonacci Numbers", "Recursive Structure",
                     "2 Background", "IV. Results"):
            assert h(line), line

    def test_markdown_heading_without_letters_is_not_a_section(self):
        from scribe.listening import detect_sections
        text = "# #&#&\n\nbody one\n\n# &#&#\n\nbody two\n"
        assert detect_sections(text) == []


class TestRepeatedParagraphLint:
    """scribe#13: a paragraph said twice is a defect; catch it before synthesis."""

    LONG = ("The committee reviewed the proposal at length and concluded that the "
            "schedule could not be met without additional staff, which the budget did "
            "not allow for in the current fiscal year, so the item was deferred.")

    def test_duplicate_long_paragraph_is_flagged(self):
        from scribe.listening import Chapter, lint_script
        ch = Chapter("c", [f"{self.LONG}\n\nSomething else entirely.\n\n{self.LONG.upper()}"])
        findings = lint_script([ch])
        assert any(f.startswith("repeated paragraph x1") for f in findings)

    def test_short_repeats_and_unique_text_are_not(self):
        from scribe.listening import Chapter, lint_script
        ch = Chapter("c", ["Chapter One.\n\nBody text here.\n\nChapter One.\n\n" + self.LONG])
        assert not any("repeated" in f for f in lint_script([ch]))

    def test_repeat_across_segments_and_chapters_counts(self):
        from scribe.listening import Chapter, lint_script
        chapters = [Chapter("a", [self.LONG]), Chapter("b", ["intro", self.LONG])]
        assert any("repeated paragraph" in f for f in lint_script(chapters))
