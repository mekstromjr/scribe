"""Build the "listening script" — the text that actually gets spoken.

Deliberately rule-based, not an LLM pass. Rewriting a full article "for listening" with
the CPU model would roughly double every job's wall clock (generation is the expensive
direction at ~15 tok/s, and a rewrite generates the WHOLE document, not a summary) and
would risk silent paraphrasing — the reader should hear the article, not the model's
memory of it. Deterministic stripping removes the things that are genuinely painful in
audio (URLs, citation markers, figure captions, markdown syntax) and nothing else. An
LLM polish pass can slot in here if the pipeline moves to a cloud model.

The script is split into chapters (Summary first, then the article) so the m4b gets
chapter markers — one tap in the Audiobookshelf player replays or skips the summary.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from scribe.document import Document
from scribe.note import note_title
from scribe.summarize import Summary

# HTML that survives extraction (trafilatura keeps some inline tags). <sup> blocks go
# ENTIRELY — in extracted articles they are footnote/citation markers, and hearing
# "sup one" every few words is what made the first test unlistenable. Other inline
# tags lose only their brackets.
_SUP = re.compile(r"<sup\b[^>]*>.*?</sup>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(r"</?(?:sub|span|em|strong|i|b|u|small|br|a)\b[^>]*/?>", re.IGNORECASE)
# trafilatura backslash-escapes literal brackets (citations arrive as
# "[\[1\]](#cite_note-...)"); unescape FIRST so the link and citation rules see them.
_ESCAPED_BRACKET = re.compile(r"\\([\[\]])")
_ANGLE_URL = re.compile(r"<https?://[^>]+>")

# Markdown constructs, in stripping order (images before links — an image IS a link
# with a bang, and the link rule alone would leave its alt text plus a stray '!').
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Label may be EMPTY: Wikipedia citations arrive as links whose label is itself a
# bracketed marker ("[[2]](#cite_note-…)"), so the citation rule runs first, empties
# the label, and this rule then swallows the husk.
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1")
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

# Citation/footnote markers: [1], [12], [^3], [a], [note 4] — but not [sic]-style
# bracketed words, which are the author's own text. Runs BEFORE the link rule: a
# Wikipedia citation is a link whose label is the marker ("[[1]](#cite_note-…)"), and
# removing the marker first leaves an empty-labeled link the link rule then swallows.
_CITATION = re.compile(r"\[\^?(?:\d+|[a-z]|note \d+)\]", re.IGNORECASE)

# Caption-ish lines. Anchored to line start and short lines only: an article ABOUT
# photography legitimately starts sentences with "Photo" mid-paragraph; a caption is a
# short standalone line.
_CAPTION = re.compile(
    r"^(?:figure|fig\.|photo|image|photograph|illustration|source|credit|caption)\b[^\n]{0,200}$",
    re.IGNORECASE | re.MULTILINE,
)

_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# --- Document hygiene -----------------------------------------------------------
# Ported from Michael's pre-scribe TTS pipeline (MekVault/Misc/Scripts/tts-pipeline,
# text_cleaner.py) — rules battle-tested against exactly the junk that made Speech
# Central unbearable: journal boilerplate, page furniture, photo credits. Curated:
# the chart-data heuristics and aggressive line-rejoining stayed behind (higher
# false-positive risk than their payoff here, where extraction is already cleaner).
_HYGIENE_LINE_RULES = [
    # PDF page furniture
    re.compile(r"^\s*\d+/\d+\s*$", re.MULTILINE),                      # "3/54"
    re.compile(r"^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),                  # "- 12 -"
    re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),                      # bare page number
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$", re.MULTILINE),     # print-dialog date
    # Journal boilerplate
    re.compile(r"^\s*VOL\.?\s+\d+\s+ISSUE\s+\d+.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Downloaded from\s+\S+.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(
        r"^\s*Published by\s+(?:American Association|AAAS|Wiley|Elsevier|Springer|Nature).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:Copyright|©|\(c\))\s+\d{4}.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*All rights reserved\.?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Reprints?\s+and\s+[Pp]ermissions?.*$", re.MULTILINE),
    re.compile(
        r"^\s*\*?\s*(?:Corresponding author|To whom correspondence).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*[\w.+-]+@[\w-]+\.[\w.-]+\s*$", re.MULTILINE),     # email-only line
    # Image furniture beyond the caption rule
    re.compile(
        r"^\s*(?:PHOTOGRAPH|PHOTO|IMAGE|ILLUSTRATION)\s*(?:BY|:)\s*.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:A\s+)?[Ss]creenshot\s+(?:from|of)\s+.*$", re.MULTILINE),
    re.compile(r"^\s*Fig(?:ure)?\.?\s+\d+.*$", re.MULTILINE | re.IGNORECASE),
    # Web furniture
    re.compile(r"^\s*\d+\s*COMMENTS?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:SUBSCRIBE|Sign up for our newsletter\b.*)$", re.MULTILINE),
]
_HYGIENE_INLINE_RULES = [
    re.compile(r"\(cid:\d+\)"),                                        # PDF cid refs
    re.compile(r"ISSN\s*[\d\-Xx]+", re.IGNORECASE),
    re.compile(r"(?:doi|DOI)[:\s]+10\.\d{4,}/\S+"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M", re.IGNORECASE),
    re.compile(r"\b\w+\.(?:jpg|jpeg|png|gif|svg|webp)\b", re.IGNORECASE),
]
_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")
_REPEAT_HEADER_THRESHOLD = 3


def _document_hygiene(text: str) -> str:
    """Strip page furniture, journal boilerplate, and credit lines; heal hyphenation.

    Repeated-header removal first: a running header repeated on every page would
    otherwise be spoken dozens of times, and it is only detectable by counting —
    no single line looks wrong in isolation.
    """
    counts = Counter(line.strip() for line in text.splitlines() if line.strip())
    repeated = {
        line for line, n in counts.items()
        if n >= _REPEAT_HEADER_THRESHOLD and len(line) > 3
    }
    if repeated:
        text = "\n".join(
            line for line in text.splitlines() if line.strip() not in repeated
        )
    for rule in _HYGIENE_LINE_RULES:
        text = rule.sub("", text)
    for rule in _HYGIENE_INLINE_RULES:
        text = rule.sub("", text)
    return _HYPHEN_BREAK.sub(r"\1\2", text)

# End matter: nobody wants a bibliography narrated. Everything from the first of these
# headings onward is dropped — in articles they only appear as trailing sections.
# The '#' is OPTIONAL: a PDF text layer has no markdown, so its bibliography is a
# bare "References" line — and narrating a bibliography is the single worst thing
# this pipeline could do to a listener.
_END_MATTER = re.compile(
    r"^\s*#{0,6}\s*(?:references|external links|see also|further reading|bibliography|"
    r"notes|footnotes|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Wikipedia section-edit markers — bare, or as a full "[[edit](…index.php?…)]" link —
# and the parenthesized husks of anchor-only links. Empty brackets are what remains
# after a citation is stripped out of a link label; they must go too or the voice
# gets a spurious pause-and-click where the marker was.
# Whitespace-tolerant: the block arrives as "[\n[edit](/w/index.php?…)]" — the outer
# bracket sits on its own line.
_EDIT_LINK = re.compile(r"\[?\s*\[edit\]\s*\([^)]*\)\s*\]?", re.IGNORECASE)
_EDIT_MARKER = re.compile(r"\[edit\]", re.IGNORECASE)
_ANCHOR_HUSK = re.compile(r"\(#[^)\s]*\)")
_EMPTY_BRACKETS = re.compile(r"\[\s*\]")
# Last resort, after every structured rule has run: a bracket wrapping one short word
# is the author's own ("[sic]") and stays; any OTHER surviving bracket is markup
# residue in some nesting the rules above did not anticipate (multi-line figure
# blocks, links whose labels contain links, ...). Brackets are never spoken usefully,
# so deleting the character loses nothing the listener could have heard.
_AUTHOR_BRACKET = re.compile(r"\[(\w{1,12})\]")
_STRAY_BRACKET = re.compile(r"[\[\]]")
_OPEN_SENTINEL, _CLOSE_SENTINEL = "\x00", "\x01"


# --- PDF text-layer reflow (scribe#10) -------------------------------------------
# PDFium returns one line per printed line and NO paragraph marks, so a whole page
# arrives as one "paragraph" and every segment boundary landed at a page break, mid
# sentence (33 of 57 boundaries on a 62-page chapter, measured 2026-09-16). The reflow
# rebuilds paragraphs from line shape: a paragraph ends where a line is short for its
# page AND ends a sentence. Everything else is a soft wrap and gets joined.

_TERMINAL = ('.', '!', '?', '"', '”', '’', ')', ':')
# Missing-glyph codepoints: U+FFFE/U+FFFF and the private-use area, which is where
# PDFium lands a glyph the font maps to no Unicode (ligatures, math minus, footnote
# superscripts in LaTeX-era PDFs).
_MISSING_GLYPH = re.compile(r"[\ufffe\uffff\ue000-\uf8ff]+")
_MISSING_IN_WORD = re.compile(r"(?<=[a-z])[\ufffe\uffff\ue000-\uf8ff]+(?=[a-z])")
# Between a word and a Capital it is a footnote superscript glued to the next
# sentence ("Each\ufffe\ufffeSee, I told you"): a space, not a ligature.
_MISSING_BETWEEN = re.compile(r"(?<=[a-z.,;:!?])[\ufffe\uffff\ue000-\uf8ff]+(?=[A-Z])")
_MISSING_WORD_START = re.compile(r"(?<![A-Za-z])[\ufffe\uffff\ue000-\uf8ff]+(?=[a-z]{2})")
# Ligature candidates, longest first; a small lexicon decides. Best effort: a wrong
# guess is still a word-shaped sound, which beats a hole or a "replacement character".
_LIGATURES = ("ffi", "ffl", "ff", "fi", "fl")
_LIGATURE_TEXT = """
affair affairs affect affected affecting affects affirm affix afflict affluent afford
affordable affords afield aflame afloat amplifier amplify artificial baffle beneficial benefit
benefits briefly buffalo buffer buffers butterfly caffeine camouflage certificate certified
certify chaffing clarify classified classify cliff cliffs codify coefficient coefficients
coffee confidence confident configuration configure confine confirm conflict conflicts cuff
defiance deficit define defined defines defining definite definitely definition definitions
deflate deflect diff differ difference differences different differential differentiate differs
difficult difficulties difficulty diffuse diffusion dignified edifice effect effective
effectively effects efficacy efficiency efficient efficiently effort efforts field fields
fierce fifth fifty fight figure figures file filed files filing fill filled film filter final
finally finance financial find finding findings fine finger finish finished finite fire firm
first fish fist fit fitness five fix fixed fixes flag flags flame flat flavor flaw flawless
fleet flesh flew flex flexibility flexible flick flight flip float floated floating flock flood
floor flop floppy flour flourish flow flowed flowing flows fluctuate fluent fluid flush flux
fly flyer gaffe giraffe graffiti gratified griffin handoff huff identified identifier identify
infinite infinity inflate inflation inflect inflexible inflict inflow influence influenced
influences influential jiffy justified justify kickoff layoff magnificent modified modifier
modify muffin muffle notified notify off offer offered offering offers office officer official
officially offline offset offspring overflow pacific payoff profile profit profitable puff
purified qualified qualify raffle ratified rectify refine refined reflect reflected reflecting
reflection reflects reflex reflux riff rifle ruffle satisfied satisfy scaffold scientific scoff
scuffle sheriff shuffle significant significantly signified simplify sniff snowflake specific
specifically specification specified specifies specify staff stiff stifle stuff suffer
suffering suffice sufficient sufficiently suffix tariff terrified testify toffee traffic trifle
unified uniform unify verified verify waffle whiff workflow
"""
_LIGATURE_WORDS = frozenset(_LIGATURE_TEXT.split())


def _repair_missing_glyphs(text: str) -> str:
    """Replace missing-glyph runs: a plausible ligature inside or in front of a word,
    nothing everywhere else (footnote superscripts, math minus signs)."""
    def guess(before: str, after: str, at_start: bool) -> str:
        for lig in _LIGATURES:
            if (before + lig + after).lower() in _LIGATURE_WORDS:
                return lig
        return "fi" if at_start else "ff"

    def in_word(m: re.Match[str]) -> str:
        start = m.start()
        i = start
        while i > 0 and text[i - 1].isalpha():
            i -= 1
        j = m.end()
        while j < len(text) and text[j].isalpha():
            j += 1
        return guess(text[i:start], text[m.end():j], at_start=False)

    def at_start(m: re.Match[str]) -> str:
        j = m.end()
        while j < len(text) and text[j].isalpha():
            j += 1
        return guess("", text[m.end():j], at_start=True)

    text = _MISSING_IN_WORD.sub(in_word, text)
    text = _MISSING_BETWEEN.sub(" ", text)
    text = _MISSING_WORD_START.sub(at_start, text)
    return _MISSING_GLYPH.sub("", text)


def _short_line_run(lines: list[str], i: int, min_run: int = 3, max_len: int = 32) -> int:
    """Length of the run of consecutive short, sentence-less lines starting at i, if it
    is at least ``min_run`` long; else 0. Pseudocode blocks and figure labels ("F5",
    "F3 F4", "return 0") arrive exactly like this and are noise when spoken."""
    n = 0
    while i + n < len(lines):
        ln = lines[i + n].strip()
        if not ln or len(ln) > max_len or ln.endswith(_TERMINAL):
            break
        n += 1
    return n if n >= min_run else 0


def reflow_pdf_text(text: str) -> str:
    """Rebuild paragraphs from a PDF text layer: join soft wraps, keep real breaks,
    heal sentences split across pages, drop pseudocode/figure-label runs.

    A break is a paragraph end when the line ends a sentence AND is short for its
    page (the ragged last line), or when the next line looks like a heading. A blank
    line is a break only if the sentence actually ended; page-join blanks inside a
    sentence are joined. Everything else is a soft wrap.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    body_lens = sorted(len(ln) for ln in lines if len(ln) >= 40)
    typical = body_lens[len(body_lens) // 2] if body_lens else 80
    short = 0.8 * typical

    out: list[str] = []
    para: list[str] = []
    i = 0
    while i < len(lines):
        run = _short_line_run(lines, i)
        if run:
            i += run
            continue
        ln = lines[i].strip()
        if not ln:
            # Blank line: a real break only if the sentence ended.
            if para and para[-1].endswith(_TERMINAL):
                out.append(" ".join(para))
                para = []
            i += 1
            continue
        if _looks_like_pdf_heading(ln) and (not para or para[-1].endswith(_TERMINAL)):
            if para:
                out.append(" ".join(para))
                para = []
            out.append(ln)
            i += 1
            continue
        if para and para[-1].endswith("-") and ln[:1].islower():
            para[-1] = para[-1][:-1] + ln     # hyphenated wrap
        else:
            para.append(ln)
        ends = ln.endswith(_TERMINAL) and len(ln) < short
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if ends or (nxt and _looks_like_pdf_heading(nxt) and ln.endswith(_TERMINAL)):
            out.append(" ".join(para))
            para = []
        i += 1
    if para:
        out.append(" ".join(para))
    return "\n\n".join(p for p in out if p.strip())


def prepare_document(text: str) -> str:
    """Document-level pass: hygiene, then cut everything from the end matter on.

    Separate from the body pass so heading detection can run on text that still HAS
    its headings — the body pass turns them into spoken sentences, which destroys the
    structure chapters are built from.
    """
    text = _repair_missing_glyphs(text)
    text = _document_hygiene(text)
    m = _END_MATTER.search(text)
    return text[: m.start()] if m else text


def clean_body(text: str) -> str:
    """Body pass: strip what is painful to hear; keep every sentence the author wrote.

    Safe to run per-section: every rule here is local to the text it is given.
    """
    text = _FENCE.sub(" Code example omitted. ", text)
    text = _EDIT_LINK.sub("", text)
    text = _EDIT_MARKER.sub("", text)
    text = _ANCHOR_HUSK.sub("", text)
    text = _SUP.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _ESCAPED_BRACKET.sub(r"\1", text)
    text = _ANGLE_URL.sub("", text)
    text = _IMAGE.sub("", text)
    text = _CITATION.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    text = _TABLE_ROW.sub("", text)
    text = _CAPTION.sub("", text)
    # Headings become spoken sentences: a pause-inducing period, not a hash.
    text = _HEADING.sub(lambda m: f"{m.group(1).strip().rstrip('.:')}." , text)
    text = _INLINE_CODE.sub(r"\1", text)
    # Emphasis twice: bold-italic nests (*** outside, * inside after one pass). Then
    # sweep unpaired leftovers — sources contain unbalanced "**" the pair rule cannot
    # match. Single "*" stays (it can be the author's own character).
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = re.sub(r"\*{2,}", "", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _AUTHOR_BRACKET.sub(rf"{_OPEN_SENTINEL}\1{_CLOSE_SENTINEL}", text)
    text = _STRAY_BRACKET.sub("", text)
    text = text.replace(_OPEN_SENTINEL, "[").replace(_CLOSE_SENTINEL, "]")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def clean_for_listening(text: str) -> str:
    """Full cleaning pipeline for a whole document."""
    return clean_body(prepare_document(text))


# Sentence-end candidates that are not: common abbreviations, initials, list numbers,
# decimals broken by a space. Splitting there put a prosody reset inside "et al. 2020"
# and "Fig. 3" on every boundary the paragraph splitter could not use.
_ABBREVIATIONS = re.compile(
    r"\b(?:e\.g|i\.e|et al|etc|vs|cf|viz|approx|resp|Fig|Figs|Eq|Eqs|Sec|Ch|Chap|Vol|No|Nos|"
    r"pp?|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|Inc|Ltd|Co|Corp|U\.S|U\.K|Ph\.D|a\.m|p\.m|"
    r"[A-Z])\.$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"\u201d\u2019)]))\s+(?=[A-Z0-9\"\u201c(])"
)


def _split_sentences(para: str) -> list[str]:
    parts = _SENTENCE_END.split(para)
    out: list[str] = []
    for part in parts:
        # A bare list marker ("3.") is not a sentence; "in 2020." is.
        if out and (_ABBREVIATIONS.search(out[-1]) or re.fullmatch(r"\d{1,3}\.", out[-1].strip())):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def split_segments(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into segments of at most ``max_chars``.

    A paragraph longer than the cap is split at sentence boundaries as a fallback;
    only a pathological single sentence ever gets hard-cut.
    """
    segments: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            segments.append("\n\n".join(current))
            current, size = [], 0

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            sentences = _split_sentences(para)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) + 1 > max_chars:
                    segments.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
                # A single sentence over the cap: hard-cut rather than send an
                # unbounded request.
                while len(buf) > max_chars:
                    segments.append(buf[:max_chars])
                    buf = buf[max_chars:]
            if buf:
                segments.append(buf)
            continue
        if size and size + len(para) + 2 > max_chars:
            flush()
        current.append(para)
        size += len(para) + 2
    flush()
    return segments


@dataclass
class Chapter:
    title: str
    segments: list[str]


# What SHOULD never survive cleaning. The script is deterministic, so artifacts are
# knowable before a single second is synthesized — lint findings mean a cleaning rule
# is missing, and hearing about it here costs nothing while hearing it in the audio
# costs a re-listen (this is how the "sup"/footnote noise was caught: by ear, late).
_LINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("html tag", re.compile(r"</?[a-z][a-z0-9]*\b[^>]*>", re.IGNORECASE)),
    ("bracket residue", re.compile(r"[\[\]]")),
    ("url", re.compile(r"https?://|www\.", re.IGNORECASE)),
    ("backslash escape", re.compile(r"\\[\[\]()*_#]")),
    ("anchor husk", re.compile(r"\(#[^)\s]*\)")),
    ("markdown emphasis", re.compile(r"(\*{1,3}|_{2,3})\S")),
    ("markdown heading", re.compile(r"^#{1,6}\s", re.MULTILINE)),
    ("code fence", re.compile(r"```")),
    ("citation marker", re.compile(r"\bcite[_-]?(?:note|ref)", re.IGNORECASE)),
]


def lint_script(chapters: list[Chapter]) -> list[str]:
    """Report likely-vocalized artifacts left in a listening script.

    Returns human-readable findings ("bracket residue x12, e.g. ...context..."), empty
    when the script is clean. Callers decide severity: the CLI prints them, the
    pipeline logs them and synthesizes anyway — a slightly noisy audiobook still
    beats no audiobook, but the finding tells us which cleaning rule to add next.
    """
    findings: list[str] = []
    text = "\n\n".join(seg for ch in chapters for seg in ch.segments)
    # The cleaner deliberately keeps short author brackets ("[sic]", "[if]", IPA like
    # "[aː]") — the lint must not cry wolf about what is kept on purpose, or real
    # findings drown and the report gets ignored.
    text = _AUTHOR_BRACKET.sub(r"\1", text)
    for name, pattern in _LINT_PATTERNS:
        hits = list(pattern.finditer(text))
        if not hits:
            continue
        i = hits[0].start()
        context = " ".join(text[max(0, i - 40): i + 40].split())
        findings.append(f"{name} x{len(hits)}, e.g. ...{context}...")
    return findings


# --- Structure detection (scribe#3) ---------------------------------------------
# Chapters follow the source's own sections, so a textbook chapter arrives as
# tappable sections in the player instead of one 40-minute block.

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# "3.2 Decidability", "Chapter 4", "IV. Results" — numbering is the strongest signal
# a PDF text layer offers, since it carries no heading markup at all.
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:(?:chapter|section|part)\s+)?"
    r"(?:\d+(?:\.\d+){0,2}\.?|[IVXLC]{1,6}\.)"
    r"\s+(\S.{0,78})$",
    re.IGNORECASE,
)

# A chapter shorter than this is not worth a player entry — it merges into its
# predecessor. ~700 chars is roughly 45 seconds of speech.
_MIN_CHAPTER_CHARS = 700
# Beyond this the chapter list stops being navigation and becomes a wall.
_MAX_CHAPTERS = 20


_DIACRITIC_RESIDUE = re.compile(
    r"\s?[\u00af\u02d9\u02d8\u00b4\u0060\u005e\u00a8\u02dc]\s?|\.(?=[a-z])"
)


def _looks_like_pdf_heading(line: str) -> bool:
    """Heuristic heading test for text layers that carry no markup.

    Deliberately strict: a false positive splits a paragraph mid-thought and puts a
    chapter marker inside a sentence, which is worse than a missed heading (whose
    only cost is a longer chapter).
    """
    # Detached diacritics from LaTeX text layers ("Matr ¯ avr ¯ .tta") are not
    # letters and would sink the ratio; judge the line without them.
    s = _DIACRITIC_RESIDUE.sub("", line).strip()
    if not (3 <= len(s) <= 80) or s.endswith((".", ",", ";", ":", "?", "!")):
        return False
    # Mostly-letters test, before anything else: it is what separates a heading from
    # a table row or a formula. ") O(1) O(logk(n))" scores 0.53 and is rejected;
    # "3.2 Decidability" scores 0.81 and survives.
    m = _NUMBERED_HEADING.match(s)
    # Judge the TITLE part: "3.1. Matravrtta" is a heading even though its numbering
    # drags the whole line's letter ratio under the bar.
    core = m.group(1) if m else s
    if sum(c.isalpha() or c.isspace() for c in core) / len(core) < 0.75:
        return False
    words = s.split()
    # Author lines carry affiliation marks; table headers repeat their tokens; a real
    # heading has at least one word of three letters. None of those is navigation.
    if any(c in s for c in "\u2217\u2020\u2021*") or not any(
        sum(ch.isalpha() for ch in w) >= 3 for w in words
    ):
        return False
    if len(words) >= 4 and len(set(words)) <= len(words) // 2:
        return False
    # Figure text rendered as one line: interpunct/bullet separators, or a "word"
    # longer than any English heading word (glyphs of two labels interleaved).
    if any(c in s for c in "\u00b7\u2022") or any(len(w) > 16 for w in words):
        return False
    if m:
        # "3.1 Matravrtta" yes; "0 if j > n" (a formula line) no: the title must
        # start with a letter and read as a title, not a clause.
        title = m.group(1)
        return title[:1].isalpha() and title[:1].isupper()
    if not (1 <= len(words) <= 10):
        return False
    # ALL CAPS needs two real words: "ALTR U" is a garbled small-caps identifier.
    if s.isupper():
        return sum(1 for w in words if sum(c.isalpha() for c in w) >= 3) >= 2
    return all(w[0].isupper() or not w[0].isalpha() for w in words) and any(
        w[0].isupper() for w in words
    )


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Split prepared text into (heading, body) sections, or [] if it has no structure.

    Markdown headings win when present (the web path keeps them); the PDF heuristic is
    the fallback. Returns [] rather than guessing when nothing is confident enough —
    the caller then produces today's single "Full article" chapter.
    """
    # A heading needs letters: PDF figure debris like "# #&#&" satisfies the markdown
    # shape and used to become a chapter called "#&#&#&#".
    headings = [m for m in _MD_HEADING.finditer(text)
                if sum(c.isalpha() for c in m.group(2)) >= 2]
    if headings:
        # Coarsest level that actually divides the document. The `# Title` line is
        # usually alone at level 1, so this naturally lands on `##`.
        by_level: dict[int, list[re.Match[str]]] = {}
        for m in headings:
            by_level.setdefault(len(m.group(1)), []).append(m)
        for level in sorted(by_level):
            if len(by_level[level]) >= 2:
                marks = by_level[level]
                sections = []
                # Text before the FIRST heading is the article's lead — dropping it
                # would silently lose the opening paragraphs (and on Wikipedia, the
                # definition itself). Any higher-level heading in there is the
                # document title, which the summary chapter already announced.
                lead = _MD_HEADING.sub("", text[: marks[0].start()]).strip()
                if lead:
                    sections.append(("Introduction", lead))
                for i, m in enumerate(marks):
                    end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
                    sections.append((m.group(2).strip(), text[m.end(): end].strip()))
                return sections
        return []

    lines = text.splitlines()

    def _starts_a_section(i: int) -> bool:
        if not _looks_like_pdf_heading(lines[i]):
            return False
        prev = lines[i - 1].strip() if i else ""
        at_break = not prev or prev.endswith((".", "!", "?", '."', '.”'))
        if not at_break:
            return False
        # Numbering is a strong enough signal to stand on a paragraph break alone.
        # Bare title case is not — a capitalized sentence fragment wrapped onto its
        # own line looks identical — so it still requires a real blank line.
        return bool(_NUMBERED_HEADING.match(lines[i].strip())) or not prev

    marks = [i for i in range(len(lines)) if _starts_a_section(i)]
    if len(marks) < 2:
        return []
    sections = []
    lead = "\n".join(lines[: marks[0]]).strip()
    if lead:
        sections.append(("Introduction", lead))
    for n, i in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        sections.append((lines[i].strip(), "\n".join(lines[i + 1: end]).strip()))
    return sections


def _chapter_title(heading: str) -> str:
    """Player-friendly chapter label: cleaned of markup, truncated at a word."""
    # Leading symbol residue ("\u21223.2 Aside") is a glyph the font could not map.
    heading = _DIACRITIC_RESIDUE.sub("", heading)
    label = clean_body(heading).rstrip(".").strip().lstrip("\u2122\u00a9\u00ae\u2020\u2021*# ")
    label = label or heading.strip()
    if len(label) <= 60:
        return label
    return label[:60].rsplit(" ", 1)[0] + "..."


def _consolidate(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge runs too short to be worth a chapter, then cap the total.

    A merged chapter keeps the FIRST heading as its title and the later headings stay
    in the spoken text, so nothing is lost to the listener — only the player's
    navigation list is coarsened.
    """
    merged: list[tuple[str, str]] = []
    for title, body in sections:
        if merged and len(body) < _MIN_CHAPTER_CHARS:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body}\n\n{title}.\n\n{body}".strip())
        else:
            merged.append((title, body))

    while len(merged) > _MAX_CHAPTERS:
        # Fold the shortest chapter into its neighbour until the list fits.
        i = min(range(1, len(merged)), key=lambda n: len(merged[n][1]))
        title, body = merged.pop(i)
        prev_title, prev_body = merged[i - 1]
        merged[i - 1] = (prev_title, f"{prev_body}\n\n{title}.\n\n{body}".strip())
    return merged


def build_script(doc: Document, summary: Summary, *, max_chars: int) -> list[Chapter]:
    """Summary chapter first, then the article — the agreed listening order."""
    title = note_title(doc, summary).strip() or "Untitled"
    summary_text = clean_for_listening(
        f"{title}.\n\nSummary.\n\n{summary.tldr}\n\n{summary.summary}"
    )
    chapters = [Chapter("Summary", split_segments(summary_text, max_chars))]

    prepared = prepare_document(doc.text)
    if doc.kind == "pdf":
        # Text layers carry line breaks, not paragraphs (see reflow_pdf_text).
        prepared = reflow_pdf_text(prepared)
    sections = _consolidate(detect_sections(prepared))
    lead = "End of summary. The full article begins now."

    if len(sections) >= 2:
        for n, (heading, body) in enumerate(sections):
            # The heading is spoken at the top of its own chapter — a listener who
            # jumps to a chapter should hear what it is.
            spoken = clean_body(f"{_DIACRITIC_RESIDUE.sub('', heading)}.\n\n{body}")
            if not spoken:
                continue
            if n == 0:
                spoken = f"{lead}\n\n{spoken}"
            chapters.append(
                Chapter(_chapter_title(heading), split_segments(spoken, max_chars))
            )
        # Every section cleaned away to nothing: fall through to the flat chapter
        # rather than shipping a summary-only audiobook.
        if len(chapters) > 1:
            return chapters

    article_text = clean_body(prepared)
    if article_text:
        chapters.append(
            Chapter("Full article", split_segments(f"{lead}\n\n{article_text}", max_chars))
        )
    return chapters
