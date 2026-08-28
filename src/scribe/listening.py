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
_END_MATTER = re.compile(
    r"^#{1,6}\s*(?:references|external links|see also|further reading|bibliography|"
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


def clean_for_listening(text: str) -> str:
    """Strip what is painful to hear; keep every sentence the author wrote."""
    text = _document_hygiene(text)
    m = _END_MATTER.search(text)
    if m:
        text = text[: m.start()]
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
            sentences = re.split(r"(?<=[.!?])\s+", para)
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


def build_script(doc: Document, summary: Summary, *, max_chars: int) -> list[Chapter]:
    """Summary chapter first, then the article — the agreed listening order."""
    title = (doc.title or summary.title or "Untitled").strip()
    summary_text = clean_for_listening(
        f"{title}.\n\nSummary.\n\n{summary.tldr}\n\n{summary.summary}"
    )
    article_text = clean_for_listening(doc.text)
    chapters = [Chapter("Summary", split_segments(summary_text, max_chars))]
    if article_text:
        chapters.append(
            Chapter(
                "Full article",
                split_segments(
                    f"End of summary. The full article begins now.\n\n{article_text}",
                    max_chars,
                ),
            )
        )
    return chapters
