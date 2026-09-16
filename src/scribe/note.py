"""Render a vault note.

The shape follows the conventions already in use in the vault rather than inventing a new
one: inline `tags:` array carrying `aigen`, a `created:` date, an `H1` title, a `> Source`
blockquote, and a `## TL;DR` — matching `Atlas/Tours (Ai)/`. The `Sources:` frontmatter key
comes from `Misc/Templates/Reading.md`.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import quote

from scribe.document import Document, Method
from scribe.summarize import Summary

# Characters Obsidian and/or the filesystem will not tolerate in a note name.
_UNSAFE = re.compile(r'[\\/:*?"<>|#\^\[\]]+')
_HEADING = re.compile(r"^(#{1,6})(\s)", re.MULTILINE)
_WS = re.compile(r"\s+")


def slugify(title: str, *, max_len: int = 80) -> str:
    """Turn a model-written title into a safe vault filename (spaces preserved — Obsidian
    note names are human-facing, so this is not a URL slug)."""
    cleaned = _WS.sub(" ", _UNSAFE.sub("", title)).strip(" .")
    return (cleaned[:max_len].strip() or "Untitled note")


def demote_headings(markdown: str, floor: int = 3) -> str:
    """Shift the model's headings so they nest UNDER the note's own structure.

    The summary arrives with its own `#`/`##` hierarchy, which collides with the note's
    `# Title` and `## Summary`. Left alone, Obsidian's outline shows the summary's sections
    as siblings of `## Summary` rather than children, so the note reads flat.

    The whole block is shifted by a single offset rather than clamped per-heading, so the
    model's own relative nesting is preserved.
    """
    levels = [len(m.group(1)) for m in _HEADING.finditer(markdown)]
    if not levels:
        return markdown
    shift = floor - min(levels)
    if shift <= 0:
        return markdown

    def bump(match: re.Match[str]) -> str:
        # Markdown has no h7; deeper headings clamp rather than overflow into literal '#'.
        return "#" * min(6, len(match.group(1)) + shift) + match.group(2)

    return _HEADING.sub(bump, markdown)


def _callout(body: str) -> str:
    """Wrap raw text in a collapsed Obsidian callout.

    Every line needs the '> ' prefix, blank lines included — a bare empty line terminates
    the callout and dumps the remainder into the note body. The trailing '-' on '[!note]-'
    is what makes it collapsed by default, which matters because the raw text of a 40-page
    PDF would otherwise dominate the note.
    """
    lines = ["> [!note]- Full extracted text"]
    for line in body.splitlines():
        lines.append(f"> {line}" if line.strip() else ">")
    return "\n".join(lines)


def _provenance(doc: Document, summary: Summary, model: str) -> str:
    methods = sorted({p.method.value for p in doc.pages if p.method is not Method.SKIPPED})
    parts = [
        f"Ingested: {date.today().isoformat()} via scribe",
        f"Pages: {len(doc.pages)}",
        f"Extraction: {'+'.join(methods) or 'none'}",
        f"Model: {model}",
    ]
    if doc.ocr_pages:
        parts.append(f"OCR: {doc.ocr_pages} page(s), {doc.seconds:.0f}s")
    skipped = [p for p in doc.pages if p.method is Method.SKIPPED]
    if skipped:
        # Surfaced in the note itself: a summary built from partial text should say so,
        # and say why -- a runaway OCR page and a page-count cap are different problems.
        reasons = sorted({p.reason for p in skipped if p.reason})
        detail = f" ({'; '.join(reasons)})" if reasons else ""
        parts.append(f"**{len(skipped)} page(s) SKIPPED{detail}**")
    if summary.sections > 1:
        # Say so plainly: a map-reduced summary is genuinely lower resolution than a
        # single pass, because no one pass saw the whole argument.
        parts.append(f"Summarized in {summary.sections} sections (map-reduce)")
    if summary.truncated_chars:
        parts.append(f"**{summary.truncated_chars} chars truncated to fit context**")
    return " · ".join(parts)


# Titles that name the file's export path rather than its content. A slug has no spaces
# and leans on digits or separators; office suites stamp their own prefix; some
# generators leave a placeholder. Any of these loses to the model's title (scribe#9).
_JUNK_PREFIXES = ("microsoft word - ", "microsoft powerpoint - ", "untitled", "document",
                  "scan", "img_", "image", "download", "final", "draft", "copy of ")
_JUNK_SUFFIXES = (".pdf", ".doc", ".docx", ".tex", ".pptx", ".txt")


def is_junk_title(title: str | None) -> bool:
    """True when a candidate title is a filename in disguise, not a title.

    '03-dynprog', 'monetary20250618a1', 'Microsoft Word - final.docx' are all junk;
    'Federal Reserve issues FOMC statement' and 'Liber Abaci' are not. The test is
    deliberately lenient toward real titles: a false 'junk' only costs falling through
    to the model's title, which is a decent title too.
    """
    s = (title or "").strip()
    if len(s) < 3:
        return True
    low = s.lower()
    if low.startswith(_JUNK_PREFIXES) or low.endswith(_JUNK_SUFFIXES):
        return True
    if " " not in s:
        # One token: a real one-word title exists ("Dune"), but a token with digits or
        # separators in it is a slug.
        return any(c.isdigit() or c in "-_." for c in s)
    letters = sum(c.isalpha() or c.isspace() for c in s) / len(s)
    return letters < 0.6


def note_title(doc: Document, summary: Summary) -> str:
    """Best-effort title ladder (scribe#9): the source's own title when it has a real
    one, then the model's, then the filename stem as a last resort.

    For a link, `doc.title` is the page's <title>, which is what you would search for
    later. For a file, the extractor sets `doc.title` from PDF metadata and leaves it
    None when that is empty or junk (LaTeX PDFs ship no title; Word stamps
    "Microsoft Word - x.docx"), so the model's title takes over. The filename stem only
    wins when the model produced nothing, because "03-dynprog" is not a title.
    """
    own = (doc.title or "").strip()
    if own and not is_junk_title(own):
        return own
    model = (summary.title or "").strip()
    if model and not is_junk_title(model):
        return model
    stem = Path(doc.source).stem if doc.kind != "link" else doc.source
    if not is_junk_title(stem):
        return stem
    # Everything is junk: the stem is the most searchable junk, but an empty one loses.
    return stem or own or model


def obsidian_uri(vault_name: str, note_path: str) -> str:
    """Deep link that opens the note in Obsidian on macOS or iOS.

    The `.md` suffix is dropped: Obsidian resolves by note name, and leaving it on makes
    the link miss.
    """
    path = note_path[:-3] if note_path.endswith(".md") else note_path
    return f"obsidian://open?vault={quote(vault_name, safe='')}&file={quote(path, safe='')}"


def render(
    doc: Document,
    summary: Summary,
    *,
    model: str,
    created: date | None = None,
    attachment_link: str | None = None,
) -> str:
    created = created or date.today()
    tags = ["aigen", "scribe", *summary.tags]
    if doc.kind == "link":
        source_line = f"[{doc.title or doc.source}]({doc.source})"
    elif attachment_link:
        # Wikilink to the copy committed alongside this note, so the original is one
        # click away rather than only named.
        source_line = f"[[{attachment_link}]]"
    else:
        source_line = f"`{doc.source}`"

    return "\n".join(
        [
            "---",
            f"tags: [{', '.join(tags)}]",
            f"created: {created.isoformat()}",
            f"Sources: {doc.source}",
            f"type: {doc.kind}",
            "---",
            f"# {note_title(doc, summary)}",
            "",
            f"> Source: {source_line}",
            f"> {_provenance(doc, summary, model)}",
            "",
            "## TL;DR",
            "",
            summary.tldr,
            "",
            "---",
            "",
            "## Summary",
            "",
            demote_headings(summary.summary),
            "",
            "---",
            "",
            "## Source Text",
            "",
            _callout(doc.text),
            "",
        ]
    )
