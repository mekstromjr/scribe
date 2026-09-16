"""Turn the rendered note into the file that gets posted to Slack.

The note used to be committed to the owner's Obsidian vault (scribe#5 removed that).
Now it is a file in the thread, in whichever format the reader asked for. `md` is the
`note.render()` output verbatim, frontmatter and callouts included, so a note worth
keeping drops into a vault as-is. `pdf` and `docx` are renderings of that same markdown
for people who will never open a markdown file: pandoc for both, with weasyprint turning
pandoc's HTML into the PDF. No TeX in the image.

Export failures are the caller's problem to degrade, not to retry: the summary already
exists, and a pandoc crash will not fix itself on a second attempt.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Literal

log = logging.getLogger("scribe.note_export")

NoteFormat = Literal["pdf", "md", "docx", "none"]
FORMATS: tuple[str, ...] = ("pdf", "md", "docx", "none")

# Obsidian callout header, e.g. "> [!note]- Full extracted text". Outside Obsidian the
# marker is line noise, so it becomes a bold lead line inside the same blockquote.
_CALLOUT = re.compile(r"^> \[!\w+\][+-]? ?(.*)$", re.MULTILINE)
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Print-oriented defaults for the PDF: readable body measure, quiet blockquotes so the
# collapsed source text does not shout, and page numbers because the source text of a
# long PDF can run to dozens of pages.
_PDF_CSS = """
@page { size: Letter; margin: 2cm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #777; } }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 20pt; margin-bottom: 0.2em; }
h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; }
h3, h4 { font-size: 12pt; }
blockquote { margin: 0.8em 0; padding: 0.2em 1em; border-left: 3px solid #bbb; color: #333; }
code, pre { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
pre { white-space: pre-wrap; }
hr { border: 0; border-top: 1px solid #ccc; margin: 1.5em 0; }
a { color: #1a4d8f; text-decoration: none; }
"""


class ExportError(RuntimeError):
    """The note could not be turned into the requested file."""


def portable_markdown(markdown: str) -> str:
    """Markdown for renderers that are not Obsidian: no frontmatter, no callout markers."""
    body = _FRONTMATTER.sub("", markdown, count=1)
    # The bold lead gets its own paragraph inside the quote (the "\n>" line); without
    # it the first line of source text runs on from the heading.
    return _CALLOUT.sub(
        lambda m: f"> **{m.group(1).strip()}**\n>" if m.group(1).strip() else ">", body
    )


def _pandoc(args: list[str], stdin: str) -> str:
    try:
        proc = subprocess.run(
            ["pandoc", *args], input=stdin, text=True, capture_output=True, timeout=300,
        )
    except FileNotFoundError as exc:
        raise ExportError("pandoc is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise ExportError("pandoc timed out") from exc
    if proc.returncode != 0:
        raise ExportError(f"pandoc failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def export_note(markdown: str, fmt: str, *, stem: str, out_dir: Path) -> Path:
    """Write the note as `fmt` into `out_dir` and return the path.

    `fmt` must be one of FORMATS other than "none"; callers decide before this point
    whether there is anything to export at all.
    """
    if fmt not in FORMATS or fmt == "none":
        raise ExportError(f"unsupported note format: {fmt!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}.{fmt}"

    if fmt == "md":
        target.write_text(markdown)
        return target

    portable = portable_markdown(markdown)
    if fmt == "docx":
        _pandoc(["-f", "markdown", "-t", "docx", "-o", str(target)], portable)
        return target

    # pdf: pandoc renders HTML, weasyprint lays it out. Imported lazily because the
    # laptop CLI and the test suite should not need weasyprint's cairo/pango stack to
    # import this module.
    # pagetitle, not title: it fills <title> (pandoc warns without one) but renders no
    # header block, so the note's own H1 stays the only title on the page.
    html = _pandoc(["-f", "markdown", "-t", "html5", "--standalone",
                    "--metadata", f"pagetitle={stem}"], portable)
    try:
        from weasyprint import CSS, HTML  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExportError("weasyprint is not installed") from exc
    except OSError as exc:
        # weasyprint loads pango/cairo through cffi AT IMPORT; a missing native lib
        # surfaces here as OSError, not ImportError.
        raise ExportError(f"weasyprint cannot load its native libraries: {exc}") from exc
    try:
        HTML(string=html).write_pdf(str(target), stylesheets=[CSS(string=_PDF_CSS)])
    except Exception as exc:  # weasyprint raises a zoo of its own types
        raise ExportError(f"PDF layout failed: {exc}") from exc
    return target
