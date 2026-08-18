"""scribe CLI. Extraction and note rendering; Slack and vault publishing land later."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scribe.config import load_settings
from scribe.extract import ExtractionError, extract
from scribe.extract.pdf import has_text_layer
from scribe.note import render, slugify
from scribe.ollama import OllamaError, health
from scribe.summarize import summarize
from scribe.vault import VaultError, publish, resolve_attachment


def _cmd_extract(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        doc = extract(settings, args.target)
    except (ExtractionError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"# {doc.title}", file=sys.stderr)
    print(f"# {doc.summary_line()}", file=sys.stderr)
    if args.per_page:
        for page in doc.pages:
            line = f"# page {page.number}: {page.method.value} ({page.seconds:.1f}s)"
            print(line, file=sys.stderr)
    print(doc.text)
    return 0


def _cmd_note(args: argparse.Namespace) -> int:
    """Extract -> summarize -> render the vault note. Prints the note; does not publish."""
    settings = load_settings()
    try:
        doc = extract(settings, args.target)
        print(f"# extracted: {doc.summary_line()}", file=sys.stderr)
        summary = summarize(settings, doc)
        print(f"# summarized in {summary.seconds:.1f}s", file=sys.stderr)
    except (ExtractionError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if summary.truncated_chars:
        print(
            f"WARNING: {summary.truncated_chars} chars dropped to fit the context window — "
            "the summary covers only the retained portion",
            file=sys.stderr,
        )

    body = render(doc, summary, model=settings.text_model)
    filename = f"{slugify(summary.title)}.md"
    if args.out_dir:
        target = Path(args.out_dir).expanduser() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        print(f"wrote {target}")
    else:
        print(f"# filename: {filename}", file=sys.stderr)
        print(body)
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    """Extract -> summarize -> render -> commit to the Obsidian vault."""
    settings = load_settings()
    try:
        doc = extract(settings, args.target)
        print(f"# extracted: {doc.summary_line()}", file=sys.stderr)
        summary = summarize(settings, doc)
        print(f"# summarized in {summary.seconds:.1f}s", file=sys.stderr)
    except (ExtractionError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if summary.truncated_chars:
        print(
            f"WARNING: {summary.truncated_chars} chars dropped to fit the context window",
            file=sys.stderr,
        )

    # Local files travel with the note so the original is one click away in the vault.
    # Links do not — the URL is already in the frontmatter.
    source_file = None
    if doc.kind in {"pdf", "image"}:
        candidate = Path(args.target).expanduser()
        if candidate.is_file():
            source_file = candidate

    try:
        # Resolved BEFORE rendering: the note wikilinks the attachment by its final
        # name, which is only known after collision resolution.
        attachment_path = resolve_attachment(settings, source_file) if source_file else None
        if source_file and attachment_path is None:
            print(
                f"note: {source_file.name} exceeds the attachment size cap — "
                "committing the note without it",
                file=sys.stderr,
            )
        body = render(
            doc, summary, model=settings.text_model, attachment_link=attachment_path
        )
        result = publish(
            settings,
            note_body=body,
            note_stem=slugify(summary.title),
            attachment=source_file,
            attachment_path=attachment_path,
        )
    except VaultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"published: {result['note']}")
    if result["attachment"]:
        print(f"attachment: {result['attachment']}")
    print(f"\nTL;DR — {summary.tldr}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Classify a PDF without spending any CPU on OCR."""
    path = Path(args.target).expanduser()
    if has_text_layer(path):
        print(f"{path.name}: has a text layer — extraction will be instant, no OCR")
    else:
        print(f"{path.name}: NO text layer — every page falls through to OCR (~20s/page)")
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    """Confirm which Ollama we are actually talking to.

    Worth its own command: the dev Mac runs a native ollama on 127.0.0.1:11434, so a
    misconfigured host answers successfully with completely different models.
    """
    settings = load_settings()
    try:
        models = health(settings)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"host: {settings.ollama_host}")
    for name in models:
        print(f"  {name}")
    missing = [m for m in (settings.ocr_model, settings.text_model) if m not in models]
    if missing:
        print(f"WARNING: expected model(s) not served here: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="scribe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extract text from a URL, PDF, or image")
    p_extract.add_argument("target")
    p_extract.add_argument(
        "--per-page", action="store_true", help="report the method and timing for each page"
    )
    p_extract.set_defaults(func=_cmd_extract)

    p_note = sub.add_parser("note", help="extract, summarize, and render a vault note")
    p_note.add_argument("target")
    p_note.add_argument("--out-dir", help="write the note here instead of stdout")
    p_note.set_defaults(func=_cmd_note)

    p_publish = sub.add_parser("publish", help="extract, summarize, and commit to the vault")
    p_publish.add_argument("target")
    p_publish.set_defaults(func=_cmd_publish)

    p_probe = sub.add_parser("probe", help="check whether a PDF has a text layer (no OCR)")
    p_probe.add_argument("target")
    p_probe.set_defaults(func=_cmd_probe)

    p_health = sub.add_parser("health", help="show which Ollama host and models are in use")
    p_health.set_defaults(func=_cmd_health)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
