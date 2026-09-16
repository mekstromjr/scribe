"""scribe CLI: extract, summarize, export a note file, and run the Slack bot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scribe.config import load_settings
from scribe.extract import ExtractionError, extract
from scribe.extract.pdf import has_text_layer
from scribe.note import note_title, render, slugify
from scribe.note_export import FORMATS, ExportError, export_note
from scribe.ollama import OllamaError, health
from scribe.summarize import summarize


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
    """Extract -> summarize -> render the note. Prints the markdown; writes nothing."""
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
    filename = f"{slugify(note_title(doc, summary))}.md"
    if args.out_dir:
        target = Path(args.out_dir).expanduser() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        print(f"wrote {target}")
    else:
        print(f"# filename: {filename}", file=sys.stderr)
        print(body)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Extract -> summarize -> render -> write the note as pdf/md/docx.

    The laptop twin of what the Slack bot posts in-thread; same renderer, same exporter,
    so a format problem reproduces here without a Slack round trip.
    """
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

    body = render(doc, summary, model=settings.text_model)
    try:
        path = export_note(
            body, args.format, stem=slugify(note_title(doc, summary)),
            out_dir=Path(args.out_dir).expanduser(),
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(f"\nTL;DR — {summary.tldr}")
    return 0


def _cmd_listen(args: argparse.Namespace) -> int:
    """Extract -> summarize -> synthesize -> package an .m4b; optionally upload to ABS.

    The local-file default exists so the whole audio path is verifiable from a laptop
    (with kokoro port-forwarded) before any Slack traffic touches it.
    """
    import shutil

    from scribe.abs import ABSError, upload
    from scribe.audio import produce_audio
    from scribe.note import note_title as _title
    from scribe.tts import TTSError
    from scribe.tts import health as tts_health

    settings = load_settings()

    if args.dry_run:
        # The whole point of a dry run is knowing about vocalized artifacts BEFORE
        # spending synthesis minutes — so it needs neither kokoro nor the summarizer.
        from scribe.listening import build_script, lint_script
        from scribe.summarize import Summary

        try:
            doc = extract(settings, args.target)
        except (ExtractionError, OllamaError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        chapters = build_script(
            doc,
            Summary(title=doc.title or args.target, tldr="(summary omitted in dry run)",
                    summary=""),
            max_chars=settings.tts_max_chars,
        )
        findings = lint_script(chapters)
        # Chapter plan first and on stderr: the structure is the thing worth checking
        # at a glance, and it stays visible when the script itself is piped away.
        print(f"chapter plan ({len(chapters)}):", file=sys.stderr)
        for ch in chapters:
            chars = sum(len(s) for s in ch.segments)
            print(f"  - {ch.title}  [{len(ch.segments)} segment(s), {chars} chars]",
                  file=sys.stderr)
        for ch in chapters:
            print(f"===== chapter: {ch.title} ({len(ch.segments)} segment(s)) =====")
            for seg in ch.segments:
                print(seg)
                print("----- segment break -----")
        if findings:
            print("\nLINT: artifacts that WILL be vocalized:", file=sys.stderr)
            for f in findings:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nLINT: clean", file=sys.stderr)
        return 0

    try:
        tts_health(settings)
        doc = extract(settings, args.target)
        print(f"# extracted: {doc.summary_line()}", file=sys.stderr)
        summary = summarize(settings, doc)
        print(f"# summarized in {summary.seconds:.1f}s", file=sys.stderr)
        title = _title(doc, summary)
        author = doc.source if doc.kind == "link" else "scribe"
        result = produce_audio(settings, doc, summary, title=title, author=author)
    except (ExtractionError, OllamaError, TTSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with result.workdir:
        print(
            f"# {result.segments} segment(s), {result.audio_seconds / 60:.1f} min of audio, "
            f"synthesized in {result.synth_seconds / 60:.1f} min "
            f"({result.audio_seconds / max(result.synth_seconds, 0.001):.1f}x realtime)",
            file=sys.stderr,
        )
        if args.upload:
            try:
                link = upload(settings, result.m4b, title=title, author=author)
            except ABSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"uploaded: {link}")
        else:
            out = Path(args.out or f"{slugify(title)}.m4b").expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.m4b, out)
            print(f"wrote {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Run the Slack Socket Mode bot. Blocks."""
    from scribe.slack_app import run

    return run()


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

    p_note = sub.add_parser("note", help="extract, summarize, and render the note markdown")
    p_note.add_argument("target")
    p_note.add_argument("--out-dir", help="write the note here instead of stdout")
    p_note.set_defaults(func=_cmd_note)

    p_export = sub.add_parser("export", help="extract, summarize, and write the note as a file")
    p_export.add_argument("target")
    p_export.add_argument(
        "--format", choices=[f for f in FORMATS if f != "none"], default="pdf",
    )
    p_export.add_argument("--out-dir", default=".", help="directory to write into")
    p_export.set_defaults(func=_cmd_export)

    p_listen = sub.add_parser(
        "listen", help="extract, summarize, synthesize, and package an .m4b"
    )
    p_listen.add_argument("target")
    p_listen.add_argument("--out", help="write the .m4b here (default: ./<title>.m4b)")
    p_listen.add_argument(
        "--upload", action="store_true",
        help="upload to the Audiobookshelf Articles library instead of writing locally",
    )
    p_listen.add_argument(
        "--dry-run", action="store_true",
        help="print the listening script and artifact lint without synthesizing "
             "(no kokoro or summarizer needed); exits 1 if artifacts are found",
    )
    p_listen.set_defaults(func=_cmd_listen)

    p_serve = sub.add_parser("serve", help="run the Slack bot (Socket Mode)")
    p_serve.set_defaults(func=_cmd_serve)

    p_probe = sub.add_parser("probe", help="check whether a PDF has a text layer (no OCR)")
    p_probe.add_argument("target")
    p_probe.set_defaults(func=_cmd_probe)

    p_health = sub.add_parser("health", help="show which Ollama host and models are in use")
    p_health.set_defaults(func=_cmd_health)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
