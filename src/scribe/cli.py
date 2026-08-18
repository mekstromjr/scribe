"""CLI for Phase 1: extraction only. No Slack, no vault writes."""

from __future__ import annotations

import argparse
import sys

from scribe.config import load_settings
from scribe.extract import ExtractionError, extract
from scribe.extract.pdf import has_text_layer
from scribe.ollama import OllamaError, health


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


def _cmd_probe(args: argparse.Namespace) -> int:
    """Classify a PDF without spending any CPU on OCR."""
    from pathlib import Path

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

    p_probe = sub.add_parser("probe", help="check whether a PDF has a text layer (no OCR)")
    p_probe.add_argument("target")
    p_probe.set_defaults(func=_cmd_probe)

    p_health = sub.add_parser("health", help="show which Ollama host and models are in use")
    p_health.set_defaults(func=_cmd_health)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
