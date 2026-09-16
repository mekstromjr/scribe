"""scribe#10 experiment: is an LLM polish pass worth adding to the listening script?

Three variants of the same ~8k-char excerpt from each real source:
  rules  -- today's path: prepare_document + clean_body (deterministic)
  llm    -- the legacy tts-pipeline repair prompt over the prepared text, no rules
  both   -- rules first, then the LLM prompt on the result

Per variant: chars, lint findings, wall seconds, similarity to its input (difflib), and a
saved script. Then one matched ~1,200-char window per variant is synthesized with the
production Kokoro so the three can be listened to blind.

Runs INSIDE the scribe pod (Ollama's name only resolves in-cluster) against the
production models; writes to /data/experiments/tts_cleanup so results survive restarts.
Usage: python run.py [--phase extract|llm|audio|all] [--out DIR] [--max-chars N]
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
import time
from pathlib import Path

import httpx

from scribe.config import load_settings
from scribe.extract import extract
from scribe.listening import Chapter, clean_body, lint_script, prepare_document
from scribe.tts import synthesize_segment

log = logging.getLogger("experiment")

# Verbatim from the legacy MekVault tts-pipeline llm_processor.py (LM Studio era), tuned on
# the owner's own article corpus. Kept word-for-word so the experiment tests THAT prompt.
SYSTEM_PROMPT = """\
You are a text-cleaning assistant for a TTS (text-to-speech) pipeline. \
Your job is to fix problems that regex cannot handle. You will receive a \
chunk of article text that was extracted from a PDF and already \
regex-cleaned. Fix the following issues ONLY:

1. Split or joined words — use context to fix. E.g. "it's away" → "it's a way", \
"selfreport" → "self-report".
2. Smushed navigation/headers stuck onto article text — e.g. \
"REPORTPOLICYTECHHow the biggest..." → "How the biggest..."
3. Orphaned sentence fragments that don't form a complete thought — remove them.
4. Paragraphs that are smushed into a single long line with no breaks — add \
paragraph breaks where topic or speaker changes.
5. Embedded wrong-topic content (e.g. a marine biology abstract in a social \
media article) — remove it entirely.
6. Using the articles existing structure, format it in markdown—formatting \
headings and subheadings, etc. Do not create formatting from nothing. \
Simply add the formatting on top of what already exists.

RULES — you MUST follow these:
- NEVER paraphrase, summarize, or add new content.
- NEVER change the meaning or wording of sentences.
- Preserve all original punctuation and quoting style.
- Return ONLY the cleaned text, no explanations or commentary.
- If no fixes are needed, return the text unchanged.
"""

# Second prompt, added after the first run (2026-09-16): the legacy prompt's rule 6 asks
# for MARKDOWN formatting, which its pipeline stripped afterwards and which a speech
# script must never contain. Same rules otherwise, framed for listening.
SPEECH_PROMPT = """\
You are a text-cleaning assistant for a text-to-speech pipeline. The text you receive \
will be READ ALOUD by a speech synthesizer exactly as written. It was extracted from a \
web page or PDF and already cleaned by rules. Fix ONLY these problems:

1. Split or joined words — use context to fix. E.g. "it's away" → "it's a way", \
"selfreport" → "self-report".
2. Navigation, header, footer, caption or citation debris stuck into the prose — remove it.
3. Orphaned fragments that do not form a complete thought — remove them.
4. Text smushed into one long block — restore paragraph breaks where the topic or speaker \
changes.
5. Content that clearly belongs to a different document — remove it entirely.

RULES — you MUST follow these:
- Plain prose only. NO markdown, NO asterisks, NO heading marks, NO brackets, NO URLs.
- NEVER paraphrase, summarize, or add new content.
- NEVER change the meaning or wording of sentences.
- Preserve the original punctuation.
- Return ONLY the cleaned text, no explanations or commentary.
- If no fixes are needed, return the text unchanged.
"""

LLM_CHUNK_CHARS = 6000  # legacy pipeline's chunk size, paragraph-packed


def truncate_at_paragraph(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    return text[: cut if cut > max_chars // 2 else max_chars].rstrip()


def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    chunks, cur, n = [], [], 0
    for para in text.split("\n\n"):
        if cur and n + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += len(para) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def llm_polish(settings, text: str, prompt: str = SYSTEM_PROMPT) -> tuple[str, float, int]:
    """Repair prompt, chunk by chunk, against the production text model.
    Returns (polished, seconds, prompt_tokens_total)."""
    out, seconds, ptok = [], 0.0, 0
    for i, chunk in enumerate(chunk_paragraphs(text, LLM_CHUNK_CHARS), 1):
        t0 = time.monotonic()
        resp = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.text_model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk},
                ],
                "options": {
                    "temperature": 0,
                    "num_thread": settings.num_thread,
                    "num_ctx": settings.context_tokens,
                },
            },
            timeout=settings.ollama_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        dt = time.monotonic() - t0
        seconds += dt
        ptok += int(data.get("prompt_eval_count") or 0)
        content = (data.get("message") or {}).get("content", "").strip()
        log.info("llm chunk %d: %d -> %d chars in %.0fs", i, len(chunk), len(content), dt)
        out.append(content if content else chunk)
    return "\n\n".join(out), seconds, ptok


def metrics(name: str, text: str, base: str, seconds: float) -> dict:
    findings = lint_script([Chapter(name, [text])])
    ratio = difflib.SequenceMatcher(None, base, text, autojunk=False).quick_ratio()
    a, b = base.splitlines(), text.splitlines()
    diff = list(difflib.unified_diff(a, b, lineterm="", n=0))
    return {
        "chars": len(text),
        "delta_chars": len(text) - len(base),
        "lint_findings": len(findings),
        "lint_samples": findings[:8],
        "similarity_to_input": round(ratio, 4),
        "lines_removed": sum(1 for d in diff if d.startswith("-") and not d.startswith("---")),
        "lines_added": sum(1 for d in diff if d.startswith("+") and not d.startswith("+++")),
        "seconds": round(seconds, 1),
    }


def phase_extract(settings, sources, out: Path, max_chars: int) -> None:
    for src in sources:
        d = out / src["id"]
        d.mkdir(parents=True, exist_ok=True)
        if (d / "raw.txt").exists():
            log.info("%s: raw exists, skipping extract", src["id"])
            continue
        t0 = time.monotonic()
        target = src["url"]
        if src["kind"] == "pdf":
            pdf = d / "source.pdf"
            if not pdf.exists():
                r = httpx.get(target, follow_redirects=True, timeout=60,
                              headers={"User-Agent": settings.user_agent})
                r.raise_for_status()
                pdf.write_bytes(r.content)
            target = str(pdf)
        doc = extract(settings, target)
        prepared = prepare_document(doc.text)
        raw = truncate_at_paragraph(prepared, max_chars)
        (d / "raw.txt").write_text(raw)
        (d / "extract.json").write_text(json.dumps({
            "title": doc.title, "kind": doc.kind, "pages": len(doc.pages),
            "full_chars": len(doc.text), "excerpt_chars": len(raw),
            "seconds": round(time.monotonic() - t0, 1),
        }, indent=2))
        log.info("%s: extracted %d chars, excerpt %d", src["id"], len(doc.text), len(raw))


def phase_llm(settings, sources, out: Path) -> None:
    for src in sources:
        d = out / src["id"]
        raw = (d / "raw.txt").read_text()
        res = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}

        if "rules" not in res:
            t0 = time.monotonic()
            rules = clean_body(raw)
            (d / "rules.txt").write_text(rules)
            res["rules"] = metrics("rules", rules, raw, time.monotonic() - t0)
            (d / "metrics.json").write_text(json.dumps(res, indent=2))
        rules = (d / "rules.txt").read_text()

        if "llm" not in res:
            llm, secs, ptok = llm_polish(settings, raw)
            (d / "llm.txt").write_text(llm)
            res["llm"] = metrics("llm", llm, raw, secs) | {"prompt_tokens": ptok}
            (d / "metrics.json").write_text(json.dumps(res, indent=2))

        if "both" not in res:
            both, secs, ptok = llm_polish(settings, rules)
            (d / "both.txt").write_text(both)
            res["both"] = metrics("both", both, rules, secs) | {"prompt_tokens": ptok}
            (d / "metrics.json").write_text(json.dumps(res, indent=2))

        if "speech" not in res:
            # rules, then the speech-framed prompt (no markdown rule)
            sp, secs, ptok = llm_polish(settings, rules, prompt=SPEECH_PROMPT)
            (d / "speech.txt").write_text(sp)
            res["speech"] = metrics("speech", sp, rules, secs) | {"prompt_tokens": ptok}
            (d / "metrics.json").write_text(json.dumps(res, indent=2))
        log.info("%s: %s", src["id"],
                 {k: (v["lint_findings"], v["seconds"]) for k, v in res.items()})


def matched_window(text: str, anchor: str, size: int) -> str:
    """The ~size-char window starting at the paragraph that best matches `anchor`'s
    opening, so all three variants are heard over the same passage."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    key = anchor[:80]
    best = max(range(len(paras)), key=lambda i: difflib.SequenceMatcher(
        None, key, paras[i][:80]).ratio(), default=0)
    buf = []
    n = 0
    for p in paras[best:]:
        buf.append(p)
        n += len(p)
        if n >= size:
            break
    return "\n\n".join(buf)


def phase_audio(settings, sources, out: Path, window: int) -> None:
    for src in sources:
        d = out / src["id"]
        raw = (d / "raw.txt").read_text()
        paras = [p for p in raw.split("\n\n") if p.strip()]
        # Anchor a third of the way in: past any front matter, inside the body.
        anchor = paras[len(paras) // 3] if paras else raw
        for variant in ("rules", "llm", "both"):
            wav = d / f"listen-{variant}.wav"
            if wav.exists():
                continue
            text = matched_window((d / f"{variant}.txt").read_text(), anchor, window)
            (d / f"listen-{variant}.txt").write_text(text)
            secs = synthesize_segment(settings, text, wav)
            log.info("%s/%s: %d chars synthesized in %.0fs", src["id"], variant, len(text), secs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["extract", "llm", "audio", "all"])
    ap.add_argument("--out", default="/data/experiments/tts_cleanup")
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--only", help="comma-separated source ids")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    sources = json.loads((Path(__file__).parent / "sources.json").read_text())
    if args.only:
        keep = set(args.only.split(","))
        sources = [s for s in sources if s["id"] in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("extract", "all"):
        phase_extract(settings, sources, out, args.max_chars)
    if args.phase in ("llm", "all"):
        phase_llm(settings, sources, out)
    if args.phase in ("audio", "all"):
        phase_audio(settings, sources, out, args.window)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
