"""Turn extracted text into the fields the vault note needs — in a single model call."""

from __future__ import annotations

from pydantic import BaseModel, Field

from scribe.chunk import chunk
from scribe.config import Settings
from scribe.document import Document
from scribe.ollama import chat_structured

# Enforced server-side by Ollama's `format` field, so the model cannot drift into prose.
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tldr": {"type": "string"},
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "tldr", "summary", "tags"],
}

PROMPT = """\
You are summarizing a document for a personal knowledge vault.

Return JSON with exactly these fields:
- "title": a short, specific title for the document. No trailing punctuation. If the \
document is a chapter, section or excerpt of a larger work and the work is identifiable \
from the text, name both, e.g. "Algorithms, Chapter 3: Dynamic Programming".
- "tldr": 2-3 sentences capturing what this document is and why it matters. This is the \
only part the reader sees in chat, so it must stand alone.
- "summary": a THOROUGH summary in markdown. Use `##` headings and bullets. Cover every \
major section. Prefer specifics — names, numbers, definitions — over generalities. Do not \
restate the tldr.
- "tags": 3-8 lowercase kebab-case topic tags. No leading '#'. Omit generic words like \
"document" or "notes".

Document source: {source}

--- DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---
"""


MAP_SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "items": {"type": "string"}}},
    "required": ["points"],
}

MAP_PROMPT = """\
This is ONE SECTION of a longer document. Extract its key points as a JSON array of \
short, self-contained statements — facts, claims, definitions, numbers, conclusions.

Be specific and terse. Do not write prose, do not add commentary, and do not speculate \
about parts of the document you cannot see.

Section {n} of {total}:

--- SECTION TEXT ---
{text}
--- END SECTION TEXT ---
"""

COLLAPSE_PROMPT = """\
These are key points already extracted from CONSECUTIVE SECTIONS of a long document, in
order. There are too many to digest at once. Condense them into FEWER, higher-level key
points as a JSON array — merge related points, keep concrete facts, numbers and
conclusions, preserve the original order, and do not add commentary or speculation.

Group {n} of {total}:

--- KEY POINTS ---
{text}
--- END KEY POINTS ---
"""

REDUCE_PROMPT = """\
Below are key points extracted from a long document, in order, section by section. Write \
the summary of the WHOLE document from them.

Return JSON with exactly these fields:
- "title": a short, specific title. No trailing punctuation. If the document is a \
chapter or section of a larger, identifiable work, name both, e.g. "Algorithms, Chapter 3: \
Dynamic Programming".
- "tldr": 2-3 sentences on what this document is and why it matters. This is the only \
part the reader sees in chat, so it must stand alone.
- "summary": a THOROUGH summary in markdown, using `##` headings and bullets. Cover the \
document end to end — including its conclusion. Prefer specifics over generalities. Do \
not restate the tldr.
- "tags": 3-8 lowercase kebab-case topic tags. No leading '#'.

Document source: {source}

--- KEY POINTS ---
{points}
--- END KEY POINTS ---
"""


class Summary(BaseModel):
    title: str
    tldr: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    seconds: float = 0.0
    truncated_chars: int = 0
    # >1 when the document was too long for one pass and was map-reduced. Surfaced in the
    # note so a flatter summary is attributable rather than mysterious.
    sections: int = 1


def _fit_to_context(text: str, settings: Settings) -> tuple[str, int]:
    """Trim the document so prompt + response fit the server's context window.

    Ollama TRUNCATES over-length input silently rather than erroring, so an unguarded long
    document would produce a confident summary of only its opening pages. Trimming here at
    least makes the loss visible and reportable.

    Chars-per-token is approximate (~4 for English prose); the reserve absorbs the error
    along with the prompt scaffolding and the generated response.
    """
    budget_chars = int((settings.context_tokens - settings.response_reserve_tokens) * 4)
    if len(text) <= budget_chars:
        return text, 0
    return text[:budget_chars], len(text) - budget_chars


def _build(data: dict, doc: Document, seconds: float, *, dropped: int = 0,
           sections: int = 1) -> Summary:
    tags = [t.strip().lstrip("#").lower() for t in data.get("tags", [])]
    return Summary(
        title=(data.get("title") or doc.title or doc.source).strip(),
        tldr=(data.get("tldr") or "").strip(),
        summary=(data.get("summary") or "").strip(),
        tags=[t for t in tags if t],
        seconds=seconds,
        truncated_chars=dropped,
        sections=sections,
    )


def _map_reduce(settings: Settings, doc: Document, budget_chars: int, abort=None) -> Summary:
    """Summarize a document too long for one pass, without losing its tail.

    Truncation drops the conclusion, which is usually the part worth reading. This costs
    roughly 20% more wall clock for full coverage. It is genuinely lower resolution — a
    chunk summarizer cannot see the whole argument — so it is a fallback, never the
    default.
    """
    chunks = chunk(doc.text, settings.chunk_chars)
    elapsed = 0.0
    points: list[str] = []
    for n, piece in enumerate(chunks, start=1):
        # Cancellation checkpoint: a canceled job stops before the next model call
        # rather than after the whole document. Chunks are the long pole (~8 min each
        # on insp1), so this is the granularity that matters.
        if abort:
            abort()
        data, secs = chat_structured(
            settings,
            MAP_PROMPT.format(n=n, total=len(chunks), text=piece),
            MAP_SCHEMA,
        )
        elapsed += secs
        points.extend(str(p).strip() for p in data.get("points", []) if str(p).strip())

    # The reduce input must itself fit. When the key points overflow (a very long
    # document), COLLAPSE them recursively — condense point-groups through the model
    # until everything fits — rather than trimming. Trimming dropped the tail sections'
    # points, which meant the summary quietly thinned toward the end of the document
    # (measured: 8,992 chars cut on a 72-page, 9-section PDF). Coverage of the whole
    # document matters more here than per-point resolution: the full-detail paths are
    # the /scribe skill and the TTS pipeline, not this summary.
    joined = "\n".join(f"- {p}" for p in points)
    rounds = 0
    while len(joined) > budget_chars and rounds < 3:
        groups = chunk(joined, settings.chunk_chars)
        collapsed: list[str] = []
        for n, piece in enumerate(groups, start=1):
            if abort:
                abort()
            data, secs = chat_structured(
                settings,
                COLLAPSE_PROMPT.format(n=n, total=len(groups), text=piece),
                MAP_SCHEMA,
            )
            elapsed += secs
            collapsed.extend(str(p).strip() for p in data.get("points", []) if str(p).strip())
        rejoined = "\n".join(f"- {p}" for p in collapsed)
        if not collapsed or len(rejoined) >= len(joined):
            # A round that fails to shrink would loop forever; fall through to the trim.
            break
        joined = rejoined
        rounds += 1

    # Safety net for the depth cap or a non-converging collapse. With ~10x condensation
    # per round this should never fire on real input, but Ollama truncates silently, so
    # an unguarded overflow would be invisible.
    joined, dropped = _fit_to_context(joined, settings)
    if abort:
        abort()
    data, secs = chat_structured(
        settings, REDUCE_PROMPT.format(source=doc.source, points=joined), SCHEMA
    )
    return _build(data, doc, elapsed + secs, dropped=dropped, sections=len(chunks))


def summarize(settings: Settings, doc: Document, abort=None) -> Summary:
    budget_chars = int((settings.context_tokens - settings.response_reserve_tokens) * 4)

    # Single pass whenever the document fits: it is both faster AND better, since the
    # model sees the whole argument at once. Chunking is only for the case where the
    # alternative is losing the tail.
    if len(doc.text) > budget_chars:
        return _map_reduce(settings, doc, budget_chars, abort=abort)

    if abort:
        abort()
    prompt = PROMPT.format(source=doc.source, text=doc.text)
    data, seconds = chat_structured(settings, prompt, SCHEMA)
    # No truncation is possible on this path — it only runs when the document fits.
    return _build(data, doc, seconds)
