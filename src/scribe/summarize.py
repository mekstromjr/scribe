"""Turn extracted text into the fields the vault note needs — in a single model call."""

from __future__ import annotations

from pydantic import BaseModel, Field

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
- "title": a short, specific title for the document. No trailing punctuation.
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


class Summary(BaseModel):
    title: str
    tldr: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    seconds: float = 0.0
    truncated_chars: int = 0


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


def summarize(settings: Settings, doc: Document) -> Summary:
    text, dropped = _fit_to_context(doc.text, settings)
    prompt = PROMPT.format(source=doc.source, text=text)
    data, seconds = chat_structured(settings, prompt, SCHEMA)

    tags = [t.strip().lstrip("#").lower() for t in data.get("tags", [])]
    return Summary(
        title=(data.get("title") or doc.title or doc.source).strip(),
        tldr=(data.get("tldr") or "").strip(),
        summary=(data.get("summary") or "").strip(),
        tags=[t for t in tags if t],
        seconds=seconds,
        truncated_chars=dropped,
    )
