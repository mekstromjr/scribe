"""Minimal Ollama client.

Adapted from recipe-pipeline's `worker/src/recipe_pipeline/ollama_client.py`. Deliberately
copied rather than shared: that pipeline is live, and refactoring it into a library is out
of scope for scribe.
"""

from __future__ import annotations

import base64
import json
import time

import httpx

from scribe.config import Settings


class OllamaError(RuntimeError):
    pass


def generate_with_image(
    settings: Settings, prompt: str, image_bytes: bytes, *, model: str | None = None
) -> tuple[str, float]:
    """Run a vision prompt against one image. Returns (text, wall_clock_seconds).

    Wall clock is measured here rather than read from the response because glm-ocr reports
    zero for load_duration/eval_count/eval_duration/total_duration — trusting those fields
    yields a confident, wrong "0.0s".
    """
    payload = {
        "model": model or settings.ocr_model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode()],
        "stream": False,
        # Deterministic: transcription should not vary run to run. num_thread matches the
        # container CPU limit — see Settings.num_thread for why the default oversubscribes.
        "options": {"temperature": 0, "num_thread": settings.num_thread},
    }
    t0 = time.monotonic()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json=payload,
            timeout=settings.ollama_timeout_seconds,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"vision request failed against {settings.ollama_host}: {exc}") from exc
    return resp.json().get("response", ""), time.monotonic() - t0


def health(settings: Settings) -> list[str]:
    """Return the model names this host serves.

    Used as an identity check: the dev Mac runs its own ollama, so a misconfigured host can
    silently answer with entirely different models. Callers should confirm the expected
    model is present before trusting any result.
    """
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"cannot reach ollama at {settings.ollama_host}: {exc}") from exc
    return [m["name"] for m in resp.json().get("models", [])]


def chat_structured(
    settings: Settings,
    prompt: str,
    schema: dict,
    *,
    model: str | None = None,
) -> tuple[dict, float]:
    """Ask the text model for JSON conforming to ``schema``. Returns (parsed, seconds).

    Ollama enforces the schema server-side via the ``format`` field, so the model cannot
    return prose that then has to be scraped. One call yields every field the note needs
    (title, tl;dr, summary, tags) instead of one round trip per field — which matters here
    because CPU generation runs at roughly 12-17 tok/s.

    ``think`` is disabled: qwen3.5 is a thinking model, and its reasoning trace would be
    generated at that same rate for output we discard.
    """
    payload = {
        "model": model or settings.text_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "think": False,
        "options": {"temperature": 0, "num_thread": settings.num_thread},
    }
    t0 = time.monotonic()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json=payload,
            timeout=settings.ollama_timeout_seconds,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"chat request failed against {settings.ollama_host}: {exc}") from exc

    content = resp.json().get("message", {}).get("content", "")
    try:
        return json.loads(content), time.monotonic() - t0
    except json.JSONDecodeError as exc:
        # Schema-enforced output should always parse; if it does not, surface the raw text
        # rather than a bare "expecting value" traceback.
        raise OllamaError(f"model returned non-JSON despite schema: {content[:300]!r}") from exc
