"""Minimal Ollama client.

Adapted from recipe-pipeline's `worker/src/recipe_pipeline/ollama_client.py`. Deliberately
copied rather than shared: that pipeline is live, and refactoring it into a library is out
of scope for scribe.
"""

from __future__ import annotations

import base64
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
        # Deterministic: transcription should not vary run to run.
        "options": {"temperature": 0},
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
