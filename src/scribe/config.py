"""Runtime settings. Env-overridable so the same code runs in-cluster and on a laptop."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCRIBE_", env_file=".env", extra="ignore")

    # In-cluster default. ollama-mini is ClusterIP-only (no Ingress) because Ollama ships
    # no authentication, so this is the only address that resolves in production.
    #
    # For laptop testing, port-forward and override:
    #   kubectl -n infra port-forward svc/ollama-mini 11500:11434
    #   SCRIBE_OLLAMA_HOST=http://127.0.0.1:11500 scribe extract file.pdf
    #
    # Do NOT use local port 11434 — the dev Mac runs its own native ollama there, and
    # "localhost:11434" resolves ambiguously between the two (kubectl binds ::1, native
    # ollama binds 127.0.0.1). That silently sends requests to the wrong server.
    ollama_host: str = "http://ollama-mini.infra.svc.cluster.local:11434"

    ocr_model: str = "glm-ocr:latest"
    text_model: str = "qwen3.5:4b"

    # A page yielding fewer than this many non-whitespace characters is treated as having
    # no usable text layer, and falls through to OCR. Slides are legitimately sparse — a
    # title-only slide can be ~20 chars — so this is deliberately low. Raising it wastes
    # ~20s of CPU OCR on pages whose text we could have read for free.
    min_page_chars: int = 24

    # Longest-edge cap for images sent to the vision model. Full 300-DPI renders cost CPU
    # in the vision encoder with no accuracy gain at this model size.
    ocr_max_edge: int = 1500

    # Render scale for pages that need OCR. PDFium's base is 72 DPI.
    #
    # 150 measured as the sweet spot: 200 DPI cost ~50% more wall clock for a byte-identical
    # transcription on the test page. Vision-encoder cost scales with pixel count, and this
    # model gains no accuracy above 150 on printed text.
    ocr_render_dpi: int = 150

    # glm-ocr on CPU is ~20s/page, so a long image-only PDF is a multi-minute job.
    # 0 disables the cap.
    max_ocr_pages: int = 0

    ollama_timeout_seconds: float = 1800.0

    # Declaring scribe honestly, with a contact address, is what Wikimedia's robot policy
    # asks for — and it empirically works better than spoofing a browser, which their edge
    # refuses with a 403 bot-policy notice. Keep the tool name and contact in place.
    user_agent: str = "scribe/0.1 (https://meklab.net; michael.ekstrom@me.com) python-httpx"


def load_settings() -> Settings:
    return Settings()
