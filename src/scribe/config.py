"""Runtime settings. Env-overridable so the same code runs in-cluster and on a laptop."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
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

    # --- OCR runaway guard (scribe#4) ---------------------------------------------
    # glm-ocr loops on sparse pages: a book's title page ran to 14,040 tokens and was
    # still going when the 16k context filled (2026-09-08). ollama's default num_predict
    # is unlimited and the server runs with --context-shift, so without a cap the ONLY
    # stop is the client timeout -- an hour per attempt, three attempts, no note.
    #
    # Sized from that document, not guessed: a dense two-page landscape spread produced
    # 739-1,475 generated tokens per page. 2x the densest observed page gives a legitimate
    # page headroom and cuts a runaway at ~11 min (slowest observed 4.5 tok/s) instead of
    # 60. A response that stops here carries done_reason "length" -- the signal that the
    # model never reached the end of the page -- and the page is SKIPPED, not summarized
    # from repetition junk. Re-measure if the OCR model or page geometry changes.
    ocr_num_predict: int = 3000
    # Per-page ceiling, separate from ollama_timeout_seconds (which is sized for a full
    # summarization call). Must exceed the time the cap above takes to reach at the
    # slowest observed rate (3000 / 4.5 tok/s ~ 11 min, plus ~1 min of prompt eval);
    # otherwise the timeout fires first and the done_reason signal is lost. A timed-out
    # page is also SKIPPED rather than requeued: the server was up and working, so a
    # retry would reproduce the same result at temperature 0.
    ocr_timeout_seconds: float = 900.0

    # MEASURED, not guessed. llama.cpp sizes its thread pool from HOST core count and
    # ignores the container's cgroup quota, so on a 10-CPU node with a 6-CPU limit it
    # oversubscribes and the threads fight CFS throttling. Pinning to the limit measured
    # +73% generation (4.0 -> 6.9 tok/s) and +48% prompt eval on qwen3.5:4b.
    #
    # Must track the CPU limit in k8s apps/ollama-mini/deployment.yaml. There is no server
    # env var for this — it is a per-request model option, so every caller must send it.
    #
    # Note: changing this value forces Ollama to reload the model (~30s), so it should be
    # set once and left alone rather than tuned per request.
    num_thread: int = 6

    # Declaring scribe honestly, with a contact address, is what Wikimedia's robot policy
    # asks for — and it empirically works better than spoofing a browser, which their edge
    # refuses with a 403 bot-policy notice. Keep the tool name and contact in place.
    user_agent: str = "scribe/0.1 (https://meklab.net; michael.ekstrom@me.com) python-httpx"

    # Must match OLLAMA_CONTEXT_LENGTH on the server (32768 as of k8s tag v1.34.1). Ollama
    # truncates over-length input SILENTLY, so scribe trims first and reports what it cut.
    context_tokens: int = 32768
    # Headroom for the prompt scaffolding and the generated summary itself.
    response_reserve_tokens: int = 6000

    # Chunk size for the map-reduce fallback (only used when a document would otherwise be
    # truncated). ~24k chars is about 6k tokens: large enough that a chunk carries real
    # context, small enough that the map step stays cheap. Generation dominates cost, so
    # more chunks is more expensive than bigger chunks.
    chunk_chars: int = 24000

    # Comma-separated Slack channel IDs where EVERY link or file is processed without
    # an @-mention (a "drop channel"). Everywhere else channels stay mention-only —
    # membership alone must not turn a discussion channel into a firehose, and the
    # message.channels subscription delivers every channel scribe is a member of.
    drop_channels: str = ""

    @property
    def drop_channel_ids(self) -> set[str]:
        return {c.strip() for c in self.drop_channels.split(",") if c.strip()}

    # --- ETA shown in the Slack ack -------------------------------------------------
    # All three MEASURED on insp1 (CT 260) on 2026-08-27, not derived. They are host
    # constants: rehoming ollama means re-measuring them, like num_thread.
    #   ocr page:  ~3 min/page incl. the amortized glm-ocr <-> qwen swap
    #   map chunk: warm average over the clean post-16k-fix Bloom filter chunks
    #   single:    intercept + per-char rate fit so small docs are not quoted the
    #              full-budget price; a budget-full single call lands at ~700s
    eta_ocr_page_seconds: float = 190.0
    eta_chunk_seconds: float = 480.0
    eta_single_base_seconds: float = 60.0
    # FALLBACK timezone for the ETA clock. The primary source is the Slack profile tz
    # of whoever sent the message (users.info, cached) — Slack keeps that current when
    # the user travels. This value covers a missing profile tz and non-Slack callers.
    # Slack's <!date> token was viewer-local but only renders 12-hour time, hence
    # server-side rendering at all.
    timezone: str = "America/Denver"
    eta_single_seconds_per_char: float = 0.02
    # Audio: production Kokoro measured 2026-09-14 at ~100 s per 3000-char segment.
    # All eta_* values are PRIORS: calibration.py learns a per-stage correction from
    # every clean completion (scribe#8), so these only matter on a fresh spool.
    eta_audio_seconds_per_char: float = 1 / 30

    # --- TTS / Audiobookshelf (home#174) --------------------------------------------
    # Kokoro (kokoro-tts in this same namespace) turns the note into an m4b that lands
    # in Audiobookshelf's Articles library and back in the Slack thread. Best-effort by
    # design: the note is the product, the audio is a bonus, so audio failures never
    # fail or requeue a job.
    tts_enabled: bool = True

    # How the full note is delivered in the Slack thread (scribe#5): a file in this
    # format, or "none" for the TL;DR reply alone. `md` is the raw note, frontmatter and
    # all, so it drops straight into an Obsidian vault; `pdf`/`docx` are for people who
    # will never open a markdown file. Runtime-toggleable from Slack like tts_enabled.
    note_format: Literal["pdf", "md", "docx", "none"] = "pdf"

    # ClusterIP-only, same reasoning as ollama: kokoro-fastapi ships no authentication.
    # Laptop testing: kubectl -n infra port-forward svc/kokoro-tts 8880
    tts_host: str = "http://kokoro-tts.infra.svc.cluster.local:8880"

    # af_bella, chosen by ear from samples on 2026-08-27 (af_heart and am_michael were
    # the runners-up). ONE voice per deployment, not per-request variety: the server
    # caches every voice tensor it loads and OOM-killed a 2Gi limit when a sampling
    # run loaded seven (k8s#145) — the 3Gi limit assumes a single cached voice.
    tts_voice: str = "af_bella"

    # Per-request text cap. Kokoro splits internally, but request-level chunking keeps
    # single-request wall clock bounded (~2 min at measured 1.6x realtime) so a
    # mid-article failure loses one segment, not the whole synthesis. Relevant beyond
    # politeness: the server blocks its event loop while generating (k8s#145 — an HTTP
    # liveness probe used to SIGKILL it mid-request for exactly this reason), and
    # shorter requests also survive port-forward flakiness in dev.
    tts_max_chars: int = 3000

    # Generous per-segment ceiling: at 1.6x realtime a 4000-char segment (~5 min of
    # audio) synthesizes in ~3 min; 15 min means something is actually wrong.
    tts_timeout_seconds: float = 900.0

    # In-cluster API endpoint vs the public link put in notes and Slack replies. The
    # API talks service-to-service; the link must open on a phone.
    abs_api_url: str = "http://audiobookshelf.prod.svc.cluster.local:80"
    # shelf.meklab.net, NOT audiobookshelf.meklab.net — the ingress host is "shelf"
    # (apps/audiobookshelf/ingress.yaml); the longer name has no DNS record and every
    # link built from it was dead on arrival.
    abs_web_url: str = "https://shelf.meklab.net"
    abs_library_name: str = "Articles"
    # Vault: secret/infra/scribe property abs-token.
    abs_token: str = ""

    # --- Slack ---------------------------------------------------------------------
    # Socket Mode needs BOTH: a bot token from installing the app, and an app-level
    # token with connections:write minted under Basic Information. See SLACK_SETUP.md —
    # the app-level token cannot be created from a manifest.
    # Durable spool for pending Slack jobs, so a restart resumes rather than silently
    # dropping queued work. In-cluster this should be a PVC mount — an emptyDir would
    # defeat the point.
    spool_dir: str = "~/.local/state/scribe/queue"

    # Attempts per job before giving up. Transient failures (ollama restarting, a
    # network blip) must not permanently lose a queued document. Permanent failures --
    # a dead link, an unsupported file type -- are not retried at all.
    max_attempts: int = 3

    slack_bot_token: str = ""
    # Slack's own docs call this the "app-level token"; it is stored locally as
    # SCRIBE_SLACK_WRITE_TOKEN (after its connections:write scope). Accept both names so
    # neither the existing environment nor the Slack-standard term has to give way.
    slack_app_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SCRIBE_SLACK_APP_TOKEN", "SCRIBE_SLACK_WRITE_TOKEN"
        ),
    )


def load_settings() -> Settings:
    return Settings()
