"""Kokoro client: text segments in, WAV files out.

WAV rather than MP3 from the server: the segments get concatenated and encoded ONCE to
AAC in package.py, and a lossy-to-lossy mp3->aac hop would pay the quality tax twice.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from scribe.config import Settings


class TTSError(RuntimeError):
    pass


def voices(settings: Settings) -> list[str]:
    """English voice ids the server actually serves.

    Asked rather than hardcoded so /scribevoice can reject a typo at command time
    instead of at synthesis time, tens of minutes later.
    """
    try:
        resp = httpx.get(f"{settings.tts_host}/v1/audio/voices", timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTSError(f"cannot list voices at {settings.tts_host}: {exc}") from exc
    raw = resp.json().get("voices", [])
    names = [v.get("id", v.get("name", "")) if isinstance(v, dict) else str(v) for v in raw]
    # Kokoro prefixes by language and gender: a=American, b=British.
    return sorted(n for n in names if n[:3] in ("af_", "am_", "bf_", "bm_"))


def health(settings: Settings) -> None:
    try:
        resp = httpx.get(f"{settings.tts_host}/health", timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTSError(f"cannot reach kokoro at {settings.tts_host}: {exc}") from exc


# Waits between transport-error retries. Sized for a server RESTART, not a blip: the
# kokoro pod has died mid-request twice now (OOM during voice sampling, then a
# SIGSEGV on 2026-08-27), and the container takes ~30-60s to come back up and load
# the model. A single quick retry just hits the corpse.
_RETRY_WAITS = (10.0, 60.0)


def synthesize_segment(settings: Settings, text: str, dest: Path) -> float:
    """Synthesize one segment to ``dest``. Returns wall-clock seconds.

    Transport errors retry patiently (see _RETRY_WAITS); an HTTP 4xx/5xx response
    means the request itself was handled and judged, so it fails immediately.
    """
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": settings.tts_voice,
        "response_format": "wav",
    }
    last: Exception | None = None
    for attempt in range(len(_RETRY_WAITS) + 1):
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{settings.tts_host}/v1/audio/speech",
                json=payload,
                timeout=settings.tts_timeout_seconds,
            )
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return time.monotonic() - t0
        except httpx.HTTPStatusError as exc:
            raise TTSError(
                f"kokoro rejected segment ({exc.response.status_code}): "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            last = exc
            if attempt < len(_RETRY_WAITS):
                time.sleep(_RETRY_WAITS[attempt])
    raise TTSError(f"tts request failed against {settings.tts_host}: {last}") from last
