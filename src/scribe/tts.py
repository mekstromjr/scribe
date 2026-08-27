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


def health(settings: Settings) -> None:
    try:
        resp = httpx.get(f"{settings.tts_host}/health", timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTSError(f"cannot reach kokoro at {settings.tts_host}: {exc}") from exc


def synthesize_segment(settings: Settings, text: str, dest: Path) -> float:
    """Synthesize one segment to ``dest``. Returns wall-clock seconds.

    One retry on transport errors only: the pod restarting mid-request (this is how the
    k8s#145 OOM manifested to clients) yields a clean retry, while an HTTP 4xx means the
    request itself is wrong and retrying would just repeat it.
    """
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": settings.tts_voice,
        "response_format": "wav",
    }
    last: Exception | None = None
    for attempt in range(2):
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
            if attempt == 0:
                time.sleep(5.0)
    raise TTSError(f"tts request failed against {settings.tts_host}: {last}") from last
