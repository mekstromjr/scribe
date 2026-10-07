"""Cover art for the audiobook (scribe#14).

A stable-diffusion.cpp sd-server draws one 512x512 image per document, of a scene the
text model describes from the finished summary, in the sender's chosen style, and the
scribe badge is stamped in the bottom-right corner. The JPEG is embedded in the m4b, and
Audiobookshelf's scanner extracts an embedded cover for any new item that has none, so no
ABS API call is needed.

Two halves, split across the two workers on purpose:

* ``plan`` runs in the SUMMARIZE worker at hand-off: one short text-model call that
  describes the scene and, for ``auto``, picks the style (``random`` draws one locally).
  The concrete style and the scene are frozen into the audio spool record, so the audio
  worker never talks to ollama (scribe#7) and a resumed job keeps the cover it was
  promised.

The scene is its own call rather than a field in the summary request on purpose: adding a
required field there shifted the summaries themselves (one third shorter on one article,
longer on others; scribe#14), and the summary is the product.
* ``submit`` runs in the AUDIO worker and generates the image on a background thread
  while Kokoro synthesizes. The two run on different machines, so on any normal-length
  document the cover is ready before the m4b is packaged.

Best-effort throughout: every failure is logged and yields no cover, never a failed job.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import httpx
from PIL import Image, ImageFilter

from scribe.config import Settings
from scribe.ollama import chat_structured
from scribe.summarize import Summary

log = logging.getLogger("scribe.cover")

# Style name -> prompt fragment appended to the scene. Fragments are the ones sampled in
# Phase 0 (scribe#14); all six are kept because they diverge more on real documents than
# on the two test scenes.
STYLES: dict[str, str] = {
    "flat": "flat minimalist illustration, limited color palette, clean shapes",
    "storybook": "detailed storybook illustration, ink and color",
    "painterly": "painterly oil painting, visible brush strokes",
    "watercolor": "soft watercolor painting, paper texture",
    "woodcut": "linocut woodcut print, bold lines, two colors",
    "poster": "vintage travel poster style, bold flat colors",
}
# Settings values that are not themselves a style.
AUTO, RANDOM, OFF = "auto", "random", "off"
CHOICES = (AUTO, RANDOM, *STYLES, OFF)

# What `auto` falls back to when the text model is unreachable or answers off-list.
FALLBACK_STYLE = "flat"

# INERT at CFG_SCALE 1.0: classifier-free guidance at 1.0 never runs the unconditional
# pass, which is where a negative prompt acts. Sent anyway so raising cfg_scale needs no
# second change. What actually keeps lettering out is the scene wording (SCENE_FIELD);
# a faint painter's signature still slips into some painterly covers (scribe#14).
NEGATIVE_PROMPT = "text, letters, words, watermark, signature, logo, blurry, deformed"

# Fixed by the model rather than tuned: SD-Turbo is trained at 512x512 and distilled for
# guidance-free sampling, so cfg 1.0 with euler_a.
SIZE = 512
CFG_SCALE = 1.0
SAMPLER = "euler_a"

# Badge placement, chosen against light, dark and busy test covers (scribe#14).
BADGE_FRACTION = 0.14
MARGIN_FRACTION = 0.03

# The model reads the summary, not the document: it is plenty for a scene, and keeps the
# prompt-eval cost of this call to a fraction of the summary's own.
SUMMARY_CHARS = 6000

SCENE_FIELD = """\
- "scene": ONE sentence describing a single concrete visual scene that captures this \
document's subject, for its cover illustration. Name physical objects, places or \
landscapes from the content. Prefer objects and settings over people and animals. \
Describe only what is seen: never mention titles, writing, words, labels, signs, \
numbers or logos, and no art-style words."""

STYLE_FIELD = """\
- "style": the art style that best suits the document, one of:
  - "flat": flat minimalist illustration. Clean and modern; suits technology, science, \
explainers, how-tos.
  - "storybook": detailed storybook illustration, ink and color. Suits narratives, history, \
biography, fiction.
  - "painterly": oil painting with visible brush strokes. Suits art, culture, ideas, \
reflective essays.
  - "watercolor": soft watercolor. Suits nature, health, travel, gentle or personal topics.
  - "woodcut": bold linocut print. Suits politics, philosophy, law, serious or historical \
argument.
  - "poster": vintage travel poster. Suits places, events, adventure, upbeat topics."""

PLAN_PROMPT = """\
You are planning the cover illustration for an audiobook made from this document.

Return JSON with exactly these fields:
{fields}

Title: {title}
TL;DR: {tldr}
Tags: {tags}

--- SUMMARY ---
{summary}
--- END SUMMARY ---
"""

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe-cover")


class CoverError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoverPlan:
    style: str        # a STYLES key, or OFF
    scene: str = ""   # empty when the model call failed; build_prompt falls back to title


def plan(settings: Settings, summary: Summary, *,
         rng: random.Random | None = None) -> CoverPlan:
    """Decide this document's cover. Runs in the summarize worker; never raises.

    A failed model call still yields a cover: the style falls back to FALLBACK_STYLE (for
    ``auto``) and the scene to the title. So does an unknown stored style (one since
    removed, a hand-edited config), rather than silently disabling covers.
    """
    choice = (settings.cover_style or OFF).strip().lower()
    if not settings.cover_host or choice == OFF:
        return CoverPlan(OFF)
    if choice == RANDOM:
        style = (rng or random).choice(list(STYLES))
    elif choice == AUTO:
        style = None  # the model picks
    elif choice in STYLES:
        style = choice
    else:
        log.warning("unknown cover_style %r; using %s", choice, FALLBACK_STYLE)
        style = FALLBACK_STYLE

    schema: dict = {"type": "object", "properties": {"scene": {"type": "string"}},
                    "required": ["scene"]}
    fields = SCENE_FIELD
    if style is None:
        schema["properties"]["style"] = {"type": "string", "enum": list(STYLES)}
        schema["required"].append("style")
        fields += "\n" + STYLE_FIELD
    prompt = PLAN_PROMPT.format(fields=fields, title=summary.title, tldr=summary.tldr,
                                tags=", ".join(summary.tags),
                                summary=summary.summary[:SUMMARY_CHARS])
    try:
        data, secs = chat_structured(settings, prompt, schema)
        if not isinstance(data, dict):
            raise TypeError(f"expected a JSON object, got {type(data).__name__}")
    except Exception as exc:  # OllamaError, or a malformed body: either way, no scene
        log.warning("cover plan failed, falling back: %s", exc)
        return CoverPlan(style or FALLBACK_STYLE)

    scene = str(data.get("scene") or "").strip()
    if style is None:
        style = str(data.get("style") or "").strip().lower()
        if style not in STYLES:
            log.warning("cover style auto-pick returned %r, using %s", style, FALLBACK_STYLE)
            style = FALLBACK_STYLE
    log.info("cover.plan style=%s (%s) seconds=%.1f scene=%r", style, choice, secs, scene)
    return CoverPlan(style, scene)


def build_prompt(scene: str, style: str, *, title: str = "") -> str:
    """Scene first, style last. With no scene (the plan call failed) the title is a
    weaker but still document-specific subject."""
    subject = (scene or title or "an open book on a desk").strip().rstrip(".")
    return f"{subject}, {STYLES[style]}"


def seed_for(job_id: str) -> int:
    """Deterministic per job, so a cover regenerated after a restart is the same image."""
    return int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)


def generate(settings: Settings, prompt: str, *, seed: int) -> bytes:
    """One txt2img call; returns the PNG bytes. Raises CoverError."""
    body = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": settings.cover_steps,
        "cfg_scale": CFG_SCALE,
        "width": SIZE,
        "height": SIZE,
        "seed": seed,
        "sampler_name": SAMPLER,
    }
    try:
        resp = httpx.post(f"{settings.cover_host}/sdapi/v1/txt2img", json=body,
                          timeout=settings.cover_timeout_seconds)
        resp.raise_for_status()
        return base64.b64decode(resp.json()["images"][0])
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise CoverError(f"image generation failed against {settings.cover_host}: {exc}") from exc


@lru_cache(maxsize=1)
def _badge() -> Image.Image:
    with files("scribe").joinpath("assets/scribe-badge.png").open("rb") as fh:
        return Image.open(fh).convert("RGBA")


def watermark(png: bytes) -> bytes:
    """Stamp the badge bottom-right with a soft shadow; return a baseline JPEG.

    JPEG because that is what an m4b cover is expected to be (mp4 `covr` takes JPEG or
    PNG, and players handle JPEG most widely), at a quality where the badge's flat
    teal stays clean.
    """
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    w = image.width
    size, margin = round(w * BADGE_FRACTION), round(w * MARGIN_FRACTION)
    badge = _badge().resize((size, size), Image.LANCZOS)

    pad = 8
    shadow = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), (pad, pad), badge.getchannel("A"))
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))

    x = y = w - size - margin
    image.alpha_composite(shadow, (x - pad + 2, y - pad + 3))
    image.alpha_composite(badge, (x, y))
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=92)
    return out.getvalue()


def make_cover(settings: Settings, *, scene: str, style: str, seed: int,
               title: str = "") -> bytes | None:
    """Generate + watermark. Never raises; None means no cover this time."""
    t0 = time.monotonic()
    try:
        png = generate(settings, build_prompt(scene, style, title=title), seed=seed)
        jpeg = watermark(png)
    except Exception as exc:
        log.warning("no cover for %r: %s", title, exc)
        return None
    log.info("cover.done style=%s seconds=%.0f bytes=%d", style, time.monotonic() - t0,
             len(jpeg))
    return jpeg


def submit(settings: Settings, *, scene: str, style: str, seed: int,
           title: str = "") -> Future[bytes | None]:
    """Start a cover on the single cover thread. One at a time, because the image
    server is one GPU and a second request would only queue behind the first there."""
    return _executor.submit(make_cover, settings, scene=scene, style=style, seed=seed,
                            title=title)


def wait(future: Future[bytes | None] | None, timeout: float) -> bytes | None:
    """The cover if it finishes within ``timeout``, else None. Never raises."""
    if future is None:
        return None
    try:
        return future.result(timeout=timeout)
    except Exception as exc:
        log.warning("cover not ready in time, packaging without it: %s", exc)
        return None
