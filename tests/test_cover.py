"""Cover art (scribe#14): style resolution, prompt, sd-server client, watermark.

No network: chat_structured and httpx.post are stubbed.
"""

from __future__ import annotations

import base64
import io
import random
from concurrent.futures import Future

import httpx
import pytest
from PIL import Image

from scribe import cover
from scribe.config import Settings
from scribe.ollama import OllamaError
from scribe.summarize import Summary


def _settings(**kw) -> Settings:
    base = {"cover_host": "http://sd.invalid:1234"}
    base.update(kw)
    return Settings(**base)


def _summary() -> Summary:
    return Summary(title="Lighthouses", tldr="About lighthouses.", summary="...",
                   tags=["maritime", "history"])


def _no_call(*a, **k):
    raise OllamaError("down")


def _fake_chat(monkeypatch, answer: dict) -> dict:
    seen: dict = {}

    def fake(settings, prompt, schema, **kw):
        seen["prompt"], seen["schema"] = prompt, schema
        return answer, 1.0

    monkeypatch.setattr(cover, "chat_structured", fake)
    return seen


def _png(color=(250, 250, 250), size=512) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


class TestPlan:
    def test_off_when_no_server_configured(self):
        assert cover.plan(_settings(cover_host="", cover_style="flat"), _summary()).style == "off"

    def test_off_is_off_and_asks_nothing(self, monkeypatch):
        monkeypatch.setattr(cover, "chat_structured", _no_call)
        assert cover.plan(_settings(cover_style="off"), _summary()) == cover.CoverPlan("off")

    def test_a_named_style_asks_only_for_the_scene(self, monkeypatch):
        seen = _fake_chat(monkeypatch, {"scene": "a lighthouse at dusk"})
        p = cover.plan(_settings(cover_style="woodcut"), _summary())
        assert p == cover.CoverPlan("woodcut", "a lighthouse at dusk")
        assert "style" not in seen["schema"]["properties"]

    def test_random_draws_a_real_style_locally(self, monkeypatch):
        seen = _fake_chat(monkeypatch, {"scene": "s"})
        rng = random.Random(1)
        picks = {cover.plan(_settings(cover_style="random"), _summary(), rng=rng).style
                 for _ in range(50)}
        assert picks <= set(cover.STYLES) and len(picks) > 1
        assert "style" not in seen["schema"]["properties"]

    def test_auto_asks_for_scene_and_style_with_an_enum(self, monkeypatch):
        seen = _fake_chat(monkeypatch, {"scene": "a jar of starter", "style": "watercolor"})
        p = cover.plan(_settings(cover_style="auto"), _summary())
        assert p == cover.CoverPlan("watercolor", "a jar of starter")
        assert seen["schema"]["properties"]["style"]["enum"] == list(cover.STYLES)
        assert seen["schema"]["required"] == ["scene", "style"]
        assert "About lighthouses." in seen["prompt"]

    def test_the_prompt_carries_the_summary_but_capped(self, monkeypatch):
        seen = _fake_chat(monkeypatch, {"scene": "s"})
        long = _summary().model_copy(update={"summary": "x" * 50_000})
        cover.plan(_settings(cover_style="flat"), long)
        assert "x" * cover.SUMMARY_CHARS in seen["prompt"]
        assert "x" * (cover.SUMMARY_CHARS + 1) not in seen["prompt"]

    def test_auto_falls_back_when_ollama_is_down(self, monkeypatch):
        monkeypatch.setattr(cover, "chat_structured", _no_call)
        assert cover.plan(_settings(cover_style="auto"), _summary()) == \
            cover.CoverPlan(cover.FALLBACK_STYLE, "")

    def test_a_malformed_ollama_body_falls_back_too(self, monkeypatch):
        """chat_structured parses the HTTP body outside its own error handling, so a
        non-JSON body surfaces as a plain ValueError. It must not escape plan()."""
        def bad_json(*a, **k):
            raise ValueError("Expecting value")

        monkeypatch.setattr(cover, "chat_structured", bad_json)
        assert cover.plan(_settings(cover_style="auto"), _summary()) == \
            cover.CoverPlan(cover.FALLBACK_STYLE, "")

    def test_a_non_object_answer_falls_back(self, monkeypatch):
        monkeypatch.setattr(cover, "chat_structured", lambda *a, **k: (["flat"], 1.0))
        assert cover.plan(_settings(cover_style="woodcut"), _summary()) == \
            cover.CoverPlan("woodcut", "")

    def test_a_named_style_survives_ollama_being_down(self, monkeypatch):
        monkeypatch.setattr(cover, "chat_structured", _no_call)
        assert cover.plan(_settings(cover_style="poster"), _summary()).style == "poster"

    def test_auto_falls_back_on_an_off_list_answer(self, monkeypatch):
        _fake_chat(monkeypatch, {"scene": "s", "style": "cubist"})
        assert cover.plan(_settings(cover_style="auto"), _summary()).style == \
            cover.FALLBACK_STYLE

    def test_an_unknown_stored_value_still_gets_a_cover(self, monkeypatch):
        _fake_chat(monkeypatch, {"scene": "s"})
        assert cover.plan(_settings(cover_style="baroque"), _summary()).style == \
            cover.FALLBACK_STYLE


class TestPrompt:
    def test_scene_then_style_fragment(self):
        p = cover.build_prompt("a brass telescope on a rooftop.", "painterly")
        assert p == "a brass telescope on a rooftop, " + cover.STYLES["painterly"]

    def test_title_stands_in_for_a_missing_scene(self):
        """Audio records spooled before scribe#14 have no scene."""
        assert cover.build_prompt("", "flat", title="Tides").startswith("Tides, ")

    def test_seed_is_stable_per_job(self):
        assert cover.seed_for("job-1") == cover.seed_for("job-1")
        assert cover.seed_for("job-1") != cover.seed_for("job-2")


class TestGenerate:
    def test_posts_the_live_settings_and_decodes_the_image(self, monkeypatch):
        seen = {}

        def fake_post(url, json, timeout):
            seen.update(url=url, body=json, timeout=timeout)
            return httpx.Response(200, json={"images": [base64.b64encode(b"PNGDATA").decode()]},
                                  request=httpx.Request("POST", url))

        monkeypatch.setattr(cover.httpx, "post", fake_post)
        out = cover.generate(_settings(cover_steps=4), "a scene", seed=7)
        assert out == b"PNGDATA"
        assert seen["url"] == "http://sd.invalid:1234/sdapi/v1/txt2img"
        body = seen["body"]
        assert (body["steps"], body["cfg_scale"], body["width"], body["height"],
                body["seed"], body["sampler_name"]) == (4, 1.0, 512, 512, 7, "euler_a")
        assert "text" in body["negative_prompt"]

    def test_server_errors_become_cover_errors(self, monkeypatch):
        def fake_post(url, json, timeout):
            return httpx.Response(500, request=httpx.Request("POST", url))

        monkeypatch.setattr(cover.httpx, "post", fake_post)
        with pytest.raises(cover.CoverError):
            cover.generate(_settings(), "a scene", seed=1)

    def test_a_malformed_body_is_a_cover_error(self, monkeypatch):
        monkeypatch.setattr(cover.httpx, "post", lambda url, json, timeout: httpx.Response(
            200, json={"nope": 1}, request=httpx.Request("POST", url)))
        with pytest.raises(cover.CoverError):
            cover.generate(_settings(), "a scene", seed=1)


class TestWatermark:
    def test_jpeg_out_badge_in_the_bottom_right_only(self):
        out = Image.open(io.BytesIO(cover.watermark(_png())))
        assert out.format == "JPEG" and out.size == (512, 512)
        # Inside the badge (its navy disk) vs the untouched white top-left corner.
        size = round(512 * cover.BADGE_FRACTION)
        margin = round(512 * cover.MARGIN_FRACTION)
        cx = 512 - margin - size // 2
        r, g, b = out.getpixel((cx, 512 - margin - size // 6))[:3]
        assert max(r, g, b) < 120, "badge disk should be dark"
        assert min(out.getpixel((10, 10))[:3]) > 240, "rest of the image untouched"


class TestMakeCoverNeverRaises:
    def test_generation_failure_is_none(self, monkeypatch):
        def boom(*a, **k):
            raise cover.CoverError("server down")

        monkeypatch.setattr(cover, "generate", boom)
        assert cover.make_cover(_settings(), scene="a scene", style="flat", seed=1) is None

    def test_success_is_watermarked_jpeg(self, monkeypatch):
        monkeypatch.setattr(cover, "generate", lambda *a, **k: _png())
        out = cover.make_cover(_settings(), scene="a scene", style="flat", seed=1)
        assert out[:2] == b"\xff\xd8"

    def test_wait_on_a_failed_future_is_none(self):
        f: Future = Future()
        f.set_exception(RuntimeError("x"))
        assert cover.wait(f, 1) is None

    def test_wait_on_a_slow_future_times_out_to_none(self):
        assert cover.wait(Future(), 0.01) is None

    def test_wait_on_no_future_is_none(self):
        assert cover.wait(None, 1) is None
