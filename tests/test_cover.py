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


def _panel_rows(img: Image.Image) -> list[int]:
    """Rows of a white cover that the title panel darkened (sampled at the centre)."""
    return [y for y in range(img.height) if max(img.getpixel((img.width // 2, y))[:3]) < 200]


class TestCoverTitle:
    @pytest.mark.parametrize("raw, want", [
        ("Lighthouse - Wikipedia", "Lighthouse"),
        ("The Federalist No. 10 | The Avalon Project", "The Federalist No. 10"),
        ("Why sourdough rises - Serious Eats", "Why sourdough rises"),
        # An em dash separates author from title; the short tail is the title.
        ("The Author — Notes on Quiet Grace", "The Author — Notes on Quiet Grace"),
        ("Lighthouses - A Short Field Guide", "Lighthouses - A Short Field Guide"),
        ("Part 1 - Origins: A History of Lighthouses on the Atlantic Coast",
         "Part 1 - Origins: A History of Lighthouses on the Atlantic Coast"),
        ("  Algorithms,   Chapter 3: Dynamic Programming ",
         "Algorithms, Chapter 3: Dynamic Programming"),
        ("Notes_on_logic_selections", "Notes on logic selections"),
        ("", ""),
    ])
    def test_site_suffix_dropped_subtitles_kept(self, raw, want):
        assert cover.cover_title(raw, strip_site=True) == want

    def test_non_web_titles_keep_a_hyphenated_subtitle(self):
        assert cover.cover_title("Lecture 5 - Dynamic Programming") == \
            "Lecture 5 - Dynamic Programming"
        assert cover.cover_title("Notes_on_logic") == "Notes on logic"

    def test_short_title_one_line_at_the_largest_size(self):
        font, lines = cover.fit_title("Lighthouses", 440, 512)
        assert lines == ["Lighthouses"]
        assert font.size == round(512 * cover.TITLE_MAX_FRACTION)

    def test_long_title_shrinks_and_never_overflows(self):
        text = "The Federalist Papers, Number Ten: The Same Subject Continued"
        font, lines = cover.fit_title(text, 440, 512)
        assert len(lines) <= cover.TITLE_MAX_LINES
        assert all(font.getlength(line) <= 440 for line in lines)
        assert " ".join(lines) == text

    def test_absurd_title_truncates_with_an_ellipsis(self):
        text = " ".join(["Lighthouse"] * 60)
        font, lines = cover.fit_title(text, 440, 512)
        assert len(lines) == cover.TITLE_MAX_LINES
        assert lines[-1].endswith("…")
        assert all(font.getlength(line) <= 440 for line in lines)

    def test_unbreakable_word_is_cut_not_overflowed(self):
        font, lines = cover.fit_title("x" * 200, 440, 512)
        assert all(font.getlength(line) <= 440 for line in lines)

    def test_panel_at_the_top_clear_of_the_badge(self):
        long = "The Federalist Papers, Number Ten: The Same Subject Continued"
        out = Image.open(io.BytesIO(cover.watermark(_png(), long)))
        rows = _panel_rows(out)
        assert rows and rows[0] >= round(512 * cover.MARGIN_FRACTION) - 1
        badge_top = 512 - round(512 * cover.MARGIN_FRACTION) - round(512 * cover.BADGE_FRACTION)
        assert rows[-1] < badge_top
        # Translucent, not opaque: the white behind still lifts the panel off black.
        y = rows[0] + 2
        assert min(out.getpixel((60, y))[:3]) > 40

    def test_no_title_no_panel(self):
        out = Image.open(io.BytesIO(cover.watermark(_png())))
        assert _panel_rows(out) == []

    def test_overlay_failure_keeps_the_cover(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("font missing")

        monkeypatch.setattr(cover, "title_overlay", boom)
        out = Image.open(io.BytesIO(cover.watermark(_png(), "Lighthouses")))
        assert out.format == "JPEG" and min(out.getpixel((256, 30))[:3]) > 240

    def test_make_cover_prints_the_title(self, monkeypatch):
        monkeypatch.setattr(cover, "generate", lambda *a, **k: _png())
        out = cover.make_cover(_settings(), scene="a scene", style="flat", seed=1,
                               title="Lighthouses")
        assert _panel_rows(Image.open(io.BytesIO(out)))[0] < 100


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


class TestCleanScene:
    """scribe#14: lettering is removed from the scene text by rule, because the negative
    prompt is inert at cfg 1.0. Cases are real scenes from the first production runs."""

    def test_the_first_production_drop(self):
        scene = ("A futuristic server room where rows of glowing servers emit streams of data "
                 "that form a chaotic swarm of digital agents attacking a locked gate "
                 "representing safety protocols, while three distinct figures stand at the "
                 "entrance holding signs labeled 'Evaluator', 'Democracy', and 'Global Pact' "
                 "to block the swarm.")
        out = cover.clean_scene(scene)
        assert out.endswith("stand at the entrance to block the swarm.")
        for word in ("sign", "label", "Evaluator", "Democracy", "Global Pact", "'"):
            assert word not in out

    def test_a_displayed_title(self):
        out = cover.clean_scene("A parchment scroll unrolling on a wooden desk, displaying the "
                                "title 'Federalist No. 10' alongside ink sketches of a map.")
        assert out == ("A parchment scroll unrolling on a wooden desk, alongside ink sketches "
                       "of a map.")

    def test_a_plaque_reading_quoted_text(self):
        out = cover.clean_scene('A wooden door with a brass plaque reading "Library" at the '
                                'end of a long hallway.')
        assert out == "A wooden door at the end of a long hallway."

    def test_unquoted_text_after_an_unambiguous_label_verb(self):
        assert cover.clean_scene("A desk with books under a banner that says Welcome, lit by "
                                 "a lamp.") == "A desk with books, lit by a lamp."

    def test_reading_a_book_is_not_lettering(self):
        s = "An old woman reading a book by a window."
        assert cover.clean_scene(s) == s

    def test_a_clean_scene_is_untouched(self):
        s = ("A towering cylindrical masonry lighthouse stands on a rugged rocky coastline with "
             "the first-order Fresnel lens glowing brightly against a twilight sky.")
        assert cover.clean_scene(s) == s

    def test_build_prompt_applies_it(self):
        p = cover.build_prompt("A gate with a sign reading 'Keep Out'.", "flat")
        assert "Keep Out" not in p and "sign" not in p


class TestOutlineSummary:
    def test_generic_chapters_are_dropped_and_real_ones_kept(self):
        chapters = ["Summary", "Full article", "Introduction", "Tower of Hanoi", "Mergesort"]
        s = cover.outline_summary("01-recursion", chapters, "01-recursion.pdf")
        assert s.title == "01-recursion"
        assert "- Tower of Hanoi" in s.summary and "- Mergesort" in s.summary
        assert "Full article" not in s.summary and "- Summary" not in s.summary
        assert s.summary.startswith("Source: 01-recursion.pdf")

    def test_no_chapters_no_body(self):
        assert cover.outline_summary("Title", ["Summary"]).summary == ""
