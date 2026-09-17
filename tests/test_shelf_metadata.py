"""scribe#11/#12: voice menu, per-person shelf metadata, and the tags on the m4b."""

from __future__ import annotations

import json
from pathlib import Path

from scribe import abs as absmod
from scribe import package
from scribe.config import Settings
from scribe.tts import describe_voice, grouped_voices, sample_text


class TestVoiceGrouping:
    def test_describe(self):
        assert describe_voice("af_bella") == "American English, female"
        assert describe_voice("bm_george") == "British English, male"
        assert describe_voice("zf_xiaobei") == "Chinese, female"
        assert describe_voice("weird") == "other"

    def test_groups_english_first_and_v0_last_within_group(self):
        g = grouped_voices(["zf_xiaobei", "af_v0bella", "af_bella", "bm_george", "af_alloy"])
        names = [n for n, _ in g]
        assert names[0] == "American English, female" and names[1] == "British English, male"
        assert names[-1] == "Chinese, female"
        assert g[0][1] == ["af_alloy", "af_bella", "af_v0bella"]

    def test_sample_text_names_the_voice_and_shares_the_passage(self):
        a, b = sample_text("af_bella"), sample_text("bm_george")
        assert a.startswith("Hello, I'm af bella.")
        assert a.split(". ", 1)[1] == b.split(". ", 1)[1]


class TestVoiceMenu:
    def test_menu_links_samples_when_known(self, tmp_path):
        from scribe.runtime_config import save_section
        from scribe.slack_app import _voice_menu

        s = Settings(slack_bot_token="x", slack_app_token="x", spool_dir=str(tmp_path / "q"))
        text = _voice_menu(s, ["af_bella", "bm_george"], "bm_george")
        assert "Scribe voice samples" not in text
        save_section(s, "voice_samples", {"url": "https://shelf.example/item/abc"})
        text = _voice_menu(s, ["af_bella", "bm_george"], "bm_george")
        assert "<https://shelf.example/item/abc|Scribe voice samples>" in text
        assert "*British English, male:* `bm_george` ←" in text
        assert "*American English, female:* `af_bella`" in text


class TestUserName:
    def _client(self, payload=None, fail=False):
        class C:
            calls = 0

            def users_info(self, user):
                C.calls += 1
                if fail:
                    raise RuntimeError("slack down")
                return {"user": payload}
        return C()

    def test_first_name_from_real_name_and_cached(self):
        from scribe.slack_app import _UserName
        c = self._client({"real_name": "Michael Ekstrom", "profile": {}})
        n = _UserName()
        assert n.get(c, "U1") == "Michael"
        assert n.get(c, "U1") == "Michael" and type(c).calls == 1

    def test_failure_yields_none_not_an_exception(self):
        from scribe.slack_app import _UserName
        assert _UserName().get(self._client(fail=True), "U9") is None


class TestM4bTags:
    def test_narrator_series_comment_are_passed_to_ffmpeg(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"x")

        monkeypatch.setattr(package.subprocess, "run", fake_run)
        monkeypatch.setattr(package, "wav_seconds", lambda p: 1.0)
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF")
        package.build_m4b([package.ChapterAudio("c", [wav])], tmp_path / "o.m4b", title="T",
                          author="A", workdir=tmp_path, narrator="bm_george",
                          grouping="Michael", comment="https://x")
        cmd = seen["cmd"]
        assert "composer=bm_george" in cmd and "grouping=Michael" in cmd
        assert "comment=https://x" in cmd


class TestAbsMetadata:
    def test_patch_payload(self, monkeypatch):
        sent = {}

        class R:
            def raise_for_status(self):
                pass

        def fake_patch(url, headers=None, json=None, timeout=None):
            sent["url"], sent["json"] = url, json
            return R()

        monkeypatch.setattr(absmod.httpx, "patch", fake_patch)
        s = Settings(abs_token="t", abs_api_url="http://abs")
        absmod.set_item_metadata(s, "item1", narrator="bm_george",
                                 tags=["Michael"], description="https://x")
        assert sent["url"].endswith("/api/items/item1/media")
        assert sent["json"] == {
            "metadata": {"narrators": ["bm_george"], "description": "https://x"},
            "tags": ["Michael"],
        }

    def test_collection_created_then_reused(self, monkeypatch):
        calls = []
        state = {"collections": []}

        class R:
            def __init__(self, data=None):
                self.data = data or {}

            def raise_for_status(self):
                pass

            def json(self):
                return self.data

        def fake_get(url, headers=None, timeout=None, **kw):
            if url.endswith("/api/libraries"):
                return R({"libraries": [{"name": "Articles", "id": "L1",
                                         "folders": [{"id": "F1"}]}]})
            return R({"collections": state["collections"]})

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url.rsplit("/api", 1)[1], json))
            if url.endswith("/api/collections"):
                state["collections"].append({"id": "C1", "name": json["name"],
                                             "libraryId": "L1",
                                             "books": [{"id": b} for b in json["books"]]})
            return R()

        monkeypatch.setattr(absmod.httpx, "get", fake_get)
        monkeypatch.setattr(absmod.httpx, "post", fake_post)
        s = Settings(abs_token="t", abs_api_url="http://abs")
        absmod.add_to_collection(s, "i1", "Michael")   # creates
        absmod.add_to_collection(s, "i2", "Michael")   # adds
        absmod.add_to_collection(s, "i1", "Michael")   # already there: no call
        assert calls == [
            ("/collections", {"libraryId": "L1", "name": "Michael", "books": ["i1"]}),
            ("/collections/C1/book", {"id": "i2"}),
        ]

    def test_empty_patch_is_a_noop(self, monkeypatch):
        called = []
        monkeypatch.setattr(absmod.httpx, "patch", lambda *a, **k: called.append(1))
        absmod.set_item_metadata(Settings(abs_token="t"), "i")
        assert called == []


class TestAudioJobCarriesPerson:
    def test_round_trip(self, tmp_path):
        from scribe.queue import AudioJob, enqueue_audio, restore_audio
        s = Settings(spool_dir=str(tmp_path / "q"))
        aj = AudioJob(id="1", channel="C", thread_ts="1", source_label="x", user="U1",
                      doc={}, summary={}, person="Michael")
        enqueue_audio(s, aj)
        assert restore_audio(s)[0].person == "Michael"
        assert "person" in json.loads((tmp_path / "q" / "1.audio").read_text())


class TestBackfillSelection:
    """The first back-fill matched 'lacks this person's tag' and stamped one person's
    name on everyone's files. Candidates must be UNCLAIMED items only."""

    ITEMS = [
        {"id": "1", "title": "A", "series": "", "narrator": "am_michael", "tags": ["Michael"]},
        {"id": "2", "title": "B", "series": "Becky", "narrator": "af_river", "tags": ["Becky"]},
        {"id": "3", "title": "C", "series": "", "narrator": "", "tags": []},
        {"id": "4", "title": "Scribe voice samples", "series": "", "narrator": "",
         "tags": ["voice samples"]},
    ]

    def test_other_persons_items_are_never_touched(self):
        got = absmod.backfill_candidates(self.ITEMS, "Becky", known_people={"Michael", "Becky"})
        assert [i["id"] for i in got] == ["2", "3"]

    def test_own_clean_items_are_skipped_and_series_leftovers_included(self):
        got = absmod.backfill_candidates(self.ITEMS, "Michael", known_people={"Michael", "Becky"})
        assert [i["id"] for i in got] == ["3"]

    def test_title_filter(self):
        got = absmod.backfill_candidates(self.ITEMS, "Becky", known_people={"Michael", "Becky"},
                                         titles={"C"})
        assert [i["id"] for i in got] == ["3"]
