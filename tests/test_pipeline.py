"""scribe#7: the audio half runs on its own worker, handed over through the spool, so
the next document summarizes while the last one synthesizes."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from scribe import slack_app
from scribe.config import Settings
from scribe.document import Document, Method, Page
from scribe.queue import (
    AudioJob,
    Job,
    complete_audio,
    enqueue,
    enqueue_audio,
    restore,
    restore_audio,
)
from scribe.slack_app import _Active, _Pending
from scribe.summarize import Summary


def _settings(tmp_path, **kw) -> Settings:
    return Settings(slack_bot_token="xoxb-x", slack_app_token="xapp-x",
                    spool_dir=str(tmp_path / "q"), abs_token="t", **kw)


def _doc(text="Heading\n\nBody text here.") -> Document:
    return Document(source="d", kind="link", title="d",
                    pages=[Page(number=1, text=text, method=Method.TEXT_LAYER)])


def _summary() -> Summary:
    return Summary(title="T", tldr="tl", summary="sum")


def _fake_export(body, fmt, *, stem, out_dir):
    path = out_dir / f"{stem}.{fmt}"
    path.write_text("x")
    return path


def _audio_job(job: Job, **overrides) -> AudioJob:
    return AudioJob(id=job.id, channel=job.channel, thread_ts=job.thread_ts,
                    source_label=job.source_label, user=job.user,
                    doc=_doc().model_dump(mode="json"),
                    summary=_summary().model_dump(mode="json"),
                    overrides={"tts_voice": "bm_george", "tts_enabled": True, **overrides})


class _Client:
    def __init__(self):
        self.posts: list[str] = []
        self.uploads: list[str] = []
        self.lock = threading.Lock()

    def chat_postMessage(self, **kw):
        with self.lock:
            self.posts.append(kw.get("text", ""))

    def files_upload_v2(self, **kw):
        with self.lock:
            self.uploads.append(kw.get("filename", ""))


class TestAudioSpool:
    def test_round_trip_in_arrival_order(self, tmp_path):
        s = _settings(tmp_path)
        j1, j2 = Job.new("C", "1", "t", "one"), Job.new("C", "2", "t", "two")
        enqueue_audio(s, _audio_job(j2))
        enqueue_audio(s, _audio_job(j1))
        got = restore_audio(s)
        assert [a.id for a in got] == [j1.id, j2.id]
        assert got[0].overrides["tts_voice"] == "bm_george"
        assert Document.model_validate(got[0].doc).text == _doc().text

    def test_audio_records_are_invisible_to_the_summarize_restore(self, tmp_path):
        """restore() deletes any *.json it cannot parse as a Job; an audio record must
        not be collateral."""
        s = _settings(tmp_path)
        j = Job.new("C", "1", "t", "one")
        enqueue_audio(s, _audio_job(j))
        assert restore(s) == []
        assert len(restore_audio(s)) == 1

    def test_complete_removes_only_the_audio_record(self, tmp_path):
        s = _settings(tmp_path)
        j = Job.new("C", "1", "t", "one")
        enqueue(s, j)
        aj = _audio_job(j)
        enqueue_audio(s, aj)
        complete_audio(s, aj)
        assert restore_audio(s) == []
        assert [x.id for x in restore(s)] == [j.id]

    def test_corrupt_record_is_dropped_not_fatal(self, tmp_path):
        s = _settings(tmp_path)
        j = Job.new("C", "1", "t", "one")
        enqueue_audio(s, _audio_job(j))
        (tmp_path / "q" / "0000-bad.audio").write_text("{nope")
        assert [a.id for a in restore_audio(s)] == [j.id]


class TestHandoff:
    def _run_summarize(self, tmp_path, monkeypatch, settings=None, user=None):
        s = settings or _settings(tmp_path)
        monkeypatch.setattr(slack_app, "extract", lambda *a, **k: _doc())
        monkeypatch.setattr(slack_app, "summarize", lambda *a, **k: _summary())
        monkeypatch.setattr(slack_app, "export_note", _fake_export)
        handed: list[AudioJob] = []
        job = Job.new("C", "1", "https://x", "x")
        job.user = user
        enqueue(s, job)
        client = _Client()
        slack_app._process(s, client, job, audio_submit=handed.append)
        return s, job, client, handed

    def test_note_posts_then_audio_is_spooled_and_submitted(self, tmp_path, monkeypatch):
        s, job, client, handed = self._run_summarize(tmp_path, monkeypatch)
        assert len(handed) == 1 and handed[0].id == job.id
        assert [a.id for a in restore_audio(s)] == [job.id], "spooled before hand-off"
        assert restore(s) == [], "summarize record completed"
        assert client.uploads, "note file posted before audio"

    def test_handoff_freezes_the_senders_settings(self, tmp_path, monkeypatch):
        from scribe.runtime_config import set_value

        s = _settings(tmp_path)
        set_value(s, "tts_voice", "bm_george", user="U1")
        _, _, _, handed = self._run_summarize(tmp_path, monkeypatch, settings=s, user="U1")
        assert handed[0].overrides == {"tts_voice": "bm_george", "tts_enabled": True}

    def test_tts_off_means_no_audio_job(self, tmp_path, monkeypatch):
        s = _settings(tmp_path, tts_enabled=False)
        _, _, _, handed = self._run_summarize(tmp_path, monkeypatch, settings=s)
        assert handed == []


class TestAudioWorker:
    def test_runs_under_frozen_overrides_not_current_config(self, tmp_path, monkeypatch):
        from scribe.runtime_config import set_value

        s = _settings(tmp_path)
        set_value(s, "tts_voice", "af_sky")  # changed AFTER hand-off
        seen = {}

        def fake_produce(settings, doc, summary, *, title, author, abort=lambda: None, **kw):
            seen["voice"] = settings.tts_voice
            raise RuntimeError("stop here")

        monkeypatch.setattr(slack_app, "produce_audio", fake_produce)
        j = Job.new("C", "1", "t", "one")
        aj = _audio_job(j)
        enqueue_audio(s, aj)
        slack_app._process_audio(s, _Client(), aj)
        assert seen["voice"] == "bm_george"
        assert restore_audio(s) == [], "completed even on failure; never requeued"

    def test_cancel_while_waiting_in_audio_spool(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        called = []
        monkeypatch.setattr(slack_app, "produce_audio",
                            lambda *a, **k: called.append(1) or (_ for _ in ()).throw(RuntimeError))
        active = _Active()
        j = Job.new("C", "1", "t", "one")
        aj = _audio_job(j)
        enqueue_audio(s, aj)
        active.add(aj)
        assert [x.id for x in active.cancel_thread("1")] == [j.id]
        client = _Client()
        slack_app._process_audio(s, client, aj, active=active)
        assert called == [], "canceled job must not synthesize"
        assert client.posts == []
        assert restore_audio(s) == []

    def test_cancel_mid_synthesis_stops_at_next_segment(self, tmp_path, monkeypatch):
        import scribe.audio as audio

        s = _settings(tmp_path)
        active = _Active()
        j = Job.new("C", "1", "t", "one")
        aj = _audio_job(j)
        active.add(aj)
        synthesized = []

        def fake_synth(settings, text, dest):
            synthesized.append(text)
            dest.write_bytes(b"RIFF")
            active.cancel_thread("1")  # cancel lands while segment 1 is in flight
            return 0.1

        monkeypatch.setattr(audio, "synthesize_segment", fake_synth)
        monkeypatch.setattr(audio, "build_script", lambda *a, **k: [
            audio.ChapterAudio.__mro__ and SimpleNamespace(title="c", segments=["a", "b", "c"])
        ])
        monkeypatch.setattr(audio, "build_m4b", lambda *a, **k: 1.0)
        client = _Client()
        slack_app._process_audio(s, client, aj, active=active)
        assert synthesized == ["a"], "one segment finishes, the rest are skipped"
        assert client.posts == [] and client.uploads == []


class TestOverlap:
    def test_second_note_posts_while_first_audio_still_synthesizing(self, tmp_path, monkeypatch):
        """The point of the ticket: with two workers, document 2's TL;DR lands while
        document 1's m4b is still being made."""
        s = _settings(tmp_path)
        monkeypatch.setattr(slack_app, "extract", lambda *a, **k: _doc())
        monkeypatch.setattr(slack_app, "summarize", lambda *a, **k: _summary())
        monkeypatch.setattr(slack_app, "export_note", _fake_export)
        release = threading.Event()
        started = threading.Event()
        timeline: list[str] = []

        def slow_produce(settings, doc, summary, *, title, author, abort=lambda: None, **kw):
            started.set()
            timeline.append("audio-start")
            release.wait(5)
            timeline.append("audio-end")
            raise RuntimeError("no m4b in tests")

        monkeypatch.setattr(slack_app, "produce_audio", slow_produce)
        client = _Client()
        orig_post = client.chat_postMessage

        def post(**kw):
            if kw.get("text", "").startswith("*d*"):
                timeline.append(f"note:{kw['thread_ts']}")
            orig_post(**kw)

        client.chat_postMessage = post

        pool = ThreadPoolExecutor(max_workers=1)
        audio_pool = ThreadPoolExecutor(max_workers=1)
        pending, audio_pending, active = _Pending(), _Pending(), _Active()

        def audio_submit(aj):
            slack_app._submit_audio(s, audio_pool, audio_pending, active, client, aj)

        j1, j2 = Job.new("C", "1", "https://a", "a"), Job.new("C", "2", "https://b", "b")
        for j in (j1, j2):
            enqueue(s, j)
            slack_app._submit(s, pool, pending, active, client, j, 1.0, audio_submit)

        assert started.wait(5)
        deadline = time.time() + 5
        while "note:2" not in timeline and time.time() < deadline:
            time.sleep(0.02)
        assert "note:2" in timeline and "audio-end" not in timeline, timeline
        release.set()
        pool.shutdown(wait=True)
        audio_pool.shutdown(wait=True)
        assert timeline.index("note:2") < timeline.index("audio-end")
