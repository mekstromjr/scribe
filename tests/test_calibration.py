"""scribe#8: the estimator learns a per-stage correction from every clean completion,
quotes the upper side of its spread, and survives restarts."""

from __future__ import annotations

import json
import math
import time

from scribe import slack_app
from scribe.calibration import FACTOR_MAX, FACTOR_MIN, Calibration, StageStats
from scribe.config import Settings
from scribe.eta import Estimate, estimate
from scribe.runtime_config import config_path, set_value
from scribe.slack_app import _ack_text, _Pending


def _settings(tmp_path, **kw):
    return Settings(slack_bot_token="x", slack_app_token="x", spool_dir=str(tmp_path / "q"), **kw)


class TestLearning:
    def test_fresh_calibration_is_the_prior(self, tmp_path):
        cal = Calibration.load(_settings(tmp_path))
        assert cal.quote("summarize_single", 100) == 100
        assert cal.expected("audio", 60) == 60

    def test_first_observation_moves_fully_then_settles(self, tmp_path):
        s = _settings(tmp_path)
        cal = Calibration()
        cal.observe(s, "summarize_single", predicted=100, actual=150)
        assert math.isclose(cal.stats["summarize_single"].factor, 1.5, rel_tol=1e-6)
        # Later observations are smoothed: a single 1.0 ratio does not erase the 1.5.
        for _ in range(3):
            cal.observe(s, "summarize_single", predicted=100, actual=100)
        f = cal.stats["summarize_single"].factor
        assert 1.0 < f < 1.5

    def test_quote_is_above_the_mean_when_residuals_scatter(self, tmp_path):
        s = _settings(tmp_path)
        cal = Calibration()
        for actual in (90, 160, 110, 170, 100, 150):
            cal.observe(s, "audio", predicted=100, actual=actual)
        st = cal.stats["audio"]
        assert st.spread > 0
        assert cal.quote("audio", 100) > cal.expected("audio", 100)

    def test_one_absurd_job_is_clamped(self, tmp_path):
        s = _settings(tmp_path)
        cal = Calibration()
        cal.observe(s, "ocr", predicted=10, actual=10_000)
        assert cal.stats["ocr"].factor <= FACTOR_MAX
        cal2 = Calibration()
        cal2.observe(s, "ocr", predicted=10_000, actual=1)
        assert cal2.stats["ocr"].factor >= FACTOR_MIN

    def test_zero_or_negative_teaches_nothing(self, tmp_path):
        s = _settings(tmp_path)
        cal = Calibration()
        assert cal.observe(s, "audio", 0, 50) is None
        assert cal.observe(s, "audio", 50, 0) is None
        assert cal.observe(s, "nope", 50, 50) is None
        assert cal.stats["audio"].n == 0


class TestPersistence:
    def test_survives_restart_and_coexists_with_settings(self, tmp_path):
        s = _settings(tmp_path)
        set_value(s, "tts_voice", "af_sky", user="U1")
        Calibration().observe(s, "summarize_map", 480, 700)
        again = Calibration.load(s)
        assert again.stats["summarize_map"].n == 1
        on_disk = json.loads(config_path(s).read_text())
        assert on_disk["users"] == {"U1": {"tts_voice": "af_sky"}}, "settings untouched"
        assert "calibration" in on_disk
        assert "calibration" not in vars(Settings)  # never a setting

    def test_corrupt_section_is_ignored(self, tmp_path):
        s = _settings(tmp_path)
        config_path(s).parent.mkdir(parents=True, exist_ok=True)
        config_path(s).write_text(json.dumps({"calibration": {"audio": "garbage", "x": 1}}))
        cal = Calibration.load(s)
        assert cal.stats["audio"] == StageStats()


class TestPendingRemainingTime:
    def test_inflight_counts_at_remaining_not_full(self):
        p = _Pending()
        p.add(100)
        p.start(100)
        time.sleep(0.05)
        n, secs = p.peek()
        assert n == 1 and secs < 100
        p.done(100)
        assert p.peek() == (0, 0.0)

    def test_queued_jobs_count_in_full(self):
        p = _Pending()
        p.add(100)
        p.start(100)
        p.add(50)
        n, secs = p.add(10)
        assert n == 2 and 149 < secs <= 150


class TestAckText:
    def _est(self, ocr=0.0, summ=100.0, audio=200.0, branch="summarize_single"):
        return Estimate(kind="pdf", chars=1000, ocr_pages=int(bool(ocr)), branch=branch,
                        ocr_seconds=ocr, summarize_seconds=summ, audio_seconds=audio)

    def test_two_times_when_audio_is_on(self, tmp_path):
        s = _settings(tmp_path)
        text, sq, aq = _ack_text(s, self._est(), Calibration(), 0, 0, tz=None, audio_on=True)
        assert text.startswith("Summary by ") and ", audio by " in text
        assert sq == 100 and aq == 200

    def test_one_time_when_audio_is_off(self, tmp_path):
        s = _settings(tmp_path)
        text, _, aq = _ack_text(s, self._est(), Calibration(), 0, 0, tz=None, audio_on=False)
        assert "audio" not in text and aq == 0

    def test_learned_factor_scales_the_quote(self, tmp_path):
        s = _settings(tmp_path)
        cal = Calibration()
        cal.observe(s, "summarize_single", 100, 150)
        _, sq, _ = _ack_text(s, self._est(), cal, 0, 0, tz=None, audio_on=False)
        assert math.isclose(sq, 150, rel_tol=1e-6)


class TestEstimateShape:
    def test_link_uses_budget_and_no_ocr(self, tmp_path):
        e = estimate(_settings(tmp_path), "https://example.com/a")
        assert e.kind == "link" and e.ocr_pages == 0 and e.ocr_seconds == 0
        assert e.summarize_seconds > 0 and e.audio_seconds > 0

    def test_unreadable_file_falls_back_without_raising(self, tmp_path):
        e = estimate(_settings(tmp_path), str(tmp_path / "missing.pdf"))
        assert e.kind == "unknown" and e.summary_raw > 0


class TestProcessLearns:
    def _fakes(self, monkeypatch, tmp_path):
        from scribe.document import Document, Method, Page
        from scribe.summarize import Summary

        def fake_extract(s, target, cache=None):
            return Document(source="d", kind="link", title="d",
                            pages=[Page(number=1, text="hi there", method=Method.TEXT_LAYER)])

        def fake_summarize(s, doc, abort=None):
            time.sleep(0.02)
            return Summary(title="t", tldr="x", summary="y")

        def fake_export(body, fmt, *, stem, out_dir):
            p = out_dir / f"{stem}.{fmt}"
            p.write_text("x")
            return p

        monkeypatch.setattr(slack_app, "extract", fake_extract)
        monkeypatch.setattr(slack_app, "summarize", fake_summarize)
        monkeypatch.setattr(slack_app, "export_note", fake_export)

        class Client:
            posts: list[str] = []

            def chat_postMessage(self, **kw):
                Client.posts.append(kw.get("text", ""))

            def files_upload_v2(self, **kw):
                pass

        Client.posts.clear()
        return Client()

    def test_clean_completion_updates_summarize_stage(self, tmp_path, monkeypatch):
        from scribe.queue import Job

        s = _settings(tmp_path, tts_enabled=False)
        client = self._fakes(monkeypatch, tmp_path)
        job = Job.new("C", "1", "https://x", "x")
        slack_app._process(s, client, job)
        assert Calibration.load(s).stats["summarize_single"].n == 1

    def test_retry_attempt_does_not_teach(self, tmp_path, monkeypatch):
        from scribe.queue import Job

        s = _settings(tmp_path, tts_enabled=False)
        client = self._fakes(monkeypatch, tmp_path)
        job = Job.new("C", "1", "https://x", "x")
        job.attempts = 1
        slack_app._process(s, client, job)
        assert Calibration.load(s).stats["summarize_single"].n == 0

    def test_tldr_quotes_audio_and_handoff_carries_prediction(self, tmp_path, monkeypatch):
        from scribe.queue import Job

        s = _settings(tmp_path, abs_token="t")
        client = self._fakes(monkeypatch, tmp_path)
        handed = []
        job = Job.new("C", "1", "https://x", "x")
        slack_app._process(s, client, job, audio_submit=handed.append)
        assert any("Audio should land by" in p for p in client.posts)
        assert handed and handed[0].script_chars > 0 and handed[0].predicted_raw > 0
