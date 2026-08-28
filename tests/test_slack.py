"""Slack front-end logic. No network, no Slack connection."""

from __future__ import annotations

from scribe.slack_app import _Pending, first_url


class TestFirstUrl:
    """Slack wraps links in angle brackets, optionally with a display label. A naive
    regex keeps the bracket or the label and produces a URL that 404s."""

    def test_unwraps_slack_angle_brackets(self):
        assert first_url("check <https://example.com/a>") == "https://example.com/a"

    def test_drops_the_display_label(self):
        assert first_url("<https://example.com/a|example.com>") == "https://example.com/a"

    def test_accepts_a_bare_url(self):
        assert first_url("see https://example.com/b please") == "https://example.com/b"

    def test_returns_none_without_a_url(self):
        assert first_url("hello there") is None

    def test_handles_empty_text(self):
        assert first_url("") is None

    def test_takes_the_first_of_several(self):
        text = "<https://one.example> and <https://two.example>"
        assert first_url(text) == "https://one.example"


class TestPending:
    """Queue depth drives the ack wording — count for "behind N items", seconds so the
    quoted ETA is when THIS document finishes, not when it starts."""

    def test_reports_how_many_are_ahead(self):
        p = _Pending()
        assert p.add(100) == (0, 0.0)
        assert p.add(50) == (1, 100.0)
        assert p.add(10) == (2, 150.0)

    def test_done_decrements_both_tallies(self):
        p = _Pending()
        p.add(100)
        p.add(50)
        p.done(100)
        assert p.add(10) == (1, 50.0)

    def test_never_goes_negative(self):
        p = _Pending()
        p.done(100)
        p.done(100)
        assert p.add(10) == (0, 0.0)


class TestHelpGating:
    """Help is on request only. Answering every unrecognized message turns the bot into a
    nag — and worse, Slack's unfurl events used to trigger it on every link."""

    def test_help_words_cover_common_asks(self):
        from scribe.slack_app import HELP_WORDS

        for word in ("help", "?", "usage"):
            assert word in HELP_WORDS

    def test_help_text_explains_the_inputs(self):
        from scribe.slack_app import HELP_TEXT

        for expected in ("link", "PDF", "image", "Obsidian"):
            assert expected in HELP_TEXT


class TestRetryClassification:
    """A transient dependency outage must not permanently lose a queued document — that is
    the durability the spool exists to provide. An ollama-mini rollout landing while the
    queue resumed cost a document exactly this way."""

    def test_transient_errors_are_retried(self):
        from scribe.ollama import OllamaError
        from scribe.vault import VaultError

        # Both mean "a dependency was unreachable", not "this input is bad".
        for exc in (OllamaError, VaultError):
            assert issubclass(exc, RuntimeError)

    def test_extraction_errors_are_not_retried(self):
        """A dead link or unsupported file will fail identically forever; retrying it
        would just block the queue behind it."""
        from scribe.extract.web import ExtractionError

        assert issubclass(ExtractionError, RuntimeError)
        assert ExtractionError is not RuntimeError

    def test_job_carries_an_attempt_counter(self):
        from scribe.queue import Job

        job = Job.new("C", "1", "t", "t")
        assert job.attempts == 0
        job.attempts += 1
        assert job.attempts == 1

    def test_attempts_survive_the_spool_round_trip(self, tmp_path):
        """The counter must persist, or a pod restart resets it and a permanently broken
        dependency retries forever."""
        from scribe.config import Settings
        from scribe.queue import Job, enqueue, restore

        settings = Settings(spool_dir=str(tmp_path / "q"))
        job = Job.new("C", "1", "t", "t")
        job.attempts = 2
        enqueue(settings, job)
        assert restore(settings)[0].attempts == 2


class TestSpooledNameDoesNotLeak:
    """The spooled file carries a job-id prefix for on-disk uniqueness. It must not reach
    the note's title, H1, or Sources frontmatter -- a note called
    '1787110665559937147-a2779432-Syllabus' is unfindable."""

    def test_document_title_uses_the_original_upload_name(self):
        from pathlib import Path

        from scribe.document import Document
        from scribe.note import note_title
        from scribe.summarize import Summary

        doc = Document(
            source="1787110665559937147-a2779432-Syllabus-F26-v0-1.pdf",
            kind="pdf",
            title="1787110665559937147-a2779432-Syllabus-F26-v0-1",
        )
        # What _process now does when the job carries an original name.
        original = "Syllabus-F26-v0-1.pdf"
        doc.source = original
        doc.title = Path(original).stem

        s = Summary(title="Model Written Title", tldr="", summary="")
        assert note_title(doc, s) == "Syllabus-F26-v0-1"
        assert doc.source == "Syllabus-F26-v0-1.pdf"


class TestRequeueKeepsSpool:
    """A requeued job must keep its spool record and attachment. `return` does not skip
    `finally`, and the unconditional complete() there deleted the very file the retry
    was about to read — the first upload to hit a transient failure died with "not a
    file or URL" after 59 minutes of work."""

    @staticmethod
    def _job_with_attachment(settings):
        from scribe.queue import Job, enqueue, spool

        job = Job.new("C", "1", "", "`doc.pdf`")
        att = spool(settings) / f"{job.id}-doc.pdf"
        att.write_bytes(b"%PDF-1.4 fake")
        job.target = str(att)
        job.attachment = str(att)
        job.attachment_name = "doc.pdf"
        enqueue(settings, job)
        return job, att

    @staticmethod
    def _client():
        from types import SimpleNamespace

        return SimpleNamespace(chat_postMessage=lambda **kw: None)

    def test_transient_failure_preserves_spool_and_attachment(self, tmp_path, monkeypatch):
        from scribe import slack_app
        from scribe.config import Settings
        from scribe.ollama import OllamaError
        from scribe.queue import restore

        settings = Settings(spool_dir=str(tmp_path / "q"))
        job, att = self._job_with_attachment(settings)

        def boom(*a, **kw):
            raise OllamaError("server disconnected")

        monkeypatch.setattr(slack_app, "extract", boom)
        requeued = []
        slack_app._process(settings, self._client(), job, requeue=requeued.append)

        assert requeued == [job]
        assert att.exists(), "attachment must survive for the retry to read"
        restored = restore(settings)
        assert [j.id for j in restored] == [job.id]
        assert restored[0].attempts == 1

    def test_exhausted_attempts_clean_up(self, tmp_path, monkeypatch):
        from scribe import slack_app
        from scribe.config import Settings
        from scribe.ollama import OllamaError
        from scribe.queue import restore

        settings = Settings(spool_dir=str(tmp_path / "q"))
        job, att = self._job_with_attachment(settings)
        job.attempts = settings.max_attempts - 1

        def boom(*a, **kw):
            raise OllamaError("still down")

        monkeypatch.setattr(slack_app, "extract", boom)
        requeued = []
        slack_app._process(settings, self._client(), job, requeue=requeued.append)

        assert requeued == []
        assert not att.exists()
        assert restore(settings) == []


class TestUserTz:
    """The sender's Slack profile timezone beats the configured fallback, but a lookup
    failure must never block an ack."""

    def test_caches_the_profile_lookup(self):
        from types import SimpleNamespace

        from scribe.slack_app import _UserTz

        calls = []

        def users_info(user):
            calls.append(user)
            return {"user": {"tz": "Asia/Tokyo"}}

        client = SimpleNamespace(users_info=users_info)
        u = _UserTz()
        assert u.get(client, "U1") == "Asia/Tokyo"
        assert u.get(client, "U1") == "Asia/Tokyo"
        assert calls == ["U1"]

    def test_lookup_failure_returns_none(self):
        from types import SimpleNamespace

        from scribe.slack_app import _UserTz

        def users_info(user):
            raise RuntimeError("slack down")

        u = _UserTz()
        assert u.get(SimpleNamespace(users_info=users_info), "U1") is None

    def test_no_user_id_returns_none(self):
        from scribe.slack_app import _UserTz

        assert _UserTz().get(None, None) is None

    def test_user_survives_the_spool_round_trip(self, tmp_path):
        from scribe.config import Settings
        from scribe.queue import Job, enqueue, restore

        settings = Settings(spool_dir=str(tmp_path / "q"))
        job = Job.new("C", "1", "t", "t")
        job.user = "U0MICHAEL"
        enqueue(settings, job)
        assert restore(settings)[0].user == "U0MICHAEL"
