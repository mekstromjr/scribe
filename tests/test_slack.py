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

        for expected in ("link", "PDF", "image", "scribeformat"):
            assert expected in HELP_TEXT


class TestRetryClassification:
    """A transient dependency outage must not permanently lose a queued document — that is
    the durability the spool exists to provide. An ollama-mini rollout landing while the
    queue resumed cost a document exactly this way."""

    def test_transient_errors_are_retried(self):
        from scribe.ollama import OllamaError

        # "A dependency was unreachable", not "this input is bad".
        assert issubclass(OllamaError, RuntimeError)

    def test_export_errors_are_not_transient(self):
        """A note export failure happens after the TL;DR is posted; it must degrade to a
        missing file, never requeue a job whose summary already exists."""
        from scribe.note_export import ExportError
        from scribe.ollama import OllamaError

        assert not issubclass(ExportError, OllamaError)

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

        from scribe.document import Document
        from scribe.note import note_title
        from scribe.summarize import Summary

        doc = Document(
            source="1787110665559937147-a2779432-Syllabus-F26-v0-1.pdf",
            kind="pdf",
            title=None,  # no PDF metadata title
        )
        # What _process now does when the job carries an original name.
        original = "Syllabus-F26-v0-1.pdf"
        doc.source = original

        # With a model title, that wins (scribe#9); the spooled prefix is nowhere.
        s = Summary(title="Model Written Title", tldr="", summary="")
        assert note_title(doc, s) == "Model Written Title"
        # Without one, the fallback is the ORIGINAL stem, never the spooled name.
        assert note_title(doc, Summary(title="", tldr="", summary="")) == "Syllabus-F26-v0-1"
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


class TestScribeFormatCommand:
    """`/scribeformat` replaced the vault on/off toggle (scribe#5): the choice is which
    file the reader gets, not whether the owner's vault fills up."""

    def _app(self, tmp_path, monkeypatch):
        from scribe import slack_app
        from scribe.config import Settings

        settings = Settings(slack_bot_token="xoxb-x", slack_app_token="xapp-x",
                            spool_dir=str(tmp_path / "q"))
        handlers: dict = {}

        class FakeApp:
            def command(self, name):
                def deco(fn):
                    handlers[name] = fn
                    return fn
                return deco

        slack_app._register_config_commands(FakeApp(), settings)
        return settings, handlers

    def _call(self, handler, text):
        out = []
        handler(ack=lambda: None, respond=out.append, command={"text": text})
        return out[-1]

    def test_bare_invocation_reports_current(self, tmp_path, monkeypatch):
        _, h = self._app(tmp_path, monkeypatch)
        assert "*pdf*" in self._call(h["/scribeformat"], "")

    def test_sets_and_persists(self, tmp_path, monkeypatch):
        from scribe.runtime_config import effective

        settings, h = self._app(tmp_path, monkeypatch)
        reply = self._call(h["/scribeformat"], "MD")
        assert "*md*" in reply
        assert effective(settings).note_format == "md"

    def test_rejects_unknown_format(self, tmp_path, monkeypatch):
        from scribe.runtime_config import effective

        settings, h = self._app(tmp_path, monkeypatch)
        reply = self._call(h["/scribeformat"], "html")
        assert "not a note format" in reply
        assert effective(settings).note_format == "pdf"

    def test_none_plus_tts_off_warns(self, tmp_path, monkeypatch):
        settings, h = self._app(tmp_path, monkeypatch)
        self._call(h["/scribetoggletts"], "off")
        assert "Both outputs are off" in self._call(h["/scribeformat"], "none")

    def test_old_vault_toggle_is_gone(self, tmp_path, monkeypatch):
        _, h = self._app(tmp_path, monkeypatch)
        assert "/scribetoggleobs" not in h


class TestPerUserCommands:
    """Commands write the invoking user's layer (scribe#6); `default` writes the shared one."""

    def _handlers(self, tmp_path):
        from scribe import slack_app
        from scribe.config import Settings

        settings = Settings(slack_bot_token="xoxb-x", slack_app_token="xapp-x",
                            spool_dir=str(tmp_path / "q"))
        handlers: dict = {}

        class FakeApp:
            def command(self, name):
                def deco(fn):
                    handlers[name] = fn
                    return fn
                return deco

        slack_app._register_config_commands(FakeApp(), settings)
        return settings, handlers

    def _call(self, handler, text, user="U1"):
        out = []
        handler(ack=lambda: None, respond=out.append, command={"text": text, "user_id": user})
        return out[-1]

    def test_format_is_per_user(self, tmp_path):
        from scribe.runtime_config import effective

        settings, h = self._handlers(tmp_path)
        assert "for you" in self._call(h["/scribeformat"], "md", user="U1")
        assert effective(settings, "U1").note_format == "md"
        assert effective(settings, "U2").note_format == "pdf"
        assert effective(settings).note_format == "pdf"

    def test_default_word_writes_the_shared_layer(self, tmp_path):
        from scribe.runtime_config import effective

        settings, h = self._handlers(tmp_path)
        reply = self._call(h["/scribeformat"], "docx default", user="U1")
        assert "for everyone by default" in reply
        assert effective(settings, "U2").note_format == "docx"
        assert effective(settings).note_format == "docx"

    def test_toggle_is_per_user(self, tmp_path):
        from scribe.runtime_config import effective

        settings, h = self._handlers(tmp_path)
        self._call(h["/scribetoggletts"], "off", user="U1")
        assert effective(settings, "U1").tts_enabled is False
        assert effective(settings, "U2").tts_enabled is True

    def test_voice_is_per_user_and_warns_when_diverging(self, tmp_path, monkeypatch):
        from scribe import slack_app
        from scribe.runtime_config import effective

        monkeypatch.setattr(slack_app, "voices", lambda s: ["af_bella", "bm_george"])
        settings, h = self._handlers(tmp_path)
        reply = self._call(h["/scribevoice"], "bm_george", user="U1")
        assert "for you" in reply and "stays loaded" in reply
        assert effective(settings, "U1").tts_voice == "bm_george"
        assert effective(settings, "U2").tts_voice == "af_bella"
        # Setting the shared default carries no divergence warning.
        assert "stays loaded" not in self._call(h["/scribevoice"], "bm_george default")

    def test_config_shows_own_and_shared(self, tmp_path):
        _, h = self._handlers(tmp_path)
        assert "shared defaults" in self._call(h["/scribeconfig"], "").lower()
        self._call(h["/scribeformat"], "none", user="U1")
        reply = self._call(h["/scribeconfig"], "", user="U1")
        assert "Your settings" in reply and "note *none*" in reply
        assert "Shared defaults" in reply and "note *pdf*" in reply
        assert "overridden: note_format" in reply

    def test_process_resolves_settings_for_the_sender(self, tmp_path, monkeypatch):
        """The whole point: a job runs under its SENDER's layer, not whoever typed last."""
        from scribe import slack_app
        from scribe.config import Settings
        from scribe.queue import Job
        from scribe.runtime_config import set_value

        settings = Settings(slack_bot_token="xoxb-x", slack_app_token="xapp-x",
                            spool_dir=str(tmp_path / "q"), tts_enabled=False)
        set_value(settings, "note_format", "md", user="U1")
        seen = {}

        def fake_extract(s, target, cache=None):
            from scribe.document import Document, Method, Page
            return Document(source="d", kind="link", title="d",
                            pages=[Page(number=1, text="hi", method=Method.TEXT_LAYER)])

        def fake_summarize(s, doc, abort=None):
            from scribe.summarize import Summary
            return Summary(title="t", tldr="x", summary="y")

        def fake_export(body, fmt, *, stem, out_dir):
            seen["fmt"] = fmt
            p = out_dir / f"{stem}.{fmt}"
            p.write_text("x")
            return p

        monkeypatch.setattr(slack_app, "extract", fake_extract)
        monkeypatch.setattr(slack_app, "summarize", fake_summarize)
        monkeypatch.setattr(slack_app, "export_note", fake_export)

        class Client:
            def chat_postMessage(self, **kw): pass
            def files_upload_v2(self, **kw): pass

        job = Job.new("C", "1", "https://x", "x")
        job.user = "U1"
        slack_app._process(settings, Client(), job)
        assert seen["fmt"] == "md"
        job2 = Job.new("C", "2", "https://x", "x")
        job2.user = "U2"
        slack_app._process(settings, Client(), job2)
        assert seen["fmt"] == "pdf"
