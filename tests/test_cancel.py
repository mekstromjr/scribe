"""Cancel is a thread reply: one user-visible message at cancel time, silent teardown at
the next pipeline checkpoint, and a durable spool cleanup so a restart cannot resurrect
a job the user already canceled."""

from __future__ import annotations

from types import SimpleNamespace

from scribe import slack_app
from scribe.config import Settings
from scribe.queue import Job, enqueue, restore
from scribe.slack_app import _Active, is_cancel


class TestIsCancel:
    def test_plain_words(self):
        for w in ("cancel", "Stop", "  nvm ", "abort!", "Nevermind."):
            assert is_cancel(w)

    def test_channel_mention_is_stripped(self):
        assert is_cancel("<@U0123ABC> cancel")

    def test_content_is_not_cancel(self):
        for t in ("https://example.com", "cancel my subscription essay", "", None):
            assert not is_cancel(t)


class TestActive:
    def test_cancel_thread_marks_only_that_thread(self):
        a = _Active()
        j1 = Job.new("C", "111", "t", "one")
        j2 = Job.new("C", "222", "t", "two")
        a.add(j1)
        a.add(j2)
        assert [j.id for j in a.cancel_thread("111")] == [j1.id]
        assert a.canceled(j1.id) and not a.canceled(j2.id)

    def test_requeue_refcount_survives_first_done_callback(self):
        a = _Active()
        j = Job.new("C", "111", "t", "one")
        a.add(j)   # first run
        a.add(j)   # requeued under the same id before the first done-callback
        a.remove(j)  # first run's callback
        assert [x.id for x in a.cancel_thread("111")] == [j.id], (
            "the retry must still be trackable"
        )

    def test_remove_clears_cancellation_state(self):
        a = _Active()
        j = Job.new("C", "111", "t", "one")
        a.add(j)
        a.cancel_thread("111")
        a.remove(j)
        assert not a.canceled(j.id)
        assert a.cancel_thread("111") == []


class TestProcessCancellation:
    @staticmethod
    def _client(sink):
        return SimpleNamespace(chat_postMessage=lambda **kw: sink.append(kw))

    def test_canceled_before_start_skips_silently(self, tmp_path, monkeypatch):
        settings = Settings(spool_dir=str(tmp_path / "q"))
        job = Job.new("C", "1", "https://example.com", "x")
        enqueue(settings, job)
        active = _Active()
        active.add(job)
        active.cancel_thread("1")

        def must_not_run(*a, **kw):
            raise AssertionError("extract must not run for a canceled job")

        monkeypatch.setattr(slack_app, "extract", must_not_run)
        replies: list = []
        slack_app._process(settings, self._client(replies), job, active=active)

        assert replies == [], "teardown is silent — the cancel command already replied"
        assert restore(settings) == []

    def test_cancel_mid_summarize_stops_at_checkpoint(self, tmp_path, monkeypatch):
        settings = Settings(
            spool_dir=str(tmp_path / "q"),
            context_tokens=2000, response_reserve_tokens=500, chunk_chars=1000,
        )
        job = Job.new("C", "1", "doc", "x")
        enqueue(settings, job)
        active = _Active()
        active.add(job)

        from scribe.document import Document, Method, Page
        doc = Document(source="d", kind="link", title="d",
                       pages=[Page(number=1, text="Head\n\n" + "word " * 4000,
                                   method=Method.TEXT_LAYER)])
        monkeypatch.setattr(slack_app, "extract", lambda *a, **kw: doc)

        calls = {"n": 0}

        def fake_chat(settings_, prompt, schema, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # Cancellation lands while the first chunk is processing.
                active.cancel_thread("1")
            return {"points": ["p"]}, 0.1

        import scribe.summarize as sz
        monkeypatch.setattr(sz, "chat_structured", fake_chat)

        published = []
        monkeypatch.setattr(slack_app, "publish",
                            lambda *a, **kw: published.append(1) or {"note": "n.md"})
        replies: list = []
        slack_app._process(settings, self._client(replies), job, active=active)

        assert calls["n"] <= 2, "must stop at the next chunk checkpoint, not finish"
        assert published == [], "a canceled job must never publish"
        assert replies == []
        assert restore(settings) == []
