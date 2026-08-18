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
    """Queue depth drives the ack wording, so the user knows they are waiting behind
    other work rather than assuming the bot stalled."""

    def test_reports_how_many_are_ahead(self):
        p = _Pending()
        assert p.add() == 0
        assert p.add() == 1
        assert p.add() == 2

    def test_done_decrements(self):
        p = _Pending()
        p.add()
        p.add()
        p.done()
        assert p.add() == 1

    def test_never_goes_negative(self):
        p = _Pending()
        p.done()
        p.done()
        assert p.add() == 0
