"""The collapse loop's job is coverage: a reduce input that overflows must be condensed
through the model, not trimmed — trimming silently thinned summaries toward the end of
long documents (measured: 8,992 chars cut on a 72-page PDF)."""

from __future__ import annotations

from scribe import summarize as sz
from scribe.config import Settings
from scribe.document import Document, Method, Page

# Tiny budgets so the tests drive the overflow paths with kilobytes, not megabytes:
# budget_chars = (2000 - 500) * 4 = 6000.
SETTINGS = Settings(context_tokens=2000, response_reserve_tokens=500, chunk_chars=1000)

REDUCE_KEYS = {"title": "T", "tldr": "tl", "summary": "s", "tags": ["x"]}


def _doc(chars: int) -> Document:
    return Document(
        source="big.pdf", kind="pdf", title="big",
        pages=[Page(number=1, text="Heading\n\n" + ("point of fact. " * (chars // 15)),
                    method=Method.TEXT_LAYER)],
    )


def test_overflowing_points_are_collapsed_not_trimmed(monkeypatch):
    calls = {"map": 0, "collapse": 0, "reduce_input": None}

    def fake_chat(settings, prompt, schema, **kw):
        if "ONE SECTION" in prompt:
            calls["map"] += 1
            # Verbose enough that the joined points overflow the 6000-char budget.
            return {"points": [f"map point {calls['map']} " + "detail " * 40]}, 0.1
        if "CONSECUTIVE SECTIONS" in prompt:
            calls["collapse"] += 1
            return {"points": [f"condensed {calls['collapse']}"]}, 0.1
        calls["reduce_input"] = prompt
        return dict(REDUCE_KEYS), 0.1

    monkeypatch.setattr(sz, "chat_structured", fake_chat)
    result = sz._map_reduce(SETTINGS, _doc(30000), budget_chars=6000)

    assert calls["collapse"] > 0, "overflow must trigger a collapse round"
    assert result.truncated_chars == 0, "nothing may be trimmed when collapse converges"
    assert len(calls["reduce_input"]) <= 6000 + len(sz.REDUCE_PROMPT)


def test_nonconverging_collapse_falls_back_to_trim(monkeypatch):
    def fake_chat(settings, prompt, schema, **kw):
        if "ONE SECTION" in prompt:
            return {"points": ["verbose " * 60]}, 0.1
        if "CONSECUTIVE SECTIONS" in prompt:
            # Pathological: condensing makes it BIGGER. The loop must bail, not spin.
            return {"points": ["even more verbose " * 80]}, 0.1
        return dict(REDUCE_KEYS), 0.1

    monkeypatch.setattr(sz, "chat_structured", fake_chat)
    result = sz._map_reduce(SETTINGS, _doc(30000), budget_chars=6000)

    assert result.truncated_chars > 0, "the safety-net trim must still report the loss"


def test_fitting_points_skip_collapse_entirely(monkeypatch):
    calls = {"collapse": 0}

    def fake_chat(settings, prompt, schema, **kw):
        if "ONE SECTION" in prompt:
            return {"points": ["terse"]}, 0.1
        if "CONSECUTIVE SECTIONS" in prompt:
            calls["collapse"] += 1
            return {"points": ["x"]}, 0.1
        return dict(REDUCE_KEYS), 0.1

    monkeypatch.setattr(sz, "chat_structured", fake_chat)
    result = sz._map_reduce(SETTINGS, _doc(30000), budget_chars=6000)

    assert calls["collapse"] == 0
    assert result.truncated_chars == 0
