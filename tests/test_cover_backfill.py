"""`scribe cover-backfill` (scribe#14): covers for shelf items made before covers existed.

ABS, ollama and the image server are all stubbed.
"""

from __future__ import annotations

import argparse

import pytest

from scribe import abs as abs_mod
from scribe import cli, cover

ITEMS = {
    "a": {"title": "Has one", "cover_path": "/metadata/items/a/cover.jpg"},
    "b": {"title": "01-recursion", "cover_path": ""},
    "c": {"title": "Plato_Meno_selections", "cover_path": ""},
}


@pytest.fixture
def shelf(monkeypatch):
    monkeypatch.setenv("SCRIBE_COVER_HOST", "http://sd.invalid")
    monkeypatch.setenv("SCRIBE_ABS_TOKEN", "t")
    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(abs_mod, "list_items",
                        lambda s: [{"id": k, "title": v["title"]} for k, v in ITEMS.items()])
    monkeypatch.setattr(abs_mod, "item_info", lambda s, i: {
        "title": ITEMS[i]["title"], "description": "", "chapters": ["Summary", "Topic"],
        "cover_path": ITEMS[i]["cover_path"]})
    monkeypatch.setattr(abs_mod, "set_cover", lambda s, i, jpeg: uploaded.__setitem__(i, jpeg))
    monkeypatch.setattr(cover, "plan", lambda s, summary: cover.CoverPlan("woodcut", "a scroll"))
    monkeypatch.setattr(cover, "make_cover", lambda s, **k: b"\xff\xd8" + k["title"].encode())
    return uploaded


def _run(**kw) -> int:
    ns = argparse.Namespace(style=None, title=None, force=False, dry_run=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return cli._cmd_cover_backfill(ns)


def test_fills_only_the_gaps(shelf):
    assert _run() == 0
    assert sorted(shelf) == ["b", "c"], "the item with a cover is skipped"


def test_force_redoes_everything(shelf):
    _run(force=True)
    assert sorted(shelf) == ["a", "b", "c"]


def test_dry_run_draws_and_uploads_nothing(shelf, monkeypatch, capsys):
    monkeypatch.setattr(cover, "make_cover", lambda *a, **k: pytest.fail("drew in a dry run"))
    assert _run(dry_run=True) == 0
    assert shelf == {}
    assert "woodcut: a scroll, " in capsys.readouterr().out


def test_one_failure_does_not_stop_the_rest(shelf, monkeypatch):
    monkeypatch.setattr(cover, "make_cover",
                        lambda s, **k: None if k["title"] == "01-recursion" else b"jpeg")
    assert _run() == 1, "nonzero exit when anything failed"
    assert sorted(shelf) == ["c"]


def test_refuses_without_an_image_server(shelf, monkeypatch):
    monkeypatch.setenv("SCRIBE_COVER_HOST", "")
    assert _run() == 2 and shelf == {}
