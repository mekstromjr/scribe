"""Audiobookshelf client. No network — httpx is stubbed at module level."""

from __future__ import annotations

import pytest

from scribe import abs as abs_mod
from scribe.abs import ABSError, _library, upload
from scribe.config import Settings

LIBS = {
    "libraries": [
        {"id": "lib-audio", "name": "Audiobooks", "folders": [{"id": "f-1"}]},
        {"id": "lib-art", "name": "Articles", "folders": [{"id": "f-2"}]},
    ]
}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def settings():
    return Settings(abs_token="fake-token")


class TestLibraryResolution:
    def test_finds_the_articles_library_by_name(self, settings, monkeypatch):
        monkeypatch.setattr(abs_mod.httpx, "get", lambda *a, **k: _Resp(LIBS))
        assert _library(settings) == ("lib-art", "f-2")

    def test_missing_library_says_how_to_fix_it(self, settings, monkeypatch):
        monkeypatch.setattr(
            abs_mod.httpx, "get", lambda *a, **k: _Resp({"libraries": []})
        )
        with pytest.raises(ABSError, match="create it"):
            _library(settings)

    def test_no_token_fails_before_any_request(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("should not have made a request")

        monkeypatch.setattr(abs_mod.httpx, "get", boom)
        with pytest.raises(ABSError, match="SCRIBE_ABS_TOKEN"):
            _library(Settings(abs_token=""))


class TestUpload:
    def _wire(self, monkeypatch, items):
        posts = []
        monkeypatch.setattr(abs_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            abs_mod.httpx, "post", lambda *a, **k: posts.append(k) or _Resp({})
        )

        def get(url, **k):
            if url.endswith("/api/libraries"):
                return _Resp(LIBS)
            return _Resp({"results": items})

        monkeypatch.setattr(abs_mod.httpx, "get", get)
        return posts

    def test_returns_item_deep_link_when_the_scan_finds_it(
        self, settings, monkeypatch, tmp_path
    ):
        items = [{"id": "item-9", "media": {"metadata": {"title": "My Article"}}}]
        posts = self._wire(monkeypatch, items)
        m4b = tmp_path / "a.m4b"
        m4b.write_bytes(b"x")
        link = upload(settings, m4b, title="My Article", author="scribe")
        assert link == f"{settings.abs_web_url}/item/item-9"
        # The upload targeted the Articles library and folder, resolved by name.
        assert posts[0]["data"]["library"] == "lib-art"
        assert posts[0]["data"]["folder"] == "f-2"

    def test_falls_back_to_the_library_link_rather_than_erroring(
        self, settings, monkeypatch, tmp_path
    ):
        # Audio already delivered: a slow scanner must not turn success into failure.
        self._wire(monkeypatch, [])
        m4b = tmp_path / "a.m4b"
        m4b.write_bytes(b"x")
        link = upload(settings, m4b, title="Unscanned", author="scribe")
        assert link == f"{settings.abs_web_url}/library/lib-art"
