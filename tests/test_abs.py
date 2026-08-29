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
    def _wire(self, monkeypatch, items, scan_raises=False):
        posts = []
        monkeypatch.setattr(abs_mod.time, "sleep", lambda s: None)

        def post(url, **k):
            if url.endswith("/scan"):
                if scan_raises:
                    raise abs_mod.httpx.ConnectError("scan unavailable")
                posts.append({**k, "_url": url})
                return _Resp({})
            posts.append({**k, "_url": url})
            return _Resp({})

        monkeypatch.setattr(abs_mod.httpx, "post", post)

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

    def test_triggers_a_scan_before_polling(self, settings, monkeypatch, tmp_path):
        # ABS's own post-upload scan races the FUSE flush for large files (a 114 MB
        # m4b sat unindexed for 30+ min on 2026-08-28), so we must ask explicitly.
        items = [{"id": "item-9", "media": {"metadata": {"title": "My Article"}}}]
        posts = self._wire(monkeypatch, items)
        m4b = tmp_path / "a.m4b"
        m4b.write_bytes(b"x")
        upload(settings, m4b, title="My Article", author="scribe")
        urls = [k.get("_url") for k in posts]
        assert any(u and u.endswith("/api/libraries/lib-art/scan") for u in urls), urls

    def test_scan_failure_does_not_lose_the_upload(self, settings, monkeypatch, tmp_path):
        # The audio is already on disk; a scan hiccup must not fail the job.
        items = [{"id": "item-9", "media": {"metadata": {"title": "My Article"}}}]
        self._wire(monkeypatch, items, scan_raises=True)
        m4b = tmp_path / "a.m4b"
        m4b.write_bytes(b"x")
        assert upload(settings, m4b, title="My Article", author="scribe").endswith("/item/item-9")

    def test_falls_back_to_the_library_link_rather_than_erroring(
        self, settings, monkeypatch, tmp_path
    ):
        # Audio already delivered: a slow scanner must not turn success into failure.
        self._wire(monkeypatch, [])
        m4b = tmp_path / "a.m4b"
        m4b.write_bytes(b"x")
        link = upload(settings, m4b, title="Unscanned", author="scribe")
        assert link == f"{settings.abs_web_url}/library/lib-art"
