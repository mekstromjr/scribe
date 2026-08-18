"""Vault publishing logic. No network — HTTP is stubbed."""

from __future__ import annotations

import pytest

from scribe import vault
from scribe.config import Settings
from scribe.vault import VaultError, resolve_attachment, unique_path


@pytest.fixture
def settings():
    return Settings(gitlab_token="fake-token")


class TestUniquePath:
    """Never overwrite: a second note about the same source is a NEW note, and silently
    replacing one the user may have edited would be data loss."""

    def test_uses_plain_name_when_free(self, settings, monkeypatch):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: False)
        assert unique_path(settings, "+", "My Note") == "+/My Note.md"

    def test_appends_counter_on_collision(self, settings, monkeypatch):
        taken = {"+/My Note.md", "+/My Note (2).md"}
        monkeypatch.setattr(vault, "file_exists", lambda s, p: p in taken)
        assert unique_path(settings, "+", "My Note") == "+/My Note (3).md"

    def test_respects_custom_suffix(self, settings, monkeypatch):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: False)
        assert unique_path(settings, "Misc/Files", "scan", ".pdf") == "Misc/Files/scan.pdf"

    def test_strips_trailing_slash_from_folder(self, settings, monkeypatch):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: False)
        assert unique_path(settings, "+/", "N") == "+/N.md"

    def test_gives_up_rather_than_looping_forever(self, settings, monkeypatch):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: True)
        with pytest.raises(VaultError, match="could not find a free filename"):
            unique_path(settings, "+", "N")


class TestResolveAttachment:
    def test_returns_path_for_small_file(self, settings, monkeypatch, tmp_path):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: False)
        f = tmp_path / "deck.pdf"
        f.write_bytes(b"x" * 1024)
        assert resolve_attachment(settings, f) == "Misc/Files/deck.pdf"

    def test_returns_none_over_size_cap(self, settings, monkeypatch, tmp_path):
        monkeypatch.setattr(vault, "file_exists", lambda s, p: False)
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"x" * 2048)
        settings.max_attachment_bytes = 1024
        # The vault repo syncs to every device, so a large binary is a lasting cost.
        assert resolve_attachment(settings, f) is None


class TestTokenGuard:
    def test_missing_token_is_a_clear_error(self):
        with pytest.raises(VaultError, match="SCRIBE_GITLAB_TOKEN"):
            vault._headers(Settings(gitlab_token=""))
