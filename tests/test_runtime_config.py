"""Runtime overrides set from Slack (scribe#2). No network, no Slack."""

from __future__ import annotations

import json

import pytest

from scribe.config import Settings
from scribe.runtime_config import ALLOWED_KEYS, config_path, effective, load, set_value


@pytest.fixture
def settings(tmp_path):
    # config.json lives beside the spool dir, both on the PVC in production.
    return Settings(spool_dir=str(tmp_path / "queue"), tts_voice="am_michael")


class TestLayering:
    def test_no_file_means_env_and_defaults_win(self, settings):
        assert effective(settings).tts_voice == "am_michael"
        assert effective(settings).tts_enabled is True

    def test_override_beats_env(self, settings):
        set_value(settings, "tts_voice", "af_bella")
        assert effective(settings).tts_voice == "af_bella"

    def test_untouched_keys_keep_following_env(self, settings):
        # The whole point of storing only what was set: a manifest change to an
        # unrelated setting still takes effect after someone flips a toggle.
        set_value(settings, "tts_enabled", False)
        assert effective(settings).tts_voice == "am_michael"

    def test_effective_does_not_mutate_the_caller_settings(self, settings):
        set_value(settings, "tts_voice", "af_bella")
        effective(settings)
        assert settings.tts_voice == "am_michael"


class TestPersistence:
    def test_survives_a_restart(self, settings, tmp_path):
        set_value(settings, "tts_voice", "bm_george")
        fresh = Settings(spool_dir=str(tmp_path / "queue"), tts_voice="am_michael")
        assert effective(fresh).tts_voice == "bm_george"

    def test_writes_are_atomic_leaving_no_partials(self, settings):
        set_value(settings, "tts_voice", "af_heart")
        set_value(settings, "tts_enabled", False)
        stray = list(config_path(settings).parent.glob(".config-*"))
        assert stray == []
        assert json.loads(config_path(settings).read_text()) == {
            "tts_voice": "af_heart", "tts_enabled": False,
        }


class TestSafety:
    def test_unknown_keys_are_refused(self, settings):
        with pytest.raises(ValueError, match="not a runtime-configurable"):
            set_value(settings, "abs_token", "sneaky")

    def test_corrupt_file_is_ignored_rather_than_fatal(self, settings):
        config_path(settings).parent.mkdir(parents=True, exist_ok=True)
        config_path(settings).write_text("{not json")
        assert load(settings) == {}
        assert effective(settings).tts_voice == "am_michael"

    def test_unknown_keys_already_in_the_file_are_dropped_on_read(self, settings):
        config_path(settings).parent.mkdir(parents=True, exist_ok=True)
        config_path(settings).write_text(json.dumps({"tts_voice": "af_sky", "evil": 1}))
        assert load(settings) == {"tts_voice": "af_sky"}

    def test_allowlist_is_exactly_the_three_toggles(self):
        assert sorted(ALLOWED_KEYS) == ["note_format", "tts_enabled", "tts_voice"]


class TestPerUser:
    """scribe#6: two people on one bot each get their own settings automatically."""

    def test_user_layer_beats_shared_beats_env(self, settings):
        set_value(settings, "tts_voice", "af_bella")            # shared
        set_value(settings, "tts_voice", "bm_george", user="U1")
        assert effective(settings, "U1").tts_voice == "bm_george"
        assert effective(settings, "U2").tts_voice == "af_bella"
        assert effective(settings).tts_voice == "af_bella"

    def test_user_without_overrides_follows_shared(self, settings):
        set_value(settings, "note_format", "md")
        assert effective(settings, "U9").note_format == "md"

    def test_users_do_not_see_each_other(self, settings):
        set_value(settings, "tts_voice", "bm_george", user="U1")
        set_value(settings, "note_format", "docx", user="U2")
        assert effective(settings, "U1").note_format == "pdf"
        assert effective(settings, "U2").tts_voice == "am_michael"

    def test_legacy_flat_file_is_the_shared_layer(self, settings):
        config_path(settings).parent.mkdir(parents=True, exist_ok=True)
        config_path(settings).write_text(json.dumps({"tts_voice": "af_sky"}))
        assert effective(settings, "U1").tts_voice == "af_sky"
        # And a write keeps the legacy key while adding the users map.
        set_value(settings, "tts_enabled", False, user="U1")
        on_disk = json.loads(config_path(settings).read_text())
        assert on_disk == {"tts_voice": "af_sky", "users": {"U1": {"tts_enabled": False}}}

    def test_unknown_keys_in_a_user_entry_are_dropped(self, settings):
        config_path(settings).parent.mkdir(parents=True, exist_ok=True)
        config_path(settings).write_text(
            json.dumps({"users": {"U1": {"evil": 1, "tts_voice": "x"}}})
        )
        assert load(settings, "U1") == {"tts_voice": "x"}

    def test_users_key_is_never_a_setting(self, settings):
        with pytest.raises(ValueError):
            set_value(settings, "users", {})

    def test_unknown_user_key_is_refused(self, settings):
        with pytest.raises(ValueError):
            set_value(settings, "abs_token", "x", user="U1")
