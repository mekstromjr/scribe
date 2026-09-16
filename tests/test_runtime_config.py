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
