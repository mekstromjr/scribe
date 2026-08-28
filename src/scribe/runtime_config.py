"""Runtime overrides set from Slack, persisted beside the spool.

Settings has three layers now, narrowest first: this file, then environment, then the
code default. The file only ever holds keys explicitly set from Slack, so a value
never touched from chat keeps following its env var — which is what makes the
deployment manifest still meaningful after someone flips something at 11pm.

On the spool PVC rather than a ConfigMap: it is the one writable, durable path the pod
already has, and a config the bot writes itself has no business round-tripping through
git.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from scribe.config import Settings

log = logging.getLogger("scribe.runtime_config")

# Only these may be set from chat. An allowlist, not an open key/value store: a typo'd
# key would otherwise sit in the file looking authoritative and doing nothing.
ALLOWED_KEYS = {"tts_voice", "tts_enabled", "vault_enabled"}


def config_path(settings: Settings) -> Path:
    return Path(settings.spool_dir).expanduser().parent / "config.json"


def load(settings: Settings) -> dict[str, Any]:
    """Read overrides. A missing or corrupt file is not an error — it means 'no
    overrides', and refusing to start the bot over a bad config file would be a much
    worse failure than ignoring it."""
    path = config_path(settings)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable runtime config at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("ignoring runtime config at %s: not a JSON object", path)
        return {}
    return {k: v for k, v in data.items() if k in ALLOWED_KEYS}


def set_value(settings: Settings, key: str, value: Any) -> None:
    """Persist one override, atomically.

    Write-to-temp-then-rename: the worker reads this file at the start of every job,
    and a partially written file would be read as 'no overrides' — silently reverting
    settings mid-queue.
    """
    if key not in ALLOWED_KEYS:
        raise ValueError(f"{key} is not a runtime-configurable setting")
    path = config_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load(settings)
    data[key] = value
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def effective(settings: Settings) -> Settings:
    """Settings with the runtime overrides applied.

    Returns a COPY: the caller's Settings stays the pristine env/default view, and a
    job that started under the old values is not mutated halfway through by someone
    typing a command.
    """
    overrides = load(settings)
    if not overrides:
        return settings
    return settings.model_copy(update=overrides)
