"""Runtime overrides set from Slack, persisted beside the spool.

Settings has four layers, narrowest first: the sending user's overrides, then the shared
overrides, then environment, then the code default (scribe#2, scribe#6). The file only
ever holds keys explicitly set from Slack, so a value never touched from chat keeps
following its env var — which is what makes the deployment manifest still meaningful
after someone flips something at 11pm.

Per-user came from a real fight over the voice: two people using one bot should each
get their own settings automatically, keyed by the Slack user id that every job already
carries. The file shape is the old flat object plus a `users` map, so a file written
before scribe#6 loads unchanged as the shared layer.

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
ALLOWED_KEYS = {"tts_voice", "tts_enabled", "note_format"}

# Reserved top-level key holding the per-user map. Not an allowed setting name, so it can
# never collide with one.
USERS_KEY = "users"
# Other reserved sections the bot writes for itself (e.g. "calibration", scribe#8).
# Read and written whole via load_section/save_section; never surfaced as settings.
RESERVED_KEYS = {USERS_KEY, "calibration"}


def config_path(settings: Settings) -> Path:
    return Path(settings.spool_dir).expanduser().parent / "config.json"


def _read(settings: Settings) -> dict[str, Any]:
    """The whole file as written, or {} when missing or corrupt. A missing or corrupt
    file is not an error — it means 'no overrides', and refusing to start the bot over a
    bad config file would be a much worse failure than ignoring it."""
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
    return data


def _allowed(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in section.items() if k in ALLOWED_KEYS}


def load(settings: Settings, user: str | None = None) -> dict[str, Any]:
    """Overrides for one layer: the shared layer by default, or one user's own entry.
    Only allowlisted keys come back, whatever the file says."""
    data = _read(settings)
    if user is None:
        return _allowed(data)
    users = data.get(USERS_KEY)
    return _allowed(users.get(user)) if isinstance(users, dict) else {}


def set_value(settings: Settings, key: str, value: Any, user: str | None = None) -> None:
    """Persist one override, atomically — into the shared layer, or into `user`'s entry.

    Write-to-temp-then-rename: the worker reads this file at the start of every job,
    and a partially written file would be read as 'no overrides' — silently reverting
    settings mid-queue.
    """
    if key not in ALLOWED_KEYS:
        raise ValueError(f"{key} is not a runtime-configurable setting")
    path = config_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read(settings)
    if user is None:
        data[key] = value
    else:
        users = data.get(USERS_KEY)
        if not isinstance(users, dict):
            users = data[USERS_KEY] = {}
        users.setdefault(user, {})[key] = value
    _write_atomic(path, data)


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_section(settings: Settings, name: str) -> dict[str, Any]:
    """A reserved, non-settings section of the file (whole), or {}."""
    if name not in RESERVED_KEYS:
        raise ValueError(f"{name} is not a reserved section")
    data = _read(settings).get(name)
    return dict(data) if isinstance(data, dict) else {}


def save_section(settings: Settings, name: str, value: dict[str, Any]) -> None:
    """Replace a reserved section atomically, leaving every other key alone."""
    if name not in RESERVED_KEYS:
        raise ValueError(f"{name} is not a reserved section")
    path = config_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read(settings)
    data[name] = value
    _write_atomic(path, data)


def effective(settings: Settings, user: str | None = None) -> Settings:
    """Settings with the runtime overrides applied: shared layer first, then `user`'s
    own entry on top when given.

    Returns a COPY: the caller's Settings stays the pristine env/default view, and a
    job that started under the old values is not mutated halfway through by someone
    typing a command.
    """
    overrides = load(settings)
    if user:
        overrides.update(load(settings, user))
    if not overrides:
        return settings
    return settings.model_copy(update=overrides)
