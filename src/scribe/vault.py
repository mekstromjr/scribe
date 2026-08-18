"""Publish a rendered note to the Obsidian vault via the GitLab API.

The vault (`mekadmin/mekvault`) syncs to Obsidian over iCloud, with Obsidian Git as
version control. The cluster cannot write to iCloud, so scribe commits to GitLab and
Obsidian Git pulls. Round-trip verified 2026-08-17.

Everything for one document lands in a SINGLE commit via the commits API rather than
one file-API call per file: an attachment that committed without its note (or vice
versa) would leave the vault with a dangling wikilink.
"""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import quote

import httpx

from scribe.config import Settings


class VaultError(RuntimeError):
    pass


def _api(settings: Settings, path: str) -> str:
    return f"{settings.gitlab_url.rstrip('/')}/api/v4/projects/{settings.vault_project_id}/{path}"


def _headers(settings: Settings) -> dict[str, str]:
    if not settings.gitlab_token:
        raise VaultError(
            "no GitLab token — set SCRIBE_GITLAB_TOKEN (needs write_repository on the vault)"
        )
    return {"PRIVATE-TOKEN": settings.gitlab_token}


def file_exists(settings: Settings, file_path: str) -> bool:
    url = _api(settings, f"repository/files/{quote(file_path, safe='')}")
    try:
        resp = httpx.get(
            url, headers=_headers(settings), params={"ref": settings.vault_branch}, timeout=30.0
        )
    except httpx.HTTPError as exc:
        raise VaultError(f"could not query {file_path}: {exc}") from exc
    if resp.status_code == 404:
        return False
    if resp.is_success:
        return True
    raise VaultError(f"unexpected {resp.status_code} querying {file_path}: {resp.text[:200]}")


def unique_path(settings: Settings, folder: str, stem: str, suffix: str = ".md") -> str:
    """Find a free path, appending ' (2)', ' (3)'... on collision.

    Never overwrites: a second note about the same source is a new note, and silently
    replacing a note the user may have edited would be data loss.
    """
    base = f"{folder.rstrip('/')}/{stem}"
    if not file_exists(settings, f"{base}{suffix}"):
        return f"{base}{suffix}"
    for n in range(2, 100):
        candidate = f"{base} ({n}){suffix}"
        if not file_exists(settings, candidate):
            return candidate
    raise VaultError(f"could not find a free filename for {stem} after 99 attempts")


def resolve_attachment(settings: Settings, attachment: Path) -> str | None:
    """Pick the vault path for a source file, or None if it is too large to commit.

    Called BEFORE rendering so the note can wikilink the attachment by its final name —
    the path is only known after collision resolution, and a note written afterwards
    would have to guess.
    """
    if attachment.stat().st_size > settings.max_attachment_bytes:
        return None
    return unique_path(settings, settings.vault_files_dir, attachment.stem, attachment.suffix)


def publish(
    settings: Settings,
    *,
    note_body: str,
    note_stem: str,
    attachment: Path | None = None,
    attachment_path: str | None = None,
) -> dict[str, str]:
    """Commit the note (and optionally its source file) in ONE commit.

    Atomic by design: an attachment committed without its note, or a note committed
    without the attachment it links, would leave the vault with a dangling wikilink.
    """
    note_path = unique_path(settings, settings.vault_notes_dir, note_stem)
    actions: list[dict[str, str]] = [
        {"action": "create", "file_path": note_path, "content": note_body}
    ]

    if attachment is not None and attachment_path:
        actions.append(
            {
                "action": "create",
                "file_path": attachment_path,
                "content": base64.b64encode(attachment.read_bytes()).decode(),
                "encoding": "base64",
            }
        )

    payload = {
        "branch": settings.vault_branch,
        "commit_message": f"scribe: {note_stem}",
        "actions": actions,
    }
    try:
        resp = httpx.post(
            _api(settings, "repository/commits"),
            headers=_headers(settings),
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise VaultError(f"commit to vault failed: {exc}") from exc

    return {"note": note_path, "attachment": attachment_path or ""}
