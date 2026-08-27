"""Upload finished audio to Audiobookshelf's Articles library.

API calls go to the in-cluster Service; the link handed to notes and Slack uses the
public hostname, because the whole point of the link is opening on a phone.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from scribe.config import Settings


class ABSError(RuntimeError):
    pass


def _headers(settings: Settings) -> dict[str, str]:
    if not settings.abs_token:
        raise ABSError("no Audiobookshelf token — set SCRIBE_ABS_TOKEN")
    return {"Authorization": f"Bearer {settings.abs_token}"}


def _library(settings: Settings) -> tuple[str, str]:
    """Resolve (library_id, folder_id) by name at call time, not config time.

    Looked up rather than configured: ids are opaque and survive nowhere outside the
    ABS database, so pinning them in config would break silently if the library were
    ever recreated. One extra GET per upload is noise next to a 20-minute synthesis.
    """
    try:
        resp = httpx.get(
            f"{settings.abs_api_url}/api/libraries", headers=_headers(settings), timeout=30.0
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ABSError(f"cannot list Audiobookshelf libraries: {exc}") from exc
    for lib in resp.json().get("libraries", []):
        if lib.get("name") == settings.abs_library_name:
            folders = lib.get("folders") or []
            if not folders:
                raise ABSError(f"library {settings.abs_library_name!r} has no folders")
            return lib["id"], folders[0]["id"]
    raise ABSError(
        f"no library named {settings.abs_library_name!r} in Audiobookshelf — create it "
        f"(media type Books) pointing at the /articles mount"
    )


def upload(settings: Settings, m4b: Path, *, title: str, author: str) -> str:
    """Upload one m4b; returns a public web link to the item.

    The upload response carries no item id, so the item is found by polling the
    library's newest additions. If it has not been scanned in time the LIBRARY link is
    returned instead — a working link to the right shelf beats an error after the
    audio was already delivered.
    """
    library_id, folder_id = _library(settings)
    try:
        with m4b.open("rb") as fh:
            resp = httpx.post(
                f"{settings.abs_api_url}/api/upload",
                headers=_headers(settings),
                # ABS reads these two ids plus title/author for the folder name it
                # creates; everything else the scanner takes from the m4b tags.
                data={
                    "title": title,
                    "author": author,
                    "library": library_id,
                    "folder": folder_id,
                },
                files={"0": (m4b.name, fh, "audio/mp4")},
                timeout=600.0,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ABSError(f"upload to Audiobookshelf failed: {exc}") from exc

    library_link = f"{settings.abs_web_url}/library/{library_id}"
    for _ in range(10):
        time.sleep(3.0)
        try:
            resp = httpx.get(
                f"{settings.abs_api_url}/api/libraries/{library_id}/items",
                headers=_headers(settings),
                params={"limit": 10, "sort": "addedAt", "desc": 1},
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            continue
        for item in resp.json().get("results", []):
            meta = (item.get("media") or {}).get("metadata") or {}
            if meta.get("title") == title:
                return f"{settings.abs_web_url}/item/{item['id']}"
    return library_link
