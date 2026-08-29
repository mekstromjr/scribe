"""Upload finished audio to Audiobookshelf's Articles library.

API calls go to the in-cluster Service; the link handed to notes and Slack uses the
public hostname, because the whole point of the link is opening on a phone.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from scribe.config import Settings

log = logging.getLogger("scribe.abs")


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

    # Ask for a scan explicitly rather than waiting for ABS to notice on its own.
    #
    # The library folder is an rclone FUSE mount (garage:media/articles), and FUSE
    # does not deliver inotify events, so ABS's file watcher never fires for what we
    # upload. ABS does scan after its own /api/upload, but that scan RACES the FUSE
    # flush to Garage: measured 2026-08-28, a 114 MB m4b uploaded 200 OK and then sat
    # unindexed for 30+ minutes, while 54 MB and 72 MB files on the same path indexed
    # fine. A manual scan surfaced it in 6s. Big files are exactly the ones worth
    # listening to, so this is not an edge case.
    try:
        httpx.post(
            f"{settings.abs_api_url}/api/libraries/{library_id}/scan",
            headers=_headers(settings),
            timeout=60.0,
        ).raise_for_status()
    except httpx.HTTPError as exc:
        # Non-fatal: the file is uploaded either way, and a later scan will find it.
        log.warning("could not trigger an Audiobookshelf scan: %s", exc)

    # Poll ~2.5 min, not 30s: the scan has to probe a large file THROUGH the FUSE
    # mount, which is far slower than a local disk read.
    for _ in range(30):
        time.sleep(5.0)
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
    # Still not indexed: the audio IS uploaded, so hand back the shelf rather than
    # failing a job whose work is done.
    log.warning("%r uploaded but not indexed in time — returning the library link", title)
    return library_link
