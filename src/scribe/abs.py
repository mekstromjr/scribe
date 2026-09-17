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


def set_item_metadata(settings: Settings, item_id: str, *, narrator: str | None = None,
                      tags: list[str] | None = None, description: str | None = None,
                      clear_series: bool = False) -> None:
    """Patch narrator / tags / description on one item (scribe#12).

    Authoritative over whatever the scanner read from the file. Raises ABSError on
    failure; callers decide how loud to be. ``clear_series`` empties the series list
    (the first cut of #12 used series per person; collections replaced that).
    """
    metadata: dict = {}
    if narrator:
        metadata["narrators"] = [narrator]
    if clear_series:
        metadata["series"] = []
    if description:
        metadata["description"] = description
    body: dict = {}
    if metadata:
        body["metadata"] = metadata
    if tags is not None:
        body["tags"] = tags
    if not body:
        return
    try:
        httpx.patch(
            f"{settings.abs_api_url}/api/items/{item_id}/media",
            headers=_headers(settings), json=body, timeout=60.0,
        ).raise_for_status()
    except httpx.HTTPError as exc:
        raise ABSError(f"metadata update failed for item {item_id}: {exc}") from exc


def add_to_collection(settings: Settings, item_id: str, name: str) -> None:
    """Put an item in the library collection called ``name``, creating it if needed.

    Collections, not series, for "whose files are these" (owner's call, 2026-09-16): a
    series is one ordered work, a collection is a named shelf of unrelated items, and
    it gets its own tab in the library. Idempotent: ABS ignores a duplicate add.
    """
    library_id, _ = _library(settings)
    try:
        resp = httpx.get(f"{settings.abs_api_url}/api/collections", headers=_headers(settings),
                         timeout=30.0)
        resp.raise_for_status()
        existing = [c for c in resp.json().get("collections", [])
                    if c.get("name") == name and c.get("libraryId") == library_id]
        if existing:
            coll_id = existing[0]["id"]
            if any((b.get("id") == item_id) for b in existing[0].get("books", [])):
                return
            httpx.post(f"{settings.abs_api_url}/api/collections/{coll_id}/book",
                       headers=_headers(settings), json={"id": item_id},
                       timeout=30.0).raise_for_status()
        else:
            httpx.post(f"{settings.abs_api_url}/api/collections", headers=_headers(settings),
                       json={"libraryId": library_id, "name": name, "books": [item_id]},
                       timeout=30.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise ABSError(f"could not add item {item_id} to collection {name!r}: {exc}") from exc


def list_items(settings: Settings) -> list[dict]:
    """Every item in the Articles library (id, title, series, narrators, tags)."""
    library_id, _ = _library(settings)
    out: list[dict] = []
    page = 0
    while True:
        try:
            resp = httpx.get(
                f"{settings.abs_api_url}/api/libraries/{library_id}/items",
                headers=_headers(settings), params={"limit": 100, "page": page},
                timeout=60.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ABSError(f"cannot list Audiobookshelf items: {exc}") from exc
        data = resp.json()
        for it in data.get("results", []):
            meta = (it.get("media") or {}).get("metadata") or {}
            out.append({
                "id": it["id"], "title": meta.get("title"),
                "series": meta.get("seriesName") or "",
                "narrator": meta.get("narratorName") or "",
                "tags": (it.get("media") or {}).get("tags") or [],
            })
        if len(out) >= data.get("total", 0) or not data.get("results"):
            return out
        page += 1


def upload(settings: Settings, m4b: Path, *, title: str, author: str,
           narrator: str | None = None, collection: str | None = None,
           tags: list[str] | None = None, description: str | None = None) -> str:
    """Upload one m4b; returns a public web link to the item.

    Once the scan surfaces the item, its narrator / tags / description are patched and
    it is added to ``collection`` (scribe#12); a failure there is logged, never
    raised, because the file is already on the shelf.

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
                try:
                    set_item_metadata(settings, item["id"], narrator=narrator,
                                      tags=tags, description=description)
                    if collection:
                        add_to_collection(settings, item["id"], collection)
                except ABSError as exc:
                    log.warning("%s", exc)
                return f"{settings.abs_web_url}/item/{item['id']}"
    # Still not indexed: the audio IS uploaded, so hand back the shelf rather than
    # failing a job whose work is done.
    log.warning("%r uploaded but not indexed in time — returning the library link", title)
    return library_link
