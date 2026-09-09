"""A durable spool for pending Slack jobs.

Without this the queue lives only in the executor's memory: a restart with four items
pending drops them silently, leaving four "On it" acks that never resolve. For a
read-later queue that is the worst failure mode available — the user believes the work is
coming and it simply is not.

Each job is one JSON file, plus the downloaded attachment beside it when there is one.
Attachments live here rather than in a tempdir precisely so they survive a restart.

Ordering is FIFO by filename: ids are `<epoch_ns>-<random>`, so lexicographic sort is
chronological and restore replays in the order the user sent things.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from scribe.config import Settings
from scribe.document import Page


@dataclass
class Job:
    id: str
    channel: str
    thread_ts: str
    target: str
    source_label: str
    attachment: str | None = None
    # The ORIGINAL upload name. The spooled file is prefixed with the job id to keep
    # concurrent uploads from colliding on disk, but the vault must get the clean name --
    # otherwise notes link attachments called "1787099069943412312-1cc31a4e-Syllabus.pdf".
    attachment_name: str | None = None
    # Slack user id of the sender, for rendering ETAs in THEIR profile timezone on the
    # restart-resume path. Optional so records spooled before this field restore fine.
    user: str | None = None
    # Retry counter. A transient dependency outage (ollama restarting, a network blip)
    # must not permanently lose a queued document -- that is the exact durability the
    # spool exists to provide.
    attempts: int = 0

    @staticmethod
    def new(channel: str, thread_ts: str, target: str, source_label: str) -> Job:
        # NANOSECOND prefix, not milliseconds. The suffix is random, so it breaks ties
        # arbitrarily rather than chronologically — at millisecond resolution several
        # links pasted in quick succession share a prefix and can replay out of order.
        # Nanosecond resolution makes a collision effectively impossible.
        jid = f"{time.time_ns():019d}-{uuid.uuid4().hex[:8]}"
        return Job(id=jid, channel=channel, thread_ts=thread_ts, target=target,
                   source_label=source_label)


def spool(settings: Settings) -> Path:
    d = Path(settings.spool_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_file(settings: Settings, job_id: str) -> Path:
    return spool(settings) / f"{job_id}.json"


def enqueue(settings: Settings, job: Job) -> None:
    """Persist a job. Written to a temp file then renamed so a crash mid-write cannot
    leave a half-written record that fails to parse on restore."""
    path = _job_file(settings, job.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(job)))
    os.replace(tmp, path)


def complete(settings: Settings, job: Job) -> None:
    """Remove the job, its attachment and its OCR page cache. Safe to call twice."""
    _job_file(settings, job.id).unlink(missing_ok=True)
    _pages_file(settings, job.id).unlink(missing_ok=True)
    if job.attachment:
        Path(job.attachment).unlink(missing_ok=True)


# NOT ".json": restore() globs "*.json" for job records and deletes anything that will not
# parse as a Job, so a cache file with that suffix would be wiped on every restart.
_PAGES_SUFFIX = ".pages"


def _pages_file(settings: Settings, job_id: str) -> Path:
    return spool(settings) / f"{job_id}{_PAGES_SUFFIX}"


class PageCache:
    """Finished OCR pages for one job, persisted beside its spool record.

    OCR is the expensive, deterministic part of a job -- minutes per page at temperature 0
    -- and until scribe#4 a requeue redid all of it, so a document whose last page hung
    re-OCR'd every good page on every attempt and never got further. With the cache, a
    retry resumes at the first page it has not finished. SKIPPED pages are cached too: they
    are exactly the ones a retry must not attempt again.

    Whole-file atomic rewrite per page, same temp-then-rename pattern as enqueue(). Pages
    are few and small, so rewriting beats a partial-write hazard.
    """

    def __init__(self, path: Path):
        self.path = path
        self._pages: dict[int, Page] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                self._pages = {int(k): Page.model_validate(v) for k, v in raw.items()}
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                # A corrupt cache costs a re-OCR, never a failed job.
                self._pages = {}

    def get(self, number: int) -> Page | None:
        return self._pages.get(number)

    def put(self, page: Page) -> None:
        self._pages[page.number] = page
        tmp = self.path.with_suffix(_PAGES_SUFFIX + ".tmp")
        tmp.write_text(json.dumps({str(k): v.model_dump() for k, v in self._pages.items()}))
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._pages)


def page_cache(settings: Settings, job_id: str) -> PageCache:
    return PageCache(_pages_file(settings, job_id))


def restore(settings: Settings) -> list[Job]:
    """Return unfinished jobs in arrival order.

    A record that will not parse is discarded rather than raising: one corrupt file must
    not stop every other queued item from resuming.
    """
    jobs: list[Job] = []
    for path in sorted(spool(settings).glob("*.json")):
        try:
            jobs.append(Job(**json.loads(path.read_text())))
        except (json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
    return jobs
