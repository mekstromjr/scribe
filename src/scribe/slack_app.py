"""Slack Socket Mode front end.

Socket Mode (an OUTBOUND WebSocket) rather than the Events API, because Slack cannot
reach this network: the cluster sits behind Tailscale CGNAT with no public ingress. This
also means scribe needs no Service, no Ingress, and no certificate — nothing connects to
it.

Work is deliberately serialized behind a single worker. ollama-mini runs
OLLAMA_NUM_PARALLEL=1 on CPU, so concurrent documents would not finish sooner — they
would thrash a shared bottleneck and make every request slower. Queueing is honest.
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from scribe.abs import ABSError, upload
from scribe.audio import produce_audio
from scribe.calibration import Calibration
from scribe.config import Settings, load_settings
from scribe.document import Document, Method
from scribe.eta import Estimate, audio_seconds, clock_at, estimate
from scribe.extract import ExtractionError, extract
from scribe.listening import build_script
from scribe.note import note_title, render, slugify
from scribe.note_export import FORMATS, ExportError, export_note
from scribe.ollama import OllamaError
from scribe.queue import (
    AudioJob,
    Job,
    complete,
    complete_audio,
    enqueue,
    enqueue_audio,
    page_cache,
    restore,
    restore_audio,
    spool,
)
from scribe.runtime_config import effective, load, load_section, set_value
from scribe.summarize import Summary, summarize
from scribe.tts import TTSError, describe_voice, grouped_voices, voices

log = logging.getLogger("scribe.slack")

# Slack wraps links as <https://x> or <https://x|label>; take the URL, drop the label.
_URL = re.compile(r"<(https?://[^|>\s]+)(?:\|[^>]*)?>")
_BARE_URL = re.compile(r"https?://\S+")

HELP_WORDS = {"help", "?", "usage", "how does scribe work?", "how does this work?"}
HELP_TEXT = (
    "Send me a *link*, *PDF*, or *image* — as a DM here, or @-mention me in a channel.\n\n"
    "I extract the text, write a thorough summary, and reply in thread with the TL;DR "
    "plus the full note as a file (PDF by default; `/scribeformat` picks pdf, md, docx, "
    "or none).\n\n"
    "Documents are processed one at a time and a long article takes several minutes — "
    "I'll tell you where you are in the queue.\n\n"
    "Sent something by mistake? Reply *cancel* in its thread and I'll stop."
)


CANCEL_WORDS = {"cancel", "stop", "abort", "nevermind", "nvm"}


class JobCanceled(Exception):
    """Raised at a pipeline checkpoint when the job's thread said to stop."""


def is_cancel(text: str) -> bool:
    """True if a thread reply is a cancel command. Mentions are stripped first so
    "@scribe cancel" in a channel works the same as "cancel" in a DM."""
    bare = re.sub(r"<@[^>]+>", "", text or "").strip().strip("!.").lower()
    return bare in CANCEL_WORDS


class _Active:
    """Tracks which jobs belong to which thread, and which have been canceled.

    Entries are COUNTED, not merely set: a requeued job is resubmitted under the same id
    before the failed run's done-callback fires, and a plain set would let that callback
    delete the retry's tracking entry.
    """

    def __init__(self) -> None:
        self._by_thread: dict[str, dict[str, Job | AudioJob]] = {}
        self._counts: dict[str, int] = {}
        self._canceled: set[str] = set()
        self._lock = threading.Lock()

    def add(self, job: Job | AudioJob) -> None:
        with self._lock:
            self._by_thread.setdefault(job.thread_ts, {})[job.id] = job
            self._counts[job.id] = self._counts.get(job.id, 0) + 1

    def remove(self, job: Job | AudioJob) -> None:
        with self._lock:
            n = self._counts.get(job.id, 0) - 1
            if n > 0:
                self._counts[job.id] = n
                return
            self._counts.pop(job.id, None)
            self._canceled.discard(job.id)
            thread = self._by_thread.get(job.thread_ts)
            if thread:
                thread.pop(job.id, None)
                if not thread:
                    del self._by_thread[job.thread_ts]

    def cancel_thread(self, thread_ts: str) -> list[Job | AudioJob]:
        """Mark every job in the thread canceled; returns them (may be empty)."""
        with self._lock:
            jobs = list(self._by_thread.get(thread_ts, {}).values())
            self._canceled.update(j.id for j in jobs)
        return jobs

    def canceled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._canceled


class _UserName:
    """Cached Slack display names (users.info real_name, falling back to display
    name), for the shelf's per-person series and tag (scribe#12). A miss returns
    None and the item simply has no series; never blocks a job."""

    TTL_SECONDS = 24 * 3600.0

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.Lock()

    def get(self, client, user_id: str | None) -> str | None:
        if not user_id:
            return None
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(user_id)
            if hit and now - hit[1] < self.TTL_SECONDS:
                return hit[0]
        name: str | None = None
        try:
            u = client.users_info(user=user_id).get("user", {}) or {}
            prof = u.get("profile") or {}
            name = (u.get("real_name") or prof.get("real_name") or prof.get("display_name")
                    or u.get("name") or None)
            # First name only: the shelf shows "Michael", not a full legal name, and
            # family members share a surname anyway.
            if name:
                name = name.split()[0]
        except Exception:
            log.warning("users.info failed for %s; item will have no series", user_id)
        with self._lock:
            self._cache[user_id] = (name, now)
        return name


class _UserTz:
    """Cached Slack profile timezones. Slack keeps a user's tz current as they travel,
    so the profile beats any configured zone — but users.info per message would be
    wasteful, and a lookup failure must never block an ack, so misses return None and
    the caller falls back to settings.timezone."""

    TTL_SECONDS = 3600.0

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.Lock()

    def get(self, client, user_id: str | None) -> str | None:
        if not user_id:
            return None
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(user_id)
            if hit and now - hit[1] < self.TTL_SECONDS:
                return hit[0]
        tz: str | None = None
        try:
            tz = client.users_info(user=user_id).get("user", {}).get("tz") or None
        except Exception:
            log.warning("users.info failed for %s; using fallback timezone", user_id)
        with self._lock:
            self._cache[user_id] = (tz, now)
        return tz


def first_url(text: str) -> str | None:
    m = _URL.search(text or "")
    if m:
        return m.group(1)
    m = _BARE_URL.search(text or "")
    return m.group(0).rstrip(">") if m else None


def download_file(settings: Settings, file_info: dict, dest: Path) -> Path:
    """Fetch a Slack upload.

    The bot token must go in an Authorization header — `url_private_download` returns an
    HTML login page rather than the bytes if the request is unauthenticated, which fails
    later and confusingly rather than here.
    """
    url = file_info.get("url_private_download") or file_info["url_private"]
    name = file_info.get("name") or "upload"
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
        follow_redirects=True,
        timeout=120.0,
    )
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("text/html"):
        raise ExtractionError(
            f"Slack returned HTML for {name} — the bot token is missing or lacks files:read"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def _audio_stage(settings: Settings, client, job, doc, summary,
                 abort=lambda: None, person: str | None = None) -> float | None:
    """Synthesize, upload to Audiobookshelf, and post the m4b in-thread.

    Never raises. Each delivery step degrades independently: an ABS outage still posts
    the file to Slack, and a Slack upload failure still leaves the ABS link.

    `settings` is the job's already-resolved view (shared + sender overrides); it is not
    re-resolved here, which would silently drop the sender's layer.
    """
    title = note_title(doc, summary)
    author = doc.source if doc.kind == "link" else "scribe"
    t0 = time.monotonic()
    try:
        result = produce_audio(settings, doc, summary, title=title, author=author,
                               abort=abort, person=person)
    except JobCanceled:
        raise
    except Exception as exc:
        log.warning("audio synthesis failed for %s: %s", job.source_label, exc)
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"_(No audio this time — synthesis failed: {exc})_",
        )
        return None
    synth_wall = time.monotonic() - t0

    with result.workdir:
        minutes = result.audio_seconds / 60
        abs_line = ""
        try:
            link = upload(
                settings, result.m4b, title=title, author=author,
                narrator=settings.tts_voice, collection=person,
                tags=[person] if person else [], description=doc.source,
            )
            abs_line = f"Listen in <{link}|Audiobookshelf> ({minutes:.0f} min)."
        except ABSError as exc:
            log.warning("ABS upload failed for %s: %s", job.source_label, exc)
            abs_line = f"_(Audiobookshelf upload failed: {exc})_"

        try:
            # files_upload_v2 needs files:write; posted into the same thread so the
            # audio sits next to the TL;DR it belongs to.
            client.files_upload_v2(
                channel=job.channel,
                thread_ts=job.thread_ts,
                file=str(result.m4b),
                filename=f"{title}.m4b",
                title=title,
                initial_comment=abs_line,
            )
        except Exception as exc:
            log.warning("Slack audio upload failed for %s: %s", job.source_label, exc)
            client.chat_postMessage(
                channel=job.channel, thread_ts=job.thread_ts,
                text=abs_line or f"_(Audio ready but both deliveries failed: {exc})_",
            )
    return synth_wall


def _note_stage(settings: Settings, client, job: Job, doc, summary) -> None:
    """Render the note and post it in-thread as a file in the configured format.

    Never raises, and never requeues: by the time this runs the summary exists and the
    TL;DR is already in the thread, so an export failure costs the file, not the job.
    Mirrors _audio_stage on purpose -- both are deliveries of a finished product.
    """
    fmt = settings.note_format
    if fmt == "none":
        return
    title = note_title(doc, summary)
    body = render(doc, summary, model=settings.text_model)
    with tempfile.TemporaryDirectory(prefix="scribe-note-") as tmp:
        try:
            path = export_note(body, fmt, stem=slugify(title), out_dir=Path(tmp))
        except ExportError as exc:
            log.warning("note export (%s) failed for %s: %s", fmt, job.source_label, exc)
            client.chat_postMessage(
                channel=job.channel, thread_ts=job.thread_ts,
                text=f"_(No {fmt} this time — rendering failed: {exc})_",
            )
            return
        try:
            client.files_upload_v2(
                channel=job.channel,
                thread_ts=job.thread_ts,
                file=str(path),
                filename=path.name,
                title=title,
            )
        except Exception as exc:
            log.warning("Slack note upload failed for %s: %s", job.source_label, exc)
            client.chat_postMessage(
                channel=job.channel, thread_ts=job.thread_ts,
                text=f"_(The {fmt} was ready but the upload to Slack failed: {exc})_",
            )


def _process(settings: Settings, client, job: Job, requeue=lambda _job: None,
             active: _Active | None = None, audio_submit=lambda _aj: None,
             audio_ahead=lambda: 0.0, tz: str | None = None,
             person: str | None = None) -> None:
    """Run the summarize half of the pipeline and reply in-thread. Never raises —
    failures are reported to Slack.

    Ends by HANDING OFF the audio half (scribe#7): the AudioJob is spooled, then
    ``audio_submit`` schedules it on the audio worker. This worker is free for the next
    document the moment the note is posted, while Kokoro grinds through the last one.
    """
    # Runtime overrides are read ONCE, here, at the start of the job: a toggle typed
    # while this document is mid-flight applies to the next one, so a job's behavior
    # never changes underneath the ack the user already received. Resolved for the
    # SENDER (scribe#6): each person gets their own voice and format automatically.
    settings = effective(settings, job.user)
    requeued = False
    # The raw per-stage predictions this job was quoted from (a PDF scan is
    # milliseconds), so each stage's actual can be learned against them (scribe#8).
    pred = estimate(settings, job.target)
    cal = Calibration.load(settings)

    def abort() -> None:
        # The one user-visible cancel message was already posted by the cancel command;
        # everything after it tears down silently.
        if active and active.canceled(job.id):
            raise JobCanceled()

    try:
        abort()
        # The page cache lives in the spool under the job id, so a requeued attempt
        # resumes OCR at the first unfinished page instead of redoing them all (scribe#4).
        t0 = time.monotonic()
        doc = extract(settings, job.target, cache=page_cache(settings, job.id))
        extract_wall = time.monotonic() - t0
        # OCR is the only extraction cost worth learning; a text-layer read is ms. Only
        # a run that OCR'd every page it meant to (no cap, no skips, no cache resume)
        # is a clean sample.
        ocr_pages = [pg for pg in doc.pages if pg.method is Method.OCR]
        if pred.ocr_pages and ocr_pages and job.attempts == 0 and not any(
            pg.method is Method.SKIPPED for pg in doc.pages
        ):
            log.info("eta.actual job=%s stage=ocr predicted=%.0f actual=%.0f pages=%d",
                     job.id, pred.ocr_seconds, extract_wall, len(ocr_pages))
            cal.observe(settings, "ocr", pred.ocr_seconds, extract_wall)
        if job.attachment_name:
            # The extractors derive `source` from the file PATH, which for an upload is
            # the SPOOLED name carrying a job-id prefix (kept so concurrent uploads cannot
            # collide on disk). Left alone that prefix reaches the Sources frontmatter and,
            # via the filename fallback, the title. `title` is NOT set from the name: the
            # ladder in note_title prefers PDF metadata, then the model (scribe#9).
            doc.source = job.attachment_name
        t0 = time.monotonic()
        summary = summarize(settings, doc, abort=abort)
        summarize_wall = time.monotonic() - t0
        if job.attempts == 0:
            log.info("eta.actual job=%s stage=%s predicted=%.0f actual=%.0f chars=%d",
                     job.id, pred.branch, pred.summarize_seconds, summarize_wall,
                     len(doc.text))
            cal.observe(settings, pred.branch, pred.summarize_seconds, summarize_wall)

        # Exact audio prediction now that the listening script can be built (rule-based,
        # cheap): quoted in the TL;DR reply and learned against by the audio worker.
        audio_on = settings.tts_enabled and bool(settings.abs_token)
        script_chars = 0
        audio_raw = 0.0
        if audio_on:
            try:
                script_chars = sum(len(seg) for ch in build_script(
                    doc, summary, max_chars=settings.tts_max_chars) for seg in ch.segments)
            except Exception:  # never let a script problem block the note
                script_chars = len(doc.text) + len(summary.summary)
            audio_raw = audio_seconds(settings, script_chars)

        # Past here the TL;DR is posted and cancel would confuse more than it saves.
        abort()
        lines = [
            f"*{note_title(doc, summary)}*",
            # The source is repeated in the body, not just implied by the thread: Slack
            # surfaces reply text in notifications, search and the Threads pane, where the
            # parent message is not visible. With several items queued, a model-written
            # title alone does not reliably identify which one this is.
            job.source_label,
            "",
            summary.tldr,
            "",
        ]
        if settings.note_format == "none":
            lines.append("_Note delivery is off (`/scribeformat`) — this summary lives only here._")
        if audio_on and not (active and active.canceled(job.id)):
            wait = cal.quote("audio", audio_raw) + audio_ahead()
            lines.append(f"_Audio should land by {clock_at(settings, wait, tz)}._")
        if summary.sections > 1:
            lines.append(
                f"_Long document — summarized in {summary.sections} sections, so this "
                f"covers the whole thing but in less detail than usual._"
            )
        if summary.truncated_chars:
            lines.append(
                f"_Note: {summary.truncated_chars} characters were trimmed to fit the "
                f"model's context, so the summary covers only part of the source._"
            )
        client.chat_postMessage(
            channel=job.channel, thread_ts=job.thread_ts, text="\n".join(lines)
        )
        # The full note follows the TL;DR as a file. Best-effort: the summary is the
        # product and is already in the thread.
        _note_stage(settings, client, job, doc, summary)
        # Audio is a separate, spooled job on its own worker. Spooled BEFORE this record
        # completes, so a crash between the two cannot lose the promised audio.
        if settings.tts_enabled and settings.abs_token:
            if active and active.canceled(job.id):
                # Canceled after the note posted: keep the note, skip only the audio.
                log.info("skipping audio for canceled job %s", job.id)
            else:
                audio_job = AudioJob(
                    id=job.id, channel=job.channel, thread_ts=job.thread_ts,
                    source_label=job.source_label, user=job.user,
                    doc=doc.model_dump(mode="json"), summary=summary.model_dump(mode="json"),
                    # The sender's resolved TTS settings, frozen at hand-off.
                    overrides={"tts_voice": settings.tts_voice,
                               "tts_enabled": settings.tts_enabled},
                    script_chars=script_chars, predicted_raw=audio_raw,
                    person=person,
                )
                enqueue_audio(settings, audio_job)
                audio_submit(audio_job)
    except JobCanceled:
        log.info("job %s canceled by its thread", job.id)
    except ExtractionError as exc:
        # Bad input -- a dead link, an unsupported file. Retrying will not help.
        log.warning("extraction failed for %s: %s", job.source_label, exc)
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"Sorry — {job.source_label} failed: {exc}",
        )
    except OllamaError as exc:
        # Transient: the model server was unreachable. Retry rather than drop
        # the job. This is what lost a document when an ollama-mini rollout happened to
        # land while the queue was resuming.
        if job.attempts + 1 < settings.max_attempts:
            job.attempts += 1
            enqueue(settings, job)
            log.warning(
                "attempt %d/%d failed for %s (%s) — requeueing",
                job.attempts, settings.max_attempts, job.source_label, exc,
            )
            requeue(job)
            requeued = True
            return
        log.error("giving up on %s after %d attempts: %s", job.source_label,
                  job.attempts + 1, exc)
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"Sorry — {job.source_label} failed after "
                 f"{job.attempts + 1} attempts: {exc}",
        )
    except Exception:
        # A crash here must not kill the worker thread and silently stop the queue.
        log.exception("unexpected failure processing %s", job.target)
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"Sorry — {job.source_label} failed unexpectedly.",
        )
    finally:
        # Clears both the spool record and the downloaded attachment. A job that failed is
        # still done: retrying it forever would block everything behind it.
        #
        # EXCEPT a requeued job: `return` does not skip `finally`, and completing here
        # deletes the very attachment the retry is about to read. That is how the first
        # upload to hit a transient ollama failure died with "not a file or URL" after
        # 59 minutes of work (2026-08-27) — the retry machinery had only ever been
        # exercised by URL jobs, which have no attachment to lose.
        if not requeued:
            complete(settings, job)


def _process_audio(settings: Settings, client, job: AudioJob,
                   active: _Active | None = None) -> None:
    """Run the audio half from its spool record. Never raises, never requeues: the note
    is already delivered, so anything that goes wrong here costs the m4b, not the job."""
    # NOT effective(): the settings the sender had at hand-off are frozen in the record.
    settings = settings.model_copy(update=job.overrides)

    def abort() -> None:
        if active and active.canceled(job.id):
            raise JobCanceled()

    try:
        abort()
        doc = Document.model_validate(job.doc)
        summary = Summary.model_validate(job.summary)
        synth_wall = _audio_stage(settings, client, job, doc, summary, abort=abort,
                                  person=job.person)
        if synth_wall and job.predicted_raw > 0:
            log.info("eta.actual job=%s stage=audio predicted=%.0f actual=%.0f script_chars=%d",
                     job.id, job.predicted_raw, synth_wall, job.script_chars)
            Calibration.load(settings).observe(settings, "audio", job.predicted_raw, synth_wall)
    except JobCanceled:
        log.info("audio for %s canceled by its thread", job.id)
    except Exception:
        log.exception("unexpected failure in audio for %s", job.source_label)
    finally:
        complete_audio(settings, job)


class _Pending:
    """Counts queued jobs — and their estimated seconds — so the ack can quote when THIS
    document will be done, not when it will merely start.

    Tracked explicitly rather than reading ThreadPoolExecutor._work_queue, which is a
    private attribute with no stability guarantee. The in-flight job counts at its
    REMAINING time (estimate minus elapsed, floored at zero), not its full estimate:
    quoting a half-finished job at full price was one reason acks ran late (scribe#8).
    """

    def __init__(self) -> None:
        self._n = 0
        self._queued_seconds = 0.0
        self._inflight: tuple[float, float] | None = None  # (est, started_at)
        self._lock = threading.Lock()

    def _ahead_locked(self) -> tuple[int, float]:
        remaining = 0.0
        if self._inflight:
            est, started = self._inflight
            remaining = max(0.0, est - (time.monotonic() - started))
        return self._n, self._queued_seconds + remaining

    def peek(self) -> tuple[int, float]:
        """(jobs ahead, seconds ahead) without adding anything."""
        with self._lock:
            return self._ahead_locked()

    def add(self, est_seconds: float) -> tuple[int, float]:
        """Returns (jobs ahead, estimated seconds ahead) as of just before this add."""
        with self._lock:
            ahead = self._ahead_locked()
            self._n += 1
            self._queued_seconds += est_seconds
        return ahead

    def start(self, est_seconds: float) -> None:
        """A queued job began running: it moves from the queued sum to in-flight."""
        with self._lock:
            self._queued_seconds = max(0.0, self._queued_seconds - est_seconds)
            self._inflight = (est_seconds, time.monotonic())

    def done(self, est_seconds: float) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)
            if self._inflight is not None:
                self._inflight = None
            else:
                self._queued_seconds = max(0.0, self._queued_seconds - est_seconds)


def _submit(settings: Settings, pool: ThreadPoolExecutor, pending: _Pending,
            active: _Active, client, job: Job, est_seconds: float,
            audio_submit=lambda _aj: None, audio_ahead=lambda: 0.0,
            tz: str | None = None, person: str | None = None) -> None:
    def requeue(j: Job) -> None:
        # Back of the queue, not the front: a document whose dependency is down should not
        # block everything behind it while it retries.
        pending.add(est_seconds)
        _submit(settings, pool, pending, active, client, j, est_seconds, audio_submit,
                audio_ahead, tz, person)

    def run() -> None:
        pending.start(est_seconds)
        _process(settings, client, job, requeue, active, audio_submit, audio_ahead, tz,
                 person)

    active.add(job)
    fut = pool.submit(run)

    def _done(_f) -> None:
        pending.done(est_seconds)
        active.remove(job)

    fut.add_done_callback(_done)


def _submit_audio(settings: Settings, pool: ThreadPoolExecutor, pending: _Pending,
                  active: _Active, client, job: AudioJob) -> None:
    """Schedule an audio job on the audio worker. Registered under the same id as its
    summarize half, so a thread cancel finds it whether it is waiting or synthesizing."""
    est = Calibration.load(settings).expected("audio", job.predicted_raw)
    pending.add(est)
    active.add(job)

    def run() -> None:
        pending.start(est)
        _process_audio(settings, client, job, active)

    fut = pool.submit(run)

    def _done(_f) -> None:
        pending.done(est)
        active.remove(job)

    fut.add_done_callback(_done)


def _ack_text(settings: Settings, est: Estimate, cal: Calibration, ahead_seconds: float,
              audio_ahead: float, *, tz: str | None, audio_on: bool) -> tuple[str, float, float]:
    """The two-line quote for the ack: summary time, then audio time (scribe#8).

    Each stage is scaled by its learned factor at the quoting (upper) side. Audio starts
    when the summary is done OR when the audio queue drains, whichever is later, then
    takes its own quoted time. Returns (text, quoted_summary_seconds, quoted_audio_seconds).
    """
    summary_q = cal.quote("ocr", est.ocr_seconds) + cal.quote(est.branch, est.summarize_seconds)
    summary_at = ahead_seconds + summary_q
    text = f"Summary by {clock_at(settings, summary_at, tz)}"
    audio_q = 0.0
    if audio_on:
        audio_q = cal.quote("audio", est.audio_seconds)
        audio_at = max(summary_at, audio_ahead) + audio_q
        text += f", audio by {clock_at(settings, audio_at, tz)}"
    return text + ".", summary_q, audio_q


def _voice_menu(settings: Settings, available: list[str], current: str) -> str:
    """The no-argument /scribevoice reply: a link to the samples on the shelf, then the
    voices grouped by language and gender (scribe#11)."""
    lines = [f"Your voice: *{current}*  ({describe_voice(current)})"]
    link = load_section(settings, "voice_samples").get("url")
    if link:
        lines.append(f"Hear every voice read the same passage: <{link}|Scribe voice samples> "
                     f"(one chapter per voice).")
    lines.append("Pick one with `/scribevoice <id>`. `v0` ids are older versions of the "
                 "same voice.")
    for group, ids in grouped_voices(available):
        marked = [f"`{v}`{' ←' if v == current else ''}" for v in ids]
        lines.append(f"*{group}:* " + ", ".join(marked))
    return "\n".join(lines)


def _register_config_commands(app: App, settings: Settings) -> None:
    """Slash commands for on-the-fly configuration (scribe#2, per-user in scribe#6).

    Every command writes the INVOKING user's own layer, so two people never fight over
    a voice. A trailing `default` word writes the shared layer everyone falls back to
    instead. Changes take effect for jobs started after the command; anything already
    running finishes under the settings it began with, so a mid-queue toggle cannot
    produce a half-configured document.
    """

    def _state_line(s: Settings) -> str:
        return (
            f"voice *{s.tts_voice}* · note *{s.note_format}* · "
            f"TTS *{'on' if s.tts_enabled else 'off'}*"
        )

    def _scope(command) -> tuple[str, str | None, str]:
        """(argument text, target user or None for the shared layer, scope label)."""
        words = (command.get("text") or "").split()
        if words and words[-1].lower() == "default":
            return " ".join(words[:-1]), None, "for everyone by default"
        return " ".join(words), command.get("user_id") or None, "for you"

    def _both_off_note(after: Settings) -> str:
        if after.note_format == "none" and not after.tts_enabled:
            return "\n_Both outputs are off — documents will only get a TL;DR in thread._"
        return ""

    @app.command("/scribevoice")
    def on_voice(ack, respond, command):
        ack()
        wanted, user, scope = _scope(command)
        current = effective(settings, user)
        try:
            available = voices(current)
        except TTSError as exc:
            respond(f"Couldn't reach the TTS server to list voices: {exc}")
            return
        if not wanted:
            respond(_voice_menu(settings, available, current.tts_voice))
            return
        if wanted not in available:
            near = [v for v in available if wanted.lower() in v.lower()]
            hint = f" Did you mean {', '.join(f'`{v}`' for v in near[:3])}?" if near else ""
            respond(f"`{wanted}` is not a voice this server serves.{hint}")
            return
        set_value(settings, "tts_voice", wanted, user=user)
        # No memory caveat here. Measured 2026-09-16 in the Kokoro pod: a voice pack
        # is 523 KB, all 68 together 35 MB, and nothing loads until a document is
        # synthesized. The old "seven voices OOM'd 2Gi" note blamed voices for the
        # per-request leak (k8s#146). Audition as many as you like.
        respond(
            f"Voice set to *{wanted}* {scope} from the next document. "
            f"{_state_line(effective(settings, user))}"
        )

    def _toggle(key: str, label: str, respond, command) -> None:
        arg, user, scope = _scope(command)
        current = effective(settings, user)
        arg = arg.strip().lower()
        # An explicit on/off is idempotent; bare invocation flips. Both are useful —
        # flipping is fastest from a phone, explicit is safe in a script.
        if arg in {"on", "off"}:
            new = arg == "on"
        elif arg:
            respond(f"Use `on`, `off`, or no argument to flip. {_state_line(current)}")
            return
        else:
            new = not getattr(current, key)
        set_value(settings, key, new, user=user)
        after = effective(settings, user)
        respond(f"{label} is now *{'on' if new else 'off'}* {scope}. "
                f"{_state_line(after)}{_both_off_note(after)}")

    @app.command("/scribeformat")
    def on_format(ack, respond, command):
        """Pick the file format the full note is delivered in, or `none` for TL;DR only."""
        ack()
        wanted, user, scope = _scope(command)
        wanted = wanted.strip().lower()
        current = effective(settings, user)
        choices = ", ".join(f"`{f}`" for f in FORMATS)
        if not wanted:
            respond(f"Your notes are delivered as *{current.note_format}*. Options: {choices}. "
                    f"Add `default` to change the shared default instead.")
            return
        if wanted not in FORMATS:
            respond(f"`{wanted}` is not a note format. Options: {choices}.")
            return
        set_value(settings, "note_format", wanted, user=user)
        after = effective(settings, user)
        respond(f"Notes will be delivered as *{wanted}* {scope} from the next document. "
                f"{_state_line(after)}{_both_off_note(after)}")

    @app.command("/scribetoggletts")
    def on_toggle_tts(ack, respond, command):
        ack()
        _toggle("tts_enabled", "TTS audio", respond, command)

    @app.command("/scribeconfig")
    def on_config(ack, respond, command):
        ack()
        user = command.get("user_id") or None
        mine = effective(settings, user)
        shared = effective(settings)
        own = load(settings, user) if user else {}
        lines = [f"Your settings: {_state_line(mine)}"]
        if own:
            lines.append(f"Shared defaults: {_state_line(shared)}")
            lines.append("_You have overridden: " + ", ".join(sorted(own)) +
                         ". Commands change your own settings; add `default` to change "
                         "the shared ones._")
        else:
            lines.append("_You are on the shared defaults. Any command you run changes "
                         "only your settings; add `default` to change everyone's._")
        lines.append(f"_Estimator calibration: {Calibration.load(settings).describe()}_")
        respond("\n".join(lines))


def build_app(settings: Settings) -> tuple[App, ThreadPoolExecutor]:
    app = App(token=settings.slack_bot_token)
    _register_config_commands(app, settings)
    # Needed to recognize our own @-mentions inside drop channels: a mention there fires
    # BOTH app_mention and message.channels for the same message, and handling both
    # would summarize the document twice.
    bot_user_id = app.client.auth_test().get("user_id", "")
    # One summarize worker: Ollama is the bottleneck and handles one request at a time.
    # Parallel documents would not finish sooner, only thrash a shared bottleneck.
    # One audio worker: Kokoro is a DIFFERENT server, so the two stages overlap — the
    # next document summarizes while the last one synthesizes (scribe#7).
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe")
    audio_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe-audio")
    pending = _Pending()
    audio_pending = _Pending()
    active = _Active()
    user_tz = _UserTz()
    user_name = _UserName()

    def audio_submit(aj: AudioJob) -> None:
        _submit_audio(settings, audio_pool, audio_pending, active, app.client, aj)

    def audio_ahead() -> float:
        return audio_pending.peek()[1]

    def quote(job: Job, tz: str | None) -> tuple[str, float]:
        """Ack text and the summary-stage seconds to book in the queue."""
        est = estimate(settings, job.target)
        cal = Calibration.load(settings)
        eff = effective(settings, job.user)
        audio_on = eff.tts_enabled and bool(eff.abs_token)
        ahead_n, ahead_seconds = pending.peek()
        text, summary_q, audio_q = _ack_text(
            settings, est, cal, ahead_seconds, audio_ahead(), tz=tz, audio_on=audio_on)
        log.info(
            "eta.ack job=%s kind=%s chars=%d ocr_pages=%d branch=%s raw_ocr=%.0f raw_sum=%.0f "
            "raw_audio=%.0f quote_summary=%.0f quote_audio=%.0f ahead_n=%d ahead=%.0f "
            "audio_ahead=%.0f",
            job.id, est.kind, est.chars, est.ocr_pages, est.branch, est.ocr_seconds,
            est.summarize_seconds, est.audio_seconds, summary_q, audio_q, ahead_n,
            ahead_seconds, audio_ahead(),
        )
        queued = f" It is queued behind {ahead_n} other item(s)." if ahead_n else ""
        return text + queued, summary_q

    def handle(event: dict, say, client) -> None:
        # Ignore our own messages, or we would answer ourselves forever.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Slack fires a SECOND message event when it unfurls a link — subtype
        # "message_changed", because the message is edited to attach the preview card.
        # That event carries no top-level text, so it used to fall through to the help
        # reply: every link produced a spurious extra message. Edits, deletions and thread
        # broadcasts arrive the same way. Only a genuine new message (no subtype) or a
        # file upload is actionable.
        subtype = event.get("subtype")
        if subtype not in (None, "file_share"):
            return

        channel = event["channel"]
        # Reply in a thread on the original message so the channel stays readable.
        thread_ts = event.get("thread_ts") or event["ts"]

        # A cancel reply IN an existing thread stops that thread's job(s). Checked before
        # link/file extraction so "cancel" can never be mistaken for content. Spool
        # records are removed HERE, durably — a pod restart must not resurrect a job the
        # user already canceled; the in-flight pipeline stops at its next checkpoint.
        if event.get("thread_ts") and not event.get("files") and is_cancel(event.get("text", "")):
            jobs = active.cancel_thread(event["thread_ts"])
            for j in jobs:
                if isinstance(j, AudioJob):
                    complete_audio(settings, j)
                else:
                    complete(settings, j)
            if jobs:
                labels = ", ".join(j.source_label for j in jobs)
                say(
                    text=f"Canceled — {labels}. If a step was mid-flight it stops at "
                         f"the next checkpoint.",
                    thread_ts=thread_ts,
                )
            else:
                say(text="Nothing is running in this thread.", thread_ts=thread_ts)
            return

        target: str | None = None
        source_label: str | None = None
        attachment: str | None = None
        attachment_name: str | None = None

        files = event.get("files") or []
        if files:
            name = files[0].get("name") or "upload"
            # Staged in the spool, not a tempdir: the bytes must outlive a restart or the
            # resumed job would have nothing to read.
            dest = spool(settings) / f"{Job.new(channel, thread_ts, '', '').id}-{name}"
            try:
                downloaded = download_file(settings, files[0], dest)
            except Exception as exc:
                say(text=f"Couldn't download that file: {exc}", thread_ts=thread_ts)
                return
            target = str(downloaded)
            attachment = str(downloaded)
            attachment_name = name
            source_label = f"`{name}`"
        else:
            target = first_url(event.get("text", ""))
            source_label = target

        if not target:
            # Help ON REQUEST only. Answering every unrecognized message turns the bot
            # into a nag; the same text lives in the app's description for discovery.
            if event.get("text", "").strip().lower().lstrip("!/") in HELP_WORDS:
                say(text=HELP_TEXT, thread_ts=thread_ts)
            else:
                log.info("no link or file in message; staying quiet")
            return

        job = Job.new(channel, thread_ts, target, source_label or target)
        job.attachment = attachment
        job.attachment_name = attachment_name
        job.user = event.get("user")
        enqueue(settings, job)

        # Acknowledge immediately. The pipeline takes minutes on CPU, so without this the
        # user has no signal anything is happening. The source is echoed so the thread
        # reads coherently top to bottom. The quoted time is when THIS document should
        # finish — its own estimate plus everything queued ahead of it.
        tz = user_tz.get(client, job.user)
        text, est = quote(job, tz)
        pending.add(est)
        say(text=f"On it — {job.source_label}. {text}", thread_ts=thread_ts)
        _submit(settings, pool, pending, active, client, job, est, audio_submit,
                audio_ahead, tz, user_name.get(client, job.user))

    @app.event("app_mention")
    def on_mention(event, say, client):
        handle(event, say, client)

    @app.event("message")
    def on_message(event, say, client):
        # DMs always; channels only when allowlisted as a DROP CHANNEL (every link or
        # file processed, no mention needed — SCRIBE_DROP_CHANNELS). Other channels stay
        # mention-only via app_mention, so inviting scribe somewhere for @-mentions never
        # turns that channel into a firehose.
        if event.get("channel_type") == "im":
            handle(event, say, client)
            return
        if (
            event.get("channel_type") == "channel"
            and event.get("channel") in settings.drop_channel_ids
        ):
            # A mention in a drop channel also arrives as app_mention — that handler owns
            # it. Without this check the same message would be processed twice.
            if bot_user_id and f"<@{bot_user_id}>" in (event.get("text") or ""):
                return
            handle(event, say, client)

    # Resume anything the previous run did not finish, in the order it arrived. Audio
    # first: those documents are furthest along and their threads already have a note,
    # so they resume silently. Summarize jobs are told explicitly — a silently resumed
    # job is indistinguishable from a stalled one, which the spool exists to prevent.
    for audio_job in restore_audio(settings):
        log.info("resuming audio job %s (%s)", audio_job.id, audio_job.source_label)
        audio_submit(audio_job)
    for job in restore(settings):
        log.info("resuming queued job %s (%s)", job.id, job.source_label)
        tz = user_tz.get(app.client, job.user)
        text, est = quote(job, tz)
        pending.add(est)
        try:
            app.client.chat_postMessage(
                channel=job.channel,
                thread_ts=job.thread_ts,
                text=f"Picking this back up after a restart — {job.source_label}. {text}",
            )
        except Exception:
            log.exception("could not notify resume for %s", job.id)
        _submit(settings, pool, pending, active, app.client, job, est, audio_submit,
                audio_ahead, tz, user_name.get(app.client, job.user))

    return app, pool


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    if not settings.slack_bot_token or not settings.slack_app_token:
        log.error("need SCRIBE_SLACK_BOT_TOKEN (xoxb-) and SCRIBE_SLACK_APP_TOKEN (xapp-)")
        return 1
    app, _pool = build_app(settings)
    log.info("connecting to Slack over Socket Mode")
    SocketModeHandler(app, settings.slack_app_token).start()
    return 0
