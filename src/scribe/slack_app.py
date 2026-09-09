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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from scribe.abs import ABSError, upload
from scribe.audio import produce_audio
from scribe.config import Settings, load_settings
from scribe.eta import estimate_seconds, eta_line
from scribe.extract import ExtractionError, extract
from scribe.note import note_title, obsidian_uri, render, slugify
from scribe.ollama import OllamaError
from scribe.queue import Job, complete, enqueue, page_cache, restore, spool
from scribe.runtime_config import effective, set_value
from scribe.summarize import summarize
from scribe.tts import TTSError, voices
from scribe.vault import VaultError, append_listen_link, publish, resolve_attachment

log = logging.getLogger("scribe.slack")

# Slack wraps links as <https://x> or <https://x|label>; take the URL, drop the label.
_URL = re.compile(r"<(https?://[^|>\s]+)(?:\|[^>]*)?>")
_BARE_URL = re.compile(r"https?://\S+")

HELP_WORDS = {"help", "?", "usage", "how does scribe work?", "how does this work?"}
HELP_TEXT = (
    "Send me a *link*, *PDF*, or *image* — as a DM here, or @-mention me in a channel.\n\n"
    "I extract the text, write a thorough summary, and save a note to your Obsidian vault "
    "inbox (`+/`). You get the TL;DR back in thread with a link that opens the note in "
    "Obsidian.\n\n"
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
        self._by_thread: dict[str, dict[str, Job]] = {}
        self._counts: dict[str, int] = {}
        self._canceled: set[str] = set()
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._by_thread.setdefault(job.thread_ts, {})[job.id] = job
            self._counts[job.id] = self._counts.get(job.id, 0) + 1

    def remove(self, job: Job) -> None:
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

    def cancel_thread(self, thread_ts: str) -> list[Job]:
        """Mark every job in the thread canceled; returns them (may be empty)."""
        with self._lock:
            jobs = list(self._by_thread.get(thread_ts, {}).values())
            self._canceled.update(j.id for j in jobs)
        return jobs

    def canceled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._canceled


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


def _audio_stage(settings: Settings, client, job: Job, doc, summary, note_path: str) -> None:
    """Synthesize, upload to Audiobookshelf, link the note, and post the m4b in-thread.

    Never raises. Each delivery step degrades independently: an ABS outage still posts
    the file to Slack, a Slack upload failure still leaves the ABS link, and a vault
    hiccup loses only the note's listen line.
    """
    settings = effective(settings)
    title = note_title(doc, summary)
    author = doc.source if doc.kind == "link" else "scribe"
    try:
        result = produce_audio(settings, doc, summary, title=title, author=author)
    except Exception as exc:
        log.warning("audio synthesis failed for %s: %s", job.source_label, exc)
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"_(No audio this time — synthesis failed: {exc})_",
        )
        return

    with result.workdir:
        minutes = result.audio_seconds / 60
        abs_line = ""
        try:
            link = upload(settings, result.m4b, title=title, author=author)
            abs_line = f"Listen in <{link}|Audiobookshelf> ({minutes:.0f} min)."
        except ABSError as exc:
            log.warning("ABS upload failed for %s: %s", job.source_label, exc)
            abs_line = f"_(Audiobookshelf upload failed: {exc})_"
        else:
            # No note to link when vault publishing is off.
            if note_path:
                try:
                    append_listen_link(settings, note_path, link)
                except VaultError as exc:
                    log.warning("listen-link commit failed for %s: %s", note_path, exc)

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


def _process(settings: Settings, client, job: Job, requeue=lambda _job: None,
             active: _Active | None = None) -> None:
    """Run the pipeline and reply in-thread. Never raises — failures are reported to Slack."""
    # Runtime overrides are read ONCE, here, at the start of the job: a toggle typed
    # while this document is mid-flight applies to the next one, so a job's behavior
    # never changes underneath the ack the user already received.
    settings = effective(settings)
    local_file = Path(job.attachment) if job.attachment else None
    requeued = False

    def abort() -> None:
        # The one user-visible cancel message was already posted by the cancel command;
        # everything after it tears down silently.
        if active and active.canceled(job.id):
            raise JobCanceled()

    try:
        abort()
        # The page cache lives in the spool under the job id, so a requeued attempt
        # resumes OCR at the first unfinished page instead of redoing them all (scribe#4).
        doc = extract(settings, job.target, cache=page_cache(settings, job.id))
        if job.attachment_name:
            # The extractors derive title/source from the file PATH, which for an upload is
            # the SPOOLED name carrying a job-id prefix (kept so concurrent uploads cannot
            # collide on disk). Left alone that prefix becomes the note's title, its H1 and
            # its Sources frontmatter -- e.g. "1787110665559937147-a2779432-Syllabus".
            doc.source = job.attachment_name
            doc.title = Path(job.attachment_name).stem
        summary = summarize(settings, doc, abort=abort)

        # Publishing is the point of no return: past here the note exists and cancel
        # would leave more mess than it saves.
        abort()
        result = {"note": ""}
        if settings.vault_enabled:
            attachment_path = (
                resolve_attachment(settings, local_file, job.attachment_name)
                if local_file
                else None
            )
            body = render(
                doc, summary, model=settings.text_model, attachment_link=attachment_path
            )
            result = publish(
                settings,
                note_body=body,
                note_stem=slugify(note_title(doc, summary)),
                attachment=local_file,
                attachment_path=attachment_path,
            )

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
        if settings.vault_enabled:
            note_name = Path(result["note"]).stem
            uri = obsidian_uri(settings.obsidian_vault_name, result["note"])
            # Deep link rather than just the name: tapping it opens the note directly
            # in Obsidian on phone or laptop, which is the whole point of a read-later
            # queue. Slack renders <uri|label>.
            lines.append(f"Saved to your vault: <{uri}|{note_name}>")
        else:
            lines.append("_Vault publishing is off — this summary lives only here._")
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
        # Audio AFTER the note reply, and best-effort: at Kokoro's measured 1.6x
        # realtime a long article synthesizes for tens of minutes, and a TTS failure
        # must never fail (or requeue) a job whose note already published.
        if settings.tts_enabled and settings.abs_token:
            if active and active.canceled(job.id):
                # Canceled after the note published: keep the note, skip only the audio.
                log.info("skipping audio for canceled job %s", job.id)
            else:
                _audio_stage(settings, client, job, doc, summary, result["note"])
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
    except (OllamaError, VaultError) as exc:
        # Transient: the model server or GitLab was unreachable. Retry rather than drop
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


class _Pending:
    """Counts queued jobs — and their estimated seconds — so the ack can quote when THIS
    document will be done, not when it will merely start.

    Tracked explicitly rather than reading ThreadPoolExecutor._work_queue, which is a
    private attribute with no stability guarantee. The seconds figure deliberately counts
    an in-flight job at its full estimate: tracking its remaining time would need worker
    progress plumbing, and overshooting a queue-wait estimate is the cheap direction to
    be wrong in.
    """

    def __init__(self) -> None:
        self._n = 0
        self._seconds = 0.0
        self._lock = threading.Lock()

    def add(self, est_seconds: float) -> tuple[int, float]:
        """Returns (jobs ahead, estimated seconds ahead) as of just before this add."""
        with self._lock:
            ahead = (self._n, self._seconds)
            self._n += 1
            self._seconds += est_seconds
        return ahead

    def done(self, est_seconds: float) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)
            self._seconds = max(0.0, self._seconds - est_seconds)


def _submit(settings: Settings, pool: ThreadPoolExecutor, pending: _Pending,
            active: _Active, client, job: Job, est_seconds: float) -> None:
    def requeue(j: Job) -> None:
        # Back of the queue, not the front: a document whose dependency is down should not
        # block everything behind it while it retries.
        pending.add(est_seconds)
        _submit(settings, pool, pending, active, client, j, est_seconds)

    active.add(job)
    fut = pool.submit(_process, settings, client, job, requeue, active)

    def _done(_f) -> None:
        pending.done(est_seconds)
        active.remove(job)

    fut.add_done_callback(_done)


def _register_config_commands(app: App, settings: Settings) -> None:
    """Slash commands for on-the-fly configuration (scribe#2).

    Changes take effect for jobs started after the command; anything already running
    finishes under the settings it began with, so a mid-queue toggle cannot produce a
    half-configured document.
    """

    def _state_line(s: Settings) -> str:
        return (
            f"voice *{s.tts_voice}* · vault *{'on' if s.vault_enabled else 'off'}* · "
            f"TTS *{'on' if s.tts_enabled else 'off'}*"
        )

    @app.command("/scribevoice")
    def on_voice(ack, respond, command):
        ack()
        current = effective(settings)
        wanted = (command.get("text") or "").strip()
        try:
            available = voices(current)
        except TTSError as exc:
            respond(f"Couldn't reach the TTS server to list voices: {exc}")
            return
        if not wanted:
            listing = "\n".join(
                f"• `{v}`{'  ← current' if v == current.tts_voice else ''}"
                for v in available
            )
            respond(f"Current voice: *{current.tts_voice}*\n\n{listing}")
            return
        if wanted not in available:
            near = [v for v in available if wanted.lower() in v.lower()]
            hint = f" Did you mean {', '.join(f'`{v}`' for v in near[:3])}?" if near else ""
            respond(f"`{wanted}` is not a voice this server serves.{hint}")
            return
        set_value(settings, "tts_voice", wanted)
        respond(
            f"Voice set to *{wanted}* for the next document. "
            f"{_state_line(effective(settings))}"
        )

    def _toggle(key: str, label: str, respond, command) -> None:
        current = effective(settings)
        arg = (command.get("text") or "").strip().lower()
        # An explicit on/off is idempotent; bare invocation flips. Both are useful —
        # flipping is fastest from a phone, explicit is safe in a script.
        if arg in {"on", "off"}:
            new = arg == "on"
        elif arg:
            respond(f"Use `on`, `off`, or no argument to flip. {_state_line(current)}")
            return
        else:
            new = not getattr(current, key)
        set_value(settings, key, new)
        after = effective(settings)
        note = ""
        if not after.vault_enabled and not after.tts_enabled:
            note = "\n_Both outputs are off — documents will only get a TL;DR in thread._"
        respond(f"{label} is now *{'on' if new else 'off'}*. {_state_line(after)}{note}")

    @app.command("/scribetoggleobs")
    def on_toggle_obs(ack, respond, command):
        ack()
        _toggle("vault_enabled", "Vault publishing", respond, command)

    @app.command("/scribetoggletts")
    def on_toggle_tts(ack, respond, command):
        ack()
        _toggle("tts_enabled", "TTS audio", respond, command)

    @app.command("/scribeconfig")
    def on_config(ack, respond, command):  # noqa: ARG001
        ack()
        respond(f"scribe: {_state_line(effective(settings))}")


def build_app(settings: Settings) -> tuple[App, ThreadPoolExecutor]:
    app = App(token=settings.slack_bot_token)
    _register_config_commands(app, settings)
    # Needed to recognize our own @-mentions inside drop channels: a mention there fires
    # BOTH app_mention and message.channels for the same message, and handling both
    # would summarize the document twice.
    bot_user_id = app.client.auth_test().get("user_id", "")
    # One worker: the model server is the bottleneck and handles one request at a time.
    # Parallel documents would not finish sooner, only thrash a shared bottleneck.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe")
    pending = _Pending()
    active = _Active()
    user_tz = _UserTz()

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
        est = estimate_seconds(settings, job.target)
        ahead_n, ahead_seconds = pending.add(est)
        queued = f" It is queued behind {ahead_n} other item(s)." if ahead_n else ""
        say(
            text=f"On it — {job.source_label}. "
                 f"{eta_line(settings, est + ahead_seconds, tz=user_tz.get(client, job.user))}"
                 f"{queued}",
            thread_ts=thread_ts,
        )
        _submit(settings, pool, pending, active, client, job, est)

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

    # Resume anything the previous run did not finish, in the order it arrived. Each
    # thread is told explicitly — a silently resumed job is indistinguishable from a
    # stalled one, which is the confusion the spool exists to prevent.
    for job in restore(settings):
        log.info("resuming queued job %s (%s)", job.id, job.source_label)
        est = estimate_seconds(settings, job.target)
        _n, ahead_seconds = pending.add(est)
        try:
            app.client.chat_postMessage(
                channel=job.channel,
                thread_ts=job.thread_ts,
                text=f"Picking this back up after a restart — {job.source_label}. "
                     + eta_line(settings, est + ahead_seconds,
                                tz=user_tz.get(app.client, job.user)),
            )
        except Exception:
            log.exception("could not notify resume for %s", job.id)
        _submit(settings, pool, pending, active, app.client, job, est)

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
