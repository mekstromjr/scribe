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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from scribe.config import Settings, load_settings
from scribe.extract import ExtractionError, extract
from scribe.note import note_title, obsidian_uri, render, slugify
from scribe.ollama import OllamaError
from scribe.queue import Job, complete, enqueue, restore, spool
from scribe.summarize import summarize
from scribe.vault import VaultError, publish, resolve_attachment

log = logging.getLogger("scribe.slack")

# Slack wraps links as <https://x> or <https://x|label>; take the URL, drop the label.
_URL = re.compile(r"<(https?://[^|>\s]+)(?:\|[^>]*)?>")
_BARE_URL = re.compile(r"https?://\S+")


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


def _process(settings: Settings, client, job: Job) -> None:
    """Run the pipeline and reply in-thread. Never raises — failures are reported to Slack."""
    local_file = Path(job.attachment) if job.attachment else None
    try:
        doc = extract(settings, job.target)
        summary = summarize(settings, doc)

        attachment_path = resolve_attachment(settings, local_file) if local_file else None
        body = render(doc, summary, model=settings.text_model, attachment_link=attachment_path)
        result = publish(
            settings,
            note_body=body,
            note_stem=slugify(note_title(doc, summary)),
            attachment=local_file,
            attachment_path=attachment_path,
        )

        note_name = Path(result["note"]).stem
        uri = obsidian_uri(settings.obsidian_vault_name, result["note"])
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
            # Deep link rather than just the name: tapping it opens the note directly
            # in Obsidian on phone or laptop, which is the whole point of a read-later
            # queue. Slack renders <uri|label>.
            f"Saved to your vault: <{uri}|{note_name}>",
        ]
        if summary.truncated_chars:
            lines.append(
                f"_Note: {summary.truncated_chars} characters were trimmed to fit the "
                f"model's context, so the summary covers only part of the source._"
            )
        client.chat_postMessage(
            channel=job.channel, thread_ts=job.thread_ts, text="\n".join(lines)
        )
    except (ExtractionError, OllamaError, VaultError) as exc:
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=f"Sorry — {job.source_label} failed: {exc}",
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
        complete(settings, job)


class _Pending:
    """Counts queued jobs so the ack can say whether the user is waiting behind others.

    Tracked explicitly rather than reading ThreadPoolExecutor._work_queue, which is a
    private attribute with no stability guarantee.
    """

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def add(self) -> int:
        with self._lock:
            ahead = self._n
            self._n += 1
        return ahead

    def done(self) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)


def _submit(settings: Settings, pool: ThreadPoolExecutor, pending: _Pending, client,
            job: Job) -> None:
    fut = pool.submit(_process, settings, client, job)
    fut.add_done_callback(lambda _f: pending.done())


def build_app(settings: Settings) -> tuple[App, ThreadPoolExecutor]:
    app = App(token=settings.slack_bot_token)
    # One worker: the model server is the bottleneck and handles one request at a time.
    # Parallel documents would not finish sooner, only thrash a shared bottleneck.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe")
    pending = _Pending()

    def handle(event: dict, say, client) -> None:
        # Ignore our own messages, or we would answer ourselves forever.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = event["channel"]
        # Reply in a thread on the original message so the channel stays readable.
        thread_ts = event.get("thread_ts") or event["ts"]

        target: str | None = None
        source_label: str | None = None
        attachment: str | None = None

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
            source_label = f"`{name}`"
        else:
            target = first_url(event.get("text", ""))
            source_label = target

        if not target:
            say(
                text="Send me a link, PDF, or image and I'll summarize it into your vault.",
                thread_ts=thread_ts,
            )
            return

        job = Job.new(channel, thread_ts, target, source_label or target)
        job.attachment = attachment
        enqueue(settings, job)

        # Acknowledge immediately. The pipeline takes minutes on CPU, so without this the
        # user has no signal anything is happening. The source is echoed so the thread
        # reads coherently top to bottom.
        ahead = pending.add()
        queued = f" It is queued behind {ahead} other item(s)." if ahead else ""
        say(
            text=f"On it — {job.source_label}. This takes a few minutes.{queued}",
            thread_ts=thread_ts,
        )
        _submit(settings, pool, pending, client, job)

    @app.event("app_mention")
    def on_mention(event, say, client):
        handle(event, say, client)

    @app.event("message")
    def on_message(event, say, client):
        # Only DMs; channel messages arrive via app_mention so we never read traffic we
        # were not addressed in.
        if event.get("channel_type") == "im":
            handle(event, say, client)

    # Resume anything the previous run did not finish, in the order it arrived. Each
    # thread is told explicitly — a silently resumed job is indistinguishable from a
    # stalled one, which is the confusion the spool exists to prevent.
    for job in restore(settings):
        log.info("resuming queued job %s (%s)", job.id, job.source_label)
        pending.add()
        try:
            app.client.chat_postMessage(
                channel=job.channel,
                thread_ts=job.thread_ts,
                text=f"Picking this back up after a restart — {job.source_label}.",
            )
        except Exception:
            log.exception("could not notify resume for %s", job.id)
        _submit(settings, pool, pending, app.client, job)

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
