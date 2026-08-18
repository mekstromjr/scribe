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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from scribe.config import Settings, load_settings
from scribe.extract import ExtractionError, extract
from scribe.note import render, slugify
from scribe.ollama import OllamaError
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


def download_file(settings: Settings, file_info: dict, dest_dir: Path) -> Path:
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
    path = dest_dir / name
    path.write_bytes(resp.content)
    return path


def _process(settings: Settings, client, channel: str, thread_ts: str, target: str,
             local_file: Path | None) -> None:
    """Run the pipeline and reply in-thread. Never raises — failures are reported to Slack."""
    try:
        doc = extract(settings, target)
        summary = summarize(settings, doc)

        attachment_path = resolve_attachment(settings, local_file) if local_file else None
        body = render(doc, summary, model=settings.text_model, attachment_link=attachment_path)
        result = publish(
            settings,
            note_body=body,
            note_stem=slugify(summary.title),
            attachment=local_file,
            attachment_path=attachment_path,
        )

        note_name = Path(result["note"]).stem
        lines = [
            f"*{summary.title}*",
            "",
            summary.tldr,
            "",
            f"Saved to your vault as `{note_name}`",
        ]
        if summary.truncated_chars:
            lines.append(
                f"_Note: {summary.truncated_chars} characters were trimmed to fit the "
                f"model's context, so the summary covers only part of the source._"
            )
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text="\n".join(lines))
    except (ExtractionError, OllamaError, VaultError) as exc:
        client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=f"Sorry — that failed: {exc}"
        )
    except Exception:
        # A crash here must not kill the worker thread and silently stop the queue.
        log.exception("unexpected failure processing %s", target)
        client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text="Sorry — that failed unexpectedly."
        )
    finally:
        if local_file is not None:
            local_file.unlink(missing_ok=True)


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


def build_app(settings: Settings) -> tuple[App, ThreadPoolExecutor]:
    app = App(token=settings.slack_bot_token)
    # One worker: the model server is the bottleneck and handles one request at a time.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scribe")
    pending = _Pending()

    def handle(event: dict, say, client) -> None:
        # Ignore our own messages, or we would answer ourselves forever.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = event["channel"]
        # Reply in a thread on the original message so the channel stays readable.
        thread_ts = event.get("thread_ts") or event["ts"]

        tmpdir = Path(tempfile.mkdtemp(prefix="scribe-"))
        local_file: Path | None = None
        target: str | None = None

        files = event.get("files") or []
        if files:
            try:
                local_file = download_file(settings, files[0], tmpdir)
                target = str(local_file)
            except Exception as exc:
                say(text=f"Couldn't download that file: {exc}", thread_ts=thread_ts)
                return
        else:
            target = first_url(event.get("text", ""))

        if not target:
            say(
                text="Send me a link, PDF, or image and I'll summarize it into your vault.",
                thread_ts=thread_ts,
            )
            return

        # Acknowledge immediately. The pipeline takes minutes on CPU, so without this the
        # user has no signal that anything is happening.
        ahead = pending.add()
        note = f" (queued behind {ahead} other job(s))" if ahead else ""
        say(
            text=f"On it — reading and summarizing{note}. This takes a few minutes.",
            thread_ts=thread_ts,
        )

        fut = pool.submit(_process, settings, client, channel, thread_ts, target, local_file)
        fut.add_done_callback(lambda _f: pending.done())

    @app.event("app_mention")
    def on_mention(event, say, client):
        handle(event, say, client)

    @app.event("message")
    def on_message(event, say, client):
        # Only DMs; channel messages arrive via app_mention so we never read traffic we
        # were not addressed in.
        if event.get("channel_type") == "im":
            handle(event, say, client)

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
