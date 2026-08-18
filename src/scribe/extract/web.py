"""Article extraction for links.

trafilatura is used rather than a hand-rolled readability pass: it strips nav, ads, and
boilerplate, and returns the article body as markdown, which is already the shape the vault
note wants.
"""

from __future__ import annotations

import httpx
import trafilatura

from scribe.config import Settings
from scribe.document import Document, Method, Page


class ExtractionError(RuntimeError):
    pass


def extract_url(settings: Settings, url: str) -> Document:
    """Fetch and extract an article.

    On the User-Agent: identifying scribe honestly, with contact info, works *better* than
    spoofing a browser. Measured against Wikipedia:

        Mozilla/5.0 (... Chrome/131 ...)                      -> 403, bot-policy notice
        scribe/0.1 (https://meklab.net; <contact>) python-httpx -> 200

    Wikimedia's robot policy asks automated clients to declare themselves and provide a
    contact address; a generic browser string reads as an evasive bot and is refused. Keep
    this descriptive.
    """
    try:
        resp = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExtractionError(f"could not fetch {url}: {exc}") from exc

    text = trafilatura.extract(
        resp.text,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        with_metadata=False,
    )
    if not text or not text.strip():
        raise ExtractionError(
            f"no article content found at {url} — the page may be JS-rendered or paywalled"
        )

    meta = trafilatura.extract_metadata(resp.text)
    title = getattr(meta, "title", None) if meta else None

    return Document(
        source=url,
        kind="link",
        title=title or url,
        pages=[Page(number=1, text=text.strip(), method=Method.WEB)],
    )
