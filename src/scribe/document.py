"""Result types for extraction.

Kept separate from the extractors so the Slack and vault layers (Phases 3-4) can depend on
the shape without importing pypdfium2 or trafilatura.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Method(StrEnum):
    """How a page's text was obtained. Surfaced in the note header so a bad transcription
    can be attributed to OCR rather than assumed to be the source's own wording."""

    TEXT_LAYER = "text-layer"
    OCR = "ocr"
    WEB = "web"
    SKIPPED = "skipped"


class Page(BaseModel):
    number: int
    text: str
    method: Method
    # Wall-clock seconds. glm-ocr returns ZERO for every duration field it reports
    # (load_duration, eval_count, total_duration), so the API's own timings are unusable
    # and this is measured by the caller instead.
    seconds: float = 0.0


class Document(BaseModel):
    source: str
    kind: str  # pdf | image | link
    title: str | None = None
    pages: list[Page] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.method is Method.OCR)

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.pages)

    def summary_line(self) -> str:
        methods = {p.method for p in self.pages}
        label = "mixed" if len(methods) > 1 else (next(iter(methods), Method.SKIPPED).value)
        return (
            f"{self.kind} · {len(self.pages)} page(s) · extraction: {label} · "
            f"{self.ocr_pages} OCR'd · {self.seconds:.1f}s"
        )
