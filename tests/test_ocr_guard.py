"""The OCR runaway guard (scribe#4). No network — ollama is stubbed at the HTTP edge.

Background: a 10-page scan hung for three hours because glm-ocr looped on the book's title
page. Nothing bounded generation, the client timeout was the only stop, and the retry path
redid the nine good pages each time only to hang on the same tenth. These tests pin the
three fixes: a generation cap whose hit is detected, a per-page timeout that skips instead
of requeueing, and finished pages surviving a requeue.
"""

from __future__ import annotations

import io

import httpx
import pypdfium2 as pdfium
import pytest
from PIL import Image

from scribe import ollama
from scribe.config import Settings
from scribe.document import Method, Page
from scribe.extract import extract
from scribe.extract import ocr as ocr_module
from scribe.extract import pdf as pdf_module
from scribe.note import _provenance
from scribe.ollama import OllamaError, OllamaTimeout, VisionResult, generate_with_image
from scribe.queue import Job, complete, enqueue, page_cache, restore
from scribe.summarize import Summary


@pytest.fixture
def settings(tmp_path):
    return Settings(spool_dir=str(tmp_path / "queue"), ocr_num_predict=3000,
                    ocr_timeout_seconds=900.0)


def _image() -> Image.Image:
    return Image.new("RGB", (40, 40), "white")


def _fake_post(response: dict | None = None, *, raises: Exception | None = None):
    """Stand in for httpx.post, recording the payload it was called with."""
    calls: list[dict] = []

    def post(url, *, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if raises:
            raise raises
        return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    return post, calls


class TestGenerateWithImage:
    def test_sends_the_generation_cap_and_ocr_timeout(self, settings, monkeypatch):
        post, calls = _fake_post({"response": "hello", "done_reason": "stop"})
        monkeypatch.setattr(ollama.httpx, "post", post)

        result = generate_with_image(settings, "transcribe", b"png")

        [call] = calls
        assert call["json"]["options"]["num_predict"] == settings.ocr_num_predict
        assert call["timeout"] == settings.ocr_timeout_seconds
        assert result == VisionResult(text="hello", seconds=result.seconds, complete=True)

    def test_hitting_the_cap_is_reported_as_incomplete(self, settings, monkeypatch):
        post, _ = _fake_post({"response": "PLATO PLATO PLATO", "done_reason": "length"})
        monkeypatch.setattr(ollama.httpx, "post", post)

        assert generate_with_image(settings, "p", b"x").complete is False

    def test_a_missing_done_reason_is_treated_as_finished(self, settings, monkeypatch):
        # Older servers omit the field; the absence of a "length" stop is the finish
        # signal, so silence must not skip a good page.
        post, _ = _fake_post({"response": "ok"})
        monkeypatch.setattr(ollama.httpx, "post", post)

        assert generate_with_image(settings, "p", b"x").complete is True

    def test_timeout_is_its_own_error(self, settings, monkeypatch):
        post, _ = _fake_post(raises=httpx.ReadTimeout("slow"))
        monkeypatch.setattr(ollama.httpx, "post", post)

        with pytest.raises(OllamaTimeout):
            generate_with_image(settings, "p", b"x")

    def test_unreachable_is_still_a_plain_ollama_error(self, settings, monkeypatch):
        # Connection refused is the transient case the requeue path exists for; it must
        # NOT be swallowed into a skipped page.
        post, _ = _fake_post(raises=httpx.ConnectError("refused"))
        monkeypatch.setattr(ollama.httpx, "post", post)

        with pytest.raises(OllamaError) as exc_info:
            generate_with_image(settings, "p", b"x")
        assert not isinstance(exc_info.value, OllamaTimeout)


class TestOcrPage:
    def test_finished_page_is_ocr_with_normalized_text(self, settings, monkeypatch):
        monkeypatch.setattr(
            ocr_module, "generate_with_image",
            lambda *a, **k: VisionResult(text=r"$\frac{1}{2}$ cup  ", seconds=2.0),
        )
        page = ocr_module.ocr_page(settings, _image(), 3)
        assert page == Page(number=3, text="1/2 cup", method=Method.OCR, seconds=2.0)

    def test_capped_page_is_skipped_and_its_junk_discarded(self, settings, monkeypatch):
        monkeypatch.setattr(
            ocr_module, "generate_with_image",
            lambda *a, **k: VisionResult(text="PLATO " * 500, seconds=660.0, complete=False),
        )
        page = ocr_module.ocr_page(settings, _image(), 10)
        assert page.method is Method.SKIPPED
        assert page.text == ""
        assert "3000 tokens" in page.reason

    def test_timed_out_page_is_skipped_not_raised(self, settings, monkeypatch):
        def boom(*a, **k):
            raise OllamaTimeout("exceeded 900s")

        monkeypatch.setattr(ocr_module, "generate_with_image", boom)
        page = ocr_module.ocr_page(settings, _image(), 10)
        assert page.method is Method.SKIPPED
        assert "timed out" in page.reason

    def test_outage_still_propagates(self, settings, monkeypatch):
        def boom(*a, **k):
            raise OllamaError("connection refused")

        monkeypatch.setattr(ocr_module, "generate_with_image", boom)
        with pytest.raises(OllamaError):
            ocr_module.ocr_page(settings, _image(), 1)


def _blank_pdf(path, pages: int) -> None:
    """Image-only stand-in: blank pages have no text layer, so every one goes to OCR."""
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    path.write_bytes(buf.getvalue())


class TestPageCacheAcrossRequeue:
    def test_second_attempt_reuses_finished_pages(self, settings, tmp_path, monkeypatch):
        pdf_path = tmp_path / "scan.pdf"
        _blank_pdf(pdf_path, 3)
        calls: list[int] = []

        def fake_ocr_page(settings, image, number):
            calls.append(number)
            method = Method.SKIPPED if number == 3 else Method.OCR
            return Page(number=number, text=f"page {number}", method=method,
                        reason="OCR runaway, cut at 3000 tokens" if number == 3 else None)

        monkeypatch.setattr(pdf_module, "ocr_page", fake_ocr_page)
        job = Job.new("C1", "1.0", str(pdf_path), "scan.pdf")

        first = extract(settings, str(pdf_path), cache=page_cache(settings, job.id))
        assert calls == [1, 2, 3]

        # Fresh cache object, same job id: what a requeued attempt constructs.
        second = extract(settings, str(pdf_path), cache=page_cache(settings, job.id))
        assert calls == [1, 2, 3], "the retry must not OCR any page again"
        assert second == first
        # The skipped page is cached too — it is precisely the one not to retry.
        assert second.pages[2].method is Method.SKIPPED

    def test_cache_survives_restore_and_dies_with_complete(self, settings):
        job = Job.new("C1", "1.0", "/x.pdf", "x.pdf")
        enqueue(settings, job)
        cache = page_cache(settings, job.id)
        cache.put(Page(number=1, text="t", method=Method.OCR))

        # restore() deletes any *.json it cannot parse as a Job; the cache must not be
        # collateral damage, and it must not be mistaken for a job either.
        assert [j.id for j in restore(settings)] == [job.id]
        assert len(page_cache(settings, job.id)) == 1

        complete(settings, job)
        assert len(page_cache(settings, job.id)) == 0
        assert not cache.path.exists()

    def test_corrupt_cache_means_reocr_not_failure(self, settings):
        job = Job.new("C1", "1.0", "/x.pdf", "x.pdf")
        cache = page_cache(settings, job.id)
        cache.put(Page(number=1, text="t", method=Method.OCR))
        cache.path.write_text("{not json")

        assert len(page_cache(settings, job.id)) == 0


class TestProvenance:
    def test_skipped_pages_say_why(self):
        from scribe.document import Document

        doc = Document(source="s.pdf", kind="pdf", pages=[
            Page(number=1, text="a", method=Method.OCR, seconds=100),
            Page(number=2, text="", method=Method.SKIPPED,
                 reason="OCR runaway, cut at 3000 tokens"),
            Page(number=3, text="", method=Method.SKIPPED, reason="OCR timed out after 900s"),
        ])
        line = _provenance(doc, Summary(title="t", tldr="", summary=""), "m")
        assert (
            "**2 page(s) SKIPPED (OCR runaway, cut at 3000 tokens; "
            "OCR timed out after 900s)**"
        ) in line


class TestRepeatCollapse:
    """scribe#13: the model transcribed a page, its running header, then the page again,
    under the token cap; two minutes of audio played twice."""

    PAGE = "\n".join([
        "Language is a complicated and versatile instrument. People learn to use it in",
        "much the same way as they learn to use other tools, such as automobiles or",
        "kitchen equipment. Youngsters who do much riding with their parents or friends",
        "seldom need formal instruction in driving a car. They acquire their knowledge by",
        "observation and imitation. In the same way, those who spend much time in the",
        "kitchen learn to use complicated kitchen appliances. The case is similar with",
        "language. Certainly in childhood, and for many of us throughout our lives, we",
        "learn the proper use of language by observing and imitating the linguistic",
        "behavior of the people we meet. There are, however, limits to this informal learning.",
    ])

    def test_page_repeated_after_a_running_header_is_cut(self):
        from scribe.extract.ocr import collapse_repetition
        text = self.PAGE + "\n4.1 PURPOSES OF DEFINITION\n" + self.PAGE
        out, cut = collapse_repetition(text)
        assert cut and out == self.PAGE

    def test_page_repeated_without_header_is_cut(self):
        from scribe.extract.ocr import collapse_repetition
        out, cut = collapse_repetition(self.PAGE + "\n" + self.PAGE)
        assert cut and out.strip() == self.PAGE

    def test_clean_page_untouched(self):
        from scribe.extract.ocr import collapse_repetition
        out, cut = collapse_repetition(self.PAGE)
        assert not cut and out == self.PAGE

    def test_legitimate_short_repeats_survive(self):
        from scribe.extract.ocr import collapse_repetition
        tail = ("More different text follows here at length, enough to make the page long "
                "but not repeated in any 160-character window at all.")
        text = "Chapter 4\n" + self.PAGE + "\nChapter 4\n" + tail
        out, cut = collapse_repetition(text)
        assert not cut and out == text
