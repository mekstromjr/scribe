# Measured behaviour

Numbers behind the defaults in `src/scribe/config.py`. They were measured on the CPU-only
Ollama host scribe runs against; on different hardware, re-measure before trusting them.

## Text layer before OCR

Lecture slides exported from PowerPoint, Keynote or LaTeX carry a selectable text layer, and
so does every article. Reading that layer is lossless and effectively instant. Running the
vision model over those pages instead is both slower and worse: vision models paraphrase
and silently drop content, while PDFium returns exactly what is embedded.

So scribe reads the text layer per page and falls back to `glm-ocr` only for pages that
yield no meaningful text (`SCRIBE_MIN_PAGE_CHARS`, default 24 non-whitespace characters;
a title-only slide can be about 20).

| Input | Pages | Path | Time |
|---|---|---|---|
| Sheet-music PDF (text layer) | 39 | text layer | 0.53 s, 0 OCR calls |
| Scanned recipe card (no text layer) | 1 | OCR fallback | ~40 s |
| Wikipedia article | 1 | trafilatura | ~1 s |

The same 39-page PDF forced through OCR would take roughly 15-25 minutes.

## OCR

- **Latency is variable, roughly 20-60 s per page.** The same page at the same DPI measured
  20.6 s and 40.3 s on different runs; it moves with node contention. Treat it as a range.
- **`glm-ocr` can loop forever on a sparse page.** A book's title page (mostly whitespace
  and a scan gutter) ran to 14,000+ tokens and was still going when the 16k context
  filled; dense pages from the same scan produced 739-1,475. Ollama's default
  `num_predict` is unlimited, so scribe caps OCR generation (`SCRIBE_OCR_NUM_PREDICT`,
  about 2x the densest measured page) and gives OCR its own timeout. A page that hits
  either is recorded as SKIPPED with the reason, and the rest of the document proceeds.
  Both signals are deterministic at temperature 0, so requeueing would only reproduce
  them. Finished OCR pages are cached in the spool so a genuine requeue resumes instead
  of redoing them.
- **Render DPI is the main OCR cost lever.** 200 DPI cost about 50% more wall clock than
  150 for a byte-identical transcription, so the default is 150.
- **`glm-ocr` reports zero for every timing field** it returns (`load_duration`,
  `eval_count`, `total_duration`). Wall clock is measured client-side; trusting the
  API's own numbers yields a confident, wrong `0.0s`.
- **`glm-ocr` emits LaTeX for typographic glyphs.** A printed `½` comes back as
  `$\frac{1}{2}$`. `normalize_latex()` undoes this for short spans while leaving genuine
  equations intact.

## `num_thread` must match the CPU allocation

llama.cpp sizes its thread pool from the host core count and ignores the cgroup quota,
so on a 10-CPU node with a 6-CPU limit it oversubscribes and the threads fight CFS
throttling. Measured on `qwen3.5:4b`:

| Config | Prompt eval | Generation |
|---|---|---|
| default (host count) | 26.6-29.3 t/s | 3.9-4.1 t/s |
| `num_thread=5` | 26.8 t/s | 6.7 t/s |
| `num_thread=6` | 43.3 t/s | 6.9 t/s |

There is no server-side setting for this; it is a per-request model option, so every
client must send it (`SCRIBE_NUM_THREAD`). Low measured CPU (4515m against a 6000m limit)
was evidence for contention, not against it: a throttled thread is not burning CPU while
it waits.

## Fetching web pages

Measured against Wikipedia, a spoofed browser User-Agent gets `403` with a bot-policy
notice, while `scribe/0.1 (...; contact)` gets `200`. Wikimedia's robot policy asks
automated clients to declare themselves; complying works better than evading.

## Cover art

One cover measured 143 s end to end at the production settings (SD-Turbo, 4 steps, full
VAE on CPU). Four steps measured cleaner than one and cost about 1.5 min more per cover.

## Text-to-speech

Production Kokoro measured about 100 s per 3,000-character segment (about 1.6x realtime).
A Kokoro voice pack is about 523 KB, so voice choice has no meaningful memory cost.
