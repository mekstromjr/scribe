# scribe

> **Read-only mirror.** This project is developed on a self-hosted GitLab instance and
> mirrored to GitHub for visibility. Issues and pull requests are not accepted here;
> the mirror is overwritten on every push. Contact: see the profile of [@mekstromjr](https://github.com/mekstromjr).

Send a link, PDF, or image to Slack; get a brief summary back in thread, followed by the
full note as a file (PDF by default, or Markdown, Word, or nothing) with a thorough summary
plus the complete extracted text.

Backed by [`ollama-mini`](https://gitlab.meklab.net/meklab/k8s/-/tree/main/apps/ollama-mini)
(CPU-only Ollama in the `infra` namespace).

**Status: Phase 4 — Slack Socket Mode bot written, not yet verified against live Slack**
(needs the app tokens; see [SLACK_SETUP.md](SLACK_SETUP.md)). Everything below Slack —
extraction, summarization, note rendering, note export — is working and verified.

## The core design decision: extract before OCR

Lecture slides exported from PowerPoint/Keynote/LaTeX carry a selectable text layer, and so
does every article. Reading that layer is lossless and effectively instant. Running the
vision model over those pages instead is both slower *and worse* — vision models paraphrase
and silently drop content, while PDFium returns exactly what is embedded.

So scribe reads the text layer per page and falls back to `glm-ocr` **only** for pages that
yield no meaningful text. Measured on real files:

| Input | Pages | Path | Time |
|---|---|---|---|
| Sheet-music PDF (text layer) | 39 | text-layer | **0.53s**, 0 OCR calls |
| Scanned recipe card (no text layer) | 1 | OCR fallback | ~40s |
| Wikipedia article | 1 | trafilatura | ~1s |

The same 39-page PDF forced through OCR would take roughly 15-25 minutes. OCR is the
exception, not the default.

## Usage

```bash
uv sync

# Which Ollama am I actually talking to? (see the port warning below)
scribe health

# Classify a PDF without spending any CPU
scribe probe deck.pdf

# Extract; --per-page shows the method and timing for each page
scribe extract deck.pdf --per-page
scribe extract https://example.com/article
scribe extract scan.png

# Extract + summarize + render the note markdown (prints it)
scribe note deck.pdf
scribe note https://example.com/article --out-dir ~/notes

# ...or write it as a file, the way the bot posts it (pdf | md | docx)
scribe export --format pdf deck.pdf
scribe export --format docx https://example.com/article --out-dir ~/Downloads

# Run the Slack bot (blocks; needs the two Slack tokens)
scribe serve
```

### Running against the cluster from a laptop

`ollama-mini` is ClusterIP-only (Ollama ships no authentication, so it has no Ingress):

```bash
kubectl -n infra port-forward svc/ollama-mini 11500:11434
SCRIBE_OLLAMA_HOST=http://127.0.0.1:11500 scribe health
```

**Do not use local port 11434.** The dev Mac runs its own native ollama there. A
`port-forward` to 11434 binds IPv6 `[::1]` while the native instance holds IPv4
`127.0.0.1`, so `localhost:11434` resolves to either one unpredictably — requests silently
go to the wrong server. `scribe health` exists to catch exactly this: it prints the host and
the models it serves, and fails if the expected models are missing.

## Measured behaviour worth knowing

- **OCR latency is variable, roughly 20-60s per page.** The same page at the same DPI
  measured 20.6s and 40.3s on different runs — it moves with node contention. Treat it as a
  range, not a constant.
- **`glm-ocr` can loop forever on a sparse page.** A book's title page (mostly whitespace
  and a scan gutter) ran to 14,000+ tokens and was still going when the 16k context filled;
  dense pages from the same scan produced 739-1,475. Ollama's default `num_predict` is
  unlimited, so scribe caps OCR generation (`SCRIBE_OCR_NUM_PREDICT`, 2x the densest measured
  page) and gives OCR its own timeout. A page that hits either is recorded as SKIPPED with the
  reason, and the rest of the document proceeds — both signals are deterministic, so
  requeueing would only reproduce them. Finished OCR pages are cached in the spool so a
  genuine requeue resumes instead of redoing them (scribe#4).
- **Render DPI is the main OCR cost lever.** 200 DPI cost ~50% more wall clock than 150 for
  a byte-identical transcription, so the default is 150.
- **`glm-ocr` reports zero for every timing field** it returns (`load_duration`,
  `eval_count`, `total_duration`). Wall clock is measured client-side; trusting the API's
  own numbers yields a confident, wrong `0.0s`.
- **`glm-ocr` emits LaTeX for typographic glyphs** — a printed `½` comes back as
  `$\frac{1}{2}$`, which Obsidian would render as MathJax. `normalize_latex()` undoes this
  for short spans while leaving genuine equations intact.
- **Set `num_thread` to the container CPU limit — this is the single biggest
  lever.** llama.cpp sizes its thread pool from *host* core count and ignores the
  cgroup quota, so on a 10-CPU node with a 6-CPU limit it oversubscribes and the
  threads fight CFS throttling. Measured on `qwen3.5:4b`:

  | Config | Prompt eval | Generation |
  |---|---|---|
  | default (host count) | 26.6-29.3 t/s | **3.9-4.1 t/s** |
  | `num_thread=5` | 26.8 t/s | 6.7 t/s |
  | `num_thread=6` | **43.3 t/s** | **6.9 t/s** |

  There is no server-side env var for this — it is a per-request model option, so
  **every client must send it**. Counter-intuitively, low measured CPU (4515m
  against a 6000m limit) is evidence *for* contention, not against it: a
  throttled thread is not burning CPU while it waits.
- **Identify honestly in the User-Agent.** Measured against Wikipedia, a spoofed browser
  string gets `403` with a bot-policy notice, while `scribe/0.1 (...; contact)` gets `200`.
  Wikimedia's robot policy asks automated clients to declare themselves; complying works
  better than evading.

## Runtime settings are per person

`/scribevoice`, `/scribeformat` and `/scribetoggletts` change the settings of whoever
runs them, keyed by Slack user id, and each document is processed under its sender's
settings. Nobody has to fight over the voice. Append `default` to a command to change
the shared layer everyone falls back to instead; `/scribeconfig` shows your settings,
the shared defaults, and which keys you have overridden.

Resolution order, narrowest first: your overrides, the shared overrides, environment,
code default. Overrides live in `config.json` beside the spool and hold only keys that
were explicitly set, so a manifest change to an untouched key still takes effect.

Voice choice is free. A Kokoro voice pack is about half a megabyte and loads only when a
document is synthesized with it, so auditioning a dozen voices costs nothing; the memory
that server uses is the model and its per-request leak, not voices.

## The listening script is rules, measured

The text sent to Kokoro is produced by deterministic rules, not a model. An experiment
on six real sources (2026-09-16, `experiments/tts_cleanup/`) compared the rules with the
legacy LLM repair prompt, alone and on top of the rules, and a speech-framed rewrite of
that prompt. The rules had the fewest artifacts on every source; every LLM variant added
markup or removed content (a chapter's epigraphs, a paper's prose). So there is no LLM
pass, and the artifact classes only a model could fix are recorded on the issue as the
spec for a targeted span-repair if feedback ever calls for one.

What the rules do for a PDF text layer, which arrives as printed lines with no paragraph
marks:

- **Reflow** rebuilds paragraphs from line shape: a paragraph ends where a line is short
  for its page and ends a sentence. Soft wraps and hyphenated wraps are joined, a sentence
  split across a page break is healed, and runs of short sentence-less lines, which is
  what pseudocode and figure labels look like, are dropped. Measured on a 62-page chapter:
  33 of 57 segment boundaries fell mid-sentence before, 2 after.
- **Missing glyphs** (U+FFFE and private-use codepoints, where LaTeX PDFs put ligatures,
  footnote superscripts and minus signs) become the ligature a small lexicon says fits
  inside a word, a space between a word and a capital, and nothing elsewhere.
- **Headings** are judged with detached diacritics removed, must contain a real word,
  and may not look like an author line, a table header or figure text. The chapter list
  on that textbook went from 3 entries to 21.
- **Sentence splitting**, the fallback for an oversized paragraph, no longer breaks on
  abbreviations, initials or bare list markers.

`scribe listen --dry-run <file>` prints the script and its lint findings without
synthesizing anything. A finding means a rule is missing.

## The estimate learns

The ack quotes two times: when the summary will be in the thread, and when the audio
will. Each is a per-stage model in things known at ack time (characters, OCR pages,
which summarize branch) times a **learned correction factor** per stage. Every clean
completion feeds back `log(actual / predicted)` into an exponentially weighted mean and
spread for that stage; quotes use the upper side of the spread, so acks land early more
often than late. The `SCRIBE_ETA_*` constants are only the priors a fresh spool starts
from. Learned state lives in the `calibration` section of `config.json` beside the spool.

- **Clean completions only.** Retries, cancels and OCR runs that skipped or capped pages
  teach nothing; one absurd job is clamped so it cannot poison the next ten estimates.
- **The in-flight job counts at its remaining time**, not its full estimate, when a
  queue wait is quoted.
- **The audio quote is re-issued exactly** in the TL;DR reply, from the real listening
  script length and the audio queue at that moment.
- **Audit trail in Loki.** Every ack logs `eta.ack ...`, every stage logs `eta.actual ...`,
  every update logs `eta.learn ...`. To see estimate vs actual per stage:

  ```
  {namespace="infra", container="scribe"} |~ "eta\\.(ack|actual|learn)"
  ```

  `scribe calibration` (or the tail of `/scribeconfig`) shows the current factors.

## Titles are a ladder, and the filename is the bottom rung

A file's name is rarely its title (`03-dynprog.pdf` is a chapter on dynamic programming),
so the note, the Slack file and the Audiobookshelf entry are named by the first rung that
holds: the PDF's own metadata Title when it is real; then the model's title, which the
prompt asks to name the larger work and chapter when the text makes that identifiable;
then the filename stem. "Real" excludes filenames in disguise: slugs, office-suite export
prefixes, placeholders like "untitled". A link keeps the page's own title, as before.

## Two workers, two stages

Summarizing is Ollama-bound and Ollama serves one request at a time, so there is one
summarize worker. Audio is Kokoro-bound, a different server, so it has its own worker:
the moment a document's note is posted, its audio half is written to the spool as an
`AudioJob` and the summarize worker moves on. The next document's TL;DR lands while the
last one's m4b is still being made.

- **The hand-off is through the spool, not memory.** An audio job that was promised
  survives a restart; on start-up audio jobs resume first, silently, because their
  threads already have a note.
- **The sender's TTS settings are frozen in the record.** A voice change typed while a
  job waits for audio applies to the next document, not the one already acknowledged.
- **Cancel still finds it.** Both halves register under the same id, so "cancel" in the
  thread drops a waiting audio job, or stops a running one at its next segment.

## Note delivery

The full note is posted into the Slack thread as a file, right after the TL;DR reply, in
whatever format `/scribeformat` selects (`pdf` by default; `md`, `docx`, or `none`).
Nothing is written anywhere else: a note worth keeping is downloaded and filed by hand,
so several people can use one bot without filling anyone's vault.

- **`md` is the renderer's output verbatim**, frontmatter, tags and Obsidian callout
  included, so it drops into a vault unchanged.
- **`pdf` and `docx` are renderings of that same markdown** with the Obsidian-only
  syntax made portable: the frontmatter block goes, the collapsed callout becomes a
  plain blockquote with a bold lead. pandoc produces the docx directly and the HTML for
  the PDF; weasyprint lays the PDF out. No TeX in the image.
- **Export is best-effort, like audio.** By the time it runs the summary is already in
  the thread, so a pandoc or upload failure costs the file and posts a one-line apology;
  it never fails or requeues the job.

## Slack

DM the bot, or @-mention it, with a link or an attached PDF/image. It acknowledges in
thread immediately, then replies with the TL;DR and the note name when the work is done.

**Socket Mode, not the Events API** — Slack cannot reach this network (the cluster is
behind Tailscale CGNAT with no public ingress), so scribe opens an *outbound* WebSocket
instead. A useful consequence: scribe needs no Service, no Ingress and no certificate,
because nothing ever connects *to* it.

The Slack CLI is **not usable** for setup here — `slack login` returns "This workspace is
not eligible for the next generation Slack platform." That gate is about Slack's
Deno-hosted platform, which needs a paid plan; Socket Mode and bot tokens are free.
[SLACK_SETUP.md](SLACK_SETUP.md) covers the web-UI path using `manifest.json`.

Two design points:

- **Work is serialized behind one worker.** ollama-mini runs `OLLAMA_NUM_PARALLEL=1` on
  CPU, so concurrent documents would not finish sooner — they would thrash a shared
  bottleneck and slow everything. The ack says how many jobs are ahead of you.
- **The immediate ack is not cosmetic.** A document takes minutes; without it there is no
  signal anything is happening.

Scopes are minimal by design: `im:history` plus `app_mentions:read` means scribe sees only
DMs sent to it and messages that @-mention it. There is no `channels:history`, so it
cannot read channel traffic it was not addressed in.

## Reused from recipe-pipeline

`src/scribe/ollama.py` and the PDF rendering in `src/scribe/extract/pdf.py` are adapted from
`recipe-pipeline/worker/src/recipe_pipeline/`. **Deliberately copied, not shared** — that
pipeline is live, and refactoring it into a library was out of scope. If either diverges
meaningfully, revisit.

Note the two solve different problems: recipe-pipeline OCRs handwritten cursive with Apple
Vision and never reads a text layer (its inputs never have one). scribe is text-layer-first
because its inputs usually do.

## Configuration

All settings are env-overridable with the `SCRIBE_` prefix (see `src/scribe/config.py`).

| Setting | Default | Notes |
|---|---|---|
| `SCRIBE_OLLAMA_HOST` | `http://ollama-mini.infra.svc.cluster.local:11434` | In-cluster address |
| `SCRIBE_OCR_MODEL` | `glm-ocr:latest` | Vision fallback |
| `SCRIBE_TEXT_MODEL` | `qwen3.5:4b` | Summarization |
| `SCRIBE_MIN_PAGE_CHARS` | `24` | Below this, a page falls through to OCR |
| `SCRIBE_OCR_RENDER_DPI` | `150` | Main OCR cost lever |
| `SCRIBE_NUM_THREAD` | `6` | **Must match the CPU limit in `apps/ollama-mini/deployment.yaml`** |
| `SCRIBE_OCR_MAX_EDGE` | `1500` | Longest-edge cap for vision input |
| `SCRIBE_MAX_OCR_PAGES` | `0` (unlimited) | Skipped pages are recorded, not silently dropped |
| `SCRIBE_OCR_NUM_PREDICT` | `3000` | Generation cap per OCR page; hitting it skips the page (runaway guard) |
| `SCRIBE_OCR_TIMEOUT_SECONDS` | `900` | Per-page OCR deadline, separate from the summarization timeout; must exceed cap / slowest tok/s |
| `SCRIBE_NOTE_FORMAT` | `pdf` | How the full note is delivered in-thread: `pdf`, `md`, `docx`, `none`; `/scribeformat` overrides at runtime |
| `SCRIBE_SLACK_BOT_TOKEN` | *(none)* | `xoxb-` from installing the app |
| `SCRIBE_SLACK_APP_TOKEN` | *(none)* | `xapp-` with `connections:write`, for Socket Mode |

## Roadmap

- **Phase 5** — deploy to `infra`, Standard tier, Deployment only (no Service/Ingress)
