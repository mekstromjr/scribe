# Design notes

Longer explanations of how scribe behaves, moved out of the README. Measurements behind
the defaults are in [measurements.md](measurements.md).

## Two workers, two stages

Summarizing is Ollama-bound and the CPU Ollama host serves one request at a time, so
there is one summarize worker: concurrent documents would not finish sooner, they would
thrash a shared bottleneck. Audio is Kokoro-bound, a different server, so it has its own
worker. The moment a document's note is posted, its audio half is written to the spool as
an `AudioJob` and the summarize worker moves on. The next document's TL;DR lands while the
last one's m4b is still being made.

- **The hand-off is through the spool, not memory.** An audio job that was promised
  survives a restart; on start-up audio jobs resume first, silently, because their
  threads already have a note.
- **The sender's TTS settings are frozen in the record.** A voice change typed while a job
  waits for audio applies to the next document, not the one already acknowledged.
- **Cancel still finds it.** Both halves register under the same id, so "cancel" in the
  thread drops a waiting audio job, or stops a running one at its next segment.
- **Failures are classified.** Transient failures (the model server restarting, a network
  blip) are retried up to `SCRIBE_MAX_ATTEMPTS`; permanent ones (a dead link, an
  unsupported file type) are not.

## Slack

DM the bot, or @-mention it, with a link or an attached PDF or image. It acknowledges in
thread immediately with an ETA, then replies with the TL;DR and the note file when the
work is done. A document takes minutes, so the immediate ack is not cosmetic.

**Socket Mode, not the Events API.** The cluster has no public ingress, so scribe opens an
outbound WebSocket to Slack instead. A consequence: scribe needs no Service, no Ingress
and no certificate, because nothing ever connects to it.

The Slack CLI is not usable for setup on a free workspace (`slack login` returns "This
workspace is not eligible for the next generation Slack platform"). That gate is about
Slack's hosted platform; Socket Mode and bot tokens are free.
[SLACK_SETUP.md](../SLACK_SETUP.md) covers the web-UI path using `manifest.json`.

Scopes are minimal: `im:history` plus `app_mentions:read` means scribe sees only DMs sent
to it and messages that @-mention it. Channels listed in `SCRIBE_DROP_CHANNELS` process
every link or file without a mention; everywhere else stays mention-only.

## Runtime settings are per person

`/scribevoice`, `/scribeformat`, `/scribecover` and `/scribetoggletts` change the settings
of whoever runs them, keyed by Slack user id, and each document is processed under its
sender's settings. Append `default` to a command to change the shared layer everyone
falls back to instead; `/scribeconfig` shows your settings, the shared defaults, and which
keys you have overridden.

Resolution order, narrowest first: your overrides, the shared overrides, environment,
code default. Overrides live in `config.json` beside the spool and hold only keys that
were explicitly set, so a change to an untouched key's default still takes effect.

`/scribevoice` with no argument links to a "voice samples" item on the Audiobookshelf
shelf, one chapter per voice reading the same passage, then lists the voice ids grouped by
language and gender. The item is built by `scribe voice-samples`.

## Whose file is this

Every m4b on the shelf is filed under its sender: the item joins a collection named with
the person's first name (from their Slack profile) and gets a matching tag, the voice id
becomes the narrator, and the source URL or filename the description. A collection rather
than a series, because a series is one ordered work and a collection is a named shelf of
unrelated items. Not the `grouping` tag either: Audiobookshelf reads that as a fallback
series tag. Older items were back-filled with `scribe abs-backfill`.

## The listening script is rules, measured

The text sent to Kokoro is produced by deterministic rules, not a model. An experiment on
six real sources (`experiments/tts_cleanup/`) compared the rules with an LLM repair prompt,
alone and on top of the rules, and a speech-framed rewrite of that prompt. The rules had
the fewest artifacts on every source; every LLM variant added markup or removed content
(a chapter's epigraphs, a paper's prose). So there is no LLM pass.

What the rules do for a PDF text layer, which arrives as printed lines with no paragraph
marks:

- **Reflow** rebuilds paragraphs from line shape: a paragraph ends where a line is short
  for its page and ends a sentence. Soft and hyphenated wraps are joined, a sentence split
  across a page break is healed, and runs of short sentence-less lines (pseudocode, figure
  labels) are dropped. On a 62-page chapter, 33 of 57 segment boundaries fell
  mid-sentence before, 2 after.
- **Missing glyphs** (U+FFFE and private-use codepoints, where LaTeX PDFs put ligatures,
  footnote superscripts and minus signs) become the ligature a small lexicon says fits
  inside a word, a space between a word and a capital, and nothing elsewhere.
- **Headings** are judged with detached diacritics removed, must contain a real word, and
  may not look like an author line, a table header or figure text. The chapter list on one
  textbook went from 3 entries to 21.
- **Sentence splitting**, the fallback for an oversized paragraph, does not break on
  abbreviations, initials or bare list markers.

`scribe listen --dry-run <file>` prints the script and its lint findings without
synthesizing anything. A finding means a rule is missing.

## The estimate learns

The ack quotes two times: when the summary will be in the thread, and when the audio will.
Each is a per-stage model in things known at ack time (characters, OCR pages, which
summarize branch) times a learned correction factor per stage. Every clean completion
feeds `log(actual / predicted)` into an exponentially weighted mean and spread for that
stage; quotes use the upper side of the spread, so acks land early more often than late.
The `SCRIBE_ETA_*` constants are only the priors a fresh spool starts from. Learned state
lives in the `calibration` section of `config.json` beside the spool.

- **Clean completions only.** Retries, cancels and OCR runs that skipped or capped pages
  teach nothing; one absurd job is clamped so it cannot poison the next ten estimates.
- **The in-flight job counts at its remaining time**, not its full estimate, when a queue
  wait is quoted.
- **The audio quote is re-issued** in the TL;DR reply, from the real listening-script
  length and the audio queue at that moment.
- **Audit trail in the logs.** Every ack logs `eta.ack ...`, every stage `eta.actual ...`,
  every update `eta.learn ...`. `scribe calibration` (or the tail of `/scribeconfig`)
  shows the current factors.

## Titles are a ladder

A file's name is rarely its title (`03-dynprog.pdf` is a chapter on dynamic programming),
so the note, the Slack file and the Audiobookshelf entry are named by the first rung that
holds: the PDF's own metadata Title when it is real; then the model's title, which the
prompt asks to name the larger work and chapter when the text makes that identifiable;
then the filename stem. "Real" excludes filenames in disguise: slugs, office-suite export
prefixes, placeholders like "untitled". A link keeps the page's own title.

## Long documents

The summarizer's context is `SCRIBE_CONTEXT_TOKENS` (must match the server's
`OLLAMA_CONTEXT_LENGTH`; Ollama truncates over-length input silently). A document that
fits is summarized in one call. One that would otherwise be truncated goes through a
map-reduce fallback over `SCRIBE_CHUNK_CHARS` chunks instead, so the server never cuts
input silently.

## Cover art

Every audiobook gets a cover drawn for it: a 512x512 image from a
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `sd-server` running
SD-Turbo, with the scribe badge in the corner. It is embedded in the m4b, and
Audiobookshelf's scanner takes an embedded cover for any new item that has none.

- **The subject comes from the document.** At hand-off, one short text-model call reads
  the finished summary and describes a single concrete scene, steering toward objects and
  places and away from lettering, because SD-Turbo cannot spell and draws people badly.
  The scene is deliberately not a field of the summary request: adding one there shifted
  the summaries themselves (one article's summary came out a third shorter), and the
  summary is the product.
- **The style is a per-person setting**, like the voice: `flat`, `storybook`, `painterly`,
  `watercolor`, `woodcut`, `poster`, `random`, `off`, or `auto` (the default), where the
  same call also picks the style that suits the document.
- **Everything is decided at hand-off** and frozen into the audio spool record. The audio
  worker never calls the text model, and the seed is derived from the job id, so a job
  resumed after a restart draws the same image.
- **It costs no wall clock on a normal document.** The image is generated on a separate
  machine while Kokoro synthesizes, and the m4b waits for it only at packaging time. Time
  spent waiting is excluded from the audio calibration.
- **Best-effort.** A dead image server, a timeout or a bad response costs the cover, not
  the audiobook.
- **Lettering is removed by rule, not just asked for.** The negative prompt is inert at
  SD-Turbo's cfg 1.0, and on abstract topics the text model still asks for "signs labeled
  ..." now and then. `cover.clean_scene` strips quoted strings, label phrases and lettered
  props from the scene before the image call.
- **Older items:** `scribe cover-backfill` draws a cover for every shelf item that has
  none and uploads it through the Audiobookshelf API (`--dry-run`, `--force`, `--title`).
- **The badge is rebuilt, not cropped:** `scripts/make_badge.py` extracts the emblem from
  `assets/scribe-logo-minimalist.jpg` by colour and draws it on a fresh disk.

## Note delivery

The full note is posted into the Slack thread as a file, right after the TL;DR reply, in
whatever format `/scribeformat` selects (`pdf` by default; `md`, `docx`, or `none`).
Nothing is written anywhere else: a note worth keeping is downloaded and filed by hand, so
several people can use one bot without filling anyone's notes.

- **`md` is the renderer's output verbatim**, frontmatter, tags and Obsidian-style callout
  included, so it drops into a Markdown vault unchanged.
- **`pdf` and `docx` are renderings of that same Markdown** with the Obsidian-only syntax
  made portable. pandoc produces the docx directly and the HTML for the PDF; weasyprint
  lays the PDF out. No TeX in the image.
- **Export is best-effort, like audio.** By the time it runs the summary is already in the
  thread, so a pandoc or upload failure costs the file and posts a one-line apology; it
  never fails or requeues the job.

## Shared code with recipe-pipeline

`src/scribe/ollama.py` and the PDF rendering in `src/scribe/extract/pdf.py` started as
copies of the equivalents in
[recipe-pipeline](https://github.com/mekstromjr/recipe-pipeline). They were copied, not
shared as a library, to keep the two deployable independently.
