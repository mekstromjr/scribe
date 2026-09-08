# scribe

> **Read-only mirror.** This project is developed on a self-hosted GitLab instance and
> mirrored to GitHub for visibility. Issues and pull requests are not accepted here;
> the mirror is overwritten on every push. Contact: see the profile of [@mekstromjr](https://github.com/mekstromjr).

Send a link, PDF, or image to Slack; get a brief summary back in thread and a full note in
the Obsidian vault with a thorough summary plus the complete extracted text.

Backed by [`ollama-mini`](https://gitlab.meklab.net/meklab/k8s/-/tree/main/apps/ollama-mini)
(CPU-only Ollama in the `infra` namespace).

**Status: Phase 4 — Slack Socket Mode bot written, not yet verified against live Slack**
(needs the app tokens; see [SLACK_SETUP.md](SLACK_SETUP.md)). Everything below Slack —
extraction, summarization, note rendering, vault publishing — is working and verified.

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

# Extract + summarize + render the vault note (prints it)
scribe note deck.pdf
scribe note https://example.com/article --out-dir ~/notes

# ...and commit it to the Obsidian vault
scribe publish deck.pdf
scribe publish https://example.com/article

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

## Vault publishing

Notes are committed to `mekadmin/mekvault` (project 2) at `+/`, the vault's own inbox
convention for new unsorted notes. Local PDFs and images travel with the note into
`Misc/Files/` and are wikilinked; links do not, since the URL is already in the
frontmatter.

Three deliberate choices:

- **One commit, not two.** The note and its attachment go up together via the commits
  API. Two separate file-API calls could leave the vault with a note whose wikilink
  points at a file that never committed — a dangling link in a repo that syncs to every
  device.
- **The attachment path resolves *before* rendering.** The note has to link the
  attachment, but its final path is only known after collision resolution. Rendering
  first would mean guessing the name.
- **`unique_path` never overwrites.** A second note about the same source is a new note;
  silently replacing one you may have edited since is data loss. Collisions append
  ` (2)`, ` (3)`, and give up after 99 rather than looping.

Needs `SCRIBE_GITLAB_TOKEN` — a project access token on the vault with
`write_repository`. The vault syncs to Obsidian over iCloud with Obsidian Git as version
control; the cluster cannot write iCloud, so scribe commits and the plugin pulls.

**If notes stop appearing, check that Obsidian Git can still push.** It commits locally
first, so a broken remote leaves the repo looking healthy while the server silently falls
behind — that failure went unnoticed for three months.

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
| `SCRIBE_GITLAB_TOKEN` | *(none)* | Project access token, `write_repository` on the vault |
| `SCRIBE_VAULT_PROJECT_ID` | `2` | `mekadmin/mekvault` |
| `SCRIBE_VAULT_NOTES_DIR` | `+` | The vault's inbox convention |
| `SCRIBE_VAULT_FILES_DIR` | `Misc/Files` | Where attachments land |
| `SCRIBE_MAX_ATTACHMENT_BYTES` | `10485760` | Above this the note publishes without the source |
| `SCRIBE_SLACK_BOT_TOKEN` | *(none)* | `xoxb-` from installing the app |
| `SCRIBE_SLACK_APP_TOKEN` | *(none)* | `xapp-` with `connections:write`, for Socket Mode |

## Roadmap

- **Phase 5** — deploy to `infra`, Standard tier, Deployment only (no Service/Ingress)
