# scribe

Send a link, PDF, or image to Slack; get a brief summary back in thread and a full note in
the Obsidian vault with a thorough summary plus the complete extracted text.

Backed by [`ollama-mini`](https://gitlab.meklab.net/meklab/k8s/-/tree/main/apps/ollama-mini)
(CPU-only Ollama in the `infra` namespace).

**Status: Phase 2 — extraction, summarization, and note rendering. CLI only.** No Slack
 and no vault writes yet.

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

# Extract + summarize + render the vault note
scribe note deck.pdf
scribe note https://example.com/article --out-dir ~/notes
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

## Roadmap

- **Phase 3** — publish to `mekadmin/mekvault` via the GitLab API
- **Phase 4** — Slack Socket Mode (outbound WebSocket; Slack cannot reach the tailnet)
- **Phase 5** — deploy to `infra`, Standard tier, Deployment only (no Service/Ingress)
