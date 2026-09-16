# scribe: Slack Socket Mode bot. No ports are exposed — the connection to Slack is
# OUTBOUND, so nothing ever connects to this container.
FROM python:3.12.11-slim-bookworm

# ffmpeg: concatenates Kokoro's WAV segments and encodes the single AAC .m4b with
# chapter markers (package.py).
# pandoc: renders the note markdown to docx, and to HTML for the PDF (note_export.py).
# libpango/libcairo and friends: weasyprint's layout engine, which turns that HTML into
# the PDF. No TeX -- a pandoc PDF via LaTeX would triple the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg pandoc \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 libcairo2 libgdk-pixbuf-2.0-0 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Manifests first so the third-party dependency layer caches across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the source.
COPY src/ src/
COPY README.md ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app

# The spool lives on a PVC mounted here. It must survive restarts or queued jobs are
# lost — the whole point of the durable queue.
ENV SCRIBE_SPOOL_DIR=/data/queue
RUN mkdir -p /data/queue && chown -R app:app /data

USER app
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["scribe"]
CMD ["serve"]
