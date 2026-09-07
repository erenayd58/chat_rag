# Two stages so the compiler toolchain and the git client needed to build the
# dependencies never ship in the image that runs. Everything installs into one
# virtualenv, which is the only thing copied forward.
FROM python:3.11-slim-bookworm AS builder

# build-essential: for any dependency without a manylinux wheel.
# git: requirements.txt pins amsc-poc to a GitHub commit.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# sentence-transformers pulls in torch, whose default wheel carries several
# gigabytes of CUDA libraries. Nothing here uses a GPU, so the CPU build is
# installed first and the dependency resolver then finds it already satisfied.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt ./
RUN pip install -r requirements.txt


FROM python:3.11-slim-bookworm

# The image does not ship .git, so the commit it was built from is stamped
# here instead. Left empty it simply reports as unknown, which is what a
# configuration snapshot should say rather than guessing.
ARG CHAT_RAG_GIT_SHA=""
ENV CHAT_RAG_GIT_SHA=${CHAT_RAG_GIT_SHA}

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Turkish filenames and chunk text pass through stdout and the log files.
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    # All runtime state under one mountable directory, and only this: the
    # vector stores, the ledger, the knowledge base records, the parser's
    # canonical-unit cache and the logs are all derived from it by
    # config/paths.py. STRUCTURED_PARSER_CACHE used to be set here as well,
    # which pinned the cache to /data image-wide -- so the build-time smoke
    # checks below, which move the data directory, could not move that one.
    # config/paths.py leaves every path exactly as it was when this is unset,
    # which is how a local checkout keeps writing to its own files.
    CHAT_RAG_DATA_DIR=/data

COPY --from=builder /opt/venv /opt/venv

# Runs as a normal user. /data is created here so a named volume inherits the
# right ownership; a bind mount takes the host's, which Docker Desktop makes
# writable.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app /data \
 && chown -R app:app /app /data

WORKDIR /app
COPY --chown=app:app . .
USER app

# Two build-time checks, so a broken image fails here rather than at the first
# request. The import smoke fails when requirements.txt pins an
# amsc revision the product code has outgrown. This is invisible in a developer
# checkout, where amsc is an editable install of the sibling chunk repository,
# and it is exactly how a clean image came to build and then not start.
# The serve smoke then starts the production server the CMD below starts,
# answers /api/health over a real socket and stops it -- which is what tells
# the difference between 'the modules import' and 'the container serves'.
# The data directory is overridden for both steps: /data is the volume mount
# point, and a build must not leave a vector store in that layer.
RUN CHAT_RAG_DATA_DIR=/tmp/import-smoke python tools/import_smoke.py \
 && CHAT_RAG_DATA_DIR=/tmp/serve-smoke python tools/serve_smoke.py \
 && rm -rf /tmp/import-smoke /tmp/serve-smoke

EXPOSE 5005

# Deliberately cheap: it answers from the Flask app itself and touches no
# model, parser or vector store. Python is already here, so no curl is added
# just to call one URL.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5005/api/health', timeout=4).status == 200 else 1)"

# The production entrypoint: waitress, one process, WAITRESS_THREADS request
# threads. Not `python app.py` -- that is the development server, and it must
# not be what a deployment reaches by default.
CMD ["python", "-m", "wsgi"]
