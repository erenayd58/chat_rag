# RAG Console — retrieval-augmented question answering over PDF documents

Upload a PDF into a knowledge base, and ask questions of it. The document is
parsed once into canonical units, chunked by a chosen method, embedded and
indexed; a question retrieves from that index, assembles a labelled context
and gets one answer that cites its sources. The console's **Viewer** screen
shows where each chunking method put its boundaries, on the same documents,
and asks the same question of several of them side by side.

The system is two repositories:

| repo | what it is |
|---|---|
| **`chat_rag`** (this one) | the product — the `/api/v1` contract and the application under it, the Next.js console in [`frontend/`](frontend/README.md), ingest jobs, retrieval, the answer chain, resource limits, observability, configuration |
| **`chunk`** (`amsc-poc`) | the chunking library, installed from a pinned commit — chunking methods, Deep Analysis, the canonical PDF adapter, the shared payload reader and the Viewer's retrieval engine, all research and benchmark code |

`chunk` is expected beside this checkout (`../chunk`).
[docs/architecture.md](docs/architecture.md) is the map: what each repo owns,
the two runtime flows, and which file to open for what.

---

## Documentation

Start here, then follow the question you have:

| doc | answers |
|---|---|
| **[docs/architecture.md](docs/architecture.md)** | What is the system, which repo owns what, where is the code for X |
| **[docs/library-api.md](docs/library-api.md)** | What `pip install chat-rag` promises: the public API, the versioning and deprecation policy, the extras, and the limitations |
| **[docs/operations.md](docs/operations.md)** | How do I run it, what are the limits, and what does *this* 503 mean |
| **[docs/configuration.md](docs/configuration.md)** | Where does a setting come from, who owns it, what wins |
| **[docs/database.md](docs/database.md)** | Where the records live, how to create the schema, what is still a file |
| **[docs/testing.md](docs/testing.md)** | What to run before calling a change done, and the order for cross-repo changes |
| **[docs/limitations.md](docs/limitations.md)** | What this system does not do, and why |
| **[docs/legacy-removal.md](docs/legacy-removal.md)** | Which Flask-era endpoint each `/api/v1` route replaced, what was kept and why |
| **[../chunk/docs/adding-a-chunker.md](../chunk/docs/adding-a-chunker.md)** | How to add a chunking method, end to end |
| **[../chunk/docs/viewer-architecture.md](../chunk/docs/viewer-architecture.md)** | How the Viewer works across both repos, and how to debug a package |
| **[../chunk/docs/library-surface.md](../chunk/docs/library-surface.md)** | What is product, research and legacy in the library, and what the console may import |
| [CHANGELOG.md](CHANGELOG.md) | What changed in the published API, release by release |
| [CHUNK_YONTEMLERI_VE_SORGU_EKRANI.md](CHUNK_YONTEMLERI_VE_SORGU_EKRANI.md) | The four chunking methods and the query screen, in plain Turkish, for a non-technical reader |

---

## First day

Ten steps from a clone to having added a chunking method. Each is explained
further down or in the doc it names.

```bash
# 1. clone both repositories side by side
git clone <chat_rag-url> chat_rag
git clone <chunk-url>    chunk
cd chat_rag

# 2. environment
python -m venv venv                       # Python 3.11–3.13
venv\Scripts\activate                     # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .                # the engine package itself (src/chat_rag)
cp env.example .env                       # PowerShell: Copy-Item env.example .env

# 3. prove the declared source installs and runs (minutes; needs Docker)
python tools/verify_reproducibility.py --local

# 4. start the backend and the console
.\start-demo.ps1                          # or: python -m asgi + npm run dev --prefix frontend
```

Then, in the browser and the terminal:

5. **upload a PDF** — open <http://localhost:3000>, create a knowledge base,
   upload a document, choose **Deep Analysis** if a provider key is configured
   and **Standard** if not. The first parse of a PDF takes minutes; the second
   takes under a second.
6. **ask a question** — `/chat`, pick the knowledge base, ask. The answer
   cites `[S1]`, `[S2]`… back to the chunks it used.
7. **open the Viewer** — <http://localhost:3000/viewer>, pick the document,
   select two methods, and step through the boundaries where they disagree.
   Then ask the same question of both on **Sorgu** and compare what each
   chunking found.
8. **look at the instruments** — `GET /api/v1/health` (three fields: what
   state the service is in, whether it may be sent traffic, and why) and
   `GET /api/ops/metrics` (counters, stage latency, error categories, budgets,
   caches).
   [docs/operations.md](docs/operations.md) reads them for you.
9. **run the tests** — `python -m pytest -q` here, `py -3.11 -m pytest` in
   `../chunk`. [docs/testing.md](docs/testing.md).
10. **add a trivial chunker** — copy `chunk/src/amsc/chunking/example.py`, add
    one `ChunkMethod` to `amsc/chunking/registry.py`, add a test. It appears in the
    upload form, the Viewer and the benchmark with no console change.
    [../chunk/docs/adding-a-chunker.md](../chunk/docs/adding-a-chunker.md).

---

## Local setup

### Prerequisites

| | |
|---|---|
| Python | **3.11–3.13**. The library (`chunk`) declares `>=3.11,<3.14`, and the reproducibility gate builds its clean environment with 3.11. |
| The `chunk` checkout | beside this one (`../chunk`), or named by `CHUNK_REPO` / `start-demo.ps1 -ChunkPath` |
| A provider key | optional. Without one: uploading, parsing, chunking, structural QA and BM25 search all work; generated answers do not. |
| Ollama | optional, for local answers or as the fallback model. It runs on the host, not in this process. |
| Docker | only for the container run and the full reproducibility gate |

### Install

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
cp env.example .env
```

Two installs, because there are two things. `requirements.txt` is the pinned
dependency set; `pip install -e .` is this repository's own engine package,
`chat_rag`, which lives in `src/chat_rag` and is what `interfaces/`, `asgi.py`,
`cli/` and `tools/` import. `--no-deps` keeps the second from re-resolving what
the first just pinned.

`requirements.txt` installs `amsc-poc` from a pinned commit of the `chunk`
repository. For library development install it editable instead, so your
working tree is what the console imports:

```bash
pip install -e ../chunk
```

### Configure

`env.example` is the map: every application default appears as a commented-out
line, and the uncommented lines are the demo profile. Copy it to `.env` and
change what you need. The full precedence rules and who owns which setting are
in [docs/configuration.md](docs/configuration.md).

The minimum for generated answers is one key:

```env
OPENROUTER_API_KEY=<your key>
```

With `.env` as shipped, that key serves all three model roles. With no key at
all the console still runs — see the table above.

### Run it by hand

```bash
python -m asgi                  # uvicorn, one process, every interface
npm run dev --prefix frontend   # the console, on http://localhost:3000
```

The backend serves <http://127.0.0.1:5005>. There is one entrypoint: `python
app.py` and `python -m wsgi` were the Flask console's development and
production servers, and both went with it
([docs/legacy-removal.md](docs/legacy-removal.md)).
[docs/operations.md](docs/operations.md) says what it runs under.

### Common startup failures

| symptom | cause |
|---|---|
| `ValueError` naming an env variable, before the server binds | a configuration value was refused. That is deliberate: bad values stop the process while someone is looking. [docs/configuration.md](docs/configuration.md) |
| `ModuleNotFoundError: amsc` | `pip install -r requirements.txt` did not run, or the pinned commit is unreachable. `python tools/import_smoke.py` says which `amsc` answered |
| the console will not start | it is a Next.js application and needs Node.js 18+ on `PATH`. The launcher runs `npm install` once when `frontend/node_modules` is missing; by hand it is `npm install --prefix frontend` |
| a port is already in use | `start-demo.ps1` recognises a server it already started and refuses a port held by something else. `-ProductPort` / `-ConsolePort` move them |
| the first upload seems to hang | it does not — layout parsing is minutes per document on CPU, and the job is running. Poll `GET /api/v1/ingest-jobs/<job_id>` |
| answers fail but search works | no provider key, or an unreachable gateway. The answer model carries the reason; retrieval never depended on it |

---

## What runs, and where

### The model chain

Three model roles, one OpenRouter key, each role configured on its own:

| Stage | Model (demo) | Configuration | Runs |
|---|---|---|---|
| Deep Analysis / Agentic chunking — proposer + verifier | `qwen/qwen3-30b-a3b-instruct-2507` | `DEEP_ANALYSIS_MODEL`, `DEEP_ANALYSIS_ENDPOINT`, `DEEP_ANALYSIS_API_KEY_ENV`, `DEEP_ANALYSIS_VERIFY` | at upload only, through `amsc.deep.pipeline` |
| Embedding — document chunks and questions, one space | `qwen/qwen3-embedding-8b` | `EMBEDDING_PROVIDER=openrouter`, `EMBEDDING_MODEL`, `EMBEDDING_ENDPOINT`, `EMBEDDING_API_KEY_ENV` | at upload (chunks) and per question (query vector) |
| Answer — reads the assembled context, cites sources | `minimax/minimax-m2.7` | `ANSWER_PROVIDER=openrouter`, `ANSWER_MODEL`, `ANSWER_ENDPOINT`, `ANSWER_API_KEY_ENV` | per question |
| Answer fallback (local, offline) | Ollama `qwen2.5:3b` | `ANSWER_FALLBACK_PROVIDER=ollama`, `ANSWER_FALLBACK_MODEL` | only when the primary answer call fails |

`RETRIEVAL_PROFILE=hybrid_rrf` is the final retrieval profile: the stored
Qwen3 vectors (cosine) and the frozen deterministic BM25 (Turkish diacritic
fold) each rank a candidate pool of 50, reciprocal-rank fusion (k = 60)
merges them with chunk-id tie-breaking, and the top hits are assembled into a
labelled context (`[S1]`, `[S2]`, …; de-duplicated, same-section neighbours
allowed, `CONTEXT_MAX_TOKENS` budget) that the answer model must cite.
Standard and Deep Analysis documents go through exactly this path — only
their chunk partition differs. No ingest-time model runs during a question.

Each knowledge base's vector store carries an `embedding_index.json`
manifest (provider, model, dimension, fingerprint). When the configured
embedding changes, the knowledge base reports **re-index required**: dense
retrieval is switched off (keyword results only, said so in the chat), new
uploads are refused with 409 until the store is rebuilt, and **Settings →
Embedding index → Re-index** re-embeds every stored chunk with the current
model. Vectors from two models are never compared.

Query responses carry the observability the Lab and the chat notices use:
retrieval mode and hit counts (dense / BM25 / fused), selected context and
its token estimate, embedding model and fingerprint, answer provider and
model, whether the fallback answered, and per-stage latency. Prompts and
keys are never stored.

### The chunking modes

Chosen at upload, never at query time.

| Mode | What runs | When the model is unavailable |
|---|---|---|
| Standard | The frozen structure-first walk (`amsc.chunking.structural`). Fast, deterministic, no model. | — |
| Deep Analysis | `amsc.deep.pipeline.chunk_document(mode="deep")`: the same structural walk, a backend LLM **proposer** (one bounded prompt per section that still has a choice), the deterministic **quality selector** (never worse than Standard on any smell type), the double-order **verifier** (a change is kept only when it wins in both orders) and the quality measurement. | The ingest still completes on the deterministic quality contract and the document is labelled with the pipeline status — never passed off as Standard. |

Deep Analysis statuses, as recorded on the document and shown under the
chunking badge: `ok` (quality checks passed), `deterministic` (LLM not
requested), `fallback_no_provider` (model or key not configured),
`fallback_provider_error` (every model call failed), `degraded` (some calls
failed; those sections kept their deterministic result). The document's
**Details** row shows quality before → after, model/verifier usage and the
structural checks (hard token cap, coverage).

Configuration is backend-only (`DEEP_ANALYSIS_MODEL`, `DEEP_ANALYSIS_ENDPOINT`,
`DEEP_ANALYSIS_API_KEY_ENV`, `DEEP_ANALYSIS_VERIFY`, …; see `env.example`).
Only the *name* of the key variable is configured; the key is read at request
time by the provider and never stored, logged or written to provenance. Chat
reads the chunks that were indexed at upload; no ingest model runs during a
question.

An upload also chooses **which chunking methods to analyse the document
with**, independently of the mode it is indexed under: the PDF is parsed once
and every chosen method runs over that one canonical, so the Viewer can
compare them side by side. `GET /api/v1/meta/chunking-methods` says which
methods this machine can run, and why one cannot.

### Default demo profile

The low-cost profile validated on the KKB documents:

```env
CHUNKER_TYPE=structure_first
RETRIEVAL_PROFILE=bm25_only
```

Structure-first chunking lets document structure decide chunk boundaries
(a chunk opens at every heading and section change, oversized units split at
table row / list item / sentence seams) and BM25-only retrieval loads no
embedding model at all: no dense vectors are computed or stored in this
profile. The V4 chunker and the `benchmark_aligned` retrieval profile stay
selectable for comparison against the library's frozen benchmark.

**First upload of a PDF is slow.** Parsing runs layout inference over every
logical page, which is a few seconds per page on CPU -- an 85-page report takes
roughly ten minutes, and essentially all of it is layout model inference rather
than anything in this repository. The resulting canonical units are cached on
disk under `.cache/canonical-units/`, keyed by the PDF content hash, so
re-ingesting the same document afterwards takes well under a second. For a
demo, upload the document once beforehand. Deleting the cache directory is
safe; it is regenerated on the next ingest. Set `STRUCTURED_PARSER_CACHE` to
move it elsewhere.

### LLM Provider Setup

**Option 1: Azure OpenAI (Cloud-based)**
```env
LLM_PROVIDER=azure
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT=gpt-4o
```

**Option 2: Ollama (Local, Free)**
```bash
# Install Ollama first from https://ollama.ai
# Pull a model
ollama pull llama2

# Configure in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

Ollama runs on the host, not in this process. With `ANSWER_PROVIDER`
unset, `LLM_PROVIDER` still selects the answer model, which is why both
names appear in `env.example`.

### The indexing chunker

A knowledge base is created with one **indexing** chunker — what its retrieval
index actually holds. This is a deliberately smaller table than the analysis
methods above (`components/chunker/registry.py`), and an analysis method is
never accepted as one:

| `CHUNKER_TYPE` | what it is |
|---|---|
| `structure_first` | the default: document structure decides the boundaries |
| `v4` | the frozen AMSC V4/A4 implementation, for comparison against the library's benchmark |

`v4` is frozen: the `amsc-poc` revision it runs
against is the one `requirements.txt` pins — that line is the only place the
commit is written, and `tests/unit/test_amsc_pin.py` checks it — and the
integration rejects any V4 config whose semantic hash differs from
`FROZEN_V4_CONFIG_HASH` in `components/chunker/frozen_v4_chunker.py`. V4 does
not accept runtime chunk-size or threshold parameters.

The current parsers return flat text. The normalization adapter therefore maps
blank-line-delimited parser blocks to ordered canonical paragraphs and does not
guess headings, pages, tables, lists, or visuals. If a parser supplies structured
unit metadata, the same adapter preserves those fields directly.

To compare two indexing chunkers, create one knowledge base with each, upload
the same document to both, and run the identical query against each in the
Lab's retrieval experimentation.

The first V4 ingestion may download `intfloat/multilingual-e5-base`; subsequent
boundary embeddings use `.cache/boundary-embeddings`.

### Retrieval profiles

| `RETRIEVAL_PROFILE` | what it is |
|---|---|
| `bm25_only` | the default: the frozen deterministic BM25 alone. No embedding model is loaded and no vectors are stored, so it needs no provider at all |
| `hybrid_rrf` | the final chain: stored dense vectors + BM25, fused by RRF, with a bounded labelled context the answer model must cite |
| `benchmark_aligned` | the frozen Phase 4/5 configuration — multilingual E5 role prefixes, deterministic long-text pooling, Unicode BM25, equal-weight RRF over a 100-result pool at `k=60`, its E5 model loaded `local_files_only` — so a console result can be compared with the library's own benchmark |

Retrieval is deterministic in all three: no query rewriting, no reranking and
no model call before the answer. Indexes are profile-specific because the
embedding models and dimensions differ, so changing profile means a new store
path and a re-ingest.

---

## Demo mode (backend + console)

The proof of concept has two faces and one browser tab. The product is how it
is used; the **Viewer** screen is what the chunking technology does underneath
— where each method put its boundaries, and why. Both are the console, so one
script starts what there is to start.

```powershell
.\start-demo.ps1      # start both, wait until each answers, open the console
.\stop-demo.ps1       # stop what start-demo started
```

| | Address | Server |
|---|---|---|
| Console (Next.js) | http://localhost:3000 | `npm run dev` in [`frontend/`](frontend/README.md) — every screen, including `/viewer` |
| Backend (chat_rag) | http://127.0.0.1:5005 | `venv\Scripts\python.exe -m asgi` (uvicorn, one process; `FLASK_HOST` / `FLASK_PORT`) |

**There is no third process, and no second backend.** The Viewer used to be
one: a server in the `chunk` repository on `:8765` that served its own HTML
page and relayed this console over `/api/demo/*`. Step 12 moved it into the
console as five screens over `/api/v1`, and Step 13 removed the relay it used,
the Flask console beside it and the server itself
([docs/legacy-removal.md](docs/legacy-removal.md)). A demo is the two rows in
the table.

The browser only ever talks to the console's own origin: the console forwards
`/api/v1/*` to the backend itself, so there is no CORS grant and exactly one
place (`CHAT_RAG_API_URL`, which the launcher sets) knows the backend address.
It is read per request, so `-ProductPort` moves the backend without rebuilding
the console.

The launcher checks readiness over HTTP, recognises servers that are already
running instead of starting a second copy, refuses a port held by something
else, writes each server's output to `.demo\logs\` and the started process ids
to `.demo\state.json` (both git-ignored). It runs `npm install` once if
`frontend
ode_modules` is missing. `stop-demo.ps1` stops only processes it can
identify as those servers; `-All` extends that to a backend or console on the
demo ports that it did not start. Nothing from `.env` is printed. Options:
`-NoBrowser`, `-NoInstall`, `-ProductPort`, `-ConsolePort`, `-TimeoutSeconds`.

### What the Viewer screen shows

Five tabs over one document, all of them reading `/api/v1`:

| tab | answers |
|---|---|
| **Genel** | what is in the system, what is ready, where to go next |
| **İncele** | the document itself, with up to three chunking methods printed onto it — the same text in the same row in every column, so a boundary one method draws and another does not is visible on the words. `‹ Fark ›` steps through the disagreements |
| **Sorgu** | one question, several methods at once (`POST /api/v1/analysis-queries`), with each one's sources and how much of the evidence they agreed on |
| **Debug** | where every boundary came from: parser, structural pass, rule layer, model proposal, verification. Only recorded values |
| **Benchmark** | the methods side by side on measurements that were actually taken; a document with no gold set is told so rather than shown a number nobody produced |

Two routes exist for it and no more: `GET /api/v1/documents/<id>/analysis/payload`
(the render model — the canonical units and, per method, the chunks *and the
unit offsets they cut at*) and `POST /api/v1/analysis-queries`. Everything else
the screen needs is what every other screen uses. See
[docs/api-v1.md](docs/api-v1.md).

**An upload is already most of what the Viewer needs.**
The Viewer reads one shape — a packaged Deep Analysis tree pinned to the
canonical it was chunked from — and an ingest already produces every expensive
input that tree needs, so nothing is computed twice:

* the canonical is the one the chunker normalised for *this* ingest, so the
  PDF is never parsed again (a document ingested before this existed has its
  canonical recovered from the parser's own cache instead);
* a **Deep Analysis** upload hands over its whole run — deep rows, Standard
  rows, selection audit, verifier verdicts, proposer audit — so **no second
  proposer or verifier call is ever made**;
* a **Standard** upload has no run to reuse, so the Deep side of the
  comparison is the *deterministic* quality contract (`use_llm=False`): zero
  provider calls, zero cost, and labelled as such rather than passed off as a
  model-backed run.

`components/viewer/analysis.py` does the packaging on a background worker, so
no HTTP call waits on it: an upload records the ingest and returns, and
`POST /api/v1/documents/<id>/analysis` only *queues* a build. Each document's
state — `missing` / `pending` / `running` / `ready` / `failed` — is on its
`analysis` block in `GET /api/v1/documents`, so the Viewer lists a ready
document in its picker, says plainly that one is still being prepared, and
picks up a method that finished while the screen was open.

An upload chooses **which chunking methods to analyse the document with**
(`methods=markdown&methods=structure-only&methods=agentic`, or the older
`deep_analysis=true`). The PDF is parsed once; every chosen method runs over
that one canonical and is packaged as its own arm, so the Viewer can compare
them side by side under a single document. Identity is the file's content
hash: uploading the same PDF again adds variants to the document that is
already there instead of making a second one, and
`POST /api/v1/documents/<document_id>/analysis/methods` adds a method later
without re-reading the file. The *analysis* is shared that way; the *choice* is not.
Each upload record keeps the methods it asked for, and that is what the Viewer
opens it on — an upload that ticked Standard and Hybrid is not shown the
Markdown and Deep Analysis variants another upload of the same file left
behind. The shared analysis keeps all of them, so neither upload costs a
second parse. `GET /api/v1/meta/chunking-methods` says which methods this
machine can actually run, and why one cannot. The methods themselves are defined once,
in the library's registry (`amsc.chunking.registry` in the chunk repository): key,
engine kind, product name, summary and capabilities. `components/viewer/methods.py`
is this console's view of that registry and adds only what the deployment
decides — availability on this machine, display order, the default. Adding a
chunking method is a library change (`chunk/docs/adding-a-chunker.md`: a
method module, one registry entry, a test) followed by
`python tools/promote_chunk_pin.py`, which bumps the `amsc-poc` pin in
`requirements.txt` and checks it; no route, template or script here names
a method. The *indexing* chunker a knowledge base is created with is a
separate, deliberately smaller table (`components/chunker/registry.py`), and an
analysis method is never accepted as one. A build interrupted by a restart
is picked up again from disk. Deleting a document here deletes its analysis;
the chunk repository's frozen benchmark trees are never reachable from this
path. Everything lives under `artifacts/viewer-live/` (git-ignored) and is
regenerable from an ingest.

These documents are a **live workspace category**, not benchmark data. They
have no gold query set, so no Hit@k or MRR is computed for them — the Viewer
says so rather than inventing numbers — and they never enter the frozen
benchmark tables or the cross-document contract table.

Presentation order: **1.** Knowledge Bases — a base and its documents;
**2.** upload a document with **Deep Analysis** and a second method;
**3.** Sohbet — an answer with sources; **4.** Viewer → **İncele** (the
methods side by side on the page) → **Sorgu** (the same question through each
of them) → **Debug** (why each boundary) → **Benchmark**.

---

## Running with Docker

The whole system, from a clean machine, in one command. Three containers: the
Next.js console, the FastAPI application and PostgreSQL with pgvector. No
queue and no model service — Ollama stays on the host, and everything else
this application needs runs in-process.

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine with Compose v2
- Ollama on the host **only if you want generated answers**. Uploading,
  parsing, structure-first chunking, structural QA and BM25 search all work
  with Ollama stopped; generation then returns an explanatory error instead of
  taking the application down.

Nothing else. No Python, no Node.js and no `chunk` checkout: both images build
from this repository and the pinned library commit, which is what
[the reproducibility gate](docs/testing.md) checks on every push.

### Start

```bash
docker compose up --build
```

Then open <http://localhost:3000>. The console lands on Knowledge Bases; Chat
is at `/chat`, search at `/search`, and the chunk-boundary Viewer at `/viewer`.

To have the QA report name the commit the image was built from — the image
does not ship `.git`, so it otherwise reports it as unknown:

```bash
CHAT_RAG_GIT_SHA=$(git rev-parse HEAD) docker compose up --build
# PowerShell: $env:CHAT_RAG_GIT_SHA = (git rev-parse HEAD); docker compose up --build
```

### What comes up, and in what order

| service | port | waits for |
|---|---|---|
| `db` — PostgreSQL 16 with pgvector | not published | — |
| `app` — the application, `python -m asgi` on uvicorn | `5005` | `db` reporting **healthy** |
| `frontend` — the console, Next.js | `3000` | `app` reporting **healthy** |

Each wait is on a health check, not on "the process exists". PostgreSQL accepts
connections a few seconds after it starts, and the application refuses to serve
without a reachable database, so `service_started` would give a container that
exits and restarts until the timing happened to work.

The database is not published at all: it is reachable from the other two
containers and from nowhere else, which is what lets its password be a compose
default rather than a secret. The application *is* published, because the
contract and `/api/ops/metrics` are consumed from outside the stack.

### The schema

There is no step to remember. `app` runs `python -m tools.migrate` before the
server starts, and that step:

- waits for a database that is accepting connections but still recovering,
  rather than exiting and being restarted until it is ready;
- holds a PostgreSQL advisory lock, so two application containers starting
  together cannot both run the same migration;
- prints the revision it moved from and the one it moved to, and prints
  "already at head" as its own outcome rather than as silence.

It is still Alembic and still reviewable — nothing in the application creates a
table. Set `CHAT_RAG_MIGRATE_ON_START=0` for a deployment that applies its
schema as a separate step; the application then starts against whatever schema
is there. To look without changing anything:

```bash
docker compose exec app python -m tools.migrate --check
```

### Stop

```bash
docker compose down
```

The application serves on uvicorn and shuts down on SIGTERM, draining in-flight
requests and the ingest jobs already running; teardown takes about two seconds,
and the compose file allows fifteen (`stop_grace_period`). The entrypoint
`exec`s the server, so the signal reaches uvicorn itself rather than a shell in
front of it. A bare `docker stop` uses the daemon's own timeout, which on some
installations is only one second — short enough to kill the process
mid-shutdown and report exit 137. `docker stop -t 10` (Docker's documented
default) or `docker compose down` both exit 0.

### Configuration

`.env.docker` holds the application's settings and contains no secrets. Put
anything private in `.env.docker.local`, which is git-ignored and overrides it.
The local `.env` is deliberately not used by the containers: it points at
`localhost`, which inside a container means the container itself.

Two settings are *structural* and live in `docker-compose.yml` rather than in
an env file, because they name containers rather than express a preference:

| | |
|---|---|
| `CHAT_RAG_DATA_DIR=/data` | where the application's files go, matching the mount |
| `CHAT_RAG_API_URL=http://app:5005` | where the console forwards `/api/v1`, by service name |

The second is read **per request** by the console
([`frontend/lib/api/proxy.ts`](frontend/lib/api/proxy.ts)), so one image runs
against any backend. It used to be a `next.config.mjs` rewrite, which Next.js
resolves at build time and freezes into the image — the console then carried
`127.0.0.1`, which inside a container is that container.

Ollama is reached at `http://host.docker.internal:11434`. That address lives in
`.env.docker`, not in the code; the compose file maps the name explicitly so it
also works on plain Linux Docker.

The database password is `${POSTGRES_PASSWORD:-chat_rag}`. Set
`POSTGRES_PASSWORD` in the shell, or in a `.env` beside the compose file (git
ignores it), for any deployment where the database is not private to the stack
— and set the matching `DATABASE_URL` in `.env.docker.local`.

### Where the data lives

Everything the stack persists is in **Docker-managed named volumes**, not in a
directory in this checkout. Running the containers never reads or writes your
local `.cache/` or `artifacts/viewer-live/`.

| volume | mounted at | holds |
|---|---|---|
| `db-data` | `db:/var/lib/postgresql/data` | the records, chunks and embeddings |
| `app-data` | `app:/data` | the parser's canonical-unit cache, packaged Viewer payloads, the embedding caches, upload staging, logs |
| `eval-runs` | `app:/app/artifacts/runs` | evaluation runs written by the CLI |
| `eval-reports` | `app:/app/artifacts/reports` | QA report packages written by the CLI |

Compose prefixes them with the project name, so they are `chat_rag_app-data`
and so on in `docker volume ls`.

They are volumes rather than `./.docker-data/...` bind mounts because of
ownership, and the difference only shows on Linux. Docker creates a missing
bind-mount source directory as `root:root` and the mount carries that
ownership into the container; the application image drops to uid 10001 before
it runs anything, so on a clean Linux host the first thing it did was

```
PermissionError: [Errno 13] Permission denied: '/data/logs'
```

Docker Desktop hides this — its filesystem translation layer presents a bind
mount as owned by whoever asks — so it only appeared on a CI runner. A named
volume is initialised from the image's content at its mount point, ownership
included, so the directory arrives owned by the user that has to write it, on
every platform and with no `chmod` anywhere.

Frozen gold sets under `artifacts/gold/` are inputs, not state: they travel
inside the image, are never mounted over and are never written to.

If you ran an earlier version, the leftover `./.docker-data/` is the previous
layout's and is no longer read. Delete it.

### Looking at the data, and getting it out

```bash
docker compose exec app ls -la /data /data/logs
docker compose exec app tail -f /data/logs/rag_*.log

# copy a report package to the host
docker compose cp app:/app/artifacts/reports ./reports
```

### Reset the stack's data

One command, and it takes the volumes with it:

```bash
docker compose down -v
```

The next `up` finds an empty database and the migration builds the schema from
nothing. This removes only the stack's own knowledge bases, stores and logs;
your local development data is untouched. Without `-v` the containers go and
the data stays, which is what an ordinary stop should do.

### The CLI, inside the container

The same commands, no separate image:

```bash
docker compose exec app python -m cli inspect --kb <name>
docker compose exec app python -m cli search  --kb <name> --query "..."
docker compose exec app python -m cli qa      --kb <name>
docker compose exec app python -m cli report  --kb <name> --gold artifacts/gold/<set>.json
docker compose exec app python -m cli eval    --kb <name> --gold artifacts/gold/<set>.json
```

Reports and runs land in the `eval-reports` and `eval-runs` volumes;
`docker compose cp` above brings one out.

### Health

`GET /api/v1/health` loads no model, parses nothing and does not touch the
vector store. That is what the application container's health check calls, and
what the console waits on before it starts. The console's own check asks itself
for a page, so it says whether the console is serving without claiming anything
about the application behind it.

```bash
docker compose ps          # STATUS shows (healthy) for all three
curl http://localhost:5005/api/v1/health
curl http://localhost:5005/api/ops/metrics
```

---

## The product API — `/api/v1`

The versioned contract, and what a new client should build against:
knowledge bases, documents and their chunking analyses, ingest jobs, questions
and searches, and enough discovery to know what this deployment can do. It is
designed to keep working while the implementation under it is replaced —
PostgreSQL, pgvector, a Next.js front end — so it exposes no file path, no
store provider and no state-file shape. All three of those replacements have
happened underneath it without the wire moving, and the Flask-era surface it
was written beside is gone
([docs/legacy-removal.md](docs/legacy-removal.md)).

`POST /api/v1/queries` is the one that can refuse you under load, with **503**
and a `Retry-After`, or **504** past a deadline; an upload is never refused for
being slow, because it is always a job.
[docs/operations.md](docs/operations.md) says what each refusal means.

**[docs/api-v1.md](docs/api-v1.md) is the contract**: every endpoint, the
request and response shapes, the refusal taxonomy, and what was deliberately
left out of it. `GET /api/v1/openapi.json` is the same list, generated from
the routers and their models, for a client generator to read.

```bash
curl localhost:5005/api/v1/meta/chunking-methods   # what this deployment can chunk with
curl localhost:5005/api/v1/knowledge-bases
curl -X POST localhost:5005/api/v1/queries -H 'content-type: application/json' \
     -d '{"knowledge_base_id":"<id>","question":"..."}'
curl localhost:5005/api/v1/openapi.json            # the contract, machine-readable
```

## The Python API — `chat_rag`

The other way in, for a program rather than a browser. Same engine, same use
cases, same PostgreSQL; no HTTP anywhere.

```python
from chat_rag import Engine, EngineConfig

with Engine(EngineConfig(retrieval_profile="hybrid_rrf")) as engine:
    kb = engine.knowledge_bases.create("Reports")
    document = kb.ingest("report.pdf")          # a path, bytes or an open file
    document.analysis().request()               # queue its chunking analysis
    hits = kb.search("liquidity")               # retrieval, no answer model
    answer = kb.ask("What changed?")            # an answer with its citations
    print(answer.text, answer.grounded)
```

An `Engine` owns its own container, its own runtime — connection pool, provider
budgets, counters, packaging queue — its own session and, given a `data_dir`,
its own files, so two of them in one process share nothing but the database.
It is not the process default and installs no log handler: a library speaks for
itself, not for the program it is imported into.

```bash
pip install chat-rag              # the engine; PostgreSQL is required, not optional
pip install chat-rag[local]       # + sentence-transformers, ollama
pip install chat-rag[pdf]         # + pymupdf, pymupdf4llm, python-docx
pip install chat-rag[all]         # both
```

The extras degrade by refusing rather than by pretending — without `[local]`,
asking for a local model names the extra to install. The wheel installs with
nothing else named: it carries where `amsc-poc` comes from. This repository's
own install is unchanged; `requirements.txt` names everything explicitly.

**[docs/library-api.md](docs/library-api.md) is the contract**: the thirty
published names, what a refusal means and how to catch it, how the schema is
created (`engine.migrate()` — no `alembic.ini`), what 0.x promises, how a
removal is announced, and the limitations a consumer should know about.
[CHANGELOG.md](CHANGELOG.md) records what changes.

## The console — `frontend/`

A Next.js application over `/api/v1`, and nothing else. Knowledge bases,
documents and their ingestion, a document's chunking analysis, search and
chat with citations.

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000, against localhost:5005
```

It holds no catalogue of its own: the chunking methods, the retrieval methods
and the model chain are read from `/api/v1/meta/...` at run time, so a method
added to the library's registry appears in the picker without a line changing.
`frontend/lib/api/` is the only place it calls `fetch` — `client.ts` for a
screen, `proxy.ts` for this server forwarding `/api/v1` to the application —
and the contract's refusal `type`, not a status code, is what its screens
branch on. In the deployed stack it is a container of its own, and the only
port a browser needs. [frontend/README.md](frontend/README.md) is the rest.

## The operator surface

One route, off the contract on purpose: `GET /api/ops/metrics`. Counters,
stage and job latency over a bounded window, error categories, the state of
every cache and the effective non-secret configuration — what an operator
reads once `GET /api/v1/health` has said to look closer. Its *contents* are
free to change with the internals they report on, which is why it is not
versioned and why no client should build against it.
[docs/operations.md](docs/operations.md) reads it for you.

There used to be a whole surface here: the Flask-era **console API**, the one
the rendered screens and the Viewer's relay spoke. It is gone, along with the
screens, the templates, the static JavaScript and the second entrypoint;
[docs/legacy-removal.md](docs/legacy-removal.md) is the record of what each of
its endpoints became.

## The offline CLI

For evaluating a knowledge base without the browser:

```bash
python -m cli inspect --kb <name>                                  # how it is configured
python -m cli search  --kb <name> --query "..."                    # one query, with sources
python -m cli qa      --kb <name>                                  # structural QA over the corpus
python -m cli report  --kb <name> --gold artifacts/gold/<set>.json # an exportable QA package
python -m cli eval    --kb <name> --gold artifacts/gold/<set>.json # a frozen gold set through retrieval
python -m cli gold    --help                                       # freeze reviewed marks into a gold set
```

The same commands run inside the container — see *Running with Docker*.
