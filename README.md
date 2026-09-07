# RAG Console — retrieval-augmented question answering over PDF documents

Upload a PDF into a knowledge base, and ask questions of it. The document is
parsed once into canonical units, chunked by a chosen method, embedded and
indexed; a question retrieves from that index, assembles a labelled context
and gets one answer that cites its sources. A companion **Viewer** shows where
each chunking method put its boundaries, on the same documents.

The system is two repositories:

| repo | what it is |
|---|---|
| **`chat_rag`** (this one) | the product — Flask console and API, ingest jobs, retrieval, the answer chain, resource limits, observability, configuration |
| **`chunk`** (`amsc-poc`) | the chunking library, installed from a pinned commit — chunking methods, Deep Analysis, the canonical PDF adapter, the Viewer page builder and server, all research and benchmark code |

`chunk` is expected beside this checkout (`../chunk`).
[docs/architecture.md](docs/architecture.md) is the map: what each repo owns,
the two runtime flows, and which file to open for what.

---

## Documentation

Start here, then follow the question you have:

| doc | answers |
|---|---|
| **[docs/architecture.md](docs/architecture.md)** | What is the system, which repo owns what, where is the code for X |
| **[docs/operations.md](docs/operations.md)** | How do I run it, what are the limits, and what does *this* 503 mean |
| **[docs/configuration.md](docs/configuration.md)** | Where does a setting come from, who owns it, what wins |
| **[docs/testing.md](docs/testing.md)** | What to run before calling a change done, and the order for cross-repo changes |
| **[docs/limitations.md](docs/limitations.md)** | What this system does not do, and why |
| **[../chunk/docs/adding-a-chunker.md](../chunk/docs/adding-a-chunker.md)** | How to add a chunking method, end to end |
| **[../chunk/docs/viewer-architecture.md](../chunk/docs/viewer-architecture.md)** | How the Viewer works across both repos, and how to debug a package |
| **[../chunk/docs/library-surface.md](../chunk/docs/library-surface.md)** | What is product, research and legacy in the library, and what the console may import |
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
python setup_nltk.py
cp env.example .env                       # PowerShell: Copy-Item env.example .env

# 3. prove the declared source installs and runs (minutes; needs Docker)
python tools/verify_reproducibility.py --local

# 4. start the product and the Viewer
.\start-demo.ps1                          # or: python app.py  (see "Running it by hand")
```

Then, in the browser and the terminal:

5. **upload a PDF** — open <http://127.0.0.1:5005>, create a knowledge base,
   upload a document, choose **Deep Analysis** if a provider key is configured
   and **Standard** if not. The first parse of a PDF takes minutes; the second
   takes under a second.
6. **ask a question** — `/chat`, pick the knowledge base, ask. The answer
   cites `[S1]`, `[S2]`… back to the chunks it used.
7. **open the Viewer** — <http://127.0.0.1:8765>, pick the document, select
   two methods, and step through the boundaries where they disagree.
8. **look at the instruments** — `GET /api/health` (three fields: alive,
   ready, what an operator should do) and `GET /api/ops/metrics` (counters,
   stage latency, error categories, budgets, caches).
   [docs/operations.md](docs/operations.md) reads them for you.
9. **run the tests** — `python -m pytest -q` here, `py -3.11 -m pytest` in
   `../chunk`. [docs/testing.md](docs/testing.md).
10. **add a trivial chunker** — copy `chunk/src/amsc/example_chunker.py`, add
    one `ChunkMethod` to `amsc/methods.py`, add a test. It appears in the
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
python setup_nltk.py             # NLTK data used by the legacy chunker
cp env.example .env
```

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
python app.py        # development: Werkzeug, loopback only, debugger on
python -m wsgi       # production: waitress, all interfaces, no debugger
```

Both serve <http://127.0.0.1:5005>. Which one is running is not a detail —
[docs/operations.md](docs/operations.md) says why.

### Common startup failures

| symptom | cause |
|---|---|
| `ValueError` naming an env variable, before the server binds | a configuration value was refused. That is deliberate: bad values stop the process while someone is looking. [docs/configuration.md](docs/configuration.md) |
| `ModuleNotFoundError: amsc` | `pip install -r requirements.txt` did not run, or the pinned commit is unreachable. `python tools/import_smoke.py` says which `amsc` answered |
| the Viewer link is dead, or `start-demo.ps1` fails on the Viewer | the Viewer page is a build artifact, not in version control. The launcher builds it; by hand it is `py -3.11 -m amsc.viewer_v3 --output artifacts/viewer-v3/index.html` in `../chunk` |
| a port is already in use | `start-demo.ps1` recognises a server it already started and refuses a port held by something else. `-ProductPort` / `-ViewerPort` move them |
| the first upload seems to hang | it does not — layout parsing is minutes per document on CPU, and the job is running. Poll `GET /api/ingest/jobs/<job_id>` |
| answers fail but search works | no provider key, or an unreachable gateway. The answer model carries the reason; retrieval never depended on it |

---

## What runs, and where

### The model chain

Three model roles, one OpenRouter key, each role configured on its own:

| Stage | Model (demo) | Configuration | Runs |
|---|---|---|---|
| Deep Analysis / Agentic chunking — proposer + verifier | `qwen/qwen3-30b-a3b-instruct-2507` | `DEEP_ANALYSIS_MODEL`, `DEEP_ANALYSIS_ENDPOINT`, `DEEP_ANALYSIS_API_KEY_ENV`, `DEEP_ANALYSIS_VERIFY` | at upload only, through `amsc.deep_pipeline` |
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
| Standard | The frozen structure-first walk (`amsc.structural_chunker`). Fast, deterministic, no model. | — |
| Deep Analysis | `amsc.deep_pipeline.chunk_document(mode="deep")`: the same structural walk, a backend LLM **proposer** (one bounded prompt per section that still has a choice), the deterministic **quality selector** (never worse than Standard on any smell type), the double-order **verifier** (a change is kept only when it wins in both orders) and the quality measurement. | The ingest still completes on the deterministic quality contract and the document is labelled with the pipeline status — never passed off as Standard. |

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
compare them side by side. `GET /api/demo/methods` says which methods this
machine can run, and why one cannot.

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
profile. The legacy and V4 chunkers and the `legacy` / `benchmark_aligned`
retrieval profiles remain selectable for comparison.

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
| `structure_first` | the demo profile: document structure decides the boundaries |
| `legacy` | the original `SemanticChunker` — word windows with overlap |
| `v4` | the frozen AMSC V4/A4 implementation, for comparison |

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

`RETRIEVAL_PROFILE=legacy` preserves the existing chat_rag retrieval behavior.
`RETRIEVAL_PROFILE=benchmark_aligned` selects the frozen Phase 4/5 profile:
multilingual E5 role prefixes,
normalized deterministic long-text pooling, Unicode BM25, and equal-weight RRF
with a 100-result pool and `k=60`. Query expansion, contextualization, and
reranking are disabled in this profile. Its E5 model is loaded with
`local_files_only=true`, matching the frozen benchmark configuration.

Indexes are profile-specific because the embedding models and dimensions differ.
Use a new vector-database path/collection and re-ingest documents when changing
profiles; do not point `benchmark_aligned` at an index created by `legacy`.

---

## Demo mode (product + Agentic Chunking Viewer)

The proof of concept has two faces: this product (how it is used) and the
chunk repository's **Viewer v3** (what the chunking technology does
underneath — where each method put its boundaries, and why). They stay
separate servers; one script starts both for a presentation.

```powershell
.\start-demo.ps1      # start both, wait until each answers, open the product
.\stop-demo.ps1       # stop what start-demo started
```

| | Address | Server |
|---|---|---|
| Product (chat_rag) | http://127.0.0.1:5005 | `venv\Scripts\python.exe app.py` (the development server; `FLASK_PORT`, reloader off) |
| Viewer (chunk, Viewer v3) | http://127.0.0.1:8765 | `py -3.11 -m amsc.viewer_server --viewer artifacts/viewer-v3/index.html --console-url http://127.0.0.1:5005` in the chunk repo |

The chunk repository is expected next to this one (`..\chunk`); override with
`-ChunkPath` or `CHUNK_REPO`. The Viewer page itself is a build artifact and is
not in version control, so on a fresh clone the launcher builds it first --
`py -3.11 -m amsc.viewer_v3 --output artifacts\viewer-v3\index.html`, the
product shell, which carries no corpus of its own and reads every document
live from this console. A page that is already there is served unchanged.
How the two repositories divide the Viewer up, and what to look at when a
document's analysis fails, is in `..\chunk\docs\viewer-architecture.md`.

The launcher checks readiness over HTTP
(`/api/health` on both), recognises servers that are already running instead
of starting a second copy, refuses a port held by something else, writes the
servers' output to `.demo\logs\` and the started process ids to
`.demo\state.json` (both git-ignored). `stop-demo.ps1` stops only processes it
can identify as those servers; `-All` extends that to a product/viewer server
on the demo ports that it did not start. Nothing from `.env` is printed; the
Viewer's chat gets `OPENROUTER_API_KEY` from the environment or `.env`
(without it the Viewer runs BM25-only, no answers — use `-Lexical` to force
that). Options: `-NoBrowser`, `-OpenViewer` (second tab), `-ProductPort`,
`-ViewerPort`, `-TimeoutSeconds`.

The two windows share one state. `GET /api/demo/workspace` returns this
console's knowledge bases, their documents and chunk counts as a read-only
snapshot; the Viewer's server reads it (`--console-url`, which `start-demo.ps1`
points back here) and serves it to its own page. So a knowledge base created
here, or a document ingested into it, appears in the Viewer's workspace strip
on its next refresh — there is no second copy of that state to keep in step,
and the browser never has to reach a second origin. The snapshot carries names,
counts and ingest metadata only: no absolute paths and no full file hashes.

**A document uploaded here becomes a document you can analyse over there.**
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
no HTTP call waits on it: an upload records the ingest and returns, and the
Viewer's refresh (`?prepare=1`) only *queues* what is missing. Each document's
state — `missing` / `pending` / `running` / `ready` / `failed` — travels with
it in the workspace snapshot, so the Viewer lists a ready document in its own
document picker and shows one that is still being prepared as a disabled entry
saying so.

An upload chooses **which chunking methods to analyse the document with**
(`methods=markdown&methods=structure-only&methods=agentic`, or the older
`deep_analysis=true`). The PDF is parsed once; every chosen method runs over
that one canonical and is packaged as its own arm, so the Viewer can compare
them side by side under a single document. Identity is the file's content
hash: uploading the same PDF again adds variants to the document that is
already there instead of making a second one, and
`POST /api/demo/viewer-analysis/<doc_id>/methods` adds a method later without
re-reading the file. The *analysis* is shared that way; the *choice* is not.
Each upload record keeps the methods it asked for, and that is what the Viewer
opens it on — an upload that ticked Standard and Hybrid is not shown the
Markdown and Deep Analysis variants another upload of the same file left
behind. The shared analysis keeps all of them, so neither upload costs a
second parse. `GET /api/demo/methods` says which methods this machine
can actually run, and why one cannot. The methods themselves are defined once,
in the library's registry (`amsc.methods` in the chunk repository): key,
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

Inside the product, **Tools → Agentic Chunking Viewer** (sidebar, with a
live/offline dot) and the card at the top of **Lab** open the Viewer in a new
tab. The address comes from `VIEWER_URL` (default `http://127.0.0.1:8765/`;
empty hides the link).

Presentation order: **1.** chat_rag — a knowledge base and its documents;
**2.** upload a document with **Deep Analysis** (status and quality summary
under the chunking badge, *Details* for before/after); **3.** Chat — an answer
with sources; **4.** Agentic Chunking Viewer; **5.** Sunum (the four methods
side by side) → Debug (why each boundary) → Benchmark; then back to the
product.

---

## Running with Docker

One container runs the whole application. There is no separate database,
queue or model service: Ollama stays on the host, and everything else runs
in-process.

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine with Compose v2
- Ollama on the host **only if you want generated answers**. Uploading,
  parsing, structure-first chunking, structural QA and BM25 search all work
  with Ollama stopped; generation then returns an explanatory error instead of
  taking the application down.

### Start

```bash
docker compose up --build
```

To have the QA report name the commit the image was built from -- the image
does not ship `.git`, so it otherwise reports it as unknown:

```bash
CHAT_RAG_GIT_SHA=$(git rev-parse HEAD) docker compose up --build
# PowerShell: $env:CHAT_RAG_GIT_SHA = (git rev-parse HEAD); docker compose up --build
```

Then open <http://localhost:5005>. The app lands on Knowledge Bases;
Chat is at `/chat` and the technical tools (retrieval quality review,
parser output, chunk browser) are under `/lab`.

### Stop

```bash
docker compose down
```

The container serves on waitress and shuts down on SIGTERM, draining in-flight
requests; teardown takes about two seconds, and the compose file allows fifteen
(`stop_grace_period`). A bare `docker stop` uses the daemon's own timeout, which
on some installations is only one second -- short enough to kill the process
mid-shutdown and report exit 137. `docker stop -t 10` (Docker's documented
default) or `docker compose down` both exit 0.

### Configuration

`.env.docker` holds the container's settings and contains no secrets. Put
anything private in `.env.docker.local`, which is git-ignored and overrides
it. The local `.env` is deliberately not used by the container: it points at
`localhost`, which inside a container means the container itself.

Ollama is reached at `http://host.docker.internal:11434`. That address lives
in `.env.docker`, not in the code; the compose file maps the name explicitly
so it also works on plain Linux Docker.

### Where the data lives

Everything the container persists is under `./.docker-data`, which is a
different place from the paths a local checkout uses. Running the container
never reads or writes your local `chroma_db/`, `.knowledge_bases.json`,
`.ingested_documents.json` or `.cache/`.

```
.docker-data/
  state/      knowledge_bases.json, ingested_documents.json, gold_set.json
  chroma/     one vector store per knowledge base
  faiss/      the same, for knowledge bases using the FAISS provider
  cache/      the parser's canonical-unit cache
  logs/       application logs
  artifacts/  evaluation runs and QA reports written by the CLI
```

Frozen gold sets under `artifacts/gold/` are inputs, not state: they travel
inside the image and are never written to.

### Reset the container's data

Stop the container first, then delete the one directory:

```bash
docker compose down
rm -rf ./.docker-data          # PowerShell: Remove-Item -Recurse -Force .docker-data
```

This removes only the container's knowledge bases, stores and logs. Your local
development data is untouched.

### The CLI, inside the container

The same commands, no separate image:

```bash
docker compose exec app python -m cli inspect --kb <name>
docker compose exec app python -m cli search  --kb <name> --query "..."
docker compose exec app python -m cli qa      --kb <name>
docker compose exec app python -m cli report  --kb <name> --gold artifacts/gold/<set>.json
docker compose exec app python -m cli eval    --kb <name> --gold artifacts/gold/<set>.json
```

Reports and runs land in `./.docker-data/artifacts/` on the host.

### Health

`GET /api/health` answers from the Flask app alone -- it loads no model, parses
nothing and does not touch the vector store. That is what the container's
healthcheck calls.

```bash
docker compose ps          # STATUS shows (healthy)
```

---

## The console API

The screens are `/` (knowledge bases), `/chat`, `/documents`, `/chunks` and
`/lab`. Everything they do is an HTTP call you can make yourself:

| group | endpoints |
|---|---|
| knowledge bases | `GET|POST /api/kb`, `GET|PUT|DELETE /api/kb/<kb_id>`, `GET /api/kb/options`, `GET /api/kb/<kb_id>/embedding-index`, `POST /api/kb/<kb_id>/reindex-embeddings` |
| documents | `POST /api/documents/upload`, `GET /api/documents`, `DELETE /api/documents/<doc_id>`, `GET /api/documents/<doc_id>/chunks`, `GET /api/documents/<doc_id>/canonical-units` |
| ingest jobs | `GET /api/ingest/jobs`, `GET|DELETE /api/ingest/jobs/<job_id>` |
| asking | `POST /api/query`, `POST /api/clear` |
| the Lab | `POST /api/chunks/search-vector`, `POST /api/chunks/search-bm25`, `POST /api/experiment/search_chunks`, `POST /api/experiment/rank_chunks` |
| chunks | `GET|POST /api/chunks`, `GET|PUT|DELETE /api/chunks/<chunk_id>` |
| gold set | `GET|POST /api/goldset`, `DELETE /api/goldset/<entry_id>` |
| the Viewer bridge | `GET /api/demo/viewer`, `GET /api/demo/workspace`, `GET /api/demo/methods`, `GET|POST /api/demo/viewer-analysis/<doc_id>`, `POST /api/demo/viewer-analysis/<doc_id>/methods`, `GET /api/demo/viewer-analysis/<doc_id>/payload`, `GET /api/demo/viewer-analysis/<doc_id>/chunks` |
| instruments | `GET /api/health`, `GET /api/ops/metrics`, `GET /api/stats`, `GET /api/models`, `GET /api/retrieval/capabilities` |

`POST /api/documents/upload` and `POST /api/query` are the two that can refuse
you under load, with **503** and a `Retry-After`, or **504** past a deadline.
[docs/operations.md](docs/operations.md) says what each refusal means.

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
