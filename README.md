# /Users/murseltasgin/projects/chat_rag/README.md
# Advanced RAG System with Conversational Context

A modular, production-ready Retrieval-Augmented Generation (RAG) system with sophisticated ingestion and query pipelines.

## Features

- **Semantic Chunking**: Context-preserving document chunking with overlap
- **Contextual RAG**: Document metadata enrichment and context awareness
- **Query Understanding**: Automatic query clarification and expansion
- **Hybrid Retrieval**: Combines vector search (semantic) with BM25 (keyword)
- **Intelligent Reranking**: LLM-based or Cross-Encoder reranking for optimal results
- **Conversation Tracking**: Multi-turn conversation support with reference resolution
- **Smart Search Strategy**: Automatically selects optimal retrieval method
- **Multiple LLM Providers**: Support for Azure OpenAI and Ollama
- **Multiple Vector DBs**: Support for ChromaDB and FAISS
- **Knowledge Base Management**: Create and manage multiple knowledge bases
- **Document Tracking**: Automatic tracking to avoid re-processing documents
- **Web Interface**: Modern browser-based UI for document management and chat
- **CLI Interface**: Command-line chat application with automatic ingestion
- **Modular Architecture**: Pluggable components following SOLID principles

## Architecture

```
chat_rag/
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py         # Central config from .env
├── core/                   # Core data models and exceptions
│   ├── __init__.py
│   ├── models.py          # Data models (DocumentChunk, RetrievalResult, etc.)
│   └── exceptions.py      # Custom exceptions
├── components/            # Pluggable components
│   ├── chunker/          # Text chunking strategies
│   │   ├── base.py       # Base chunker interface
│   │   └── semantic_chunker.py
│   ├── embedding/        # Embedding models
│   │   ├── base.py       # Base embedding interface
│   │   └── sentence_transformer_embedding.py
│   ├── vectordb/         # Vector database providers
│   │   ├── base.py       # Base vectordb interface
│   │   ├── chroma_vectordb.py  # ChromaDB implementation
│   │   └── faiss_vectordb.py   # FAISS implementation
│   ├── llm/              # LLM providers
│   │   ├── base.py       # Base LLM interface
│   │   ├── azure_openai_llm.py  # Azure OpenAI implementation
│   │   └── ollama_llm.py        # Ollama implementation
│   ├── retriever/        # Retrieval strategies
│   │   ├── base.py       # Base retriever interface
│   │   └── hybrid_retriever.py
│   ├── query_processor/  # Query understanding and enhancement
│   │   └── query_enhancer.py
│   ├── reranker/         # Result reranking
│   │   ├── base.py      # Base reranker interface
│   │   ├── reranker.py  # LLM-based reranker
│   │   └── cross_encoder_reranker.py  # Cross-encoder reranker
│   ├── conversation/     # Conversation management
│   │   └── conversation_manager.py
│   ├── document_processor/ # Document preprocessing
│   │   └── document_processor.py
│   ├── parsers/            # Document parsers
│   │   ├── base.py        # Base parser interface
│   │   ├── pdf_parser.py  # PDF parsing
│   │   ├── docx_parser.py # DOCX parsing
│   │   ├── markdown_parser.py  # Markdown parsing
│   │   ├── text_parser.py # Plain text parsing
│   │   ├── image_parser.py # Image OCR parsing
│   │   └── parser_factory.py  # Parser factory
│   └── contextual_enhancer/ # Contextual enrichment
│       └── contextual_enhancer.py
│   └── knowledgebase/    # Knowledge base management
│       └── manager.py    # Multi-KB manager
├── pipeline/             # Main orchestration
│   ├── __init__.py
│   └── rag_pipeline.py   # Main RAG pipeline
├── utils/                # Utilities
│   ├── logger.py         # Logging utilities
│   └── document_tracker.py # Document ingestion tracking
├── main_new.py           # CLI chat application
├── app.py                # Web application (Flask)
├── requirements.txt      # Dependencies
└── env.example          # Environment variables template
```

## The model chain

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

## Demo mode (product + Agentic Chunking Viewer)

The proof of concept has two faces: this product (how it is used) and the
chunk repository's **Viewer v2** (what the chunking technology does
underneath — Sunum / Sorgu / Debug / Benchmark). They stay separate servers;
one script starts both for a presentation.

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
re-reading the file. `GET /api/demo/methods` says which methods this machine
can actually run, and why one cannot. The methods themselves are defined once,
in the library's registry (`amsc.methods` in the chunk repository): key,
engine kind, product name, summary and capabilities. `components/viewer/methods.py`
is this console's view of that registry and adds only what the deployment
decides — availability on this machine, display order, the default. Adding a
chunking method is a library change (`chunk/docs/adding-a-chunker.md`: a
partition function, one registry entry, a test) followed by bumping the
`amsc-poc` pin in `requirements.txt`; no route, template or script here names
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

## Bounded ingest (uploads as jobs)

An upload is an **ingest job**. `POST /api/documents/upload` validates the
request, stages the file and queues the job; the parse, the chunking, any
Deep Analysis model calls, the embeddings, the store write and the ledger
write happen on an ingest worker. Two answers are possible:

* `async=1` (what the console sends): **202** at once with `job_id` and the
  job; poll `GET /api/ingest/jobs/<job_id>` until `status` is terminal.
* otherwise the request waits for the job (up to `INGEST_SYNC_WAIT` seconds)
  and answers exactly as before: **200** with the document, **409** when the
  store must be re-indexed first, **503** when Deep Analysis cannot run on this
  knowledge base, **500** on failure. A job still running when the wait runs
  out is answered **202** with the job to poll.

Job states: `queued` → `running` → `succeeded` | `failed` | `timed_out` |
`cancelled` | `interrupted`. A full queue refuses the upload with **503**,
`overloaded: true` and a `Retry-After` header; nothing is queued and nothing
is kept. The same file uploaded again for the same knowledge base and methods
while the first is still in flight is *attached* to that job (`attached:
true`, same `job_id`). One ingest runs per knowledge base at a time;
different knowledge bases run in parallel up to `INGEST_WORKERS`.

| Setting | Default | Bounds |
|---|---|---|
| `INGEST_WORKERS` | 2 | jobs running at once (parsing, chunking, local embedding) |
| `INGEST_QUEUE_CAPACITY` | 8 | jobs waiting behind busy workers |
| `INGEST_JOB_TIMEOUT` | 1800 s | how long a job may run once started |
| `INGEST_SYNC_WAIT` | 840 s | how long a synchronous upload waits before 202 |
| `INGEST_SYNC_WAITERS` | half of `WAITRESS_THREADS` | request threads that may block on an upload |
| `INGEST_JOB_RETENTION` | 3600 s | how long a finished job stays queryable |
| `PROVIDER_MAX_INFLIGHT` | 8 | Deep proposer + verifier calls in flight, process-wide |
| `DEEP_ANALYSIS_CONCURRENCY` | 8 | one Deep job's own call pool |
| `EMBEDDING_MAX_INFLIGHT` | 4 | embedding requests in flight, process-wide |

**The resource model.** Every externally multiplying call is bounded, and by
its own limit so that one service cannot starve another:

| Work | Bound |
|---|---|
| Deep proposer / verifier calls | `PROVIDER_MAX_INFLIGHT` global, `DEEP_ANALYSIS_CONCURRENCY` per job |
| Embedding requests (remote provider) | `EMBEDDING_MAX_INFLIGHT` global |
| Embedding batches (local model) | `INGEST_WORKERS` — CPU on the worker |
| Parsing, chunking, indexing | `INGEST_WORKERS` |
| Viewer packaging | its single thread; it makes no provider call |
| Answer model at query time | `ANSWER_MAX_INFLIGHT` global; request threads in a query by `QUERY_MAX_ACTIVE`; the whole query by `QUERY_TIMEOUT` (see *Bounded queries*) |

Both caps are taken per call, not per job: a Deep job's pool of eight shares
`PROVIDER_MAX_INFLIGHT` slots call by call with every other Deep job, and the
number in flight across the process never exceeds it. A budget guarantees
**boundedness, not fairness** — no ordering among waiters is promised, and a
waiter cannot hang because every wait is bounded by the job's own deadline.

**Deadline semantics: cooperative, with the network boundary enforced.** The
deadline is checked at stage boundaries — after parsing, after chunking,
before the store write — and before every outbound call, so a job that runs
out of time stops with nothing committed. Work already inside a stage runs to
the end of that stage; a parse is not interrupted. The one place where the
overshoot would otherwise be a whole `DEEP_ANALYSIS_TIMEOUT` is a network
call, so each call's socket timeout is clamped to the time the job has left.
A ledger that cannot be written takes the store rows back out.

**Restart.** Jobs live in memory and none is resumed — nothing was committed,
because the ledger write is a job's last act. What does survive is the
*answer*: every job journals its transitions (`INGEST_JOB_RETENTION` applies
to those records too), and start-up settles whatever was in flight against
the ledger. A client holding a `job_id` from before a restart gets
`succeeded` when the ledger holds the document that job wrote, and
`interrupted` — "the server stopped, nothing was registered, upload it
again" — when it does not. **404** now means only that the job is older than
the retention window. The staging directory is swept at start-up, because a
file there belongs to no job.

`GET /api/health` and `GET /api/ingest/jobs` show the capacity picture,
including both budgets and how many finished jobs are retained;
`DELETE /api/ingest/jobs/<job_id>` cancels a queued job at once and a running
one at its next boundary.

## Bounded queries (chat under limits)

A question is answered on the request thread that received it: retrieval,
context assembly and the answer-model call all happen there, because a chat
answer is synchronous. What multiplies under load is therefore request
threads held for the length of a provider call, and three bounds close that
(`components/query/limits.py`, `config/query.py`):

| Setting | Default | Bounds |
|---|---|---|
| `QUERY_MAX_ACTIVE` | `WAITRESS_THREADS - INGEST_SYNC_WAITERS - 1` (3) | request threads inside a query at once |
| `ANSWER_MAX_INFLIGHT` | 4 | answer-model calls in flight, process-wide, every session |
| `QUERY_TIMEOUT` | 180 s | one question, start to answer |
| `ANSWER_SLOT_WAIT` | 30 s | how long a question waits for an answer slot before it is refused |

**Admission is immediate and never queues.** `POST /api/query` either enters
now or is refused now with **503**, `overloaded: true`, `reason: admission`
and a `Retry-After` (derived from the median recent query time, within 2–30
seconds). A queued question would hold the very thread the limit exists to
keep free. The default is derived so that questions and synchronous uploads
together can never take every request thread, which is what keeps
`/api/health`, `/api/ops/metrics` and job polling answerable under any burst
of either; `tests/integration/test_query_starvation.py` proves it on the
real server, and the process warns at start-up when an explicit setting
gives that guarantee up. The chat page puts the question back in the box
with a "busy, try again in N seconds" notice.

**The answer budget is a third budget on purpose.** In the demo all three
roles — Deep Analysis, embeddings, answers — reach one gateway with one
key, so one cap on "OpenRouter calls" is the obvious alternative, and the
wrong one. A Deep call is one short vote among dozens made in bulk from a
background worker; an answer is one long interactive completion a person is
watching. One semaphore would let a Deep job holding eight slots make every
chat wait for an ingest, and a busy afternoon of chat stall the ingest
queue. The fallback answer model is also a local Ollama whose cost is this
host's CPU, which Deep calls to a remote gateway must not spend. Separate
budgets give each path a floor whatever the other is doing. A question that
cannot get an answer slot within `ANSWER_SLOT_WAIT` is refused as
`reason: answer_capacity`, the same **503** shape, so "raise
`QUERY_MAX_ACTIVE`" and "raise `ANSWER_MAX_INFLIGHT`" stay distinguishable.

**Deadline semantics: cooperative, with the network boundary enforced** —
the same claim ingest makes, and no stronger. The deadline is checked before
every outbound call and while waiting for a slot; each answer attempt's
socket timeout is clamped to the time left and a retry there is no time for
is skipped; the fallback model is not tried with no time left; the query's
own embedding call goes through the ingest embedding wrapper, which already
honours the guard. What is *not* interrupted: a lexical index rebuild on the
request thread, a store lookup, a local model's forward pass, and an Ollama
call — its client timeout is fixed at construction and `chat` takes none per
call, so it is refused before it starts but never shortened. A question can
overshoot `QUERY_TIMEOUT` by the longest such stage, never by a whole
provider timeout on top. When it passes, the answer is **504**,
`timed_out: true`. Capacity — admission, the answer slot, the pipeline
lease — is released on every exit.

**Local models are loaded once per name.** A pipeline is built per browser
session and knowledge base; each used to load its own copy of the
sentence-transformers embedder (and, on the legacy profile, the
cross-encoder), so `PIPELINE_CACHE_MAX` was also a multiplier on model
memory. Both are now shared process-wide by model name; `caches.local_models`
on the metrics endpoint shows what is resident and how often it was loaded.

**The Lab's search endpoints run under the same limits.** `POST
/api/chunks/search-vector`, `/api/chunks/search-bm25`,
`/api/experiment/search_chunks` and `/api/experiment/rank_chunks` do the
front half of a query on the request thread — embed the question, search the
store, build the lexical index if this pipeline has not built it yet — so
under no limit at all they were a way around `QUERY_MAX_ACTIVE`: a burst of
them could hold every request thread, each waiting an unbounded time for an
embedding slot. They take the same admission counter (the bound is on
request threads doing retrieval, whichever endpoint asked), the same
deadline, the same pipeline lease and the same telemetry, under
`mode: lab.*` so an operator can tell them from chat. They make no
answer-model call, so they are given no answer budget. Their answers are
unchanged apart from the two refusals every query path shares: **503** when
admission is full, **504** past the deadline.

**A limit is never a fallback.** The enhancement paths degrade on purpose —
a clarification the model could not produce falls back to the original
question, a strategy to hybrid, a document summary to the title — and that
stays. It must not extend to a deadline, a refused budget or a
cancellation: swallowed into a fallback, those become a query that runs on
past its deadline making calls that are refused in turn and then answers
from heuristics as though nothing happened. `RESOURCE_CONTROL_EXCEPTIONS`
(`core/exceptions.py`) names the four, and every fallback handler in the
query enhancer, the contextual enhancer, the LLM reranker and the pipeline
re-raises them first.

**Every query is measured with the ingest instrument.** A query is a trace
of kind `query` with stages `retrieve`, `rerank` (legacy profile), `context`
and `answer`, plus provider seconds and slot wait recorded by the budget
wrapper. `GET /api/ops/metrics` carries `metrics.queries` — active and peak
active, p50/p95/max per stage, provider wait, outcomes (`succeeded`,
`failed`, `timed_out`, `rejected`) and the last few traces — and `query`
(admission and answer-budget counters, the limits, the deadline semantics
in one sentence). `/api/health` carries one line: `query.active`,
`max_active`, `answer_inflight`, `answer_limit`, and reports `overloaded`
while every query slot is in use. The response's `metadata.query` carries
the same timing for that one question, so a slow answer can be correlated
with the metrics by `query_id`. The log carries the question's length, the
stage times and the outcome; never the question, a chunk or the answer. The
legacy profile's step-by-step retrieval trace — which used to print the
question, the clarified rewrite, every generated variation and the
conversation so far to stdout, a deployment's log stream — is at `DEBUG`
with the rest of the content-bearing output, so `LOG_FILE_LEVEL=DEBUG` still
gets all of it and the default gets none of it.

## Operating it: health, metrics and the caches

`GET /api/health` is the small one, for a probe: liveness, readiness and a
line of capacity. It answers three different questions with three fields,
and they are not the same question:

| Field | Question | When it changes |
|---|---|---|
| `status` | Is the process alive? | Never, while it answers at all. It is the historical `healthy` value, kept so existing probes and the serve smoke keep working. |
| `ready` | May traffic be sent here? | Never, in practice. It stays true while overloaded and while degraded, because refusing traffic during an overload makes the overload worse and a degraded process still serves reads. |
| `state` | What should an operator do? | `ok`, `overloaded` (the ingest queue is full, so uploads are being refused; chat and search still work -- wait, do not restart) or `degraded` (something needs a person, named in `reasons`). |

`degraded` has two causes: the knowledge base records cannot be read, or the
last `DEGRADED_AFTER_JOBS` (5) ingest jobs all failed -- the "healthy but
broken" case, where the process is serving and the queue is empty because
every job dies. It is a ratio over the bounded metrics window rather than a
latch, so a service that recovers stops reporting it. Where both apply,
`degraded` wins over `overloaded`: being full is transient, and failing is not.

`reasons` are written for an operator, and go through the same redaction as
everything else here -- a storage error names what failed, not where the data
root lives.

`GET /api/ops/metrics` is the one to read when health says to look closer:

* **counters** — jobs accepted, succeeded, failed, timed out, cancelled,
  rejected, and restart-settled;
* **stage latency** — p50/p95/max for `parse`, `chunk`, `deep_analysis`,
  `embed`, `index`, `ledger` and `viewer_stage`, plus queue wait and total job
  time, over a bounded window of recent jobs;
* **errors** — counts by category (`configuration`, `provider`, `storage`,
  `timeout`, `overloaded`, …) with a few example messages each;
* **capacity** — workers, queue, both provider budgets and how long callers
  have spent waiting for a slot;
* **caches** — the pipeline cache's size against its bound, and whether the
  Hybrid boundary model is resident.

`?recent=N` (max 25) sets how many individual job traces come back. Every
part of this answer is bounded by construction, so its size does not grow
with uptime. A single job's own timing is on its job record, at
`GET /api/ingest/jobs/<job_id>`.

**What this endpoint may contain.** Counts, durations, categories, states,
and the ids an operator needs to correlate a job with a log line -- job ids,
knowledge base ids, chunking modes. No document text, no chunk, no filename,
no temp path, no key. The one place arbitrary text could arrive is the few
example messages kept per error category, and an exception string is written
for a developer standing in a source tree: an ordinary `[Errno 2]` names the
account, the deployment's layout and the document, none of which a credential
filter would catch. Every such message is therefore redacted where it is
stored -- credential shapes blanked, absolute paths replaced by `<path>`,
length capped -- so what is served says *what* failed and not *where*. The
endpoint has no access boundary of its own because the application has none
to reuse: everything it serves is aggregate by construction, and strictly
less than `/api/kb` and `/api/chunks` already return to the same caller.

**Where the logs go, and how big they get.** This application owns its file
sink -- it is not a container's stdout that something else rotates. It writes
`logs/rag_<timestamp>.log` under the data root, and to the console as well.
That file used to grow in two directions at once: without a size limit, and
with one more file per restart that nothing ever removed. Both are bounded
now, by `LOG_MAX_BYTES` x (`LOG_BACKUPS` + 1) for a single run and
`LOG_RUNS_KEPT` for the directory -- a little under half a gigabyte at the
defaults, and no logging platform involved. Console output stays the
deployment's to collect.

**The file handler is at `INFO` by default, and that is deliberate.** This
system's request and retrieval dumps -- full prompts, full retrieved chunks,
the whole answer context -- are written at `DEBUG`, and a log file is the
most-copied artefact a service has: tailed, shipped, pasted into tickets. A
default that puts a copy of the corpus there is a decision nobody makes on
purpose, so it is not the default. Everything an operator needs stays at
`INFO`: the lifecycle events below, start-up, ingest and every error.

A developer debugging retrieval opts in explicitly with `LOG_FILE_LEVEL=DEBUG`,
and the process says so in a warning line at start-up. An unrecognised value
falls back to `INFO`, not to `DEBUG` -- a typo in a deployment's configuration
must not be the thing that starts writing document text to disk.

**Operational logs.** Lifecycle events are written to the `RAG.ops` logger as
one line each, in a stable `event=… job_id=… kb_id=…` shape:
`ingest.job.accepted`, `.started`, `.succeeded`, `.failed`, `.timed_out`,
`.cancelled`, `.rejected`, `.attached`, `.restart_settled`. Values are
redacted before they are written -- document text, chunks, prompts and
anything credential-shaped cannot reach a log line through this path, which
is what makes it safe to leave on and to paste into a ticket.

**What is bounded, and by what**

| Resource | Bound |
|---|---|
| Built pipelines (models, store handles, lexical indexes) | `PIPELINE_CACHE_MAX` (8), `PIPELINE_CACHE_TTL` (1800 s) |
| Job registry and journal | `INGEST_JOB_RETENTION` (3600 s), 500 records |
| Metrics window | 200 traces, 5 messages per error category |
| Hybrid boundary model | one shared instance, loaded on first use |
| Staged uploads | deleted with the job; swept at start-up |
| `logs/` | `LOG_MAX_BYTES` (10 MB) x (`LOG_BACKUPS` + 1) per run, `LOG_RUNS_KEPT` (10) runs kept; `LOG_FILE_LEVEL` is `INFO`, so no content is written |

A pipeline is evicted only when nothing is using it: an ingest job leases its
pipeline for the length of the job, so a burst of browser traffic cannot
close the store a job is writing to. When a document is ingested, every
*other* pipeline for that knowledge base drops its lexical index and rebuilds
it on the next query -- without that, a document uploaded in one browser was
missing from keyword search in another until the process restarted.

**Deliberately not bounded**, with the reason:

| Grows with | Why it is left alone |
|---|---|
| `artifacts/viewer-live/` — one directory per analysed document | Product data, not a cache: it is what the Viewer reads. It is deleted with its document and is regenerable from an ingest. Bounding it would mean deleting analyses a user still expects to open. |
| `.cache/canonical-units/`, `.cache/embeddings/`, `.cache/boundary-embeddings/` | Content-addressed caches on disk, not in memory. They trade disk for a re-parse or a re-embed, and both are safe to delete at any time. Capping them needs an eviction policy and a size accounting that this product has no evidence it needs yet. |
| Chat history per session | Already bounded by `MAX_CONVERSATION_HISTORY`, and it goes when its pipeline is evicted. |

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

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Ollama installed locally if using Ollama LLM provider

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chat_rag
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python setup_nltk.py
```

5. **Configure environment**
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration
# For Azure OpenAI: Set AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT
# For Ollama: Set LLM_PROVIDER=ollama and OLLAMA_MODEL
```

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

See [Ollama Guide](docs/OLLAMA_GUIDE.md) for more details.

## Configuration

All configuration is centralized in `config/settings.py` and reads from `.env` file. See `env.example` for all available options.

### Key Configuration Options

```env
# LLM Provider (azure or ollama)
LLM_PROVIDER=azure

# Azure OpenAI (when LLM_PROVIDER=azure)
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT=gpt-4o

# Ollama (when LLM_PROVIDER=ollama)
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database Provider (chroma or faiss)
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_PATH=./chroma_db

# Chunking
CHUNKER_TYPE=legacy  # legacy or v4
CHUNK_SIZE=512
CHUNK_OVERLAP=128
MIN_CHUNK_SIZE=50

# Retrieval
RETRIEVAL_PROFILE=legacy  # legacy or benchmark_aligned
DEFAULT_TOP_K=5
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3

# Conversation
ENABLE_CONVERSATION=true
MAX_CONVERSATION_HISTORY=10

# Reranker Configuration
RERANKER_TYPE=cross_encoder  # Options: 'llm' or 'cross_encoder'
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Document Input
DOCUMENTS_INPUT_PATH=./documents
DOCUMENTS_RECURSIVE=true

# Logging
LOG_LEVEL=INFO
LOG_TOKEN_USAGE=true
```

For complete configuration options, see `env.example`.

### Frozen V4/A4 chunking

`CHUNKER_TYPE=legacy` preserves the existing `SemanticChunker`. Setting
`CHUNKER_TYPE=v4`, or selecting `v4` while creating a knowledge base in the
web UI, uses the Phase-5 AMSC V4/A4 implementation. The dependency is pinned to
`erenayd58/chunk` commit
`1e7f7186c13729c739ccb3170da0892f7350cb27`; the integration rejects any V4
config whose semantic hash differs from `f29f805deee9189c`. V4 does not accept
runtime chunk-size or threshold parameters.

The current parsers return flat text. The normalization adapter therefore maps
blank-line-delimited parser blocks to ordered canonical paragraphs and does not
guess headings, pages, tables, lists, or visuals. If a parser supplies structured
unit metadata, the same adapter preserves those fields directly.

To run the minimal product demo:

1. Install `requirements.txt` and start `python app.py` (or `python -m wsgi`).
2. Create one knowledge base with chunker `legacy` and another with `v4`.
3. Upload a document to either knowledge base and ask a question from the chat.
4. Open `/documents` to inspect the stored chunks and retrieval results.
5. To compare the same query, select each knowledge base in turn in Retrieval
   Experimentation and run the identical query.

The first V4 ingestion may download `intfloat/multilingual-e5-base`; subsequent
boundary embeddings use `.cache/boundary-embeddings`.

### Retrieval profiles

`RETRIEVAL_PROFILE=legacy` preserves the existing chat_rag retrieval behavior.
`RETRIEVAL_PROFILE=benchmark_aligned` selects the Phase 4/5 profile pinned at
commit `1e7f7186c13729c739ccb3170da0892f7350cb27`: multilingual E5 role prefixes,
normalized deterministic long-text pooling, Unicode BM25, and equal-weight RRF
with a 100-result pool and `k=60`. Query expansion, contextualization, and
reranking are disabled in this profile. Its E5 model is loaded with
`local_files_only=true`, matching the frozen benchmark configuration.

Indexes are profile-specific because the embedding models and dimensions differ.
Use a new vector-database path/collection and re-ingest documents when changing
profiles; do not point `benchmark_aligned` at an index created by `legacy`.

## Running it

There are two entrypoints, and which one is running is not a detail.

| | Command | Server | Binds | Debugger |
|---|---|---|---|---|
| Development | `python app.py` | Werkzeug | `127.0.0.1` | on (`FLASK_DEBUG=false` turns it off) |
| Production | `python -m wsgi` | waitress | `0.0.0.0` | none |

`python app.py` is for a developer at a keyboard: it keeps the reloader and the
traceback page, and it listens on loopback only so neither is offered to the
network the machine has joined. It is not a production runtime and is no longer
what a deployment reaches -- the container's `CMD` is `python -m wsgi`.

`python -m wsgi` serves the same application on waitress: **one process** with a
bounded pool of request threads (`WAITRESS_THREADS`, default 8), plus the one
background thread that packages documents for the Viewer. One process is a
deliberate choice, not a limitation of the server -- the packaging queue lives
in memory, the per-knowledge-base pipeline cache is a module global, and the
vector store is an embedded database rather than a database server, so a second
worker process would duplicate all three. `wsgi.py` says so in more detail.

It stops on SIGTERM (what `docker stop` and service managers send) as well as on
Ctrl+C, draining in-flight requests first.

### Where state goes

One setting decides: `CHAT_RAG_DATA_DIR`. Set it, and the knowledge base
records, the ingest ledger, the gold set, the vector stores, the parser's
canonical-unit cache and the logs all live under it. Leave it unset -- a local
checkout -- and every path stays exactly where it has always been, relative to
the working directory.

`VECTOR_DB_PATH` still names the fallback vector store outright, for a
deployment that really does keep it elsewhere. But it is honoured only from the
actual environment: a value for it in `.env` is ignored once a data directory
has been declared, because `.env` describes a developer's own layout and a
deployment that has named its data directory has not asked for that layout. The
start-up banner says which paths are in effect and names anything it refused.

Two checks prove a build can run at all, both cheap enough for the image build:

```bash
python tools/import_smoke.py   # the declared dependencies satisfy every import
python tools/serve_smoke.py    # `python -m wsgi` binds, answers /api/health, stops
```

### Does a clean clone work?

Green tests do not answer that. They once stayed green through a
`requirements.txt` that could not be installed at all, because every machine
running them already had the package and an editable checkout of the sibling
library. So there is a separate gate, and it is one command:

```bash
python tools/verify_reproducibility.py
```

It clones this repository from the remote, checks the clone carries no state
from your machine, installs the pinned `amsc` revision into a fresh Python 3.11
environment, imports it, builds the Viewer v3 product shell from it, then
builds and runs the container and asks it for `/api/health`. Every check
reports PASS, FAIL or SKIP -- SKIP means a capability is missing (no Docker, no
Python 3.11, no network) or a tier was not asked for, never that something was
checked and forgiven.

| Flag | What it adds |
|---|---|
| `--with-host-install` | installs `requirements.txt` into a fresh venv on this machine as well (several minutes, ~1 GB of wheels) |
| `--local` | clones this checkout instead of the remote, to run the gate before pushing |
| `--no-docker` | skips the container checks |
| `--keep` | leaves the temporary clone and environments behind for inspection |

The same command runs in CI on every push to `main` and
`refactor/productionization` (`.github/workflows/reproducibility.yml`), so what
fails there fails here too, with the same output.

## Quick Start

### 1. Installation (5 minutes)

```bash
# Clone and navigate
git clone <repository-url>
cd chat_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup NLTK data
python setup_nltk.py

# Configure environment
cp env.example .env
# Edit .env with your credentials
```

### 2. Choose Your Interface

**Option A: CLI Chat (Simple)**
```bash
# Add documents to ./documents folder
# Start CLI application
python main_new.py
```

**Option B: Web Application (Full Features)**
```bash
# Development server (reloader and debugger, loopback only)
python app.py
# Open browser: http://127.0.0.1:5005
```

For anything that is not a developer at a keyboard, run the production server
instead -- see [Running it](#running-it):
```bash
python -m wsgi
```

### 3. Start Chatting

- CLI: Type questions directly in the terminal
- Web: Use the browser interface to chat and manage documents

See detailed usage sections below for more information.

## Usage

### CLI Chat Application

The command-line interface provides an interactive chat experience with automatic document ingestion.

**Start the CLI application:**
```bash
python main_new.py
```

**What happens when you start:**
1. Configuration is loaded from `.env` file
2. RAG pipeline is initialized with your settings
3. Documents are automatically scanned from `DOCUMENTS_INPUT_PATH` (default: `./documents`)
4. New documents are ingested (already processed documents are skipped)
5. Interactive chat session begins

**Available CLI Commands:**
- Type your question and press Enter to chat
- `help` - Show available commands
- `stats` - Display document statistics (total documents, chunks, size, etc.)
- `clear` - Clear conversation history
- `exit` or `quit` - Exit the application

**Example CLI Session:**
```bash
$ python main_new.py

================================================================================
  RAG CONVERSATIONAL CHAT
  Retrieval-Augmented Generation with Document Ingestion
================================================================================

⚙️  Loading configuration...
✓ Configuration loaded

🚀 Initializing RAG pipeline...
✓ Pipeline initialized

================================================================================
DOCUMENT INGESTION
================================================================================

📂 Scanning for documents in: ./documents
   Found 2 new document(s)
   Skipping 0 already ingested document(s)

📥 Ingesting new documents...

[1/2] Processing: report.pdf
  ✓ Success: 15 chunks created

[2/2] Processing: notes.pdf
  ✓ Success: 22 chunks created

✓ Successfully ingested 2 new document(s)

📊 DOCUMENT STATISTICS
================================================================================
Total Documents: 2
Total Chunks: 37
Total Size: 2.45 MB

================================================================================
💬 CHAT MODE
================================================================================

You: What is the main topic?
Assistant: The main topic covers project documentation and requirements...

📚 Show sources? (y/n): y

📄 Sources:
1. report.pdf
   Section: Introduction
   Relevance Score: 0.856
   Preview: The document discusses...

You: 
```

### Web Application

The web application provides a modern browser-based interface with advanced features.

**Start the web application:**
```bash
python app.py
```

**Access the application:**
Open your browser to: `http://localhost:5005`

**Web Application Features:**
- 🎨 Modern UI with gradient design
- 💬 Real-time conversational chat with multi-turn support
- 📚 Source citations with relevance scores
- 📊 Document and knowledge base statistics
- 📁 Document management (upload, view, delete)
- 🔍 Chunk browsing and editing
- 🗄️ Multiple knowledge base support
- 🗑️ Clear conversation history
- 📱 Fully responsive design

**Important Notes:**
- The web app runs on port **5005** (not 5000)
- Documents can be managed through the web interface at `/documents`
- Multiple knowledge bases can be created and managed
- Each knowledge base can have its own vector database, embedding model, and chunker configuration

### Knowledge Base Management

The system supports multiple knowledge bases, each with its own configuration:

**Creating a Knowledge Base (via Web UI):**
1. Click "➕ New KB" button in the web interface
2. Configure:
   - Name: Descriptive name for the KB
   - Vector DB Provider: chroma or faiss
   - Embedding Model: Model name for embeddings
   - Chunker Config: Chunking parameters
   - Vector DB Path: Storage location (optional)

**Using Knowledge Bases:**
- Each KB has a unique ID
- Documents are ingested into specific KBs
- Queries can target specific KBs or use the default
- KBs can be managed through the web interface

**Programmatic KB Management:**
```python
from components.knowledgebase.manager import KnowledgeBaseManager

kb_manager = KnowledgeBaseManager()

# Create a new KB
kb = kb_manager.create(
    name="Technical Documentation",
    vector_db_provider="faiss",
    embedding_model_name="all-MiniLM-L6-v2"
)

# List all KBs
all_kbs = kb_manager.list()

# Get a specific KB
kb_config = kb_manager.get(kb_id="abc12345")

# Update a KB
kb_manager.update(kb_id="abc12345", updates={"name": "Updated Name"})

# Delete a KB
kb_manager.delete(kb_id="abc12345")
```

### Document Ingestion

**Automatic Ingestion (CLI):**
- Documents in `./documents` folder are automatically ingested on startup
- Already processed documents are skipped (tracked in `.ingested_documents.json`)

**Manual Ingestion (Web UI):**
- Open a knowledge base and use **Upload Document**
- Choose the chunking mode per document: **Standard** or **Deep Analysis**
- Documents are processed and indexed automatically

**Chunking modes (chosen at upload, never at query time):**

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

**Programmatic Ingestion:**
```python
from pipeline import RAGPipeline
from config import Settings

settings = Settings()
pipeline = RAGPipeline(settings=settings)

# Ingest from directory
results = pipeline.ingest_documents_from_directory(
    directory_path="./my_documents",
    recursive=True
)

# Ingest single file
chunks = pipeline.ingest_document_from_file("./documents/report.pdf")
```

### Basic Usage

```python
from pipeline import RAGPipeline
from config import Settings

# Initialize
settings = Settings()
rag_pipeline = RAGPipeline(settings=settings)

# Ingest documents from default directory (set in .env: DOCUMENTS_INPUT_PATH)
results = rag_pipeline.ingest_documents_from_directory()

# Or specify a custom directory
results = rag_pipeline.ingest_documents_from_directory(
    directory_path="./my_documents",
    recursive=True,
    file_pattern="*.pdf"  # Optional: filter by file type
)

# Ingest a single document
chunks = rag_pipeline.ingest_document_from_file("./documents/report.pdf")

# Retrieve relevant information
results, metadata = rag_pipeline.retrieve(
    query="What is John Smith's role?",
    top_k=5
)

# Format results for LLM
context = rag_pipeline.get_retrieval_context(results)
print(context)
```

### Conversational Usage

```python
# Turn 1
results1, _ = rag_pipeline.retrieve("What is John Smith's role?")
assistant_response = "John Smith is a Senior Software Engineer."
rag_pipeline.add_assistant_response(assistant_response)

# Turn 2 - Ambiguous reference resolved automatically
results2, _ = rag_pipeline.retrieve("How old is he?")
# System automatically clarifies to "How old is John Smith?"

# View conversation history
print(rag_pipeline.get_conversation_summary())
```

### Custom Components

You can replace any component with your own implementation:

```python
from components.llm import BaseLLM
from components.embedding import BaseEmbedding
from components.reranker import CrossEncoderReranker

# Use cross-encoder reranker for better performance
cross_encoder = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
rag_pipeline = RAGPipeline(reranker=cross_encoder, settings=settings)

# Custom LLM
class MyCustomLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Your implementation
        pass
    
    def get_name(self):
        return "MyCustomLLM"
    
    def get_model_name(self):
        return "custom-model"

# Use custom component
custom_llm = MyCustomLLM()
rag_pipeline = RAGPipeline(llm_model=custom_llm, settings=settings)
```

## Adding New Components

### Adding a New LLM Provider

1. Create a new file: `components/llm/my_llm.py`
2. Implement `BaseLLM` interface
3. Import in `components/llm/__init__.py`
4. Use in pipeline initialization

```python
from components.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Implementation
        pass
    
    def get_name(self):
        return "MyLLM"
    
    def get_model_name(self):
        return "my-model-v1"
```

### Adding a New Vector Database

1. Create: `components/vectordb/my_vectordb.py`
2. Implement `BaseVectorDB` interface
3. Import in `components/vectordb/__init__.py`

## Design Principles

This codebase follows these key principles:

1. **Abstraction**: No hardcoded technology-specific code in main components
2. **Separation of Concerns**: Data access, business logic, and presentation are separated
3. **Strategy Pattern**: Algorithms are pluggable (retrieval, chunking, etc.)
4. **Interface Segregation**: Components depend only on methods they use
5. **Modularity**: Code is organized in small, focused modules (< 500 lines per file)
6. **Centralized Configuration**: All config through central settings module
7. **Resilience**: Exception handling and graceful degradation
8. **Extensibility**: Easy to add new components without changing existing code

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run end-to-end tests
python -m pytest tests/e2e/

# Run example with cross-encoder reranker
python examples/cross_encoder_reranker_example.py
```

## Reranking Strategies

The system supports two reranking strategies:

1. **LLM Reranker**: Uses language model for relevance assessment (flexible but slower)
2. **Cross-Encoder Reranker**: Uses specialized cross-encoder model (fast and accurate)

See [Reranker Guide](docs/RERANKER_GUIDE.md) for detailed comparison and usage.

**Quick Start with Cross-Encoder:**
```python
# Set in .env
RERANKER_TYPE=cross_encoder

# Or in code
from components.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()
pipeline = RAGPipeline(reranker=reranker)
```

## Logging and Metrics

The system includes:
- Standard logging for debugging
- Token usage tracking
- Performance metrics
- Input/output logging for LLM calls

## Health Checks

For API deployments, health check endpoints verify:
- LLM connectivity
- Vector database status
- Embedding model availability

## Contributing

1. Follow the existing code structure
2. Keep files under 500-600 lines
3. Keep functions/methods under 20-30 lines
4. Add unit tests for new components
5. Update documentation

## License

[Your License]

## Support

For issues and questions, please open a GitHub issue.

