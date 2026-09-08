# Architecture — what the system is, which repo owns what

The system is two repositories that ship as one product.

| repo | what it is | what it owns |
|---|---|---|
| **`chat_rag`** (this one) | the product: a Flask console, its HTTP API, the runtime that serves it | uploads, ingest jobs, knowledge bases, retrieval, the answer chain, resource limits, observability, configuration, the demo launcher |
| **`chunk`** (`amsc-poc`) | the chunking library, installed as a pinned dependency | chunking methods and their registry, Deep Analysis, the canonical PDF adapter, the retrieval primitives the benchmark froze, the Viewer page builder and its server, all research and benchmark code |

They are expected side by side (`../chunk`), and the console installs the
library from an immutable commit named in `requirements.txt`. Nothing in
`chat_rag` edits `amsc` in place; a library change is a commit, a push and a
pin bump — see [testing.md](testing.md#changes-that-cross-both-repos).

`chunk` never imports `chat_rag`. `chat_rag` imports only the twenty-odd
`amsc` modules declared as the console API — see
[Product, research, legacy](#product-research-legacy).

---

## The two runtime flows

Everything the product does is one of these two. Both are bounded end to end;
the numbers are in [configuration.md](configuration.md) and the behaviour under
load is in [operations.md](operations.md).

### Upload → a document you can search

```
POST /api/documents/upload
  │  read the multipart body                      interfaces/http/{v1,legacy}
  │  validate, stage the file, hash it            application/ingest.py
  ▼
ingest job (queued → running)                     components/ingest/jobs.py
  │
  ├── parse            PDF → canonical units      components/parsers/structured_pdf_parser.py
  │                    cached by content hash     .cache/canonical-units/
  ├── chunk            units → chunk rows         components/chunker/factory.py → amsc
  │     Standard       amsc.chunking.structural
  │     Deep Analysis  amsc.deep.pipeline          proposer → selector → verifier
  ├── embed            chunk text → vectors       components/embedding/
  ├── index            vectors + lexical rows     components/vectordb/   (files)
  ├── ledger           the document is registered utils/document_tracker.py
  │                    -> a row in PostgreSQL     storage/
  └── viewer stage     queue the analysis         components/viewer/analysis.py
        ▼
      background worker packages the run for the Viewer  (never blocks the request)
```

The ledger write is the job's **last** act, which is what makes a restart
answerable: a document in the ledger was fully committed, and anything else
was not. Nothing is resumed — see *Restart* in
[operations.md](operations.md#bounded-ingest-uploads-are-jobs).

Each stage checks the job deadline at its boundary, and every outbound model
call is made with the time the job has left. Deep Analysis and embedding calls
are additionally capped process-wide, by separate budgets.

### Question → an answer with sources

```
POST /api/query
  │  read the body                                interfaces/http/{v1,legacy}
  │  admission: a slot now, or 503 now            application/query.py
  │                                               components/query/limits.py
  ▼
retrieve            dense (vectors) + BM25 (lexical)    components/retriever/
  │                 fused by RRF on the hybrid_rrf profile
  ├── context       pick sources, label [S1]…, budget    components/context/assembler.py
  └── answer        one completion, cites the labels     components/llm/
        ▼
      response + metadata.query (timings, models, hit counts, query_id)
```

The whole question runs **on the request thread that received it** — that is
why the limits are about request threads. No ingest-time model runs during a
question: chat reads the chunks that were written at upload.

---

## The application boundary

One direction, and it is the whole rule:

```
interfaces/http/
  v1/                     THE CONTRACT.  The surface a client builds against.
      routers/            FastAPI: one router per product concept
      schemas/            the API's own Pydantic types
      errors.py           the one refusal-to-status table
      openapi.py          what the generated document cannot infer
  legacy/                 compatibility: Flask, what the console still speaks
  coexistence.py          one process, both frameworks — for this step only
      │                   Both surfaces read a request, call one use case, and
      │                   turn the answer or the refusal into their own shape.
      ▼
application/              The product's behaviour.  Plain functions over a
      │                   Services container; no framework, no request,
      │                   no status codes.
      ▼
components/  config/  core/  pipeline/  utils/  amsc
      │                   Stores, models, limits, telemetry, the chunking
      │                   library.
      ▼
storage/                  PostgreSQL, behind repository interfaces.  Every SQL
                          statement this application runs is in one module;
                          nothing above it imports SQLAlchemy.
```

### One store, and what is in it

```
application / domain
   └── PostgreSQL          knowledge bases, documents (the ingest ledger),
                           content identity and analysis state, ingest jobs,
                           the gold set,
                           chunk rows + metadata + embeddings (pgvector)
```

**PostgreSQL is authoritative for everything durable.** Step 8 moved the
relational records off the filesystem; Step 9 moved the vectors, and there is
no second store to keep in step -- deleting a knowledge base takes its
vectors with it in the same transaction, by a foreign key rather than by a
caller remembering to remove a directory.

The lexical (BM25) index is unchanged and is not stored anywhere: it is built
in memory from the chunk rows, which are now read out of `chunk_vectors`
instead of out of a Chroma collection. The retrieval algorithm did not move.

The large regenerable artifacts of document processing -- the canonical units,
each method's packaged `chunks.jsonl`, a Deep run tree, the assembled Viewer
payload, the caches -- are still files under the data directory, addressed by
a *row* (`contents.content_key` names the directory) rather than being the
record. [database.md](database.md) is the schema, the migrations and the
reasoning.

`/api/v1` is **FastAPI** and the console's surface is still **Flask**, in one
process over one container. `interfaces/http/coexistence.py` mounts the ASGI
application inside the WSGI one and registers its routes — read from FastAPI's
own table, never written out twice — so `python -m wsgi` keeps serving both.
`asgi.py` is the same FastAPI application standing alone, which is what a
deployment runs once nothing needs the console's screens. There is no
synchronisation between the two surfaces and there is nothing to synchronise:
they are two adapters over one application.

Nothing in `application/` imports Flask — `tests/application` fails if it
ever does — and nothing below `interfaces/http/` builds a response. Three
consequences worth stating:

* **the container is the seam.** `application/services.py` composes the whole
  application (`build_services()`), and every use case takes it as its first
  argument. The CLI calls `default_services()` and gets the same object the
  Flask app was given, which is why `python -m cli` measures what the console
  would actually return — and why it now imports no web framework at all.
* **a refusal is a meaning, not a code.** `application/errors.py` has six:
  invalid request, not found, conflict, unavailable, not-ready, processing
  failed. Overload, deadline and interruption are not among them because
  `core/exceptions.py` already owns those, raised by the subsystem that owns
  the limit. Each adapter has exactly one file that maps them to a status —
  `v1/envelope.py` and `legacy/responses.py` — and the two are free to
  disagree, which they do: an analysis that is selected but not built yet is
  a **409** `not_ready` on the contract and a **404** on the compatibility
  surface, because that is what the console was written against.
* **two surfaces, one application.** `/api/v1` is a second adapter, never a
  second implementation: both call the same use cases, so a knowledge base
  created through one is the record the other lists, with nothing
  synchronising them. [api-v1.md](api-v1.md) is the contract, and
  `interfaces/http/legacy/` is the directory to delete when nothing speaks
  the old surface any more.

### Adding an endpoint

1. put the decision in the `application/` module for its behaviour group, and
   raise from `application/errors.py` when it refuses;
2. add the route to the matching router in `interfaces/http/v1/routers/`:
   read the inputs, call the use case, return a model from
   `interfaces/http/v1/schemas/`. Declare the response model on the route —
   the schemas forbid undeclared fields, which is what keeps a storage
   column out of an answer. Do not catch the refusal: `v1/errors.py` is the
   only place a status code is decided;
3. publish it in [api-v1.md](api-v1.md) — `tests/migration/test_http_surface.py`
   compares that document against the live routing table and fails on drift in
   either direction. (The same is true of the README's *console API* table for
   the compatibility surface, which should not be growing — and a route added
   there is also a row in [legacy-removal.md](legacy-removal.md), which
   `tests/migration/test_legacy_removal_map.py` holds to the same standard.)

---

## Repository map — `chat_rag`

Only the parts worth knowing. Each row says what it owns and when you would
open it.

| path | owns | touch it when |
|---|---|---|
| `application/` | **the product's behaviour, with no web framework under it**: one module per behaviour group (`knowledge_bases`, `documents`, `ingest`, `chunks`, `query`, `workspace`, `catalogue`, `goldsets`, `ops`), plus `errors.py` (what a refusal means) and `services.py` (the container everything is handed) | changing what the product *does* |
| `interfaces/http/v1/` | **the product contract** (`/api/v1`), as a FastAPI application: `routers/` (one per concept), `schemas/` (the API's own Pydantic types), `errors.py` (the one refusal-to-status table), `openapi.py` (the refusal body, the two headers and the status the document must not advertise), `application.py` (the app and its lifespan) — see [api-v1.md](api-v1.md) | adding or changing a supported endpoint |
| `interfaces/http/legacy/` | the Flask-era surface the old rendered screens still speak; one directory to delete when they are gone | keeping those screens working |
| `interfaces/http/coexistence.py` | the ASGI-inside-WSGI bridge that lets one process serve both surfaces over one container; temporary, and deleted with the legacy directory | debugging why a `/api/v1` request behaves differently through the console's port |
| `runtime/bootstrap.py` | what a process does before it serves: the banner, restart settlement, the staging sweep, the development server's options | changing start-up or restart behaviour |
| `app.py` | the Flask application itself: the app object, the session key, CORS, and which container the blueprints are given | changing framework-level wiring |
| `wsgi.py` | the production entrypoint today (`python -m wsgi`, waitress, one process, both surfaces) | changing how the server is served or shut down |
| `asgi.py` | the ASGI entrypoint (`python -m asgi`, uvicorn): `/api/v1` alone, and what a deployment runs once the console's screens are gone | changing FastAPI's own start-up, shutdown or thread pool |
| `pipeline/rag_pipeline.py` | the pipeline object: builds the embedder / store / retriever / answer model from settings, and runs a query | changing retrieval or the answer chain end to end |
| `components/ingest/jobs.py` | the job system: queue, workers, states, retention, cancellation | changing upload concurrency or job lifecycle |
| `components/ingest/limits.py` | **the one owner of deadlines and provider budgets**: `current_guard()`, `deadline_timeout()`, the Deep and embedding semaphores | adding a transport, or anything that calls out over the network |
| `components/ingest/journal.py` | the record of job transitions (`ingest_jobs`), so a restart can answer for a job | changing restart semantics |
| `components/ingest/pipelines.py` | the bounded pipeline cache and its leases | changing caching or eviction |
| `components/query/limits.py` | query admission, the answer budget, `QueryGuard` | changing what a busy server does to a question |
| `components/observability/` | `telemetry.py` (traces, stages, error categories, the bounded window) and `events.py` (the `RAG.ops` one-line event log) | adding a metric or an operational event |
| `components/chunker/factory.py` + `registry.py` | which **indexing** chunker a knowledge base may be created with (`structure_first`, `v4`) — deliberately *not* the analysis-method registry | adding an indexing chunker |
| `components/viewer/methods.py` | this deployment's view of `amsc.chunking.registry`: availability on this machine, display order, the default | changing which analysis methods are offered |
| `components/viewer/analysis.py` | the packaging worker and each document's `missing`/`pending`/`running`/`ready`/`failed` state | debugging a Viewer package |
| `application/analysis_query.py` | asking one document's analysis arms — the engine, its models and its bounds; the only path that compares chunkers | changing what the Viewer's *Sorgu* does |
| `components/retriever/` | the retrieval profiles (`bm25_only`, `hybrid_rrf`, `benchmark_aligned`) | changing how candidates are found or fused |
| `components/llm/` | the answer transports: OpenAI-compatible, Ollama, Azure, the unavailable carrier and the fallback pair | adding a provider — see [Adding a provider](#adding-a-provider) |
| `components/parsers/` | file → text/units, and the parser factory that picks one | adding a file type |
| `storage/` | **PostgreSQL**: `models.py` (the schema and the reasoning behind each edge), `engine.py` (one pooled engine, one session per unit of work), `repositories.py` (every SQL statement this application runs), `migrations/` (Alembic) | changing what is persisted — read [database.md](database.md) first |
| `config/` | six owners, one each: `paths` (where files go), `database` (`DATABASE_URL` and the pool), `runtime` (server), `ingest`, `query`, `settings` (models, endpoints, retrieval) | any setting — read [configuration.md](configuration.md) first |
| `utils/logger.py` | the sixth config owner: log levels and rotation, deliberately fail-safe | changing logging |
| `utils/document_tracker.py` | the ingest ledger, as a façade over `DocumentRepository`: a document is a row addressed by its `doc_id`, never by a path | changing what a registered document records |
| `cli/` | `python -m cli` — eval, search, qa, inspect, report, gold | offline evaluation of a knowledge base |
| `tools/` | `import_smoke.py`, `serve_smoke.py`, `verify_reproducibility.py`, `import_legacy_state.py` (a pre-Step-8 installation's JSON records into PostgreSQL) | proving a build works — see [testing.md](testing.md) |
| `tests/` | `unit/` (fast, no network), `application/` (the use cases with no Flask at all), `integration/` (the real Flask app), `migration/` (the contracts a platform change must keep), `storage/` (the repositories, the invariants, the transactions, the concurrency and the Alembic gate), `conftest.py` (moves the process out of the checkout, blanks keys, builds and truncates the test database) | always |
| `evaluation/experiment-log.md` | the record of the retrieval experiments behind the shipped context budget and top-k | asking why a number is what it is |
| `frontend/` | **the console**: a Next.js application over `/api/v1` and nothing else. `app/viewer/` and `components/viewer/` are the Viewer's five screens; `lib/viewer/rows.ts` is the alignment rule the comparison is built on | changing a screen |
| `templates/`, `static/` | the Flask-era screens, served beside the console until Step 13 | keeping an old screen working |
| `start-demo.ps1` / `stop-demo.ps1` | the demo launcher: starts the backend and the Next.js console, waits until each answers | running the demo |

## Repository map — `chunk`

The library's own map is
[chunk/docs/package-layout.md](../../chunk/docs/package-layout.md); this table
is only the part this console depends on.

| path | owns | touch it when |
|---|---|---|
| `src/amsc/chunking/registry.py` | **the chunking-method registry** — one `ChunkMethod` per method: wire key, engine kind, product label, summary, capabilities, partition callable | adding an analysis method ([adding-a-chunker.md](../../chunk/docs/adding-a-chunker.md)) |
| `src/amsc/chunking/structural.py` | Standard: the frozen structure-first walk | never, lightly — it is the baseline every method is compared against |
| `src/amsc/deep/pipeline.py` | Deep Analysis, the production entry point: `chunk_document(units, mode=…)` | changing Deep's orchestration |
| `src/amsc/deep/proposer.py` / `verifier.py` / `selector.py` | its proposer, its double-order verifier, its deterministic selector | as above |
| `src/amsc/providers.py` | how a generative provider is called, and nothing about chunking: the protocol, the OpenAI-compatible transport, cache-first parallel calls | changing provider transport |
| `src/amsc/canonical/adapter.py` + `prepare.py` | PDF → canonical units, and the manifest that pins them | changing canonical extraction |
| `src/amsc/viewer/corpus.py` | **the payload reader the Viewer and this console share** — the cross-repo data contract | changing what the Viewer reads |
| `src/amsc/viewer/chat/` | **the engine behind the Viewer's *Sorgu***: one retrieval index per analysis arm, and the comparison across them. Console API since Step 12 — `chat_rag` runs it in process | changing how an arm is retrieved or answered |
| `src/amsc/viewer/build.py` + `template.py` | the standalone Viewer page and its build. Not the product's Viewer any more; kept for that repository's own corpus | serving the frozen benchmark out of `chunk` |
| `src/amsc/viewer/server.py` | that page's server (`python -m amsc.viewer.server`). No part of running the product | as above |
| `src/amsc/surface.py` | the product / service / research / legacy declaration, enforced against the real import graph | adding a module off the product path |
| `src/amsc/document/io.py` | reading and writing artifact files, including `sha256_file` | changing artifact I/O |
| `src/amsc/chunking/example.py` | the documented template for a new method — copy it | adding a method |
| `src/amsc/chunking/method.py` | the `ChunkMethod` / `PartitionResult` types, in a leaf module so a method module can import them and the registry can import the method | adding a method |
| `evaluation/` | frozen benchmark results, pinned by hash | never; it is a record |
| `configs/` | benchmark and checkpoint configurations, inputs to frozen runs | running a benchmark |

---

## Product, research, legacy

`amsc` is organised by domain, and everything under `src/amsc/research/` is
off the product path. Which module is which is **declared** in
`src/amsc/surface.py`, by dotted path, and **enforced** against the real import
graph, in both repos:

* `chat_rag` product code may import only `surface.CONSOLE_API` — the modules
  the console genuinely calls. `tests/unit/test_amsc_surface.py` fails on
  anything else.
* Nothing reachable from a library entry point may be research or legacy.
  `chunk/tests/unit/test_library_surface.py` fails with the import chain that
  broke it.

The full explanation, the current console API and what each remaining legacy
module is kept for is
**[chunk/docs/library-surface.md](../../chunk/docs/library-surface.md)**.

Two rules of thumb: do not import an `amsc` module that is not in
`CONSOLE_API`, and do not build on anything under `RESEARCH` or `LEGACY` —
the benchmarks depend on those staying frozen, not on them staying useful.

---

## Adding a provider

The answer transports live in `components/llm/`. To add one:

1. implement `BaseLLM` (`generate`, `get_name`, `get_model_name`);
2. read the deadline: call `current_guard()` before the request and pass
   `deadline_timeout(configured)` as the socket timeout, both from
   `components.ingest.limits`. This is checked by
   `tests/unit/test_provider_surface.py`;
3. export it from `components/llm/__init__.py`;
4. select it by name in `RAGPipeline._build_answer_model`, and document that
   name in `env.example`. Both are checked by the same test — an adapter
   nothing can select is dead optionality and the suite says so.

Parsers work the same way at a smaller scale: implement `BaseParser`, and
either add it to `ParserFactory._register_default_parsers` or register it at
runtime with `ParserFactory.register_parser`.

Adding a **chunking method** is a library change, not a console change:
[chunk/docs/adding-a-chunker.md](../../chunk/docs/adding-a-chunker.md).

---

## Where to go next

| question | doc |
|---|---|
| How do I run it? | [../README.md](../README.md) |
| What does a setting do, and who owns it? | [configuration.md](configuration.md) |
| It is refusing uploads / queries. What now? | [operations.md](operations.md) |
| How do I test a change, and in what order across repos? | [testing.md](testing.md) |
| What is still not solved? | [limitations.md](limitations.md) |
| How does the Viewer work? | [chunk/docs/viewer-architecture.md](../../chunk/docs/viewer-architecture.md) |
| How do I add a chunking method? | [chunk/docs/adding-a-chunker.md](../../chunk/docs/adding-a-chunker.md) |
