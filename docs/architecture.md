# Architecture — what the system is, which repo owns what

The system is two repositories that ship as one product.

| repo | what it is | what it owns |
|---|---|---|
| **`chat_rag`** (this one) | the product: the `/api/v1` contract, the application under it, the Next.js console, and the runtime that serves them | uploads, ingest jobs, knowledge bases, retrieval, the answer chain, resource limits, observability, configuration, the demo launcher |
| **`chunk`** (`amsc-poc`) | the chunking library, installed as a pinned dependency | chunking methods and their registry, Deep Analysis, the canonical PDF adapter, the retrieval primitives the benchmark froze, the shared payload reader and the Viewer's retrieval engine, all research and benchmark code |

They are expected side by side (`../chunk`), and the console installs the
library from an immutable commit named in `requirements.txt`. Nothing in
`chat_rag` edits `amsc` in place; a library change is a commit, a push and a
pin bump — see [testing.md](testing.md#changes-that-cross-both-repos).

`chunk` never imports `chat_rag`. `chat_rag` imports only the twenty-odd
`amsc` modules declared as the console API — see
[Product, research, legacy](#product-research-legacy).

---

## What ships

Three containers, brought up by one command
(`docker compose up --build`, [README](../README.md#running-with-docker)):

```
      browser
         │  :3000
         ▼
  ┌──────────────┐   /api/v1/*    ┌──────────────┐   SQL + pgvector   ┌────────┐
  │  frontend    │───────────────▶│     app      │───────────────────▶│   db   │
  │  Next.js     │  CHAT_RAG_API_ │  FastAPI on  │   DATABASE_URL     │ pg16 + │
  │  the console │  URL, per req. │  uvicorn     │                    │ vector │
  └──────────────┘                └──────────────┘                    └────────┘
         │                               │  :5005                          ▲
         │                               └── /api/ops/metrics ──▶ operator │
         │                                                                 │
         └── the browser never learns the app's address ───────────────────┘
```

Three facts about that picture are load-bearing:

* **the console forwards, it does not redirect.** Every `/api/v1` call is
  same-origin from the browser's point of view, so there is no CORS grant and
  no preflight in front of an upload. `frontend/lib/api/proxy.ts` reads
  `CHAT_RAG_API_URL` **per request**, so one console image runs against any
  backend — it was a build-time `rewrites()` entry until Step 14, and the
  address was frozen into the image.
* **the schema is applied before the server serves**, by `tools/migrate.py`
  from the container entrypoint: it waits for a recovering database, holds an
  advisory lock so two containers cannot both migrate, and logs the revisions.
  Nothing in the application creates a table.
* **each service waits on the one below it being *healthy*,** not merely
  started. [operations.md](operations.md) has the table.

Ollama, when it is used, runs on the host and is reached at
`host.docker.internal`. Nothing in the code knows that address.

---

## The two runtime flows

Everything the product does is one of these two. Both are bounded end to end;
the numbers are in [configuration.md](configuration.md) and the behaviour under
load is in [operations.md](operations.md).

### Upload → a document you can search

```
POST /api/v1/documents
  │  read the multipart body                      interfaces/http/v1
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
POST /api/v1/queries
  │  read the body                                interfaces/http/v1
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

The repository root holds what serves; `src/chat_rag` holds what it serves.
The engine is a distribution of its own (`pyproject.toml`), so the boundary is
what the wheel contains rather than a rule asking to be respected: no web
framework can reach it, because none of it ships beside one.
`tests/unit/test_distribution.py` builds a real wheel and reads that claim off
the artifact — the library and its Alembic migrations are in it, `interfaces/`,
`asgi.py`, `cli/` and `tools/` are not, and `fastapi` is not a declared
dependency.

```
interfaces/http/          NOT IN THE PACKAGE.  The adapter, at the repository
  v1/                     root with asgi.py, cli/ and tools/.
      routers/            FastAPI: one router per product concept
      schemas/            the API's own Pydantic types
      errors.py           the one refusal-to-status table
      openapi.py          what the generated document cannot infer
  operator.py             one route, off the contract on purpose:
      │                   /api/ops/metrics, whose body is free to change
      │                   A router reads a request, calls one use case, and
      │                   turns the answer or the refusal into a wire shape.
      ▼
─── src/chat_rag/ ──────  THE PACKAGE.  `pip install chat-rag`.
      │
api/                      The published Python API: Engine, EngineConfig, and
      │                   the values its calls answer with.  A second caller
      │                   of the use cases, beside the adapter above — never
      │                   between it and them.
      │
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

### The public Python API

Two callers of the use cases, and they are peers. `interfaces/http` serves
`/api/v1` to a browser; [`chat_rag.api`](../src/chat_rag/api/__init__.py)
answers a Python program:

```python
from chat_rag import Engine, EngineConfig

with Engine(EngineConfig(retrieval_profile="hybrid_rrf")) as engine:
    kb = engine.knowledge_bases.create("Reports")
    document = kb.ingest("report.pdf")
    document.analysis().request()
    hits = kb.search("liquidity")
    answer = kb.ask("What changed?")
```

Neither goes through the other, and neither owns the vocabulary: both project
the same use-case dictionaries into their own types, so a Python rename is not
an HTTP break and an HTTP addition is not a Python one.

The facade adds three things and no behaviour. It **owns a container and a
runtime** (`build_services`, the same composition `asgi.py` performs), it
**owns a session id** — the pipeline cache's key. The HTTP surface uses one
shared entry per knowledge base for every caller (a pipeline is the knowledge
base's store handle and lexical index, not the caller's, and the surface
carries no session of its own); a library engine has an id of its own so two
engines in one process cannot share a cache entry — and it **runs
every call inside its own activation**, so the store, the budgets, the counters
and the packaging queue a call reaches for are that engine's. A rule it appears
to enforce lives in a use case; a refusal it raises is that use case's own
`application.errors` exception.

`EngineConfig` is the stated subset of `Settings`, not a second configuration
system: a field left unset means *this caller did not say*, and the answer is
then whatever `Settings.from_env()` reads, under the precedence `config` has
always had. `data_dir` is the one that makes two engines genuinely separate —
packaged analyses, staged uploads, the parser cache and both embedding caches
land under it. The database is not a file and is not covered by it: two
engines under two roots still share every row unless `database_url` says
otherwise.

The surface is **enforced**, not described. `chat_rag.api.__all__` is the
published list; `tests/unit/test_public_surface.py` checks that the exports are
exactly it, that both import paths agree, and that no internal type appears in
a public signature — reading resolved annotations rather than source text, and
holding parameters to a stricter rule than returns. What that promise means
across versions, and how a removal is announced, is
[library-api.md](library-api.md).

`tests/integration/test_public_api.py` drives the whole flow — ingest, analyse,
compare, search, ask — over the real PostgreSQL, with only the answer model and
the embedding model replaced.

### What a running engine owns

Seven things outlive any one call: the connection pool, three provider
budgets, the metrics registry, the Viewer packager's queue and worker, and the
analysis-query engine. Each was a module global built on first use — right for
one process with one configuration, and wrong for a library, where a second
`Services` would have spent the first one's provider slots and written to its
database.

They are a **`Runtime`** now ([chat_rag/runtime.py](../src/chat_rag/runtime.py)),
and a `Services` owns one:

```
Services ── runtime ──┬── database        one pool, from settings.database
                      ├── provider_budget | embedding_budget | answer_budget
                      ├── metrics         this engine's counters
                      ├── packager        the Viewer queue, worker and locks
                      ├── analysis        the analysis-query engine and indexes
                      └── paths           where this engine's files go
```

Deep code is not handed one. `storage.session_scope()`, `limits.provider_budget()`,
`telemetry.metrics()` and every reader in `config.paths` kept their names and
resolve through `chat_rag.runtime`, which answers with the activated runtime if
there is one and the environment or the **process default** otherwise.

A second engine is reached by activation. `Services.activate()` sets it for a
block; a pipeline carries the engine that built it and activates it around its
construction and every operation; the two worker threads (ingest jobs, the
packager) are handed their runtime, because a `ContextVar` is not inherited by
a thread. The record stores skip all of that — they are handed their `Database`
when the container is composed.

**Files.** `runtime.paths` used to be a value the runtime reported and nothing
read: every writer — the packager, upload staging, the parser cache, both
embedding caches — resolved through the process environment, so a second engine
given its own data root still wrote into the first one's directories.
`config.paths.current()` resolves through the activated engine now, which is
what makes `EngineConfig(data_dir=…)` real. A *configured* engine is exactly
what it was configured with; an *environment-derived* one — which is what
`build_services()` with no settings composes, and therefore what the product
runs — reads the environment when asked, so no product path moved.

**Who speaks for the process.** The default is the runtime a caller that was
never handed one resolves to: Alembic, `tools/migrate.py`, a CLI command, a
test reaching a repository. `build_services(install_default=True)` is the
product's, and only the first container in a process is ever installed.
`chat_rag.api.Engine` passes `False`: being the first container in somebody
else's process is an accident of ordering, and taking the default on it would
hand this engine's pool, budgets, counters and packaging queue to code that
never asked — and then dispose that pool when the `with` block ended.

`tests/unit/test_engine_isolation.py` states the whole claim as behaviour: a
slot taken in one engine is not missing from the other, a trace recorded in one
is invisible in the other, a build queued in one is not in the other's queue,
and no two engines resolve any path to the same directory.
`tests/integration/test_public_api.py` runs two engines against two data roots
and looks at what is on disk afterwards.

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

`/api/v1` is **FastAPI**, served by `python -m asgi` on uvicorn, and it is the
whole HTTP surface bar one operator route. There was a second one until Step 13
— the Flask console, its rendered screens and the Viewer's relay, with the
FastAPI application mounted inside the WSGI process by a bridge — and
[legacy-removal.md](legacy-removal.md) is the record of what each of its
endpoints became. What is left is what that plan said would be left: this
package, and `asgi.py` as the entrypoint.

Nothing in `application/` or `api/` imports a web framework —
`tests/application` fails if either ever does — and nothing below
`interfaces/http/` builds a response. Three consequences worth stating:

* **the container is the seam.** `application/services.py` composes the whole
  application (`build_services()`), and every use case takes it as its first
  argument. The CLI calls `default_services()` and gets the same object the
  server is given, which is why `python -m cli` measures what the console
  would actually return — and why it imports no web framework at all.
* **a refusal is a meaning, not a code.** `application/errors.py` has six:
  invalid request, not found, conflict, unavailable, not-ready, processing
  failed. Overload, deadline and interruption are not among them because
  `core/exceptions.py` already owns those, raised by the subsystem that owns
  the limit. Exactly one file maps them to a status — `v1/errors.py` — and
  a router never names one beyond what its own decorator declares.
* **one surface, one application.** `/api/v1` is an adapter, never an
  implementation: it calls the same use cases the CLI does, over the same
  container. [api-v1.md](api-v1.md) is the contract; the one route beside it
  is `/api/ops/metrics`, which is deliberately not on it because its body
  reports internals that are free to change.

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
   either direction. It also fails on a route served *outside* `/api/v1`: there
   is one, it is declared there, and a second is a legacy surface growing back.

---

## Repository map — `chat_rag`

Only the parts worth knowing. Each row says what it owns and when you would
open it.

| path | owns | touch it when |
|---|---|---|
| `application/` | **the product's behaviour, with no web framework under it**: one module per behaviour group (`knowledge_bases`, `documents`, `ingest`, `chunks`, `query`, `workspace`, `catalogue`, `analysis_query`, `ops`), plus `errors.py` (what a refusal means) and `services.py` (the container everything is handed) | changing what the product *does* |
| `api/` | **the published Python API**: `engine.py` (`Engine` — its own `Services`, `Runtime` and session id), `config.py` (`EngineConfig`, the stated subset of `Settings`), `resources.py` (`KnowledgeBase`, `Document`, `IngestJob`, `Analysis`) and `results.py` (`Hit`, `Answer`, `Source`, `Health`, `Comparison`). Delegates only | changing what a Python caller can reach, or what it is called |
| `interfaces/http/v1/` | **the product contract** (`/api/v1`), as a FastAPI application: `routers/` (one per concept), `schemas/` (the API's own Pydantic types), `errors.py` (the one refusal-to-status table), `openapi.py` (the refusal body, the two headers and the status the document must not advertise), `application.py` (the app and its lifespan) — see [api-v1.md](api-v1.md) | adding or changing a supported endpoint |
| `interfaces/http/operator.py` | the one route served outside the contract: `GET /api/ops/metrics`, kept at its old path and body through the removal of the surface that used to serve it | changing what an operator can read |
| `interfaces/http/__init__.py` | `create_app()` — the contract plus that route, over one container — and the walker that answers "what does this application serve" for the tests and the documents | adding a surface beside the contract |
| `runtime/bootstrap.py` | what a process does before it serves: the banner, restart settlement, the staging sweep | changing start-up or restart behaviour |
| `asgi.py` | **the entrypoint** (`python -m asgi`, uvicorn, one process): the composed container, the application, the lifespan and the worker-thread pool | changing how the server starts, stops or sizes itself |
| `docker-entrypoint.sh`, `tools/migrate.py` | what happens between "the container started" and "the server serves": wait for the database, take an advisory lock, bring the schema to head, then `exec` the entrypoint above | changing how a deployment gets its schema |
| `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | the deployment: two images and the three services, their health checks and their start-up order. Read as a contract by `tests/unit/test_deployment.py` | changing how the system is shipped or brought up |
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
| `tests/` | `unit/` (fast, no network), `application/` (the use cases with no web framework at all), `integration/` (the real application over a test client, and two that start the real server), `migration/` (the contracts a platform change must keep), `storage/` (the repositories, the invariants, the transactions, the concurrency and the Alembic gate), `conftest.py` (moves the process out of the checkout, blanks keys, builds and truncates the test database) | always |
| `evaluation/experiment-log.md` | the record of the retrieval experiments behind the shipped context budget and top-k | asking why a number is what it is |
| `frontend/` | **the console**: a Next.js application over `/api/v1` and nothing else. `app/viewer/` and `components/viewer/` are the Viewer's five screens; `lib/viewer/rows.ts` is the alignment rule the comparison is built on | changing a screen |
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
