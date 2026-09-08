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
  ├── index            vectors + lexical rows     components/vectordb/
  ├── ledger           the document is registered utils/document_tracker.py
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
  v1/                     THE CONTRACT.  The surface a client builds against
  legacy/                 compatibility: what the console still speaks
      │                   Both read a request, call one use case, and turn
      │                   the answer or the refusal into their own shape.
      ▼
application/              The product's behaviour.  Plain functions over a
      │                   Services container; no framework, no request,
      │                   no status codes.
      ▼
components/  config/  core/  pipeline/  utils/  amsc
                          Stores, models, limits, telemetry, the chunking
                          library.
```

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
2. add the route to the matching blueprint in `interfaces/http/v1/`: read the
   inputs, call the use case, return `resource(...)` or `collection(...)`,
   and project the result in `v1/resources.py`;
3. publish it in [api-v1.md](api-v1.md) — `tests/migration/test_http_surface.py`
   compares that document against the live routing table and fails on drift in
   either direction. (The same is true of the README's *console API* table for
   the compatibility surface, which should not be growing.)

---

## Repository map — `chat_rag`

Only the parts worth knowing. Each row says what it owns and when you would
open it.

| path | owns | touch it when |
|---|---|---|
| `application/` | **the product's behaviour, with no web framework under it**: one module per behaviour group (`knowledge_bases`, `documents`, `ingest`, `chunks`, `query`, `workspace`, `catalogue`, `goldsets`, `ops`), plus `errors.py` (what a refusal means) and `services.py` (the container everything is handed) | changing what the product *does* |
| `interfaces/http/v1/` | **the product contract** (`/api/v1`): the resource projections, the envelope and the refusal-to-status table a FastAPI port has to reproduce — see [api-v1.md](api-v1.md) | adding or changing a supported endpoint |
| `interfaces/http/legacy/` | the Flask-era surface the console and the Viewer's relay still speak; one directory to delete when they do not | keeping the current screens working |
| `runtime/bootstrap.py` | what a process does before it serves: the banner, restart settlement, the staging sweep, the development server's options | changing start-up or restart behaviour |
| `app.py` | the Flask application itself: the app object, the session key, CORS, and which container the blueprints are given | changing framework-level wiring |
| `wsgi.py` | the production entrypoint (`python -m wsgi`, waitress, one process) | changing how the server is served or shut down |
| `pipeline/rag_pipeline.py` | the pipeline object: builds the embedder / store / retriever / answer model from settings, and runs a query | changing retrieval or the answer chain end to end |
| `components/ingest/jobs.py` | the job system: queue, workers, states, retention, cancellation | changing upload concurrency or job lifecycle |
| `components/ingest/limits.py` | **the one owner of deadlines and provider budgets**: `current_guard()`, `deadline_timeout()`, the Deep and embedding semaphores | adding a transport, or anything that calls out over the network |
| `components/ingest/journal.py` | the on-disk record of job transitions, so a restart can answer for a job | changing restart semantics |
| `components/ingest/pipelines.py` | the bounded pipeline cache and its leases | changing caching or eviction |
| `components/query/limits.py` | query admission, the answer budget, `QueryGuard` | changing what a busy server does to a question |
| `components/observability/` | `telemetry.py` (traces, stages, error categories, the bounded window) and `events.py` (the `RAG.ops` one-line event log) | adding a metric or an operational event |
| `components/chunker/factory.py` + `registry.py` | which **indexing** chunker a knowledge base may be created with (`structure_first`, `v4`) — deliberately *not* the analysis-method registry | adding an indexing chunker |
| `components/viewer/methods.py` | this deployment's view of `amsc.chunking.registry`: availability on this machine, display order, the default | changing which analysis methods are offered |
| `components/viewer/analysis.py` | the packaging worker and each document's `missing`/`pending`/`running`/`ready`/`failed` state | debugging a Viewer package |
| `components/retriever/` | the retrieval profiles (`bm25_only`, `hybrid_rrf`, `benchmark_aligned`) | changing how candidates are found or fused |
| `components/llm/` | the answer transports: OpenAI-compatible, Ollama, Azure, the unavailable carrier and the fallback pair | adding a provider — see [Adding a provider](#adding-a-provider) |
| `components/parsers/` | file → text/units, and the parser factory that picks one | adding a file type |
| `config/` | five owners, one each: `paths` (where state goes), `runtime` (server), `ingest`, `query`, `settings` (models, endpoints, retrieval) | any setting — read [configuration.md](configuration.md) first |
| `utils/logger.py` | the sixth config owner: log levels and rotation, deliberately fail-safe | changing logging |
| `utils/document_tracker.py` | the ingest ledger, written atomically under a per-file lock | changing what a registered document records |
| `cli/` | `python -m cli` — eval, search, qa, inspect, report, gold | offline evaluation of a knowledge base |
| `tools/` | `import_smoke.py`, `serve_smoke.py`, `verify_reproducibility.py` | proving a build works — see [testing.md](testing.md) |
| `tests/` | `unit/` (fast, no network), `application/` (the use cases with no Flask at all), `integration/` (the real Flask app), `migration/` (the contracts a platform change must keep), `conftest.py` (moves the process out of the checkout, blanks keys) | always |
| `evaluation/experiment-log.md` | the record of the retrieval experiments behind the shipped context budget and top-k | asking why a number is what it is |
| `templates/`, `static/` | the console UI | changing a screen |
| `start-demo.ps1` / `stop-demo.ps1` | the demo launcher: builds the Viewer shell if missing, starts both servers, waits for health | running the demo |

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
| `src/amsc/viewer/build.py` + `template.py` | the Viewer product page and its build | changing the Viewer |
| `src/amsc/viewer/server.py` | the Viewer's own server process (`python -m amsc.viewer.server`) | changing how the Viewer is served |
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
