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
  │  validate, stage the file, hash it            app.py
  ▼
ingest job (queued → running)                     components/ingest/jobs.py
  │
  ├── parse            PDF → canonical units      components/parsers/structured_pdf_parser.py
  │                    cached by content hash     .cache/canonical-units/
  ├── chunk            units → chunk rows         components/chunker/factory.py → amsc
  │     Standard       amsc.structural_chunker
  │     Deep Analysis  amsc.deep_pipeline          proposer → selector → verifier
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
  │  admission: a slot now, or 503 now            components/query/limits.py
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

## Repository map — `chat_rag`

Only the parts worth knowing. Each row says what it owns and when you would
open it.

| path | owns | touch it when |
|---|---|---|
| `app.py` | the Flask app: every route, and the process-wide singletons (pipeline cache, ingest manager, budgets, admission) built at import | adding an endpoint, changing what a route returns |
| `wsgi.py` | the production entrypoint (`python -m wsgi`, waitress, one process) | changing how the server is served or shut down |
| `pipeline/rag_pipeline.py` | the pipeline object: builds the embedder / store / retriever / answer model from settings, and runs a query | changing retrieval or the answer chain end to end |
| `components/ingest/jobs.py` | the job system: queue, workers, states, retention, cancellation | changing upload concurrency or job lifecycle |
| `components/ingest/limits.py` | **the one owner of deadlines and provider budgets**: `current_guard()`, `deadline_timeout()`, the Deep and embedding semaphores | adding a transport, or anything that calls out over the network |
| `components/ingest/journal.py` | the on-disk record of job transitions, so a restart can answer for a job | changing restart semantics |
| `components/ingest/pipelines.py` | the bounded pipeline cache and its leases | changing caching or eviction |
| `components/query/limits.py` | query admission, the answer budget, `QueryGuard` | changing what a busy server does to a question |
| `components/observability/` | `telemetry.py` (traces, stages, error categories, the bounded window) and `events.py` (the `RAG.ops` one-line event log) | adding a metric or an operational event |
| `components/chunker/factory.py` + `registry.py` | which **indexing** chunker a knowledge base may be created with (`structure_first`, `v4`) — deliberately *not* the analysis-method registry | adding an indexing chunker |
| `components/viewer/methods.py` | this deployment's view of `amsc.methods`: availability on this machine, display order, the default | changing which analysis methods are offered |
| `components/viewer/analysis.py` | the packaging worker and each document's `missing`/`pending`/`running`/`ready`/`failed` state | debugging a Viewer package |
| `components/retriever/` | the retrieval profiles (`bm25_only`, `hybrid_rrf`, `benchmark_aligned`) | changing how candidates are found or fused |
| `components/llm/` | the answer transports: OpenAI-compatible, Ollama, Azure, the unavailable carrier and the fallback pair | adding a provider — see [Adding a provider](#adding-a-provider) |
| `components/parsers/` | file → text/units, and the parser factory that picks one | adding a file type |
| `config/` | five owners, one each: `paths` (where state goes), `runtime` (server), `ingest`, `query`, `settings` (models, endpoints, retrieval) | any setting — read [configuration.md](configuration.md) first |
| `utils/logger.py` | the sixth config owner: log levels and rotation, deliberately fail-safe | changing logging |
| `utils/document_tracker.py` | the ingest ledger, written atomically under a per-file lock | changing what a registered document records |
| `cli/` | `python -m cli` — eval, search, qa, inspect, report, gold | offline evaluation of a knowledge base |
| `tools/` | `import_smoke.py`, `serve_smoke.py`, `verify_reproducibility.py` | proving a build works — see [testing.md](testing.md) |
| `tests/` | `unit/` (fast, no network), `integration/` (the real Flask app), `migration/` (the contracts a platform change must keep), `conftest.py` (moves the process out of the checkout, blanks keys) | always |
| `evaluation/experiment-log.md` | the record of the retrieval experiments behind the shipped context budget and top-k | asking why a number is what it is |
| `templates/`, `static/` | the console UI | changing a screen |
| `start-demo.ps1` / `stop-demo.ps1` | the demo launcher: builds the Viewer shell if missing, starts both servers, waits for health | running the demo |

## Repository map — `chunk`

| path | owns | touch it when |
|---|---|---|
| `src/amsc/methods.py` | **the chunking-method registry** — one `ChunkMethod` per method: wire key, engine kind, product label, summary, capabilities, partition callable | adding an analysis method ([adding-a-chunker.md](../../chunk/docs/adding-a-chunker.md)) |
| `src/amsc/structural_chunker.py` | Standard: the frozen structure-first walk | never, lightly — it is the baseline every method is compared against |
| `src/amsc/deep_pipeline.py` | Deep Analysis, the production entry point: `chunk_document(units, mode=…)` | changing Deep's orchestration |
| `src/amsc/deep_proposer.py` / `deep_verifier.py` / `deep_analysis.py` | its proposer, its double-order verifier, its deterministic selector | as above |
| `src/amsc/provider_calls.py` | how a generative provider is called, and nothing about chunking: the protocol, the OpenAI-compatible transport, cache-first parallel calls | changing provider transport |
| `src/amsc/checkpoint_adapter.py` + `prepare_full_checkpoint.py` | PDF → canonical units, and the manifest that pins them | changing canonical extraction |
| `src/amsc/viewer_corpus.py` | **the payload reader both Viewer pages share** — the cross-repo data contract | changing what the Viewer reads |
| `src/amsc/viewer_v3.py` + `viewer_v3_template.py` | the Viewer product page and its build | changing the Viewer |
| `src/amsc/viewer_server.py` | the Viewer's own server process (`python -m amsc.viewer_server`) | changing how the Viewer is served |
| `src/amsc/surface.py` | the product / service / research / legacy declaration, enforced against the real import graph | adding a module off the product path |
| `src/amsc/io.py` | reading and writing artifact files, including `sha256_file` | changing artifact I/O |
| `src/amsc/example_chunker.py` | the documented template for a new method — copy it | adding a method |
| `src/amsc/chunk_method.py` | the `ChunkMethod` / `PartitionResult` types, in a leaf module so a method module can import them and the registry can import the method | adding a method |
| `evaluation/` | frozen benchmark results, pinned by hash | never; it is a record |
| `configs/` | benchmark and checkpoint configurations, inputs to frozen runs | running a benchmark |

---

## Product, research, legacy

`amsc` is one flat namespace holding product code, experiments and legacy
side by side. Which is which is **declared** in `src/amsc/surface.py` and
**enforced** against the real import graph, in both repos:

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
