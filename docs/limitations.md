# Known limitations

Everything here is true of the code as it stands. Each entry says what the
limit is, why it is where it is, and what would have to change — so the next
person can tell a deliberate boundary from an unfinished one.

Nothing already fixed is listed. This is not a changelog.

---

## Runtime shape

**One process, by design.** `python -m wsgi` runs a single waitress process
with a bounded thread pool (`WAITRESS_THREADS`, 8) plus one background thread
that packages documents for the Viewer. Three things make a second worker
process wrong rather than merely unnecessary: the Viewer packaging queue lives
in memory, the per-knowledge-base pipeline cache is a module global, and the
vector store is an embedded database rather than a database server. A second
process would duplicate all three and they would disagree.

*To scale out* you would need an external queue, a shared pipeline registry
and a vector store that is a service. That is a different deployment shape,
not a configuration change; `wsgi.py` says the same thing beside the code.

**Horizontal scaling is not a container setting.** Running two containers
against one mounted data root is not supported: two processes would open the
same Chroma store and keep separate ledgers, job registries and caches. The
resource limits in `config/` remain the real capacity dial whatever the
container is given.

**The job registry is in memory; only the *answer* survives a restart.** No
job is resumed. Nothing was committed either — the ledger write is a job's
last act — so the exposure is bounded, and every transition is journalled so
start-up can settle whatever was in flight against the ledger. A client
holding a `job_id` from before a restart gets `succeeded` if the ledger has
the document, `interrupted` if it does not. What you cannot do is pick a
half-finished ingest back up.

**Deadlines are cooperative.** Checked at stage boundaries and before every
outbound call, with each socket timeout clamped to the time left. Work already
inside a stage runs to the end of that stage: a parse is not interrupted, a
local model's forward pass is not interrupted, and an Ollama call cannot be
shortened at all — its client timeout is fixed at construction and `chat`
takes none per call, so the deadline can refuse to start one but not stop one.
A job or query can therefore overshoot by the longest such stage. It cannot
overshoot by a whole provider timeout on top; that is the line the clamping
holds.

**Budgets bound, they do not schedule.** `PROVIDER_MAX_INFLIGHT`,
`EMBEDDING_MAX_INFLIGHT` and `ANSWER_MAX_INFLIGHT` guarantee that the number
of calls in flight never exceeds the cap. They promise no ordering and no
fairness between waiters. A waiter cannot hang, because every wait is bounded
by the caller's own deadline.

---

## Resources and data

**Local models are memory, and the pipeline cache is the dial.** A pipeline
holds an embedding model, a store handle and a whole knowledge base's lexical
index. Embedding and cross-encoder models are shared process-wide by name
(`caches.local_models` shows what is resident), which took the model copies
out of `PIPELINE_CACHE_MAX` — but the lexical indexes and store handles still
scale with it. On a small host, lower `PIPELINE_CACHE_MAX` before anything
else.

**First parse of a PDF is slow and that is the layout model, not this code.**
Layout inference runs over every logical page: seconds per page on CPU, so an
85-page report is roughly ten minutes. The canonical units are then cached by
content hash under `.cache/canonical-units/`, and re-ingesting the same
document takes under a second. For a demo, upload beforehand.

**There is no upload size cap.** `MAX_FILE_SIZE_MB` used to be read and never
applied; Phase 8 removed the setting rather than leave a knob that turns
nothing. Adding a real cap is a feature — a check in the upload route and a
refusal shape — not a configuration change. Today the practical bound is the
ingest job deadline and the disk.

**Three on-disk caches are deliberately unbounded.**
`.cache/canonical-units/`, `.cache/embeddings/` and
`.cache/boundary-embeddings/` are content-addressed: they trade disk for a
re-parse or a re-embed and are safe to delete at any moment. Capping them
needs an eviction policy and size accounting the product has no evidence it
needs yet.

**`artifacts/viewer-live/` grows with the corpus, on purpose.** It is product
data, not a cache — it is what the Viewer reads. Each directory is deleted
with its document and is regenerable from an ingest. Bounding it would mean
deleting analyses a user still expects to open.

**A knowledge base is pinned to its embedding space.** Change the configured
embedding model and the store reports **re-index required**: dense retrieval
switches off, new uploads are refused with 409, and Settings → Embedding index
→ Re-index rebuilds it. Vectors from two models are never compared, so the
re-index is the whole corpus and not an increment.

---

## Code and boundaries

**Import-time side effects remain in three places.** They are documented
rather than fixed because removing them is a startup redesign:

| what | effect |
|---|---|
| `app.py` | importing it builds `Settings`, the pipeline, the KB and gold managers, the budgets and the ingest manager — so `import app` opens the vector store | 
| `utils/logger.py` | importing it creates the log directory and opens a log file |
| `config/__init__.py` | importing it applies `.env` to the process environment (deliberate, and the one dotenv read in the application) |

`tests/conftest.py` works around the first two by moving the whole session out
of the checkout before any application module is imported. A CLI run
(`python -m cli`) still writes a `logs/` directory wherever it is invoked.

**`amsc` is one flat namespace with research and legacy inside it.** Product,
service, research and legacy are *declared* in `amsc/surface.py` and enforced
against the real import graph rather than separated into packages. The reason
is in [chunk/docs/library-surface.md](../../chunk/docs/library-surface.md):
about thirty modules are documented `python -m amsc.<module>` entry points and
several write their own dotted name into artifacts that tests pin, so moving
them would need a re-export shim per module — the duplicate-surface problem,
not a fix for it.

**Legacy that stays, and why.** `amsc.legacy_chat_rag` is a pinned
reproduction of a public chunker kept as a benchmark candidate;
`amsc.viewer_v2` and its template are the earlier Viewer page, kept for the
research build's provenance arm and as a manual fallback. Both have real
callers. Neither is something to build on.

**Some public utility methods have no caller.** `BaseLLM.generate_json`,
`DocumentTracker.get_ingested_files` / `get_file_info` / `clear_all`, and
`RAGPipeline.add_assistant_response` / `get_conversation_summary` are retained
deliberately: they are the public surface of small classes with an obvious
shape, and the evidence for removing them was weaker than the cost of breaking
a caller outside this repository. They are not dead code by accident — Phase 8
looked at each and decided. `ParserFactory.register_parser` is in the same
position and *is* used by the extension recipe in
[architecture.md](architecture.md).

**Three duplicated small helpers remain.** `_ensure_parent` and the
tmp-write-then-`os.replace` dance appear in the two record managers, the
Viewer analysis writer and the ingest journal. They were left alone in Phase 8
because there is no correct owner: `config/paths.py` explicitly states it
creates no directories, and inventing a utility module for three lines is the
dumping ground the cleanup was avoiding. It needs an ownership decision first.

**The `amsc` pin is only as fresh as the last push.** Locally the library is
an editable install of `../chunk`, so an unpinned library change appears to
work. Only `tools/verify_reproducibility.py` and a clean install disagree.
See [testing.md](testing.md).

---

## Product scope

**No authentication, anywhere.** There is no login, no per-user data and no
access boundary on any endpoint. `/api/ops/metrics` has none for a specific
reason — everything it serves is aggregate and strictly less than `/api/kb`
already returns to the same caller — but the honest summary is that this
application assumes a trusted network. Putting it on an untrusted one is a
deployment decision that needs a reverse proxy in front.

**The demo launcher is Windows PowerShell.** `start-demo.ps1` and
`stop-demo.ps1` are the supported one-command demo path and have no bash
equivalent. The two servers they start are plain commands
([README](../README.md)), so the demo can be run by hand anywhere.

**Live workspace documents carry no gold set.** Documents ingested through the
console can be analysed and compared in the Viewer, but no Hit@k or MRR is
computed for them — there is nothing to score against. The Viewer says so
rather than inventing numbers, and they never enter the frozen benchmark
tables.

**The benchmark corpus is frozen and Turkish.** Every retrieval and chunking
number in `chunk/evaluation/` was measured on the KKB 2024 report with a fixed
gold query set. The deterministic BM25 uses a Turkish diacritic fold. Numbers
do not transfer to another corpus or another language; a new corpus needs its
own gold set.

**The documentation is in two languages.** The product docs here and the
library's boundary docs (`library-surface.md`, `viewer-architecture.md`) are in
English; the library's own README, the chunking recipe
(`adding-a-chunker.md`), the Viewer notes and the decision record are in
Turkish, as is the stakeholder explainer
[CHUNK_YONTEMLERI_VE_SORGU_EKRANI.md](../CHUNK_YONTEMLERI_VE_SORGU_EKRANI.md).
Each file is internally consistent and none is a translation of another, so
nothing is out of date because of it — but a reader of only one language will
find part of the map closed.

**Provider adapters are transports, not integrations.** The OpenAI-compatible
adapter sends the minimal payload (`model` + `messages`) and assumes nothing
else about the gateway. The Azure adapter is a supported deployment shape with
its own settings but is not what the demo runs, and the judge/provider adapter
in `amsc` is marked `adapter_only_not_verified` because it has not been
verified against a live service.
