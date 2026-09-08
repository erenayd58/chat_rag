# Operations — running it, its limits, and what to do when it says no

The operator's and the on-call developer's doc: the entrypoint, every bound
the process places on itself, what `GET /api/v1/health` and
`GET /api/ops/metrics` mean, and a symptom-first troubleshooting table.

Settings named here are documented once, in
[configuration.md](configuration.md), which is what decides their defaults.
The tables below give the shipped value so the prose is readable, not as a
second source of truth.

---

## Running it

There is one entrypoint, and it is what the container's `CMD` runs.

| | Command | Server | Binds |
|---|---|---|---|
| The backend | `python -m asgi` | uvicorn | `0.0.0.0` (`FLASK_HOST`) |
| The console | `npm run dev` / `npm start` in `frontend/` | Next.js | `localhost:3000` |

`python -m asgi` serves `/api/v1` and one operator route: **one process** with a
bounded pool of worker threads (`WAITRESS_THREADS`, default 8), plus the one
background thread that packages documents for the Viewer. Every handler on this
surface is a synchronous `def` -- it retrieves, it reads a store, it waits on a
provider -- so Starlette runs it in that pool, which is why the pool is the
request concurrency every ingest and query ration is sized against.

One process is a deliberate choice, not a limitation of the server: the
packaging queue lives in memory, the per-knowledge-base pipeline cache is a
module global, and the provider budgets are semaphores that only mean what they
say inside one address space, so a second worker process would duplicate all
three. `asgi.py` says so in more detail.

It stops on SIGTERM (what `docker stop` and service managers send) as well as on
Ctrl+C: uvicorn stops accepting, then the lifespan drains the ingest jobs
already running and returns the database pool.

There were two other entrypoints until Step 13 -- `python app.py` (Werkzeug,
loopback, the debugger) and `python -m wsgi` (waitress) -- and they served the
Flask console, its rendered screens and the Viewer's relay. Both went with that
surface ([legacy-removal.md](legacy-removal.md)); nothing reads `FLASK_DEBUG`
any more, and there is no debug mode to leave on by accident.

### The database

`python -m asgi` refuses to start without a reachable `DATABASE_URL`, and says
which host it could not reach.
That is deliberate: the knowledge bases, the ingest ledger, the content
identities and their analysis state, the ingest journal and the gold set are
rows, and every screen begins by listing knowledge bases. There is no degraded
mode that serves without one.

Create the schema before the first start, and after any upgrade that ships a
migration:

```bash
alembic upgrade head
```

Nothing in the application creates a table. [database.md](database.md) is the
schema, the migrations, the pool and the import path for an installation whose
records are still JSON files.

### Where the rest of the state goes

One setting decides: `CHAT_RAG_DATA_DIR`. Set it, and the parser's
canonical-unit cache, the packaged Viewer artifacts, the embedding caches, the
upload staging area and the logs all live under it. Leave it unset -- a local
checkout -- and every path stays exactly where it has always been, relative to
the working directory. The records and the vectors are not among them: they
are in the database `DATABASE_URL` names.

`STRUCTURED_PARSER_CACHE` still names the parser cache outright, for a
deployment that really does keep it elsewhere. But it is honoured only from the
actual environment: a value for it in `.env` is ignored once a data directory
has been declared, because `.env` describes a developer's own layout and a
deployment that has named its data directory has not asked for that layout. The
start-up banner says which paths are in effect and names anything it refused.

Proving that a *clean clone* can run at all is a separate question with its
own gate: [testing.md](testing.md).

---

## Bounded ingest (uploads as jobs)

An upload is an **ingest job**. `POST /api/v1/documents` validates the
request, stages the file and queues the job; the parse, the chunking, any
Deep Analysis model calls, the embeddings, the store write and the ledger
write happen on an ingest worker. Two answers are possible:

* `async=1` (what the console sends): **202** at once with `job_id` and the
  job; poll `GET /api/v1/ingest-jobs/<job_id>` until `status` is terminal.
  There is no second, synchronous answer: this contract always answers 202.
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

`GET /api/v1/health` and `GET /api/v1/ingest-jobs` show the capacity picture,
including both budgets and how many finished jobs are retained;
`DELETE /api/v1/ingest-jobs/<job_id>` cancels a queued job at once and a running
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

**Admission is immediate and never queues.** `POST /api/v1/queries` either enters
now or is refused now with **503**, `overloaded: true`, `reason: admission`
and a `Retry-After` (derived from the median recent query time, within 2–30
seconds). A queued question would hold the very thread the limit exists to
keep free. The default is derived so that questions and synchronous uploads
together can never take every request thread, which is what keeps
`/api/v1/health`, `/api/ops/metrics` and job polling answerable under any burst
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
sentence-transformers embedder, so `PIPELINE_CACHE_MAX` was also a multiplier
on model memory. They are now shared process-wide by model name;
`caches.local_models` on the metrics endpoint shows what is resident and how
often it was loaded.

**Search runs under the same limits.** `POST /api/v1/searches` does the
front half of a query on the request thread — embed the question, search the
store, build the lexical index if this pipeline has not built it yet — so
under no limit at all it was a way around `QUERY_MAX_ACTIVE`: a burst of
searches could hold every request thread, each waiting an unbounded time for an
embedding slot. (There were three such endpoints, one per retriever leg, until
they became one resource with a `method`.) It takes the same admission counter
(the bound is on request threads doing retrieval, whichever route asked), the
same deadline, the same pipeline lease and the same telemetry, under
`mode: lab.*` so an operator can tell a search from a question. It makes no
answer-model call, so it is given no answer budget. Its answers are
unchanged apart from the two refusals every query path shares: **503** when
admission is full, **504** past the deadline.

**A limit is never a fallback.** The answer path degrades on purpose — a
gateway that is down becomes an answer saying so rather than a broken
request — and that stays. It must not extend to a deadline, a refused budget
or a cancellation: swallowed into a fallback, those become a query that runs
on past its deadline making calls that are refused in turn and then answers
as though nothing happened. `RESOURCE_CONTROL_EXCEPTIONS`
(`core/exceptions.py`) names the four, and every handler that degrades
re-raises them first.

**Every query is measured with the ingest instrument.** A query is a trace
of kind `query` with stages `retrieve`, `context` and `answer`, plus
provider seconds and slot wait recorded by the budget
wrapper. `GET /api/ops/metrics` carries `metrics.queries` — active and peak
active, p50/p95/max per stage, provider wait, outcomes (`succeeded`,
`failed`, `timed_out`, `rejected`) and the last few traces — and `query`
(admission and answer-budget counters, the limits, the deadline semantics
in one sentence). `/api/v1/health` carries one line: `query.active`,
`max_active`, `answer_inflight`, `answer_limit`, and reports `overloaded`
while every query slot is in use. The response's `metadata.query` carries
the same timing for that one question, so a slow answer can be correlated
with the metrics by `query_id`. The log carries the question's length, the
stage times and the outcome; never the question, a chunk or the answer.
Every content-bearing dump — the prompt, the retrieved chunks, the assembled
context — is at `DEBUG`, so `LOG_FILE_LEVEL=DEBUG` gets all of it and the
default gets none of it.

## Health, metrics and the caches

`GET /api/v1/health` is the small one, for a probe: liveness, readiness and a
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
`GET /api/v1/ingest-jobs/<job_id>`.

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
less than `/api/v1/knowledge-bases` and a knowledge base's chunks already
return to the same caller.

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

---

## Troubleshooting

Symptom first. Every entry names where to look before deciding anything.

### An upload is refused with 503

```json
{"error": "...", "overloaded": true, "retry_after": 12}
```

**Means:** the ingest queue is full — `INGEST_WORKERS` jobs are running and
`INGEST_QUEUE_CAPACITY` are waiting. Nothing was queued and nothing was kept;
the staged file is gone.

**Look at:** `GET /api/v1/health` → `ingest.running`, `ingest.queued`,
`ingest.queue_capacity`, and `state` (it reads `overloaded`).
`GET /api/ops/metrics` → `metrics.counters.rejected`, and `metrics.stages` for
which stage is spending the time.

**Next:** wait — do not restart; the queue drains. If it is chronic, read
`metrics.stages` first. A `parse` p95 in the minutes is first-time PDFs on CPU
(expected, and cached afterwards); a slow `deep_analysis` is the provider.
Raise `INGEST_QUEUE_CAPACITY` to absorb bursts; raise `INGEST_WORKERS` only if
the host has the CPU, because parsing and local embedding run there.

### A question is refused with 503

Two different refusals, deliberately distinguishable by `reason`:

| `reason` | means | next |
|---|---|---|
| `admission` | every request thread allowed inside a query is in use (`QUERY_MAX_ACTIVE`) | raise `WAITRESS_THREADS` and let `QUERY_MAX_ACTIVE` re-derive, or raise it explicitly — but not past `WAITRESS_THREADS - INGEST_SYNC_WAITERS - 1`, or health and job polling lose their reserved thread. The process warns at start-up if you do. |
| `answer_capacity` | the query got in, but no answer-model slot came free within `ANSWER_SLOT_WAIT` (`ANSWER_MAX_INFLIGHT`) | raise `ANSWER_MAX_INFLIGHT` if the gateway tolerates it. This is a provider bound, not a thread bound. |

**Look at:** `GET /api/v1/health` → `query.active` / `max_active` and
`query.answer_inflight` / `answer_limit`. `GET /api/ops/metrics` →
`query.admission`, `query.answer_budget` (including how long callers waited)
and `metrics.queries.outcomes`.

### A question returns 504

**Means:** the question passed `QUERY_TIMEOUT`. The deadline is cooperative:
checked before every outbound call and while waiting for a slot, with each
answer attempt's socket timeout clamped to the time left. A stage already
running is not interrupted, so an overshoot by one stage is expected; an
overshoot by a whole provider timeout is not.

**Look at:** `metrics.queries.stages` (p50/p95/max for `retrieve`,
`context`, `answer`) and the response's own `metadata.query`, which carries
the same timing for that one question under a `query_id` that is also in the
log.

**Next:** a slow `retrieve` on a cold pipeline is usually the lexical index
being built for the first time; a slow `answer` is the gateway. Raise
`QUERY_TIMEOUT` only after reading which stage spent the time.

### A job is stuck, or ended `timed_out`

**Look at:** `GET /api/v1/ingest-jobs/<job_id>` — state, stage timings and error
category. `GET /api/v1/ingest-jobs` lists what is queued and running.

| state | means |
|---|---|
| `queued` | waiting for a worker. One ingest runs per knowledge base at a time, so a second upload to the *same* KB waits even with workers free. |
| `running` past its stage p95 | usually `parse` on a first-seen PDF (minutes per document on CPU), or `deep_analysis` waiting on provider slots. |
| `timed_out` | it passed `INGEST_JOB_TIMEOUT`. Nothing was committed — the ledger write is a job's last act. Upload it again. |
| `interrupted` | the process stopped mid-job and start-up settled it against the ledger: nothing was registered, upload it again. |
| `failed` | a real error. `error_category` says which kind; `metrics.errors` keeps a few redacted example messages per category. |

**Next:** `DELETE /api/v1/ingest-jobs/<job_id>` cancels — a queued job at once, a
running one at its next stage boundary. A **404** means only that the job is
older than `INGEST_JOB_RETENTION`.

### The Deep or embedding budget is saturated

**Means:** calls are waiting for a slot, not failing. `PROVIDER_MAX_INFLIGHT`
(Deep proposer + verifier) and `EMBEDDING_MAX_INFLIGHT` are process-wide and
separate on purpose, so neither path can starve the other.

**Look at:** `/api/ops/metrics` → `ingest.budgets`, which carries the in-flight
count, the limit and the seconds callers have spent waiting for a slot.
`/api/v1/health` → `ingest.provider_inflight` / `provider_limit` and
`ingest.embedding_inflight` / `embedding_limit`.

**Next:** a long wait with jobs still completing is the budget doing its job.
Raise the cap only if the gateway's own rate limit is higher than yours — and
remember `DEEP_ANALYSIS_CONCURRENCY` is one job's pool while
`PROVIDER_MAX_INFLIGHT` is every job together.

### `state: degraded`

**Means:** something needs a person; `reasons` names it. Two causes: the
knowledge base records cannot be read, or the last `DEGRADED_AFTER_JOBS` (5)
ingest jobs all failed — the "healthy but broken" case where the queue is
empty because every job dies.

It is a ratio over the bounded metrics window rather than a latch, so a
service that recovers stops reporting it. `degraded` outranks `overloaded`:
being full is transient, failing is not. `ready` stays true either way,
because a degraded process still serves reads.

**Look at:** `reasons`, then `metrics.errors` for the category, then the
`RAG.ops` log lines (`ingest.job.failed`, one line per job).

### `state: overloaded`

The ingest queue is full and uploads are being refused; chat and search still
work. Wait, do not restart. See the first entry above.

### A Viewer analysis never becomes `ready`

Each document carries its own state on its `analysis` block in
`GET /api/v1/documents`: `missing` → `pending` → `running` → `ready` |
`failed`.

**Look at:** `GET /api/v1/documents/<document_id>/analysis` for that document's
state and error, `metrics.stages.viewer_stage` for how long packaging takes,
and `artifacts/viewer-live/<doc_id>/` on disk for what was written.

**Next:** `POST /api/v1/documents/<document_id>/analysis` re-queues one
document. Packaging runs on a single background thread and makes no provider
call, so a stuck package is never a budget problem. What each file in a packaged tree is, and which step
writes it, is in
[chunk/docs/viewer-architecture.md](../../chunk/docs/viewer-architecture.md).

If the Viewer *screen* shows nothing rather than a document's analysis being
stuck: it is a screen of the Next.js console (`/viewer`), so check that the
console is running and that `GET /api/v1/documents/<id>/analysis` says
`ready`. There is no second server to start; the standalone page in the
`chunk` checkout is not part of this product.

### The pipeline cache looks wrong

A pipeline holds an embedding model, a store handle and a whole knowledge
base's lexical index, so it is the largest memory dial in the process.

**Look at:** `/api/ops/metrics` → `caches.pipelines` (size against
`PIPELINE_CACHE_MAX`) and `caches.local_models` (which models are resident and
how often each was loaded — they are shared process-wide by name, so that
count should stay small).

**Expected behaviour:** a pipeline is evicted only when nothing is using it;
an ingest job leases its pipeline for the length of the job, so browser
traffic cannot close a store a job is writing to. After an ingest, every
*other* pipeline for that knowledge base drops its lexical index and rebuilds
it on the next query — which is why the first question after an upload can be
slower.

### The process will not start without a database

`Cannot start: DATABASE_URL is not set` -- set it (`env.example` shows the
form) and run `alembic upgrade head`.

`Cannot start: cannot reach the database at postgresql+psycopg://host:5432/name`
-- the process could not open a connection within `DATABASE_CONNECT_TIMEOUT`.
The credential is not in that message; the host and the database name are.
Check that the server is up, that the network reaches it, and that the
database named in the URL exists -- `alembic upgrade head` creates the *schema*
but not the database.

`relation "knowledge_bases" does not exist` at the first request means the
database is reachable and the migrations have not been run.

### The process will not start

**Means:** a configuration value was refused, and the message names the
variable. Configuration is validated at start-up on purpose, so a bad value
stops the process while someone is looking rather than failing the first
request.

**Look at:** the variable name in the error, then
[configuration.md](configuration.md) for who owns it. Logging is the one group
that does *not* refuse to start: an unrecognised level falls back to `INFO`
and says so in a warning line.

Cross-setting rules are checked together (`cross_check` in
`config/runtime.py`): a synchronous upload that would outlive its connection
(`INGEST_SYNC_WAIT` past `WAITRESS_CHANNEL_TIMEOUT`) is refused outright; a
thread ration that leaves nothing free is reported as a warning and served at
`/api/ops/metrics` → `configuration.warnings`.

### The reproducibility gate fails

`python tools/verify_reproducibility.py` — see
[testing.md](testing.md). It installs from the **declared source**, not from
your working tree, so it fails on things a green suite cannot see.

| check | almost always means |
|---|---|
| `pin.shape` | the `amsc-poc` line in `requirements.txt` is malformed, or does not end at a 40-character sha |
| `chunk.install` | the pinned commit is on no remote branch — push the `chunk` branch |
| `chunk.import` | the pinned revision lacks a symbol the product imports — bump the pin with `python tools/promote_chunk_pin.py` |
| `clone.clean` | developer state got committed; it belongs in `.gitignore` |
| `docker.*` | Docker is not running, or the image build genuinely broke |
| `state.untouched` | the gate wrote to your real data — a bug, not a flake |

---

## Where the log lines are

| what | where |
|---|---|
| lifecycle events, one line each, stable `event=… job_id=… kb_id=…` shape | the `RAG.ops` logger: `ingest.job.accepted`, `.started`, `.succeeded`, `.failed`, `.timed_out`, `.cancelled`, `.rejected`, `.attached`, `.restart_settled` |
| everything else an operator needs | `INFO`, in `logs/rag_<timestamp>.log` under the data root, and on the console |
| prompts, retrieved chunks, the answer context | `DEBUG` only, and only in the file handler — `LOG_FILE_LEVEL=DEBUG` opts in, and the process warns at start-up that it did |

Values are redacted before they are written: document text, chunks, prompts
and anything credential-shaped cannot reach a log line through the ops path,
which is what makes it safe to paste into a ticket.
