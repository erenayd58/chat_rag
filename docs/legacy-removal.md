# The legacy surface, and the order it came out in

The Flask-era API was served beside [`/api/v1`](api-v1.md) for five steps.
Both called the same use cases over one container, so they could not disagree
about behaviour — only about spelling. This page was the plan for deleting the
older spelling: every endpoint it served, who called it, what answers it on the
contract, and which step removed it.

**All three waves are done.** Step 13 took the last of it, and what this page
is now is the record: what was removed, what replaced it, and — at the end —
what was deliberately *not* removed and why. It is kept rather than deleted
because every row is a question somebody will ask again ("where did
`/api/demo/workspace` go?"), and because a plan that survives its own execution
is the only kind worth writing.

It exists in this shape because "the legacy surface" was never one decision. A
route the console's JavaScript called came out when that screen was rewritten
(Step 11); a route the Viewer relayed came out when the Viewer moved (Step 12),
and that was a change in the *other* repository; and a handful came out only
when the bridge itself went (Step 13). Removing them in one commit would have
meant breaking all three at once.

`tests/migration/test_legacy_removal_map.py` holds this page against the
application's real routing table and against the tree: nothing on it is served
any more, the files wave 3 deleted are gone, the framework is not imported, its
dependencies are not declared, and no source file still reaches for a removed
path. So this page cannot quietly become a story about work that was undone —
which is the only reason to trust a record written by the person who did it.

---

## How to read it

**Replacement** is what a client calls instead. `—` means nothing on
`/api/v1` answers this and the reason is in [api-v1.md](api-v1.md) under *Not
here, on purpose*; removing such a route was a decision to stop offering it,
not a migration.

**Caller** is what called it at the time the row was written, found by reading
the console's JavaScript, the Viewer's relay (`amsc.viewer.server` in the
`chunk` repository), the container's health check and this repository's own
tooling. *none* means nothing in either repository called it.

**Wave** was when it went:

| wave | what it took | what had to be true first | step |
|---|---|---|---|
| **1** | routes whose only caller was a console screen | that screen is served by the new front end and calls `/api/v1` | 11 → 13 |
| **2** | routes the Viewer relayed | the Viewer speaks `/api/v1`, or the route is promoted; a `chunk` change and a pin bump | 12 → 13 |
| **3** | the bridge, the blueprints, the templates, the static JavaScript and the second entrypoint | waves 1 and 2 are done and the health check has moved | 13 |

Waves 1 and 2 each had two halves: the *caller* went in Step 11 or 12, and the
*route* went in Step 13 with the surface that served it. A route with no caller
is not a route that can be deleted while the screen serving the old console is
still reachable.

---

## Wave 1 — the console's own screens

Every one of these had a `/api/v1` answer already.

**Step 11 met wave 1's precondition.** The console is a Next.js application in
[`../frontend/`](../frontend/README.md), and it speaks `/api/v1` and nothing
else — `frontend/tests/surface.test.ts` reads its source and fails on any
Flask-era path, and `frontend/tests/live/console.test.tsx` drives the real
screens against a running server. The caller column below is therefore a record
of the **Flask-era screens**, which were served beside the new front end until
Step 13 and came out with it. Three of them had no `/api/v1` answer to migrate
to — `GET /api/stats`, the chunk editor and the gold set — and the new console
does not offer them; see *The Lab's own affordances* below.

### Knowledge bases — `/` and `/kb/<kb_id>`

| legacy | replacement | caller |
|---|---|---|
| `GET /api/kb` | `GET /api/v1/knowledge-bases` | `kb_list.js`, `api.js` |
| `POST /api/kb` | `POST /api/v1/knowledge-bases` | `kb_list.js` |
| `GET /api/kb/<kb_id>` | `GET /api/v1/knowledge-bases/<kb_id>` | `kb_detail.js` |
| `PUT /api/kb/<kb_id>` | `PATCH /api/v1/knowledge-bases/<kb_id>` | `kb_detail.js` |
| `DELETE /api/kb/<kb_id>` | `DELETE /api/v1/knowledge-bases/<kb_id>` | `kb_detail.js`, `kb_list.js` |
| `GET /api/kb/<kb_id>/embedding-index` | `GET /api/v1/knowledge-bases/<kb_id>/embedding-index` | `kb_detail.js` |
| `POST /api/kb/<kb_id>/reindex-embeddings` | `POST /api/v1/knowledge-bases/<kb_id>/embedding-index/rebuild` | `kb_detail.js` |

`PUT` became `PATCH` on purpose: the legacy route read a whole record and wrote
the two fields it was allowed to, which is a `PATCH` that was spelled `PUT`.
`DELETE` became **204** with no body, so "how many vectors went" is no longer
reported to a client; it went in the same transaction as the record, which is
what `tests/integration/test_kb_api.py` reads off the store.

### Documents and their ingestion

| legacy | replacement | caller |
|---|---|---|
| `GET /api/documents` | `GET /api/v1/documents` | `chat.js`, `kb_detail.js`, `kb_list.js`, `lab.js` |
| `POST /api/documents/upload` | `POST /api/v1/documents` | `kb_detail.js` |
| `DELETE /api/documents/<doc_id>` | `DELETE /api/v1/documents/<document_id>` | `kb_detail.js` |
| `GET /api/documents/<doc_id>/chunks` | `GET /api/v1/documents/<document_id>/chunks` | `lab.js` |
| `GET /api/documents/<doc_id>/canonical-units` | `GET /api/v1/documents/<document_id>/units` | `lab.js` |
| `GET /api/ingest/jobs` | `GET /api/v1/ingest-jobs` | `kb_detail.js` |
| `GET /api/ingest/jobs/<job_id>` | `GET /api/v1/ingest-jobs/<job_id>` | `kb_detail.js` |
| `DELETE /api/ingest/jobs/<job_id>` | `DELETE /api/v1/ingest-jobs/<job_id>` | none |
| `GET /api/stats` | — | `kb_detail.js` |

Two things a client had to know here, and one of them stopped being a choice:

- **the upload is always asynchronous.** The legacy route could block until the
  job settled; `/api/v1` answers **202** with the job and nothing else. That
  synchronous mode was the reason the ingest path needed a semaphore to stop
  uploads holding every request thread, and it was not inherited. What became
  of the ration is in *What was kept, and why* below.
- **`GET /api/stats` had no replacement and needed none.** It was a count of
  documents, chunks and bytes for one knowledge base. `GET /api/v1/documents`
  carries every number in it per document, and `GET /api/v1/health` carries the
  capacity half. Its use case (`application.documents.statistics`) went with
  it. If a screen wants the total without walking the collection, that is a new
  endpoint to design, not a legacy one to keep.
- **`deep_analysis=true` was a second spelling** of the method list, kept for
  the old upload form. Deep Analysis is a registry key like every other method
  (`methods=agentic`), and the boolean went with the form.

### Asking, and the Lab

| legacy | replacement | caller |
|---|---|---|
| `POST /api/query` | `POST /api/v1/queries` | `chat.js` |
| `POST /api/chunks/search-vector` | `POST /api/v1/searches` (`method: "vector"`) | `lab.js` |
| `POST /api/chunks/search-bm25` | `POST /api/v1/searches` (`method: "bm25"`) | `lab.js` |
| `POST /api/experiment/search_chunks` | `POST /api/v1/searches` | `lab.js` |
| `GET /api/chunks` | `GET /api/v1/knowledge-bases/<kb_id>/chunks` | `lab.js` |
| `GET /api/models` | `GET /api/v1/meta/models` | `lab.js` |
| `GET /api/retrieval/capabilities` | `GET /api/v1/meta/retrieval-methods` | `lab.js` |
| `GET /api/demo/methods` | `GET /api/v1/meta/chunking-methods` | `kb_detail.js` |

Four search routes became one resource with a `method`. The three legacy ones
differed only in which retriever leg they called, which is a parameter and not
three endpoints; `application.chunks.search_vector` and `search_bm25` went with
them, leaving one `experiment_search`. The bound they all run under did not
move — `tests/integration/test_search_query_limits.py` drives every method
through it.

### The Lab's own affordances, which were not promoted

| legacy | replacement | caller |
|---|---|---|
| `GET /api/chunks/<chunk_id>` | — | none |
| `PUT /api/chunks/<chunk_id>` | — | `lab.js` |
| `DELETE /api/chunks/<chunk_id>` | — | `lab.js` |
| `GET /api/goldset` | — | `lab.js` |
| `POST /api/goldset` | — | `lab.js` |
| `DELETE /api/goldset/<entry_id>` | — | `lab.js` |

Editing an indexed chunk changes the corpus behind the ingest ledger's back,
and the gold set is an offline evaluation input driven by `python -m cli`.
Both were deliberate omissions from the contract, and **the decision they
forced was about the Lab screen, not about the API**: either the Lab is not
rebuilt, or these are promoted first with the ledger question answered.

**Step 11 decided: the Lab is not rebuilt.** Its two useful halves were, and
they are ordinary product screens on the contract — *Search* is
`POST /api/v1/searches` with the retrieval method from
`GET /api/v1/meta/retrieval-methods`, and *Analysis* is a document's chunking
variants over `GET|POST /api/v1/documents/<document_id>/analysis`. What was
left behind is exactly the three affordances with no `/api/v1` answer, and
nothing was invented to replace them.

**What that costs, stated plainly.** With those routes gone, nothing writes a
runtime gold entry any more: `python -m cli gold export` still freezes what is
in the store into a file, and `python -m cli eval` still runs a frozen set
through retrieval, but marking an answer in a screen is not a thing this
product does. That is the consequence of not rebuilding the Lab, and it is
recorded here rather than discovered later. `application.goldsets` — the use
case those three routes called — went with them; `components/goldset` and its
CLI callers did not.

---

## Wave 2 — the Viewer's relay

Called by `amsc.viewer.server` in the `chunk` repository, over HTTP, from a
separate process. **Step 12 removed the caller**: the Viewer is a screen of
this front end, it reads `/api/v1` directly, and no `:8765` process is part of
the product. **Step 13 removed the routes**, and the server that used to call
them (`amsc.viewer.server`, with its launcher `chunk/tools/serve_viewer_v3.ps1`).

| legacy | replacement | caller |
|---|---|---|
| `GET /api/demo/workspace` | `GET /api/v1/knowledge-bases` | none |
| `POST /api/demo/viewer-analysis/<doc_id>` | `POST /api/v1/documents/<document_id>/analysis` | none |
| `GET /api/demo/viewer-analysis/<doc_id>/payload` | `GET /api/v1/documents/<document_id>/analysis/payload` | none |
| `GET /api/demo/viewer-analysis/<doc_id>/chunks` | `GET /api/v1/documents/<document_id>/analysis/methods/<method>/chunks` | none |
| `GET /api/demo/viewer` | — | `api.js` |
| `GET /api/demo/viewer-analysis/<doc_id>` | `GET /api/v1/documents/<document_id>/analysis` | none |
| `POST /api/demo/viewer-analysis/<doc_id>/methods` | `POST /api/v1/documents/<document_id>/analysis/methods` | none |

The two that had no replacement got one, and both were kept as small as the
parity actually needed:

- **`/payload`** became `GET /api/v1/documents/<document_id>/analysis/payload`.
  What it carries is still the render model the packager writes, and it is
  published as **pass-through** for exactly the reason this row used to give
  for not promoting it — the shape belongs to the analysis, it grows a field
  whenever a chunker records something new, and no version number should be
  spent on it. What is contractual is the resource around it: which document,
  which content, which methods are in it, and a **409 `not_ready`** while
  nothing is built.
- **`/api/demo/workspace`** did not need one. It answered "every knowledge
  base, every document, and where each document's analysis got to" in one
  request, which is `GET /api/v1/knowledge-bases` and `GET /api/v1/documents`
  — the latter already carrying each document's `analysis` block. The
  `?prepare=1` half, a bulk queue of every missing analysis, was not promoted:
  the screen queues the document a reader actually opened. Both halves of the
  use case (`application.workspace.snapshot` and `prepare_missing`) went with
  the routes, and so did `probe_viewer` and the `VIEWER_URL` setting it read —
  there is no companion process to probe.

One route the Viewer relayed had no `/api/v1` answer at all, and it was not the
payload. The Viewer's *Sorgu* asks one question of one document through several
chunking methods at once, over indexes built from the analysis arms — a
comparison of chunkers, which no knowledge-base query can make because a
knowledge base has one chunker. That is now
`POST /api/v1/analysis-queries`, and the engine behind it is the library's own
(`amsc.viewer.chat`), which moved from `SERVICE` to `CONSOLE_API` in the same
step. The relay never appeared in this table because it was not a console
route: the old Viewer server answered it itself, out of its own process.

`GET /api/demo/viewer` was a probe of whether a companion Viewer process was
running. There is no companion process; it went with the console screen that
showed the link.

---

## Wave 3 — what was left when both were done

| what | why it survived waves 1 and 2 | what happened to it |
|---|---|---|
| `GET /api/health` | the container's `HEALTHCHECK`, `docker-compose.yml`, `start-demo.ps1`, `tools/serve_smoke.py` and `tools/verify_reproducibility.py` called it | those five moved to `GET /api/v1/health` first, then the route went. The two `/api/v1` fields a probe reads are `state` and `ready`; `status: "healthy"`, the third word the old body carried for probes that predated them, is not on the contract |
| `/api/ops/metrics` (GET) | an operator surface whose contents are deliberately free to change, so it was never promoted | **kept**, at the same path, with the same body. See below |
| `GET /`, `GET /kb/<kb_id>`, `GET /chat`, `GET /lab` | the rendered screens | gone, with `templates/` and `static/` |

With those gone, the deletion was `interfaces/http/legacy/`,
`interfaces/http/coexistence.py`, `interfaces/http/context.py`, `templates/`,
`static/`, `app.py`, `wsgi.py`, the console-API table in
[../README.md](../README.md), and the Flask, flask-cors and waitress
dependencies. [`asgi.py`](../asgi.py) was already the same application standing
alone; it is the entrypoint now, and `python -m asgi` is what the image runs.

### `/api/ops/metrics` was kept, deliberately

It is the one route served outside `/api/v1`. Its *contents* are free to change
with the internals they report on — counters, stage latencies, cache occupancy,
the effective configuration — which is why it was never promoted to a versioned
path and why promoting it now would be a promise nobody wants to keep. It is
not unused: it is what an operator reads once `GET /api/v1/health` has told
them to look closer, and `docs/operations.md` publishes it.

So it moved framework and nothing else. `interfaces/http/operator.py` serves
the same path, answers the same body (including the `success` field, for the
scripts that read it), and reads `?recent=` the same forgiving way the Flask
route did — a value that is not a number falls back to the default rather than
refusing the request, which is the opposite of what the contract does and is
right for the endpoint somebody curls when something is wrong.
`tests/migration/test_http_surface.py` declares it as the one route outside the
contract and fails if a second appears.

---

## What was kept, and why

Not everything the legacy surface reached for went with it. Two things stayed,
and both are recorded here because "it has no caller" was not sufficient reason
to remove them.

**The synchronous-upload ration.** `INGEST_SYNC_WAIT` and
`INGEST_SYNC_WAITERS`, `services.sync_waiters`, and
`application.ingest.await_settlement` / `outcome` are the application-layer
path a synchronous upload took, and no route calls them: `POST /api/v1/documents`
has always answered **202**. They are kept because `INGEST_SYNC_WAITERS` is
*subtracted from the `QUERY_MAX_ACTIVE` default* — the reservation is what
guarantees questions can never take every request thread — so removing it would
raise the number of questions a deployment answers at once. That is a change to
what a deployment does, not a cleanup, and it belongs to whoever decides it
deliberately. `tests/unit/test_configuration.py` states the derivation and this
reason beside it.

**`FLASK_HOST`, `FLASK_PORT`, `WAITRESS_THREADS`, `WAITRESS_CHANNEL_TIMEOUT`.**
Flask and waitress are gone; the names are not. They are what every `.env`, the
compose file and the container image already carry, and renaming a setting is a
*silent* change — the old name simply stops being read and the default takes
over. `config/runtime.py` owns all four and says so at the top.

---

## What must not move back

- **Nothing was removed from `application/` that a `/api/v1` route calls.**
  Where removing a route left a use case with no caller at all, the use case
  went with it and is named in the rows above; where it left one the contract
  still calls, nothing changed.
- **A route removed here is removed from the documents in the same commit.**
  `tests/migration/test_http_surface.py` compares `docs/api-v1.md` to the
  routing table in both directions, and `test_legacy_removal_map.py` compares
  this page to it.
- **The Viewer's routes could not be removed from this repository alone.** The
  Viewer was pinned; a running Viewer built against the old relay would have
  failed on a live document with no message a user could act on. That is why
  wave 2 was two steps and two repositories.
