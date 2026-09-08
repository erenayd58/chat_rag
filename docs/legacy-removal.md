# The legacy surface, and the order it comes out in

The Flask-era API is still served beside [`/api/v1`](api-v1.md). Both call the
same use cases over one container, so they cannot disagree about behaviour —
only about spelling. This page is the plan for deleting the older spelling:
every endpoint it still serves, who calls it, what answers it on the contract,
and which step removes it.

It exists because "the legacy surface" is not one decision. A route the
console's JavaScript calls comes out when that screen is rewritten (Step 11); a
route the Viewer relays comes out when the Viewer is moved (Step 12), and that
is a change in the *other* repository; and a handful come out only when the
bridge itself goes (Step 13). Removing them in one commit would mean breaking
all three at once.

`tests/migration/test_legacy_removal_map.py` holds the tables below against the
application's real routing table, in both directions. A route added or removed
without a row here is a failure, so this page cannot quietly go stale — which
is the only reason to trust a plan written before the work.

---

## How to read it

**Replacement** is what a client should call instead. `—` means nothing on
`/api/v1` answers this and the reason is in [api-v1.md](api-v1.md) under *Not
here, on purpose*; removing such a route is a decision to stop offering it, not
a migration.

**Caller** is what actually calls it today, found by reading the console's
JavaScript, the Viewer's relay (`amsc.viewer.server` in the `chunk`
repository), the container's health check and this repository's own tooling.
*none* means nothing in either repository calls it.

**Wave** is when it goes:

| wave | what it takes | what has to be true first |
|---|---|---|
| **1** | routes whose only caller is a console screen | that screen is served by the new front end and calls `/api/v1` |
| **2** | routes the Viewer relays | the Viewer speaks `/api/v1`, or the route is promoted; a `chunk` change and a pin bump |
| **3** | the bridge, the blueprints, the templates and the static JavaScript | waves 1 and 2 are done and the health check has moved |

---

## Wave 1 — the console's own screens

Every one of these has a `/api/v1` answer already. Nothing outside
`static/js/` calls them.

**Step 11 met wave 1's precondition.** The console is a Next.js application in
[`../frontend/`](../frontend/README.md) now, and it speaks `/api/v1` and
nothing else — `frontend/tests/surface.test.ts` reads its source and fails on
any Flask-era path, and `frontend/tests/live/console.test.tsx` drives the real
screens against a running server. The caller column below is therefore a
record of the **Flask-era screens**, which are still served beside the new
front end and still call these routes; they and their JavaScript come out
together in wave 3, and until then removing a route here would break a screen
that is still reachable. Three of them had no `/api/v1` answer to migrate to —
`GET /api/stats`, the chunk editor and the gold set — and the new console does
not offer them; see *The Lab's own affordances* below.

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

`PUT` becomes `PATCH` on purpose: the legacy route reads a whole record and
writes the two fields it is allowed to, which is a `PATCH` that was spelled
`PUT`.

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

Two things a front end has to know here:

- **the upload is always asynchronous on the contract.** The legacy route can
  block until the job settles; `/api/v1` answers **202** with the job and
  nothing else. The synchronous mode is the reason that adapter needs a
  semaphore to stop uploads holding every request thread, and it is not
  inherited.
- **`GET /api/stats` has no replacement and needs none.** It is a count of
  documents, chunks and bytes for one knowledge base. `GET /api/v1/documents`
  carries every number in it per document, and `GET /api/v1/health` carries the
  capacity half. If a screen wants the total without walking the collection,
  that is a new endpoint to design, not a legacy one to keep.

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

Four search routes become one resource with a `method`. The three legacy ones
differ only in which retriever leg they call, which is a parameter and not
three endpoints.

### The Lab's own affordances, which are not promoted

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
Both are deliberate omissions from the contract, and **the decision they force
is about the Lab screen, not about the API**: either the Lab is not rebuilt,
or these are promoted first with the ledger question answered.

**Step 11 decided: the Lab is not rebuilt.** Its two useful halves were, and
they are ordinary product screens on the contract — *Search* is
`POST /api/v1/searches` with the retrieval method from
`GET /api/v1/meta/retrieval-methods`, and *Analysis* is a document's chunking
variants over `GET|POST /api/v1/documents/<document_id>/analysis`. What was
left behind is exactly the three affordances with no `/api/v1` answer, and
nothing was invented to replace them: editing an indexed chunk stays a lab
affordance rather than becoming a product operation, and the gold set stays
what it is, an offline input to `python -m cli`. The Flask Lab still serves
all three until wave 3 takes the screen.

---

## Wave 2 — the Viewer's relay

Called by `amsc.viewer.server` in the `chunk` repository, over HTTP, from a
separate process. **Step 12 removed the caller**: the Viewer is a screen of
this front end now, it reads `/api/v1` directly, and no `:8765` process is
part of the product any more. Every row below therefore has a replacement and
a caller of `none`; the routes themselves come out in Step 13 with the rest of
the legacy layer.

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
  the screen queues the document a reader actually opened.

One route the Viewer relayed had no `/api/v1` answer at all, and it was not the
payload. The Viewer's *Sorgu* asks one question of one document through several
chunking methods at once, over indexes built from the analysis arms — a
comparison of chunkers, which no knowledge-base query can make because a
knowledge base has one chunker. That is now
`POST /api/v1/analysis-queries`, and the engine behind it is the library's own
(`amsc.viewer.chat`), which moved from `SERVICE` to `CONSOLE_API` in the same
step. The relay never appeared in this table because it was not a console
route: the old Viewer server answered it itself, out of its own process.

`GET /api/demo/viewer` is a probe of whether a companion Viewer process is
running. There is no companion process any more; it goes with the console
screen that shows the link.

---

## Wave 3 — what is left when both are done

| what | why it survives waves 1 and 2 |
|---|---|
| `GET /api/health` | the container's `HEALTHCHECK`, `docker-compose.yml`, `start-demo.ps1`, `tools/serve_smoke.py` and `tools/verify_reproducibility.py` call it. `GET /api/v1/health` answers the same question; the removal is those five edits, and it must not be done before them |
| `GET /api/ops/metrics` | an operator surface whose contents are deliberately free to change, so it was never promoted. It has no in-repository caller and it is not a client contract; it moves or goes when someone decides where an operator looks |
| `GET /`, `GET /kb/<kb_id>`, `GET /chat`, `GET /lab` | the rendered screens. They go when the front end serves them |

With those gone, the deletion is `interfaces/http/legacy/`,
`interfaces/http/coexistence.py`, `templates/`, `static/`, the console-API
table in [../README.md](../README.md) and the Flask dependency;
[`asgi.py`](../asgi.py) is already the same application standing alone under
uvicorn, and `python -m wsgi` stops being the entrypoint.

---

## What must not move before its wave

- **Nothing is removed from `application/`.** Both surfaces call the same use
  cases; deleting a blueprint deletes a spelling, never a behaviour. If
  removing a route would leave a use case with no caller, that is a separate
  decision with its own reason.
- **A route removed here is removed from the README's console-API table in the
  same commit.** `tests/migration/test_http_surface.py` compares that table to
  the routing table in both directions and will fail otherwise.
- **The Viewer's four routes cannot be removed from this repository alone.**
  The Viewer is pinned; a running Viewer built against the old relay would
  fail on a live document with no message a user can act on.
