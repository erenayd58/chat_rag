# `/api/v1` — the product contract

The surface a client builds against. It is versioned because it has to keep
working while everything under it is replaced: the state files by PostgreSQL,
Chroma by pgvector, the templates by a Next.js front end. None of those is a
reason to change anything on this page — and neither was the first of them,
which has already happened. These routes were Flask and are now **FastAPI**,
at the same URLs, with the same statuses and the same bodies.

The Flask-era surface is still served beside it — see *The console API* in
[../README.md](../README.md). That one is compatibility, this one is the
contract. Both call the same application layer, over one container in one
process, so they cannot disagree about behaviour, only about spelling.

---

## Conventions

**Identity.** `id` is the resource's own. Everything else that names one is
suffixed: `knowledge_base_id`, `document_id`, `job_id`, `content_id`. All
field names are `snake_case`.

Two identities are kept apart everywhere, because conflating them either loses
an upload or shares a choice that is not shared:

| | what it identifies |
|---|---|
| `id` on a document | **the upload** — one file, ingested into one knowledge base |
| `content_id` | **the bytes** — shared by every upload of the same document, and what an analysis and its variants belong to |

**Shapes.** Three, and no fourth:

```jsonc
// a resource — the object itself, no envelope
{ "id": "kb-yillik", "name": "Yillik raporlar", ... }

// a collection — always both keys, even on one page
{ "items": [ ... ], "page": { "offset": 0, "limit": 50, "total": 214 } }

// a refusal
{ "error": { "type": "not_found", "message": "...", "details": { ... } } }
```

An action with nothing to return answers **204** with no body.

**Pagination.** `?offset=` and `?limit=` on every collection. `limit` defaults
to 50 and is capped at 200. Values that do not parse fall back to the default
rather than being refused — a page size is a presentation decision, and
refusing `?limit=abc` breaks a pasted link to teach nobody anything.

**Refusals.** `type` is what a client branches on. It does not change when a
message is reworded.

| `type` | status | what it means |
|---|---|---|
| `invalid_request` | 400 | the request or its payload is wrong; nothing was created |
| `not_found` | 404 | the resource is not here |
| `not_ready` | 409 | it exists and is not finished — `details.state` says where it got to |
| `conflict` | 409 | it exists, and its current state refuses this |
| `unavailable` | 503 | a capability this deployment cannot provide right now |
| `overloaded` | 503 | no capacity; refused rather than queued. `Retry-After`, and `details.reason` names the limit |
| `timeout` | 504 | the deadline passed |
| `internal` | 500 | the server failed |

`POST /api/v1/documents` and `POST /api/v1/queries` are the two that can be
refused under load. Nothing is queued behind a refusal and nothing is kept.

**States.** Two vocabularies, each with one meaning:

- an ingest job: `queued` → `running` → `succeeded` | `failed` | `timed_out` |
  `cancelled` | `interrupted`;
- a document's analysis: `missing` | `pending` | `running` | `ready` |
  `failed`.

**Not on the wire, deliberately.** File paths, the ingest ledger's key, the
vector store's provider or location, state-file names, provider keys. Those
are what the persistence migration changes, and no client should have to.

**Pass-through fields.** A chunk's `metadata` and a query's `diagnostics`
carry what the product produced. They are useful and they are explicitly *not*
contractual: pinning them would freeze internals this contract exists to leave
free.

---

## Discovery

| endpoint | answers |
|---|---|
| `GET /api/v1/meta/chunking-methods` | every chunking method, from the library's registry |
| `GET /api/v1/meta/retrieval-methods` | which retrieval methods a knowledge base's retriever can serve |
| `GET /api/v1/meta/models` | the configured model chain — names and endpoints, never a key |
| `GET /api/v1/health` | liveness, readiness, and one line of capacity |
| `GET /api/v1/openapi.json` | this contract, machine-readable |

`chunking-methods` is the one that matters architecturally. It is a projection
of `amsc.chunking.registry` and **there is no second method catalogue** — not
in this repository, not in the console's JavaScript, not in the Viewer, and
not in whatever the front end becomes. Adding a chunking method costs its
implementation, its registration in that registry and its own tests; no API
list is edited. A method this machine cannot run is listed as
`"available": false` with `unavailable_reason`, never hidden, so a picker can
explain the gap instead of silently dropping the option.

`openapi.json` is **generated** from the routers and their request and
response models, so it cannot drift from what is served: there is no second
document to keep in step, and a field that is not declared on a response model
cannot appear in either the schema or the answer. Point any OpenAPI viewer or
client generator at it. The interactive documentation pages are deliberately
not served — they fetch their JavaScript from a public CDN, and nothing else
this product serves needs the network to render.

```jsonc
{ "items": [ {
    "key": "markdown",
    "label": "Markdown",
    "summary": "...",
    "engine": "structural",
    "available": true,
    "unavailable_reason": null,
    "uses_model": false,
    "default": false,
    "orchestration": false,   // an orchestration runs over a baseline partition
    "baseline": null          // ...and names it here
} ], "page": { ... } }
```

## Knowledge bases

| endpoint | |
|---|---|
| `GET /api/v1/knowledge-bases` | list |
| `POST /api/v1/knowledge-bases` | create → **201**, `Location` |
| `GET /api/v1/knowledge-bases/<kb_id>` | one |
| `PATCH /api/v1/knowledge-bases/<kb_id>` | `name` and `extra` only |
| `DELETE /api/v1/knowledge-bases/<kb_id>` | → **204** |
| `GET /api/v1/knowledge-bases/<kb_id>/chunks` | browse the corpus; `?search=` filters by phrase |
| `GET /api/v1/knowledge-bases/<kb_id>/embedding-index` | do the stored vectors belong to the configured model |
| `POST /api/v1/knowledge-bases/<kb_id>/embedding-index/rebuild` | re-embed every chunk |

A knowledge base's chunker, embedding model and storage are fixed at creation,
because the corpus that gets ingested depends on them. Deleting one takes its
corpus; its documents' ledger rows and analyses deliberately survive, so a user
does not lose the record that a file was ever ingested.

## Documents

| endpoint | |
|---|---|
| `GET /api/v1/documents` | list; `?knowledge_base_id=` narrows |
| `POST /api/v1/documents` | upload → **202**, the job, `Location` |
| `GET /api/v1/documents/<document_id>` | one, with its analysis |
| `DELETE /api/v1/documents/<document_id>` | → **204** |
| `GET /api/v1/documents/<document_id>/chunks` | what it was *indexed* as |
| `GET /api/v1/documents/<document_id>/units` | the parser's canonical reading, before any chunker |

Uploading is `multipart/form-data`: `file`, `knowledge_base_id`, and `methods`
(repeated, or one comma-separated field). It is **always asynchronous** — the
answer is the job. One upload is one parse and one canonical, and every
selected method runs over that same canonical, so three methods cost one parse.
What is *indexed* for retrieval stays the knowledge base's own chunker; the
methods are an analysis choice and do not change it.

The same bytes submitted again while the first upload is still in flight are
attached to that job rather than parsed twice (`attached_uploads` says so).

Deleting a document takes its chunks, its ledger row and its own analysis. It
does **not** take the content: another upload of the same bytes keeps the
shared analysis and its variants, and only the last upload of a content takes
that down with it.

## Analysis

| endpoint | |
|---|---|
| `GET /api/v1/documents/<document_id>/analysis` | where it got to — always **200** |
| `POST /api/v1/documents/<document_id>/analysis` | queue or retry → **202** |
| `POST /api/v1/documents/<document_id>/analysis/methods` | add variants → **202** |
| `GET /api/v1/documents/<document_id>/analysis/methods/<method>/chunks` | one method's rows |

The analysis resource keeps two levels apart, because they are two facts and a
screen that merges them lies in one direction or the other:

```jsonc
{
  "status": "ready",
  "content_id": "<sha256 of the bytes>",
  "selected_methods": ["standard", "markdown"],  // what THIS upload asked for
  "ready_methods":    ["standard"],              // selected ∩ built  ← what it may be asked about
  "failed_methods":   [],
  "content": {                                   // the shared analysis of these bytes
    "requested_methods": ["standard", "markdown", "agentic"],
    "ready_methods":     ["standard", "agentic"],
    "shared_with_document_ids": ["doc-1", "doc-2"]
  }
}
```

`ready_methods` at the document level is the whole `visible = selected ∩ ready`
rule: it is what this upload may be asked about, never another upload's
variants. Asking for a method outside it gets one of three answers, and the
difference matters to a client:

- **400** — not a method this deployment knows about;
- **404** — a real method, and not one *this upload* selected. The content may
  well have it, from another upload; it is still not this document's to serve;
- **409 `not_ready`** — selected, not built yet. Keep polling; the body carries
  the state.

## Ingest jobs

| endpoint | |
|---|---|
| `GET /api/v1/ingest-jobs` | list; `?knowledge_base_id=`, `?active=true` |
| `GET /api/v1/ingest-jobs/<job_id>` | one |
| `DELETE /api/v1/ingest-jobs/<job_id>` | cancel — returns the job, because the answer may be `succeeded` |

A job id outlives the process that minted it: every transition is journalled
and start-up settles anything in flight against the ingest ledger
(`restart_settled` says a restart decided this one). A **404** therefore means
one thing only — the job finished longer ago than jobs are kept. It never means
the job was lost; the document list is the record of what was ingested.

Cancelling a running job that has already written its document answers
`succeeded`, not `cancelled`. The ledger write is a job's last act, so a
document in the ledger was fully committed.

## Asking

| endpoint | |
|---|---|
| `POST /api/v1/queries` | retrieve, then answer with citations |
| `POST /api/v1/searches` | retrieve only: the ranked chunks |

Both are POSTs and neither creates anything: a question is user text of
unbounded length that should not land in an access log or be cached in
between.

A query answers with `citations` — the sources it was given, each carrying the
chunk verbatim and whether the answer actually cited it — and `grounded`,
false when the model answered without citing any of them. A client that shows
the answer without those two cannot tell an answer from a guess.

A search takes `method`: `hybrid`, `bm25` or `vector`. One the configured
retriever cannot serve is **400** with the reason — a lexical-only profile has
no vectors, and saying so is the answer. `GET /api/v1/meta/retrieval-methods`
is how a client knows in advance.

---

## Not here, on purpose

Each of these is served by the legacy surface and was not promoted:

| what | why not |
|---|---|
| `PUT`/`DELETE` on a single chunk | editing an indexed chunk changes the corpus behind the ingest ledger's back. It is a lab affordance, not a product operation |
| the gold set | an offline evaluation input, driven by `python -m cli`. It is not part of what a console client does |
| `/api/demo/viewer`, `/api/demo/workspace` | a probe of a companion dev server, and a snapshot that `GET /api/v1/documents` now answers truthfully per document |
| `/api/ops/metrics` | an operator surface whose contents are deliberately free to change. `GET /api/v1/health` is the stable half |
| a synchronous upload | it exists on the legacy surface because it always did, and it is the reason that adapter needs a semaphore to stop uploads holding every request thread |
