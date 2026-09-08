# `/api/v1` — the product contract

The surface a client builds against. It is versioned because it has to keep
working while everything under it is replaced: the state files by PostgreSQL,
Chroma by pgvector, the templates by a Next.js front end. None of those is a
reason to change anything on this page — and none of the first three was.
These routes were Flask and are now **FastAPI**, at the same URLs, with the
same statuses and the same bodies; the records and then the vectors moved into
PostgreSQL underneath them, and this page did not change for either.

The Flask-era surface that was served beside it is gone
([legacy-removal.md](legacy-removal.md)). This is the whole HTTP surface bar
one route: `/api/ops/metrics`, an operator's reading, deliberately not on
this contract because its body reports internals that are free to change.

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

Three things a generated document cannot infer are declared for it, in
`interfaces/http/v1/openapi.py`, so a generated client gets them as types
rather than as prose: the refusal body above on **every** operation, the
`Location` header on the two answers that send one, and the `Retry-After` on
the 503. Nothing else is added, and the **422** FastAPI documents by itself is
taken back out, because this surface answers a payload it cannot read with 400
`invalid_request` and a document advertising a status the server never sends is
worse than none.

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
| `GET /api/v1/documents/<document_id>/analysis/payload` | the whole analysis, as a reader's view of it |

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

### The payload

`.../analysis/payload` answers the same analysis as one document rather than as
one method at a time: the parser's canonical units in reading order, and per
ready method the chunks **plus the unit offsets each one cuts at**.

Those offsets are why it exists beside the per-method rows. `.../methods/<m>/chunks`
says what a method produced; only the payload says where a chunk starts and
ends *inside* a canonical unit, which is what lets several methods be drawn
down one column of text and compared on the page instead of by chunk number.
Nothing else on this contract carries it, and a client cannot compute it: the
mapping is written by the packager, from the chunker's own record of what it
consumed.

It answers **409 `not_ready`** while nothing this upload selected has been
built, with the analysis state, so a screen polls rather than gives up. Its
`payload` object is **pass-through**, for the same reason a chunk's `metadata`
is: it is a render model, it grows a field whenever a chunker records something
new, and freezing it here would spend a version number on internals this
contract exists to leave free. `ready_methods` beside it is not pass-through —
it is exactly the arms the payload carries.

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
| `POST /api/v1/analysis-queries` | one question, one document, through each chunking method |

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

### Asking the analysis instead of the corpus

A query and a search both read a knowledge base: one document set, chunked the
one way its knowledge base ingests. An **analysis query** reads one document's
*analysis arms* — the same document chunked several ways, one index per method,
built from the packaged rows — so the only thing that differs between arms is
the chunker. That is what makes the answer a comparison of chunkers, and it is
the one question this contract answers that a knowledge base cannot.

```jsonc
{ "document_id": "doc-1", "question": "...", "methods": ["standard", "agentic"],
  "top_k": 5, "answer": true }
```

`methods` absent means every method this upload has ready. A name this
deployment does not know is **400** with the supported list; a name it knows
that is not built for *this* upload is **404** with what is — the same two
refusals, meaning the same two things, as everywhere else on the analysis.
`answer: false` stops after retrieval.

The answer is one entry per method, always in the same shape whether one ran or
four: a comparison of one is still a comparison, and a client that branches on
the count has two rendering paths where it needs one. Each entry carries its
own `status` (`ok`, `insufficient`, `no_answer_model`, `answer_error`), so an
arm that could not be answered does not fail the request — in a comparison the
other arms are still the answer — and `unit_overlap`, how much of its retrieved
context the other arms also retrieved, which is the number that says whether
two chunkers found the same evidence or different evidence. `sources` is
pass-through, like a chunk's `metadata`.

It runs under the same admission and deadline as `/queries`. Retrieval plus an
answer-model call per arm is not a lighter thing than a query, and leaving it
outside the bound would make it the way around it.

---

## Not here, on purpose

Each of these was served by the Flask-era surface and was not promoted. That
surface is gone, so each row is now a thing the product does not offer at all:

| what | why not |
|---|---|
| `PUT`/`DELETE` on a single chunk | editing an indexed chunk changes the corpus behind the ingest ledger's back. It was a lab affordance, not a product operation |
| the gold set | an offline evaluation input, driven by `python -m cli`. It is not part of what a console client does — and with the Lab gone, nothing writes a runtime entry any more |
| `/api/stats` | a count of documents, chunks and bytes. `GET /api/v1/documents` carries every number in it per document, and `GET /api/v1/health` the capacity half |
| a workspace snapshot | one request answering "every base, every document, every analysis state". It is `GET /api/v1/knowledge-bases` and `GET /api/v1/documents`, the latter already carrying each document's `analysis` block |
| a synchronous upload | it existed because it always had, and it was why that adapter needed a semaphore to stop uploads holding every request thread. An upload is a job here, always |
| `/api/ops/metrics` | **still served**, at that path, and deliberately not versioned: its contents are free to change with the internals they report on. `GET /api/v1/health` is the stable half |

[legacy-removal.md](legacy-removal.md) is the other half of this table: every
endpoint the Flask-era surface served, what here replaced it, and what was
kept.
