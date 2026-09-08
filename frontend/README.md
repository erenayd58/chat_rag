# The console

A Next.js front end over [`/api/v1`](../docs/api-v1.md), and nothing else. It
holds no catalogue of its own: the chunking methods, the retrieval methods and
the model chain are all read from `/api/v1/meta/...` at run time, so a method
added to the library's registry appears here without a line changing.

## Running it

```bash
npm install
npm run dev            # http://localhost:3000
```

The backend is expected on `http://127.0.0.1:5005` (`python -m wsgi` in the
repository root, or `python -m asgi` once the legacy surface is gone). Point
it somewhere else with `CHAT_RAG_API_URL`; `next.config.mjs` rewrites
`/api/v1/*` there, so the browser only ever calls this server's own origin and
there is no CORS to configure.

```bash
npm run build          # production build
npm test               # vitest
npm run typecheck      # tsc --noEmit
```

## How it is laid out

```text
app/          one directory per screen (App Router)
app/viewer/   the Viewer: five tabs, and the only stylesheet of its own
components/   the shared UI: shell, primitives, pickers, dialogs
components/viewer/  the Viewer's screens and its two floating cards
lib/api/      every call to /api/v1, one module per resource
lib/viewer/   the payload's shape, its vocabulary, and the alignment rule
lib/          formatting, polling, hooks
types/        the contract's shapes, mirroring docs/api-v1.md
```

## The Viewer

`/viewer` is the one screen with a design of its own. The console is a control
surface; the Viewer is a *reading* surface -- a document printed on paper with
several chunkings drawn onto it -- so its tokens are scoped to `.v-root` in
`app/viewer/viewer.css` and the shell around it stays the console's.

The comparison it exists for is `lib/viewer/rows.ts`: the canonical units are
sliced at the union of every selected method's cut offsets, so the same text
lands in the same grid row in every column and a boundary one method draws and
another does not is visible on the words. It is a pure function and is tested
as one (`tests/viewerRows.test.ts`).

Two routes serve it and no more --
`GET /api/v1/documents/<id>/analysis/payload` and
`POST /api/v1/analysis-queries`. **No method key appears anywhere under
`app/viewer/` or `components/viewer/`**: the chips are the registry's keys
from `/api/v1/meta/chunking-methods`, narrowed to what a document has ready,
and "is this an orchestration, and over what" is that catalogue's own
`orchestration` / `baseline`.

`lib/api/client.ts` is the only module that calls `fetch`. It turns the
contract's refusal body into an `ApiError` carrying `type`, `status`,
`details` and `retryAfter`, which is what every screen branches on -- no
component reads a status code.
