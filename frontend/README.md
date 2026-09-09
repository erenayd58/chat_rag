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

The backend is expected on `http://127.0.0.1:5005` (`python -m asgi` in the
repository root). Point it somewhere else with `CHAT_RAG_API_URL`.

```bash
npm run build          # production build
npm test               # vitest
npm run typecheck      # tsc --noEmit
```

In the deployed stack this is a container of its own
([`Dockerfile`](Dockerfile)), built by `docker compose up --build` beside the
application and the database, and it is the only published port.

## How it reaches the application

The browser never learns where the backend is. Every `/api/v1/...` call goes to
this server's own origin and this server forwards it, so the contract is
same-origin -- no CORS, and no preflight in front of a multipart upload -- and
exactly one setting knows the application's address.

That forwarding is [`lib/api/proxy.ts`](lib/api/proxy.ts), reached through the
catch-all route `app/api/v1/[...path]/route.ts`. It **was** a `rewrites()`
entry in `next.config.mjs`, which reads like configuration and is not: Next.js
resolves `rewrites()` during `next build` and writes the destination into
`.next/routes-manifest.json`. The built console therefore carried the developer
default `http://127.0.0.1:5005`, which inside a container is that container --
so every screen failed against an application that was healthy one hop away,
and setting `CHAT_RAG_API_URL` on the container changed nothing. The route
handler reads the address on each request instead, and
[`tests/proxy.test.ts`](tests/proxy.test.ts) holds that property.

`proxy.ts` is the second and last module allowed to call `fetch`
(`tests/surface.test.ts` enforces the pair): `client.ts` is how a *screen*
reaches the contract, and this is how this server reaches the application. It
streams the body rather than buffering it, and passes status, body and
`Retry-After` through untouched -- the refusal taxonomy is the application's.

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
