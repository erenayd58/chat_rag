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
components/   the shared UI: shell, primitives, pickers, dialogs
lib/api/      every call to /api/v1, one module per resource
lib/          formatting, polling, hooks
types/        the contract's shapes, mirroring docs/api-v1.md
```

`lib/api/client.ts` is the only module that calls `fetch`. It turns the
contract's refusal body into an `ApiError` carrying `type`, `status`,
`details` and `retryAfter`, which is what every screen branches on -- no
component reads a status code.
