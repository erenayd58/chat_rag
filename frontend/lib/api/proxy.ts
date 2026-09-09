/**
 * The console's own origin, forwarding `/api/v1` to the application.
 *
 * The browser must never learn where the backend is: every call from
 * `client.ts` goes to this server, which passes it on. That keeps the contract
 * same-origin -- no CORS, and no preflight in front of a multipart upload --
 * and leaves exactly one place that knows the application's address.
 *
 * **Why this is a route handler and not a `rewrites()` entry.** It was one.
 * `next.config.mjs` read `CHAT_RAG_API_URL` and returned a rewrite, which
 * looks like configuration and is not: Next.js evaluates `rewrites()` during
 * `next build` and writes the resolved destination into
 * `.next/routes-manifest.json`. The built console therefore carried
 * `http://127.0.0.1:5005` -- the developer default, frozen at build time --
 * and setting `CHAT_RAG_API_URL` on the container changed nothing. Inside a
 * container `127.0.0.1` is the container, so every screen failed to reach an
 * application that was running and healthy one network hop away.
 *
 * This module reads the address on each request instead, so the same image
 * runs against a compose service, a staging host or a developer's laptop with
 * no rebuild. `tests/proxy.test.ts` holds that property.
 *
 * The body is streamed rather than buffered, so uploading a large PDF does not
 * put the file in this process's memory on its way through.
 */

/** Where the application is, resolved per request -- never at build time. */
export function backendOrigin(): string {
  const configured = (process.env.CHAT_RAG_API_URL || '').trim();
  return (configured || 'http://127.0.0.1:5005').replace(/\/+$/, '');
}

/**
 * Headers that describe one TCP hop and must not be copied onto the next.
 *
 * `content-length` is here because the body is forwarded as a stream: undici
 * frames it chunked, and a length copied off the incoming request would
 * contradict what is actually sent. `accept-encoding` is dropped so the
 * answer arrives decoded and this process does not have to re-frame it.
 *
 * `expect` is here for a sharper reason. A client uploading a large file
 * commonly sends `Expect: 100-continue` -- curl does it for any body over a
 * kilobyte, and so do several HTTP libraries -- and the Node server in front
 * of this module has already answered it by the time the body arrives.
 * Forwarding it makes undici refuse the whole request with
 * `NotSupportedError: expect header not supported`, which surfaces as the
 * useless `fetch failed`. The upload dies partway through with a 502 and the
 * application never sees it.
 *
 * It was invisible for exactly the reason it is worth a paragraph: **browsers
 * never send `Expect`**. Every screen worked, the live suite passed, and the
 * first `curl` upload of a real PDF failed at about four megabytes.
 */
const HOP_BY_HOP_REQUEST = [
  'host', 'connection', 'keep-alive', 'transfer-encoding', 'upgrade',
  'proxy-connection', 'te', 'trailer', 'content-length', 'accept-encoding',
  'expect',
];

/**
 * The same, on the way back, plus the two that would describe a body this
 * process has already decoded: `fetch` gunzips transparently, so forwarding
 * the original `content-encoding` and `content-length` would hand the browser
 * a plain body labelled as compressed.
 */
const HOP_BY_HOP_RESPONSE = [
  'connection', 'keep-alive', 'transfer-encoding', 'upgrade',
  'proxy-connection', 'trailer', 'content-encoding', 'content-length',
];

function without(headers: Headers, drop: string[]): Headers {
  const kept = new Headers(headers);
  for (const name of drop) kept.delete(name);
  return kept;
}

/**
 * Forward one request to the application and return its answer verbatim.
 *
 * Status, body and every header that is not hop-by-hop are passed through
 * untouched: the refusal taxonomy `client.ts` branches on -- the `type` in the
 * body, and `Retry-After` on a 503 -- is the application's, and a proxy that
 * rewrote any of it would be a second opinion about the contract.
 */
export async function forward(request: Request, path: string[]): Promise<Response> {
  const incoming = new URL(request.url);
  const target = new URL(
    `/api/v1/${path.map(encodeURIComponent).join('/')}${incoming.search}`,
    backendOrigin(),
  );

  const hasBody = request.method !== 'GET' && request.method !== 'HEAD';
  const init: RequestInit & { duplex?: 'half' } = {
    method: request.method,
    headers: without(request.headers, HOP_BY_HOP_REQUEST),
    redirect: 'manual',
    // Stream the body through rather than reading it first: an upload is a
    // whole PDF, and buffering it here would double the memory a large one
    // costs and delay the application's first byte until the last.
    ...(hasBody ? { body: request.body, duplex: 'half' as const } : {}),
  };

  let response: Response;
  try {
    response = await fetch(target, init);
  } catch (error) {
    // The application is not there. Answer in the contract's own shape so the
    // console reports "the server was not there" rather than parsing an HTML
    // error page Next.js would otherwise render.
    //
    // `cause` is carried because without it this is always the string "fetch
    // failed", which says nothing: a refused connection, an unknown host and a
    // body that could not be streamed all read the same. The cause is what
    // tells them apart, and it is the message an operator needs.
    const cause = (error as { cause?: unknown })?.cause;
    return Response.json(
      {
        error: {
          type: 'network',
          message: `The application could not be reached at ${backendOrigin()}.`,
          details: {
            reason: error instanceof Error ? error.message : String(error),
            cause: cause instanceof Error ? `${cause.name}: ${cause.message}`
                 : cause === undefined ? null : String(cause),
          },
        },
      },
      { status: 502 },
    );
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: without(response.headers, HOP_BY_HOP_RESPONSE),
  });
}
