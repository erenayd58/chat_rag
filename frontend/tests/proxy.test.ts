/**
 * The console's origin forwards `/api/v1` to the application, and it resolves
 * *where* the application is on every request.
 *
 * That last clause is the whole reason this file exists. The forwarding was a
 * `rewrites()` entry in `next.config.mjs` reading `CHAT_RAG_API_URL`, which
 * reads like configuration and is not: Next.js resolves `rewrites()` during
 * `next build` and freezes the destination into `.next/routes-manifest.json`.
 * The built console carried the developer default `http://127.0.0.1:5005`, and
 * inside a container `127.0.0.1` is the container -- so every screen failed
 * against an application that was healthy one hop away, and setting the
 * variable on the container changed nothing. The bug was invisible in
 * development, where the default happens to be right.
 *
 * So the property under test is not "it forwards" but "it asks again".
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { backendOrigin, forward } from '@/lib/api/proxy';

const ORIGINAL = process.env.CHAT_RAG_API_URL;

/** Records what was forwarded and answers with something recognisable. */
function record(answer?: Response) {
  const calls: { url: string; init: RequestInit }[] = [];
  const stub = vi.fn(async (input: string | URL | Request, init: RequestInit = {}) => {
    calls.push({ url: String(input), init });
    return answer ?? new Response('{"ok":true}', { status: 200 });
  });
  vi.stubGlobal('fetch', stub);
  return calls;
}

beforeEach(() => {
  delete process.env.CHAT_RAG_API_URL;
});

afterEach(() => {
  vi.unstubAllGlobals();
  if (ORIGINAL === undefined) delete process.env.CHAT_RAG_API_URL;
  else process.env.CHAT_RAG_API_URL = ORIGINAL;
});

describe('where the application is', () => {
  it('is read at request time, not frozen at build time', async () => {
    const calls = record();

    process.env.CHAT_RAG_API_URL = 'http://app:5005';
    await forward(new Request('http://console/api/v1/health'), ['health']);

    // The same module, the same process, a different deployment.
    process.env.CHAT_RAG_API_URL = 'http://elsewhere:9000';
    await forward(new Request('http://console/api/v1/health'), ['health']);

    expect(calls.map((call) => call.url)).toEqual([
      'http://app:5005/api/v1/health',
      'http://elsewhere:9000/api/v1/health',
    ]);
  });

  it('falls back to the developer default when nothing says otherwise', () => {
    expect(backendOrigin()).toBe('http://127.0.0.1:5005');
  });

  it('ignores a trailing slash rather than forwarding a doubled one', () => {
    process.env.CHAT_RAG_API_URL = 'http://app:5005/';
    expect(backendOrigin()).toBe('http://app:5005');
  });

  it('treats an empty value as unset', () => {
    process.env.CHAT_RAG_API_URL = '   ';
    expect(backendOrigin()).toBe('http://127.0.0.1:5005');
  });
});

describe('what it forwards', () => {
  it('keeps the path, the query string and the method', async () => {
    const calls = record();
    process.env.CHAT_RAG_API_URL = 'http://app:5005';

    await forward(
      new Request('http://console/api/v1/documents?knowledge_base_id=kb-1&limit=50', {
        method: 'GET',
      }),
      ['documents'],
    );

    expect(calls[0].url).toBe('http://app:5005/api/v1/documents?knowledge_base_id=kb-1&limit=50');
    expect(calls[0].init.method).toBe('GET');
  });

  it('drops the headers that describe this hop and keeps the rest', async () => {
    const calls = record();

    await forward(
      new Request('http://console/api/v1/queries', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          connection: 'keep-alive',
          'accept-encoding': 'gzip',
          'x-session-id': 'session-7',
        },
        body: '{}',
      }),
      ['queries'],
    );

    const sent = new Headers(calls[0].init.headers);
    expect(sent.get('content-type')).toBe('application/json');
    // The session id is how the contract scopes a pipeline; losing it here
    // would silently give every screen a different session.
    expect(sent.get('x-session-id')).toBe('session-7');
    for (const gone of ['host', 'connection', 'content-length', 'accept-encoding']) {
      expect(sent.get(gone)).toBeNull();
    }
  });

  it('does not forward Expect, which undici refuses outright', async () => {
    // `Expect: 100-continue` is what curl sends for any body over a kilobyte,
    // and what several HTTP libraries send for a large upload. The Node server
    // in front of this module has already answered it; forwarding it makes
    // undici reject the request with `NotSupportedError: expect header not
    // supported`, so the upload dies partway through as a 502 and the
    // application never sees it.
    //
    // Browsers never send it. That is why every screen worked, the live suite
    // passed, and the first `curl` upload of a real PDF failed at about four
    // megabytes.
    const calls = record();

    await forward(
      new Request('http://console/api/v1/documents', {
        method: 'POST',
        headers: { 'content-type': 'multipart/form-data; boundary=x', expect: '100-continue' },
        body: 'a large PDF',
      }),
      ['documents'],
    );

    expect(new Headers(calls[0].init.headers).get('expect')).toBeNull();
  });

  it('streams a body instead of reading it first', async () => {
    const calls = record();

    await forward(
      new Request('http://console/api/v1/documents', { method: 'POST', body: 'a PDF, really' }),
      ['documents'],
    );

    // `duplex: 'half'` is what makes a stream body legal; without it the whole
    // upload would have to be in this process's memory at once.
    expect((calls[0].init as { duplex?: string }).duplex).toBe('half');
    expect(calls[0].init.body).toBeDefined();
  });

  it('sends no body on a GET', async () => {
    const calls = record();
    await forward(new Request('http://console/api/v1/health'), ['health']);
    expect(calls[0].init.body).toBeUndefined();
  });
});

describe('what it answers', () => {
  it('passes the application refusal through untouched', async () => {
    // The contract's overload answer. `client.ts` branches on `type` and on
    // Retry-After; a proxy that rewrote either would be a second opinion
    // about the contract.
    record(
      new Response('{"error":{"type":"overloaded","message":"busy"}}', {
        status: 503,
        headers: { 'content-type': 'application/json', 'retry-after': '30' },
      }),
    );

    const response = await forward(
      new Request('http://console/api/v1/documents', { method: 'POST', body: 'x' }),
      ['documents'],
    );

    expect(response.status).toBe(503);
    expect(response.headers.get('retry-after')).toBe('30');
    expect(await response.json()).toEqual({ error: { type: 'overloaded', message: 'busy' } });
  });

  it('strips the encoding headers that describe a body fetch already decoded', async () => {
    record(
      new Response('{"ok":true}', {
        status: 200,
        headers: {
          'content-type': 'application/json',
          'content-encoding': 'gzip',
        },
      }),
    );

    const response = await forward(new Request('http://console/api/v1/health'), ['health']);

    expect(response.headers.get('content-encoding')).toBeNull();
    expect(response.headers.get('content-length')).toBeNull();
    expect(response.headers.get('content-type')).toBe('application/json');
  });

  it('answers an unreachable application in the contract shape', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('fetch failed');
      }),
    );
    process.env.CHAT_RAG_API_URL = 'http://app:5005';

    const response = await forward(new Request('http://console/api/v1/health'), ['health']);

    expect(response.status).toBe(502);
    const body = await response.json();
    // `network` is the client's own name for "the server was not there", so a
    // console that cannot reach the application says that rather than failing
    // to parse an HTML error page.
    expect(body.error.type).toBe('network');
    expect(body.error.message).toContain('http://app:5005');
  });
});
