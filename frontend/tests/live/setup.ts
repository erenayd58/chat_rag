/**
 * The live suite's shim, and why it needs one.
 *
 * jsdom gives the test three browser objects that Node's `fetch` will not
 * take, so this wrapper translates them and nothing else:
 *
 *   - a **relative URL** -- a browser resolves `/api/v1/...` against the
 *     page's origin; Node demands an absolute one;
 *   - an **AbortSignal** -- jsdom installs its own class over the one undici
 *     accepts, so an abort is honoured here by rejecting the promise instead;
 *   - a **FormData** -- undici serialises only its own, and silently drops a
 *     jsdom `File`, which the server then reports as "a file is required". The
 *     multipart body is therefore encoded here, exactly as a browser would.
 *
 * The screens, the API client and the polling are the shipped ones.
 */

import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import { reset } from '@/lib/methodCatalogue';

export const BASE_URL = process.env.CHAT_RAG_CONSOLE_URL || 'http://127.0.0.1:3000';

const real = globalThis.fetch;

const aborted = () => new DOMException('Aborted', 'AbortError');

const CRLF = '\r\n';

/** jsdom's Blob has no `arrayBuffer()`, so read it the way a browser could. */
function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
}

/** One multipart body, byte for byte what a browser would have sent. */
async function multipart(form: FormData): Promise<{ body: Uint8Array; contentType: string }> {
  const boundary = `----jsdomFormBoundary${Math.random().toString(16).slice(2)}`;
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];
  for (const [name, value] of form.entries()) {
    if (typeof value === 'string') {
      parts.push(
        encoder.encode(
          `--${boundary}${CRLF}Content-Disposition: form-data; name="${name}"${CRLF}${CRLF}${value}${CRLF}`,
        ),
      );
      continue;
    }
    const file = value as File;
    parts.push(
      encoder.encode(
        `--${boundary}${CRLF}Content-Disposition: form-data; name="${name}"; filename="${file.name}"${CRLF}` +
          `Content-Type: ${file.type || 'application/octet-stream'}${CRLF}${CRLF}`,
      ),
    );
    parts.push(await bytesOf(file));
    parts.push(encoder.encode(CRLF));
  }
  parts.push(encoder.encode(`--${boundary}--${CRLF}`));

  const body = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    body.set(part, offset);
    offset += part.length;
  }
  return { body, contentType: `multipart/form-data; boundary=${boundary}` };
}

globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
  const url =
    typeof input === 'string' && input.startsWith('/') ? `${BASE_URL}${input}` : (input as RequestInfo);
  const { signal, ...rest } = init;

  if (rest.body instanceof FormData) {
    const { body, contentType } = await multipart(rest.body);
    rest.body = body as unknown as BodyInit;
    rest.headers = { ...(rest.headers as Record<string, string>), 'Content-Type': contentType };
  }

  if (!signal) return real(url, rest);
  if (signal.aborted) throw aborted();
  return Promise.race([
    real(url, rest),
    new Promise<Response>((_, reject) => {
      signal.addEventListener('abort', () => reject(aborted()), { once: true });
    }),
  ]);
}) as typeof fetch;

afterEach(() => {
  cleanup();
  // The chunking catalogue is shared for a page load, not for a run.
  reset();
});
