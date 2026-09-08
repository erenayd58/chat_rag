/**
 * The client, held to the contract's conventions.
 *
 * These are the rules `docs/api-v1.md` states and every screen depends on: a
 * refusal is `{error:{type,message,details}}` and `type` is what a caller
 * branches on; 204 has no body; `Retry-After` accompanies a 503; a page is
 * `{items, page}` and a collection is walked with `offset`/`limit`.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, collect, queryString, request } from '@/lib/api/client';

function answer(body: unknown, init: { status?: number; headers?: Record<string, string> } = {}) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('queryString', () => {
  it('leaves out what nobody set', () => {
    expect(queryString({ a: 1, b: undefined, c: null, d: '' })).toBe('?a=1');
  });

  it('is empty rather than a bare question mark', () => {
    expect(queryString({})).toBe('');
  });
});

describe('request', () => {
  it('calls the versioned prefix and returns the body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(answer({ id: 'kb-1' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(request('/knowledge-bases/kb-1')).resolves.toEqual({ id: 'kb-1' });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/knowledge-bases/kb-1');
  });

  it('answers a 204 with nothing, and does not try to parse it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
    await expect(request('/knowledge-bases/kb-1', { method: 'DELETE' })).resolves.toBeUndefined();
  });

  it.each([
    [400, 'invalid_request'],
    [404, 'not_found'],
    [409, 'not_ready'],
    [409, 'conflict'],
    [503, 'unavailable'],
    [504, 'timeout'],
    [500, 'internal'],
  ])('turns a %i into ApiError of type %s', async (status, type) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(answer({ error: { type, message: 'nope' } }, { status })),
    );
    const error = (await request('/queries', { method: 'POST', json: {} }).catch(
      (cause) => cause,
    )) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.type).toBe(type);
    expect(error.status).toBe(status);
    expect(error.message).toBe('nope');
  });

  it('carries details and Retry-After off an overload', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        answer(
          { error: { type: 'overloaded', message: 'busy', details: { reason: 'queue_full' } } },
          { status: 503, headers: { 'Retry-After': '12' } },
        ),
      ),
    );
    const error = (await request('/documents', { method: 'POST' }).catch((cause) => cause)) as ApiError;
    expect(error.type).toBe('overloaded');
    expect(error.retryAfter).toBe(12);
    expect(error.details).toEqual({ reason: 'queue_full' });
    expect(error.isTransient).toBe(true);
  });

  it('names a body it did not write rather than guessing a type', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>502</html>', { status: 502 })));
    const error = (await request('/health').catch((cause) => cause)) as ApiError;
    expect(error.type).toBe('internal');
    expect(error.status).toBe(502);
  });

  it('tells a connection failure apart from a refusal', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed to fetch')));
    const error = (await request('/health').catch((cause) => cause)) as ApiError;
    expect(error.type).toBe('network');
    expect(error.status).toBe(0);
  });

  it('does not set a content type on a multipart body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(answer({ id: 'job-1' }, { status: 202 }));
    vi.stubGlobal('fetch', fetchMock);
    const form = new FormData();
    form.append('file', new File(['x'], 'x.txt'));

    await request('/documents', { method: 'POST', form });

    expect(fetchMock.mock.calls[0][1].headers).toEqual({});
    expect(fetchMock.mock.calls[0][1].body).toBe(form);
  });
});

describe('collect', () => {
  it('walks every page of a collection', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(answer({ items: [1, 2], page: { offset: 0, limit: 200, total: 3 } }))
      .mockResolvedValueOnce(answer({ items: [3], page: { offset: 2, limit: 200, total: 3 } }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(collect<number>('/documents')).resolves.toEqual([1, 2, 3]);
    expect(fetchMock.mock.calls[0][0]).toContain('offset=0');
    expect(fetchMock.mock.calls[1][0]).toContain('offset=2');
  });

  it('stops on an empty page rather than looping for ever', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(answer({ items: [], page: { offset: 0, limit: 200, total: 9 } }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(collect('/documents')).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
