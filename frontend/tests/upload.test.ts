/**
 * The upload contract: 202, a job, and a poll that knows what a 404 means.
 *
 * The two things a front end gets wrong here are treating the 202 as if the
 * document existed, and treating a forgotten job as a failure. Both are held
 * to below.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as documents from '@/lib/api/documents';
import { FORGOTTEN_JOB, followIngestJob, isActive, isForgotten } from '@/lib/jobs';

function answer(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const job = (status: string, extra: Record<string, unknown> = {}) => ({
  id: 'job-1',
  status,
  methods: [],
  attached_uploads: 0,
  ...extra,
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('upload', () => {
  it('posts multipart with one repeated field per method, and returns the job', async () => {
    const fetchMock = vi.fn().mockResolvedValue(answer(job('queued'), 202));
    vi.stubGlobal('fetch', fetchMock);

    const accepted = await documents.upload({
      file: new File(['hello'], 'rapor.pdf', { type: 'application/pdf' }),
      knowledgeBaseId: 'kb-1',
      methods: ['a-method', 'another-method'],
    });

    expect(accepted.status).toBe('queued');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/v1/documents');
    expect(init.method).toBe('POST');
    const form = init.body as FormData;
    expect(form.get('knowledge_base_id')).toBe('kb-1');
    expect(form.getAll('methods')).toEqual(['a-method', 'another-method']);
    expect((form.get('file') as File).name).toBe('rapor.pdf');
  });
});

describe('followIngestJob', () => {
  it('polls until the job settles', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(answer(job('queued', { queue_position: 2 })))
      .mockResolvedValueOnce(answer(job('running')))
      .mockResolvedValueOnce(answer(job('succeeded', { result: { chunk_count: 12 } })));
    vi.stubGlobal('fetch', fetchMock);
    vi.useFakeTimers();

    const seen: string[] = [];
    const followed = followIngestJob('job-1', { onUpdate: (update) => seen.push(update.status!) });
    await vi.runAllTimersAsync();
    const settled = await followed;
    vi.useRealTimers();

    expect(seen).toEqual(['queued', 'running', 'succeeded']);
    expect(isForgotten(settled)).toBe(false);
    expect(settled.status).toBe('succeeded');
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('reads a 404 as "no longer on record", not as a failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(answer({ error: { type: 'not_found', message: 'gone' } }, 404)),
    );

    const settled = await followIngestJob('job-1');
    expect(isForgotten(settled)).toBe(true);
    expect(settled.status).toBe(FORGOTTEN_JOB);
  });

  it('re-raises a refusal that is not a 404', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(answer({ error: { type: 'internal', message: 'boom' } }, 500)),
    );
    await expect(followIngestJob('job-1')).rejects.toMatchObject({ type: 'internal' });
  });
});

describe('isActive', () => {
  it.each([
    ['queued', true],
    ['running', true],
    ['succeeded', false],
    ['failed', false],
    ['timed_out', false],
    ['cancelled', false],
    ['interrupted', false],
  ])('%s -> %s', (status, expected) => {
    expect(isActive({ status })).toBe(expected);
  });
});
