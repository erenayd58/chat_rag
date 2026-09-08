/**
 * Ingest jobs: what became of a submitted upload.
 *
 * A **404** here means one thing only -- the job finished longer ago than jobs
 * are kept. It never means the job was lost; the document list is the record
 * of what was ingested, which is why `lib/jobs.ts` treats it as an answer
 * rather than as a failure.
 */

import { request } from './client';
import type { IngestJob, IngestJobCollection } from '@/types/api';

const BASE = '/ingest-jobs';

export function list(
  options: { knowledgeBaseId?: string | null; activeOnly?: boolean; signal?: AbortSignal } = {},
): Promise<IngestJobCollection> {
  return request<IngestJobCollection>(BASE, {
    query: {
      knowledge_base_id: options.knowledgeBaseId,
      active: options.activeOnly ? 'true' : undefined,
    },
    signal: options.signal,
  });
}

export function get(jobId: string, signal?: AbortSignal): Promise<IngestJob> {
  return request<IngestJob>(`${BASE}/${encodeURIComponent(jobId)}`, { signal });
}

/**
 * Cancel one, and read what actually happened.
 *
 * The answer may be `succeeded`: the ledger write is a job's last act, so a
 * job that has already written its document was finished, not cancelled.
 */
export function cancel(jobId: string): Promise<IngestJob> {
  return request<IngestJob>(`${BASE}/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}
