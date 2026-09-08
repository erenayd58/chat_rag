/**
 * Following an ingest job from 202 to a terminal state.
 *
 * The upload contract is three steps and this is the middle one:
 *
 *     POST /api/v1/documents  ->  202 + the job
 *     GET  /api/v1/ingest-jobs/<id>   until it settles
 *     the document is in GET /api/v1/documents
 *
 * The delay grows, because a Deep Analysis run takes minutes and polling it
 * every second is a request per second that answers `running`. A 404 is an
 * answer, not a failure: a job id outlives the process that minted it, so the
 * only thing a 404 can mean is that it finished longer ago than jobs are kept.
 */

import { ApiError } from './api/client';
import * as ingestJobs from './api/ingestJobs';
import { ACTIVE_JOB_STATES, type IngestJob } from '@/types/api';

const FIRST_DELAY_MS = 800;
const DELAY_STEP_MS = 400;
const MAX_DELAY_MS = 3000;

/** What a caller gets back when the job is no longer on record. */
export const FORGOTTEN_JOB = 'forgotten';

export type FollowedJob = IngestJob | { status: typeof FORGOTTEN_JOB; id: string | null };

export function isActive(job: { status?: string | null } | null | undefined): boolean {
  if (!job || !job.status) return false;
  return (ACTIVE_JOB_STATES as readonly string[]).includes(job.status);
}

export function isForgotten(job: FollowedJob): job is { status: typeof FORGOTTEN_JOB; id: string | null } {
  return job.status === FORGOTTEN_JOB;
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });

/**
 * Poll one job until it settles.
 *
 * `onUpdate` is called with every reading, so a screen can show the queue
 * position and the running state without owning a timer of its own.
 */
export async function followIngestJob(
  jobId: string,
  options: { onUpdate?: (job: IngestJob) => void; signal?: AbortSignal } = {},
): Promise<FollowedJob> {
  let delay = FIRST_DELAY_MS;
  for (;;) {
    let job: IngestJob;
    try {
      job = await ingestJobs.get(jobId, options.signal);
    } catch (error) {
      if (error instanceof ApiError && error.type === 'not_found') {
        return { status: FORGOTTEN_JOB, id: jobId };
      }
      throw error;
    }
    options.onUpdate?.(job);
    if (!isActive(job)) return job;
    await wait(delay, options.signal);
    delay = Math.min(delay + DELAY_STEP_MS, MAX_DELAY_MS);
  }
}
