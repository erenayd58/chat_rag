/**
 * The two calls only the Viewer makes.
 *
 * Everything else the Viewer needs is an ordinary console call — knowledge
 * bases, documents and their analysis state are `lib/api/knowledgeBases.ts`
 * and `lib/api/documents.ts`, and the method catalogue is `lib/api/meta.ts`.
 * What is here is what no other screen asks for: a document's prepared render
 * model, and a question put to several chunking methods at once.
 */

import { request } from './client';
import type { AnalysisPayload, AnalysisQueryRequest, AnalysisQueryResult } from '@/types/api';

const DOCUMENTS = '/documents';

/**
 * One document's whole analysis: the canonical units, and per ready method
 * the chunks *and the unit offsets they cut at*.
 *
 * Those offsets are why this is not the per-method rows. They are what lets
 * several methods be drawn down one column of text and compared on the page
 * rather than by chunk number, and a client cannot compute them.
 *
 * **409 `not_ready`** while nothing this upload selected has been built. That
 * is a state to poll, not a failure — `lib/errors.ts` says as much and the
 * screen shows the progress rather than an error.
 */
export function payload(documentId: string, signal?: AbortSignal): Promise<AnalysisPayload> {
  return request<AnalysisPayload>(
    `${DOCUMENTS}/${encodeURIComponent(documentId)}/analysis/payload`,
    { signal },
  );
}

/**
 * One question, one document, through each named chunking method.
 *
 * The only call on this contract that compares chunkers: a knowledge base has
 * one, so `POST /queries` cannot. `methods` absent means every method this
 * upload has ready. `answer: false` stops after retrieval.
 *
 * An arm that could not be answered comes back with its own `status` and its
 * sources rather than failing the request — in a comparison the other arms
 * are still the answer.
 */
export function query(
  input: AnalysisQueryRequest,
  signal?: AbortSignal,
): Promise<AnalysisQueryResult> {
  return request<AnalysisQueryResult>('/analysis-queries', {
    method: 'POST',
    json: input,
    signal,
  });
}
