/**
 * Asking: an answer with citations, or the ranked chunks on their own.
 *
 * Both are POSTs and neither creates anything -- a question is user text that
 * should not land in an access log or be cached in between.
 */

import { request } from './client';
import type { Answer, QueryRequest, SearchRequest, SearchResults } from '@/types/api';

/** Retrieve, then answer. The answer carries its citations and `grounded`. */
export function ask(payload: QueryRequest, signal?: AbortSignal): Promise<Answer> {
  return request<Answer>('/queries', { method: 'POST', json: payload, signal });
}

/**
 * Retrieval without an answer.
 *
 * `method` names a retrieval method; one the configured retriever cannot serve
 * is a 400 with the reason, which is why the picker is filled from
 * `/api/v1/meta/retrieval-methods` rather than from a list written here.
 */
export function search(payload: SearchRequest, signal?: AbortSignal): Promise<SearchResults> {
  return request<SearchResults>('/searches', { method: 'POST', json: payload, signal });
}
