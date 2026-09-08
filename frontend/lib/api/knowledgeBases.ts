/** Knowledge bases: the collection a document is ingested into. */

import { collect, request } from './client';
import type {
  Chunk,
  Collection,
  EmbeddingIndex,
  EmbeddingReindex,
  KnowledgeBase,
  KnowledgeBaseCreate,
  KnowledgeBaseUpdate,
} from '@/types/api';

const BASE = '/knowledge-bases';

export function list(signal?: AbortSignal): Promise<KnowledgeBase[]> {
  return collect<KnowledgeBase>(BASE, { signal });
}

export function get(kbId: string, signal?: AbortSignal): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`${BASE}/${encodeURIComponent(kbId)}`, { signal });
}

/** Create one. The chunker, embedding model and storage are fixed here. */
export function create(payload: KnowledgeBaseCreate): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(BASE, { method: 'POST', json: payload });
}

/** `name` and `extra` only: everything else was decided at creation. */
export function update(kbId: string, changes: KnowledgeBaseUpdate): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`${BASE}/${encodeURIComponent(kbId)}`, {
    method: 'PATCH',
    json: changes,
  });
}

/** Takes the corpus with it; the documents' ledger rows survive. */
export function remove(kbId: string): Promise<void> {
  return request<void>(`${BASE}/${encodeURIComponent(kbId)}`, { method: 'DELETE' });
}

export function embeddingIndex(kbId: string, signal?: AbortSignal): Promise<EmbeddingIndex> {
  return request<EmbeddingIndex>(`${BASE}/${encodeURIComponent(kbId)}/embedding-index`, { signal });
}

export function rebuildEmbeddingIndex(kbId: string): Promise<EmbeddingReindex> {
  return request<EmbeddingReindex>(`${BASE}/${encodeURIComponent(kbId)}/embedding-index/rebuild`, {
    method: 'POST',
    json: {},
  });
}

/** Browse the corpus a page at a time; `search` filters by phrase. */
export function chunks(
  kbId: string,
  options: { offset?: number; limit?: number; search?: string; signal?: AbortSignal } = {},
): Promise<Collection<Chunk>> {
  return request<Collection<Chunk>>(`${BASE}/${encodeURIComponent(kbId)}/chunks`, {
    query: { offset: options.offset, limit: options.limit, search: options.search },
    signal: options.signal,
  });
}
