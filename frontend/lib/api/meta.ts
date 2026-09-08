/**
 * Discovery: what this deployment can offer.
 *
 * There is no second catalogue. The chunking methods are a projection of the
 * library's registry and are read from the API every time a picker is opened;
 * a method key does not appear anywhere in this front end's source.
 */

import { request } from './client';
import type {
  ChunkingMethod,
  Collection,
  Health,
  ModelChain,
  RetrievalMethodCollection,
} from '@/types/api';

/** Every chunking method, including the ones this machine cannot run. */
export async function chunkingMethods(signal?: AbortSignal): Promise<ChunkingMethod[]> {
  const found = await request<Collection<ChunkingMethod>>('/meta/chunking-methods', { signal });
  return found.items;
}

/** What one knowledge base's retriever can actually serve, and its default. */
export function retrievalMethods(
  knowledgeBaseId?: string | null,
  signal?: AbortSignal,
): Promise<RetrievalMethodCollection> {
  return request<RetrievalMethodCollection>('/meta/retrieval-methods', {
    query: { knowledge_base_id: knowledgeBaseId },
    signal,
  });
}

/** The configured model chain: names and endpoints, never a key. */
export function models(
  knowledgeBaseId?: string | null,
  signal?: AbortSignal,
): Promise<ModelChain> {
  return request<ModelChain>('/meta/models', {
    query: { knowledge_base_id: knowledgeBaseId },
    signal,
  });
}

export function health(signal?: AbortSignal): Promise<Health> {
  return request<Health>('/health', { signal });
}
