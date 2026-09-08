/**
 * Documents, and the analysis that belongs to their bytes.
 *
 * Uploading is **always asynchronous**: `POST /api/v1/documents` answers 202
 * with an ingest job and nothing else, so `upload` returns a job and the
 * caller follows it (`lib/jobs.ts`). There is no synchronous mode on this
 * contract and this front end does not pretend there is one.
 */

import { collect, request } from './client';
import type {
  Analysis,
  AnalysisChunks,
  CanonicalUnitCollection,
  Chunk,
  Collection,
  DocumentWithAnalysis,
  IngestJob,
} from '@/types/api';

const BASE = '/documents';

export function list(
  knowledgeBaseId?: string | null,
  signal?: AbortSignal,
): Promise<DocumentWithAnalysis[]> {
  return collect<DocumentWithAnalysis>(BASE, {
    query: { knowledge_base_id: knowledgeBaseId },
    signal,
  });
}

export function get(documentId: string, signal?: AbortSignal): Promise<DocumentWithAnalysis> {
  return request<DocumentWithAnalysis>(`${BASE}/${encodeURIComponent(documentId)}`, { signal });
}

/**
 * Submit one file. Answers the accepted job, never the document.
 *
 * `methods` is one repeated form field per method, which is what the contract
 * documents; an unknown or unavailable key is dropped by the server and an
 * empty selection falls back to the registry's default.
 */
export function upload(input: {
  file: File;
  knowledgeBaseId: string;
  methods: string[];
}): Promise<IngestJob> {
  const form = new FormData();
  form.append('file', input.file);
  form.append('knowledge_base_id', input.knowledgeBaseId);
  for (const method of input.methods) form.append('methods', method);
  return request<IngestJob>(BASE, { method: 'POST', form });
}

/** Takes its chunks, its ledger row and its own analysis; not the content. */
export function remove(documentId: string): Promise<void> {
  return request<void>(`${BASE}/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
}

/** What this document was *indexed* as, by the knowledge base's own chunker. */
export function chunks(
  documentId: string,
  options: { offset?: number; limit?: number; knowledgeBaseId?: string | null; signal?: AbortSignal } = {},
): Promise<Collection<Chunk>> {
  return request<Collection<Chunk>>(`${BASE}/${encodeURIComponent(documentId)}/chunks`, {
    query: {
      offset: options.offset,
      limit: options.limit,
      knowledge_base_id: options.knowledgeBaseId,
    },
    signal: options.signal,
  });
}

/** The parser's canonical reading, before any chunker touched it. */
export function units(
  documentId: string,
  options: {
    offset?: number;
    limit?: number;
    knowledgeBaseId?: string | null;
    pageFrom?: number | string;
    pageTo?: number | string;
    unitType?: string;
    signal?: AbortSignal;
  } = {},
): Promise<CanonicalUnitCollection> {
  return request<CanonicalUnitCollection>(`${BASE}/${encodeURIComponent(documentId)}/units`, {
    query: {
      offset: options.offset,
      limit: options.limit,
      knowledge_base_id: options.knowledgeBaseId,
      page_from: options.pageFrom,
      page_to: options.pageTo,
      unit_type: options.unitType,
    },
    signal: options.signal,
  });
}

/* ------------------------------------------------------------------ */
/* The analysis                                                        */
/* ------------------------------------------------------------------ */

/** Where it got to. Always 200, even when there is nothing yet. */
export function analysis(documentId: string, signal?: AbortSignal): Promise<Analysis> {
  return request<Analysis>(`${BASE}/${encodeURIComponent(documentId)}/analysis`, { signal });
}

/** Queue it, or retry a failed one. */
export function requestAnalysis(documentId: string): Promise<Analysis> {
  return request<Analysis>(`${BASE}/${encodeURIComponent(documentId)}/analysis`, {
    method: 'POST',
    json: {},
  });
}

/** Add chunking variants to a document that is already here. */
export function addAnalysisMethods(documentId: string, methods: string[]): Promise<Analysis> {
  return request<Analysis>(`${BASE}/${encodeURIComponent(documentId)}/analysis/methods`, {
    method: 'POST',
    json: { methods },
  });
}

/**
 * One method's rows.
 *
 * Three refusals mean three different things here and the screen shows them
 * apart: 400 the deployment does not know the method, 404 this upload did not
 * select it (another upload of the same bytes may well have), 409 `not_ready`
 * it is selected and still being built.
 */
export function analysisMethodChunks(
  documentId: string,
  method: string,
  options: { offset?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<AnalysisChunks> {
  const path = `${BASE}/${encodeURIComponent(documentId)}/analysis/methods/${encodeURIComponent(method)}/chunks`;
  return request<AnalysisChunks>(path, {
    query: { offset: options.offset, limit: options.limit },
    signal: options.signal,
  });
}
