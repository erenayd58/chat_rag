/**
 * The shapes `/api/v1` answers with.
 *
 * Mirrors `docs/api-v1.md` and the response models it is generated from. Two
 * fields are deliberately open -- a chunk's `metadata` and an answer's
 * `diagnostics` -- because the contract publishes them as pass-through and
 * pinning them here would freeze internals the contract exists to leave free.
 */

export interface Page {
  offset: number;
  limit: number;
  total: number;
}

export interface Collection<T> {
  items: T[];
  page: Page;
}

/** The refusal body. `type` is what a client branches on. */
export interface ApiErrorBody {
  error: {
    type: ApiErrorType;
    message: string;
    details?: Record<string, unknown>;
  };
}

export type ApiErrorType =
  | 'invalid_request'
  | 'not_found'
  | 'not_ready'
  | 'conflict'
  | 'unavailable'
  | 'overloaded'
  | 'timeout'
  | 'internal'
  | 'network';

/* ------------------------------------------------------------------ */
/* Discovery                                                           */
/* ------------------------------------------------------------------ */

export interface ChunkingMethod {
  key: string;
  label: string;
  summary: string;
  engine: string;
  available: boolean;
  unavailable_reason: string | null;
  uses_model: boolean;
  default: boolean;
  /** An orchestration runs over a baseline partition rather than being one. */
  orchestration: boolean;
  baseline: string | null;
}

export interface RetrievalMethod {
  name: string | null;
  label: string | null;
  available: boolean;
  unavailable_reason: string | null;
}

export type RetrievalMethodCollection = Collection<RetrievalMethod> & {
  default: string | null;
};

export interface ModelChain {
  chain: Record<string, unknown>;
}

export interface Health {
  state: string;
  ready: boolean;
  reasons: unknown[];
  checked_at: string | null;
  capacity: {
    ingest: { running: number; queued: number; queue_capacity: number; workers: number };
    query: { active: number; max_active: number };
  };
}

/* ------------------------------------------------------------------ */
/* Knowledge bases                                                     */
/* ------------------------------------------------------------------ */

export interface KnowledgeBase {
  id: string | null;
  name: string | null;
  chunker: Record<string, unknown>;
  retrieval_method: string | null;
  embedding_model: string | null;
  extra: Record<string, unknown>;
}

export interface KnowledgeBaseCreate {
  name?: string;
  chunker?: Record<string, unknown>;
  embedding_model?: string;
  retrieval_method?: string;
  extra?: Record<string, unknown>;
}

export interface KnowledgeBaseUpdate {
  name?: string;
  extra?: Record<string, unknown>;
}

/** The store's own manifest report, passed through and not contractual. */
export interface EmbeddingIndex {
  state?: string;
  compatible?: boolean;
  dense_available?: boolean;
  reason?: string;
  stored?: { model?: string; dimension?: number; chunk_count?: number; fingerprint?: string };
  current?: { model?: string; dimension?: number; fingerprint?: string };
  [key: string]: unknown;
}

export interface EmbeddingReindex {
  result: Record<string, unknown>;
  index: Record<string, unknown>;
}

/* ------------------------------------------------------------------ */
/* Documents and their analysis                                        */
/* ------------------------------------------------------------------ */

export const ANALYSIS_STATES = ['missing', 'pending', 'running', 'ready', 'failed'] as const;
export type AnalysisState = (typeof ANALYSIS_STATES)[number];

export interface ContentAnalysis {
  requested_methods: string[];
  ready_methods: string[];
  shared_with_document_ids: string[];
}

export interface Analysis {
  status: AnalysisState | string;
  content_id: string | null;
  /** What *this upload* asked for. */
  selected_methods: string[];
  /** selected and built, both -- what this upload may be asked about. */
  ready_methods: string[];
  failed_methods: string[];
  unit_count: number | null;
  deep_source: string | null;
  error: string | null;
  updated_at: string | null;
  /** The shared analysis of these bytes, across every upload of them. */
  content: ContentAnalysis;
}

export interface DocumentRecord {
  id: string | null;
  knowledge_base_id: string | null;
  name: string;
  /** The bytes, not the upload. */
  content_id: string | null;
  size_bytes: number;
  chunk_count: number;
  chunking_mode: string | null;
  status: string;
  ingested_at: string | null;
  ingest_job_id: string | null;
}

export type DocumentWithAnalysis = DocumentRecord & { analysis: Analysis };

export interface Chunk {
  id: string | null;
  document_id: string | null;
  content: string;
  chunk_index: number | null;
  total_chunks: number | null;
  section: string | null;
  chunking_mode: string | null;
  metadata: Record<string, unknown>;
}

export type ScoredChunk = Chunk & {
  score: number | null;
  retrieval_method: string | null;
};

export type SearchResults = Collection<ScoredChunk> & {
  method: string | null;
  knowledge_base_id: string | null;
};

export interface CanonicalUnit {
  id: string | null;
  order: number | null;
  type: string | null;
  text: string;
  heading_level: number | null;
  section_path: unknown[];
  source: Record<string, unknown>;
}

export type CanonicalUnitCollection = Collection<CanonicalUnit> & { pages: unknown[] };

/** One chunking method's rows. The rows are the chunker's own shape. */
export type AnalysisChunks = Collection<Record<string, unknown>> & {
  method: string;
  engine: string | null;
  content_id: string | null;
};

/* ------------------------------------------------------------------ */
/* Ingest jobs                                                         */
/* ------------------------------------------------------------------ */

export const ACTIVE_JOB_STATES = ['queued', 'running'] as const;

export type IngestJobStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'timed_out'
  | 'cancelled'
  | 'interrupted';

export interface IngestJob {
  id: string | null;
  status: IngestJobStatus | string | null;
  knowledge_base_id: string | null;
  document_id: string | null;
  content_id: string | null;
  name: string | null;
  methods: string[];
  queue_position: number | null;
  /** Uploads of the same bytes attached to this job rather than parsed twice. */
  attached_uploads: number;
  submitted_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  run_seconds: number | null;
  restart_settled: boolean;
  error: { type: string; message: string | null } | null;
  result: {
    document_id: string | null;
    chunk_count: number | null;
    chunking_mode: string | null;
    deep_analysis: unknown;
    analysis_status: unknown;
  } | null;
}

export interface IngestCapacity {
  running: number;
  queued: number;
  queue_capacity: number;
  workers: number;
}

export type IngestJobCollection = Collection<IngestJob> & { capacity: IngestCapacity };

/* ------------------------------------------------------------------ */
/* Asking                                                              */
/* ------------------------------------------------------------------ */

export interface QueryRequest {
  question: string;
  knowledge_base_id?: string | null;
  top_k?: number;
  temperature?: number;
  max_tokens?: number;
}

export interface SearchRequest {
  query: string;
  knowledge_base_id?: string | null;
  method?: string;
  limit?: number;
}

export interface Citation {
  label: string | null;
  chunk_id: string | null;
  document_id: string | null;
  document: string | null;
  section: string | null;
  pages: unknown[];
  chunking_mode: string | null;
  /** Whether the answer actually cited this source. */
  used: boolean;
  score: number | null;
  content: string;
}

export interface Answer {
  answer: string;
  citations: Citation[];
  knowledge_base_id: string | null;
  retrieval_method: string | null;
  /** False when the model answered without citing any of its sources. */
  grounded: boolean;
  timing: { query_id: string | null; total_seconds: number | null; stages: Record<string, unknown> };
  diagnostics: Record<string, unknown>;
}
