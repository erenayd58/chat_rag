/**
 * The one place this front end calls `fetch`.
 *
 * Every screen goes through `lib/api/*`, and every one of those modules goes
 * through this file. That is what makes the contract's refusal taxonomy a
 * single translation rather than a habit repeated in a dozen components: a
 * refusal arrives as `{ error: { type, message, details } }` and leaves here
 * as an `ApiError` carrying `type`. **Nothing outside this module reads a
 * status code**, because `type` is what the contract says a client branches
 * on and it does not change when a message is reworded.
 *
 * See `docs/api-v1.md`, *Refusals*.
 */

import type { ApiErrorBody, ApiErrorType, Collection } from '@/types/api';

/** The version prefix, in one place. */
export const API_PREFIX = '/api/v1';

/** The contract's ceiling on a page. */
export const MAX_PAGE_SIZE = 200;

/**
 * A refusal, or a transport failure dressed as one.
 *
 * `type` is the contract's name for it; `status` is HTTP's opinion and is kept
 * only for a diagnostic line. A connection that never reached the server is
 * `type: 'network'` with `status: 0`, so a caller can tell "the server said
 * no" from "the server was not there".
 */
export class ApiError extends Error {
  readonly type: ApiErrorType;
  readonly status: number;
  readonly details: Record<string, unknown>;
  /** Seconds the server asked us to wait, from `Retry-After` on a 503. */
  readonly retryAfter: number | null;

  constructor(
    type: ApiErrorType,
    message: string,
    options: { status?: number; details?: Record<string, unknown>; retryAfter?: number | null } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.type = type;
    this.status = options.status ?? 0;
    this.details = options.details ?? {};
    this.retryAfter = options.retryAfter ?? null;
  }

  /** True when waiting and trying again is the right answer. */
  get isTransient(): boolean {
    return this.type === 'overloaded' || this.type === 'timeout';
  }
}

export type QueryValue = string | number | boolean | null | undefined;

/** `?a=1&b=2`, with nothing written for a value nobody set. */
export function queryString(params: Record<string, QueryValue> = {}): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  /** A JSON body. Mutually exclusive with `form`. */
  json?: unknown;
  /** A multipart body; the browser sets the content type, so we must not. */
  form?: FormData;
  query?: Record<string, QueryValue>;
  signal?: AbortSignal;
}

function retryAfterOf(response: Response): number | null {
  const raw = response.headers.get('Retry-After');
  if (!raw) return null;
  const seconds = Number(raw);
  return Number.isFinite(seconds) ? seconds : null;
}

function refusal(response: Response, body: unknown): ApiError {
  const envelope = body as ApiErrorBody | null;
  const error = envelope && typeof envelope === 'object' ? envelope.error : undefined;
  if (error && typeof error.type === 'string') {
    return new ApiError(error.type, error.message || 'İstek reddedildi.', {
      status: response.status,
      details: error.details,
      retryAfter: retryAfterOf(response),
    });
  }
  // A body this surface did not write: a proxy, a gateway, a crash. Named by
  // the only thing that is known about it.
  return new ApiError('internal', `İstek başarısız oldu (HTTP ${response.status}).`, {
    status: response.status,
    retryAfter: retryAfterOf(response),
  });
}

/**
 * One request against `/api/v1`.
 *
 * Resolves with the parsed body, or with `undefined` for the 204 the contract
 * answers an action with nothing to return. Rejects with `ApiError` and
 * nothing else.
 */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const init: RequestInit = {
    method: options.method ?? 'GET',
    headers: {},
    signal: options.signal,
  };
  if (options.json !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(options.json);
  }
  if (options.form !== undefined) {
    init.body = options.form;
  }

  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}${queryString(options.query)}`, init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError('network', 'Sunucuya ulaşılamadı. Bağlantınızı kontrol edin.', {
      details: { cause: String(cause) },
    });
  }

  if (response.status === 204) return undefined as T;

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) throw refusal(response, body);
  return body as T;
}

/**
 * Every item of a collection, a page at a time.
 *
 * The contract caps a page at 200 and always answers `page.total`, so a screen
 * that needs the whole set (a knowledge base's documents, to total them) walks
 * it here rather than each screen inventing its own loop.
 */
export async function collect<T>(
  path: string,
  options: { query?: Record<string, QueryValue>; signal?: AbortSignal } = {},
): Promise<T[]> {
  const items: T[] = [];
  let offset = 0;
  for (;;) {
    const page = await request<Collection<T>>(path, {
      query: { ...options.query, offset, limit: MAX_PAGE_SIZE },
      signal: options.signal,
    });
    items.push(...page.items);
    offset += page.items.length;
    if (!page.items.length || offset >= page.page.total) return items;
  }
}
