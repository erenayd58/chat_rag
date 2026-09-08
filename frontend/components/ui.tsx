'use client';

/**
 * The primitives every screen is built from.
 *
 * They exist so that the three states a remote call has -- loading, refused,
 * empty -- look the same on every screen and are impossible to skip: a table
 * that silently renders nothing when its request failed is the failure mode
 * this file is here to prevent.
 */

import type { ReactNode } from 'react';
import { explain, errorMessage } from '@/lib/errors';

/* ------------------------------------------------------------------ */
/* States                                                              */
/* ------------------------------------------------------------------ */

export function Loading({ label = 'Yükleniyor…' }: { label?: string }) {
  return (
    <div className="loading-state" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </div>
  );
}

export function Empty({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}
      {action}
    </div>
  );
}

/**
 * A refusal, shown the way the taxonomy describes it.
 *
 * The title and the advice come from `lib/errors`, keyed by the contract's
 * `type`; the server's own message is shown underneath as detail, so a
 * reworded message never changes what the screen tells someone to do.
 */
export function Failed({
  error,
  onRetry,
  title,
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const explanation = explain(error);
  const detail = errorMessage(error);
  return (
    <div className="error-state" role="alert">
      <h3>{title ?? explanation.title}</h3>
      <p>{explanation.detail}</p>
      {detail && detail !== explanation.title ? <div className="error-detail">{detail}</div> : null}
      {onRetry && explanation.retryable ? (
        <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
          Tekrar dene
        </button>
      ) : null}
    </div>
  );
}

/** Loading, then refused, then the content. In that order, every time. */
export function Async<T>({
  state,
  children,
  empty,
  loadingLabel,
}: {
  state: { data: T | undefined; error: unknown; loading: boolean; reload: () => void };
  children: (data: T) => ReactNode;
  empty?: ReactNode;
  loadingLabel?: string;
}) {
  if (state.loading && state.data === undefined) return <Loading label={loadingLabel} />;
  if (state.error && state.data === undefined) return <Failed error={state.error} onRetry={state.reload} />;
  if (state.data === undefined) return empty ? <>{empty}</> : null;
  return <>{children(state.data)}</>;
}

/* ------------------------------------------------------------------ */
/* Small pieces                                                        */
/* ------------------------------------------------------------------ */

export type Tone = 'neutral' | 'success' | 'warn' | 'danger' | 'accent';

export function Badge({
  tone = 'neutral',
  dot = false,
  title,
  children,
}: {
  tone?: Tone;
  dot?: boolean;
  title?: string;
  children: ReactNode;
}) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {dot ? <span className="dot" aria-hidden="true" /> : null}
      {children}
    </span>
  );
}

export function DefRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="def-row">
      <span className="def-key">{label}</span>
      <span className="def-val">{children}</span>
    </div>
  );
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: T; label: string; count?: number | null }[];
  active: T;
  onChange: (key: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          role="tab"
          aria-selected={tab.key === active}
          className={`tab${tab.key === active ? ' active' : ''}`}
          onClick={() => onChange(tab.key)}
        >
          {tab.label}
          {tab.count === null || tab.count === undefined ? null : (
            <span className="tab-count">{tab.count}</span>
          )}
        </button>
      ))}
    </div>
  );
}

/**
 * Offset paging over a collection.
 *
 * Says how many there are because the contract always answers `page.total`,
 * and a browser that cannot tell 50 rows from 5000 is guessing.
 */
export function Pager({
  offset,
  limit,
  total,
  onOffset,
  busy,
}: {
  offset: number;
  limit: number;
  total: number;
  onOffset: (offset: number) => void;
  busy?: boolean;
}) {
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);
  return (
    <div className="pager">
      <span>
        {from}–{to} / {total}
      </span>
      <span className="pager-buttons">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={busy || offset <= 0}
          onClick={() => onOffset(Math.max(0, offset - limit))}
        >
          Önceki
        </button>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={busy || offset + limit >= total}
          onClick={() => onOffset(offset + limit)}
        >
          Sonraki
        </button>
      </span>
    </div>
  );
}

export function InlineError({ children }: { children: ReactNode }) {
  return (
    <div className="inline-error" role="alert">
      {children}
    </div>
  );
}
