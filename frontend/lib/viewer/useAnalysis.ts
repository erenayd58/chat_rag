'use client';

/**
 * One document's analysis, followed until it is worth showing.
 *
 * An analysis is built on a worker and a Deep Analysis run can take minutes,
 * so "open a document" is not one request — it is a state to watch. This hook
 * is the whole of that watching, in one place, because getting it wrong shows
 * up as either a spinner that never resolves or a page that quietly keeps the
 * first two methods after the third finished.
 *
 * Two requests, deliberately not one:
 *
 * * `GET .../analysis` is small, always **200**, and says where the build got
 *   to. It is what is polled while anything is pending or running.
 * * `GET .../analysis/payload` carries a whole document's text and every
 *   chunk's provenance. It is fetched only when the set of ready methods
 *   *changes* — first time, and again when a method finishes — so a document
 *   sitting open costs one small poll every few seconds and nothing else.
 *
 * Polling stops when the build settles. A `ready` analysis that later gains a
 * method does so through an explicit action on this screen, and that action
 * calls `reload`.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import api from '@/lib/api';
import { ApiError } from '@/lib/api/client';
import type { Analysis, AnalysisPayload } from '@/types/api';
import type { ViewerDoc } from './model';

/** How often to ask a building analysis where it got to. */
const POLL_MS = 3000;

export interface AnalysisView {
  /** missing | pending | running | ready | failed — the packager's own word. */
  status: string;
  state: Analysis | null;
  doc: ViewerDoc | null;
  /** What this upload may be asked about: selected and built, both. */
  readyMethods: string[];
  /** True while nothing has arrived yet. */
  loading: boolean;
  /** True while the packager is still working. */
  building: boolean;
  error: unknown;
  reload: () => void;
}

const IDLE: AnalysisView = {
  status: 'missing',
  state: null,
  doc: null,
  readyMethods: [],
  loading: false,
  building: false,
  error: null,
  reload: () => {},
};

export function useAnalysis(documentId: string): AnalysisView {
  const [state, setState] = useState<Analysis | null>(null);
  const [payload, setPayload] = useState<AnalysisPayload | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  // The ready-method set the payload on screen was built from. A payload is
  // re-fetched when this changes and never otherwise.
  const shown = useRef<string>('');

  const reload = useCallback(() => {
    shown.current = '';
    setNonce((value) => value + 1);
  }, []);

  useEffect(() => {
    shown.current = '';
    setState(null);
    setPayload(null);
    setError(null);
  }, [documentId]);

  useEffect(() => {
    if (!documentId) return;
    const controller = new AbortController();
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const read = async () => {
      try {
        const found = await api.documents.analysis(documentId, controller.signal);
        if (!live) return;
        setState(found);
        setError(null);

        const signature = found.ready_methods.join('|');
        if (found.ready_methods.length && signature !== shown.current) {
          const built = await api.analysis.payload(documentId, controller.signal);
          if (!live) return;
          shown.current = signature;
          setPayload(built);
        }
        if (found.status === 'pending' || found.status === 'running') {
          timer = setTimeout(read, POLL_MS);
        }
      } catch (cause) {
        if (!live || controller.signal.aborted) return;
        // A payload that is not ready yet is a state, not a failure: the
        // analysis record already said so and this screen shows the progress.
        if (cause instanceof ApiError && cause.type === 'not_ready') {
          timer = setTimeout(read, POLL_MS);
          return;
        }
        setError(cause);
      } finally {
        if (live) setLoading(false);
      }
    };

    setLoading(true);
    read();
    return () => {
      live = false;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [documentId, nonce]);

  if (!documentId) return IDLE;

  const status = state?.status ?? 'missing';
  return {
    status,
    state,
    doc: (payload?.payload as unknown as ViewerDoc) ?? null,
    readyMethods: payload?.ready_methods ?? state?.ready_methods ?? [],
    loading: loading && !state,
    building: status === 'pending' || status === 'running',
    error,
    reload,
  };
}
