'use client';

/**
 * One loading state, done once.
 *
 * Every screen here has the same three states -- loading, an error a person
 * can act on, and the data -- and getting one of them wrong is how a console
 * ends up showing an empty table that is really a failed request. This hook
 * is what makes those three the same three everywhere, and it aborts the
 * request it started when the component goes away.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface AsyncState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  /** Run it again, keeping whatever is on screen until the answer arrives. */
  reload: () => void;
  /** Replace the data locally, for an edit whose answer is the new record. */
  set: (value: T) => void;
}

export function useAsync<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    setLoading(true);
    loadRef
      .current(controller.signal)
      .then((value) => {
        if (!live) return;
        setData(value);
        setError(null);
      })
      .catch((cause) => {
        if (!live || controller.signal.aborted) return;
        setError(cause);
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  const set = useCallback((value: T) => setData(value), []);

  return { data, error, loading, reload, set };
}
