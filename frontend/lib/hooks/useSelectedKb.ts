'use client';

/**
 * The knowledge base a person is working in, remembered between screens.
 *
 * Chat, Search and the document screens all need one, and asking for it again
 * on every navigation is the kind of small friction that makes a console feel
 * unfinished. It is a presentation preference and lives in `localStorage`;
 * nothing about the corpus depends on it.
 */

import { useCallback, useEffect, useState } from 'react';

const KEY = 'chat_rag.selected_kb';

function read(): string {
  if (typeof window === 'undefined') return '';
  try {
    return window.localStorage.getItem(KEY) ?? '';
  } catch {
    return '';
  }
}

export function useSelectedKb(): [string, (kbId: string) => void] {
  // Empty on the first render on purpose: the server render has no storage,
  // and reading it during render would make the two disagree.
  const [selected, setSelected] = useState('');

  useEffect(() => {
    setSelected(read());
  }, []);

  const select = useCallback((kbId: string) => {
    setSelected(kbId);
    try {
      window.localStorage.setItem(KEY, kbId);
    } catch {
      /* a browser with storage turned off still works, it just forgets */
    }
  }, []);

  return [selected, select];
}
