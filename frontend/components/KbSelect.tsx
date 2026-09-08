'use client';

/** The knowledge-base picker Chat, Search and Analysis all share. */

import { useEffect } from 'react';
import api from '@/lib/api';
import { useAsync } from '@/lib/hooks/useAsync';
import type { KnowledgeBase } from '@/types/api';

export function KbSelect({
  value,
  onChange,
  id = 'kbSelect',
  placeholder = 'Bilgi tabanı seçin',
  autoSelectSingle = true,
}: {
  value: string;
  onChange: (kbId: string) => void;
  id?: string;
  placeholder?: string;
  /** With exactly one knowledge base there is nothing to choose. */
  autoSelectSingle?: boolean;
}) {
  const state = useAsync<KnowledgeBase[]>((signal) => api.knowledgeBases.list(signal), []);
  const items = state.data;

  useEffect(() => {
    if (!items) return;
    const known = items.some((kb) => kb.id === value);
    if (value && !known) {
      // The remembered knowledge base is gone; do not keep pointing at it.
      onChange('');
      return;
    }
    if (!value && autoSelectSingle && items.length === 1 && items[0].id) onChange(items[0].id);
  }, [items, value, onChange, autoSelectSingle]);

  if (state.error && !items) {
    return (
      <select className="select" id={id} disabled>
        <option>Bilgi tabanları okunamadı</option>
      </select>
    );
  }

  return (
    <select
      className="select"
      id={id}
      value={value}
      disabled={state.loading && !items}
      onChange={(event) => onChange(event.target.value)}
    >
      <option value="">{state.loading && !items ? 'Yükleniyor…' : placeholder}</option>
      {(items ?? []).map((kb) => (
        <option key={kb.id ?? ''} value={kb.id ?? ''}>
          {kb.name || kb.id}
        </option>
      ))}
    </select>
  );
}
