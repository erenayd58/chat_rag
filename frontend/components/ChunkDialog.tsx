'use client';

/**
 * One chunk, in full, with the facts the API recorded about it.
 *
 * A citation, a search hit and a browsed chunk are the same object in three
 * places, so they open the same dialog. The facts are whatever the caller
 * could name from the response -- nothing is computed into a confidence, and a
 * score is shown as the number the retriever produced.
 */

import { Modal } from './Modal';
import { DefRow } from './ui';

export interface ChunkFact {
  label: string;
  value: string;
}

export function ChunkDialog({
  title,
  facts,
  text,
  onClose,
}: {
  title: string;
  facts: ChunkFact[];
  text: string;
  onClose: () => void;
}) {
  return (
    <Modal
      title={title}
      onClose={onClose}
      wide
      footer={
        <button type="button" className="btn btn-secondary" onClick={onClose}>
          Kapat
        </button>
      }
    >
      <div style={{ marginBottom: 16 }}>
        {facts.map((fact) => (
          <DefRow key={fact.label} label={fact.label}>
            <span className="mono">{fact.value}</span>
          </DefRow>
        ))}
      </div>
      <div className="chunk-full">{text || '(boş)'}</div>
    </Modal>
  );
}
