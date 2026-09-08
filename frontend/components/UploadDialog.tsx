'use client';

/**
 * Upload, and then wait properly.
 *
 * `POST /api/v1/documents` answers **202** with an ingest job -- never the
 * document. So this dialog does the three steps the contract describes and
 * shows each of them: it submits, it follows the job (queue position, then
 * running), and only when the job settles does it say what happened. Closing
 * the dialog stops the polling; the job carries on and the strip above the
 * document list keeps showing it, because a browser tab is not what an
 * ingestion depends on.
 *
 * Two answers this screen has to tell apart:
 *   `attached_uploads` -- the same bytes were already in flight, so this
 *   upload joined that job instead of parsing them twice;
 *   `overloaded` (503) -- the queue is full, nothing was queued and nothing
 *   was kept. The file stays chosen so one click resends it.
 */

import { useEffect, useRef, useState } from 'react';
import api, { ApiError } from '@/lib/api';
import { explain } from '@/lib/errors';
import { followIngestJob, isForgotten } from '@/lib/jobs';
import { failureNote, progressNote } from '@/lib/jobText';
import { MethodPicker } from './MethodPicker';
import { Modal } from './Modal';
import { InlineError } from './ui';
import type { IngestJob } from '@/types/api';

type Phase = 'choosing' | 'submitting' | 'following';

export function UploadDialog({
  knowledgeBaseId,
  onClose,
  onSettled,
}: {
  knowledgeBaseId: string;
  onClose: () => void;
  /** Called whenever the job list may have changed, and when one settles. */
  onSettled: (job: IngestJob | null) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [methods, setMethods] = useState<string[]>([]);
  const [phase, setPhase] = useState<Phase>('choosing');
  const [note, setNote] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => abort.current?.abort(), []);

  const busy = phase !== 'choosing';

  const submit = async () => {
    if (!file || !methods.length) return;
    setError(null);
    setPhase('submitting');
    setNote('Doküman gönderiliyor…');
    const controller = new AbortController();
    abort.current = controller;

    let accepted: IngestJob;
    try {
      accepted = await api.documents.upload({ file, knowledgeBaseId, methods });
    } catch (cause) {
      setPhase('choosing');
      const explanation = explain(cause);
      const detail = cause instanceof ApiError ? cause.message : '';
      setError(`${explanation.title}. ${explanation.detail}${detail ? ` (${detail})` : ''}`);
      return;
    }

    onSettled(accepted);
    if (accepted.attached_uploads > 0) {
      setNote('Aynı doküman zaten işleniyor; bu yükleme ona bağlandı.');
    }
    setPhase('following');

    try {
      const settled = await followIngestJob(accepted.id ?? '', {
        signal: controller.signal,
        onUpdate: (job) => setNote(progressNote(job)),
      });
      if (!isForgotten(settled) && settled.status === 'succeeded') {
        onSettled(settled);
        onClose();
        return;
      }
      setPhase('choosing');
      setError(failureNote(settled));
      onSettled(isForgotten(settled) ? null : settled);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return;
      setPhase('choosing');
      const explanation = explain(cause);
      setError(`${explanation.title}. ${explanation.detail}`);
    }
  };

  return (
    <Modal
      title="Doküman yükle"
      onClose={() => {
        abort.current?.abort();
        onClose();
      }}
      footer={
        <>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              abort.current?.abort();
              onClose();
            }}
          >
            {busy ? 'Arka planda devam et' : 'Vazgeç'}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !file || !methods.length}
            onClick={submit}
          >
            Yükle ve analiz et
          </button>
        </>
      }
    >
      <div className="field">
        <span className="field-label">Dosya</span>
        <input
          ref={fileInput}
          type="file"
          accept=".pdf,.txt,.md,.docx"
          style={{ display: 'none' }}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          className="btn btn-secondary btn-block"
          style={{ justifyContent: 'flex-start' }}
          disabled={busy}
          onClick={() => fileInput.current?.click()}
        >
          {file ? file.name : 'Dosya seç…'}
        </button>
        <div className="field-hint">PDF, TXT, MD, DOCX</div>
      </div>

      <div className="field">
        <span className="field-label">Chunking methods</span>
        <MethodPicker selected={methods} onChange={setMethods} />
      </div>

      {busy ? (
        <div className="field">
          <div className="progress-indeterminate" />
          <div className="field-hint">{note}</div>
        </div>
      ) : null}

      {error ? <InlineError>{error}</InlineError> : null}
    </Modal>
  );
}
