'use client';

/**
 * A document's chunking analysis: where it got to, and what it produced.
 *
 * The contract keeps two levels apart and so does this panel, because merging
 * them lies in one direction or the other:
 *
 *   - **this upload** -- `selected_methods` is what it asked for and
 *     `ready_methods` is `selected ∩ built`, which is the only set it may be
 *     asked about;
 *   - **the content** -- the shared analysis of the same bytes, which may hold
 *     variants another upload asked for. Those are visible as a fact and are
 *     deliberately not offered as something to inspect here.
 *
 * While the analysis is `pending` or `running` the state is re-read on a
 * timer, so a queued document finishes on screen without a refresh.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import api, { ApiError } from '@/lib/api';
import { explain } from '@/lib/errors';
import { formatDate, formatNumber, humanise, truncate } from '@/lib/format';
import {
  ROW_ID_KEYS,
  ROW_SECTION_KEYS,
  ROW_SIZE_KEYS,
  ROW_TEXT_KEYS,
  pickNumber,
  pickString,
  rawRow,
} from '@/lib/rows';
import { useAsync } from '@/lib/hooks/useAsync';
import { MethodPicker, useChunkingMethods } from './MethodPicker';
import { Modal } from './Modal';
import { useToast } from './Toast';
import { Async, Badge, DefRow, Empty, Failed, InlineError, Loading, Pager, type Tone } from './ui';
import type { Analysis, AnalysisChunks } from '@/types/api';

const POLL_MS = 3000;
const PAGE_SIZE = 25;

const STATE_TONE: Record<string, Tone> = {
  ready: 'success',
  running: 'warn',
  pending: 'neutral',
  failed: 'danger',
  missing: 'neutral',
};

const STATE_LABEL: Record<string, string> = {
  ready: 'Hazır',
  running: 'Çalışıyor',
  pending: 'Sırada',
  failed: 'Başarısız',
  missing: 'Analiz yok',
};

export function AnalysisStateBadge({ status }: { status: string }) {
  return (
    <Badge tone={STATE_TONE[status] ?? 'neutral'} dot>
      {STATE_LABEL[status] ?? humanise(status)}
    </Badge>
  );
}

export function AnalysisPanel({ documentId }: { documentId: string }) {
  const toast = useToast();
  const state = useAsync<Analysis>((signal) => api.documents.analysis(documentId, signal), [documentId]);
  const analysis = state.data;
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const [inspect, setInspect] = useState<string | null>(null);

  const inFlight = analysis?.status === 'pending' || analysis?.status === 'running';
  const { reload } = state;

  useEffect(() => {
    if (!inFlight) return;
    const timer = setInterval(reload, POLL_MS);
    return () => clearInterval(timer);
  }, [inFlight, reload]);

  const run = useCallback(async () => {
    setBusy(true);
    try {
      const next = await api.documents.requestAnalysis(documentId);
      state.set(next);
      toast.success('Analiz sıraya alındı.');
    } catch (error) {
      toast.error(`${explain(error).title}. ${explain(error).detail}`);
    } finally {
      setBusy(false);
    }
  }, [documentId, state, toast]);

  return (
    <Async state={state}>
      {(data) => (
        <>
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Analiz durumu</div>
                <div className="card-sub">
                  Bir yükleme kendi seçtiği yöntemleri sorabilir; içerik daha fazlasını taşıyor olabilir.
                </div>
              </div>
              <div className="row">
                <AnalysisStateBadge status={data.status} />
                <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={run}>
                  {data.status === 'failed' ? 'Yeniden dene' : 'Analizi çalıştır'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={busy}
                  onClick={() => setAdding(true)}
                >
                  Yöntem ekle
                </button>
              </div>
            </div>
            <div className="card-pad">
              {data.error ? <InlineError>{data.error}</InlineError> : null}

              <DefRow label="Bu yüklemenin seçtiği yöntemler">
                <MethodChips
                  keys={data.selected_methods}
                  ready={data.ready_methods}
                  failed={data.failed_methods}
                />
              </DefRow>
              <DefRow label="Sorulabilir (seçilen ∩ hazır)">
                {data.ready_methods.length ? (
                  <span className="badge-list" style={{ justifyContent: 'flex-end' }}>
                    {data.ready_methods.map((method) => (
                      <button
                        key={method}
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => setInspect(method)}
                      >
                        <MethodLabel methodKey={method} /> ↗
                      </button>
                    ))}
                  </span>
                ) : (
                  <span className="dim">henüz yok</span>
                )}
              </DefRow>
              <DefRow label="Birim sayısı">{formatNumber(data.unit_count)}</DefRow>
              {data.deep_source ? (
                <DefRow label="Deep kaynağı">
                  <MethodLabel methodKey={data.deep_source} />
                </DefRow>
              ) : null}
              <DefRow label="Son güncelleme">{formatDate(data.updated_at)}</DefRow>
              <DefRow label="İçerik kimliği">
                <span className="mono">{data.content_id ? truncate(data.content_id, 20) : '—'}</span>
              </DefRow>
            </div>
          </div>

          <div className="card section">
            <div className="card-header">
              <div>
                <div className="card-title">Paylaşılan içerik analizi</div>
                <div className="card-sub">
                  Aynı baytların her yüklemesi bu analizi paylaşır. Buradaki bir varyant başka bir
                  yüklemenin seçimi olabilir; bu doküman üzerinden sorulamaz.
                </div>
              </div>
            </div>
            <div className="card-pad">
              <DefRow label="İstenen yöntemler">
                <MethodChips keys={data.content.requested_methods} ready={data.content.ready_methods} />
              </DefRow>
              <DefRow label="Hazır yöntemler">
                <MethodChips keys={data.content.ready_methods} ready={data.content.ready_methods} />
              </DefRow>
              <DefRow label="Aynı içeriği paylaşan yüklemeler">
                {formatNumber(data.content.shared_with_document_ids.length)}
              </DefRow>
            </div>
          </div>

          {adding ? (
            <AddMethodsDialog
              documentId={documentId}
              analysis={data}
              onClose={() => setAdding(false)}
              onDone={(next) => {
                state.set(next);
                setAdding(false);
                toast.success('Yöntemler sıraya alındı.');
              }}
            />
          ) : null}

          {inspect ? (
            <MethodChunksDialog
              documentId={documentId}
              method={inspect}
              onClose={() => setInspect(null)}
            />
          ) : null}
        </>
      )}
    </Async>
  );
}

/* ------------------------------------------------------------------ */
/* Method names, spelled the way the registry spells them              */
/* ------------------------------------------------------------------ */

/** The registry's label for a key, or the key itself until it answers. */
export function MethodLabel({ methodKey }: { methodKey: string }) {
  const state = useChunkingMethods();
  const label = state.data?.find((method) => method.key === methodKey)?.label;
  return <>{label ?? methodKey}</>;
}

function MethodChips({
  keys,
  ready = [],
  failed = [],
}: {
  keys: string[];
  ready?: string[];
  failed?: string[];
}) {
  if (!keys.length) return <span className="dim">—</span>;
  return (
    <span className="badge-list" style={{ justifyContent: 'flex-end' }}>
      {keys.map((key) => (
        <Badge
          key={key}
          tone={failed.includes(key) ? 'danger' : ready.includes(key) ? 'success' : 'neutral'}
        >
          <MethodLabel methodKey={key} />
        </Badge>
      ))}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Adding variants                                                     */
/* ------------------------------------------------------------------ */

function AddMethodsDialog({
  documentId,
  analysis,
  onClose,
  onDone,
}: {
  documentId: string;
  analysis: Analysis;
  onClose: () => void;
  onDone: (next: Analysis) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const locked = useMemo(() => analysis.selected_methods, [analysis.selected_methods]);

  const submit = async () => {
    const added = selected.filter((key) => !locked.includes(key));
    if (!added.length) {
      setError('Zaten seçili olmayan bir yöntem seçin.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onDone(await api.documents.addAnalysisMethods(documentId, added));
    } catch (cause) {
      const explanation = explain(cause);
      setError(`${explanation.title}. ${explanation.detail}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title="Analiz yöntemi ekle"
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Vazgeç
          </button>
          <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>
            Ekle ve çalıştır
          </button>
        </>
      }
    >
      <p className="field-hint" style={{ marginTop: 0, marginBottom: 12 }}>
        Doküman yeniden okunmaz: eklenen her yöntem aynı kanonik metin üzerinde çalışır.
      </p>
      <MethodPicker selected={selected} onChange={setSelected} locked={locked} />
      {error ? (
        <div style={{ marginTop: 12 }}>
          <InlineError>{error}</InlineError>
        </div>
      ) : null}
    </Modal>
  );
}

/* ------------------------------------------------------------------ */
/* Inspecting one method's rows                                        */
/* ------------------------------------------------------------------ */

function MethodChunksDialog({
  documentId,
  method,
  onClose,
}: {
  documentId: string;
  method: string;
  onClose: () => void;
}) {
  const [offset, setOffset] = useState(0);
  const state = useAsync<AnalysisChunks>(
    (signal) =>
      api.documents.analysisMethodChunks(documentId, method, {
        offset,
        limit: PAGE_SIZE,
        signal,
      }),
    [documentId, method, offset],
  );
  const [open, setOpen] = useState<number | null>(null);

  return (
    <Modal
      title="Yöntem parçaları"
      onClose={onClose}
      wide
      footer={
        <button type="button" className="btn btn-secondary" onClick={onClose}>
          Kapat
        </button>
      }
    >
      <div className="row" style={{ marginBottom: 12 }}>
        <Badge tone="accent">
          <MethodLabel methodKey={method} />
        </Badge>
        {state.data?.engine ? <Badge>{humanise(state.data.engine)}</Badge> : null}
        {state.data ? <span className="dim">{formatNumber(state.data.page.total)} parça</span> : null}
      </div>

      {state.loading && !state.data ? <Loading label="Parçalar okunuyor…" /> : null}
      {state.error && !state.data ? <MethodChunksError error={state.error} onRetry={state.reload} /> : null}

      {state.data ? (
        state.data.items.length ? (
          <>
            <div className="chunk-list">
              {state.data.items.map((row, index) => {
                const text = pickString(row, ROW_TEXT_KEYS) ?? rawRow(row);
                const section = pickString(row, ROW_SECTION_KEYS);
                const id = pickString(row, ROW_ID_KEYS);
                const size = pickNumber(row, ROW_SIZE_KEYS);
                const isOpen = open === index;
                return (
                  <button
                    key={id ?? index}
                    type="button"
                    className="chunk-row"
                    onClick={() => setOpen(isOpen ? null : index)}
                  >
                    <div className="chunk-head">
                      <span className="chunk-index">#{offset + index + 1}</span>
                      {section ? <span className="chunk-section">{section}</span> : null}
                      {size !== null ? <Badge>{formatNumber(size)} token</Badge> : null}
                    </div>
                    <div className={isOpen ? 'chunk-full' : 'chunk-text'}>{text}</div>
                  </button>
                );
              })}
            </div>
            <Pager
              offset={state.data.page.offset}
              limit={state.data.page.limit}
              total={state.data.page.total}
              onOffset={setOffset}
              busy={state.loading}
            />
          </>
        ) : (
          <Empty title="Bu yöntem hiç parça üretmedi" />
        )
      ) : null}
    </Modal>
  );
}

/**
 * The three refusals this endpoint can give, told apart.
 *
 * They mean three different things and a client that shows one message for all
 * three costs the reader the difference between "wait" and "stop asking".
 */
function MethodChunksError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  if (error instanceof ApiError && error.type === 'not_ready') {
    return (
      <div className="inline-note">
        Bu yöntem seçildi ama henüz üretilmedi. Analiz bitince burada görünür.
        <div style={{ marginTop: 8 }}>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
            Tekrar bak
          </button>
        </div>
      </div>
    );
  }
  if (error instanceof ApiError && error.type === 'not_found') {
    return (
      <div className="inline-note">
        Bu yöntem bu yüklemenin seçimlerinde yok. Aynı içeriğin başka bir yüklemesinde olabilir; bu
        doküman üzerinden sorulamaz. Eklemek için <strong>Yöntem ekle</strong>.
      </div>
    );
  }
  return <Failed error={error} onRetry={onRetry} />;
}
