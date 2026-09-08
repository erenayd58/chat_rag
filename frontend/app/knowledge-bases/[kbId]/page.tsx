'use client';

/**
 * One knowledge base: what is in it, what was uploaded, and its settings.
 *
 * Three tabs, and the split is the one the old console had because it was the
 * right one: an overview that answers "is there anything here", the documents
 * with their ingestion and analysis state, and the settings that can still be
 * changed after creation -- which the contract limits to the name, the extra
 * fields and re-embedding.
 */

import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { AnalysisStateBadge } from '@/components/AnalysisPanel';
import { IngestJobs } from '@/components/IngestJobs';
import { useConfirm } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import { UploadDialog } from '@/components/UploadDialog';
import { Async, Badge, DefRow, Empty, Failed, InlineError, Loading } from '@/components/ui';
import api from '@/lib/api';
import { toastMessage } from '@/lib/errors';
import { formatBytes, formatDate, formatNumber, humanise } from '@/lib/format';
import { useAsync } from '@/lib/hooks/useAsync';
import { successNote } from '@/lib/jobText';
import type { DocumentWithAnalysis, EmbeddingIndex, KnowledgeBase } from '@/types/api';

type Tab = 'overview' | 'documents' | 'settings';

const TABS: { key: Tab; label: string }[] = [
  { key: 'overview', label: 'Genel bakış' },
  { key: 'documents', label: 'Documents' },
  { key: 'settings', label: 'Ayarlar' },
];

export default function KnowledgeBasePage() {
  const params = useParams<{ kbId: string }>();
  const kbId = decodeURIComponent(params.kbId);
  const [tab, setTab] = useState<Tab>('overview');

  const kb = useAsync<KnowledgeBase>((signal) => api.knowledgeBases.get(kbId, signal), [kbId]);
  const documents = useAsync<DocumentWithAnalysis[]>(
    (signal) => api.documents.list(kbId, signal),
    [kbId],
  );
  const [uploading, setUploading] = useState(false);
  const [jobToken, setJobToken] = useState(0);
  const toast = useToast();

  const reloadDocuments = documents.reload;
  const onJobsDrained = useCallback(() => reloadDocuments(), [reloadDocuments]);

  return (
    <div className="page">
      <div className="breadcrumb">
        <Link href="/knowledge-bases">Knowledge Bases</Link>
        <span>/</span>
        <span>{kb.data?.name ?? '…'}</span>
      </div>

      <div className="page-header">
        <div>
          <h1 className="page-title">{kb.data?.name ?? ' '}</h1>
          <p className="page-desc">
            {typeof kb.data?.extra?.description === 'string' ? kb.data.extra.description : ''}
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setUploading(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <path d="M17 8l-5-5-5 5" />
            <path d="M12 3v12" />
          </svg>
          Doküman yükle
        </button>
      </div>

      {kb.error && !kb.data ? <Failed error={kb.error} onRetry={kb.reload} /> : null}

      {kb.data ? (
        <>
          <div className="tabs" role="tablist">
            {TABS.map((entry) => (
              <button
                key={entry.key}
                type="button"
                role="tab"
                aria-selected={entry.key === tab}
                className={`tab${entry.key === tab ? ' active' : ''}`}
                onClick={() => setTab(entry.key)}
              >
                {entry.label}
                {entry.key === 'documents' && documents.data ? (
                  <span className="tab-count">{documents.data.length}</span>
                ) : null}
              </button>
            ))}
          </div>

          {tab === 'overview' ? <Overview kb={kb.data} documents={documents} /> : null}

          {tab === 'documents' ? (
            <div className="card">
              <IngestJobs knowledgeBaseId={kbId} onDrained={onJobsDrained} refreshToken={jobToken} />
              <Documents
                kbId={kbId}
                state={documents}
                onUpload={() => setUploading(true)}
                onChanged={documents.reload}
              />
            </div>
          ) : null}

          {tab === 'settings' ? <Settings kb={kb.data} onSaved={kb.set} /> : null}
        </>
      ) : kb.loading ? (
        <Loading label="Bilgi tabanı yükleniyor…" />
      ) : null}

      {uploading ? (
        <UploadDialog
          knowledgeBaseId={kbId}
          onClose={() => setUploading(false)}
          onSettled={(job) => {
            setJobToken((value) => value + 1);
            if (job?.status === 'succeeded') toast.success(successNote(job));
            documents.reload();
          }}
        />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Overview                                                            */
/* ------------------------------------------------------------------ */

function Overview({
  kb,
  documents,
}: {
  kb: KnowledgeBase;
  documents: ReturnType<typeof useAsync<DocumentWithAnalysis[]>>;
}) {
  const rows = documents.data ?? [];
  const chunks = rows.reduce((total, document) => total + (document.chunk_count ?? 0), 0);
  const bytes = rows.reduce((total, document) => total + (document.size_bytes ?? 0), 0);
  const latest = rows.reduce((newest, document) => {
    const value = document.ingested_at ?? '';
    return value > newest ? value : newest;
  }, '');
  const chunker = typeof kb.chunker?.type === 'string' ? kb.chunker.type : null;

  return (
    <>
      <div className="stats-row">
        <div className="card stat-tile">
          <div className="stat-label">Documents</div>
          <div className="stat-value">{documents.data ? formatNumber(rows.length) : '—'}</div>
        </div>
        <div className="card stat-tile">
          <div className="stat-label">Parça</div>
          <div className="stat-value">{documents.data ? formatNumber(chunks) : '—'}</div>
        </div>
        <div className="card stat-tile">
          <div className="stat-label">Toplam boyut</div>
          <div className="stat-value">{documents.data ? formatBytes(bytes) : '—'}</div>
        </div>
        <div className="card stat-tile">
          <div className="stat-label">Son indeksleme</div>
          <div className="stat-value stat-value-sm">{latest ? formatDate(latest) : '—'}</div>
        </div>
      </div>

      <div className="card section">
        <div className="card-header">
          <div>
            <div className="card-title">Yapılandırma</div>
            <div className="card-sub">
              Bölümleyici, embedding modeli ve depolama oluşturulurken sabitlendi.
            </div>
          </div>
        </div>
        <div className="card-pad">
          <DefRow label="İndeksleme bölümleyicisi">{chunker ? humanise(chunker) : '—'}</DefRow>
          <DefRow label="Retrieval method">{(kb.retrieval_method ?? '—').toUpperCase()}</DefRow>
          <DefRow label="Embedding modeli">{kb.embedding_model ?? '—'}</DefRow>
          <DefRow label="Knowledge base ID">
            <span className="mono">{kb.id}</span>
          </DefRow>
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Documents                                                           */
/* ------------------------------------------------------------------ */

function Documents({
  kbId,
  state,
  onUpload,
  onChanged,
}: {
  kbId: string;
  state: ReturnType<typeof useAsync<DocumentWithAnalysis[]>>;
  onUpload: () => void;
  onChanged: () => void;
}) {
  const confirm = useConfirm();
  const toast = useToast();

  const remove = async (document: DocumentWithAnalysis) => {
    const ok = await confirm({
      title: 'Dokümanı sil',
      message: `"${document.name}" ve bu bilgi tabanındaki ${document.chunk_count} parçası silinsin mi? İçerik analizi, aynı baytların başka bir yüklemesi varsa korunur.`,
      confirmLabel: 'Sil',
      danger: true,
    });
    if (!ok || !document.id) return;
    try {
      await api.documents.remove(document.id);
      toast.success('Doküman silindi.');
      onChanged();
    } catch (error) {
      toast.error(toastMessage(error));
    }
  };

  return (
    <Async
      state={state}
      loadingLabel="Dokümanlar yükleniyor…"
      empty={<Empty title="Henüz doküman yok" />}
    >
      {(documents) =>
        documents.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Doküman</th>
                  <th>Durum</th>
                  <th>İndeksleme</th>
                  <th>Analiz</th>
                  <th>Parça</th>
                  <th>Yüklendi</th>
                  <th>Boyut</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <tr key={document.id ?? ''}>
                    <td className="cell-name">
                      <Link href={`/documents/${encodeURIComponent(document.id ?? '')}`}>
                        {document.name}
                      </Link>
                    </td>
                    <td>
                      <DocumentStatus status={document.status} />
                    </td>
                    <td>
                      {document.chunking_mode ? (
                        <Badge tone="accent">{humanise(document.chunking_mode)}</Badge>
                      ) : (
                        <span className="dim">—</span>
                      )}
                    </td>
                    <td>
                      <AnalysisStateBadge status={document.analysis.status} />
                    </td>
                    <td>{formatNumber(document.chunk_count)}</td>
                    <td className="nowrap muted">{formatDate(document.ingested_at)}</td>
                    <td className="muted">{formatBytes(document.size_bytes)}</td>
                    <td className="cell-actions">
                      <Link
                        className="btn btn-ghost btn-sm"
                        href={`/documents/${encodeURIComponent(document.id ?? '')}`}
                      >
                        İncele
                      </Link>
                      <button
                        type="button"
                        className="btn btn-danger-ghost btn-sm"
                        onClick={() => remove(document)}
                      >
                        Sil
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty
            title="Henüz doküman yok"
            description="Bu bilgi tabanına bir doküman yükleyin; arama ve sohbet için indekslensin."
            action={
              <button type="button" className="btn btn-primary" onClick={onUpload}>
                Doküman yükle
              </button>
            }
          />
        )
      }
    </Async>
  );
}

function DocumentStatus({ status }: { status: string }) {
  if (status === 'indexed')
    return (
      <Badge tone="success" dot>
        İndekslendi
      </Badge>
    );
  if (status === 'processing')
    return (
      <Badge tone="warn" dot>
        İşleniyor
      </Badge>
    );
  if (status === 'failed')
    return (
      <Badge tone="danger" dot>
        Başarısız
      </Badge>
    );
  return <Badge>{humanise(status)}</Badge>;
}

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

function Settings({ kb, onSaved }: { kb: KnowledgeBase; onSaved: (next: KnowledgeBase) => void }) {
  const router = useRouter();
  const confirm = useConfirm();
  const toast = useToast();
  const [name, setName] = useState(kb.name ?? '');
  const [description, setDescription] = useState(
    typeof kb.extra?.description === 'string' ? kb.extra.description : '',
  );
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    if (!name.trim()) {
      setError('Bir ad gerekli.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const next = await api.knowledgeBases.update(kb.id ?? '', {
        name: name.trim(),
        extra: { ...kb.extra, description: description.trim() },
      });
      onSaved(next);
      toast.success('Ayarlar kaydedildi.');
    } catch (cause) {
      setError(toastMessage(cause));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    const ok = await confirm({
      title: 'Bilgi tabanını sil',
      message: `"${kb.name}" ve arama indeksi silinsin mi? Bu geri alınamaz. Dokümanların ingest kayıtları ve analizleri korunur.`,
      confirmLabel: 'Sil',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.knowledgeBases.remove(kb.id ?? '');
      toast.success('Bilgi tabanı silindi.');
      router.push('/knowledge-bases');
    } catch (cause) {
      toast.error(toastMessage(cause));
    }
  };

  return (
    <>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Genel</div>
        </div>
        <div className="card-pad">
          <div className="field" style={{ maxWidth: 460 }}>
            <label className="field-label" htmlFor="settingsName">
              Ad
            </label>
            <input
              className="input"
              id="settingsName"
              maxLength={120}
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="field" style={{ maxWidth: 460 }}>
            <label className="field-label" htmlFor="settingsDescription">
              Açıklama
            </label>
            <textarea
              className="textarea"
              id="settingsDescription"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>
          {error ? (
            <div style={{ marginBottom: 12 }}>
              <InlineError>{error}</InlineError>
            </div>
          ) : null}
          <button type="button" className="btn btn-primary" disabled={busy} onClick={save}>
            Değişiklikleri kaydet
          </button>
        </div>
      </div>

      <EmbeddingIndexCard kbId={kb.id ?? ''} />

      <div className="card section">
        <div className="card-header">
          <div className="card-title" style={{ color: 'var(--danger)' }}>
            Dikkat
          </div>
        </div>
        <div className="card-pad row" style={{ justifyContent: 'space-between' }}>
          <div className="muted" style={{ fontSize: 13, maxWidth: '60ch' }}>
            Bilgi tabanını silmek kaydını ve arama indeksini kaldırır. Dokümanların ingest kayıtları ve
            içerik analizleri kasıtlı olarak korunur; bir dosyanın bir kez yüklendiği bilgisi kaybolmaz.
          </div>
          <button type="button" className="btn btn-danger" onClick={remove}>
            Bilgi tabanını sil
          </button>
        </div>
      </div>
    </>
  );
}

/**
 * Do the stored vectors belong to the configured embedding model?
 *
 * The body is the store's own manifest report and is published as
 * pass-through, so this card reads the keys every profile answers and shows
 * the rest of the reason verbatim rather than pretending to understand it.
 */
function EmbeddingIndexCard({ kbId }: { kbId: string }) {
  const toast = useToast();
  const confirm = useConfirm();
  const state = useAsync<EmbeddingIndex>((signal) => api.knowledgeBases.embeddingIndex(kbId, signal), [kbId]);
  const [busy, setBusy] = useState(false);

  const rebuild = async () => {
    const ok = await confirm({
      title: 'Vektörleri yeniden üret',
      message:
        'Bu bilgi tabanının anlamsal indeksi geçerli embedding modeliyle yeniden üretilsin mi? Her parça yeniden vektörlenir; dokümanlar ve parçalar değişmez.',
      confirmLabel: 'Yeniden indeksle',
    });
    if (!ok) return;
    setBusy(true);
    try {
      const done = await api.knowledgeBases.rebuildEmbeddingIndex(kbId);
      const result = done.result as { chunks?: number; model?: string; seconds?: number };
      toast.success(
        `Yeniden indekslendi: ${formatNumber(result.chunks ?? null)} parça · ${result.model ?? '—'}`,
      );
      state.reload();
    } catch (error) {
      toast.error(toastMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card section">
      <div className="card-header">
        <div>
          <div className="card-title">Embedding indeksi</div>
          <div className="card-sub">
            Anlamsal arama, sorunun vektörünü burada saklanan vektörlerle karşılaştırır; ikisi de aynı
            modelden gelmelidir.
          </div>
        </div>
      </div>
      <div className="card-pad">
        <Async state={state} loadingLabel="Kontrol ediliyor…">
          {(index) => {
            const stored = index.stored ?? {};
            const current = index.current ?? {};
            const canRebuild = index.state !== 'not_applicable' && (stored.chunk_count ?? 0) > 0;
            return (
              <>
                <DefRow label="Durum">
                  <IndexStateBadge state={index.state} />
                </DefRow>
                <DefRow label="Geçerli embedding modeli">
                  {current.model ?? '—'}
                  {current.dimension ? ` · ${current.dimension} boyut` : ''}
                </DefRow>
                <DefRow label="Kayıtlı vektörler">
                  {stored.model ?? (stored.chunk_count ? 'model bilinmiyor' : '—')}
                  {stored.dimension ? ` · ${stored.dimension} boyut` : ''}
                </DefRow>
                <DefRow label="Kayıtlı parça">{formatNumber(stored.chunk_count ?? null)}</DefRow>
                <DefRow label="Parmak izi (kayıtlı / geçerli)">
                  <span className="mono">{`${stored.fingerprint ?? '—'} / ${current.fingerprint ?? '—'}`}</span>
                </DefRow>
                <div className="row" style={{ marginTop: 14 }}>
                  <button
                    type="button"
                    className={`btn ${
                      index.state === 'reindex_required' || index.state === 'no_dense_index'
                        ? 'btn-primary'
                        : 'btn-secondary'
                    }`}
                    disabled={busy || !canRebuild}
                    onClick={rebuild}
                  >
                    Geçerli modelle yeniden indeksle
                  </button>
                  {index.reason ? <span className="muted" style={{ fontSize: 12.5 }}>{index.reason}</span> : null}
                </div>
              </>
            );
          }}
        </Async>
      </div>
    </div>
  );
}

function IndexStateBadge({ state }: { state?: string }) {
  if (state === 'compatible')
    return (
      <Badge tone="success" dot>
        Güncel
      </Badge>
    );
  if (state === 'empty') return <Badge>Boş</Badge>;
  if (state === 'not_applicable') return <Badge>Bu profilde kullanılmıyor</Badge>;
  if (state === 'no_dense_index')
    return (
      <Badge tone="warn" dot>
        Anlamsal indeks yok
      </Badge>
    );
  if (state === 'reindex_required')
    return (
      <Badge tone="warn" dot>
        Yeniden indeksleme gerekiyor
      </Badge>
    );
  return <Badge>{humanise(state)}</Badge>;
}
