'use client';

/**
 * Knowledge Bases -- the collection everything else hangs off.
 *
 * The counts on each card are the documents' own numbers, totalled from
 * `GET /api/v1/documents`. There is no `/stats` on the contract and none is
 * wanted: every number a card shows is one the document collection already
 * answered, so nothing here is a figure this screen made up.
 */

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useMemo, useState } from 'react';
import { Modal, useConfirm } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import { Async, Badge, Empty, InlineError } from '@/components/ui';
import api from '@/lib/api';
import { toastMessage } from '@/lib/errors';
import { formatDate, formatNumber, humanise } from '@/lib/format';
import { useAsync } from '@/lib/hooks/useAsync';
import type { DocumentWithAnalysis, KnowledgeBase } from '@/types/api';

interface Totals {
  documents: number;
  chunks: number;
  latest: string;
}

function totalsByKb(documents: DocumentWithAnalysis[]): Map<string, Totals> {
  const totals = new Map<string, Totals>();
  for (const document of documents) {
    const kbId = document.knowledge_base_id;
    if (!kbId) continue;
    const current = totals.get(kbId) ?? { documents: 0, chunks: 0, latest: '' };
    current.documents += 1;
    current.chunks += document.chunk_count ?? 0;
    if ((document.ingested_at ?? '') > current.latest) current.latest = document.ingested_at ?? '';
    totals.set(kbId, current);
  }
  return totals;
}

export default function KnowledgeBasesPage() {
  const [creating, setCreating] = useState(false);
  const state = useAsync(
    async (signal) => {
      const [bases, documents] = await Promise.all([
        api.knowledgeBases.list(signal),
        api.documents.list(null, signal),
      ]);
      return { bases, documents };
    },
    [],
  );

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Knowledge Bases</h1>
          <p className="page-desc">
            Her bilgi tabanı kendi dokümanlarını ve kendi arama indeksini taşır. Bölümleme yöntemi,
            embedding modeli ve depolama oluşturulurken belirlenir; ingest edilen külliyat onlara bağlıdır.
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setCreating(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
          Yeni bilgi tabanı
        </button>
      </div>

      <Async state={state} loadingLabel="Bilgi tabanları yükleniyor…">
        {({ bases, documents }) =>
          bases.length ? (
            <KbGrid bases={bases} totals={totalsByKb(documents)} onChanged={state.reload} />
          ) : (
            <div className="card">
              <Empty
                title="Henüz bilgi tabanı yok"
                description="Bir bilgi tabanı oluşturun, içine doküman yükleyin ve sorularınızı sormaya başlayın."
                action={
                  <button type="button" className="btn btn-primary" onClick={() => setCreating(true)}>
                    Yeni bilgi tabanı
                  </button>
                }
              />
            </div>
          )
        }
      </Async>

      {creating ? <CreateDialog onClose={() => setCreating(false)} /> : null}
    </div>
  );
}

function KbGrid({
  bases,
  totals,
  onChanged,
}: {
  bases: KnowledgeBase[];
  totals: Map<string, Totals>;
  onChanged: () => void;
}) {
  const router = useRouter();
  const confirm = useConfirm();
  const toast = useToast();

  const remove = async (kb: KnowledgeBase) => {
    const stats = totals.get(kb.id ?? '');
    const ok = await confirm({
      title: 'Bilgi tabanını sil',
      message: `"${kb.name}"${
        stats?.documents ? ` ve içindeki ${stats.documents} doküman` : ''
      } silinsin mi? Arama indeksi de kaldırılır ve bu geri alınamaz.`,
      confirmLabel: 'Sil',
      danger: true,
    });
    if (!ok || !kb.id) return;
    try {
      await api.knowledgeBases.remove(kb.id);
      toast.success('Bilgi tabanı silindi.');
      onChanged();
    } catch (error) {
      toast.error(toastMessage(error));
    }
  };

  return (
    <div className="grid-cards">
      {bases.map((kb) => {
        const stats = totals.get(kb.id ?? '') ?? { documents: 0, chunks: 0, latest: '' };
        const description = typeof kb.extra?.description === 'string' ? kb.extra.description : '';
        const chunker = typeof kb.chunker?.type === 'string' ? kb.chunker.type : null;
        return (
          <div
            key={kb.id ?? ''}
            className="card kb-card"
            role="link"
            tabIndex={0}
            onClick={() => router.push(`/knowledge-bases/${encodeURIComponent(kb.id ?? '')}`)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') router.push(`/knowledge-bases/${encodeURIComponent(kb.id ?? '')}`);
            }}
          >
            <div className="row" style={{ alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div style={{ minWidth: 0 }}>
                <div className="kb-card-name">{kb.name}</div>
                {description ? <div className="kb-card-desc">{description}</div> : null}
              </div>
              {stats.documents ? (
                <Badge tone="success" dot>
                  İndekslendi
                </Badge>
              ) : (
                <Badge dot>Boş</Badge>
              )}
            </div>

            <div className="kb-card-stats">
              <span>
                <strong>{formatNumber(stats.documents)}</strong> doküman
              </span>
              <span>
                <strong>{formatNumber(stats.chunks)}</strong> parça
              </span>
            </div>

            <div className="kb-card-foot">
              <span className="dim" style={{ fontSize: '11.5px' }}>
                {stats.latest ? `Güncellendi ${formatDate(stats.latest)}` : 'Henüz doküman yok'}
              </span>
              <span className="badge-list">
                {chunker ? <Badge title="Bu bilgi tabanının indeksleme bölümleyicisi">{humanise(chunker)}</Badge> : null}
                {kb.retrieval_method ? <Badge>{kb.retrieval_method.toUpperCase()}</Badge> : null}
              </span>
            </div>

            <div className="kb-card-actions" onClick={(event) => event.stopPropagation()}>
              <Link
                className="btn btn-secondary btn-sm"
                href={`/knowledge-bases/${encodeURIComponent(kb.id ?? '')}`}
              >
                Aç
              </Link>
              <button type="button" className="btn btn-danger-ghost btn-sm" onClick={() => remove(kb)}>
                Sil
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Creating one.
 *
 * Name and description only. The chunker, the embedding model, the store and
 * the retrieval method are left out of the payload on purpose, so the
 * deployment's own defaults apply -- naming one here would be this front end
 * holding an opinion about the library's registry, which is the thing it must
 * not do.
 */
function CreateDialog({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const toast = useToast();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError('Bir ad gerekli.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await api.knowledgeBases.create({
        name: trimmed,
        extra: description.trim() ? { description: description.trim() } : {},
      });
      toast.success('Bilgi tabanı oluşturuldu.');
      router.push(`/knowledge-bases/${encodeURIComponent(created.id ?? '')}`);
    } catch (cause) {
      setError(toastMessage(cause));
      setBusy(false);
    }
  };

  const footer = useMemo(
    () => (
      <>
        <button type="button" className="btn btn-secondary" onClick={onClose}>
          Vazgeç
        </button>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>
          Oluştur
        </button>
      </>
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [busy, name, description],
  );

  return (
    <Modal title="Yeni bilgi tabanı" onClose={onClose} footer={footer}>
      <div className="field">
        <label className="field-label" htmlFor="kbName">
          Ad
        </label>
        <input
          className="input"
          id="kbName"
          value={name}
          maxLength={120}
          autoFocus
          placeholder="örn. Yıllık faaliyet raporları"
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') submit();
          }}
        />
      </div>
      <div className="field">
        <label className="field-label" htmlFor="kbDescription">
          Açıklama <span className="dim">(isteğe bağlı)</span>
        </label>
        <textarea
          className="textarea"
          id="kbDescription"
          value={description}
          placeholder="Bu bilgi tabanı neyi içeriyor?"
          onChange={(event) => setDescription(event.target.value)}
        />
      </div>
      {error ? <InlineError>{error}</InlineError> : null}
    </Modal>
  );
}
