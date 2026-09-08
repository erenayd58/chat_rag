'use client';

/**
 * One document: what it was indexed as, what the parser read, and its analysis.
 *
 * The three tabs are three different things and the contract keeps them apart:
 * `/chunks` is what the **knowledge base's own chunker** produced and is what
 * retrieval actually searches; `/units` is the parser's canonical reading
 * before any chunker touched it; the analysis is the chunking *variants*,
 * which do not change what is indexed.
 */

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useState } from 'react';
import { AnalysisPanel, AnalysisStateBadge } from '@/components/AnalysisPanel';
import { ChunkDialog, type ChunkFact } from '@/components/ChunkDialog';
import { useConfirm } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import { Async, Badge, Empty, Pager, Tabs } from '@/components/ui';
import api from '@/lib/api';
import { toastMessage } from '@/lib/errors';
import { formatBytes, formatDate, formatNumber, humanise, truncate } from '@/lib/format';
import { useAsync } from '@/lib/hooks/useAsync';
import type {
  CanonicalUnit,
  CanonicalUnitCollection,
  Chunk,
  Collection,
  DocumentWithAnalysis,
} from '@/types/api';

type Tab = 'analysis' | 'chunks' | 'units';
const PAGE_SIZE = 25;

export default function DocumentPage() {
  const params = useParams<{ documentId: string }>();
  const documentId = decodeURIComponent(params.documentId);
  const [tab, setTab] = useState<Tab>('analysis');
  const confirm = useConfirm();
  const toast = useToast();

  const state = useAsync<DocumentWithAnalysis>((signal) => api.documents.get(documentId, signal), [
    documentId,
  ]);

  const remove = async (document: DocumentWithAnalysis) => {
    const ok = await confirm({
      title: 'Dokümanı sil',
      message: `"${document.name}" ve ${document.chunk_count} parçası silinsin mi?`,
      confirmLabel: 'Sil',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.documents.remove(documentId);
      toast.success('Doküman silindi.');
      window.location.href = `/knowledge-bases/${encodeURIComponent(document.knowledge_base_id ?? '')}`;
    } catch (error) {
      toast.error(toastMessage(error));
    }
  };

  return (
    <div className="page page-wide">
      <Async state={state} loadingLabel="Doküman yükleniyor…">
        {(document) => (
          <>
            <div className="breadcrumb">
              <Link href="/knowledge-bases">Knowledge Bases</Link>
              <span>/</span>
              <Link href={`/knowledge-bases/${encodeURIComponent(document.knowledge_base_id ?? '')}`}>
                Documents
              </Link>
              <span>/</span>
              <span>{document.name}</span>
            </div>

            <div className="page-header">
              <div>
                <h1 className="page-title">{document.name}</h1>
                <div className="row" style={{ marginTop: 8 }}>
                  <Badge tone={document.status === 'indexed' ? 'success' : 'neutral'} dot>
                    {humanise(document.status)}
                  </Badge>
                  <AnalysisStateBadge status={document.analysis.status} />
                  {document.chunking_mode ? (
                    <Badge tone="accent" title="Bu dokümanın indekslendiği bölümleme">
                      {humanise(document.chunking_mode)}
                    </Badge>
                  ) : null}
                  <span className="muted" style={{ fontSize: 12.5 }}>
                    {formatNumber(document.chunk_count)} parça · {formatBytes(document.size_bytes)} ·{' '}
                    {formatDate(document.ingested_at)}
                  </span>
                </div>
              </div>
              <button type="button" className="btn btn-danger-ghost" onClick={() => remove(document)}>
                Dokümanı sil
              </button>
            </div>

            <Tabs<Tab>
              tabs={[
                { key: 'analysis', label: 'Analysis' },
                { key: 'chunks', label: 'İndekslenen parçalar', count: document.chunk_count },
                { key: 'units', label: 'Kanonik birimler', count: document.analysis.unit_count },
              ]}
              active={tab}
              onChange={setTab}
            />

            {tab === 'analysis' ? <AnalysisPanel documentId={documentId} /> : null}
            {tab === 'chunks' ? (
              <IndexedChunks documentId={documentId} kbId={document.knowledge_base_id} />
            ) : null}
            {tab === 'units' ? <CanonicalUnits documentId={documentId} kbId={document.knowledge_base_id} /> : null}
          </>
        )}
      </Async>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* What retrieval searches                                             */
/* ------------------------------------------------------------------ */

function IndexedChunks({ documentId, kbId }: { documentId: string; kbId: string | null }) {
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState<Chunk | null>(null);
  const state = useAsync<Collection<Chunk>>(
    (signal) =>
      api.documents.chunks(documentId, {
        offset,
        limit: PAGE_SIZE,
        knowledgeBaseId: kbId,
        signal,
      }),
    [documentId, kbId, offset],
  );

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">İndekslenen parçalar</div>
          <div className="card-sub">
            Bilgi tabanının kendi bölümleyicisinin ürettiği parçalar — arama bunları tarar.
          </div>
        </div>
      </div>
      <Async state={state} loadingLabel="Parçalar yükleniyor…">
        {(page) =>
          page.items.length ? (
            <>
              <div className="chunk-list">
                {page.items.map((chunk, index) => (
                  <button
                    key={chunk.id ?? index}
                    type="button"
                    className="chunk-row"
                    onClick={() => setOpen(chunk)}
                  >
                    <div className="chunk-head">
                      <span className="chunk-index">
                        #{chunk.chunk_index ?? offset + index + 1}
                        {chunk.total_chunks ? ` / ${chunk.total_chunks}` : ''}
                      </span>
                      {chunk.section ? <span className="chunk-section">{chunk.section}</span> : null}
                      {chunk.chunking_mode ? <Badge>{humanise(chunk.chunking_mode)}</Badge> : null}
                    </div>
                    <div className="chunk-text">{chunk.content}</div>
                  </button>
                ))}
              </div>
              <Pager
                offset={page.page.offset}
                limit={page.page.limit}
                total={page.page.total}
                onOffset={setOffset}
                busy={state.loading}
              />
            </>
          ) : (
            <Empty
              title="Parça yok"
              description="Bu doküman için indekslenmiş parça bulunamadı."
            />
          )
        }
      </Async>
      {open ? (
        <ChunkDialog
          title={open.section || `Parça ${open.chunk_index ?? ''}`}
          text={open.content}
          facts={chunkFacts(open)}
          onClose={() => setOpen(null)}
        />
      ) : null}
    </div>
  );
}

function chunkFacts(chunk: Chunk): ChunkFact[] {
  return [
    { label: 'Parça kimliği', value: chunk.id ?? '—' },
    { label: 'Sıra', value: chunk.chunk_index === null ? '—' : String(chunk.chunk_index) },
    { label: 'Toplam', value: chunk.total_chunks === null ? '—' : String(chunk.total_chunks) },
    { label: 'Bölüm', value: chunk.section ?? '—' },
    { label: 'Bölümleme', value: chunk.chunking_mode ? humanise(chunk.chunking_mode) : '—' },
  ];
}

/* ------------------------------------------------------------------ */
/* What the parser read                                                */
/* ------------------------------------------------------------------ */

function CanonicalUnits({ documentId, kbId }: { documentId: string; kbId: string | null }) {
  const [offset, setOffset] = useState(0);
  const [unitType, setUnitType] = useState('');
  const [open, setOpen] = useState<CanonicalUnit | null>(null);

  const state = useAsync<CanonicalUnitCollection>(
    (signal) =>
      api.documents.units(documentId, {
        offset,
        limit: PAGE_SIZE,
        knowledgeBaseId: kbId,
        unitType: unitType || undefined,
        signal,
      }),
    [documentId, kbId, offset, unitType],
  );

  const types = Array.from(
    new Set((state.data?.items ?? []).map((unit) => unit.type).filter(Boolean) as string[]),
  );

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Kanonik birimler</div>
          <div className="card-sub">
            Ayrıştırıcının okuması — hiçbir bölümleyici dokunmadan önceki hâli.
          </div>
        </div>
        <div className="row">
          <select
            className="select input-inline"
            value={unitType}
            aria-label="Birim türü"
            onChange={(event) => {
              setUnitType(event.target.value);
              setOffset(0);
            }}
          >
            <option value="">Tüm türler</option>
            {types.map((type) => (
              <option key={type} value={type}>
                {humanise(type)}
              </option>
            ))}
          </select>
        </div>
      </div>
      <Async state={state} loadingLabel="Birimler yükleniyor…">
        {(page) =>
          page.items.length ? (
            <>
              <div className="chunk-list">
                {page.items.map((unit, index) => (
                  <button
                    key={unit.id ?? index}
                    type="button"
                    className="chunk-row"
                    onClick={() => setOpen(unit)}
                  >
                    <div className="chunk-head">
                      <span className="chunk-index">#{unit.order ?? offset + index + 1}</span>
                      {unit.type ? <Badge>{humanise(unit.type)}</Badge> : null}
                      {unit.heading_level ? <Badge tone="accent">H{unit.heading_level}</Badge> : null}
                      {unit.section_path.length ? (
                        <span className="chunk-section">{unit.section_path.join(' › ')}</span>
                      ) : null}
                    </div>
                    <div className="chunk-text">{unit.text}</div>
                  </button>
                ))}
              </div>
              <Pager
                offset={page.page.offset}
                limit={page.page.limit}
                total={page.page.total}
                onOffset={setOffset}
                busy={state.loading}
              />
            </>
          ) : (
            <Empty title="Birim yok" description="Bu filtre için kanonik birim bulunamadı." />
          )
        }
      </Async>
      {open ? (
        <ChunkDialog
          title={truncate(open.section_path.join(' › ') || open.type || 'Birim', 70)}
          text={open.text}
          facts={[
            { label: 'Birim kimliği', value: open.id ?? '—' },
            { label: 'Sıra', value: open.order === null ? '—' : String(open.order) },
            { label: 'Tür', value: open.type ? humanise(open.type) : '—' },
            {
              label: 'Başlık düzeyi',
              value: open.heading_level === null ? '—' : String(open.heading_level),
            },
            { label: 'Bölüm yolu', value: open.section_path.join(' › ') || '—' },
          ]}
          onClose={() => setOpen(null)}
        />
      ) : null}
    </div>
  );
}
