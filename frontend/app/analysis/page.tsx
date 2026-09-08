'use client';

/**
 * Analysis -- the chunking variants, across a knowledge base.
 *
 * The per-document panel is the same component the document screen uses, so
 * there is one implementation of "what does this document's analysis say" and
 * two ways in. What this screen adds is the view across a knowledge base:
 * which documents have an analysis, which are still building, which failed --
 * and the deployment's method catalogue, read from the registry, as the
 * reference for what any of it can be asked for.
 */

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { AnalysisPanel, AnalysisStateBadge, MethodLabel } from '@/components/AnalysisPanel';
import { KbSelect } from '@/components/KbSelect';
import { useChunkingMethods } from '@/components/MethodPicker';
import { Async, Badge, Empty, Failed, Loading } from '@/components/ui';
import api from '@/lib/api';
import { formatNumber, humanise } from '@/lib/format';
import { useAsync } from '@/lib/hooks/useAsync';
import { useSelectedKb } from '@/lib/hooks/useSelectedKb';
import type { DocumentWithAnalysis } from '@/types/api';

export default function AnalysisPage() {
  const [kbId, setKbId] = useSelectedKb();
  const [documentId, setDocumentId] = useState<string>('');

  const documents = useAsync<DocumentWithAnalysis[]>(
    (signal) => (kbId ? api.documents.list(kbId, signal) : Promise.resolve([])),
    [kbId],
  );

  // Keep the selection inside the knowledge base that is on screen.
  useEffect(() => {
    const rows = documents.data;
    if (!rows) return;
    if (!rows.some((document) => document.id === documentId)) {
      setDocumentId(rows[0]?.id ?? '');
    }
  }, [documents.data, documentId]);

  return (
    <div className="page page-wide">
      <div className="page-header">
        <div>
          <h1 className="page-title">Analysis</h1>
          <p className="page-desc">
            Bir doküman bir kez okunur; seçilen her bölümleme yöntemi aynı kanonik metin üzerinde
            çalışır. İndekslenen külliyat değişmez — analiz, karşılaştırma içindir.
          </p>
        </div>
      </div>

      <MethodCatalogue />

      <div className="card section">
        <div className="card-header">
          <div className="card-title">Doküman seçin</div>
          <div className="row">
            <KbSelect value={kbId} onChange={setKbId} />
          </div>
        </div>
        {!kbId ? (
          <Empty title="Bir bilgi tabanı seçin" description="Analiz durumu doküman başına tutulur." />
        ) : (
          <Async state={documents} loadingLabel="Dokümanlar yükleniyor…">
            {(rows) =>
              rows.length ? (
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Doküman</th>
                        <th>Analiz</th>
                        <th>Seçilen yöntemler</th>
                        <th>Hazır</th>
                        <th>Birim</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((document) => (
                        <tr key={document.id ?? ''}>
                          <td className="cell-name">
                            <Link href={`/documents/${encodeURIComponent(document.id ?? '')}`}>
                              {document.name}
                            </Link>
                          </td>
                          <td>
                            <AnalysisStateBadge status={document.analysis.status} />
                          </td>
                          <td>
                            <span className="badge-list">
                              {document.analysis.selected_methods.length ? (
                                document.analysis.selected_methods.map((key) => (
                                  <Badge
                                    key={key}
                                    tone={
                                      document.analysis.failed_methods.includes(key)
                                        ? 'danger'
                                        : document.analysis.ready_methods.includes(key)
                                          ? 'success'
                                          : 'neutral'
                                    }
                                  >
                                    <MethodLabel methodKey={key} />
                                  </Badge>
                                ))
                              ) : (
                                <span className="dim">—</span>
                              )}
                            </span>
                          </td>
                          <td>{formatNumber(document.analysis.ready_methods.length)}</td>
                          <td>{formatNumber(document.analysis.unit_count)}</td>
                          <td className="cell-actions">
                            <button
                              type="button"
                              className={`btn btn-sm ${
                                documentId === document.id ? 'btn-primary' : 'btn-secondary'
                              }`}
                              onClick={() => setDocumentId(document.id ?? '')}
                            >
                              {documentId === document.id ? 'Seçili' : 'Seç'}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty
                  title="Bu bilgi tabanında doküman yok"
                  description="Önce bir doküman yükleyin; analiz yüklemenin bir parçası olarak çalışır."
                />
              )
            }
          </Async>
        )}
      </div>

      {documentId ? (
        <div className="section">
          <AnalysisPanel key={documentId} documentId={documentId} />
        </div>
      ) : null}
    </div>
  );
}

/**
 * The deployment's chunking methods, exactly as the registry reports them.
 *
 * Including the ones this machine cannot run, with the reason -- a picker that
 * hides an unavailable option leaves the reader wondering whether it exists.
 */
function MethodCatalogue() {
  const state = useChunkingMethods();

  if (state.loading && !state.data) return <Loading label="Yöntem kataloğu okunuyor…" />;
  if (state.error && !state.data) return <Failed error={state.error} onRetry={state.reload} />;

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Chunking methods</div>
          <div className="card-sub">
            Kütüphanenin kayıt defterinden okunur; bu arayüzde ikinci bir liste yoktur.
          </div>
        </div>
      </div>
      <div className="card-pad">
        <div className="grid-cards">
          {(state.data ?? []).map((method) => (
            <div
              key={method.key}
              className="card"
              style={{ padding: '14px 16px', boxShadow: 'none', background: 'var(--surface-2)' }}
            >
              <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ fontWeight: 620, fontSize: 13.5 }}>{method.label}</div>
                {method.available ? (
                  method.default ? (
                    <Badge tone="accent">varsayılan</Badge>
                  ) : null
                ) : (
                  <Badge tone="warn">kullanılamıyor</Badge>
                )}
              </div>
              <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>
                {method.available ? method.summary : (method.unavailable_reason ?? method.summary)}
              </div>
              <div className="badge-list" style={{ marginTop: 10 }}>
                <Badge title="Bu yöntemi çalıştıran motor">{humanise(method.engine)}</Badge>
                {method.uses_model ? <Badge tone="accent">model destekli</Badge> : null}
                {method.orchestration ? (
                  <Badge title={method.baseline ? `${method.baseline} üzerinde çalışır` : undefined}>
                    orkestrasyon
                  </Badge>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
