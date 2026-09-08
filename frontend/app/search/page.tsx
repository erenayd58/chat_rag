'use client';

/**
 * Search -- retrieval without an answer.
 *
 * `POST /api/v1/searches` takes a `method`, and which methods a knowledge base
 * can actually serve is a capability question: a lexical-only profile has no
 * vectors and asking it for one is a 400 with the reason. So the picker is
 * filled from `GET /api/v1/meta/retrieval-methods` for the selected knowledge
 * base, an unavailable method is offered disabled with the server's reason,
 * and the default is the server's default.
 *
 * A score is the retriever's own number. It is shown as that and never as a
 * percentage, a confidence or a relevance bar -- rank fusion does not produce
 * a probability and drawing one would be inventing a metric.
 */

import { useEffect, useState } from 'react';
import { ChunkDialog } from '@/components/ChunkDialog';
import { KbSelect } from '@/components/KbSelect';
import { Badge, Empty, Failed, InlineError, Loading } from '@/components/ui';
import api, { ApiError } from '@/lib/api';
import { explain } from '@/lib/errors';
import { formatNumber, formatScore, humanise } from '@/lib/format';
import { useAsync } from '@/lib/hooks/useAsync';
import { useSelectedKb } from '@/lib/hooks/useSelectedKb';
import type { RetrievalMethodCollection, ScoredChunk, SearchResults } from '@/types/api';

const LIMITS = [10, 20, 50];

export default function SearchPage() {
  const [kbId, setKbId] = useSelectedKb();
  const [query, setQuery] = useState('');
  const [method, setMethod] = useState('');
  const [limit, setLimit] = useState(LIMITS[0]);
  const [results, setResults] = useState<SearchResults | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [open, setOpen] = useState<ScoredChunk | null>(null);

  const methods = useAsync<RetrievalMethodCollection>(
    (signal) => api.meta.retrievalMethods(kbId || null, signal),
    [kbId],
  );

  // The server names the default and says which methods it can serve; a
  // selection it can no longer serve falls back rather than being sent.
  useEffect(() => {
    const found = methods.data;
    if (!found) return;
    const usable = found.items.filter((entry) => entry.available && entry.name).map((entry) => entry.name!);
    if (!usable.includes(method)) setMethod(found.default && usable.includes(found.default) ? found.default : (usable[0] ?? ''));
  }, [methods.data, method]);

  const run = async () => {
    if (!kbId || !query.trim()) return;
    setRunning(true);
    setError(null);
    try {
      setResults(
        await api.asking.search({
          query: query.trim(),
          knowledge_base_id: kbId,
          method: method || undefined,
          limit,
        }),
      );
    } catch (cause) {
      setError(cause);
      setResults(null);
    } finally {
      setRunning(false);
    }
  };

  const blocked = (methods.data?.items ?? []).filter(
    (entry) => !entry.available && entry.unavailable_reason,
  );

  return (
    <div className="page page-wide">
      <div className="page-header">
        <div>
          <h1 className="page-title">Search</h1>
          <p className="page-desc">
            Cevap üretmeden, yalnız getirilen parçaları görün. Bir sorunun hangi bölümlere dayandığını
            anlamak için en doğrudan yol.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-pad">
          <div className="toolbar">
            <div className="field">
              <label className="field-label" htmlFor="kbSelect">
                Knowledge Base
              </label>
              <KbSelect value={kbId} onChange={setKbId} />
            </div>
            <div className="field">
              <label className="field-label" htmlFor="methodSelect">
                Retrieval method
              </label>
              <select
                className="select"
                id="methodSelect"
                value={method}
                disabled={methods.loading && !methods.data}
                onChange={(event) => setMethod(event.target.value)}
              >
                {(methods.data?.items ?? []).map((entry) => (
                  <option
                    key={entry.name ?? ''}
                    value={entry.name ?? ''}
                    disabled={!entry.available}
                    title={entry.unavailable_reason ?? undefined}
                  >
                    {entry.label ?? entry.name}
                    {entry.available ? '' : ' — kullanılamıyor'}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label className="field-label" htmlFor="limitSelect">
                Sonuç
              </label>
              <select
                className="select"
                id="limitSelect"
                value={limit}
                onChange={(event) => setLimit(Number(event.target.value))}
              >
                {LIMITS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
            <div className="field toolbar-grow">
              <label className="field-label" htmlFor="queryInput">
                Sorgu
              </label>
              <input
                className="input"
                id="queryInput"
                value={query}
                placeholder="Aranacak ifade ya da soru…"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') run();
                }}
              />
            </div>
            <button
              type="button"
              className="btn btn-primary"
              disabled={running || !kbId || !query.trim()}
              onClick={run}
            >
              Ara
            </button>
          </div>

          {blocked.length ? (
            <div className="field-hint">
              {blocked
                .map((entry) => `${entry.label ?? entry.name}: ${entry.unavailable_reason}`)
                .join(' · ')}
            </div>
          ) : null}
          {!kbId ? <div className="field-hint">Başlamak için bir bilgi tabanı seçin.</div> : null}
        </div>
      </div>

      <div className="section">
        {running ? <Loading label="Getiriliyor…" /> : null}
        {!running && error ? <SearchError error={error} /> : null}
        {!running && !error && results ? (
          results.items.length ? (
            <div className="card">
              <div className="card-header">
                <div className="card-title">{formatNumber(results.items.length)} sonuç</div>
                <div className="row">
                  {results.method ? <Badge tone="accent">{humanise(results.method)}</Badge> : null}
                </div>
              </div>
              <div className="chunk-list">
                {results.items.map((chunk, index) => (
                  <button
                    key={chunk.id ?? index}
                    type="button"
                    className="chunk-row"
                    onClick={() => setOpen(chunk)}
                  >
                    <div className="result-head">
                      <span className="result-rank">#{index + 1}</span>
                      {chunk.section ? <span className="chunk-section">{chunk.section}</span> : null}
                      {chunk.retrieval_method ? <Badge>{humanise(chunk.retrieval_method)}</Badge> : null}
                      <span className="result-score">skor {formatScore(chunk.score)}</span>
                    </div>
                    <div className="chunk-text">{chunk.content}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="card">
              <Empty
                title="Eşleşme yok"
                description="Bu sorgu için parça bulunamadı. Farklı bir ifade ya da başka bir retrieval method deneyin."
              />
            </div>
          )
        ) : null}
        {!running && !error && !results ? (
          <div className="card">
            <Empty
              title="Henüz bir sorgu çalıştırılmadı"
              description={
                kbId
                  ? 'Bir ifade yazıp Ara deyin. Sonuçlar, retrieval yönteminin verdiği sırayla ve kendi skorlarıyla listelenir.'
                  : 'Bir bilgi tabanı seçin, ardından aramak istediğiniz ifadeyi yazın.'
              }
            />
          </div>
        ) : null}
      </div>

      {open ? (
        <ChunkDialog
          title={open.section || 'Parça'}
          text={open.content}
          facts={[
            { label: 'Parça kimliği', value: open.id ?? '—' },
            { label: 'Doküman', value: open.document_id ?? '—' },
            { label: 'Bölüm', value: open.section ?? '—' },
            { label: 'Bulan yöntem', value: open.retrieval_method ? humanise(open.retrieval_method) : '—' },
            { label: 'Skor', value: formatScore(open.score) },
            { label: 'Bölümleme', value: open.chunking_mode ? humanise(open.chunking_mode) : '—' },
          ]}
          onClose={() => setOpen(null)}
        />
      ) : null}
    </div>
  );
}

/** A 400 here usually means the profile cannot serve the method that was asked for. */
function SearchError({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.type === 'invalid_request') {
    const supported = error.details?.supported;
    return (
      <InlineError>
        {error.message}
        {Array.isArray(supported) ? ` Desteklenen: ${supported.join(', ')}.` : ''}
      </InlineError>
    );
  }
  const explanation = explain(error);
  return <Failed error={error} title={explanation.title} />;
}
