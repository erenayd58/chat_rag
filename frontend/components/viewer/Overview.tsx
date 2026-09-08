'use client';

/**
 * Genel — what is in the system, what is ready, and where to go next.
 *
 * Three questions and nothing else. Every number is counted from what the
 * contract answered: the knowledge bases from `GET /api/v1/knowledge-bases`,
 * the documents and their analysis state from `GET /api/v1/documents`, which
 * carries an `analysis` block per row precisely so a screen like this does not
 * need a request per document.
 *
 * A document still being packaged is shown as such and stays clickable — the
 * screen it opens says where the build got to and updates when a method
 * becomes ready — because hiding it would leave a reader who just uploaded
 * something staring at a list that does not contain their file.
 */

import type { ChunkingMethod, DocumentWithAnalysis, KnowledgeBase } from '@/types/api';
import { Marks } from './Menu';
import type { Mode } from './Bar';

export interface QueryRun {
  question: string;
  method: string;
  sources: number | null;
  at: number;
}

const STATE_CHIP: Record<string, { tone: string; text: string }> = {
  ready: { tone: 'ok', text: 'hazır' },
  running: { tone: 'run', text: 'işleniyor' },
  pending: { tone: 'wait', text: 'kuyrukta' },
  failed: { tone: 'err', text: 'hata' },
  missing: { tone: 'wait', text: 'analiz yok' },
};

export function relativeTime(at: number): string {
  const seconds = (Date.now() - at) / 1000;
  if (seconds < 90) return 'az önce';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} dk önce`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} sa önce`;
  return hours < 48 ? 'dün' : `${Math.round(hours / 24)} gün önce`;
}

export function Overview({
  knowledgeBases,
  documents,
  kbId,
  catalogue,
  history,
  onKb,
  onDocument,
  onMode,
  onRerun,
  onRefresh,
  loading,
}: {
  knowledgeBases: KnowledgeBase[];
  documents: DocumentWithAnalysis[];
  kbId: string;
  catalogue: ChunkingMethod[];
  history: QueryRun[];
  onKb: (kbId: string) => void;
  onDocument: (documentId: string) => void;
  onMode: (mode: Mode) => void;
  onRerun: (run: QueryRun) => void;
  onRefresh: () => void;
  loading: boolean;
}) {
  const ready = documents.filter((entry) => entry.analysis.status === 'ready');
  const chunks = documents.reduce((total, entry) => total + (entry.chunk_count || 0), 0);
  const variants = documents.reduce(
    (total, entry) => total + entry.analysis.ready_methods.length,
    0,
  );

  const stats: [string, string | number, string][] = [
    [
      'Bilgi tabanı',
      knowledgeBases.length,
      kbId ? 'seçili bilgi tabanı okunuyor' : 'bir tanesini seçin',
    ],
    ['Doküman', documents.length, kbId ? 'seçili bilgi tabanında' : 'bir bilgi tabanı seçin'],
    ['Hazır analiz', ready.length, "İncele ve Sorgu'ya açık"],
    [
      'Parça',
      chunks.toLocaleString('tr-TR'),
      `indekslenmiş · ${variants} analiz varyantı`,
    ],
  ];

  const recent = [...documents].sort(
    (a, b) => Date.parse(b.ingested_at ?? '') - Date.parse(a.ingested_at ?? ''),
  );

  return (
    <div className="home">
      <div>
        <div className="kicker">Chunk Viewer</div>
        <h1 className="ql1">Genel bakış</h1>
        <p className="hsub">
          Bir dokümanın hangi yöntemle nasıl parçalandığını sayfanın üzerinde inceleyin, aynı
          içerikte yöntemleri karşılaştırın, dokümana soru sorun.
        </p>
      </div>

      <div className="statrow">
        <Marks />
        {stats.map(([key, value, note]) => (
          <div className="stat" key={key}>
            <div className="k">{key}</div>
            <div className="v">{value}</div>
            <div className="s">{note}</div>
          </div>
        ))}
      </div>

      <div className="hpanels">
        <div className="hpanel">
          <Marks />
          <div className="hphead">
            <span className="t">Bilgi tabanları</span>
            <button type="button" onClick={onRefresh}>
              Yenile
            </button>
          </div>
          {knowledgeBases.length ? (
            knowledgeBases.map((entry) => {
              const own = entry.id === kbId ? documents : null;
              return (
                <button
                  key={entry.id ?? ''}
                  type="button"
                  className="hrow"
                  onClick={() => {
                    onKb(entry.id ?? '');
                    onMode('incele');
                  }}
                >
                  <span className="n">{entry.name || entry.id}</span>
                  <span className="m">
                    {own
                      ? `${own.length} doküman · ${ready.length} hazır`
                      : (entry.retrieval_method ?? 'bilgi tabanı')}
                  </span>
                  <span className={`schip ${own && ready.length ? 'ok' : 'wait'}`}>
                    {entry.id === kbId ? 'seçili' : 'aç'}
                  </span>
                </button>
              );
            })
          ) : loading ? (
            <div className="hquiet">
              <span className="spin" />
              Bilgi tabanları okunuyor…
            </div>
          ) : (
            <div className="hquiet">Bu konsolda bilgi tabanı yok.</div>
          )}
        </div>

        <div className="hpanel">
          <Marks />
          <div className="hphead">
            <span className="t">Son eklenenler</span>
            <span className="r">
              {recent.length ? `${Math.min(6, recent.length)} / ${recent.length}` : ''}
            </span>
          </div>
          {!kbId ? (
            <div className="hquiet">Dokümanları görmek için bir bilgi tabanı seçin.</div>
          ) : recent.length ? (
            recent.slice(0, 6).map((entry) => {
              const chip = STATE_CHIP[entry.analysis.status] ?? STATE_CHIP.missing;
              const at = Date.parse(entry.ingested_at ?? '');
              return (
                <button
                  key={entry.id ?? ''}
                  type="button"
                  className="hrow"
                  onClick={() => {
                    onDocument(entry.id ?? '');
                    onMode('incele');
                  }}
                >
                  <span className="n">{entry.name}</span>
                  <span className="m">
                    {entry.analysis.ready_methods
                      .map(
                        (key) =>
                          catalogue.find((method) => method.key === key)?.label ?? key,
                      )
                      .join(', ') || (Number.isNaN(at) ? '' : relativeTime(at))}
                  </span>
                  <span className={`schip ${chip.tone}`}>{chip.text}</span>
                </button>
              );
            })
          ) : (
            <div className="hquiet">Bu bilgi tabanında doküman yok.</div>
          )}
        </div>
      </div>

      {history.length ? (
        <>
          <div className="qsect2" style={{ marginTop: 34 }}>
            Son sorgular
          </div>
          <div className="qhist">
            {history.map((run, index) => (
              <button
                key={`${run.at}-${index}`}
                type="button"
                className="qhrow"
                onClick={() => onRerun(run)}
              >
                <span className="hq">{run.question}</span>
                <span className="hm">{run.method}</span>
                <span className="hn">{run.sources === null ? '' : `${run.sources} parça`}</span>
                <span className="ht">{relativeTime(run.at)}</span>
              </button>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
