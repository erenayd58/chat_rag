'use client';

/**
 * Viewer — the chunking screens, inside the console.
 *
 * This used to be a second server on `:8765` serving one HTML file, relaying
 * everything it needed off this console's Flask-era demo routes. It is a
 * screen now: five tabs over `/api/v1`, the same navigation and the same
 * knowledge-base selection as every other screen, and no second process.
 *
 * What this file owns is the state the five screens share and nothing else —
 * which knowledge base, which document, which screen, which methods are on the
 * board, which page, which chunk is open. The data is in
 * `lib/viewer/useAnalysis.ts`, the alignment in `lib/viewer/rows.ts`, and each
 * tab is its own component.
 *
 * **No method name appears anywhere under this route.** The catalogue is
 * `GET /api/v1/meta/chunking-methods`, a projection of the library's registry;
 * which methods a document has is its analysis's `ready_methods`; and the two
 * questions the screens ask about a method — is it an orchestration, and what
 * does it orchestrate — are the catalogue's own `orchestration` and `baseline`
 * fields. A chunker added to the library appears here with nothing changed.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Bar, type Mode } from '@/components/viewer/Bar';
import { Benchmark } from '@/components/viewer/Benchmark';
import { Board, type Selection } from '@/components/viewer/Board';
import { ChunkCard, SourceCard } from '@/components/viewer/ChunkCard';
import { Debug } from '@/components/viewer/Debug';
import type { Anchor } from '@/components/viewer/Menu';
import { Overview, type QueryRun } from '@/components/viewer/Overview';
import { Query } from '@/components/viewer/Query';
import { useChunkingMethods } from '@/components/MethodPicker';
import { Failed } from '@/components/ui';
import api from '@/lib/api';
import { useAsync } from '@/lib/hooks/useAsync';
import { useSelectedKb } from '@/lib/hooks/useSelectedKb';
import { methodsOf } from '@/lib/viewer/model';
import { buildBoard } from '@/lib/viewer/rows';
import { useAnalysis } from '@/lib/viewer/useAnalysis';
import type {
  AnalysisSource,
  ChunkingMethod,
  DocumentWithAnalysis,
  KnowledgeBase,
  ModelChain,
} from '@/types/api';
import './viewer.css';

/** How many methods fit on the board before it stops being a comparison. */
const MAX_COLUMNS = 3;
/** How many questions the session remembers. Never persisted anywhere. */
const HISTORY = 6;

type Card =
  | { kind: 'chunk'; method: string; chunk: number; anchor: Anchor }
  | { kind: 'source'; source: AnalysisSource; methodLabel: string; anchor: Anchor };

export default function ViewerPage() {
  const [kbId, setKbId] = useSelectedKb();
  const [documentId, setDocumentId] = useState('');
  const [mode, setMode] = useState<Mode>('home');
  const [selected, setSelected] = useState<string[]>([]);
  const [page, setPage] = useState<number | null>(null);
  const [differenceIndex, setDifferenceIndex] = useState(-1);
  const [card, setCard] = useState<Card | null>(null);
  const [scrollTo, setScrollTo] = useState<{ row?: number; method?: string; chunk?: number } | null>(
    null,
  );
  const [history, setHistory] = useState<QueryRun[]>([]);
  const [seed, setSeed] = useState('');

  const catalogue = useChunkingMethods();
  const knowledgeBases = useAsync<KnowledgeBase[]>(
    (signal) => api.knowledgeBases.list(signal),
    [],
  );
  const documents = useAsync<DocumentWithAnalysis[]>(
    (signal) => (kbId ? api.documents.list(kbId, signal) : Promise.resolve([])),
    [kbId],
  );
  const models = useAsync<ModelChain | null>(
    (signal) => api.meta.models(kbId || null, signal).catch(() => null),
    [kbId],
  );
  const analysis = useAnalysis(documentId);

  const methods = catalogue.data ?? [];
  const order = useMemo(() => methods.map((method) => method.key), [methods]);
  const doc = analysis.doc;
  // What this document can be looked at with: the registry's order, narrowed
  // to the arms the payload actually carries.
  const available = useMemo(() => methodsOf(doc, order), [doc, order]);

  // A document that vanished from the knowledge base is not still open.
  useEffect(() => {
    const rows = documents.data;
    if (!rows || !documentId) return;
    if (!rows.some((entry) => entry.id === documentId)) setDocumentId('');
  }, [documents.data, documentId]);

  // Opening a document clears the board: its methods, its pages and any card
  // belong to the previous one.
  useEffect(() => {
    setSelected([]);
    setPage(null);
    setDifferenceIndex(-1);
    setCard(null);
  }, [documentId]);

  // The board opens on the first available method, so a reader who picked a
  // document sees the document rather than another chooser. A method that
  // finished packaging while the page was open is offered, not forced.
  useEffect(() => {
    if (!available.length) return;
    setSelected((current) => {
      const kept = current.filter((method) => available.includes(method));
      return kept.length ? kept : available.slice(0, 1);
    });
  }, [available.join('|')]);

  const board = useMemo(() => buildBoard(doc, selected), [doc, selected]);
  const pages = doc?.pages ?? [];

  useEffect(() => {
    if (!pages.length) return;
    setPage((current) => (current !== null && pages.includes(current) ? current : pages[0]));
  }, [pages.join(',')]);

  const remember = useCallback((run: QueryRun) => {
    setHistory((current) => [run, ...current].slice(0, HISTORY));
  }, []);

  const toggleMethod = (method: string) => {
    setDifferenceIndex(-1);
    setSelected((current) => {
      if (current.includes(method)) {
        return current.length > 1 ? current.filter((value) => value !== method) : current;
      }
      return current.length >= MAX_COLUMNS ? current : [...current, method];
    });
  };

  const stepDifference = (delta: number) => {
    if (!board.differences.length) return;
    const next =
      differenceIndex < 0
        ? delta > 0
          ? 0
          : board.differences.length - 1
        : (differenceIndex + delta + board.differences.length) % board.differences.length;
    setDifferenceIndex(next);
    const row = board.rows[board.differences[next]];
    if (row.unit.p !== page) setPage(row.unit.p);
    setCard(null);
    setScrollTo({ row: row.index });
  };

  /** Open a retrieved chunk on the board. False when it is not in this payload. */
  const jump = useCallback(
    (method: string, chunkId: string): boolean => {
      const arm = doc?.arms?.[method];
      if (!arm) return false;
      const index = arm.chunks.findIndex((chunk) => chunk.id === chunkId);
      if (index < 0) return false;
      const chunk = arm.chunks[index];
      setSelected((current) => (current.includes(method) ? current : [method]));
      setMode('incele');
      setPage(chunk.pg?.[0] ?? doc!.pages[0]);
      setCard(null);
      setScrollTo({ method, chunk: index });
      return true;
    },
    [doc],
  );

  const jumpToPage = useCallback(
    (target: number) => {
      setMode('incele');
      setPage(target);
      setCard(null);
      setScrollTo(null);
      window.scrollTo({ top: 0 });
    },
    [],
  );

  // Arrow keys page through the document; n / p step the differences. Only on
  // the board, and never while someone is typing.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
      if (event.key === 'Escape') {
        setCard(null);
        return;
      }
      if (mode !== 'incele' || !doc || !selected.length || page === null) return;
      const at = pages.indexOf(page);
      if (event.key === 'ArrowLeft' && at > 0) setPage(pages[at - 1]);
      else if (event.key === 'ArrowRight' && at < pages.length - 1) setPage(pages[at + 1]);
      else if (event.key === 'n') stepDifference(1);
      else if (event.key === 'p') stepDifference(-1);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const labelOf = (key: string) => methods.find((method) => method.key === key)?.label ?? key;
  // Which method is an orchestration, and over what, is the registry's answer
  // and not this screen's. Both the Debug pipeline and the Benchmark comparison
  // are built from it, so a second orchestration would work with no edit here.
  const orchestration =
    methods.find((method) => method.orchestration && available.includes(method.key))?.key ?? null;
  const baseline =
    methods.find((method) => method.key === orchestration)?.baseline ?? null;

  const currentKb = (knowledgeBases.data ?? []).find((entry) => entry.id === kbId) ?? null;
  const currentDocument =
    (documents.data ?? []).find((entry) => entry.id === documentId) ?? null;

  return (
    <div className="v-root">
      <Bar
        knowledgeBases={knowledgeBases.data ?? []}
        kbId={kbId}
        onKb={(next) => {
          setKbId(next);
          setDocumentId('');
        }}
        documents={documents.data ?? []}
        documentId={documentId}
        onDocument={setDocumentId}
        mode={mode}
        onMode={(next) => {
          setMode(next);
          setCard(null);
        }}
        catalogue={methods}
        doc={doc}
        methods={available}
        selected={selected}
        onToggleMethod={toggleMethod}
        pages={pages}
        page={page}
        onPage={(next) => {
          setPage(next);
          setCard(null);
          window.scrollTo({ top: 0 });
        }}
        differences={board.differences.length}
        differenceIndex={differenceIndex}
        onStepDifference={stepDifference}
      />

      <main className="v-stage">
        {mode === 'home' ? (
          <Overview
            knowledgeBases={knowledgeBases.data ?? []}
            documents={documents.data ?? []}
            kbId={kbId}
            catalogue={methods}
            history={history}
            onKb={(next) => {
              setKbId(next);
              setDocumentId('');
            }}
            onDocument={setDocumentId}
            onMode={setMode}
            onRerun={(run) => {
              setSeed(run.question);
              setMode('sorgu');
            }}
            onRefresh={() => {
              knowledgeBases.reload();
              documents.reload();
            }}
            loading={knowledgeBases.loading}
          />
        ) : mode === 'sorgu' ? (
          <Query
            knowledgeBase={currentKb}
            documents={documents.data ?? []}
            documentId={documentId}
            catalogue={methods}
            models={models.data ?? null}
            history={history}
            seed={seed}
            onRemember={remember}
            onJump={jump}
            onSource={(source, methodLabel, anchor) =>
              setCard({ kind: 'source', source, methodLabel, anchor })
            }
          />
        ) : (
          <Stage
            kbId={kbId}
            documentId={documentId}
            documentName={currentDocument?.name ?? ''}
            analysis={analysis}
            mode={mode}
            selected={selected}
            available={available}
            catalogue={methods}
            board={board}
            page={page}
            scrollTo={scrollTo}
            orchestration={orchestration}
            baseline={baseline}
            onSelectChunk={(selection, anchor) =>
              setCard(
                selection && anchor
                  ? { kind: 'chunk', method: selection.method, chunk: selection.chunk, anchor }
                  : null,
              )
            }
            selection={
              card?.kind === 'chunk' ? { method: card.method, chunk: card.chunk } : null
            }
            onToggleMethod={toggleMethod}
            onJumpToPage={jumpToPage}
          />
        )}
      </main>

      {card?.kind === 'chunk' && doc ? (
        <ChunkCard
          doc={doc}
          method={card.method}
          methodLabel={labelOf(card.method)}
          index={card.chunk}
          anchor={card.anchor}
          onClose={() => setCard(null)}
        />
      ) : null}
      {card?.kind === 'source' ? (
        <SourceCard
          source={card.source}
          methodLabel={card.methodLabel}
          anchor={card.anchor}
          onClose={() => setCard(null)}
          onJump={
            card.source.chunk_id && card.source.arm && doc?.arms?.[card.source.arm]
              ? () => jump(card.source.arm as string, card.source.chunk_id as string)
              : null
          }
        />
      ) : null}
    </div>
  );
}

/**
 * İncele, Debug and Benchmark all need the same thing first: a document whose
 * analysis is far enough along to show. The three empty states — no knowledge
 * base, no document, an analysis still building — are therefore answered once,
 * here, rather than three times with three wordings.
 */
function Stage({
  kbId,
  documentId,
  documentName,
  analysis,
  mode,
  selected,
  available,
  catalogue,
  board,
  page,
  scrollTo,
  orchestration,
  baseline,
  selection,
  onSelectChunk,
  onToggleMethod,
  onJumpToPage,
}: {
  kbId: string;
  documentId: string;
  documentName: string;
  analysis: ReturnType<typeof useAnalysis>;
  mode: Mode;
  selected: string[];
  available: string[];
  catalogue: ChunkingMethod[];
  board: ReturnType<typeof buildBoard>;
  page: number | null;
  scrollTo: { row?: number; method?: string; chunk?: number } | null;
  orchestration: string | null;
  baseline: string | null;
  selection: Selection | null;
  onSelectChunk: (selection: Selection | null, anchor: Anchor | null) => void;
  onToggleMethod: (method: string) => void;
  onJumpToPage: (page: number) => void;
}) {
  if (!kbId) {
    return (
      <div className="hero">
        <h1>Bir doküman, birden çok parçalama.</h1>
        <p>
          Bilgi tabanını ve dokümanı seçin; her yöntemin metni nerede kestiğini doğrudan sayfanın
          üzerinde görün.
        </p>
        <Steps stage={0} />
      </div>
    );
  }
  if (!documentId) {
    return (
      <div className="hero">
        <h1>Doküman seçin</h1>
        <p>İncelemek istediğiniz dokümanı üstteki kırıntı yolundan seçin.</p>
        <Steps stage={1} />
      </div>
    );
  }
  if (analysis.error) {
    return (
      <div className="hero">
        <Failed error={analysis.error} onRetry={analysis.reload} />
      </div>
    );
  }
  if (!analysis.doc) {
    return (
      <div className="hero">
        <h1>{documentName || documentId}</h1>
        {analysis.status === 'failed' ? (
          <p style={{ color: 'var(--bad)' }}>
            Analiz başarısız: {analysis.state?.error ?? 'bilinmeyen hata'}
          </p>
        ) : analysis.building || analysis.loading ? (
          <p>
            <span className="spin" />
            Analiz hazırlanıyor — doküman yeniden okunmaz, yalnız eksik yöntemler paketlenir.
          </p>
        ) : (
          <p>Bu doküman için hazır bir analiz yok.</p>
        )}
        <Steps stage={1} />
      </div>
    );
  }

  const doc = analysis.doc;

  if (mode === 'bench') {
    return (
      <Benchmark
        doc={doc}
        methods={available}
        catalogue={catalogue}
        orchestration={orchestration}
      />
    );
  }
  if (mode === 'debug') {
    return (
      <Debug
        doc={doc}
        methods={available}
        catalogue={catalogue}
        baseline={baseline}
        onJumpToPage={onJumpToPage}
      />
    );
  }

  if (!selected.length) {
    return (
      <div className="hero">
        <h1>{doc.label}</h1>
        <p>
          {doc.meta.pageCount ? `${doc.meta.pageCount} sayfa · ` : ''}Bir parçalama yöntemi
          seçin — ikincisini seçtiğinizde aynı sayfa yan yana karşılaştırılır.
        </p>
        <Steps stage={2} />
        <div className="mcards">
          {available.map((method) => {
            const entry = catalogue.find((candidate) => candidate.key === method);
            return (
              <button
                key={method}
                type="button"
                className="mcard"
                onClick={() => onToggleMethod(method)}
              >
                <span className="n">{entry?.label ?? method}</span>
                {entry?.summary ? <span className="s">{entry.summary}</span> : null}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (page === null) return null;

  return (
    <Board
      doc={doc}
      methods={selected}
      board={board}
      page={page}
      catalogue={catalogue}
      selection={selection}
      onSelect={onSelectChunk}
      scrollTo={scrollTo}
    />
  );
}

/** Where the reader is in "knowledge base → document → method". */
function Steps({ stage }: { stage: number }) {
  const items = ['Bilgi tabanı', 'Doküman', 'Yöntem'];
  return (
    <div className="steps">
      {items.map((text, index) => (
        <div
          key={text}
          className={`st${index < stage ? ' done' : index === stage ? ' now' : ''}`}
        >
          <b>{index < stage ? '✓' : index + 1}</b>
          {text}
        </div>
      ))}
    </div>
  );
}
