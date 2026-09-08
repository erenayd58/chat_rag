'use client';

/**
 * Sorgu — one question, put to the chunkers rather than to the corpus.
 *
 * This is the screen the whole `POST /api/v1/analysis-queries` endpoint exists
 * for. Chat asks a knowledge base, which has one chunker and therefore one
 * answer. Here the same question runs through several chunking methods of one
 * document at once, over indexes built from the analysis's own rows, and what
 * comes back is a comparison: the same answer from four cuttings, the sources
 * each one found, and how much of the evidence they agreed on.
 *
 * Two scopes:
 *
 * * **one document** — every selected method runs, side by side. One request.
 * * **the whole knowledge base** — one method, every ready document. That is a
 *   loop over the same endpoint with `answer: false` to find which document
 *   holds the evidence, and then one answered request against the best. No
 *   merged answer pretending to come from one document, and no second endpoint
 *   invented to avoid the loop.
 *
 * A method chip is a key from the deployment's registry intersected with what
 * a document actually has ready; there is no method name in this file.
 */

import { useEffect, useRef, useState } from 'react';
import api from '@/lib/api';
import { errorMessage } from '@/lib/errors';
import { formatPages, sourceSection } from '@/lib/viewer/model';
import type {
  AnalysisArmResult,
  AnalysisQueryResult,
  AnalysisSource,
  ChunkingMethod,
  DocumentWithAnalysis,
  KnowledgeBase,
  ModelChain,
} from '@/types/api';
import { Marks, anchorOf, type Anchor } from './Menu';
import { relativeTime, type QueryRun } from './Overview';

const TOP_K = [3, 5, 8, 10];

type Scope = { kind: 'kb' } | { kind: 'doc'; id: string; name: string };

export interface QueryProps {
  knowledgeBase: KnowledgeBase | null;
  documents: DocumentWithAnalysis[];
  documentId: string;
  catalogue: ChunkingMethod[];
  models: ModelChain | null;
  history: QueryRun[];
  /** A question handed over from another screen, run once. */
  seed: string;
  onRemember: (run: QueryRun) => void;
  /** Open a retrieved chunk on the İncele board. */
  onJump: (method: string, chunkId: string) => boolean;
  onSource: (source: AnalysisSource, methodLabel: string, anchor: Anchor) => void;
}

export function Query(props: QueryProps) {
  const [text, setText] = useState(props.seed);
  const [scope, setScope] = useState<Scope>(() => initialScope(props));
  const [methods, setMethods] = useState<string[]>([]);
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [result, setResult] = useState<AnalysisQueryResult | null>(null);
  const [kbResult, setKbResult] = useState<KbResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  // A new question supersedes any still running.
  const run = useRef(0);

  const label = (key: string) =>
    props.catalogue.find((method) => method.key === key)?.label ?? key;

  const candidates = props.documents
    .filter((entry) => entry.analysis.status === 'ready' && entry.analysis.ready_methods.length)
    .map((entry) => ({
      id: entry.id ?? '',
      name: entry.name,
      methods: entry.analysis.ready_methods,
    }));

  const offered = offeredMethods(props.catalogue, candidates, scope);

  // Keep the selection inside what the current scope can actually run, and
  // never leave it empty: a screen with no method selected has no question.
  useEffect(() => {
    setMethods((current) => {
      const kept = current.filter((method) => offered.includes(method));
      if (scope.kind === 'kb') return kept.length ? [kept[0]] : offered.slice(0, 1);
      return kept.length ? kept : offered.slice(0, 1);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offered.join('|'), scope.kind, scope.kind === 'doc' ? scope.id : '']);

  useEffect(() => {
    if (props.seed) setText(props.seed);
  }, [props.seed]);

  const ask = async () => {
    const question = text.trim();
    if (!question || busy || !methods.length) return;
    const ticket = (run.current += 1);
    setBusy(true);
    setFailure(null);
    setResult(null);
    setKbResult(null);
    setProgress(null);

    try {
      if (scope.kind === 'doc') {
        const found = await api.analysis.query({
          document_id: scope.id,
          question,
          methods,
          top_k: topK,
        });
        if (ticket !== run.current) return;
        setResult(found);
        props.onRemember({
          question,
          method:
            methods.length > 1
              ? `Karşılaştırma · ${methods.length} yöntem`
              : label(methods[0]),
          sources: found.arms[0]?.sources.length ?? 0,
          at: Date.now(),
        });
      } else {
        const found = await askKnowledgeBase({
          question,
          method: methods[0],
          candidates,
          topK,
          onProgress: (done, total) => {
            if (ticket === run.current) setProgress({ done, total });
          },
          alive: () => ticket === run.current,
        });
        if (ticket !== run.current) return;
        setKbResult(found);
        props.onRemember({
          question,
          method: `Bilgi tabanı · ${label(methods[0])}`,
          sources: found.best?.result.arms[0]?.sources.length ?? 0,
          at: Date.now(),
        });
      }
    } catch (cause) {
      if (ticket !== run.current) return;
      setFailure(errorMessage(cause));
    } finally {
      if (ticket === run.current) {
        setBusy(false);
        setProgress(null);
      }
    }
  };

  if (!props.knowledgeBase) {
    return (
      <div className="hero">
        <div className="kicker">Sorgu</div>
        <h1>Bilgi tabanı gerekli</h1>
        <p>Soru sormak için önce bir bilgi tabanı seçin.</p>
      </div>
    );
  }

  const kbName = props.knowledgeBase.name || props.knowledgeBase.id || '';
  const comparing = scope.kind === 'doc' && methods.length > 1;

  return (
    <div className="qwrap">
      <div className="qmaincol">
        <div className="kicker">
          {scope.kind === 'kb'
            ? `Sorgu · ${kbName}`
            : `Sorgu · ${comparing ? `${methods.length} yöntem` : label(methods[0] ?? '')}`}
        </div>
        <h1 className="ql1">
          {scope.kind === 'kb' ? 'Bilgi tabanına sor' : 'Bu dokümana sor'}
        </h1>
        <p className="qsub">
          {scope.kind === 'kb'
            ? `${kbName} içindeki ${candidates.length} hazır doküman aranır; en iyi eşleşen doküman cevaplanır.`
            : `${scope.name} · aynı soru seçili her yöntemin parçaları üzerinde koşar; yalnız chunker değişir.`}
        </p>

        <div className="qpanel">
          <Marks />
          <label className="qlab" htmlFor="v-question">
            Soru
          </label>
          <textarea
            id="v-question"
            className="qin"
            rows={3}
            value={text}
            placeholder={
              scope.kind === 'kb'
                ? 'Bu bilgi tabanına doğal dilde bir soru sorun…'
                : 'Bu dokümana doğal dilde bir soru sorun…'
            }
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                ask();
              }
            }}
          />
          <div className="qctl">
            <label className="qf">
              <span>Kapsam</span>
              <select
                value={scope.kind === 'kb' ? '__kb' : scope.id}
                onChange={(event) => {
                  const value = event.target.value;
                  const found = candidates.find((entry) => entry.id === value);
                  setScope(
                    value === '__kb' || !found
                      ? { kind: 'kb' }
                      : { kind: 'doc', id: found.id, name: found.name },
                  );
                  setResult(null);
                  setKbResult(null);
                  setFailure(null);
                }}
              >
                <option value="__kb">Tüm bilgi tabanı</option>
                {candidates.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="qf qmths">
              <span>{scope.kind === 'kb' ? 'Yöntem' : 'Yöntemler'}</span>
              <div className="qmchips">
                {offered.map((method) => (
                  <button
                    key={method}
                    type="button"
                    className={`mchip${methods.includes(method) ? ' on' : ''}`}
                    aria-pressed={methods.includes(method)}
                    onClick={() =>
                      setMethods((current) => toggle(current, method, scope.kind === 'kb'))
                    }
                  >
                    {label(method)}
                  </button>
                ))}
                {offered.length ? null : (
                  <span className="qmnote">Bu kapsamda hazır analiz yok.</span>
                )}
              </div>
            </div>
          </div>
          <div className="qbottom">
            <span className="qmnote">
              {comparing
                ? `${methods.length} yöntem karşılaştırılır`
                : scope.kind === 'kb'
                  ? 'Bilgi tabanı aramasında tek yöntem koşar'
                  : ''}
            </span>
            <button
              type="button"
              className="primary qgo"
              disabled={busy || !offered.length}
              onClick={ask}
            >
              Sor
            </button>
          </div>
        </div>

        <div className="qout">
          {busy ? (
            <div className="qnote">
              <span className="spin" />
              {progress
                ? `Bilgi tabanı aranıyor · ${progress.done}/${progress.total} doküman`
                : comparing
                  ? 'Aynı soru seçili yöntemlerle koşuluyor…'
                  : 'Cevap aranıyor…'}
            </div>
          ) : failure ? (
            <div className="qnote err">{failure}</div>
          ) : kbResult ? (
            <KbAnswer
              found={kbResult}
              label={label}
              onAskDocument={(id, name) => {
                setScope({ kind: 'doc', id, name });
                setKbResult(null);
              }}
              onSource={props.onSource}
              onJump={props.onJump}
            />
          ) : result ? (
            <Comparison
              result={result}
              label={label}
              onSource={props.onSource}
              onJump={props.onJump}
              single={result.arms.length === 1}
            />
          ) : null}
        </div>

        {props.history.length ? (
          <>
            <div className="qsect2">Son sorgular</div>
            <div className="qhist">
              {props.history.map((entry, index) => (
                <button
                  key={`${entry.at}-${index}`}
                  type="button"
                  className="qhrow"
                  onClick={() => setText(entry.question)}
                >
                  <span className="hq">{entry.question}</span>
                  <span className="hm">{entry.method}</span>
                  <span className="hn">
                    {entry.sources === null ? '' : `${entry.sources} parça`}
                  </span>
                  <span className="ht">{relativeTime(entry.at)}</span>
                </button>
              ))}
            </div>
          </>
        ) : null}
      </div>

      <aside className="qside">
        <div className="qcard">
          <Marks />
          <span className="livechip">canlı</span>
          <h3>Parametreler</h3>
          <label className="qf qcf">
            <span>Top-k</span>
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {TOP_K.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <dl className="qkv">
            <dt>Embedding</dt>
            <dd>{result?.embedding_model ?? chainValue(props.models, 'embedding') ?? '—'}</dd>
            <dt>Cevap modeli</dt>
            <dd>{result?.answer_model ?? chainValue(props.models, 'answer') ?? '—'}</dd>
            <dt>Kapsam</dt>
            <dd>{scope.kind === 'kb' ? `Tüm bilgi tabanı · ${kbName}` : scope.name}</dd>
            <dt>Yöntem</dt>
            <dd>
              {scope.kind === 'kb'
                ? `${label(methods[0] ?? '')} · ${
                    candidates.filter((entry) => entry.methods.includes(methods[0])).length
                  } doküman aranır`
                : methods.map(label).join(', ') || '—'}
            </dd>
            <dt>Kaynak</dt>
            <dd>{kbName}</dd>
          </dl>
        </div>
      </aside>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The answers                                                         */
/* ------------------------------------------------------------------ */

function Comparison({
  result,
  label,
  single,
  onSource,
  onJump,
}: {
  result: AnalysisQueryResult;
  label: (key: string) => string;
  single: boolean;
  onSource: QueryProps['onSource'];
  onJump: QueryProps['onJump'];
}) {
  if (single) return <Arm arm={result.arms[0]} label={label} onJump={onJump} />;
  return (
    <>
      {result.arms.map((arm) => (
        <div className="qcmpcol" key={arm.method}>
          <div className="qsect" style={{ marginTop: 0 }}>
            {label(arm.method)}
            {arm.unit_overlap === null ? null : (
              <span className="ov">
                diğer yöntemlerle örtüşme %{Math.round(arm.unit_overlap * 100)}
              </span>
            )}
          </div>
          {arm.error ? <div className="qnote warn">{arm.error}</div> : null}
          {arm.answer?.text ? <div className="qans sm">{arm.answer.text}</div> : null}
          {arm.sources.length ? (
            <div className="qmini">
              {arm.sources.map((source, index) => (
                <button
                  key={`${arm.method}-${index}`}
                  type="button"
                  className="qminisrc"
                  onClick={(event) =>
                    onSource(source, label(arm.method), anchorOf(event.currentTarget))
                  }
                >
                  <span className="slab">{source.label}</span> {formatPages(source.pages)} ·{' '}
                  {source.token_count} tk{source.used ? ' · kullanıldı' : ''}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ))}
    </>
  );
}

/** One method's answer, with its sources spelled out. */
function Arm({
  arm,
  label,
  onJump,
}: {
  arm: AnalysisArmResult | undefined;
  label: (key: string) => string;
  onJump: QueryProps['onJump'];
}) {
  if (!arm) return null;
  return (
    <>
      {arm.error ? <div className="qnote warn">{arm.error}</div> : null}
      {arm.answer?.text ? <div className="qans">{arm.answer.text}</div> : null}
      {arm.answer?.sufficient === false ? (
        <div className="qnote warn" style={{ marginTop: 10 }}>
          Model, kaynakları bu soru için yetersiz buldu.
        </div>
      ) : null}
      {arm.note ? (
        <div className="qnote warn" style={{ marginTop: 10 }}>
          {arm.note}
        </div>
      ) : null}
      {arm.sources.length ? (
        <>
          <div className="qsect">Kaynaklar · {label(arm.method)}</div>
          {arm.sources.map((source, index) => (
            <SourceRow
              key={`${arm.method}-${index}`}
              source={source}
              method={arm.method}
              onJump={onJump}
            />
          ))}
        </>
      ) : arm.error ? null : (
        <div className="qnote">Bu soruya kaynak bulunamadı.</div>
      )}
    </>
  );
}

function SourceRow({
  source,
  method,
  onJump,
}: {
  source: AnalysisSource;
  method: string;
  onJump: QueryProps['onJump'];
}) {
  const [open, setOpen] = useState(false);
  const section = sourceSection(source);
  return (
    <div className={`qsrc${source.used ? ' used' : ''}`}>
      <button type="button" className="qsrchead" onClick={() => setOpen((value) => !value)}>
        <span className="slab">{source.label}</span>
        <span className="sinfo">{section || source.chunk_id}</span>
        <span className="smeta">
          {formatPages(source.pages)} · {source.token_count} tk
          {source.used ? ' · cevapta kullanıldı' : ''}
        </span>
      </button>
      {open ? (
        <div className="qsrcbody">
          <div className="qtxt">{source.text ?? ''}</div>
          <button
            type="button"
            className="qjump"
            onClick={() => {
              if (!source.chunk_id) return;
              onJump(source.arm ?? method, source.chunk_id);
            }}
          >
            İncele görünümünde aç
          </button>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Asking a whole knowledge base                                       */
/* ------------------------------------------------------------------ */

interface Candidate {
  id: string;
  name: string;
  methods: string[];
}

interface KbResult {
  question: string;
  method: string;
  best: { document: Candidate; result: AnalysisQueryResult } | null;
  others: { document: Candidate; source: AnalysisSource }[];
  skipped: number;
  failed: number;
}

/**
 * Every ready document searched with the chosen method, then the best one
 * answered.
 *
 * Retrieval first (`answer: false`) so the answer model is asked exactly once,
 * about the document that actually holds the evidence. The runners-up stay one
 * click away rather than being merged into an answer no single document
 * supports.
 */
async function askKnowledgeBase({
  question,
  method,
  candidates,
  topK,
  onProgress,
  alive,
}: {
  question: string;
  method: string;
  candidates: Candidate[];
  topK: number;
  onProgress: (done: number, total: number) => void;
  alive: () => boolean;
}): Promise<KbResult> {
  const able = candidates.filter((entry) => entry.methods.includes(method));
  const skipped = candidates.length - able.length;
  const found: { document: Candidate; score: number; sources: AnalysisSource[] }[] = [];
  let failed = 0;

  for (let index = 0; index < able.length; index += 1) {
    if (!alive()) break;
    const document = able[index];
    try {
      const searched = await api.analysis.query({
        document_id: document.id,
        question,
        methods: [method],
        top_k: topK,
        answer: false,
      });
      const sources = searched.arms[0]?.sources ?? [];
      if (sources.length) {
        found.push({ document, score: scoreOf(sources[0]), sources });
      }
    } catch {
      failed += 1;
    }
    onProgress(index + 1, able.length);
  }

  found.sort((a, b) => b.score - a.score);
  if (!found.length) {
    return { question, method, best: null, others: [], skipped, failed };
  }

  const winner = found[0];
  const answered = await api.analysis.query({
    document_id: winner.document.id,
    question,
    methods: [method],
    top_k: topK,
  });
  return {
    question,
    method,
    best: { document: winner.document, result: answered },
    others: found.slice(1, 6).map((entry) => ({
      document: entry.document,
      source: entry.sources[0],
    })),
    skipped,
    failed,
  };
}

/**
 * How well a document matched, from the source the engine ranked first.
 *
 * Retrieval scores are comparable only within one answer, so this is a
 * ranking heuristic and nothing more: the top source's own score where the
 * engine reported one, and its position otherwise.
 */
function scoreOf(source: AnalysisSource): number {
  const score = source.score ?? source.rrf_score;
  return typeof score === 'number' ? score : 0;
}

function KbAnswer({
  found,
  label,
  onAskDocument,
  onSource,
  onJump,
}: {
  found: KbResult;
  label: (key: string) => string;
  onAskDocument: (id: string, name: string) => void;
  onSource: QueryProps['onSource'];
  onJump: QueryProps['onJump'];
}) {
  if (!found.best) {
    return (
      <div className="qnote">
        Bu soruya bilgi tabanında kaynak bulunamadı.
        {found.skipped
          ? ` (${found.skipped} doküman '${label(found.method)}' yöntemiyle hazır olmadığı için aranmadı.)`
          : ''}
      </div>
    );
  }
  return (
    <>
      <div className="qsect" style={{ marginTop: 0 }}>
        En iyi eşleşme · {found.best.document.name}
      </div>
      <Arm arm={found.best.result.arms[0]} label={label} onJump={onJump} />
      {found.others.length ? (
        <>
          <div className="qsect">Diğer eşleşen dokümanlar</div>
          {found.others.map((other) => (
            <div className="qsrc" key={other.document.id}>
              <div className="qsrchead">
                <span className="sinfo">{other.document.name}</span>
                <span className="smeta">
                  {sourceSection(other.source)
                    ? `${sourceSection(other.source)} · `
                    : ''}
                  {formatPages(other.source.pages)}
                </span>
                <button
                  type="button"
                  className="qjump"
                  onClick={() => onAskDocument(other.document.id, other.document.name)}
                >
                  Bu dokümanda cevapla
                </button>
              </div>
            </div>
          ))}
        </>
      ) : null}
      {found.skipped ? (
        <div className="qoff" style={{ marginTop: 12 }}>
          {found.skipped} doküman &apos;{label(found.method)}&apos; yöntemiyle hazır olmadığı için
          aranmadı.
        </div>
      ) : null}
      {found.failed ? <div className="qoff">{found.failed} doküman aranamadı.</div> : null}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Small decisions                                                     */
/* ------------------------------------------------------------------ */

function initialScope(props: QueryProps): Scope {
  const current = props.documents.find((entry) => entry.id === props.documentId);
  if (current && current.analysis.status === 'ready') {
    return { kind: 'doc', id: current.id ?? '', name: current.name };
  }
  return { kind: 'kb' };
}

/** The methods this scope can actually run, in the registry's order. */
function offeredMethods(
  catalogue: ChunkingMethod[],
  candidates: Candidate[],
  scope: Scope,
): string[] {
  const order = catalogue.map((method) => method.key);
  if (scope.kind === 'doc') {
    const found = candidates.find((entry) => entry.id === scope.id);
    return found ? order.filter((key) => found.methods.includes(key)) : [];
  }
  const seen = new Set<string>();
  for (const entry of candidates) for (const method of entry.methods) seen.add(method);
  return order.filter((key) => seen.has(key));
}

/** A knowledge-base search runs one method; a document compares several. */
function toggle(current: string[], method: string, single: boolean): string[] {
  if (single) return [method];
  if (current.includes(method)) {
    return current.length > 1 ? current.filter((value) => value !== method) : current;
  }
  return [...current, method];
}

function chainValue(models: ModelChain | null, key: string): string | null {
  const chain = models?.chain as Record<string, any> | undefined;
  const entry = chain?.[key];
  if (!entry) return null;
  if (typeof entry === 'string') return entry;
  return entry.model ?? entry.name ?? entry.model_name ?? null;
}
