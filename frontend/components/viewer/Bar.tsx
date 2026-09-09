'use client';

/**
 * The Viewer's own bar: where you are, what you are looking at, and where the
 * page you are reading is.
 *
 * Three fixed groups (breadcrumb, screens, navigation) and one open-ended one:
 * a chip per chunking method this document has ready. The method lane is the
 * only group whose size the registry decides, so it is the one that takes the
 * leftover width and scrolls inside it — four methods or fourteen, nothing is
 * clipped and nothing overlaps. **No method name appears in this file**; every
 * chip is a key from `GET /api/v1/meta/chunking-methods` intersected with what
 * this document actually has.
 */

import { useRef, useState } from 'react';
import type { ChunkingMethod, DocumentWithAnalysis, KnowledgeBase } from '@/types/api';
import { deepNote, type ViewerDoc } from '@/lib/viewer/model';
import { Menu, anchorOf, type Anchor } from './Menu';

export type Mode = 'home' | 'incele' | 'sorgu' | 'debug' | 'bench';

export const TABS: { key: Mode; label: string }[] = [
  { key: 'home', label: 'Genel' },
  { key: 'incele', label: 'İncele' },
  { key: 'sorgu', label: 'Sorgu' },
  { key: 'debug', label: 'Debug' },
  { key: 'bench', label: 'Benchmark' },
];

export interface BarProps {
  knowledgeBases: KnowledgeBase[];
  kbId: string;
  onKb: (kbId: string) => void;
  documents: DocumentWithAnalysis[];
  documentId: string;
  onDocument: (documentId: string) => void;
  mode: Mode;
  onMode: (mode: Mode) => void;
  catalogue: ChunkingMethod[];
  doc: ViewerDoc | null;
  methods: string[];
  selected: string[];
  onToggleMethod: (method: string) => void;
  /** Page navigation, shown only on İncele with a board on screen. Empty for
   *  a document whose format carries no page numbers, and `page` is null then:
   *  the board is the whole document and there is nothing to step through. */
  pages: number[];
  page: number | null;
  onPage: (page: number) => void;
  differences: number;
  differenceIndex: number;
  onStepDifference: (delta: number) => void;
}

/** The analysis states a document row shows, as one word each. */
const DOC_STATE: Record<string, { text: string; chip: string; disabled?: boolean }> = {
  ready: { text: '', chip: 'ok' },
  running: { text: 'analiz hazırlanıyor…', chip: 'run', disabled: true },
  pending: { text: 'analiz kuyrukta…', chip: 'wait', disabled: true },
  failed: { text: 'analiz başarısız', chip: 'err' },
  missing: { text: 'analiz yok', chip: 'wait' },
};

export function Bar(props: BarProps) {
  const [menu, setMenu] = useState<{ kind: 'kb' | 'doc'; anchor: Anchor } | null>(null);
  const kbButton = useRef<HTMLButtonElement | null>(null);
  const docButton = useRef<HTMLButtonElement | null>(null);

  const kb = props.knowledgeBases.find((entry) => entry.id === props.kbId) ?? null;
  const document = props.documents.find((entry) => entry.id === props.documentId) ?? null;
  const labels = new Map(props.catalogue.map((method) => [method.key, method.label]));
  const labelOf = (key: string) => labels.get(key) ?? key;

  const ready = props.documents.filter((entry) => entry.analysis.status === 'ready').length;
  const onBoard = props.mode === 'incele' && !!props.doc && props.selected.length > 0;
  const stepsDifferences = props.selected.length > 1 && props.differences > 0;

  return (
    <>
      <header className="v-bar">
        <div className="brand">
          Chunk Viewer<em>v3</em>
        </div>
        <nav className="path">
          <button
            type="button"
            ref={kbButton}
            className={`pick${kb ? '' : ' unset'}`}
            onClick={(event) => setMenu({ kind: 'kb', anchor: anchorOf(event.currentTarget) })}
          >
            <span className="k">Bilgi tabanı</span>
            <span className="v">{kb ? kb.name || kb.id : 'Seç'}</span>
          </button>
          <span className="sep">/</span>
          <button
            type="button"
            ref={docButton}
            className={`pick${props.kbId && !document ? ' unset' : ''}`}
            disabled={!props.kbId}
            onClick={(event) => setMenu({ kind: 'doc', anchor: anchorOf(event.currentTarget) })}
          >
            <span className="k">Doküman</span>
            <span className="v">{document ? document.name : props.kbId ? 'Seç' : '—'}</span>
          </button>
        </nav>

        <div className="tabs" role="tablist">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={tab.key === props.mode}
              className={tab.key === props.mode ? 'on' : undefined}
              onClick={() => props.onMode(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {props.mode === 'incele' && props.doc && props.methods.length ? (
          <div className="chips">
            {props.methods.map((method) => {
              const at = props.selected.indexOf(method);
              const note = noteFor(props.catalogue, method, props.doc);
              return (
                <button
                  key={method}
                  type="button"
                  className={`chip${at >= 0 ? ' on' : ''}`}
                  aria-pressed={at >= 0}
                  onClick={() => props.onToggleMethod(method)}
                >
                  {at >= 0 && props.selected.length > 1 ? <span className="ord">{at + 1}</span> : null}
                  {labelOf(method)}
                  {note ? <span className="note">{note}</span> : null}
                </button>
              );
            })}
          </div>
        ) : (
          <div style={{ flex: '1 1 var(--chipsw)', minWidth: 0 }} />
        )}

        <div className="right">
          {props.kbId ? (
            <span className={`pill${ready ? ' on' : ''}`}>
              <span className="dot" />
              {`${ready}/${props.documents.length} doküman hazır`}
            </span>
          ) : null}

          {onBoard && (props.pages.length > 0 || stepsDifferences) ? (
            <div className="nav">
              {stepsDifferences ? (
                <div className="grp">
                  <button type="button" title="Önceki ayrışma" onClick={() => props.onStepDifference(-1)}>
                    ‹ Fark
                  </button>
                  <span className="dpos">
                    {(props.differenceIndex >= 0 ? props.differenceIndex + 1 : '–') +
                      ' / ' +
                      props.differences}
                  </span>
                  <button type="button" title="Sonraki ayrışma" onClick={() => props.onStepDifference(1)}>
                    Fark ›
                  </button>
                  <span className="vr" />
                </div>
              ) : null}
              {props.pages.length ? (
                <div className="grp">
                  <button
                    type="button"
                    title="Önceki sayfa"
                    disabled={props.pages.indexOf(props.page as number) <= 0}
                    onClick={() => step(props, -1)}
                  >
                    ‹
                  </button>
                  <span className="lbl">Sayfa</span>
                  <select
                    aria-label="Sayfa"
                    value={String(props.page ?? '')}
                    onChange={(event) => props.onPage(Number(event.target.value))}
                  >
                    {props.pages.map((page) => (
                      <option key={page} value={page}>
                        {page}
                      </option>
                    ))}
                  </select>
                  <span className="lbl">/ {props.pages[props.pages.length - 1]}</span>
                  <button
                    type="button"
                    title="Sonraki sayfa"
                    disabled={props.pages.indexOf(props.page as number) >= props.pages.length - 1}
                    onClick={() => step(props, 1)}
                  >
                    ›
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </header>

      {menu?.kind === 'kb' ? (
        <Menu anchor={menu.anchor} onClose={() => setMenu(null)}>
          <div className="sect">Bilgi tabanları</div>
          {props.knowledgeBases.length ? (
            props.knowledgeBases.map((entry) => (
              <button
                key={entry.id ?? ''}
                type="button"
                className={`it${entry.id === props.kbId ? ' cur' : ''}`}
                onClick={() => {
                  props.onKb(entry.id ?? '');
                  setMenu(null);
                }}
              >
                <span className="n">{entry.name || entry.id}</span>
                <span className="m">{entry.retrieval_method || 'bilgi tabanı'}</span>
              </button>
            ))
          ) : (
            <div className="quiet">Bu konsolda bilgi tabanı yok.</div>
          )}
        </Menu>
      ) : null}

      {menu?.kind === 'doc' ? (
        <Menu anchor={menu.anchor} onClose={() => setMenu(null)}>
          <div className="sect">{kb ? kb.name || kb.id : 'Dokümanlar'}</div>
          {props.documents.length ? (
            props.documents.map((entry) => {
              const state = DOC_STATE[entry.analysis.status] ?? DOC_STATE.missing;
              const ready = entry.analysis.ready_methods;
              return (
                <button
                  key={entry.id ?? ''}
                  type="button"
                  className={`it${entry.id === props.documentId ? ' cur' : ''}`}
                  onClick={() => {
                    props.onDocument(entry.id ?? '');
                    setMenu(null);
                  }}
                >
                  <span className="n">{entry.name}</span>
                  <span className={`m${entry.analysis.status === 'failed' ? ' err' : ''}`}>
                    {state.text || ready.map(labelOf).join(', ') || 'hazır'}
                  </span>
                </button>
              );
            })
          ) : (
            <div className="quiet">Bu bilgi tabanında doküman yok.</div>
          )}
        </Menu>
      ) : null}
    </>
  );
}

function step(props: BarProps, delta: number) {
  const at = props.pages.indexOf(props.page as number) + delta;
  if (at >= 0 && at < props.pages.length) props.onPage(props.pages[at]);
}

/**
 * The three words an orchestration has to say about itself, or none.
 *
 * Which method that is comes from the catalogue's `orchestration` flag, never
 * from a key: the note belongs to whichever method declares itself one.
 */
export function noteFor(
  catalogue: ChunkingMethod[],
  method: string,
  doc: ViewerDoc | null,
): string | null {
  const entry = catalogue.find((candidate) => candidate.key === method);
  if (!entry?.orchestration) return null;
  return deepNote(doc);
}
