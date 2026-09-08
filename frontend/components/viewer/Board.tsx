'use client';

/**
 * İncele — one page of the document, with every selected chunking method
 * printed onto it.
 *
 * The board is a CSS grid whose rows are the slices `lib/viewer/rows.ts`
 * computed: the same canonical text lands in the same row in every column, so
 * a boundary one method draws and another does not is a line in one column and
 * not in the other, on the same words. That is the whole comparison, and it is
 * why this is not three lists side by side.
 *
 * A cell is tinted by its chunk's index modulo three, so consecutive chunks
 * are told apart without a legend; a cell no method covers is a ghost at a
 * third opacity, because the text is still there and only its ownership is
 * absent. The gutter's diamond marks a row where the methods disagree, and it
 * is what ‹ Fark › steps through.
 */

import { useEffect, useRef } from 'react';
import type { ChunkingMethod } from '@/types/api';
import {
  DECISION_MARKED,
  REASON_SHORT,
  type Chunk,
  type Unit,
  type ViewerDoc,
} from '@/lib/viewer/model';
import type { Board as BoardModel, BoardRow } from '@/lib/viewer/rows';
import { continuations } from '@/lib/viewer/rows';
import { Marks, anchorOf, type Anchor } from './Menu';

export interface Selection {
  method: string;
  chunk: number;
}

export function Board({
  doc,
  methods,
  board,
  page,
  catalogue,
  selection,
  onSelect,
  scrollTo,
}: {
  doc: ViewerDoc;
  methods: string[];
  board: BoardModel;
  page: number;
  catalogue: ChunkingMethod[];
  selection: Selection | null;
  onSelect: (selection: Selection | null, anchor: Anchor | null) => void;
  /** A row to bring into view once — a ‹ Fark › step, or a jump from Sorgu. */
  scrollTo: { row?: number; method?: string; chunk?: number } | null;
}) {
  const wrap = useRef<HTMLDivElement | null>(null);
  const label = (key: string) =>
    catalogue.find((method) => method.key === key)?.label ?? key;

  const pageRows = board.rows.filter((row) => row.unit.p === page);
  const running = continuations(doc, methods, pageRows);
  const columns = methods.length;
  const gutter = columns > 1;

  useEffect(() => {
    if (!scrollTo || !wrap.current) return;
    const selector =
      scrollTo.row !== undefined
        ? `[data-row="${scrollTo.row}"]`
        : `[data-method="${scrollTo.method}"][data-chunk="${scrollTo.chunk}"]`;
    const target = wrap.current.querySelector(selector);
    if (target) target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [scrollTo, page]);

  return (
    <div className="wrap" ref={wrap}>
      <section className={`sheet c${Math.min(columns, 3)}`}>
        <Marks />
        <header className="shead">
          <span className="dl">
            {doc.label}
            {doc.live?.kbName ? ` · ${doc.live.kbName}` : ''}
          </span>
          <span className="pl">Sayfa {page}</span>
        </header>

        <div
          className="board"
          style={{
            gridTemplateColumns: `${gutter ? '26px ' : ''}repeat(${columns},minmax(0,1fr))`,
          }}
        >
          {columns > 1 ? (
            <div className="grow">
              {gutter ? <div className="gut chead" /> : null}
              {methods.map((method) => (
                <div className="chead" key={method}>
                  {label(method)}
                </div>
              ))}
            </div>
          ) : null}

          {methods.some((method) => running[method] !== null) ? (
            <div className="grow">
              {gutter ? <div className="gut" /> : null}
              {methods.map((method) => {
                const index = running[method];
                const chunk = index === null ? null : doc.arms[method]?.chunks[index];
                return (
                  <div className="cont" key={method}>
                    {chunk ? `‹ Parça ${chunk.num} önceki sayfadan devam ediyor` : ''}
                  </div>
                );
              })}
            </div>
          ) : null}

          {pageRows.map((row) => (
            <div className="grow" key={row.index} data-row={row.index}>
              {gutter ? (
                <div className="gut">
                  {row.differs ? <span className="d" title="Yöntemler burada ayrışıyor" /> : null}
                </div>
              ) : null}
              {methods.map((method) => (
                <Cell
                  key={method}
                  row={row}
                  method={method}
                  doc={doc}
                  selection={selection}
                  onSelect={onSelect}
                />
              ))}
            </div>
          ))}
        </div>

        {pageRows.length ? null : (
          <div className="emptypg">Bu sayfada canonical içerik yok.</div>
        )}
      </section>
    </div>
  );
}

function Cell({
  row,
  method,
  doc,
  selection,
  onSelect,
}: {
  row: BoardRow;
  method: string;
  doc: ViewerDoc;
  selection: Selection | null;
  onSelect: (selection: Selection | null, anchor: Anchor | null) => void;
}) {
  const own = row.own[method];
  if (own === null || own === undefined) {
    return (
      <div className="cell ghost" data-method={method}>
        <UnitSlice row={row} />
      </div>
    );
  }

  const chunk = doc.arms[method]?.chunks[own];
  const opens = row.opens[method] && chunk;
  const selected = selection?.method === method && selection.chunk === own;

  return (
    <div
      className={`cell k${own % 3}${opens ? ' cb' : ''}${selected ? ' sel' : ''}`}
      data-method={method}
      data-chunk={own}
      onClick={(event) => {
        event.stopPropagation();
        if (selected) onSelect(null, null);
        else onSelect({ method, chunk: own }, anchorOf(event.currentTarget));
      }}
    >
      {opens && chunk ? <BoundaryLabel doc={doc} chunk={chunk} over={!!row.overlaps[method]} /> : null}
      <UnitSlice row={row} />
    </div>
  );
}

/**
 * Why this chunk starts here, in one line, on the boundary itself.
 *
 * A boundary Deep Analysis moved is marked, but only when the run's decision
 * trail is actually in the payload: a mark with nothing behind it would be a
 * claim rather than a record.
 */
function BoundaryLabel({ doc, chunk, over }: { doc: ViewerDoc; chunk: Chunk; over: boolean }) {
  const short = REASON_SHORT[chunk.rs] ?? 'Yeni parça';
  const deep = !!(doc.story && chunk.dec && DECISION_MARKED.has(chunk.dec.status));
  return (
    <span className={`bl${deep ? ' deep' : ''}`}>
      <b>{chunk.num}</b> · {short}
      {over ? ' · örtüşme' : ''}
    </span>
  );
}

/**
 * One slice of a canonical unit.
 *
 * The packager pre-renders a unit's HTML (a table stays a table, a list stays
 * a list) and that rendering is only correct for the *whole* unit — so it is
 * used when the slice is the whole unit, and a partial slice falls back to the
 * plain text it actually covers. `pre-line` keeps the shape of a list or a
 * table row that was cut.
 *
 * That pre-rendering is the one place this screen sets HTML rather than text.
 * It is markup **this product built**, in `amsc.viewer.corpus`, out of the
 * parser's canonical units with the document's own text escaped on the way in;
 * the tags are the packager's own table and list structure and nothing from
 * the file survives as markup. The `x` field beside it is the same text
 * unescaped, and it is what every partial slice renders — as text.
 */
function UnitSlice({ row }: { row: BoardRow }) {
  const unit: Unit = row.unit;
  const whole = row.start === 0 && row.end === unit.x.length;
  const text = unit.x.slice(row.start, row.end);

  if (unit.t === 'heading') {
    const level = Math.min(Math.max(unit.l ?? 2, 1), 4);
    if (whole && unit.h) {
      return <div className={`hx l${level}`} dangerouslySetInnerHTML={{ __html: unit.h }} />;
    }
    return (
      <div className={`hx l${level}`}>{text.replace(/^#{1,6}\s+/, '').replace(/\*\*/g, '')}</div>
    );
  }
  if (whole && unit.h) {
    return <div className="tx" dangerouslySetInnerHTML={{ __html: unit.h }} />;
  }
  const pre = unit.t === 'list' || unit.t === 'table' ? ' pre' : '';
  return <div className={`tx${pre}`}>{text}</div>;
}
