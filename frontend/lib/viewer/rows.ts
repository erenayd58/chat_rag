/**
 * The one alignment rule: the same text, in the same grid row, in every column.
 *
 * This is what makes the İncele view a comparison rather than three lists side
 * by side. Take the canonical units in reading order; for each unit, collect
 * every offset any selected method cuts at, and slice the unit at the union of
 * them. A slice is then a row, and in that row each method shows the chunk that
 * covers exactly those offsets — so a boundary one method draws and another
 * does not is visible as a line in one column and not the other, on the same
 * words.
 *
 * Ownership comes from the mapping segments, never from first-chunk membership
 * alone, and it follows the reading flow: among the chunks covering a slice,
 * take the nearest one at or ahead of where that column already is. A heading
 * repeated into several chunks' provenance, or a chunk resuming after an
 * isolated table, must not fake a new boundary — so a boundary is drawn only
 * when the flow moves *forward* into a chunk it has not visited, and every
 * chunk therefore opens at most once.
 *
 * A row is a **difference** when more than one method covers it, at least one
 * of them opens a chunk there and at least one does not: that is precisely
 * "the methods disagree here", and it is what the ‹ Fark › navigation steps
 * through.
 *
 * Ported from the Viewer v3 page, which computed the same rows in the browser
 * from the same payload. Kept as a pure function of (document, methods) so it
 * can be tested without rendering anything.
 */

import type { Arm, Segment, Unit, ViewerDoc } from './model';

export interface BoardRow {
  unit: Unit;
  /** the slice of the unit this row shows */
  start: number;
  end: number;
  /** method -> the index of the chunk that owns this slice, or null */
  own: Record<string, number | null>;
  /** method -> whether this row is where that chunk opens */
  opens: Record<string, boolean>;
  /** method -> whether more than one of its chunks covers this slice */
  overlaps: Record<string, boolean>;
  /** the methods disagree about a boundary here */
  differs: boolean;
  index: number;
}

export interface Board {
  rows: BoardRow[];
  /** the indexes of the rows where the methods disagree */
  differences: number[];
}

const EMPTY: Board = { rows: [], differences: [] };

function segmentsOf(arm: Arm, unitId: string): Segment[] {
  const rows = arm.seg?.[unitId];
  if (!rows) return [];
  return [...rows].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}

export function buildBoard(doc: ViewerDoc | null, methods: string[]): Board {
  if (!doc || !methods.length) return EMPTY;

  const rows: BoardRow[] = [];
  const differences: number[] = [];
  // Where each column's reading flow has got to. Null until that method has
  // covered anything at all.
  const at: Record<string, number | null> = {};
  for (const method of methods) at[method] = null;

  for (const unit of doc.units) {
    const length = unit.x.length;
    const cuts = new Set<number>([0, length]);
    const covers: Record<string, Segment[]> = {};

    for (const method of methods) {
      const segments = segmentsOf(doc.arms[method], unit.i);
      covers[method] = segments;
      for (const segment of segments) {
        if (segment[1] > 0 && segment[1] < length) cuts.add(segment[1]);
        if (segment[2] > 0 && segment[2] < length) cuts.add(segment[2]);
      }
    }

    const offsets = [...cuts].sort((a, b) => a - b);
    for (let k = 0; k + 1 < offsets.length; k += 1) {
      const start = offsets[k];
      const end = offsets[k + 1];
      const row: BoardRow = {
        unit,
        start,
        end,
        own: {},
        opens: {},
        overlaps: {},
        differs: false,
        index: rows.length,
      };

      let anyOpens = false;
      let anyContinues = false;
      let covered = 0;

      for (const method of methods) {
        const covering = covers[method].filter(
          (segment) => segment[1] <= start && segment[2] >= end,
        );
        if (!covering.length) {
          row.own[method] = null;
          continue;
        }
        covered += 1;

        const indexes = covering.map((segment) => segment[0]);
        const here = at[method];
        let owner: number;
        if (here === null) {
          owner = Math.min(...indexes);
        } else {
          const ahead = indexes.filter((index) => index >= here);
          owner = ahead.length ? Math.min(...ahead) : Math.max(...indexes);
        }

        row.own[method] = owner;
        row.overlaps[method] = covering.length > 1;
        row.opens[method] = here === null || owner > here;
        if (here !== null) {
          if (row.opens[method]) anyOpens = true;
          else anyContinues = true;
        }
        at[method] = here === null ? owner : Math.max(here, owner);
      }

      if (methods.length > 1 && covered > 1 && anyOpens && anyContinues) {
        row.differs = true;
        differences.push(row.index);
      }
      rows.push(row);
    }
  }

  return { rows, differences };
}

/**
 * The chunk running into this page from the previous one, per method.
 *
 * A reader who lands on page 7 has to be told that the first column of text is
 * the middle of chunk 12 rather than its start; without this the page looks
 * like every method opened a chunk at the top.
 */
export function continuations(
  doc: ViewerDoc,
  methods: string[],
  pageRows: BoardRow[],
): Record<string, number | null> {
  const found: Record<string, number | null> = {};
  for (const method of methods) {
    const first = pageRows.find((row) => row.own[method] !== null && row.own[method] !== undefined);
    found[method] = first && !first.opens[method] ? (first.own[method] as number) : null;
  }
  return found;
}
