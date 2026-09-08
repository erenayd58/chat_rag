/**
 * The alignment rule, on its own.
 *
 * `buildBoard` is the whole İncele view: everything else on that screen is
 * layout over what it decides. It is a pure function of (document, methods),
 * so it is tested here without rendering anything — and it is worth testing on
 * its own because the two properties it has to hold are easy to state and easy
 * to break by accident:
 *
 * 1. **every chunk opens at most once.** A chunk whose provenance mentions a
 *    unit twice — a heading repeated into it, a resumption after a table —
 *    must not draw a second boundary.
 * 2. **a difference is a disagreement.** A row is marked only when more than
 *    one method covers it, at least one opens a chunk there, and at least one
 *    does not. Two methods that both open, or both continue, agree.
 */

import { describe, expect, it } from 'vitest';
import { buildBoard, continuations } from '@/lib/viewer/rows';
import type { ViewerDoc } from '@/lib/viewer/model';

/** Three units on one page, with two chunkings that cut in different places. */
function corpus(): ViewerDoc {
  const units = [
    { i: 'u1', t: 'heading', p: 1, x: 'Takipteki Alacaklar', l: 2 },
    { i: 'u2', t: 'paragraph', p: 1, x: 'Birinci paragraf. Ikinci cumle burada.' },
    { i: 'u3', t: 'paragraph', p: 2, x: 'Ucuncu birim, ikinci sayfada.' },
  ];
  return {
    label: 'Rapor',
    id: 'doc-1',
    kind: 'arms-only',
    units,
    pages: [1, 2],
    meta: {},
    arms: {
      // One chunk for the whole first page, a second for the next page.
      alpha: {
        kind: 'a',
        chunks: [chunk('a-0', 1, ['u1', 'u2']), chunk('a-1', 2, ['u3'])],
        seg: {
          u1: [[0, 0, units[0].x.length, 'whole']],
          u2: [[0, 0, units[1].x.length, 'whole']],
          u3: [[1, 0, units[2].x.length, 'whole']],
        },
      },
      // The same text, cut in the middle of the second unit.
      beta: {
        kind: 'b',
        chunks: [chunk('b-0', 1, ['u1', 'u2']), chunk('b-1', 1, ['u2']), chunk('b-2', 2, ['u3'])],
        seg: {
          u1: [[0, 0, units[0].x.length, 'whole']],
          u2: [
            [0, 0, 18, 'partial'],
            [1, 18, units[1].x.length, 'partial'],
          ],
          u3: [[2, 0, units[2].x.length, 'whole']],
        },
      },
    },
  } as unknown as ViewerDoc;
}

function chunk(id: string, page: number, unitIds: string[]) {
  return { id, num: Number(id.split('-')[1]) + 1, n: 40, pg: [page], u: unitIds, rs: 'new_section' };
}

describe('the board aligns methods on the text', () => {
  it('slices a unit at the union of every method cut', () => {
    const board = buildBoard(corpus(), ['alpha', 'beta']);
    const inU2 = board.rows.filter((row) => row.unit.i === 'u2');
    // beta cut at 18, so the unit is two rows even though alpha did not cut.
    expect(inU2.map((row) => [row.start, row.end])).toEqual([
      [0, 18],
      [18, 38],
    ]);
    // ...and alpha shows its single chunk in both of them.
    expect(inU2.map((row) => row.own.alpha)).toEqual([0, 0]);
    expect(inU2.map((row) => row.own.beta)).toEqual([0, 1]);
  });

  it('opens each chunk exactly once', () => {
    const board = buildBoard(corpus(), ['alpha', 'beta']);
    for (const method of ['alpha', 'beta']) {
      const opened = board.rows
        .filter((row) => row.opens[method])
        .map((row) => row.own[method]);
      expect(opened).toEqual([...new Set(opened)]);
    }
  });

  it('marks a row where one method opens and another continues', () => {
    const board = buildBoard(corpus(), ['alpha', 'beta']);
    expect(board.differences).toHaveLength(1);
    const row = board.rows[board.differences[0]];
    // Exactly the second half of u2: beta starts a chunk, alpha runs on.
    expect(row.unit.i).toBe('u2');
    expect(row.start).toBe(18);
    expect(row.opens.beta).toBe(true);
    expect(row.opens.alpha).toBe(false);
  });

  it('finds no difference when only one method is shown', () => {
    const board = buildBoard(corpus(), ['beta']);
    expect(board.differences).toEqual([]);
    expect(board.rows.every((row) => !row.differs)).toBe(true);
  });

  it('is empty without a document or without a method', () => {
    expect(buildBoard(null, ['alpha']).rows).toEqual([]);
    expect(buildBoard(corpus(), []).rows).toEqual([]);
  });

  it('names the chunk running into a page from the one before it', () => {
    const doc = corpus();
    const board = buildBoard(doc, ['alpha', 'beta']);
    const firstPage = board.rows.filter((row) => row.unit.p === 1);
    const secondPage = board.rows.filter((row) => row.unit.p === 2);

    // Page one opens both methods' first chunks: nothing runs in.
    expect(continuations(doc, ['alpha', 'beta'], firstPage)).toEqual({
      alpha: null,
      beta: null,
    });
    // Page two opens new chunks too, so still nothing runs in.
    expect(continuations(doc, ['alpha', 'beta'], secondPage)).toEqual({
      alpha: null,
      beta: null,
    });
  });
});
