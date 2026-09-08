/**
 * Reading a row the contract calls pass-through.
 *
 * A chunking method's rows are the chunker's own JSONL, published as an open
 * object precisely so that a new chunker is not blocked on this front end
 * learning its field names. So nothing here *requires* a field: it looks for
 * the few names a chunker is likely to have used, and when it finds none the
 * screen falls back to showing the row as it arrived rather than inventing a
 * blank.
 */

export function pickString(row: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === 'string' && value.trim()) return value;
  }
  return null;
}

export function pickNumber(row: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return null;
}

export const ROW_TEXT_KEYS = ['text', 'content', 'chunk_text', 'body'];
export const ROW_SECTION_KEYS = ['heading', 'section_title', 'section', 'title'];
export const ROW_ID_KEYS = ['chunk_id', 'id', 'uid'];
export const ROW_SIZE_KEYS = ['token_count', 'tokens', 'n_tokens', 'length'];

/** The row as it arrived, formatted. The last resort, and an honest one. */
export function rawRow(row: Record<string, unknown>): string {
  try {
    return JSON.stringify(row, null, 2);
  } catch {
    return String(row);
  }
}
