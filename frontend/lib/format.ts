/**
 * Presentation only. Nothing here invents a fact.
 *
 * Every label is derived from something the API actually said: a state name, a
 * count, a byte size, an ISO timestamp. Where the API has no answer the answer
 * shown is an em dash, never a guess and never a number this front end made up.
 */

/** A locale date and time, or an em dash. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  return `${date.toLocaleDateString('tr-TR', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })} ${date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('tr-TR');
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value < 10 ? `${value.toFixed(2)} sn` : `${value.toFixed(1)} sn`;
}

/**
 * A machine name made readable: `deep_analysis` becomes `Deep analysis`.
 *
 * Deliberately mechanical. A lookup table here would be a second catalogue of
 * the library's identities, which is the one thing this front end must not
 * hold -- so an unfamiliar value from a new chunker renders correctly on the
 * day it is added rather than falling through to "unknown".
 */
export function humanise(value: string | null | undefined): string {
  if (!value) return '—';
  const words = value.replace(/[_-]+/g, ' ').trim();
  return words.charAt(0).toLocaleUpperCase('tr-TR') + words.slice(1);
}

/** `p. 4` / `pp. 4–9`, from whatever the source recorded. */
export function formatPages(pages: unknown[] | null | undefined): string | null {
  if (!pages || !pages.length) return null;
  if (pages.length === 1) return `s. ${String(pages[0])}`;
  return `s. ${String(pages[0])}–${String(pages[pages.length - 1])}`;
}

/** A score as the API gave it. Never rendered as a percentage or a confidence. */
export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return '—';
  return score.toFixed(4);
}

export function truncate(text: string, limit: number): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}…`;
}
