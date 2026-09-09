/**
 * The render model `GET .../analysis/payload` answers with, and its vocabulary.
 *
 * The contract publishes `payload` as pass-through, so it is typed here — next
 * to the code that reads it — rather than in `types/api.ts`, which mirrors what
 * the contract does pin. The keys are short because the payload carries a whole
 * document's text and every chunk's provenance, and it is written by the
 * packager in the `chunk` repository (`amsc.viewer.corpus`): this file is a
 * reader of that shape, never a second definition of it.
 *
 * Everything a *method* is comes from `GET /api/v1/meta/chunking-methods` and
 * from the payload's own `live.methods`. There is no method key in this file
 * and there must not be one: a chunker added to the library is one file there,
 * and the Viewer lists it because the registry does.
 *
 * The words below are the other kind of vocabulary — boundary reasons,
 * structural smells, Deep Analysis decisions. Those are recorded *by* a chunker
 * on a chunk, they are a closed enum of the library's, and an unrecognised one
 * is shown as it came rather than dropped.
 */

/* ------------------------------------------------------------------ */
/* The payload                                                         */
/* ------------------------------------------------------------------ */

/** One canonical unit: what the parser read, before any chunker saw it. */
export interface Unit {
  /** unit id */
  i: string;
  /** type: heading | paragraph | list | table | ... */
  t: string;
  /** the page it was read from, or null when the format has no pages */
  p: number | null;
  /** the text itself */
  x: string;
  /** pre-rendered HTML, or 0 when it is the same as the escaped text */
  h?: string | 0;
  /** heading level */
  l?: number | null;
  /** over the representation ceiling */
  big?: unknown;
  /** parser findings about this unit */
  pf?: string[];
}

/** What Deep Analysis recorded about the boundary a chunk opens at. */
export interface ChunkDecision {
  status: string;
  removed_smells?: string[];
  [key: string]: unknown;
}

/** One chunk, as the page prints it. */
export interface Chunk {
  id: string;
  /** its number within the arm, 1-based, as the chunker named it */
  num: number;
  /** token count */
  n: number;
  /** pages, or a single null when the format has none */
  pg: (number | null)[];
  /** heading, raw */
  hd?: string | null;
  /** heading, pre-rendered */
  hh?: string | null;
  /** section paths */
  sp?: unknown[];
  /** the display section path */
  sd?: string[] | null;
  /** split strategies */
  st?: string[];
  /** unit ids */
  u: string[];
  /** why this chunk starts here */
  rs: string;
  /** continuation: previous / next chunk index, and the relation's type */
  cp?: number | null;
  cn?: number | null;
  rt?: string | null;
  /** whether it opens on a heading */
  hd_open?: boolean;
  dec?: ChunkDecision;
}

/**
 * One segment of a unit that a chunk covers:
 * `[chunk index, start offset, end offset, how]`.
 *
 * This is the mapping the whole İncele view is built on. A unit one method
 * cuts inside appears in several rows here, and the board slices it at the
 * union of every column's offsets so the same text lands in the same grid row
 * whatever cut it.
 */
export type Segment = [number, number, number, string];

export interface Arm {
  /** the engine behind the method */
  kind: string;
  chunks: Chunk[];
  /** base unit id -> index of the first chunk containing it */
  m?: Record<string, number>;
  /** unit id -> the segments of it each chunk covers */
  seg: Record<string, Segment[]>;
  /** the frozen run's per-query results, when there was one */
  q?: Record<string, { f?: number | null; cov?: number | null }>;
  /** the frozen run's retrieval measurements */
  ret?: Record<string, number> | null;
  /** structural quality, as measured on the real rows */
  sq?: { fragmentation?: Record<string, number> } | null;
  tim?: Record<string, number> | null;
}

export interface LiveMethod {
  status?: string;
  seconds?: number;
  chunk_count?: number;
  source?: string;
  error?: string;
}

export interface DeepMeta {
  status?: string;
  mode?: string;
  model?: string;
  promptVersion?: string;
  calls?: { total?: number };
  timing?: Record<string, number>;
  totals?: { standard?: Record<string, number>; deep?: Record<string, number> };
  smellTotal?: { standard?: number; deep?: number };
  storyCounts?: Record<string, number>;
  regressions?: number | null;
  retrieval?: { deep?: Record<string, number>; standard?: Record<string, number> };
  estTokens?: { prompt: number; completion: number };
  estCostUsd?: number | null;
  proposer?: { call_count?: number; call_status?: { ok?: number } };
  verifier?: { group_count?: number; accepted?: number; reverted?: number };
  selection?: { vote_count?: number; forbidden_vote_count?: number };
  [key: string]: unknown;
}

export interface StorySection {
  i: number;
  h?: string | null;
  pg?: (number | null)[];
  tt?: number;
  st: string;
  cons?: boolean;
  rv?: string | null;
  sz?: boolean;
  std?: number[];
  fin?: number[];
  gr?: { rm?: string[]; in?: string[] }[];
  pr?: { a?: boolean; r?: string }[];
}

export interface GoldQuery {
  id: string;
  q: string;
  pg?: number[];
}

/** One document, prepared. */
export interface ViewerDoc {
  label: string;
  id: string;
  kind: string;
  units: Unit[];
  arms: Record<string, Arm>;
  /** every page the units name; `[null]` when the format has no pages */
  pages: (number | null)[];
  gold?: GoldQuery[];
  parser?: { count: number; findings: { t: string; r: string; p?: number }[] };
  meta: {
    pageCount?: number;
    unitCount?: number;
    timing?: Record<string, any>;
    budgets?: Record<string, number>;
    deep?: DeepMeta | null;
    [key: string]: unknown;
  };
  story?: { counts?: Record<string, number>; sections?: StorySection[] } | null;
  live?: {
    docId?: string;
    kbId?: string | null;
    kbName?: string | null;
    methods?: Record<string, LiveMethod>;
    preparedAt?: string;
  } | null;
}

/* ------------------------------------------------------------------ */
/* The vocabulary a chunker records                                    */
/* ------------------------------------------------------------------ */

/** Why a chunk starts where it does — short, for a boundary label. */
export const REASON_SHORT: Record<string, string> = {
  doc_start: 'Doküman başlangıcı',
  new_section: 'Yeni bölüm',
  label_split: 'Ara başlık',
  budget_split: 'Boyut sınırı',
  md_size: 'Boyut penceresi',
  md_heading: 'Başlıkta kesildi',
  md_overlap: 'Boyut penceresi',
};

/** The same, as a sentence, for the detail card. */
export const REASON_LONG: Record<string, string> = {
  doc_start: 'Dokümanın ilk parçası.',
  new_section: 'Yeni bir bölüm başladığı için burada yeni parça açıldı.',
  label_split: 'Bölüm içindeki bir ara başlıkta yeni parça açıldı.',
  budget_split: 'Bölüm token bütçesini aştığı için burada bölündü.',
  md_size: 'Sabit boyut penceresi dolduğu için burada bölündü.',
  md_heading: 'Boyut penceresi bir başlığa denk geldiği için burada bölündü.',
  md_overlap:
    'Sabit boyut penceresi doldu; parça, bağlam için önceki parçanın sonunu da taşır.',
};

/** The structural problems the quality contract counts. */
export const SMELLS: Record<string, string> = {
  orphan_label: 'yalnız kalan başlık',
  lead_in_cut: 'giriş cümlesinden sonra kesim',
  fragment_cut: 'birim ortasında kesim',
  table_split: 'bölünmüş tablo',
  run_split_when_fits: 'sığdığı hâlde bölünen liste',
  continuation_cut: 'devam cümlesinde kesim',
  below_min: 'çok küçük parça',
  above_soft_max: 'bütçe üstü parça',
};

/** The order the benchmark reports them in. */
export const SMELL_ORDER = [
  'orphan_label',
  'lead_in_cut',
  'continuation_cut',
  'run_split_when_fits',
  'table_split',
  'fragment_cut',
  'below_min',
  'above_soft_max',
];

/** What Deep Analysis did at a boundary, as a sentence. */
export const DECISION_TEXT: Record<string, string> = {
  llm_accepted: 'Deep Analysis bu sınırı model onayıyla yerleştirdi.',
  llm_merged: 'Deep Analysis burada iki parçayı birleştirdi.',
  det_moved: 'Deep Analysis bu sınırı kural katmanıyla taşıdı.',
  deterministic_improved: 'Deep Analysis bu sınırı kural katmanıyla iyileştirdi.',
  std_changed: "Deep Analysis, Standard'ın buradaki kesimini kaldırdı ya da taşıdı.",
  ceiling: 'Tek birim sert token tavanını aştığı için birimin içinde kesildi.',
};

/** The decisions that mark a boundary as Deep Analysis's own on the board. */
export const DECISION_MARKED = new Set([
  'llm_accepted',
  'llm_merged',
  'det_moved',
  'deterministic_improved',
  'std_changed',
]);

/** How a Deep run ended, for the debug header. */
export const DEEP_STATUS_TEXT: Record<string, string> = {
  ok: 'model koştu',
  deterministic: 'kural tabanlı — model istenmedi',
  degraded: 'kısmi model — bazı çağrılar yanıtsız',
  fallback_no_provider: 'sağlayıcıya ulaşılamadı — deterministik tamamlandı',
  fallback_provider_error: 'model yanıt vermedi — deterministik tamamlandı',
};

/** What a section's recorded decision was, and who made it. */
export const SECTION_STATUS: Record<
  string,
  { source: string; tone: string; outcome: string; outcomeTone: string }
> = {
  deterministic_improved: { source: 'Kural', tone: 'rule', outcome: 'düzeltildi', outcomeTone: 'ok' },
  llm_accepted: { source: 'Model', tone: 'model', outcome: 'kabul', outcomeTone: 'ok' },
  llm_reverted: { source: 'Model', tone: 'model', outcome: 'geri çevrildi', outcomeTone: 'no' },
  contract_reverted: {
    source: 'Kalite kontrol',
    tone: 'qc',
    outcome: 'geri alındı',
    outcomeTone: 'rv',
  },
};

/** The boundary-reason codes the debug histogram lists first. */
export const REASON_ORDER = [
  'doc_start',
  'new_section',
  'label_split',
  'budget_split',
  'md_heading',
  'md_size',
  'md_overlap',
];

export const smellNames = (list?: string[] | null): string =>
  (list ?? []).map((key) => SMELLS[key] ?? key).join(', ');

/* ------------------------------------------------------------------ */
/* Small readings of the payload                                       */
/* ------------------------------------------------------------------ */

/**
 * The pages a document actually has, in reading order.
 *
 * A parser records a page number only when the format carries one: a PDF unit
 * comes back with `p: 4`, a Markdown or plain-text unit with `p: null`, and
 * the packager reports that faithfully — such a document's `pages` is
 * `[null]`. That is one page called nothing, which is not a page, so it is
 * dropped here: a pageless document has **no** pages, and every screen that
 * asks "which pages" gets that answer instead of a null to render.
 */
export function pagesOf(doc: ViewerDoc | null): number[] {
  return numbered(doc?.pages);
}

/** The page numbers in a list, dropping the null a pageless format records. */
export function numbered(pages?: (number | null)[] | null): number[] {
  return (pages ?? []).filter((page): page is number => typeof page === 'number');
}

/** `Sayfa 4` / `Sayfa 4–6`, or nothing when a chunk names no page. */
export function formatPages(pages?: (number | null)[] | null): string {
  const found = numbered(pages);
  if (!found.length) return '';
  return found.length === 1
    ? `Sayfa ${found[0]}`
    : `Sayfa ${found[0]}–${found[found.length - 1]}`;
}

/** A chunk's section, as a breadcrumb. */
export function sectionOf(chunk: Chunk): string | null {
  if (chunk.sd && chunk.sd.length) return chunk.sd.join(' › ');
  return chunk.hd ?? null;
}

/** A retrieved source's section, cleaned of markdown. */
export function sourceSection(source: {
  section_path?: string[];
  heading?: string | null;
}): string {
  const path = source.section_path;
  const raw = path && path.length ? path.join(' › ') : (source.heading ?? '');
  return String(raw ?? '')
    .replace(/[#*]/g, '')
    .trim();
}

/**
 * What a Deep arm has to say about itself, in three words or none.
 *
 * Read from the recorded run, never guessed: an arm whose run said nothing
 * unusual carries no note at all rather than a reassuring one.
 */
export function deepNote(doc: ViewerDoc | null): string | null {
  const deep = doc?.meta?.deep;
  if (!deep) return null;
  const calls = deep.calls?.total ?? 0;
  if (deep.status === 'ok') return calls > 0 ? null : 'modele gerek olmadı';
  if (deep.status === 'deterministic') return 'kural tabanlı';
  if (deep.status === 'degraded') return 'kısmi model';
  if (deep.status === 'fallback_no_provider') return 'modelsiz tamamlandı';
  if (deep.status === 'fallback_provider_error') return 'model yanıt vermedi';
  return null;
}

/** How long a Deep run took, summed over the stages it recorded. */
export function deepSeconds(deep?: DeepMeta | null): number | null {
  const timing = deep?.timing;
  if (!timing) return null;
  let total = 0;
  let any = false;
  for (const key of Object.keys(timing)) {
    const value = timing[key];
    if (typeof value === 'number') {
      total += value;
      any = true;
    }
  }
  return any ? total : null;
}

/**
 * The methods of this document a reader may actually open, in the order the
 * registry gave.
 *
 * `order` is the catalogue's; the intersection with the payload's arms is what
 * this upload has built. A live method still packaging is excluded — it will
 * appear when its status says ready and not before.
 */
export function methodsOf(doc: ViewerDoc | null, order: string[]): string[] {
  if (!doc) return [];
  const live = doc.live?.methods;
  return order.filter((method) => {
    if (!doc.arms?.[method]) return false;
    const state = live?.[method]?.status;
    return !state || state === 'ready';
  });
}

/** The token median and 90th percentile of an arm's chunks. */
export function tokenStats(chunks: Chunk[]): { median: number | null; p90: number | null } {
  const sorted = chunks
    .map((chunk) => chunk.n)
    .filter((n): n is number => typeof n === 'number')
    .sort((a, b) => a - b);
  if (!sorted.length) return { median: null, p90: null };
  const at = (quantile: number) =>
    sorted[Math.min(sorted.length - 1, Math.round(quantile * (sorted.length - 1)))];
  return { median: at(0.5), p90: at(0.9) };
}

export const fixed3 = (value: unknown): string =>
  value === null || value === undefined ? '—' : Number(value).toFixed(3);

export const fixed1 = (value: unknown): string =>
  value === null || value === undefined ? '—' : Number(value).toFixed(1);

export const formatDuration = (seconds: number | null): string =>
  seconds === null || seconds === undefined
    ? '—'
    : seconds >= 10
      ? `${Math.round(seconds)} s`
      : `${seconds.toFixed(1)} s`;
