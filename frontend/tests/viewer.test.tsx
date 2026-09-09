/**
 * The Viewer, rendered against a stubbed `/api/v1`.
 *
 * Four things this screen has to do, and each of them is something the old
 * standalone Viewer did that the migration could quietly have lost:
 *
 * * **discover methods** — every chip is a registry key intersected with what
 *   a document has ready. The catalogue in this file uses invented method keys
 *   on purpose: if a real one ever appears in the Viewer's source, this test
 *   stops proving anything and the screen would break for a new chunker.
 * * **compare** — two methods on one page, aligned on the text, with the
 *   disagreements counted.
 * * **ask the arms** — one question, several methods, one request to
 *   `POST /api/v1/analysis-queries`.
 * * **wait honestly** — an analysis still building says so and does not
 *   pretend the document is empty.
 * * **draw a document with no pages** — Markdown and plain text carry no page
 *   number, so the packager reports `pages: [null]`. The board is the whole
 *   document then, and there is no page navigation to offer.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ViewerPage from '@/app/viewer/page';

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Two methods with names no chunker in this product has. */
const CATALOGUE = {
  items: [
    {
      key: 'alpha-cut',
      label: 'Alfa Kesim',
      summary: 'Yapısal sınırlarda keser.',
      engine: 'alpha',
      available: true,
      unavailable_reason: null,
      uses_model: false,
      default: true,
      orchestration: false,
      baseline: null,
    },
    {
      key: 'beta-cut',
      label: 'Beta Kesim',
      summary: 'Sabit boyut penceresiyle keser.',
      engine: 'beta',
      available: true,
      unavailable_reason: null,
      uses_model: false,
      default: false,
      orchestration: false,
      baseline: null,
    },
  ],
  page: { offset: 0, limit: 200, total: 2 },
};

const KBS = {
  items: [
    {
      id: 'kb-1',
      name: 'Yıllık raporlar',
      chunker: {},
      retrieval_method: 'hybrid',
      embedding_model: null,
      extra: {},
    },
  ],
  page: { offset: 0, limit: 200, total: 1 },
};

function analysis(status: string, ready: string[]) {
  return {
    status,
    content_id: 'sha-1',
    selected_methods: ['alpha-cut', 'beta-cut'],
    ready_methods: ready,
    failed_methods: [],
    unit_count: 3,
    deep_source: null,
    error: null,
    updated_at: null,
    content: { requested_methods: [], ready_methods: ready, shared_with_document_ids: [] },
  };
}

function documents(status: string, ready: string[], name = 'Faaliyet Raporu.pdf') {
  return {
    items: [
      {
        id: 'doc-1',
        knowledge_base_id: 'kb-1',
        name,
        content_id: 'sha-1',
        size_bytes: 1000,
        chunk_count: 3,
        chunking_mode: 'standard',
        status: 'indexed',
        ingested_at: '2026-01-01T00:00:00',
        ingest_job_id: null,
        analysis: analysis(status, ready),
      },
    ],
    page: { offset: 0, limit: 200, total: 1 },
  };
}

const UNITS = [
  { i: 'u1', t: 'heading', p: 1, x: 'Takipteki Alacaklar', l: 2 },
  { i: 'u2', t: 'paragraph', p: 1, x: 'Birinci cumle burada. Ikinci cumle boyle.' },
];

const PAYLOAD = {
  document_id: 'doc-1',
  content_id: 'sha-1',
  label: 'Faaliyet Raporu.pdf',
  ready_methods: ['alpha-cut', 'beta-cut'],
  payload: {
    label: 'Faaliyet Raporu.pdf',
    id: 'doc-1',
    kind: 'arms-only',
    units: UNITS,
    pages: [1],
    meta: { pageCount: 1, unitCount: 2 },
    arms: {
      'alpha-cut': {
        kind: 'alpha',
        chunks: [{ id: 'a-0', num: 1, n: 40, pg: [1], u: ['u1', 'u2'], rs: 'doc_start' }],
        seg: {
          u1: [[0, 0, UNITS[0].x.length, 'whole']],
          u2: [[0, 0, UNITS[1].x.length, 'whole']],
        },
      },
      'beta-cut': {
        kind: 'beta',
        chunks: [
          { id: 'b-0', num: 1, n: 20, pg: [1], u: ['u1', 'u2'], rs: 'doc_start' },
          { id: 'b-1', num: 2, n: 20, pg: [1], u: ['u2'], rs: 'md_size' },
        ],
        seg: {
          u1: [[0, 0, UNITS[0].x.length, 'whole']],
          u2: [
            [0, 0, 21, 'partial'],
            [1, 21, UNITS[1].x.length, 'partial'],
          ],
        },
      },
    },
  },
};

/**
 * The same document as the parser reads a Markdown file: every unit's page is
 * null, so the packager's `pages` is `[null]` — one page called nothing, which
 * is not a page.
 */
const PAGELESS = {
  ...PAYLOAD,
  label: 'rapor.md',
  payload: {
    ...PAYLOAD.payload,
    label: 'rapor.md',
    pages: [null],
    units: UNITS.map((unit) => ({ ...unit, p: null })),
    arms: Object.fromEntries(
      Object.entries(PAYLOAD.payload.arms).map(([method, arm]) => [
        method,
        { ...arm, chunks: arm.chunks.map((chunk) => ({ ...chunk, pg: [null] })) },
      ]),
    ),
  },
};

const ANSWER = {
  document_id: 'doc-1',
  question: 'takipteki alacaklar ne oldu?',
  methods: ['alpha-cut', 'beta-cut'],
  arms: [
    {
      method: 'alpha-cut',
      engine: 'alpha',
      label: 'Alfa Kesim',
      status: 'ok',
      error: null,
      answer: { text: 'Alfa cevabi [S1].', sufficient: true, sources_used: ['S1'] },
      sources: [
        {
          label: 'S1',
          chunk_id: 'a-0',
          arm: 'alpha-cut',
          text: 'Takipteki alacaklar azaldi.',
          pages: [1],
          token_count: 40,
          heading: 'Takipteki Alacaklar',
          section_path: [],
          unit_ids: ['u1'],
          used: true,
        },
      ],
      unit_overlap: 0.5,
      dense: true,
      note: null,
      seconds: 0.4,
    },
    {
      method: 'beta-cut',
      engine: 'beta',
      label: 'Beta Kesim',
      status: 'ok',
      error: null,
      answer: { text: 'Beta cevabi [S1].', sufficient: true, sources_used: ['S1'] },
      sources: [
        {
          label: 'S1',
          chunk_id: 'b-1',
          arm: 'beta-cut',
          text: 'Ikinci cumle boyle.',
          pages: [1],
          token_count: 20,
          heading: null,
          section_path: [],
          unit_ids: ['u2'],
          used: true,
        },
      ],
      unit_overlap: 0.25,
      dense: true,
      note: null,
      seconds: 0.5,
    },
  ],
  embedding_model: 'test-embedding',
  answer_model: 'test-answer',
  total_seconds: 1.2,
};

interface StubOptions {
  status?: string;
  ready?: string[];
  onQuery?: (body: unknown) => Response;
  /** The prepared analysis this document has; the paged one unless told. */
  payload?: unknown;
  name?: string;
}

function stub({
  status = 'ready',
  ready = ['alpha-cut', 'beta-cut'],
  onQuery,
  payload = PAYLOAD,
  name = 'Faaliyet Raporu.pdf',
}: StubOptions = {}) {
  const calls: { url: string; body?: unknown }[] = [];
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, body });
    if (url.startsWith('/api/v1/meta/chunking-methods')) return json(CATALOGUE);
    if (url.startsWith('/api/v1/meta/models')) return json({ chain: {} });
    if (url.startsWith('/api/v1/knowledge-bases')) return json(KBS);
    if (url.startsWith('/api/v1/analysis-queries')) {
      return onQuery ? onQuery(body) : json(ANSWER);
    }
    if (url.startsWith('/api/v1/documents?')) return json(documents(status, ready, name));
    if (url.includes('/analysis/payload')) {
      if (status !== 'ready') {
        return json(
          { error: { type: 'not_ready', message: 'building', details: { state: { status } } } },
          409,
        );
      }
      return json(payload);
    }
    if (url.includes('/analysis')) return json(analysis(status, ready));
    return json({ error: { type: 'not_found', message: url } }, 404);
  });
  vi.stubGlobal('fetch', fetcher);
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.localStorage.clear();
});

/** Open the knowledge base, then the document, through the breadcrumb. */
async function pickFromMenu(user: ReturnType<typeof userEvent.setup>, text: string) {
  // The knowledge base is also named on the overview panel behind the menu,
  // so the click is scoped to the menu that was just opened.
  const menu = await screen.findByRole('menu');
  await user.click(within(menu).getByText(text));
}

async function openKnowledgeBase(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: /Bilgi tabanı/ }));
  await pickFromMenu(user, 'Yıllık raporlar');
}

async function openDocument(
  user: ReturnType<typeof userEvent.setup>,
  name = 'Faaliyet Raporu.pdf',
) {
  await openKnowledgeBase(user);
  await user.click(await screen.findByRole('button', { name: /Doküman/ }));
  await pickFromMenu(user, name);
}

describe('Viewer', () => {
  it('offers exactly the methods the registry and the document agree on', async () => {
    stub();
    const user = userEvent.setup();
    render(<ViewerPage />);

    await user.click(await screen.findByRole('tab', { name: 'İncele' }));
    await openDocument(user);

    // Both methods appear as chips, by the label the catalogue gave them.
    expect(await screen.findByRole('button', { name: /Alfa Kesim/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Beta Kesim/ })).toBeInTheDocument();
    // The first is on the board; the second is offered, not forced.
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Alfa Kesim/ })).toHaveAttribute(
        'aria-pressed',
        'true',
      );
    });
    expect(screen.getByRole('button', { name: /Beta Kesim/ })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
  });

  it('compares two methods on the same text and counts where they differ', async () => {
    stub();
    const user = userEvent.setup();
    render(<ViewerPage />);

    await user.click(await screen.findByRole('tab', { name: 'İncele' }));
    await openDocument(user);
    await user.click(await screen.findByRole('button', { name: /Beta Kesim/ }));

    // Two columns, headed by the two labels.
    await waitFor(() => {
      expect(document.querySelectorAll('.chead').length).toBeGreaterThanOrEqual(2);
    });
    // Beta opens a second chunk mid-paragraph; alpha runs on. That is one
    // disagreement, and the bar offers it for stepping through.
    expect(await screen.findByText('– / 1')).toBeInTheDocument();
    expect(document.querySelectorAll('.gut .d')).toHaveLength(1);
  });

  it('asks one question of several methods in one request', async () => {
    const calls = stub();
    const user = userEvent.setup();
    render(<ViewerPage />);

    await user.click(await screen.findByRole('tab', { name: 'İncele' }));
    await openDocument(user);
    await user.click(await screen.findByRole('tab', { name: 'Sorgu' }));

    // Both methods, so the answer is a comparison.
    await user.click(await screen.findByRole('button', { name: 'Beta Kesim' }));
    await user.type(
      await screen.findByLabelText('Soru'),
      'takipteki alacaklar ne oldu?',
    );
    await user.click(screen.getByRole('button', { name: 'Sor' }));

    await waitFor(() => expect(screen.getByText('Alfa cevabi [S1].')).toBeInTheDocument());
    expect(screen.getByText('Beta cevabi [S1].')).toBeInTheDocument();
    // The number that makes it a comparison rather than two answers.
    expect(screen.getByText(/örtüşme %50/)).toBeInTheDocument();

    const asked = calls.filter((call) => call.url.startsWith('/api/v1/analysis-queries'));
    expect(asked).toHaveLength(1);
    expect(asked[0].body).toMatchObject({
      document_id: 'doc-1',
      question: 'takipteki alacaklar ne oldu?',
      methods: ['alpha-cut', 'beta-cut'],
    });
  });

  it('says an analysis is still being prepared instead of showing nothing', async () => {
    stub({ status: 'running', ready: [] });
    const user = userEvent.setup();
    render(<ViewerPage />);

    await user.click(await screen.findByRole('tab', { name: 'İncele' }));
    await openDocument(user);

    expect(await screen.findByText(/Analiz hazırlanıyor/)).toBeInTheDocument();
    // And no method chip is offered for a document that has none ready.
    expect(screen.queryByRole('button', { name: /Alfa Kesim/ })).not.toBeInTheDocument();
  });

  it('puts a document with no page numbers on the board whole', async () => {
    stub({ payload: PAGELESS, name: 'rapor.md' });
    const user = userEvent.setup();
    render(<ViewerPage />);

    await user.click(await screen.findByRole('tab', { name: 'İncele' }));
    await openDocument(user, 'rapor.md');

    // The board is drawn, with the document's own text on it -- a page the
    // format never recorded is not a page to wait for.
    await waitFor(() => expect(document.querySelector('.sheet')).not.toBeNull());
    expect(document.querySelectorAll('.cell[data-chunk]').length).toBeGreaterThan(0);
    expect(screen.getByText('Tüm doküman')).toBeInTheDocument();

    // And nothing offers to page through what has no pages.
    expect(screen.queryByLabelText('Sayfa')).not.toBeInTheDocument();

    // The chunk card says what it knows, and does not name a null page.
    await user.click(document.querySelector('.cell[data-chunk]') as HTMLElement);
    const card = await screen.findByRole('dialog');
    expect(within(card).getByText('40 token')).toBeInTheDocument();
  });

  it('reads the whole overview from the contract', async () => {
    stub();
    const user = userEvent.setup();
    render(<ViewerPage />);

    await openKnowledgeBase(user);

    const stats = document.querySelector('.statrow') as HTMLElement;
    await waitFor(() => {
      expect(within(stats).getByText('Hazır analiz')).toBeInTheDocument();
    });
    // One document, ready: counted, not asserted per row.
    expect(within(stats).getAllByText('1').length).toBeGreaterThan(0);
  });
});
