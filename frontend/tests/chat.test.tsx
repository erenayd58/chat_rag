/**
 * Asking, answered: citations, grounded state, and the refusals.
 *
 * The contract says a client that shows the answer without `citations` and
 * `grounded` cannot tell an answer from a guess. These render the real screen
 * against a stubbed `/api/v1` and check that both are on it -- including the
 * uncomfortable case where the model cited nothing.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatPage from '@/app/chat/page';
import { ConfirmProvider } from '@/components/Modal';
import { ToastProvider } from '@/components/Toast';

function json(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

const KBS = {
  items: [{ id: 'kb-1', name: 'Raporlar', chunker: {}, retrieval_method: 'hyb', embedding_model: null, extra: {} }],
  page: { offset: 0, limit: 200, total: 1 },
};

const ANSWER = {
  answer: 'Net kâr 2024 yılında 1,2 milyar TL oldu.',
  citations: [
    {
      label: '[1]',
      chunk_id: 'c-1',
      document_id: 'd-1',
      document: 'Faaliyet Raporu 2024.pdf',
      section: 'Finansal Sonuçlar',
      pages: [12],
      chunking_mode: null,
      used: true,
      score: 0.71,
      content: 'Net kâr 1,2 milyar TL olarak gerçekleşti.',
    },
    {
      label: '[2]',
      chunk_id: 'c-2',
      document_id: 'd-1',
      document: 'Faaliyet Raporu 2024.pdf',
      section: 'Özet',
      pages: [3, 4],
      chunking_mode: null,
      used: false,
      score: 0.4,
      content: 'Yıl boyunca büyüme sürdü.',
    },
  ],
  knowledge_base_id: 'kb-1',
  retrieval_method: 'hybrid_rrf',
  grounded: true,
  timing: { query_id: 'q-1', total_seconds: 2.5, stages: {} },
  diagnostics: {},
};

/** Routes the stub answers, so a screen's own fan-out does not need listing. */
function stubApi(onQuery: () => Response) {
  return vi.fn(async (url: string) => {
    if (url.startsWith('/api/v1/knowledge-bases')) return json(KBS);
    if (url.startsWith('/api/v1/queries')) return onQuery();
    return json({ error: { type: 'not_found', message: url } }, 404);
  });
}

function renderChat() {
  return render(
    <ToastProvider>
      <ConfirmProvider>
        <ChatPage />
      </ConfirmProvider>
    </ToastProvider>,
  );
}

async function ask(text: string) {
  const box = await screen.findByPlaceholderText(/soru sorun/i);
  await waitFor(() => expect(box).toBeEnabled());
  await userEvent.type(box, text);
  await userEvent.click(screen.getByRole('button', { name: 'Gönder' }));
}

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe('Sohbet', () => {
  it('answers an arbitrary question and renders its citations', async () => {
    vi.stubGlobal('fetch', stubApi(() => json(ANSWER)));
    renderChat();

    await ask('2024 net kârı ne kadar?');

    expect(await screen.findByText(ANSWER.answer)).toBeInTheDocument();
    expect(screen.getByText('2024 net kârı ne kadar?')).toBeInTheDocument();
    expect(screen.getAllByText('Faaliyet Raporu 2024.pdf')).toHaveLength(2);
    expect(screen.getByText('Finansal Sonuçlar')).toBeInTheDocument();
    expect(screen.getByText('s. 12')).toBeInTheDocument();
    expect(screen.getByText('s. 3–4')).toBeInTheDocument();
    expect(screen.getByText('cevapta kullanıldı')).toBeInTheDocument();
    expect(screen.getByText('Kaynağa dayalı')).toBeInTheDocument();
    expect(screen.getByText('1/2 kaynak kullanıldı')).toBeInTheDocument();
  });

  it('says so when the model cited nothing', async () => {
    vi.stubGlobal('fetch', stubApi(() => json({ ...ANSWER, grounded: false })));
    renderChat();

    await ask('kaynaksız soru');

    expect(await screen.findByText('Kaynağa dayanmıyor')).toBeInTheDocument();
    expect(screen.getByText('Bu cevap kaynak göstermedi.')).toBeInTheDocument();
  });

  it('puts the question back when the server refuses it under load', async () => {
    vi.stubGlobal(
      'fetch',
      stubApi(() =>
        json({ error: { type: 'overloaded', message: 'no capacity' } }, 503, { 'Retry-After': '9' }),
      ),
    );
    renderChat();

    await ask('meşgul sunucu');

    expect(await screen.findByText('Sunucu şu anda meşgul')).toBeInTheDocument();
    expect(screen.getByText(/9 saniye sonra/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/soru sorun/i)).toHaveValue('meşgul sunucu');
  });

  it('offers Search when answer generation is unavailable', async () => {
    vi.stubGlobal(
      'fetch',
      stubApi(() => json({ error: { type: 'unavailable', message: 'answer model unreachable' } }, 503)),
    );
    renderChat();

    await ask('model kapalı');

    expect(await screen.findByText(/Cevap üretimi şu anda kullanılamıyor/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Search' })).toHaveAttribute('href', '/search');
  });

  it('reports a passed deadline as a deadline', async () => {
    vi.stubGlobal(
      'fetch',
      stubApi(() => json({ error: { type: 'timeout', message: 'deadline passed' } }, 504)),
    );
    renderChat();

    await ask('çok uzun soru');

    expect(await screen.findByText('İşlem zaman aşımına uğradı')).toBeInTheDocument();
  });
});
