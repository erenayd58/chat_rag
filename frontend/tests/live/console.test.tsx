/**
 * The console, driven through its own screens, against a real backend.
 *
 * Nothing here is stubbed: every request goes to `/api/v1` on the running
 * server, the ingest job is polled by the shipped poller, and the answer is a
 * real answer from the configured model. What it proves is the thing unit
 * tests cannot -- that the screens, the client and the contract fit together.
 *
 *     npm run test:live      # with the console on :3000
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const pushed: string[] = [];
let routeParams: Record<string, string> = {};

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: (href: string) => pushed.push(href), replace: () => {}, refresh: () => {} }),
  usePathname: () => '/',
  useParams: () => routeParams,
  redirect: () => {},
}));

import KnowledgeBasesPage from '@/app/knowledge-bases/page';
import KnowledgeBasePage from '@/app/knowledge-bases/[kbId]/page';
import DocumentPage from '@/app/documents/[documentId]/page';
import SearchPage from '@/app/search/page';
import ChatPage from '@/app/chat/page';
import { ConfirmProvider } from '@/components/Modal';
import { ToastProvider } from '@/components/Toast';
import api from '@/lib/api';

/** A small Turkish report, so the answers it produces can be read. The live
 *  suite runs from this package's root, as `npm run test:live` does. */
const REPORT = resolve('tests/live/fixtures/rapor.md');
const STAMP = new Date().toISOString().slice(11, 19).replace(/:/g, '');
const KB_NAME = `Live ${STAMP}`;

const state: { kbId: string; documentId: string } = { kbId: '', documentId: '' };

function ui(node: React.ReactNode) {
  return render(
    <ToastProvider>
      <ConfirmProvider>{node}</ConfirmProvider>
    </ToastProvider>,
  );
}

/** Answer the confirmation dialog the destructive actions open. */
async function confirmDialog(label: string) {
  const dialog = await screen.findByRole('dialog');
  await userEvent.click(await within(dialog).findByRole('button', { name: label }));
}

beforeAll(async () => {
  const health = await api.meta.health();
  if (!health.ready) throw new Error(`the backend is not ready: ${health.state}`);
});

describe('Knowledge Bases', () => {
  it('creates one from the screen', async () => {
    ui(<KnowledgeBasesPage />);

    await userEvent.click(await screen.findByRole('button', { name: /Yeni bilgi tabanı/ }));
    await userEvent.type(screen.getByLabelText('Ad'), KB_NAME);
    await userEvent.type(screen.getByLabelText(/Açıklama/), 'canlı uçtan uca');
    await userEvent.click(screen.getByRole('button', { name: 'Oluştur' }));

    await waitFor(() => expect(pushed.at(-1)).toMatch(/^\/knowledge-bases\//));
    state.kbId = decodeURIComponent(pushed.at(-1)!.split('/').pop()!);
    expect(state.kbId).toBeTruthy();
  });

  it('lists it with its own counts', async () => {
    ui(<KnowledgeBasesPage />);
    const card = await screen.findByText(KB_NAME);
    const region = card.closest('.kb-card')!;
    expect(within(region as HTMLElement).getByText('Boş')).toBeInTheDocument();
    expect(within(region as HTMLElement).getByText('canlı uçtan uca')).toBeInTheDocument();
  });
});

describe('Documents', () => {
  it('uploads one, follows the ingest job, and lists it', async () => {
    routeParams = { kbId: state.kbId };
    const view = ui(<KnowledgeBasePage />);

    await screen.findByRole('heading', { name: KB_NAME });
    await userEvent.click(screen.getByRole('button', { name: /Doküman yükle/ }));

    // The picker is filled from the registry and preselects whatever the
    // registry marks default, so waiting for a ticked box is waiting for the
    // catalogue -- there is no method name to wait for here.
    const checked = await waitFor(
      () => {
        const found = view.container.querySelectorAll<HTMLInputElement>('input[name="method"]:checked');
        if (!found.length) throw new Error('no method preselected yet');
        return Array.from(found);
      },
      { timeout: 30000 },
    );
    expect(checked.length).toBeGreaterThan(0);

    const input = view.container.querySelector<HTMLInputElement>('input[type="file"]')!;
    const file = new File([readFileSync(REPORT)], 'rapor.md', { type: 'text/markdown' });
    fireEvent.change(input, { target: { files: [file] } });
    await screen.findByRole('button', { name: 'rapor.md' });

    await userEvent.click(screen.getByRole('button', { name: 'Yükle ve analiz et' }));

    // 202, then the poller. The dialog closes only when the job succeeded.
    await waitFor(
      () => expect(screen.queryByRole('button', { name: 'Yükle ve analiz et' })).toBeNull(),
      { timeout: 240000, interval: 500 },
    );

    await userEvent.click(screen.getByRole('tab', { name: /Documents/ }));
    const row = await screen.findByRole('link', { name: 'rapor.md' }, { timeout: 30000 });
    expect(row).toBeInTheDocument();
    state.documentId = decodeURIComponent(row.getAttribute('href')!.split('/').pop()!);
    expect(screen.getAllByText('İndekslendi').length).toBeGreaterThan(0);
  });

  it('shows the knowledge base totals it just gained', async () => {
    routeParams = { kbId: state.kbId };
    ui(<KnowledgeBasePage />);
    await screen.findByRole('heading', { name: KB_NAME });
    const documents = await screen.findByText('Documents', { selector: '.stat-label' });
    const tile = documents.closest('.stat-tile')!;
    await waitFor(() => expect(within(tile as HTMLElement).getByText('1')).toBeInTheDocument());
  });
});

describe('Analysis', () => {
  it('reports the state, and inspects one method’s chunks', async () => {
    routeParams = { documentId: state.documentId };
    ui(<DocumentPage />);

    await screen.findByRole('heading', { name: 'rapor.md' });
    await screen.findByText('Analiz durumu');
    await waitFor(() => expect(screen.getAllByText('Hazır').length).toBeGreaterThan(0), {
      timeout: 120000,
    });

    // Every ready method is offered as something to inspect.
    const inspect = await waitFor(() => {
      const buttons = screen.getAllByRole('button', { name: /↗$/ });
      if (!buttons.length) throw new Error('no ready method offered');
      return buttons[0];
    });
    await userEvent.click(inspect);

    await screen.findByText('Yöntem parçaları');
    await waitFor(() => expect(screen.getByText(/parça$/)).toBeInTheDocument(), { timeout: 30000 });
    // Two of them: the header's × and the footer's. Either closes it.
    const close = screen.getAllByRole('button', { name: 'Kapat' });
    await userEvent.click(close[close.length - 1]);
    await waitFor(() => expect(screen.queryByText('Yöntem parçaları')).toBeNull());
  });

  it('shows the indexed chunks and the canonical units apart', async () => {
    routeParams = { documentId: state.documentId };
    ui(<DocumentPage />);

    await userEvent.click(await screen.findByRole('tab', { name: /İndekslenen parçalar/ }));
    await screen.findByText(
      'Bilgi tabanının kendi bölümleyicisinin ürettiği parçalar — arama bunları tarar.',
    );
    await waitFor(() => expect(screen.getAllByText(/^#\d/).length).toBeGreaterThan(0), {
      timeout: 30000,
    });

    // This document was not parsed by the structured parser, so `/units`
    // refuses. The screen says so rather than showing an empty list.
    await userEvent.click(screen.getByRole('tab', { name: /Kanonik birimler/ }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument(), { timeout: 30000 });
    expect(screen.getByRole('alert')).toHaveTextContent('Bulunamadı');
  });
});

describe('Search', () => {
  it('runs retrieval and shows the ranked chunks', async () => {
    window.localStorage.setItem('chat_rag.selected_kb', state.kbId);
    const view = ui(<SearchPage />);

    await waitFor(() =>
      expect((screen.getByLabelText('Knowledge Base') as HTMLSelectElement).value).toBe(state.kbId),
    );
    const methodSelect = screen.getByLabelText('Retrieval method') as HTMLSelectElement;
    await waitFor(() => expect(methodSelect.value).toBeTruthy());

    await userEvent.type(screen.getByLabelText('Sorgu'), 'net kâr');
    await userEvent.click(screen.getByRole('button', { name: 'Ara' }));

    const results = await waitFor(
      () => {
        const settled = view.container.querySelector('.chunk-list, .empty-state, .error-state, .inline-error');
        if (!settled) throw new Error('still running');
        return settled;
      },
      { timeout: 120000, interval: 250 },
    );
    expect(results).toHaveClass('chunk-list');
    expect(screen.getAllByText(/^skor /).length).toBeGreaterThan(0);
  });
});

describe('Sohbet', () => {
  it('answers an arbitrary question with citations', async () => {
    window.localStorage.setItem('chat_rag.selected_kb', state.kbId);
    ui(<ChatPage />);

    const box = await screen.findByPlaceholderText(/soru sorun/i);
    await waitFor(() => expect(box).toBeEnabled());
    await userEvent.type(box, '2024 net kârı ne kadar oldu?');
    await userEvent.click(screen.getByRole('button', { name: 'Gönder' }));

    await waitFor(() => expect(screen.getByText('Kaynaklar')).toBeInTheDocument(), { timeout: 180000 });
    expect(screen.getAllByText('rapor.md').length).toBeGreaterThan(0);
    expect(screen.getByText(/kaynak kullanıldı$/)).toBeInTheDocument();
    // Grounded either way, but the screen must say which.
    const grounded = screen.queryByText('Kaynağa dayalı') ?? screen.getByText('Kaynağa dayanmıyor');
    expect(grounded).toBeInTheDocument();
  });
});

describe('Deleting', () => {
  it('removes the document, then the knowledge base', async () => {
    routeParams = { kbId: state.kbId };
    ui(<KnowledgeBasePage />);

    await screen.findByRole('heading', { name: KB_NAME });
    await userEvent.click(screen.getByRole('tab', { name: /Documents/ }));
    await screen.findByRole('link', { name: 'rapor.md' }, { timeout: 30000 });
    await userEvent.click(screen.getAllByRole('button', { name: 'Sil' })[0]);
    await confirmDialog('Sil');
    await waitFor(() => expect(screen.queryByRole('link', { name: 'rapor.md' })).toBeNull(), {
      timeout: 30000,
    });

    await userEvent.click(screen.getByRole('tab', { name: 'Ayarlar' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Bilgi tabanını sil' }));
    await confirmDialog('Sil');
    await waitFor(() => expect(pushed.at(-1)).toBe('/knowledge-bases'), { timeout: 30000 });

    await expect(api.knowledgeBases.get(state.kbId)).rejects.toMatchObject({ type: 'not_found' });
  });
});
