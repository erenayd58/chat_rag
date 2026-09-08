/**
 * The one architectural rule this front end has to keep.
 *
 * `docs/api-v1.md`: *there is no second method catalogue -- not in this
 * repository, not in the console's JavaScript, and not in whatever the front
 * end becomes.* Two tests hold that. The first renders the picker against a
 * catalogue of invented method keys and expects every one of them on screen,
 * which cannot pass if the component knows any real ones. The second reads the
 * source and fails if a real key is written down anywhere in it.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MethodPicker } from '@/components/MethodPicker';

const method = (over: Record<string, unknown>) => ({
  key: 'k',
  label: 'L',
  summary: 's',
  engine: 'e',
  available: true,
  unavailable_reason: null,
  uses_model: false,
  default: false,
  orchestration: false,
  baseline: null,
  ...over,
});

function catalogue(items: unknown[]) {
  return vi.fn().mockResolvedValue(
    new Response(JSON.stringify({ items, page: { offset: 0, limit: items.length, total: items.length } }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('MethodPicker', () => {
  it('renders whatever the registry answers, including keys it has never seen', async () => {
    vi.stubGlobal(
      'fetch',
      catalogue([
        method({ key: 'newly-invented', label: 'Newly Invented', summary: 'a method added today' }),
        method({ key: 'second-one', label: 'Second One', summary: 'and another' }),
      ]),
    );

    render(<MethodPicker selected={[]} onChange={() => {}} />);

    expect(await screen.findByText('Newly Invented')).toBeInTheDocument();
    expect(screen.getByText('Second One')).toBeInTheDocument();
    expect(screen.getByText('a method added today')).toBeInTheDocument();
  });

  it('preselects exactly the methods the registry marks default', async () => {
    vi.stubGlobal(
      'fetch',
      catalogue([
        method({ key: 'one', label: 'One' }),
        method({ key: 'two', label: 'Two', default: true }),
      ]),
    );
    const onChange = vi.fn();

    render(<MethodPicker selected={[]} onChange={onChange} />);

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(['two']));
  });

  it('shows an unavailable method disabled, with the server reason, rather than hiding it', async () => {
    vi.stubGlobal(
      'fetch',
      catalogue([
        method({
          key: 'blocked',
          label: 'Blocked',
          available: false,
          unavailable_reason: 'no model configured',
        }),
      ]),
    );

    render(<MethodPicker selected={[]} onChange={() => {}} />);

    expect(await screen.findByText('no model configured')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Blocked/ })).toBeDisabled();
  });

  it('marks an orchestration as one, because it runs over a baseline', async () => {
    vi.stubGlobal(
      'fetch',
      catalogue([
        method({ key: 'base', label: 'Base' }),
        method({ key: 'over', label: 'Over', orchestration: true, baseline: 'base', uses_model: true }),
      ]),
    );

    render(<MethodPicker selected={[]} onChange={() => {}} />);

    expect(await screen.findByText('orkestrasyon')).toBeInTheDocument();
    expect(screen.getByText('model destekli')).toBeInTheDocument();
  });
});

/* ------------------------------------------------------------------ */
/* No catalogue in the source                                          */
/* ------------------------------------------------------------------ */

/** The keys this deployment's registry answers with today. */
const REGISTERED_METHOD_KEYS = ['structure-only', 'markdown', 'agentic', 'hybrid'];

const ROOTS = ['app', 'components', 'lib'];
const SOURCE = /\.(ts|tsx)$/;

function sourceFiles(root: string, found: string[] = []): string[] {
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    if (statSync(path).isDirectory()) sourceFiles(path, found);
    else if (SOURCE.test(entry)) found.push(path);
  }
  return found;
}

describe('the method catalogue lives in the registry', () => {
  it('names no chunking method key anywhere in the source', () => {
    const offences: string[] = [];
    for (const root of ROOTS) {
      for (const file of sourceFiles(root)) {
        const text = readFileSync(file, 'utf8');
        for (const key of REGISTERED_METHOD_KEYS) {
          // A quoted key is a catalogue entry; the word inside prose is not.
          if (text.includes(`'${key}'`) || text.includes(`"${key}"`)) {
            offences.push(`${file}: ${key}`);
          }
        }
      }
    }
    expect(offences).toEqual([]);
  });
});
