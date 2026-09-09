/**
 * Every request this front end makes is on the contract.
 *
 * `docs/legacy-removal.md` wave 1: a legacy route comes out when the screen
 * that called it is served by the new front end and calls `/api/v1`. This is
 * the half of that promise this repository can hold -- the source may not name
 * a Flask-era path, and every path it does name must be one `/api/v1` serves.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOTS = ['app', 'components', 'lib', 'types'];
const SOURCE = /\.(ts|tsx)$/;

/** Flask-era paths, from the legacy removal map. */
const LEGACY_PATHS = [
  '/api/kb',
  '/api/documents',
  '/api/ingest/jobs',
  '/api/query',
  '/api/chunks',
  '/api/goldset',
  '/api/stats',
  '/api/models',
  '/api/retrieval/capabilities',
  '/api/experiment',
  '/api/demo',
  '/api/ops',
  '/api/health',
];

/** What `/api/v1` serves, as templates. Kept in step with docs/api-v1.md. */
const CONTRACT = [
  '/meta/chunking-methods',
  '/meta/retrieval-methods',
  '/meta/models',
  '/health',
  '/knowledge-bases',
  '/documents',
  '/ingest-jobs',
  '/queries',
  '/searches',
  '/analysis-queries',
];

function sourceFiles(root: string, found: string[] = []): string[] {
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    if (statSync(path).isDirectory()) sourceFiles(path, found);
    else if (SOURCE.test(entry)) found.push(path);
  }
  return found;
}

const files = ROOTS.flatMap((root) => sourceFiles(root));

describe('the console speaks /api/v1 and nothing else', () => {
  it('names no legacy route', () => {
    const offences: string[] = [];
    for (const file of files) {
      const text = readFileSync(file, 'utf8');
      for (const legacy of LEGACY_PATHS) {
        // `/api/v1/...` is not a legacy path even though it starts the same.
        const pattern = new RegExp(`['"\`]${legacy}(?![\\w-])`, 'g');
        if (pattern.test(text)) offences.push(`${file}: ${legacy}`);
      }
    }
    expect(offences).toEqual([]);
  });

  it('calls fetch from exactly one module in the browser, and one on the server', () => {
    // Two, and they are different jobs. `client.ts` is how a *screen* reaches
    // the contract, and the rule that there is only one of those is what makes
    // the refusal taxonomy a single translation. `proxy.ts` is not a screen: it
    // is this server forwarding the console's own origin to the application, so
    // the browser never learns the backend's address. Anything else that calls
    // `fetch` is a component that has gone around `client.ts`.
    const ALLOWED = [join('lib', 'api', 'client.ts'), join('lib', 'api', 'proxy.ts')];
    const callers = files.filter(
      (file) => /(?<![.\w])fetch\(/.test(readFileSync(file, 'utf8')) && !ALLOWED.includes(file),
    );
    expect(callers).toEqual([]);
  });

  it('builds every path it uses out of the contract', () => {
    const paths = new Set<string>();
    for (const file of files) {
      const text = readFileSync(file, 'utf8');
      const literal = /(?:const BASE = |request<[^>]*>\()['"`](\/[a-z0-9-]+(?:\/[a-z0-9-]+)*)/g;
      for (const match of text.matchAll(literal)) {
        paths.add(match[1]);
      }
    }
    for (const path of paths) expect(CONTRACT).toContain(path);
  });
});
