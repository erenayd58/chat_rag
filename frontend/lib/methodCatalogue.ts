/**
 * The chunking catalogue, fetched once per page load.
 *
 * Every method chip on a screen needs the registry's label for a key, and a
 * document list can hold thirty of them; without this each one would issue its
 * own `GET /api/v1/meta/chunking-methods` and the page would open with thirty
 * identical requests in flight. The catalogue is a deployment-level fact that
 * does not change while a page is open, so the *promise* is shared and every
 * caller awaits the same answer.
 *
 * A failure is not cached -- the next caller retries -- and `reset()` drops it,
 * which is what a test needs between two catalogues and what `reload()` on the
 * hook needs to mean anything.
 */

import * as meta from './api/meta';
import type { ChunkingMethod } from '@/types/api';

let inFlight: Promise<ChunkingMethod[]> | null = null;

export function chunkingMethods(): Promise<ChunkingMethod[]> {
  if (!inFlight) {
    inFlight = meta.chunkingMethods().catch((error) => {
      inFlight = null;
      throw error;
    });
  }
  return inFlight;
}

/** Forget it, so the next caller asks the server again. */
export function reset(): void {
  inFlight = null;
}
