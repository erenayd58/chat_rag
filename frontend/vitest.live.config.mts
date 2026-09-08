import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

/**
 * The live suite: the real screens against a real backend.
 *
 * Separate from `vitest.config.mts` on purpose -- `npm test` has to pass with
 * nothing running, and a suite that needs a server would make that untrue. Run
 * it with the console on :3000 and the application behind it:
 *
 *     npm run test:live
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('.', import.meta.url)) },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/live/setup.ts'],
    include: ['tests/live/**/*.test.tsx'],
    testTimeout: 300000,
    hookTimeout: 300000,
    // One knowledge base at a time: the flows create and delete real records.
    fileParallelism: false,
    sequence: { concurrent: false },
  },
});
