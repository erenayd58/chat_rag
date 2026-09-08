import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import { reset } from '@/lib/methodCatalogue';

afterEach(() => {
  cleanup();
  // The chunking catalogue is shared for a page load, not for a run.
  reset();
});
