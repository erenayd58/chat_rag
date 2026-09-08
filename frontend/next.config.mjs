/**
 * The console is served by Next.js and speaks only to `/api/v1`.
 *
 * The browser never learns where the backend is: every `/api/v1/...` request
 * goes to this server's own origin and is rewritten to the application. That
 * keeps the API same-origin (no CORS, no preflight on an upload) and leaves
 * exactly one place -- `CHAT_RAG_API_URL` -- that knows the backend's address.
 */

import { fileURLToPath } from 'node:url';

const backend = process.env.CHAT_RAG_API_URL || 'http://127.0.0.1:5005';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // This directory, not whatever ancestor happens to hold another lockfile.
  outputFileTracingRoot: fileURLToPath(new URL('.', import.meta.url)),
  async rewrites() {
    return [{ source: '/api/v1/:path*', destination: `${backend}/api/v1/:path*` }];
  },
};

export default nextConfig;
