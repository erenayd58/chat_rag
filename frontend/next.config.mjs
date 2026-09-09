/**
 * The console is served by Next.js and speaks only to `/api/v1`.
 *
 * The browser never learns where the backend is: every `/api/v1/...` request
 * goes to this server's own origin, and this server forwards it. That keeps
 * the API same-origin (no CORS, no preflight on an upload) and leaves exactly
 * one place -- `CHAT_RAG_API_URL` -- that knows the application's address.
 *
 * The forwarding used to be a `rewrites()` entry here. It is
 * `app/api/v1/[...path]/route.ts` now, because Next.js resolves `rewrites()`
 * during `next build` and freezes the destination into the build output: the
 * image carried the developer default `http://127.0.0.1:5005`, and
 * `CHAT_RAG_API_URL` on the container changed nothing. That module explains
 * the failure in full.
 */

import { fileURLToPath } from 'node:url';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // A self-contained server plus only the node_modules it actually reaches,
  // which is what the container image copies instead of the whole install.
  output: 'standalone',
  // This directory, not whatever ancestor happens to hold another lockfile.
  outputFileTracingRoot: fileURLToPath(new URL('.', import.meta.url)),
};

export default nextConfig;
