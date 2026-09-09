/**
 * `/api/v1/*` on the console's origin, served by forwarding to the application.
 *
 * The handler itself is `lib/api/proxy.ts`; this file is only the App Router's
 * way of asking for it. Node runtime because the forward streams a request
 * body, and dynamic because the answer depends on the request and on
 * `CHAT_RAG_API_URL` as it is *now* -- nothing here may be cached or
 * pre-rendered.
 */

import { forward } from '@/lib/api/proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type Context = { params: Promise<{ path: string[] }> };

async function handler(request: Request, context: Context): Promise<Response> {
  const { path } = await context.params;
  return forward(request, path);
}

export const GET = handler;
export const POST = handler;
export const PATCH = handler;
export const PUT = handler;
export const DELETE = handler;
export const HEAD = handler;
export const OPTIONS = handler;
