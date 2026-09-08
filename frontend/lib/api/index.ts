/**
 * The API layer's front door.
 *
 * A screen imports `api` and calls `api.documents.upload(...)`. It does not
 * import `fetch`, a path, or a status code -- `lib/api/client.ts` owns all
 * three, so moving a route or changing a refusal is one edit in one package.
 */

import * as asking from './asking';
import * as documents from './documents';
import * as ingestJobs from './ingestJobs';
import * as knowledgeBases from './knowledgeBases';
import * as meta from './meta';

export { ApiError, API_PREFIX, MAX_PAGE_SIZE, collect, queryString, request } from './client';

export const api = { asking, documents, ingestJobs, knowledgeBases, meta };

export default api;
