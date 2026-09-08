'use client';

/**
 * The uploads still in flight for one knowledge base.
 *
 * A refresh must not lose sight of an ingestion that is still running, so the
 * list is read from `GET /api/v1/ingest-jobs?active=true` rather than kept in
 * the page's memory. When the last active job disappears the caller is told
 * once, which is when the document list is worth reloading.
 */

import { useEffect, useRef, useState } from 'react';
import api from '@/lib/api';
import { progressNote } from '@/lib/jobText';
import { Badge } from './ui';
import type { IngestJob } from '@/types/api';

const POLL_MS = 2000;

export function IngestJobs({
  knowledgeBaseId,
  onDrained,
  refreshToken,
}: {
  knowledgeBaseId: string;
  /** Called once, when the last job in flight settles. */
  onDrained: () => void;
  /** Bump to poll again immediately -- after an upload was accepted. */
  refreshToken?: number;
}) {
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const had = useRef(false);
  const drained = useRef(onDrained);
  drained.current = onDrained;

  useEffect(() => {
    let live = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const found = await api.ingestJobs.list({
          knowledgeBaseId,
          activeOnly: true,
          signal: controller.signal,
        });
        if (!live) return;
        setJobs(found.items);
        if (found.items.length) had.current = true;
        else if (had.current) {
          had.current = false;
          drained.current();
        }
      } catch {
        // A failed poll is not worth a banner: the next one is two seconds
        // away and the document list is the record either way.
        if (!live) return;
      }
      if (live) timer = setTimeout(poll, POLL_MS);
    };

    poll();
    return () => {
      live = false;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [knowledgeBaseId, refreshToken]);

  if (!jobs.length) return null;

  return (
    <div className="ingest-jobs">
      {jobs.map((job) => (
        <div className="ingest-job" key={job.id ?? ''}>
          <Badge tone={job.status === 'running' ? 'warn' : 'neutral'} dot>
            {job.status === 'running' ? 'İşleniyor' : 'Sırada'}
          </Badge>
          <span className="ingest-job-name">{job.name ?? job.id}</span>
          <span className="ingest-job-note">{progressNote(job)}</span>
          {job.attached_uploads > 0 ? (
            <Badge tone="accent" title="Aynı içeriğin başka yüklemeleri bu işe bağlandı">
              +{job.attached_uploads} bağlı yükleme
            </Badge>
          ) : null}
        </div>
      ))}
    </div>
  );
}
