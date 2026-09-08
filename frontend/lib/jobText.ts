/**
 * What an ingest job's state means to a person.
 *
 * Every terminal state that is not `succeeded` says the same two things: what
 * happened, and whether the document was saved. The contract is explicit that
 * a job's last act is the ledger write, so "the document was not saved" is a
 * fact here and not a hedge.
 */

import type { IngestJob } from '@/types/api';
import { FORGOTTEN_JOB, type FollowedJob } from './jobs';

/** A line for a job that is still queued or running. */
export function progressNote(job: IngestJob): string {
  const methods = job.methods?.length ?? 0;
  if (job.status === 'queued') {
    return job.queue_position
      ? `Sırada bekliyor (sıra: ${job.queue_position})…`
      : 'Sırada bekliyor…';
  }
  if (job.status === 'running') {
    return methods > 1
      ? `Doküman okunuyor, ardından ${methods} yöntem çalıştırılıyor…`
      : 'Doküman okunuyor ve indeksleniyor…';
  }
  return job.status ?? '';
}

/** Why a job that did not succeed did not succeed. */
export function failureNote(job: FollowedJob): string {
  if (job.status === FORGOTTEN_JOB) {
    return 'Bu yüklemenin durumu artık saklanmıyor. Dokümanın listede olup olmadığına bakın.';
  }
  const record = job as IngestJob;
  const detail = record.error?.message ? ` ${record.error.message}` : '';
  switch (record.status) {
    case 'timed_out':
      return `Yükleme zaman aşımına uğradı; doküman kaydedilmedi.${detail}`;
    case 'cancelled':
      return 'Yükleme iptal edildi; doküman kaydedilmedi.';
    case 'interrupted':
      return 'Sunucu bu yükleme işlenirken yeniden başlatıldı; doküman kaydedilmedi. Dosyayı tekrar yükleyin.';
    default:
      return `Yükleme başarısız oldu.${detail || ' Bilinmeyen hata.'}`;
  }
}

/** The success line, from the job's own result. */
export function successNote(job: IngestJob): string {
  const name = job.name ?? 'Doküman';
  const chunks = job.result?.chunk_count;
  return chunks === null || chunks === undefined
    ? `${name} yüklendi.`
    : `${name} yüklendi — ${chunks} parça.`;
}

export function statusTone(status: string | null | undefined): 'neutral' | 'warn' | 'success' | 'danger' {
  if (status === 'succeeded') return 'success';
  if (status === 'running') return 'warn';
  if (status === 'queued') return 'neutral';
  if (!status) return 'neutral';
  return 'danger';
}
