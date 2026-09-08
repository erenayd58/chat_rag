/**
 * What a refusal means to the person looking at the screen.
 *
 * One table, keyed by the contract's `type`, so every screen says the same
 * thing about the same refusal and a reworded server message never changes
 * what a user is told. Each entry carries a title, what to do about it, and
 * whether trying again is worth anything.
 */

import { ApiError } from './api/client';
import type { ApiErrorType } from '@/types/api';

export interface Explanation {
  title: string;
  detail: string;
  /** True when the same request, sent again later, could work. */
  retryable: boolean;
}

const TABLE: Record<ApiErrorType, Explanation> = {
  invalid_request: {
    title: 'İstek kabul edilmedi',
    detail: 'Gönderilen bilgi eksik ya da geçersiz. Alanları kontrol edip tekrar deneyin.',
    retryable: false,
  },
  not_found: {
    title: 'Bulunamadı',
    detail: 'Aradığınız kayıt burada değil. Silinmiş ya da hiç oluşturulmamış olabilir.',
    retryable: false,
  },
  not_ready: {
    title: 'Henüz hazır değil',
    detail: 'Kayıt var ama işlemi bitmedi. Birazdan tekrar bakın.',
    retryable: true,
  },
  conflict: {
    title: 'Bu işlem şu anda yapılamıyor',
    detail: 'Kaydın mevcut durumu bu işleme izin vermiyor.',
    retryable: false,
  },
  unavailable: {
    title: 'Bu özellik şu anda kullanılamıyor',
    detail: 'Sunucu bu yeteneği şu an sağlayamıyor.',
    retryable: true,
  },
  overloaded: {
    title: 'Sunucu şu anda meşgul',
    detail: 'İstek kuyruğa alınmadı, reddedildi. Hiçbir şey kaybolmadı; biraz sonra tekrar deneyin.',
    retryable: true,
  },
  timeout: {
    title: 'İşlem zaman aşımına uğradı',
    detail: 'Süre sınırı doldu ve işlem durduruldu. Daha dar bir istekle tekrar deneyebilirsiniz.',
    retryable: true,
  },
  internal: {
    title: 'Sunucu hatası',
    detail: 'Sunucu bu isteği tamamlayamadı.',
    retryable: true,
  },
  network: {
    title: 'Sunucuya ulaşılamadı',
    detail: 'Bağlantı kurulamadı. Sunucunun çalıştığından emin olun.',
    retryable: true,
  },
};

const UNKNOWN: Explanation = {
  title: 'Beklenmeyen bir hata',
  detail: 'İstek tamamlanamadı.',
  retryable: true,
};

/** The refusal, explained. Anything that is not an `ApiError` is unknown. */
export function explain(error: unknown): Explanation {
  if (!(error instanceof ApiError)) return UNKNOWN;
  const known = TABLE[error.type] ?? UNKNOWN;
  if (error.type === 'overloaded' && error.retryAfter) {
    return {
      ...known,
      detail: `İstek kuyruğa alınmadı, reddedildi. Yaklaşık ${Math.round(
        error.retryAfter,
      )} saniye sonra tekrar deneyin.`,
    };
  }
  return known;
}

/** One line: what happened, in the server's own words where it had any. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Title and message together, for a toast. */
export function toastMessage(error: unknown): string {
  const { title } = explain(error);
  const message = errorMessage(error);
  return message && message !== title ? `${title}: ${message}` : title;
}
