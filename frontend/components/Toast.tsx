'use client';

/** Transient confirmations. Never the only place a failure is reported. */

import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';

type Tone = 'info' | 'success' | 'error';

interface ToastMessage {
  id: number;
  text: string;
  tone: Tone;
}

interface ToastApi {
  show: (text: string, tone?: Tone) => void;
  success: (text: string) => void;
  error: (text: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const LIFETIME: Record<Tone, number> = { info: 4000, success: 4000, error: 6500 };

export function ToastProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ToastMessage[]>([]);
  const nextId = useRef(1);

  const show = useCallback((text: string, tone: Tone = 'info') => {
    const id = nextId.current++;
    setMessages((current) => [...current, { id, text, tone }]);
    setTimeout(() => {
      setMessages((current) => current.filter((message) => message.id !== id));
    }, LIFETIME[tone]);
  }, []);

  const api = useMemo<ToastApi>(
    () => ({
      show,
      success: (text: string) => show(text, 'success'),
      error: (text: string) => show(text, 'error'),
    }),
    [show],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="toast-container" aria-live="polite" aria-atomic="false">
        {messages.map((message) => (
          <div key={message.id} className={`toast${message.tone === 'info' ? '' : ` toast-${message.tone}`}`}>
            {message.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (!api) throw new Error('useToast must be used inside <ToastProvider>');
  return api;
}
