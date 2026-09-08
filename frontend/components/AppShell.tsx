'use client';

/**
 * The shell: navigation, and one honest line about the server.
 *
 * The health line is `GET /api/v1/health` and nothing else -- state, readiness
 * and the capacity the contract publishes. It is not a made-up uptime figure
 * and it goes grey rather than green when the request itself fails.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState, type ReactNode } from 'react';
import api from '@/lib/api';
import type { Health } from '@/types/api';

const HEALTH_INTERVAL_MS = 30000;

const NAV: { href: string; label: string; icon: ReactNode; section?: string }[] = [
  {
    href: '/knowledge-bases',
    label: 'Knowledge Bases',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
        <path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" />
      </svg>
    ),
  },
  {
    href: '/chat',
    label: 'Sohbet',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    href: '/search',
    label: 'Search',
    section: 'Investigations',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="7" />
        <path d="M21 21l-4.3-4.3" />
      </svg>
    ),
  },
  {
    href: '/viewer',
    label: 'Viewer',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M12 3v18" />
        <path d="M6 8h3M6 12h3M6 16h3" />
        <path d="M15 8h3M15 13h3" />
      </svg>
    ),
  },
  {
    href: '/analysis',
    label: 'Analysis',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 2v6L4.5 18.5A2 2 0 0 0 6.24 21h11.52a2 2 0 0 0 1.74-2.5L14 8V2" />
        <path d="M8.5 2h7" />
        <path d="M7 15h10" />
      </svg>
    ),
  },
];

function HealthLine() {
  const [health, setHealth] = useState<Health | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    const controller = new AbortController();
    const read = async () => {
      try {
        const state = await api.meta.health(controller.signal);
        if (!live) return;
        setHealth(state);
        setFailed(false);
      } catch {
        if (!live) return;
        setFailed(true);
      }
    };
    read();
    const timer = setInterval(read, HEALTH_INTERVAL_MS);
    return () => {
      live = false;
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  if (failed) {
    return (
      <div className="health-line" title="GET /api/v1/health yanıt vermedi">
        <span className="health-dot down" />
        Sunucuya ulaşılamıyor
      </div>
    );
  }
  if (!health) {
    return (
      <div className="health-line">
        <span className="health-dot" />
        Durum okunuyor…
      </div>
    );
  }

  const tone = health.ready ? 'ok' : health.state === 'down' ? 'down' : 'warn';
  const capacity = health.capacity;
  return (
    <div
      className="health-line"
      title={`Yükleme: ${capacity.ingest.running} çalışıyor, ${capacity.ingest.queued}/${capacity.ingest.queue_capacity} kuyrukta · Sorgu: ${capacity.query.active}/${capacity.query.max_active}`}
    >
      <span className={`health-dot ${tone}`} />
      {health.ready ? 'Hazır' : health.state}
      {capacity.ingest.running > 0 ? ` · ${capacity.ingest.running} yükleme` : ''}
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? '';

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-name">RAG Console</div>
            <div className="brand-sub">Doküman analizi</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((item) => (
            <div key={item.href}>
              {item.section ? <div className="nav-section">{item.section}</div> : null}
              <Link
                href={item.href}
                className={`nav-link${pathname.startsWith(item.href) ? ' active' : ''}`}
              >
                {item.icon}
                <span>{item.label}</span>
              </Link>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <HealthLine />
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
