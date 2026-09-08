import type { Metadata, Viewport } from 'next';
import { AppShell } from '@/components/AppShell';
import { ConfirmProvider } from '@/components/Modal';
import { ToastProvider } from '@/components/Toast';
import './globals.css';

export const metadata: Metadata = {
  title: 'RAG Console',
  description: 'Bilgi tabanları, dokümanlar ve belgelerinizden kaynaklı cevaplar.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body>
        <ToastProvider>
          <ConfirmProvider>
            <AppShell>{children}</AppShell>
          </ConfirmProvider>
        </ToastProvider>
      </body>
    </html>
  );
}
