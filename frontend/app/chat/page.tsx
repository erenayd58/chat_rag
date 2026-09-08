'use client';

/**
 * Sohbet -- ask anything, over one knowledge base.
 *
 * The two things the contract insists a client show are shown: `citations`,
 * each carrying the chunk verbatim and whether the answer actually cited it,
 * and `grounded`, which is false when the model answered without citing any of
 * them. A screen that shows the answer without those two cannot tell an answer
 * from a guess, so both are rendered on every answer -- including the
 * uncomfortable case where `grounded` is false.
 *
 * The refusals are told apart, because they need different things from the
 * person reading:
 *   **503 overloaded** nothing was queued and nothing was kept; the question
 *   is put back in the box so one click resends it after `Retry-After`;
 *   **504 timeout** the deadline passed -- ask something narrower;
 *   **503 unavailable** the answer model could not be reached, and retrieval
 *   still works, so Search is the honest thing to offer instead.
 */

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { ChunkDialog } from '@/components/ChunkDialog';
import { KbSelect } from '@/components/KbSelect';
import { useConfirm } from '@/components/Modal';
import { Badge } from '@/components/ui';
import api, { ApiError } from '@/lib/api';
import { explain } from '@/lib/errors';
import { formatPages, formatScore, formatSeconds, humanise } from '@/lib/format';
import { useSelectedKb } from '@/lib/hooks/useSelectedKb';
import type { Answer, Citation } from '@/types/api';

const TOP_K = 5;

interface Turn {
  id: number;
  question: string;
  at: string;
  answer?: Answer;
  refusal?: { kind: 'overloaded' | 'timeout' | 'unavailable' | 'other'; title: string; detail: string };
}

const clock = () =>
  new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });

export default function ChatPage() {
  const [kbId, setKbId] = useSelectedKb();
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState(false);
  const [open, setOpen] = useState<Citation | null>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const confirm = useConfirm();
  const nextId = useRef(1);

  useEffect(() => {
    const node = scroll.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns, pending]);

  const send = async () => {
    const asked = question.trim();
    if (!asked || !kbId || pending) return;
    const id = nextId.current++;
    setTurns((current) => [...current, { id, question: asked, at: clock() }]);
    setQuestion('');
    setPending(true);

    try {
      const answer = await api.asking.ask({
        question: asked,
        knowledge_base_id: kbId,
        top_k: TOP_K,
      });
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, answer } : turn)));
    } catch (error) {
      const refusal = describe(error);
      // Overload and timeout both refused the question rather than answering
      // it badly. Putting the text back is the difference between "try again"
      // and "type it again".
      if (refusal.kind === 'overloaded' || refusal.kind === 'timeout') setQuestion(asked);
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, refusal } : turn)));
    } finally {
      setPending(false);
      input.current?.focus();
    }
  };

  const clear = async () => {
    if (!turns.length) return;
    const ok = await confirm({
      title: 'Sohbeti temizle',
      message: 'Bu ekrandaki soru ve cevaplar silinsin mi?',
      confirmLabel: 'Temizle',
    });
    if (ok) setTurns([]);
  };

  return (
    <div className="chat-layout">
      <div className="chat-topbar">
        <label className="chat-topbar-label" htmlFor="kbSelect">
          Knowledge Base
        </label>
        <div style={{ width: 280 }}>
          <KbSelect value={kbId} onChange={setKbId} />
        </div>
        <span className="row-end" />
        <button type="button" className="btn btn-ghost btn-sm" onClick={clear} disabled={!turns.length}>
          Sohbeti temizle
        </button>
      </div>

      <div className="chat-scroll" ref={scroll}>
        <div className="chat-thread">
          {turns.length === 0 && !pending ? (
            <div className="empty-state">
              <h3>Dokümanlarınıza sorun</h3>
              <p>
                Yukarıdan bir bilgi tabanı seçin ve sorunuzu yazın. Cevap, dayandığı doküman
                bölümlerini kaynak olarak gösterir; hiçbirine dayanmıyorsa bunu da söyler.
              </p>
            </div>
          ) : null}

          {turns.map((turn) => (
            <div key={turn.id}>
              <div className="msg msg-user">
                <div className="msg-bubble">{turn.question}</div>
                <div className="msg-time">{turn.at}</div>
              </div>
              {turn.answer ? (
                <AnswerBlock answer={turn.answer} onOpen={setOpen} />
              ) : turn.refusal ? (
                <RefusalBlock refusal={turn.refusal} />
              ) : null}
            </div>
          ))}

          {pending ? (
            <div className="msg msg-assistant">
              <div className="msg-bubble" style={{ width: 'fit-content' }}>
                <span className="typing">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="chat-composer">
        <div className="composer-inner">
          <div className="composer-row">
            <input
              ref={input}
              className="input"
              value={question}
              disabled={!kbId || pending}
              placeholder="Bu bilgi tabanındaki dokümanlara bir soru sorun…"
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') send();
              }}
            />
            <button
              type="button"
              className="btn btn-primary"
              disabled={!kbId || pending || !question.trim()}
              onClick={send}
            >
              Gönder
            </button>
          </div>
          <div className="composer-hint">
            {!kbId
              ? 'Başlamak için bir bilgi tabanı seçin.'
              : pending
                ? 'Cevap bekleniyor…'
                : 'Cevaplar yalnız indekslenmiş dokümanlardan üretilir.'}
          </div>
        </div>
      </div>

      {open ? (
        <ChunkDialog
          title={`${open.label ? `${open.label} · ` : ''}${open.document ?? 'Kaynak'}`}
          text={open.content}
          facts={[
            { label: 'Doküman', value: open.document ?? '—' },
            { label: 'Bölüm', value: open.section ?? '—' },
            { label: 'Sayfa', value: formatPages(open.pages) ?? '—' },
            { label: 'Bölümleme', value: open.chunking_mode ? humanise(open.chunking_mode) : '—' },
            { label: 'Cevapta kullanıldı', value: open.used ? 'evet' : 'hayır' },
            { label: 'Skor', value: formatScore(open.score) },
            { label: 'Parça kimliği', value: open.chunk_id ?? '—' },
          ]}
          onClose={() => setOpen(null)}
        />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* One answer                                                          */
/* ------------------------------------------------------------------ */

function AnswerBlock({ answer, onOpen }: { answer: Answer; onOpen: (citation: Citation) => void }) {
  const used = answer.citations.filter((citation) => citation.used).length;
  return (
    <div className="msg msg-assistant">
      <div className="msg-bubble">{answer.answer || '(boş cevap)'}</div>

      <div className="msg-meta">
        {answer.grounded ? (
          <Badge tone="success" dot title="Cevap, gösterilen kaynaklardan en az birine atıf yaptı">
            Kaynağa dayalı
          </Badge>
        ) : (
          <Badge
            tone="warn"
            dot
            title="Model verilen kaynaklardan hiçbirine atıf yapmadı; cevabı kaynaklarla doğrulayın"
          >
            Kaynağa dayanmıyor
          </Badge>
        )}
        {answer.retrieval_method ? <Badge>{humanise(answer.retrieval_method)}</Badge> : null}
        {answer.citations.length ? (
          <Badge>
            {used}/{answer.citations.length} kaynak kullanıldı
          </Badge>
        ) : null}
        {answer.timing.total_seconds !== null ? (
          <span className="dim" style={{ fontSize: 11.5 }}>
            {formatSeconds(answer.timing.total_seconds)}
          </span>
        ) : null}
      </div>

      {!answer.grounded && answer.citations.length ? (
        <div className="msg-notice">
          <strong>Bu cevap kaynak göstermedi.</strong>
          Aşağıdaki bölümler modele verildi ama cevapta hiçbirine atıf yapılmadı. Cevabı doğrudan
          kaynaklardan doğrulayın.
        </div>
      ) : null}

      {answer.citations.length ? (
        <div className="sources-block">
          <div className="sources-label">Kaynaklar</div>
          {answer.citations.map((citation, index) => (
            <button
              key={citation.chunk_id ?? index}
              type="button"
              className={`source-card${citation.used ? ' source-card-used' : ''}`}
              onClick={() => onOpen(citation)}
            >
              <div className="source-head">
                {citation.label ? <span className="source-label">{citation.label}</span> : null}
                <span className="source-doc">{citation.document ?? 'Doküman'}</span>
                {formatPages(citation.pages) ? <Badge>{formatPages(citation.pages)}</Badge> : null}
                {citation.used ? <Badge tone="success">cevapta kullanıldı</Badge> : null}
              </div>
              {citation.section ? <div className="source-section">{citation.section}</div> : null}
              {citation.content ? <div className="source-preview">{citation.content}</div> : null}
            </button>
          ))}
        </div>
      ) : (
        <div className="msg-notice">
          <strong>Kaynak bulunamadı.</strong>
          Bu soru için eşleşen bir doküman bölümü getirilemedi.
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* One refusal                                                         */
/* ------------------------------------------------------------------ */

function describe(error: unknown): NonNullable<Turn['refusal']> {
  const explanation = explain(error);
  if (error instanceof ApiError) {
    if (error.type === 'overloaded') {
      return { kind: 'overloaded', title: explanation.title, detail: explanation.detail };
    }
    if (error.type === 'timeout') {
      return { kind: 'timeout', title: explanation.title, detail: explanation.detail };
    }
    if (error.type === 'unavailable') {
      return { kind: 'unavailable', title: explanation.title, detail: error.message };
    }
  }
  return { kind: 'other', title: explanation.title, detail: `${explanation.detail}` };
}

function RefusalBlock({ refusal }: { refusal: NonNullable<Turn['refusal']> }) {
  if (refusal.kind === 'unavailable') {
    return (
      <div className="msg msg-assistant">
        <div className="msg-notice">
          <strong>Cevap üretimi şu anda kullanılamıyor.</strong>
          Dil modeline ulaşılamadı. Getirme (retrieval) çalışmaya devam ediyor: bu soruya hangi
          bölümlerin geleceğini <Link href="/search">Search</Link> ekranında görebilirsiniz.
          <div className="dim" style={{ marginTop: 6, fontSize: 12 }}>
            {refusal.detail}
          </div>
        </div>
      </div>
    );
  }
  if (refusal.kind === 'overloaded' || refusal.kind === 'timeout') {
    return (
      <div className="msg msg-assistant">
        <div className="msg-notice">
          <strong>{refusal.title}</strong>
          {refusal.detail} Sorunuz kaybolmadı; kutuya geri kondu.
        </div>
      </div>
    );
  }
  return (
    <div className="msg msg-assistant">
      <div className="msg-error">
        <strong>{refusal.title}</strong>
        {refusal.detail}
      </div>
    </div>
  );
}
