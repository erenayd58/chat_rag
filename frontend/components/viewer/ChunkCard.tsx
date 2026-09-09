'use client';

/**
 * What one chunk is, and why it starts where it does.
 *
 * The card answers three questions in the order a reader asks them: which
 * section this is, why a boundary was drawn here, and — when Deep Analysis
 * recorded a decision at this boundary — what it did and which structural
 * problem that removed. The technical identity (chunk id, engine, split
 * strategies, boundary code) is folded away, because it is the answer to a
 * fourth question that only comes up when the first three do not settle it.
 *
 * Everything here is *recorded*. A field the packager did not write is absent
 * rather than filled with a plausible value.
 */

import {
  DECISION_TEXT,
  REASON_LONG,
  formatPages,
  sectionOf,
  smellNames,
  sourceSection,
  type Chunk,
  type ViewerDoc,
} from '@/lib/viewer/model';
import type { AnalysisSource } from '@/types/api';
import { Popover, type Anchor } from './Menu';

export function ChunkCard({
  doc,
  method,
  methodLabel,
  index,
  anchor,
  onClose,
}: {
  doc: ViewerDoc;
  method: string;
  methodLabel: string;
  index: number;
  anchor: Anchor;
  onClose: () => void;
}) {
  const chunk: Chunk | undefined = doc.arms[method]?.chunks[index];
  if (!chunk) return null;

  const section = sectionOf(chunk);
  const reason = REASON_LONG[chunk.rs] ?? null;
  const decision = doc.story && chunk.dec ? DECISION_TEXT[chunk.dec.status] : null;
  const removed = chunk.dec ? smellNames(chunk.dec.removed_smells) : '';
  const continuation =
    chunk.rt === 'TOKEN_BUDGET_CONTINUATION' ? 'önceki parçanın bütçe devamı' : (chunk.rt ?? '—');

  return (
    <Popover anchor={anchor} onClose={onClose}>
      <div className="t">
        Parça {chunk.num}
        <span className="mth">{methodLabel}</span>
      </div>
      <div className="meta">
        {[formatPages(chunk.pg), `${chunk.n} token`].filter(Boolean).join(' · ')}
      </div>
      {section ? (
        <div className="row">
          <span className="k">Bölüm</span>
          {section}
        </div>
      ) : null}
      {reason ? (
        <div className="row">
          <span className="k">Neden burada başladı?</span>
          {reason}
        </div>
      ) : null}
      {decision ? (
        <div className="deepbx">
          <b>Deep Analysis</b> — {decision}
          {removed ? (
            <>
              <br />
              Giderilen: {removed}.
            </>
          ) : null}
        </div>
      ) : null}
      <details>
        <summary>Teknik ayrıntı</summary>
        <dl>
          <dt>id</dt>
          <dd>{chunk.id}</dd>
          <dt>motor</dt>
          <dd>{doc.arms[method]?.kind}</dd>
          {chunk.st?.length ? (
            <>
              <dt>strateji</dt>
              <dd>{chunk.st.join(', ')}</dd>
            </>
          ) : null}
          <dt>birim</dt>
          <dd>{chunk.u.length}</dd>
          <dt>sınır kodu</dt>
          <dd>{chunk.rs}</dd>
          <dt>devam</dt>
          <dd>{continuation}</dd>
        </dl>
      </details>
    </Popover>
  );
}

/**
 * A retrieved passage, in the same card, with a way into İncele.
 *
 * The jump is offered only when this document's payload actually carries that
 * chunk — a source from a method the reader is not looking at is still a
 * source, and a link that would land nowhere is worse than no link.
 */
export function SourceCard({
  source,
  methodLabel,
  anchor,
  onClose,
  onJump,
}: {
  source: AnalysisSource;
  methodLabel: string;
  anchor: Anchor;
  onClose: () => void;
  onJump: (() => void) | null;
}) {
  const section = sourceSection(source);
  return (
    <Popover anchor={anchor} width={360} onClose={onClose}>
      <div className="t">
        {source.label ?? 'Kaynak'}
        <span className="mth">{methodLabel}</span>
      </div>
      <div className="meta">
        {[
          formatPages(source.pages),
          `${source.token_count} token`,
          source.used ? 'cevapta kullanıldı' : '',
        ]
          .filter(Boolean)
          .join(' · ')}
      </div>
      {section ? (
        <div className="row">
          <span className="k">Bölüm</span>
          {section}
        </div>
      ) : null}
      <div className="ptxt">{source.text ?? ''}</div>
      {onJump ? (
        <div className="pfoot">
          <button
            type="button"
            className="qjump"
            onClick={() => {
              onClose();
              onJump();
            }}
          >
            İncele görünümünde aç →
          </button>
        </div>
      ) : null}
    </Popover>
  );
}
