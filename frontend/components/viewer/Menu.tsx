'use client';

/**
 * The breadcrumb's dropdown, and the card that explains a chunk.
 *
 * Both are anchored to something the reader clicked and both are dismissed by
 * Escape or by a click outside, so they are one file: the positioning is the
 * only interesting part and doing it twice is how the two drift apart.
 *
 * Position is measured from the anchor's rectangle at open time and clamped to
 * the viewport — a chunk in the right-hand column opens its card to the left,
 * and one near the bottom opens upwards, rather than either being cut off.
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';

export interface Anchor {
  rect: DOMRect;
}

export function anchorOf(element: HTMLElement): Anchor {
  return { rect: element.getBoundingClientRect() };
}

/** A dropdown under its anchor: the knowledge-base and document pickers. */
export function Menu({
  anchor,
  onClose,
  children,
}: {
  anchor: Anchor;
  onClose: () => void;
  children: ReactNode;
}) {
  const menu = useRef<HTMLDivElement | null>(null);
  const [style, setStyle] = useState<{ left: number; top: number }>({
    left: anchor.rect.left,
    top: anchor.rect.bottom + 6,
  });

  useLayoutEffect(() => {
    const width = menu.current?.offsetWidth ?? 290;
    setStyle({
      left: Math.max(8, Math.min(anchor.rect.left, window.innerWidth - width - 12)),
      top: anchor.rect.bottom + 6,
    });
  }, [anchor]);

  useEscape(onClose);

  return (
    <div
      className="v-layer"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="v-menu" role="menu" ref={menu} style={style}>
        {children}
      </div>
    </div>
  );
}

/**
 * A floating card beside its anchor: a chunk's detail, a retrieved source.
 *
 * No backdrop, deliberately. The reader is comparing this card with the text
 * behind it — a scrim would hide the very thing the card is about — so it
 * closes on Escape, on scroll, and on a click that is not inside it.
 */
export function Popover({
  anchor,
  width = 312,
  onClose,
  children,
}: {
  anchor: Anchor;
  width?: number;
  onClose: () => void;
  children: ReactNode;
}) {
  const card = useRef<HTMLDivElement | null>(null);
  const [style, setStyle] = useState<{ left: number; top: number }>({ left: -9999, top: -9999 });

  useLayoutEffect(() => {
    const rect = anchor.rect;
    const height = card.current?.offsetHeight ?? 240;
    const w = Math.min(width, window.innerWidth - 20);
    let left = rect.right + 14;
    let top = Math.max(64, Math.min(rect.top, window.innerHeight - height - 12));
    if (left + w > window.innerWidth - 10) left = Math.max(10, rect.left - w - 14);
    if (left < 10) {
      left = Math.min(window.innerWidth - w - 10, Math.max(10, rect.left));
      top = Math.min(rect.bottom + 10, window.innerHeight - height - 12);
    }
    setStyle({ left, top: Math.max(10, top) });
  }, [anchor, width]);

  useEscape(onClose);

  useEffect(() => {
    const dismiss = () => onClose();
    window.addEventListener('scroll', dismiss, { passive: true });
    const outside = (event: MouseEvent) => {
      if (card.current && !card.current.contains(event.target as Node)) onClose();
    };
    // Deferred: the click that opened this card would otherwise close it.
    const id = setTimeout(() => document.addEventListener('mousedown', outside), 0);
    return () => {
      clearTimeout(id);
      window.removeEventListener('scroll', dismiss);
      document.removeEventListener('mousedown', outside);
    };
  }, [onClose]);

  return (
    <div
      className="v-pop"
      role="dialog"
      ref={card}
      style={{ ...style, width: Math.min(width, typeof window === 'undefined' ? width : window.innerWidth - 20) }}
    >
      {children}
    </div>
  );
}

function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
}

/** The registration marks a framed panel wears — the Viewer's signature. */
export function Marks() {
  return (
    <>
      <i className="cm tl" />
      <i className="cm tr" />
      <i className="cm bl" />
      <i className="cm br" />
    </>
  );
}
