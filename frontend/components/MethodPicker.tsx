'use client';

/**
 * Which chunking methods to analyse a document with.
 *
 * **There is no method list in this file.** Every option, its label, its
 * summary, whether this machine can run it, whether it uses a model and which
 * one is preselected all come from `GET /api/v1/meta/chunking-methods`, which
 * is a projection of the library's registry. A method added there appears here
 * without a line changing; a method this deployment cannot run is shown
 * disabled with the server's own reason rather than being hidden, so a picker
 * explains the gap instead of pretending the option never existed.
 *
 * An orchestration is marked as one, because it runs *over* a baseline
 * partition rather than being a partition, and showing the two the same way
 * misleads.
 */

import { useEffect, useMemo } from 'react';
import { useAsync } from '@/lib/hooks/useAsync';
import * as catalogue from '@/lib/methodCatalogue';
import { Failed, Loading } from './ui';
import type { ChunkingMethod } from '@/types/api';

/**
 * The catalogue, shared.
 *
 * Every caller on a page awaits one request -- see `lib/methodCatalogue.ts`
 * for why that matters on a screen full of method chips. A failure is not
 * kept, so `reload()` after one really does ask the server again; a success
 * is, because the registry does not change while a page is open.
 */
export function useChunkingMethods() {
  return useAsync<ChunkingMethod[]>(() => catalogue.chunkingMethods(), []);
}

export function MethodPicker({
  selected,
  onChange,
  /** Methods already built for this document; ticked and locked. */
  locked = [],
  hint,
}: {
  selected: string[];
  onChange: (methods: string[]) => void;
  locked?: string[];
  hint?: string;
}) {
  const state = useChunkingMethods();
  const methods = state.data;

  // The registry says which method an upload gets by default, so the form
  // preselects exactly that and no method name lives in this file.
  const defaults = useMemo(
    () => (methods ?? []).filter((m) => m.available && m.default).map((m) => m.key),
    [methods],
  );

  useEffect(() => {
    if (!methods || selected.length) return;
    if (defaults.length) onChange(defaults);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [methods, defaults]);

  if (state.loading && !methods) return <Loading label="Yöntemler okunuyor…" />;
  if (state.error && !methods) return <Failed error={state.error} onRetry={state.reload} />;
  if (!methods?.length) {
    return <div className="field-hint">Bu kurulumda tanımlı bir bölümleme yöntemi yok.</div>;
  }

  const toggle = (key: string, on: boolean) => {
    onChange(on ? [...new Set([...selected, key])] : selected.filter((value) => value !== key));
  };

  const byKey = new Map(methods.map((method) => [method.key, method]));

  return (
    <>
      <div className="method-list">
        {methods.map((method) => {
          const isLocked = locked.includes(method.key);
          const checked = isLocked || selected.includes(method.key);
          const disabled = !method.available || isLocked;
          return (
            <label
              key={method.key}
              className={`method-option${checked ? ' selected' : ''}${disabled ? ' disabled' : ''}`}
            >
              <input
                type="checkbox"
                name="method"
                value={method.key}
                checked={checked}
                disabled={disabled}
                onChange={(event) => toggle(method.key, event.target.checked)}
              />
              <span className="method-body">
                <span className="method-name">
                  {method.label}
                  {method.uses_model ? <span className="badge badge-accent">model destekli</span> : null}
                  {method.orchestration ? (
                    <span
                      className="badge badge-neutral"
                      title={
                        method.baseline
                          ? `${byKey.get(method.baseline)?.label ?? method.baseline} bölümlemesi üzerinde çalışır`
                          : 'Bir temel bölümleme üzerinde çalışır'
                      }
                    >
                      orkestrasyon
                    </span>
                  ) : null}
                  {isLocked ? <span className="badge badge-success">hazır</span> : null}
                  {!method.available ? <span className="badge badge-warn">kullanılamıyor</span> : null}
                </span>
                <span className="method-desc">
                  {method.available ? method.summary : (method.unavailable_reason ?? method.summary)}
                </span>
              </span>
            </label>
          );
        })}
      </div>
      <div className="field-hint">{hint ?? defaultHint(selected.length)}</div>
    </>
  );
}

function defaultHint(count: number): string {
  if (!count) return 'En az bir yöntem seçin.';
  if (count === 1) {
    return 'Doküman bir kez okunur. Karşılaştırma için ikinci bir yöntem seçebilirsiniz.';
  }
  return `Doküman bir kez okunur; ${count} yöntem aynı metin üzerinde çalışır ve yan yana karşılaştırılabilir.`;
}
