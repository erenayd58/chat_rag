'use client';

/**
 * Debug — where every boundary in this document came from.
 *
 * The pipeline, the recorded decisions, what the model was actually asked and
 * what the parser found. **Nothing here is estimated.** A value the artifacts
 * did not record is either hidden or written as unmeasured, because the whole
 * point of this screen is that it can be trusted against a run.
 *
 * A document with no Deep Analysis run has no decision trail, and rather than
 * an empty table it gets the one thing every document does record: the
 * boundary reason of every chunk, per method, as a histogram. That is a real
 * answer to "why does this cut exist" for a deterministic chunker.
 */

import { Fragment, useState } from 'react';
import type { ChunkingMethod } from '@/types/api';
import {
  DEEP_STATUS_TEXT,
  REASON_ORDER,
  REASON_SHORT,
  SECTION_STATUS,
  SMELLS,
  deepSeconds,
  formatDuration,
  type StorySection,
  type ViewerDoc,
} from '@/lib/viewer/model';
import { Marks } from './Menu';

const FILTERS: [string, string][] = [
  ['all', 'Tümü'],
  ['rule', 'Kural'],
  ['model', 'Model'],
  ['rev', 'Geri çevrilen'],
];

/**
 * "Kural" and "Model" filter by the decision's **source** — a reverted model
 * proposal still counts as the model's — and "Geri çevrilen" by its outcome.
 */
function matches(status: string, filter: string): boolean {
  if (filter === 'all') return true;
  if (filter === 'rule') return status === 'deterministic_improved';
  if (filter === 'model') return status === 'llm_accepted' || status === 'llm_reverted';
  return status === 'llm_reverted' || status === 'contract_reverted';
}

export function Debug({
  doc,
  methods,
  catalogue,
  baseline,
  onJumpToPage,
}: {
  doc: ViewerDoc;
  methods: string[];
  catalogue: ChunkingMethod[];
  /** The partition an orchestration starts from, as the registry declares it. */
  baseline: string | null;
  onJumpToPage: (page: number) => void;
}) {
  const [filter, setFilter] = useState('all');
  const [openSection, setOpenSection] = useState<number | null>(null);
  const label = (key: string) => catalogue.find((m) => m.key === key)?.label ?? key;

  const deep = doc.meta.deep ?? null;
  const story = doc.story ?? null;
  const counts = story?.counts ?? deep?.storyCounts ?? null;
  const calls = deep?.calls?.total ?? 0;
  const timing = doc.meta.timing ?? {};
  const seconds = deepSeconds(deep);

  const chips: string[] = [];
  if (deep?.mode) chips.push(`mod: ${deep.mode}`);
  if (deep?.status && DEEP_STATUS_TEXT[deep.status]) chips.push(DEEP_STATUS_TEXT[deep.status]);
  if (deep?.model && calls > 0) chips.push(deep.model.split('/').pop() as string);
  if (deep?.promptVersion) chips.push(deep.promptVersion);
  if (seconds !== null) chips.push(formatDuration(seconds));

  const baselineArm = baseline && doc.arms[baseline] ? baseline : null;
  const baselineSeconds =
    baselineArm && timing[baselineArm]?.chunk_ms_median != null
      ? timing[baselineArm].chunk_ms_median / 1000
      : (doc.live?.methods?.[baselineArm ?? '']?.seconds ?? null);

  const steps = [
    {
      n: '01',
      name: 'Parser',
      time: timing.parse?.parse_ms ? formatDuration(timing.parse.parse_ms / 1000) : null,
      detail: `${doc.meta.pageCount ?? '?'} sayfa · ${doc.meta.unitCount ?? '?'} birim${
        doc.parser?.count ? ` · ${doc.parser.count} bulgu` : ''
      }`,
    },
    counts
      ? {
          n: '02',
          name: 'Yapısal sınır',
          time: baselineSeconds != null ? formatDuration(baselineSeconds) : null,
          detail: `${counts.standard_kept} bölüm dokunulmadan geçti · ${counts.sections ?? '?'} bölüm`,
        }
      : {
          n: '02',
          name: 'Yapısal sınır',
          time: baselineSeconds != null ? formatDuration(baselineSeconds) : null,
          detail: baselineArm
            ? `${doc.arms[baselineArm].chunks.length} parça üretildi`
            : 'bu dokümanda koşmadı',
          dim: !baselineArm,
        },
    counts
      ? {
          n: '03',
          name: 'Kural katmanı',
          time: deep?.timing?.selection != null ? formatDuration(deep.timing.selection) : null,
          detail: `${counts.deterministic_improved} bölümde sınır düzeltildi`,
        }
      : { n: '03', name: 'Kural katmanı', detail: 'karar izi kayıtlı değil', dim: true },
    deep && calls > 0 && counts
      ? {
          n: '04',
          name: 'Model önerisi',
          time: deep.timing?.llm_calls != null ? formatDuration(deep.timing.llm_calls) : null,
          detail: `${counts.llm_consulted_sections} bölüm modele danışıldı · ${
            deep.proposer?.call_count ?? 0
          } çağrı`,
        }
      : {
          n: '04',
          name: 'Model önerisi',
          detail: deep ? 'modele danışılmadı' : 'Deep koşusu yok',
          dim: true,
        },
    deep?.verifier?.group_count
      ? {
          n: '05',
          name: 'Doğrulama',
          time:
            deep.timing?.verifier_calls != null ? formatDuration(deep.timing.verifier_calls) : null,
          detail: `${deep.verifier.accepted} öneri kabul · ${deep.verifier.reverted} geri çevrildi (${deep.verifier.group_count} grup ×2 sıra)`,
        }
      : { n: '05', name: 'Doğrulama', detail: 'çalışmadı', dim: true },
  ];

  let section = 0;
  const no = () => String((section += 1)).padStart(2, '0');

  const rows = story ? decisionRows(story.sections ?? []) : [];
  const shown = rows.filter((row) => matches(row.section.st, filter));

  return (
    <div className="bench">
      <div className="bhead">
        <div>
          <div className="kicker">Debug · {doc.label}</div>
          <h1 className="ql1">Bölümleme izi</h1>
          <p className="qsub">
            Her sınırın nereden geldiği: parser tabanı, yapısal sınır, kural katmanı, model
            önerisi ve doğrulama kararı. Yalnız kayıtlı değerler gösterilir.
          </p>
        </div>
        {chips.length ? (
          <div className="bchips">
            {chips.map((chip) => (
              <span className="bchip" key={chip}>
                {chip}
              </span>
            ))}
          </div>
        ) : null}
      </div>

      <div className="bsec">
        <span className="no">{no()}</span>
        <h2>Boru hattı</h2>
      </div>
      <div className="pipe">
        <Marks />
        {steps.map((step) => (
          <div className={`pstep${step.dim ? ' dim' : ''}`} key={step.n}>
            <span className="pn">{step.n}</span>
            {step.time ? <span className="pt">{step.time}</span> : null}
            <div className="nm">{step.name}</div>
            <div className="ds">{step.detail}</div>
          </div>
        ))}
      </div>

      {story ? (
        <>
          <div className="bsec">
            <span className="no">{no()}</span>
            <h2>Sınır kararları</h2>
            <span className="bn">
              {rows.length} kayıt · {counts?.sections ?? '?'} bölüm ·{' '}
              {counts?.standard_kept ?? '?'} bölümde temel bölümleme korundu
            </span>
            <div className="dfilters">
              {FILTERS.map(([key, text]) => (
                <button
                  key={key}
                  type="button"
                  className={filter === key ? 'on' : undefined}
                  onClick={() => {
                    setFilter(key);
                    setOpenSection(null);
                  }}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
          <div className="hpanel" style={{ marginTop: 18 }}>
            <Marks />
            <table className="btable">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Bölüm</th>
                  <th style={{ textAlign: 'left' }}>Başlık</th>
                  <th style={{ textAlign: 'left' }}>Kaynak</th>
                  <th style={{ textAlign: 'left' }}>Tetikleyen</th>
                  <th>Kesim</th>
                  <th>Sonuç</th>
                </tr>
              </thead>
              <tbody>
                {shown.length ? (
                  shown.map((row) => {
                    const open = openSection === row.section.i;
                    return (
                      <Fragment key={row.section.i}>
                        <tr
                          className={`drow${open ? ' openrow' : ''}`}
                          onClick={() => setOpenSection(open ? null : row.section.i)}
                        >
                          <td>§{String(row.section.i).padStart(3, '0')}</td>
                          <td className="l">
                            {(row.section.h ?? '(başlıksız bölüm)').slice(0, 70)}
                            {row.section.pg?.length ? (
                              <span className="tech">s. {row.section.pg[0]}</span>
                            ) : null}
                          </td>
                          <td style={{ textAlign: 'left' }}>
                            <span className={`srcchip ${row.config.tone}`}>
                              {row.config.source}
                            </span>
                          </td>
                          <td className="l">
                            {row.trigger ? (
                              <span className="tech" style={{ marginLeft: 0 }}>
                                {row.trigger}
                              </span>
                            ) : (
                              '—'
                            )}
                          </td>
                          <td>
                            {row.section.std && row.section.fin
                              ? `${row.section.std.length} → ${row.section.fin.length}`
                              : '—'}
                          </td>
                          <td>
                            <span className={`res ${row.config.outcomeTone}`}>
                              {row.config.outcome}
                            </span>
                          </td>
                        </tr>
                        {open ? (
                          <tr className="ddetail">
                            <td colSpan={6}>
                              <Detail row={row} onJumpToPage={onJumpToPage} />
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={6} className="l" style={{ color: 'var(--faint)' }}>
                      Bu filtrede kayıt yok.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="bnote">
            Yalnız bir şeyin değiştiği bölümler listelenir; satıra tıklayınca kayıtlı karar
            ayrıntısı açılır. Bölüm başına süre kaydedilmez — toplam süreler boru hattında.
          </div>
        </>
      ) : (
        <>
          <div className="bsec">
            <span className="no">{no()}</span>
            <h2>Sınır nedenleri</h2>
            <span className="bn">karar izi yalnız orkestrasyon koşularında kaydedilir</span>
          </div>
          <div style={{ marginTop: 18 }}>
            <ReasonHistogram doc={doc} methods={methods} label={label} />
          </div>
        </>
      )}

      <div className="bgrid">
        <div>
          <div className="bsec" style={{ marginTop: 0 }}>
            <span className="no">{no()}</span>
            <h2>Model kullanımı</h2>
          </div>
          <div style={{ marginTop: 18 }}>
            <ModelPanel doc={doc} />
          </div>
        </div>
        <div>
          <div className="bsec" style={{ marginTop: 0 }}>
            <span className="no">{no()}</span>
            <h2>Parser bulguları</h2>
            <span className="bn">
              {doc.parser?.findings.length ?? 0} kayıt · canonical üzerinde
            </span>
          </div>
          <div className="dlog" style={{ marginTop: 18 }}>
            <ParserLog doc={doc} />
          </div>
        </div>
      </div>
    </div>
  );
}

interface DecisionRow {
  section: StorySection;
  config: (typeof SECTION_STATUS)[string];
  removed: string[];
  added: string[];
  trigger: string | null;
}

function decisionRows(sections: StorySection[]): DecisionRow[] {
  const rows: DecisionRow[] = [];
  for (const section of sections) {
    const config = SECTION_STATUS[section.st];
    if (!config) continue;
    const removed: string[] = [];
    const added: string[] = [];
    for (const group of section.gr ?? []) {
      for (const key of group.rm ?? []) removed.push(key);
      for (const key of group.in ?? []) added.push(key);
    }
    let trigger: string | null = removed.length ? removed.join(' · ') : null;
    if (!trigger && section.pr?.length) {
      const reasons = [
        ...new Set(
          section.pr
            .filter((proposal) => (section.st === 'llm_accepted') === !!proposal.a && proposal.r)
            .map((proposal) => proposal.r as string),
        ),
      ];
      trigger = reasons.join(' · ') || null;
    }
    if (!trigger && section.rv) trigger = String(section.rv);
    rows.push({ section, config, removed, added, trigger });
  }
  return rows;
}

function Detail({
  row,
  onJumpToPage,
}: {
  row: DecisionRow;
  onJumpToPage: (page: number) => void;
}) {
  const section = row.section;
  const cells: [string, string][] = [];
  const add = (key: string, value: unknown) => {
    if (value !== null && value !== undefined && value !== '') cells.push([key, String(value)]);
  };
  add('Sayfa', (section.pg ?? []).join(', '));
  add('Bölüm tokeni', section.tt);
  if (section.std && section.fin) {
    add('Kesim sayısı', `${section.std.length} → ${section.fin.length}`);
  }
  add(
    'Giderilen',
    row.removed.length ? row.removed.map((key) => SMELLS[key] ?? key).join(', ') : null,
  );
  add(
    'Eklenen problem',
    row.added.length ? row.added.map((key) => SMELLS[key] ?? key).join(', ') : null,
  );
  if (section.pr?.length) {
    add('Model önerisi', `${section.pr.length} · kabul ${section.pr.filter((p) => p.a).length}`);
  }
  if (section.cons !== undefined) add('Modele danışıldı', section.cons ? 'evet' : 'hayır');
  if (section.sz) add('Boyut ödünleşimi', 'evet — daha az problem karşılığında');
  if (section.rv) add('Geri alma nedeni', String(section.rv));

  return (
    <div className="dg">
      {cells.map(([key, value]) => (
        <span key={key}>
          <span className="k">{key}</span>
          <span className="v2">{value}</span>
        </span>
      ))}
      {section.pg?.length ? (
        <span>
          <span className="k">Görünüm</span>
          <button
            type="button"
            className="qjump"
            onClick={(event) => {
              event.stopPropagation();
              onJumpToPage(section.pg![0]);
            }}
          >
            İncele&apos;de aç →
          </button>
        </span>
      ) : null}
    </div>
  );
}

function ReasonHistogram({
  doc,
  methods,
  label,
}: {
  doc: ViewerDoc;
  methods: string[];
  label: (key: string) => string;
}) {
  const codes = new Set<string>();
  const counts: Record<string, Record<string, number>> = {};
  for (const method of methods) {
    counts[method] = {};
    for (const chunk of doc.arms[method]?.chunks ?? []) {
      codes.add(chunk.rs);
      counts[method][chunk.rs] = (counts[method][chunk.rs] ?? 0) + 1;
    }
  }
  const order = [
    ...REASON_ORDER.filter((code) => codes.has(code)),
    ...[...codes].filter((code) => !REASON_ORDER.includes(code)),
  ];

  return (
    <div className="hpanel">
      <Marks />
      <div className="hphead">
        <span className="t">Sınır nedenleri — yöntem başına</span>
        <span className="r">her parçanın kayıtlı başlama nedeni</span>
      </div>
      <table className="btable">
        <thead>
          <tr>
            <th>Neden</th>
            {methods.map((method) => (
              <th key={method}>{label(method)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {order.map((code) => (
            <tr key={code}>
              <td>
                {REASON_SHORT[code] ?? code}
                <span className="tech">{code}</span>
              </td>
              {methods.map((method) => (
                <td key={method}>{counts[method][code] || '–'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelPanel({ doc }: { doc: ViewerDoc }) {
  const deep = doc.meta.deep ?? null;
  const calls = deep?.calls?.total ?? 0;
  const counts = doc.story?.counts ?? deep?.storyCounts ?? null;

  if (!deep || calls === 0) {
    return (
      <div className="hpanel">
        <Marks />
        <div className="hphead">
          <span className="t">Model kullanımı</span>
        </div>
        <div className="hquiet">
          {deep
            ? `${(deep.status && DEEP_STATUS_TEXT[deep.status]) || 'Bu koşuda model çağrısı kaydı yok'} — kazanç kural katmanından.`
            : 'Bu dokümanda orkestrasyon koşusu yok; model hiç devrede olmadı.'}
        </div>
      </div>
    );
  }

  const rows: [string, string | number][] = [];
  if (counts?.llm_consulted_sections != null) {
    rows.push([
      'Modele danışılan bölüm',
      `${counts.llm_consulted_sections} / ${counts.sections ?? '?'}`,
    ]);
  }
  if (deep.proposer?.call_count != null) {
    rows.push([
      'Öneri çağrısı — başarılı',
      `${deep.proposer.call_status?.ok ?? 0} / ${deep.proposer.call_count}`,
    ]);
  }
  if (deep.verifier?.group_count != null) {
    rows.push(['Doğrulama çağrısı (her grup 2 sıra)', 2 * deep.verifier.group_count]);
  }
  if (deep.selection?.vote_count != null) {
    rows.push([
      'İşaretlenen / yasaklanan sınır oyu',
      `${deep.selection.vote_count} / ${deep.selection.forbidden_vote_count ?? 0}`,
    ]);
  }
  if (deep.verifier?.accepted != null) {
    rows.push(['Kabul / geri çevrilen grup', `${deep.verifier.accepted} / ${deep.verifier.reverted}`]);
  }
  if (counts?.contract_reverted != null) {
    rows.push(['Kalite kontrolünün geri aldığı', counts.contract_reverted]);
  }
  if (deep.estTokens) {
    rows.push([
      'Token — tahmini (karakterden)',
      `${Math.round(deep.estTokens.prompt / 1000)}K / ${Math.round(deep.estTokens.completion / 1000)}K`,
    ]);
  }
  if (deep.estCostUsd != null) rows.push(['Yaklaşık maliyet (liste fiyatı)', `≈ $${deep.estCostUsd}`]);

  return (
    <div className="hpanel">
      <Marks />
      <div className="hphead">
        <span className="t">Model kullanımı</span>
        {deep.model ? <span className="r">{deep.model}</span> : null}
      </div>
      {rows.map(([key, value]) => (
        <div className="bkrow" key={key}>
          <span>{key}</span>
          <span className="v">{String(value)}</span>
        </div>
      ))}
    </div>
  );
}

function ParserLog({ doc }: { doc: ViewerDoc }) {
  const findings = doc.parser?.findings ?? [];
  const oversized = doc.units.filter((unit) => unit.big).length;
  if (!findings.length) {
    return (
      <>
        <div className="more">parser bulgusu yok</div>
        {oversized ? <div className="more">temsil tavanı üstü birim: {oversized}</div> : null}
      </>
    );
  }
  return (
    <>
      {findings.slice(0, 14).map((finding, index) => (
        <div key={`${finding.t}-${index}`}>
          <span className="u">{finding.t}</span> {finding.r}
        </div>
      ))}
      {findings.length > 14 ? (
        <div className="more">… +{findings.length - 14} kayıt daha</div>
      ) : null}
      {oversized ? <div className="more">temsil tavanı üstü birim: {oversized}</div> : null}
    </>
  );
}
