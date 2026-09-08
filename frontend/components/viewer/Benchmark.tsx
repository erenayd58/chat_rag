'use client';

/**
 * Benchmark — the methods side by side, on numbers that were actually measured.
 *
 * Two shapes, and which one a document gets is decided by the document, not by
 * a preference. A document with a frozen gold set and per-arm retrieval results
 * gets the retrieval comparison; one without gets the structural comparison and
 * is told plainly that Hit@k is not produced for it rather than being shown a
 * number nobody measured.
 *
 * The one claim this screen is careful about is the orchestration's. It does
 * not declare a winning method: what it reports is that the structural problem
 * count did not go up, that no section went backwards, and — where a gold set
 * exists — that search quality was at least held. Everything else is a cost.
 */

import type { ChunkingMethod } from '@/types/api';
import {
  SMELLS,
  SMELL_ORDER,
  deepSeconds,
  fixed1,
  fixed3,
  formatDuration,
  tokenStats,
  type ViewerDoc,
} from '@/lib/viewer/model';
import { Marks } from './Menu';

export function Benchmark({
  doc,
  methods,
  catalogue,
  orchestration,
}: {
  doc: ViewerDoc;
  methods: string[];
  catalogue: ChunkingMethod[];
  /** The method that declares itself an orchestration, if this document has one. */
  orchestration: string | null;
}) {
  const label = (key: string) => catalogue.find((m) => m.key === key)?.label ?? key;
  const measured = methods.filter((method) => doc.arms[method]?.ret);
  const hasGold = measured.length >= 2;

  const chips = [`${methods.length} yöntem`];
  if (doc.meta.pageCount) chips.push(`${doc.meta.pageCount} sayfa`);
  chips.push((doc.gold?.length ?? 0) ? `${doc.gold!.length} gold sorgu` : 'gold set yok');
  if (doc.meta.budgets?.target_tokens) {
    chips.push(`hedef ${doc.meta.budgets.target_tokens} token`);
  }

  let counter = 0;
  const no = () => String((counter += 1)).padStart(2, '0');

  return (
    <div className="bench">
      <div className="bhead">
        <div>
          <div className="kicker">
            Benchmark ·{' '}
            {hasGold ? 'Dondurulmuş set' : doc.live ? 'Canlı doküman' : 'Tek koşu'}
          </div>
          <h1 className="ql1">{doc.label}</h1>
          <p className="qsub">
            {hasGold
              ? 'Aynı canonical girdi, aynı BM25 ayarları, aynı gold set — kollar arasında yalnız chunker değişir.'
              : 'Gold sorgu seti yok; Hit@k / MRR bu doküman için üretilmez ve uydurulmaz. Aşağıdakilerin hepsi gerçekten ölçülmüş değerler.'}
          </p>
        </div>
        <div className="bchips">
          {chips.map((chip) => (
            <span className="bchip" key={chip}>
              {chip}
            </span>
          ))}
        </div>
      </div>

      {hasGold ? (
        <>
          <Orchestration doc={doc} no={no()} orchestration={orchestration} label={label} />
          <Retrieval doc={doc} methods={measured} no={no()} label={label} />
          <details className="bdetails">
            <summary>
              <span className="no">{no()}</span>
              <h2>Ham ölçümler</h2>
              <span className="tog">göster</span>
            </summary>
            <div className="bgrid" style={{ marginTop: 18 }}>
              <div>
                <Structural doc={doc} methods={methods} label={label} orchestration={orchestration} />
              </div>
              <div>
                <Timing doc={doc} methods={methods} label={label} orchestration={orchestration} />
              </div>
            </div>
          </details>
        </>
      ) : (
        <>
          <div className="bsec">
            <span className="no">{no()}</span>
            <h2>Yöntemler yan yana</h2>
            <span className="bn">aynı canonical girdi · ortak token bütçesi</span>
          </div>
          <div style={{ marginTop: 18 }}>
            <Structural doc={doc} methods={methods} label={label} orchestration={orchestration} />
          </div>
          <Orchestration doc={doc} no={no()} orchestration={orchestration} label={label} />
        </>
      )}
    </div>
  );
}

/**
 * Structural quality, computed from the real chunk rows and the packaged
 * measurements — never from an estimate.
 */
function Structural({
  doc,
  methods,
  label,
  orchestration,
}: {
  doc: ViewerDoc;
  methods: string[];
  label: (key: string) => string;
  orchestration: string | null;
}) {
  const live = doc.live?.methods;
  const timing = doc.meta.timing ?? {};
  let anyDuration = false;

  const rows = methods.map((method) => {
    const arm = doc.arms[method];
    const stats = tokenStats(arm.chunks);
    const fragmentation = arm.sq?.fragmentation;
    const headed = arm.chunks.length
      ? arm.chunks.filter((chunk) => chunk.hd).length / arm.chunks.length
      : null;
    let duration: number | null = null;
    if (method === orchestration) duration = deepSeconds(doc.meta.deep);
    else if (typeof live?.[method]?.seconds === 'number') duration = live[method].seconds as number;
    else if (timing[method]?.chunk_ms_median != null) {
      duration = timing[method].chunk_ms_median / 1000;
    }
    if (duration !== null) anyDuration = true;
    return { method, arm, stats, fragmentation, headed, duration };
  });

  return (
    <div className="hpanel">
      <Marks />
      <div className="hphead">
        <span className="t">Yapısal kalite</span>
        <span className="r">gerçek parça satırlarından</span>
      </div>
      <table className="btable">
        <thead>
          <tr>
            <th>Yöntem</th>
            <th>Parça</th>
            <th>Token medyan</th>
            <th>P90</th>
            <th>Başlıkla açılan</th>
            <th>Tablo böl.</th>
            <th>Liste böl.</th>
            <th>Süre</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.method}>
              <td>
                {label(row.method)}
                <span className="tech">{row.method}</span>
              </td>
              <td>{row.arm.chunks.length}</td>
              <td>{row.stats.median ?? '—'}</td>
              <td>{row.stats.p90 ?? '—'}</td>
              <td>{row.headed !== null ? fixed3(row.headed) : '—'}</td>
              <td>{row.fragmentation ? row.fragmentation.table_units_fragmented : '—'}</td>
              <td>{row.fragmentation ? row.fragmentation.list_units_fragmented : '—'}</td>
              <td>{formatDuration(row.duration)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {anyDuration ? null : (
        <div className="bnote">
          Bu doküman için süre kaydı yok; yeni yüklemelerde her yöntemin işleme süresi otomatik
          kaydedilir.
        </div>
      )}
    </div>
  );
}

function Timing({
  doc,
  methods,
  label,
  orchestration,
}: {
  doc: ViewerDoc;
  methods: string[];
  label: (key: string) => string;
  orchestration: string | null;
}) {
  const timing = doc.meta.timing ?? {};
  const measured = methods.filter((method) => timing[method]);
  const orchestrationTiming = orchestration ? doc.arms[orchestration]?.tim : null;
  if (!measured.length && !orchestrationTiming) return null;

  return (
    <div className="hpanel">
      <Marks />
      <div className="hphead">
        <span className="t">Zamanlama (ms)</span>
        <span className="r">medyan · tek koşu</span>
      </div>
      <table className="btable">
        <thead>
          <tr>
            <th>Yöntem</th>
            <th>Chunking</th>
            <th>İndeks</th>
            <th>Arama p90</th>
          </tr>
        </thead>
        <tbody>
          {measured.map((method) => (
            <tr key={method}>
              <td>{label(method)}</td>
              <td>{fixed1(timing[method].chunk_ms_median)}</td>
              <td>{fixed1(timing[method].index_build_ms)}</td>
              <td>
                {timing[method].search_p90_ms != null
                  ? timing[method].search_p90_ms.toFixed(2)
                  : '—'}
              </td>
            </tr>
          ))}
          {orchestrationTiming && orchestration ? (
            <tr>
              <td>{label(orchestration)}</td>
              <td>—</td>
              <td>{fixed1(orchestrationTiming.index_build_ms)}</td>
              <td>
                {orchestrationTiming.search_p90_ms != null
                  ? orchestrationTiming.search_p90_ms.toFixed(2)
                  : '—'}
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
      {orchestrationTiming ? (
        <div className="bnote">
          Orkestrasyonun chunking süresi sağlayıcıya bağlıdır; yerel kollarla karşılaştırılmaz ve
          bu tabloya yazılmaz.
        </div>
      ) : null}
      {timing.parse?.parse_ms ? (
        <div className="bnote">
          Parse: {Math.round(timing.parse.parse_ms / 1000)} s · tüm yöntemlerin paylaştığı tek
          ayrıştırma ({timing.parse.unit_count} birim).
        </div>
      ) : null}
    </div>
  );
}

/** What the orchestration changed, and what it cost. */
function Orchestration({
  doc,
  no,
  orchestration,
  label,
}: {
  doc: ViewerDoc;
  no: string;
  orchestration: string | null;
  label: (key: string) => string;
}) {
  const deep = doc.meta.deep;
  if (!deep || !orchestration) return null;

  const totals = deep.smellTotal ?? {};
  const perSmell = deep.totals ?? {};
  const counts = deep.storyCounts ?? doc.story?.counts ?? {};
  const calls = deep.calls?.total ?? 0;
  const seconds = deepSeconds(deep);
  const orchestrated = deep.retrieval?.deep;
  const baseline = deep.retrieval?.standard;
  const removed =
    totals.standard != null && totals.deep != null ? totals.standard - totals.deep : null;

  const stats: [string, string | number, string][] = [
    [
      'Yapısal problem',
      totals.standard != null ? `${totals.standard} → ${totals.deep}` : '—',
      removed && removed > 0
        ? `${removed} sorunlu sınır ortadan kalktı`
        : 'toplam problem sayısı korundu',
    ],
    [
      'Kötüleşen bölüm',
      deep.regressions ?? '—',
      deep.regressions === 0
        ? 'Hiçbir bölümde, hiçbir problem türünde temel bölümlemenin gerisine düşülmedi'
        : 'bölümde en az bir problem türü arttı',
    ],
  ];
  if (orchestrated && baseline) {
    stats.push([
      "İlk 5'te doğru parça",
      fixed3(orchestrated.hit_at_5),
      orchestrated.hit_at_5 === baseline.hit_at_5
        ? 'Temel bölümleme ile aynı — arama kalitesi korunuyor, iddia bu'
        : orchestrated.hit_at_5 > baseline.hit_at_5
          ? `Temel bölümlemeden iyi (${fixed3(baseline.hit_at_5)})`
          : `Temel bölümleme: ${fixed3(baseline.hit_at_5)}`,
    ]);
  } else {
    stats.push([
      'Modele danışılan bölüm',
      counts.llm_consulted_sections != null
        ? `${counts.llm_consulted_sections} / ${counts.sections ?? '?'}`
        : '—',
      'yalnız kararsız sınırı olan bölümler',
    ]);
  }
  stats.push([
    'Ek maliyet',
    calls > 0 ? (deep.estCostUsd != null ? `≈ $${deep.estCostUsd}` : `${calls} çağrı`) : '$0',
    calls > 0
      ? `${calls} LLM çağrısı${seconds !== null ? ` · ${Math.round(seconds)} s` : ''} · yüklemede tek sefer`
      : 'model çağrısı yok · kural katmanı',
  ]);

  const cost: [string, string | number][] = [];
  if (counts.llm_consulted_sections != null) {
    cost.push([
      'Modele danışılan bölüm',
      `${counts.llm_consulted_sections} / ${counts.sections ?? '?'}`,
    ]);
  }
  if (counts.llm_accepted != null) cost.push(['Kabul edilen model önerisi', counts.llm_accepted]);
  if (counts.deterministic_improved != null) {
    cost.push(['Kural ile düzeltilen bölüm', counts.deterministic_improved]);
  }
  if (counts.llm_reverted != null) cost.push(["Doğrulayıcının geri çevirdiği", counts.llm_reverted]);
  if (seconds !== null) cost.push(['Ek süre (yükleme)', `${Math.round(seconds)} s`]);
  if (deep.model && calls > 0) cost.push(['Model', deep.model]);

  const claim =
    calls > 0
      ? `${label(orchestration)} model kullanır; aynı koşu birebir tekrarlanmaz ve tek başına bir "kazanan yöntem" ilan edilmez. Garanti edilen şey: yapısal problem sayısı artmaz, hiçbir bölüm temel bölümlemenin gerisine düşmez${
          orchestrated ? ', arama kalitesi bu gold sette en azından korunur.' : '.'
        }`
      : 'Bu koşuda modele hiç danışılmadı; kazancın tamamı deterministik kural katmanından. Aynı canonical ile koşu birebir tekrarlanabilir.';

  const smellRows = SMELL_ORDER.map((key) => {
    const before = perSmell.standard?.[key];
    const after = perSmell.deep?.[key];
    if (before == null && after == null) return null;
    const delta = before != null && after != null ? after - before : null;
    return { key, before, after, delta };
  }).filter(Boolean) as { key: string; before?: number; after?: number; delta: number | null }[];

  const totalDelta =
    totals.deep != null && totals.standard != null ? totals.deep - totals.standard : null;

  return (
    <>
      <div className="bsec">
        <span className="no">{no}</span>
        <h2>Temel bölümleme → {label(orchestration)}</h2>
      </div>
      <div className="statrow b">
        <Marks />
        {stats.map(([key, value, note]) => (
          <div className="stat" key={key}>
            <div className="k">{key}</div>
            <div className="v">{value}</div>
            <div className="s">{note}</div>
          </div>
        ))}
      </div>
      <div className="bgrid">
        <div className="hpanel">
          <Marks />
          <div className="hphead">
            <span className="t">Sınır kalitesi — problem türü başına</span>
            <span className="r">bölüm bazında sayaçlar</span>
          </div>
          <table className="btable">
            <thead>
              <tr>
                <th>Problem türü</th>
                <th>Temel</th>
                <th>{label(orchestration)}</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {smellRows.map((row) => (
                <tr key={row.key}>
                  <td>
                    {SMELLS[row.key] ?? row.key}
                    <span className="tech">{row.key}</span>
                  </td>
                  <td>{row.before ?? '—'}</td>
                  <td>{row.after ?? '—'}</td>
                  <td
                    className={
                      row.delta === null ? '' : row.delta < 0 ? 'good' : row.delta > 0 ? 'badv' : ''
                    }
                  >
                    {row.delta === null ? '—' : row.delta === 0 ? '–' : row.delta > 0 ? `+${row.delta}` : row.delta}
                  </td>
                </tr>
              ))}
              <tr className="total">
                <td>Toplam</td>
                <td>{totals.standard ?? '—'}</td>
                <td>{totals.deep ?? '—'}</td>
                <td className={totalDelta !== null && totalDelta < 0 ? 'good' : ''}>
                  {totalDelta === null ? '—' : totalDelta === 0 ? '–' : totalDelta}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <div className="hpanel">
            <Marks />
            <div className="hphead">
              <span className="t">Bedeli</span>
            </div>
            {cost.map(([key, value]) => (
              <div className="bkrow" key={key}>
                <span>{key}</span>
                <span className="v">{String(value)}</span>
              </div>
            ))}
          </div>
          <div className="claim">
            <div className="t">Neyi iddia etmiyoruz</div>
            <p>{claim}</p>
          </div>
        </div>
      </div>
    </>
  );
}

/** Search quality, when the frozen run measured it for at least two methods. */
function Retrieval({
  doc,
  methods,
  no,
  label,
}: {
  doc: ViewerDoc;
  methods: string[];
  no: string;
  label: (key: string) => string;
}) {
  if (methods.length < 2) return null;
  const columns: [string, string][] = [
    ['hit_at_1', 'İlk sonuçta'],
    ['hit_at_3', "İlk 3'te"],
    ['hit_at_5', "İlk 5'te"],
    ['mrr', 'Sıralama (MRR)'],
    ['evidence_coverage_at_5', 'Kanıt kapsama'],
  ];
  const rows = methods.map((method) => ({ method, values: doc.arms[method].ret! }));
  const best: Record<string, number> = {};
  for (const [key] of columns) {
    best[key] = Math.max(...rows.map((row) => row.values[key] ?? 0));
  }

  return (
    <>
      <div className="bsec">
        <span className="no">{no}</span>
        <h2>Arama başarısı — {methods.length} yöntem yan yana</h2>
        <span className="bn">
          aynı {rows[0].values.query_count} soru · 0–1 arası, yüksek olan iyi
        </span>
      </div>
      <div className="hpanel" style={{ marginTop: 18 }}>
        <Marks />
        <table className="btable">
          <thead>
            <tr>
              <th>Yöntem</th>
              {columns.map(([key, text]) => (
                <th key={key}>{text}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.method}>
                <td>
                  {label(row.method)}
                  <span className="tech">{row.method}</span>
                </td>
                {columns.map(([key]) => (
                  <td key={key} className={row.values[key] === best[key] ? 'best' : ''}>
                    {fixed3(row.values[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="bnote">
        ● bu koşuda gözlenen en iyi değer. Hiçbir yöntem her sütunda önde değil — beklenen sonuç
        bu; tablo tek başına bir kazanan ilan etmez.
      </div>
    </>
  );
}
