/* Lab page: retrieval quality review, chunk search, parser units,
 * chunk browser. Technical tooling, deliberately separate from the
 * product flow. */
'use strict';

function labKbId() {
  return $('#labKbSelect').value || '';
}

let labDocs = [];

/* ------------------------------------------------------------------ */
/* Tabs                                                                */
/* ------------------------------------------------------------------ */
$all('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    $all('.tab').forEach((t) => t.classList.remove('active'));
    $all('.tab-panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('#panel-' + tab.dataset.tab).classList.add('active');
  });
});

/* ------------------------------------------------------------------ */
/* Chunk metadata helpers (shared by all tabs)                         */
/* ------------------------------------------------------------------ */
function chunkMeta(item) { return (item && item.metadata) || {}; }

function chunkSection(item) {
  const md = chunkMeta(item);
  if (md.heading) return md.heading;
  if (md.section_title) return md.section_title;
  try {
    const paths = JSON.parse(md.section_paths_json || '[]');
    if (paths.length && paths[0].length) return paths[0].join(' > ');
  } catch (e) { /* fall through */ }
  return null;
}

function chunkPageList(item) {
  const md = chunkMeta(item);
  try {
    const pages = JSON.parse(md.pages_json || '[]');
    if (pages.length) return pages;
  } catch (e) { /* fall through */ }
  for (const key of ['page', 'page_number', 'page_start']) {
    if (md[key] !== undefined && md[key] !== null && md[key] !== '') return [md[key]];
  }
  return [];
}

function chunkPages(item) {
  const pages = chunkPageList(item);
  if (pages.length === 1) return String(pages[0]);
  if (pages.length > 1) return pages[0] + '-' + pages[pages.length - 1];
  return null;
}

function chunkUnitIds(item) {
  try { return JSON.parse(chunkMeta(item).unit_ids_json || '[]'); } catch (e) { return []; }
}

function chunkDocName(item) {
  const md = chunkMeta(item);
  return md.doc_title || md.file_name || md.doc_id || 'document';
}

function chunkScore(item) {
  const score = item && (item.score !== undefined ? item.score : item.similarity_score);
  return (score === null || score === undefined) ? null : Number(score);
}

function openChunkView(title, meta, text) {
  $('#chunkViewTitle').textContent = title || 'Chunk';
  $('#chunkViewMeta').textContent = meta || '';
  $('#chunkViewText').textContent = text || '';
  openModal('chunkViewModal');
}

/* ------------------------------------------------------------------ */
/* Chunk edit modal (shared by search + browser tabs)                  */
/* ------------------------------------------------------------------ */
let chunkEditCallback = null;

function openChunkEdit(chunkId, content, onSaved) {
  $('#chunkEditId').textContent = chunkId;
  $('#chunkEditContent').value = content;
  $('#chunkEditError').innerHTML = '';
  chunkEditCallback = onSaved || null;
  openModal('chunkEditModal');
}

$('#chunkEditSave').addEventListener('click', async () => {
  const chunkId = $('#chunkEditId').textContent;
  const content = $('#chunkEditContent').value;
  try {
    await api('/api/chunks/' + encodeURIComponent(chunkId) + '?kb_id=' + encodeURIComponent(labKbId()), {
      method: 'PUT',
      json: { content: content }
    });
    closeModal('chunkEditModal');
    toast('Chunk updated', 'success');
    if (chunkEditCallback) chunkEditCallback();
  } catch (e) {
    $('#chunkEditError').innerHTML = '<div class="error-state" style="margin-top:10px;">' + escapeHtml(e.message) + '</div>';
  }
});

async function deleteChunk(chunkId, onDeleted) {
  const ok = await confirmDialog({
    title: 'Delete chunk',
    message: 'Delete chunk ' + chunkId + ' from the index?',
    confirmLabel: 'Delete',
    danger: true
  });
  if (!ok) return;
  try {
    await api('/api/chunks/' + encodeURIComponent(chunkId) + '?kb_id=' + encodeURIComponent(labKbId()), { method: 'DELETE' });
    toast('Chunk deleted', 'success');
    if (onDeleted) onDeleted();
  } catch (e) {
    toast('Delete failed: ' + e.message, 'error');
  }
}

/* ------------------------------------------------------------------ */
/* Retrieval Quality (gold set review)                                 */
/* ------------------------------------------------------------------ */
const rqState = { items: [], query: '', method: '', topK: 0, correct: null, entryId: null };
const rqGold = { kbId: null, byQuestion: new Map() };

function rqNormalizeQuestion(question) {
  // Must match the server's identity rule exactly, or a mark saved
  // under one spelling is never found again. Python's casefold maps
  // 'I' to 'i'; a Turkish-locale lowercase would map it to 'ı' and
  // the two sides would disagree on the same question.
  return String(question || '').split(/\s+/).filter(Boolean).join(' ').toLowerCase();
}

async function rqLoadGold(kbId) {
  rqGold.kbId = kbId;
  rqGold.byQuestion = new Map();
  if (!kbId) return;
  try {
    const data = await api('/api/goldset?kb_id=' + encodeURIComponent(kbId));
    (data.entries || []).forEach((entry) => {
      rqGold.byQuestion.set(rqNormalizeQuestion(entry.question), entry);
    });
  } catch (e) { /* review still works without the gold set */ }
}

function rqGoldFor(query) {
  return rqGold.byQuestion.get(rqNormalizeQuestion(query)) || null;
}

async function rqRefreshMethods() {
  const select = $('#rqMethod');
  const previous = select.value;
  try {
    const data = await api('/api/retrieval/capabilities?kb_id=' + encodeURIComponent(labKbId()));
    select.innerHTML = (data.methods || []).map((m) => {
      const label = m.available ? m.label : m.label + ' — unavailable';
      return '<option value="' + escapeHtml(m.name) + '"' + (m.available ? '' : ' disabled') +
        (m.reason ? ' title="' + escapeHtml(m.reason) + '"' : '') + '>' + escapeHtml(label) + '</option>';
    }).join('');
    const stillOk = (data.methods || []).some((m) => m.name === previous && m.available);
    select.value = stillOk ? previous : (data.default || '');
    const blocked = (data.methods || []).filter((m) => !m.available && m.reason);
    $('#rqMethodNote').textContent = blocked.length
      ? blocked.map((m) => m.label + ': ' + m.reason).join('  ') : '';
  } catch (e) { /* leave existing options in place */ }
}

async function rqRun() {
  const query = $('#rqQuery').value.trim();
  const method = $('#rqMethod').value;
  const topK = parseInt($('#rqTopK').value || '5', 10);
  const kbId = labKbId();
  const resBox = $('#rqResults');
  if (!kbId) { toast('Select a knowledge base first.', 'error'); return; }
  if (!query) { toast('Enter a question.', 'error'); return; }
  resBox.innerHTML = '<div class="loading-state"><div class="spinner"></div>Running retrieval…</div>';
  try {
    const data = await api('/api/experiment/search_chunks', {
      method: 'POST',
      json: { query: query, method: method, top_k: topK, kb_id: kbId }
    });
    rqState.items = data.chunks || [];
    rqState.query = query;
    rqState.method = method;
    rqState.topK = topK;
    if (rqGold.kbId !== kbId) await rqLoadGold(kbId);
    const entry = rqGoldFor(query);
    rqState.correct = entry ? entry.correct_chunk_id : null;
    rqState.entryId = entry ? entry.entry_id : null;
    rqRender();
  } catch (e) {
    resBox.innerHTML = '<div class="error-state">' + escapeHtml(e.message) + '</div>';
  }
}

function rqRender() {
  const resBox = $('#rqResults');
  const { items, query, method, topK, correct } = rqState;
  if (!items.length) {
    resBox.innerHTML = '<div class="card empty-state"><h3>No results</h3><p>No sources were returned for this query.</p></div>';
    return;
  }

  const top = items[0];
  const markedIndex = correct ? items.findIndex((it) => it.chunk_id === correct) : -1;
  const topScore = chunkScore(top);
  let verdict;
  if (markedIndex >= 0) {
    verdict = '<span style="color:var(--success); font-weight:600;">Yes — rank ' + (markedIndex + 1) + '</span>';
  } else if (correct) {
    verdict = '<span style="color:var(--danger); font-weight:600;">No — the confirmed source is not in these results</span>';
  } else {
    verdict = '<span style="color:var(--text-2);">Not marked yet</span>';
  }

  let html =
    '<div class="card card-pad" style="margin-bottom:14px;">' +
      '<div style="font-weight:600; margin-bottom:10px;">' + escapeHtml(method.toUpperCase()) + ' · Top ' + topK +
        ' · <span style="font-weight:400; color:var(--text-2);">&ldquo;' + escapeHtml(query) + '&rdquo; · ' + items.length + ' sources</span></div>' +
      '<div class="stats-row">' +
        '<div><div class="stat-label">Top-1 section</div><div style="font-weight:600; font-size:13px; overflow-wrap:anywhere;">' + escapeHtml(chunkSection(top) || '—') + '</div></div>' +
        '<div><div class="stat-label">Top-1 pages</div><div style="font-weight:600; font-size:13px;">' + escapeHtml(chunkPages(top) || '—') + '</div></div>' +
        '<div><div class="stat-label">Top-1 score</div><div style="font-weight:600; font-size:13px;">' + (topScore === null ? '—' : topScore.toFixed(4)) + '</div></div>' +
        '<div><div class="stat-label">Correct source found?</div><div style="font-size:13px;">' + verdict + '</div></div>' +
      '</div>' +
    '</div>';

  items.forEach((item, idx) => {
    const content = item.content || '';
    const preview = content.length > 420 ? content.slice(0, 420) + '…' : content;
    const section = chunkSection(item);
    const pages = chunkPages(item);
    const score = chunkScore(item);
    const isMarked = correct && item.chunk_id === correct;
    const details = {
      chunk_id: item.chunk_id,
      retrieval_method: item.retrieval_method || null,
      score: score,
      metadata: chunkMeta(item)
    };
    html +=
      '<div class="chunk-card' + (isMarked ? ' rq-marked' : '') + '">' +
        '<div class="chunk-card-head">' +
          '<div style="display:flex; gap:10px; align-items:center; min-width:0;">' +
            '<span class="rank-badge">' + (idx + 1) + '</span>' +
            '<div style="min-width:0;">' +
              '<div style="font-weight:600; overflow-wrap:anywhere;">' + escapeHtml(section || '(no section heading)') + '</div>' +
              '<div style="font-size:12px; color:var(--text-2);">' + escapeHtml(chunkDocName(item)) + (pages ? ' · p. ' + escapeHtml(pages) : '') + '</div>' +
            '</div>' +
          '</div>' +
          '<div style="display:flex; gap:6px; align-items:center;">' +
            '<span class="badge badge-neutral">score ' + (score === null ? '—' : score.toFixed(4)) + '</span>' +
          '</div>' +
        '</div>' +
        '<div style="font-size:13px; color:var(--text-2); white-space:pre-wrap; overflow-wrap:anywhere; margin-bottom:10px;">' + escapeHtml(preview) + '</div>' +
        '<div style="display:flex; gap:8px; flex-wrap:wrap;">' +
          '<button class="btn btn-secondary btn-sm rq-open" data-idx="' + idx + '">Open chunk</button>' +
          '<button class="btn ' + (isMarked ? 'btn-primary' : 'btn-secondary') + ' btn-sm rq-mark" data-idx="' + idx + '">' +
            (isMarked ? '✓ Correct source' : 'Mark as correct source') + '</button>' +
        '</div>' +
        '<details class="details-box"><summary>Details</summary><div class="details-body">' + escapeHtml(JSON.stringify(details, null, 2)) + '</div></details>' +
      '</div>';
  });
  resBox.innerHTML = html;

  $all('.rq-open', resBox).forEach((btn) => btn.addEventListener('click', () => {
    const item = rqState.items[Number(btn.dataset.idx)];
    openChunkView(
      chunkSection(item) || 'Chunk',
      chunkDocName(item) + (chunkPages(item) ? ' · p. ' + chunkPages(item) : '') + ' · rank ' + (Number(btn.dataset.idx) + 1),
      item.content || ''
    );
  }));
  $all('.rq-mark', resBox).forEach((btn) => btn.addEventListener('click', () => rqToggleCorrect(Number(btn.dataset.idx))));
}

async function rqToggleCorrect(idx) {
  const item = rqState.items[idx];
  if (!item) return;
  const kbId = labKbId();
  if (!kbId) { toast('Select a knowledge base first.', 'error'); return; }

  if (rqState.correct === item.chunk_id) {
    const entry = rqGoldFor(rqState.query);
    if (entry) {
      try {
        await api('/api/goldset/' + encodeURIComponent(entry.entry_id), { method: 'DELETE' });
        rqGold.byQuestion.delete(rqNormalizeQuestion(rqState.query));
      } catch (e) { toast('Could not remove the entry: ' + e.message, 'error'); return; }
    }
    rqState.correct = null;
    rqState.entryId = null;
    rqRender();
    return;
  }

  const md = chunkMeta(item);
  const payload = {
    question: rqState.query,
    kb_id: kbId,
    document_id: md.doc_id || null,
    document_title: md.doc_title || md.file_name || null,
    correct_chunk_id: item.chunk_id,
    section: chunkSection(item),
    pages: chunkPageList(item),
    unit_ids: chunkUnitIds(item),
    evidence: item.content || '',
    retrieval_method: item.retrieval_method || rqState.method,
    found_at_rank: idx + 1
  };
  try {
    const data = await api('/api/goldset', { method: 'POST', json: payload });
    rqGold.byQuestion.set(rqNormalizeQuestion(rqState.query), data.entry);
    rqState.correct = data.entry.correct_chunk_id;
    rqState.entryId = data.entry.entry_id;
    rqRender();
    toast('Rank ' + (idx + 1) + ' saved as the correct source.', 'success');
  } catch (e) {
    toast('Could not save: ' + e.message, 'error');
  }
}

$('#rqRun').addEventListener('click', rqRun);
$('#rqQuery').addEventListener('keydown', (e) => { if (e.key === 'Enter') rqRun(); });

/* ------------------------------------------------------------------ */
/* Chunk Search                                                        */
/* ------------------------------------------------------------------ */
function renderChunkCards(container, chunks, opts) {
  const options = opts || {};
  if (!chunks.length) {
    container.innerHTML = '<div class="card empty-state"><h3>No results</h3><p>Try a different query or method.</p></div>';
    return;
  }
  container.innerHTML = chunks.map((chunk, idx) => {
    const score = chunkScore(chunk);
    const section = chunkSection(chunk);
    const pages = chunkPages(chunk);
    return (
      '<div class="chunk-card" data-idx="' + idx + '">' +
        '<div class="chunk-card-head">' +
          '<div style="min-width:0;">' +
            '<div style="font-weight:600; overflow-wrap:anywhere;">' + escapeHtml(section || chunkDocName(chunk)) + '</div>' +
            '<div class="mono">' + escapeHtml(chunk.chunk_id) + '</div>' +
          '</div>' +
          '<div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">' +
            (pages ? '<span class="badge badge-neutral">p. ' + escapeHtml(pages) + '</span>' : '') +
            (score !== null ? '<span class="badge badge-neutral">score ' + score.toFixed(4) + '</span>' : '') +
            '<button class="btn btn-secondary btn-sm chunk-edit">Edit</button>' +
            '<button class="btn btn-danger-ghost btn-sm chunk-del">Delete</button>' +
          '</div>' +
        '</div>' +
        '<div class="chunk-text">' + escapeHtml(chunk.content || '') + '</div>' +
      '</div>'
    );
  }).join('');

  $all('.chunk-card', container).forEach((card) => {
    const chunk = chunks[Number(card.dataset.idx)];
    $('.chunk-edit', card).addEventListener('click', () => {
      openChunkEdit(chunk.chunk_id, chunk.content || '', options.refresh);
    });
    $('.chunk-del', card).addEventListener('click', () => deleteChunk(chunk.chunk_id, options.refresh));
  });
}

async function runSearch() {
  const query = $('#searchQuery').value.trim();
  const method = $('#searchMethod').value;
  const kbId = labKbId();
  const container = $('#searchResults');
  if (!kbId) { toast('Select a knowledge base first.', 'error'); return; }
  if (!query) { toast('Enter a query.', 'error'); return; }
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Searching…</div>';
  try {
    let data;
    if (method === 'vector') {
      data = await api('/api/chunks/search-vector', { method: 'POST', json: { query: query, kb_id: kbId, offset: 0, limit: 50 } });
    } else if (method === 'bm25') {
      data = await api('/api/chunks/search-bm25', { method: 'POST', json: { query: query, kb_id: kbId, offset: 0, limit: 50 } });
    } else {
      data = await api('/api/chunks?search=' + encodeURIComponent(query) + '&offset=0&limit=50&kb_id=' + encodeURIComponent(kbId));
    }
    renderChunkCards(container, data.chunks || [], { refresh: runSearch });
  } catch (e) {
    container.innerHTML = '<div class="error-state">' + escapeHtml(e.message) + '</div>';
  }
}

$('#searchRun').addEventListener('click', runSearch);
$('#searchQuery').addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });

/* ------------------------------------------------------------------ */
/* Parser Units                                                        */
/* ------------------------------------------------------------------ */
const puState = { offset: 0, total: 0, limit: 100 };

function parserDocId() { return $('#parserDoc').value || ''; }

async function loadParserUnits(offset) {
  const container = $('#parserResults');
  const docId = parserDocId();
  if (!docId) { container.innerHTML = '<div class="card empty-state"><h3>Select a document</h3><p>Parser output is available for documents ingested through the structured parser.</p></div>'; return; }
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Loading canonical units…</div>';
  const params = new URLSearchParams({ offset: String(offset), limit: String(puState.limit) });
  if (labKbId()) params.set('kb_id', labKbId());
  const from = $('#puFrom').value;
  const to = $('#puTo').value;
  const type = $('#puType').value;
  if (from) params.set('page_from', from);
  if (to) params.set('page_to', to);
  if (type) params.set('unit_type', type);

  try {
    const data = await api('/api/documents/' + encodeURIComponent(docId) + '/canonical-units?' + params);
    puState.offset = data.offset;
    puState.total = data.total;
    const docPages = data.pages_in_document || [];
    const pageRange = docPages.length ? (docPages[0] + '-' + docPages[docPages.length - 1]) : 'n/a';
    $('#puSubtitle').textContent =
      data.total_units_in_document + ' canonical units in document · ' + data.total + ' match the filter · pages ' + pageRange;
    $('#puRange').textContent = (data.offset + 1) + '-' + (data.offset + data.returned) + ' / ' + data.total;
    container.innerHTML = (data.units || []).map((unit) => {
      const src = unit.source || {};
      const bits = [];
      const push = (label, value) => {
        if (value !== undefined && value !== null && value !== '') {
          bits.push('<span style="margin-right:12px;"><strong>' + label + '</strong> ' + escapeHtml(String(value)) + '</span>');
        }
      };
      push('order', unit.order);
      push('id', unit.unit_id);
      push('type', unit.type);
      push('h-level', unit.heading_level);
      push('page', src.page);
      push('block', src.block);
      push('column', src.logical_column);
      push('reading-order', src.layout_reading_order_index);
      push('origin', src.content_origin);
      const section = (unit.section_path || []).join(' > ');
      const text = unit.text || '';
      return (
        '<div class="chunk-card">' +
          '<div class="mono" style="line-height:1.9; margin-bottom:6px;">' + bits.join('') + '</div>' +
          (section ? '<div style="font-size:12px; color:var(--accent); margin-bottom:6px;">§ ' + escapeHtml(section) + '</div>' : '') +
          '<details class="details-box"><summary>text (' + text.length + ' chars)</summary>' +
          '<div class="chunk-text" style="margin-top:8px;">' + escapeHtml(text) + '</div></details>' +
        '</div>'
      );
    }).join('') || '<div class="card empty-state"><h3>No units</h3><p>Nothing matches the current filter.</p></div>';
  } catch (e) {
    $('#puSubtitle').textContent = '';
    $('#puRange').textContent = '';
    container.innerHTML = '<div class="error-state">' + escapeHtml(e.message) + '</div>';
  }
}

$('#puApply').addEventListener('click', () => loadParserUnits(0));
$('#puPrev').addEventListener('click', () => {
  const next = Math.max(0, puState.offset - puState.limit);
  if (next !== puState.offset) loadParserUnits(next);
});
$('#puNext').addEventListener('click', () => {
  const next = puState.offset + puState.limit;
  if (next < puState.total) loadParserUnits(next);
});
$('#parserDoc').addEventListener('change', () => loadParserUnits(0));

/* ------------------------------------------------------------------ */
/* Chunk Browser                                                       */
/* ------------------------------------------------------------------ */
async function loadDocChunks() {
  const docId = $('#chunksDoc').value;
  const container = $('#chunksResults');
  if (!docId) { toast('Select a document first.', 'error'); return; }
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Loading chunks…</div>';
  try {
    const data = await api('/api/documents/' + encodeURIComponent(docId) + '/chunks?kb_id=' + encodeURIComponent(labKbId()));
    renderChunkCards(container, data.chunks || [], { refresh: loadDocChunks });
  } catch (e) {
    container.innerHTML = '<div class="error-state">' + escapeHtml(e.message) + '</div>';
  }
}

$('#chunksLoad').addEventListener('click', loadDocChunks);

/* ------------------------------------------------------------------ */
/* KB + document wiring                                                */
/* ------------------------------------------------------------------ */
async function refreshDocSelectors() {
  const kbId = labKbId();
  labDocs = [];
  const placeholder = '<option value="">Select a document</option>';
  if (kbId) {
    try {
      const data = await api('/api/documents?kb_id=' + encodeURIComponent(kbId));
      labDocs = data.documents || [];
    } catch (e) { /* selectors stay empty */ }
  }
  const options = placeholder + labDocs.map((doc) => {
    const name = (doc.metadata && doc.metadata.original_filename) || doc.file_name;
    return '<option value="' + escapeHtml(doc.doc_id) + '">' + escapeHtml(name) + '</option>';
  }).join('');
  $('#parserDoc').innerHTML = options;
  $('#chunksDoc').innerHTML = options;
}

$('#labKbSelect').addEventListener('change', async () => {
  setStoredKb(labKbId());
  $('#rqResults').innerHTML = '';
  $('#searchResults').innerHTML = '';
  $('#parserResults').innerHTML = '';
  $('#chunksResults').innerHTML = '';
  await Promise.all([rqRefreshMethods(), rqLoadGold(labKbId()), refreshDocSelectors()]);
});

(async function init() {
  try {
    await populateKbSelect($('#labKbSelect'));
  } catch (e) {
    toast('Could not load knowledge bases: ' + e.message, 'error');
  }
  await Promise.all([rqRefreshMethods(), rqLoadGold(labKbId()), refreshDocSelectors()]);
})();
