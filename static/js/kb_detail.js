/* Knowledge Base detail page: Overview | Documents | Settings + upload. */
'use strict';

const KB_ID = $('#kbDetailPage').dataset.kbId;
let currentKb = null;

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
/* KB header + overview                                                */
/* ------------------------------------------------------------------ */
async function loadKb() {
  try {
    const data = await api('/api/kb/' + encodeURIComponent(KB_ID));
    currentKb = data.kb;
  } catch (e) {
    $('#kbTitle').textContent = 'Knowledge base not found';
    toast(e.message, 'error');
    return;
  }
  const desc = (currentKb.extra && currentKb.extra.description) || '';
  $('#kbTitle').textContent = currentKb.name;
  $('#crumbName').textContent = currentKb.name;
  $('#kbDescription').textContent = desc;
  $('#settingsName').value = currentKb.name;
  $('#settingsDesc').value = desc;

  const chunker = currentKb.chunker && currentKb.chunker.type;
  $('#configRows').innerHTML =
    '<div class="def-row"><span class="def-key">Chunking profile</span><span class="def-val">' + escapeHtml(chunkerTypeLabel(chunker)) + '</span></div>' +
    '<div class="def-row"><span class="def-key">Retrieval method</span><span class="def-val">' + escapeHtml((currentKb.retrieval_method || 'hybrid').toUpperCase()) + '</span></div>' +
    '<div class="def-row"><span class="def-key">Vector store</span><span class="def-val">' + escapeHtml((currentKb.vector_db_provider || 'chroma').toUpperCase()) + '</span></div>' +
    '<div class="def-row"><span class="def-key">Knowledge base ID</span><span class="def-val mono">' + escapeHtml(KB_ID) + '</span></div>';
}

async function loadStats() {
  try {
    const data = await api('/api/stats?kb_id=' + encodeURIComponent(KB_ID));
    const stats = data.stats;
    $('#statDocs').textContent = stats.total_documents;
    $('#statChunks').textContent = stats.total_chunks;
    $('#statSize').textContent = stats.total_size_mb + ' MB';
    $('#statLatest').textContent = stats.latest_ingestion ? fmtDate(stats.latest_ingestion) : '—';
  } catch (e) {
    /* tiles stay at placeholder */
  }
}

/* ------------------------------------------------------------------ */
/* Documents tab                                                       */
/* ------------------------------------------------------------------ */

/**
 * The Deep Analysis summary the backend wrote at ingest, or null for a
 * Standard document or a record from before the summary existed. The
 * wording (label, tone, detail) is the backend's, so the console never
 * invents a status of its own.
 */
function deepAnalysisSummary(doc) {
  if (!doc || doc.chunking_mode !== 'deep_analysis') return null;
  const snap = doc.pipeline_snapshot;
  const options = snap && snap.ingest_options;
  const summary = options && options.deep_analysis_summary;
  return summary && summary.label ? summary : null;
}

function deepStatusIcon(tone) {
  if (tone === 'success') return '✓';
  if (tone === 'danger') return '✕';
  if (tone === 'warn') return '⚠';
  return '•';
}

function fmtPair(pair, unit) {
  if (!pair) return '—';
  const s = pair.standard, d = pair.deep;
  if (s === undefined || d === undefined) return '—';
  return escapeHtml(String(s)) + ' → ' + escapeHtml(String(d)) + (unit ? ' ' + unit : '');
}

function defRow(key, value) {
  return '<div class="def-row"><span class="def-key">' + key + '</span><span class="def-val">' + value + '</span></div>';
}

/** The collapsible detail block under a Deep Analysis document row. */
function deepAnalysisDetails(d) {
  const proposer = d.proposer, verifier = d.verifier, checks = d.checks || {}, sections = d.sections || {};
  let smellRows = '';
  if (d.smells) {
    Object.keys(d.smells).forEach((key) => {
      smellRows += defRow(escapeHtml(key.replace(/_/g, ' ')), fmtPair(d.smells[key]));
    });
  }
  const quality =
    defRow('Boundary smells', fmtPair(d.smell_total)) +
    defRow('Chunks', fmtPair(d.chunk_count)) +
    defRow('Structural regressions', escapeHtml(String(d.structural_regression_count == null ? '—' : d.structural_regression_count))) +
    smellRows;
  const llm = d.uses_llm
    ? defRow('Model', escapeHtml(d.model_id || '—')) +
      (d.verifier_model_id && d.verifier_model_id !== d.model_id ? defRow('Verifier model', escapeHtml(d.verifier_model_id)) : '') +
      (proposer ? defRow('Proposer calls', escapeHtml(String(proposer.answered_count)) + ' answered' + (proposer.failed_count ? ', ' + escapeHtml(String(proposer.failed_count)) + ' failed' : '') + ' of ' + escapeHtml(String(proposer.call_count))) : '') +
      (verifier ? defRow('Verifier', escapeHtml(String(verifier.accepted)) + ' accepted / ' + escapeHtml(String(verifier.reverted)) + ' reverted of ' + escapeHtml(String(verifier.group_count)) + ' change groups') : '') +
      defRow('Sections changed by the model', escapeHtml(String(sections.changed_by_llm == null ? '—' : sections.changed_by_llm)))
    : defRow('Model', 'not used' + (d.model_id ? ' (' + escapeHtml(d.model_id) + ')' : '')) +
      (proposer ? defRow('Proposer calls', escapeHtml(String(proposer.failed_count)) + ' failed of ' + escapeHtml(String(proposer.call_count))) : '');
  const contract =
    defRow('Sections moved by the quality contract', escapeHtml(String(sections.moved_by_contract == null ? '—' : sections.moved_by_contract))) +
    defRow('Hard token cap', checks.hard_cap_ok === undefined ? '—' : (checks.hard_cap_ok ? 'held' : 'BREACHED') + ' (max ' + escapeHtml(String(checks.max_token_count)) + ' / ' + escapeHtml(String(checks.hard_max_tokens)) + ')') +
    defRow('Coverage', checks.coverage_ok === undefined ? '—' : (checks.coverage_ok ? 'every unit once' : 'MISMATCH'));
  return (
    '<div class="deep-detail">' +
      '<div class="deep-detail-head deep-status-' + escapeHtml(d.tone) + '">' + deepStatusIcon(d.tone) + ' ' + escapeHtml(d.headline || d.label) + '</div>' +
      '<div class="deep-detail-text">' + escapeHtml(d.detail || '') + (d.fallback_reason ? ' ' + escapeHtml(d.fallback_reason) : '') + '</div>' +
      '<div class="deep-detail-grid">' +
        '<div><div class="deep-detail-title">Quality (Standard → Deep)</div>' + quality + '</div>' +
        '<div><div class="deep-detail-title">Language model</div>' + llm + '</div>' +
        '<div><div class="deep-detail-title">Contract &amp; checks</div>' + contract + '</div>' +
      '</div>' +
      '<div class="deep-detail-foot">Final status: <code>' + escapeHtml(d.status) + '</code></div>' +
    '</div>'
  );
}

function statusBadge(doc) {
  const status = doc.status || 'indexed';
  if (status === 'indexed') return '<span class="badge badge-success"><span class="dot"></span>Indexed</span>';
  if (status === 'processing') return '<span class="badge badge-warn"><span class="dot"></span>Processing</span>';
  if (status === 'failed') return '<span class="badge badge-danger"><span class="dot"></span>Failed</span>';
  return '<span class="badge badge-neutral">' + escapeHtml(status) + '</span>';
}

async function loadDocuments() {
  const container = $('#docsContainer');
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Loading documents…</div>';
  let docs;
  try {
    const data = await api('/api/documents?kb_id=' + encodeURIComponent(KB_ID));
    docs = data.documents || [];
  } catch (e) {
    container.innerHTML = '<div class="error-state" style="margin:16px;">Could not load documents: ' + escapeHtml(e.message) + '</div>';
    return;
  }

  if (docs.length === 0) {
    container.innerHTML =
      '<div class="empty-state">' +
      '<h3>No documents yet</h3>' +
      '<p>Upload a PDF into this knowledge base to index it for search and chat.</p>' +
      '<button class="btn btn-primary" onclick="openUploadModal()">Upload Document</button>' +
      '</div>';
    return;
  }

  const rows = docs.map((doc, index) => {
    const name = (doc.metadata && doc.metadata.original_filename) || doc.file_name;
    const mode = chunkingModeLabel(doc);
    const deep = deepAnalysisSummary(doc);
    let modeCell = mode
      ? '<span class="badge ' + (mode === 'Deep Analysis' ? 'badge-accent' : 'badge-neutral') + '">' + escapeHtml(mode) + '</span>'
      : '<span style="color:var(--text-3);">—</span>';
    if (deep) {
      modeCell += '<div class="deep-status deep-status-' + escapeHtml(deep.tone) + '">' +
        deepStatusIcon(deep.tone) + ' ' + escapeHtml(deep.label) + '</div>';
    }
    const detailsButton = deep
      ? '<button class="btn btn-ghost btn-sm doc-details" aria-expanded="false">Details</button>'
      : '';
    const detailRow = deep
      ? '<tr class="doc-detail-row" data-for="' + index + '" hidden><td colspan="7">' + deepAnalysisDetails(deep) + '</td></tr>'
      : '';
    return (
      '<tr data-index="' + index + '">' +
        '<td style="font-weight:550; overflow-wrap:anywhere; min-width:200px;">' + escapeHtml(name) + '</td>' +
        '<td>' + statusBadge(doc) + '</td>' +
        '<td>' + modeCell + '</td>' +
        '<td>' + (doc.chunk_count || 0) + '</td>' +
        '<td style="white-space:nowrap; color:var(--text-2);">' + fmtDate(doc.ingested_at) + '</td>' +
        '<td style="color:var(--text-2);">' + fmtSize(doc.file_size) + '</td>' +
        '<td style="white-space:nowrap; text-align:right;">' +
          detailsButton +
          '<button class="btn btn-ghost btn-sm doc-reprocess" disabled title="Re-upload the file to reprocess it — automatic reprocessing is planned">Reprocess</button>' +
          '<button class="btn btn-danger-ghost btn-sm doc-delete">Delete</button>' +
        '</td>' +
      '</tr>' + detailRow
    );
  }).join('');

  container.innerHTML =
    '<table class="table">' +
      '<thead><tr>' +
        '<th>Document</th><th>Status</th><th>Chunking</th><th>Chunks</th><th>Indexed at</th><th>Size</th><th style="text-align:right;">Actions</th>' +
      '</tr></thead>' +
      '<tbody>' + rows + '</tbody>' +
    '</table>';

  $all('.doc-details', container).forEach((btn) => {
    btn.addEventListener('click', () => {
      const row = btn.closest('tr');
      const detail = container.querySelector('.doc-detail-row[data-for="' + row.dataset.index + '"]');
      if (!detail) return;
      const open = detail.hidden;
      detail.hidden = !open;
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      btn.textContent = open ? 'Hide details' : 'Details';
    });
  });

  $all('.doc-delete', container).forEach((btn) => {
    btn.addEventListener('click', async () => {
      const row = btn.closest('tr');
      const doc = docs[Number(row.dataset.index)];
      const name = (doc.metadata && doc.metadata.original_filename) || doc.file_name;
      const ok = await confirmDialog({
        title: 'Delete document',
        message: 'Delete "' + name + '" and its ' + (doc.chunk_count || 0) + ' chunks from this knowledge base?',
        confirmLabel: 'Delete',
        danger: true
      });
      if (!ok) return;
      try {
        await api('/api/documents/' + encodeURIComponent(doc.doc_id) + '?kb_id=' + encodeURIComponent(KB_ID), { method: 'DELETE' });
        toast('Document deleted', 'success');
        loadDocuments();
        loadStats();
      } catch (e) {
        toast('Delete failed: ' + e.message, 'error');
      }
    });
  });
}

/* ------------------------------------------------------------------ */
/* Upload                                                              */
/* ------------------------------------------------------------------ */
let selectedFile = null;

function openUploadModal() {
  selectedFile = null;
  $('#fileInput').value = '';
  $('#chooseFileBtn').textContent = 'Choose a file…';
  $('#uploadSubmit').disabled = true;
  $('#uploadError').innerHTML = '';
  $('#uploadProgress').style.display = 'none';
  openModal('uploadModal');
}

// Reflect the chosen chunking mode on the option cards.
$all('input[name="chunkMode"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    $('#optStandard').classList.toggle('selected', radio.value === 'standard' && radio.checked);
    $('#optDeep').classList.toggle('selected', radio.value === 'deep_analysis' && radio.checked);
    if (radio.checked && radio.value === 'standard') $('#optDeep').classList.remove('selected');
    if (radio.checked && radio.value === 'deep_analysis') $('#optStandard').classList.remove('selected');
  });
});

$('#uploadBtn').addEventListener('click', openUploadModal);
$('#chooseFileBtn').addEventListener('click', () => $('#fileInput').click());

$('#fileInput').addEventListener('change', (event) => {
  selectedFile = event.target.files[0] || null;
  $('#chooseFileBtn').textContent = selectedFile ? selectedFile.name : 'Choose a file…';
  $('#uploadSubmit').disabled = !selectedFile;
});

$('#uploadSubmit').addEventListener('click', async () => {
  if (!selectedFile) return;
  const submit = $('#uploadSubmit');
  const cancel = $('#uploadCancel');
  const progress = $('#uploadProgress');
  const errorBox = $('#uploadError');
  errorBox.innerHTML = '';
  submit.disabled = true;
  cancel.disabled = true;
  progress.style.display = 'block';

  const deepAnalysis = (document.querySelector('input[name="chunkMode"]:checked') || {}).value === 'deep_analysis';
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('kb_id', KB_ID);
  form.append('deep_analysis', deepAnalysis ? 'true' : 'false');

  try {
    const data = await api('/api/documents/upload', { method: 'POST', form: form });
    closeModal('uploadModal');
    let note = data.filename + ' indexed — ' + data.chunks_created + ' chunks created';
    let tone = 'success';
    if (data.chunking_mode === 'deep_analysis' && data.deep_analysis) {
      const d = data.deep_analysis;
      note += '. Deep Analysis: ' + d.label;
      if (d.tone !== 'success') tone = '';
    }
    toast(note, tone);
    loadDocuments();
    loadStats();
  } catch (e) {
    errorBox.innerHTML = '<div class="error-state" style="margin-top:12px;">' + escapeHtml(e.message) + '</div>';
  } finally {
    progress.style.display = 'none';
    submit.disabled = !selectedFile;
    cancel.disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */
$('#settingsSave').addEventListener('click', async () => {
  const name = $('#settingsName').value.trim();
  const description = $('#settingsDesc').value.trim();
  const errorBox = $('#settingsError');
  errorBox.innerHTML = '';
  if (!name) {
    errorBox.innerHTML = '<div class="error-state" style="margin-bottom:12px;">A name is required.</div>';
    return;
  }
  try {
    const extra = Object.assign({}, (currentKb && currentKb.extra) || {}, { description: description });
    await api('/api/kb/' + encodeURIComponent(KB_ID), { method: 'PUT', json: { name: name, extra: extra } });
    toast('Settings saved', 'success');
    loadKb();
  } catch (e) {
    errorBox.innerHTML = '<div class="error-state" style="margin-bottom:12px;">' + escapeHtml(e.message) + '</div>';
  }
});

$('#deleteKbBtn').addEventListener('click', async () => {
  const ok = await confirmDialog({
    title: 'Delete knowledge base',
    message: 'Delete "' + ((currentKb && currentKb.name) || 'this knowledge base') + '" and its search index? This cannot be undone.',
    confirmLabel: 'Delete',
    danger: true
  });
  if (!ok) return;
  try {
    await api('/api/kb/' + encodeURIComponent(KB_ID), { method: 'DELETE' });
    toast('Knowledge base deleted', 'success');
    window.location.href = '/';
  } catch (e) {
    toast('Delete failed: ' + e.message, 'error');
  }
});

/* ------------------------------------------------------------------ */
loadKb();
loadStats();
loadDocuments();
