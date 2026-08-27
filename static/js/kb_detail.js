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
    const modeBadge = mode
      ? '<span class="badge ' + (mode === 'Deep Analysis' ? 'badge-accent' : 'badge-neutral') + '">' + escapeHtml(mode) + '</span>'
      : '<span style="color:var(--text-3);">—</span>';
    return (
      '<tr data-index="' + index + '">' +
        '<td style="font-weight:550; overflow-wrap:anywhere; min-width:200px;">' + escapeHtml(name) + '</td>' +
        '<td>' + statusBadge(doc) + '</td>' +
        '<td>' + modeBadge + '</td>' +
        '<td>' + (doc.chunk_count || 0) + '</td>' +
        '<td style="white-space:nowrap; color:var(--text-2);">' + fmtDate(doc.ingested_at) + '</td>' +
        '<td style="color:var(--text-2);">' + fmtSize(doc.file_size) + '</td>' +
        '<td style="white-space:nowrap; text-align:right;">' +
          '<button class="btn btn-ghost btn-sm doc-reprocess" disabled title="Re-upload the file to reprocess it — automatic reprocessing is planned">Reprocess</button>' +
          '<button class="btn btn-danger-ghost btn-sm doc-delete">Delete</button>' +
        '</td>' +
      '</tr>'
    );
  }).join('');

  container.innerHTML =
    '<table class="table">' +
      '<thead><tr>' +
        '<th>Document</th><th>Status</th><th>Chunking</th><th>Chunks</th><th>Indexed at</th><th>Size</th><th style="text-align:right;">Actions</th>' +
      '</tr></thead>' +
      '<tbody>' + rows + '</tbody>' +
    '</table>';

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
    if (data.chunking_mode === 'deep_analysis' && data.boundary_judge) {
      const j = data.boundary_judge;
      note += ' (Deep Analysis: ' + (j.judge_call_count || 0) + ' judge calls, ' +
        (j.split_votes || 0) + ' split / ' + (j.keep_votes || 0) + ' keep' +
        (j.fallback_count ? ', ' + j.fallback_count + ' fallback' : '') + ')';
    }
    toast(note, 'success');
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
