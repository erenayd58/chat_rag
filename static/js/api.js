/* Shared client helpers: API wrapper, toasts, modals, confirm dialog,
 * formatting. Loaded on every page before the page script. */

'use strict';

/** Query helper. */
function $(sel, root) { return (root || document).querySelector(sel); }
function $all(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text === null || text === undefined ? '' : String(text);
  return div.innerHTML;
}

/**
 * Fetch a JSON API endpoint. Resolves with the parsed body when the
 * backend reports success; rejects with an Error carrying the backend
 * message (and the parsed body on `error.body`) otherwise.
 */
async function api(path, options) {
  const opts = options || {};
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.json);
  }
  if (opts.form !== undefined) {
    init.body = opts.form; // FormData: browser sets the content type
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch (e) {
    throw new Error('Bağlantı hatası: ' + e.message);
  }
  let body = null;
  try { body = await response.json(); } catch (e) { /* non-JSON body */ }
  if (!response.ok || (body && body.success === false)) {
    const message = (body && body.error) || ('İstek başarısız (' + response.status + ')');
    const err = new Error(message);
    err.status = response.status;
    err.body = body;
    throw err;
  }
  return body;
}

/* ------------------------------------------------------------------ */
/* Toasts                                                              */
/* ------------------------------------------------------------------ */
function toast(message, type) {
  const container = $('#toastContainer');
  if (!container) return;
  const el = document.createElement('div');
  el.className = 'toast' + (type === 'error' ? ' toast-error' : type === 'success' ? ' toast-success' : '');
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => { el.remove(); }, type === 'error' ? 6500 : 4000);
}

/* ------------------------------------------------------------------ */
/* Modals                                                              */
/* ------------------------------------------------------------------ */
function openModal(id) {
  const backdrop = $('#' + id);
  if (backdrop) backdrop.classList.add('open');
}

function closeModal(id) {
  const backdrop = $('#' + id);
  if (backdrop) backdrop.classList.remove('open');
}

document.addEventListener('click', (event) => {
  if (event.target.classList && event.target.classList.contains('modal-backdrop')) {
    event.target.classList.remove('open');
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    $all('.modal-backdrop.open').forEach((el) => el.classList.remove('open'));
  }
});

/**
 * Promise-based confirm dialog replacing native confirm().
 * confirmDialog({ title, message, confirmLabel, danger }) -> Promise<boolean>
 */
function confirmDialog(opts) {
  return new Promise((resolve) => {
    const backdrop = $('#confirmModal');
    if (!backdrop) { resolve(window.confirm(opts.message || 'Emin misiniz?')); return; }
    $('#confirmTitle').textContent = opts.title || 'Confirm';
    $('#confirmMessage').textContent = opts.message || '';
    const okBtn = $('#confirmOk');
    okBtn.textContent = opts.confirmLabel || 'Confirm';
    okBtn.className = 'btn ' + (opts.danger ? 'btn-danger' : 'btn-primary');

    const done = (value) => {
      backdrop.classList.remove('open');
      okBtn.removeEventListener('click', onOk);
      $('#confirmCancel').removeEventListener('click', onCancel);
      resolve(value);
    };
    const onOk = () => done(true);
    const onCancel = () => done(false);
    okBtn.addEventListener('click', onOk);
    $('#confirmCancel').addEventListener('click', onCancel);
    backdrop.classList.add('open');
  });
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */
function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '—';
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
    + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function fmtSize(bytes) {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/** Product label for a chunking mode / chunker type. */
function chunkingModeLabel(doc) {
  if (doc && doc.chunking_mode === 'deep_analysis') return 'Deep Analysis';
  if (doc && doc.chunking_mode === 'standard') return 'Standard';
  // Older records: fall back to the ingest-time snapshot.
  const snap = doc && doc.pipeline_snapshot;
  const chunker = snap && snap.pipeline && snap.pipeline.chunker;
  if (chunker === 'structure_first') return 'Standard';
  if (chunker === 'v4') return 'Dondurulmuş V4';
  if (chunker === 'legacy') return 'Anlamsal (eski)';
  return null;
}

function chunkerTypeLabel(type) {
  if (type === 'structure_first') return 'Standard (yapı öncelikli)';
  if (type === 'v4') return 'Dondurulmuş V4';
  if (type === 'legacy') return 'Anlamsal (eski)';
  return type || '—';
}

/* ------------------------------------------------------------------ */
/* Shared KB helpers                                                   */
/* ------------------------------------------------------------------ */
const SELECTED_KB_KEY = 'chat_rag.selected_kb';

function getStoredKb() {
  try { return localStorage.getItem(SELECTED_KB_KEY) || ''; } catch (e) { return ''; }
}

function setStoredKb(kbId) {
  try { localStorage.setItem(SELECTED_KB_KEY, kbId || ''); } catch (e) { /* ignore */ }
}

async function loadKbList() {
  const data = await api('/api/kb');
  return data.knowledge_bases || [];
}

/**
 * Fill a <select> with the KB list. Restores the stored selection when
 * still present. Returns the KB list.
 */
async function populateKbSelect(selectEl, opts) {
  const options = opts || {};
  let kbs = [];
  try {
    kbs = await loadKbList();
  } catch (e) {
    selectEl.innerHTML = '<option value="">Failed to load</option>';
    throw e;
  }
  const placeholder = options.placeholder || 'Bilgi tabanı seçin';
  selectEl.innerHTML = '<option value="">' + escapeHtml(placeholder) + '</option>'
    + kbs.map((kb) => '<option value="' + escapeHtml(kb.kb_id) + '">' + escapeHtml(kb.name) + '</option>').join('');
  const stored = getStoredKb();
  if (stored && kbs.some((kb) => kb.kb_id === stored)) selectEl.value = stored;
  return kbs;
}

/* ------------------------------------------------------------------ */
/* Companion viewer status                                             */
/* ------------------------------------------------------------------ */
/**
 * Mark the "Agentic Bölümleme Viewer" link with whether its server answers.
 * The probe runs on the backend (a local address the browser may not be
 * allowed to reach cross-origin); the link itself works regardless.
 */
async function probeViewerStatus() {
  const dot = $('#viewerStatus');
  const card = $('#viewerCardState');
  if (!dot && !card) return;
  let status = null;
  try {
    status = await api('/api/demo/viewer');
  } catch (e) {
    status = { reachable: false };
  }
  const live = !!(status && status.reachable);
  if (dot) {
    dot.classList.toggle('nav-status-live', live);
    dot.classList.toggle('nav-status-off', !live);
    dot.title = live ? 'Viewer running' : 'Viewer not running — start it with .\start-demo.ps1';
  }
  if (card) {
    card.textContent = live ? 'Running' : 'Çalışmıyor';
    card.classList.toggle('viewer-state-live', live);
    card.classList.toggle('viewer-state-off', !live);
    card.title = live ? '' : 'Start it with .\start-demo.ps1 (or python -m amsc.viewer_server in the chunk repo)';
  }
}

probeViewerStatus();
