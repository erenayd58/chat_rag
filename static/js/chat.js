/* Chat page: KB-scoped conversation with source cards. */
'use strict';

let docModeByDocId = {};

function currentKbId() {
  return $('#kbSelect').value || '';
}

function setComposerEnabled(enabled, hint) {
  $('#queryInput').disabled = !enabled;
  $('#sendBtn').disabled = !enabled;
  $('#composerHint').textContent = hint || (enabled
    ? 'Answers are generated from the indexed documents only.'
    : 'Select a knowledge base to start.');
}

async function refreshDocModes(kbId) {
  docModeByDocId = {};
  if (!kbId) return;
  try {
    const data = await api('/api/documents?kb_id=' + encodeURIComponent(kbId));
    (data.documents || []).forEach((doc) => {
      const label = chunkingModeLabel(doc);
      if (doc.doc_id && label) docModeByDocId[doc.doc_id] = label;
    });
  } catch (e) { /* badges are optional */ }
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */
function timeNow() {
  return new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function removeEmptyState() {
  const empty = $('#chatEmpty');
  if (empty) empty.remove();
}

function scrollToBottom() {
  const scroll = $('#chatScroll');
  scroll.scrollTop = scroll.scrollHeight;
}

function addUserMessage(text) {
  removeEmptyState();
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = '<div class="msg-bubble"></div><div class="msg-time">' + timeNow() + '</div>';
  $('.msg-bubble', el).textContent = text;
  $('#chatThread').appendChild(el);
  scrollToBottom();
}

function sourcePages(source) {
  const pages = source.pages;
  if (!pages || !pages.length) return null;
  if (pages.length === 1) return 'p. ' + pages[0];
  return 'pp. ' + pages[0] + '–' + pages[pages.length - 1];
}

function renderSources(sources) {
  if (!sources || !sources.length) return '';
  const cards = sources.slice(0, 5).map((source) => {
    const pages = sourcePages(source);
    const mode = source.doc_id ? docModeByDocId[source.doc_id] : null;
    const score = (source.score === null || source.score === undefined)
      ? null : Math.round(source.score * 100);
    return (
      '<div class="source-card">' +
        '<div class="source-head">' +
          '<span class="source-doc">' + escapeHtml(source.document || 'Document') + '</span>' +
          (pages ? '<span class="badge badge-neutral">' + escapeHtml(pages) + '</span>' : '') +
          (score !== null ? '<span class="badge badge-neutral">' + score + '% relevance</span>' : '') +
          (mode ? '<span class="badge badge-accent">' + escapeHtml(mode) + '</span>' : '') +
        '</div>' +
        (source.section ? '<div class="source-section">' + escapeHtml(source.section) + '</div>' : '') +
        (source.content_preview ? '<div class="source-preview">' + escapeHtml(source.content_preview) + '</div>' : '') +
      '</div>'
    );
  }).join('');
  return '<div class="sources-block"><div class="sources-label">Sources</div>' + cards + '</div>';
}

function addAssistantMessage(answer, sources) {
  removeEmptyState();
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.innerHTML =
    '<div class="msg-bubble"></div>' +
    renderSources(sources) +
    '<div class="msg-time">' + timeNow() + '</div>';
  $('.msg-bubble', el).textContent = answer;
  $('#chatThread').appendChild(el);
  scrollToBottom();
}

function addNotice(html, kind) {
  removeEmptyState();
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.innerHTML = '<div class="' + (kind === 'error' ? 'msg-error' : 'msg-notice') + '">' + html + '</div>';
  $('#chatThread').appendChild(el);
  scrollToBottom();
}

function addTyping() {
  removeEmptyState();
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.id = 'typingIndicator';
  el.innerHTML = '<div class="msg-bubble" style="width:fit-content;"><span class="typing"><span></span><span></span><span></span></span></div>';
  $('#chatThread').appendChild(el);
  scrollToBottom();
}

function removeTyping() {
  const el = $('#typingIndicator');
  if (el) el.remove();
}

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */
async function sendQuestion() {
  const input = $('#queryInput');
  const question = input.value.trim();
  const kbId = currentKbId();
  if (!question || !kbId) return;

  input.value = '';
  setComposerEnabled(false, 'Waiting for the answer…');
  addUserMessage(question);
  addTyping();

  try {
    const data = await api('/api/query', {
      method: 'POST',
      json: { question: question, kb_id: kbId, top_k: 5 }
    });
    removeTyping();
    addAssistantMessage(data.answer, data.sources);
  } catch (e) {
    removeTyping();
    if (e.body && e.body.generation_unavailable) {
      addNotice(
        '<strong>Answer generation is currently unavailable</strong> — the language model could not be reached. ' +
        'Document retrieval still works: you can inspect what would have been retrieved in the <a href="/lab">Lab</a>.' +
        '<details class="details-box"><summary>Details</summary><div class="details-body">' + escapeHtml(e.message) + '</div></details>'
      );
    } else {
      addNotice('<strong>The question could not be processed.</strong><br>' + escapeHtml(e.message), 'error');
    }
  } finally {
    setComposerEnabled(true);
    $('#queryInput').focus();
  }
}

$('#sendBtn').addEventListener('click', sendQuestion);
$('#queryInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendQuestion();
});

$('#clearBtn').addEventListener('click', async () => {
  const ok = await confirmDialog({
    title: 'Clear conversation',
    message: 'Clear the current conversation history?',
    confirmLabel: 'Clear'
  });
  if (!ok) return;
  try {
    await api('/api/clear', { method: 'POST', json: { kb_id: currentKbId() || null } });
  } catch (e) { /* clearing the view is still fine */ }
  $('#chatThread').innerHTML =
    '<div class="empty-state" id="chatEmpty"><h3>Conversation cleared</h3><p>Ask a new question to continue.</p></div>';
});

$('#kbSelect').addEventListener('change', async () => {
  const kbId = currentKbId();
  setStoredKb(kbId);
  setComposerEnabled(!!kbId);
  await refreshDocModes(kbId);
  if (kbId) $('#queryInput').focus();
});

/* ------------------------------------------------------------------ */
(async function init() {
  try {
    await populateKbSelect($('#kbSelect'));
  } catch (e) {
    toast('Could not load knowledge bases: ' + e.message, 'error');
  }
  const kbId = currentKbId();
  setComposerEnabled(!!kbId);
  if (kbId) refreshDocModes(kbId);
})();
