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
    ? 'Cevaplar yalnız indekslenmiş dokümanlardan üretilir.'
    : 'Başlamak için bir bilgi tabanı seçin.');
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

/** Product label for a source's chunking mode. */
function sourceModeLabel(source) {
  if (source.chunking_mode === 'deep_analysis') return 'Deep Analysis';
  if (source.chunking_mode === 'standard') return 'Standard';
  return source.doc_id ? (docModeByDocId[source.doc_id] || null) : null;
}

/**
 * How a source was found. The fused score is a rank-fusion value, not a
 * probability, so it is never shown as a percentage: the card shows the
 * rank and the retrieval legs, and the raw numbers live in the detail view.
 */
function sourceLegsLabel(source) {
  const legs = source.legs || [];
  if (legs.length === 2) return 'semantic + keyword match';
  if (legs[0] === 'dense') return 'semantic match';
  if (legs[0] === 'lexical') return 'keyword match';
  if (source.expanded_from) return 'same-section continuation';
  return null;
}

let lastSources = [];

function renderSources(sources) {
  if (!sources || !sources.length) return '';
  lastSources = sources;
  const cards = sources.map((source, index) => {
    const pages = sourcePages(source);
    const mode = sourceModeLabel(source);
    const legs = sourceLegsLabel(source);
    const heading = source.heading || source.section;
    return (
      '<div class="source-card source-card-clickable' + (source.used ? ' source-card-used' : '') + '" data-source-index="' + index + '" role="button" tabindex="0" title="Parça metnini göster">' +
        '<div class="source-head">' +
          (source.label ? '<span class="source-label">' + escapeHtml(source.label) + '</span>' : '') +
          '<span class="source-doc">' + escapeHtml(source.document || 'Doküman') + '</span>' +
          (pages ? '<span class="badge badge-neutral">' + escapeHtml(pages) + '</span>' : '') +
          (mode ? '<span class="badge badge-accent">' + escapeHtml(mode) + '</span>' : '') +
          (source.used ? '<span class="badge badge-success">Used in answer</span>' : '') +
        '</div>' +
        (heading ? '<div class="source-section">' + escapeHtml(heading) + '</div>' : '') +
        (legs ? '<div class="source-meta">' + escapeHtml(legs) + (source.rank ? ' · rank ' + source.rank : '') + '</div>' : '') +
        (source.content_preview ? '<div class="source-preview">' + escapeHtml(source.content_preview) + '</div>' : '') +
      '</div>'
    );
  }).join('');
  return '<div class="sources-block"><div class="sources-label">Sources</div>' + cards + '</div>';
}

function openSourceModal(source) {
  const modal = $('#chunkModal');
  if (!modal) return;
  const pages = sourcePages(source);
  const mode = sourceModeLabel(source);
  $('#chunkModalTitle').textContent = (source.label ? source.label + ' · ' : '') + (source.document || 'Chunk');
  const facts = [
    ['Section', source.heading || source.section || '—'],
    ['Pages', pages || '—'],
    ['Bölümleme', mode || '—'],
    ['Bulan yöntem', sourceLegsLabel(source) || '—'],
    ['Birleşik sıra', source.rank ? String(source.rank) : '—'],
    ['Anlamsal sıra / kelime sırası', (source.dense_rank || '—') + ' / ' + (source.bm25_rank || '—')],
    ['Birleşik skor', (source.score === null || source.score === undefined) ? '—' : Number(source.score).toFixed(4)],
    ['Parça kimliği', source.chunk_id || '—'],
  ];
  $('#chunkModalFacts').innerHTML = facts.map((row) =>
    '<div class="def-row"><span class="def-key">' + escapeHtml(row[0]) + '</span><span class="def-val mono">' + escapeHtml(String(row[1])) + '</span></div>'
  ).join('');
  $('#chunkModalText').textContent = source.content || source.content_preview || '';
  openModal('chunkModal');
}

function answerNotices(metadata) {
  if (!metadata) return '';
  let html = '';
  if (metadata.reindex_required) {
    html += '<div class="msg-notice-inline">Semantic search is off for this knowledge base — its vectors were built with another embedding model. ' +
      'Keyword results were used. Yeniden indeksle it under the knowledge base\'s <strong>Settings</strong>.</div>';
  }
  const answer = metadata.answer || {};
  if (answer.fallback_used) {
    html += '<div class="msg-notice-inline">Answered by the local fallback model — the primary answer service was unavailable.</div>';
  } else if (answer.skipped === 'no_sources') {
    html += '<div class="msg-notice-inline">No matching passages were found, so no answer was generated.</div>';
  }
  return html;
}

function addAssistantMessage(answer, sources, metadata) {
  removeEmptyState();
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.innerHTML =
    '<div class="msg-bubble"></div>' +
    answerNotices(metadata) +
    renderSources(sources) +
    '<div class="msg-time">' + timeNow() + '</div>';
  $('.msg-bubble', el).textContent = answer;
  $all('.source-card-clickable', el).forEach((card) => {
    const open = () => openSourceModal(lastSources[Number(card.dataset.sourceIndex)]);
    card.addEventListener('click', open);
    card.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
  });
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
    addAssistantMessage(data.answer, data.sources, data.metadata);
  } catch (e) {
    removeTyping();
    if (e.body && e.body.overloaded) {
      // The server refused the question rather than queue it: every query
      // slot (or every answer-model slot) is in use. The question is put
      // back so one click re-sends it after the suggested pause.
      const wait = Math.max(1, Math.round(Number(e.body.retry_after_seconds) || 5));
      input.value = question;
      addNotice(
        '<strong>The server is busy right now.</strong> ' +
        (e.body.reason === 'answer_capacity'
          ? 'Every answer-model slot is in use. '
          : 'Every question slot is in use. ') +
        'Your question was not lost — try sending it again in about ' + wait + ' seconds.'
      );
    } else if (e.body && e.body.timed_out) {
      input.value = question;
      addNotice(
        '<strong>The question took too long to answer</strong> and was stopped at the time limit. ' +
        'Try again, or ask a narrower question.',
        'error'
      );
    } else if (e.body && e.body.generation_unavailable) {
      addNotice(
        '<strong>Answer generation is currently unavailable</strong> — the language model could not be reached. ' +
        'Doküman retrieval still works: you can inspect what would have been retrieved in the <a href="/lab">Lab</a>.' +
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
    title: 'Sohbeti temizle',
    message: 'Sohbet geçmişi silinsin mi?',
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
    toast('Bilgi tabanları yüklenemedi: ' + e.message, 'error');
  }
  const kbId = currentKbId();
  setComposerEnabled(!!kbId);
  if (kbId) refreshDocModes(kbId);
})();
