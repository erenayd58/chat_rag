/* Knowledge Bases list page. */
'use strict';

async function loadKbPage() {
  const container = $('#kbContainer');
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Loading knowledge bases…</div>';
  let kbs, documents;
  try {
    const [kbData, docData] = await Promise.all([api('/api/kb'), api('/api/documents')]);
    kbs = kbData.knowledge_bases || [];
    documents = docData.documents || [];
  } catch (e) {
    container.innerHTML = '<div class="error-state">Bilgi tabanları yüklenemedi: ' + escapeHtml(e.message) + '</div>';
    return;
  }

  if (kbs.length === 0) {
    container.innerHTML =
      '<div class="card empty-state">' +
      '<h3>Henüz bilgi tabanı yok</h3>' +
      '<p>Bir bilgi tabanı oluşturun, içine PDF yükleyin ve sorularınızı sormaya başlayın.</p>' +
      '<button class="btn btn-primary" onclick="openModal(\'createKbModal\')">Yeni bilgi tabanı</button>' +
      '</div>';
    return;
  }

  // Aggregate document stats per KB.
  const byKb = {};
  documents.forEach((doc) => {
    const id = doc.kb_id;
    if (!id) return;
    if (!byKb[id]) byKb[id] = { docs: 0, chunks: 0, latest: '' };
    byKb[id].docs += 1;
    byKb[id].chunks += doc.chunk_count || 0;
    if ((doc.ingested_at || '') > byKb[id].latest) byKb[id].latest = doc.ingested_at;
  });

  const cards = kbs.map((kb) => {
    const stats = byKb[kb.kb_id] || { docs: 0, chunks: 0, latest: '' };
    const desc = (kb.extra && kb.extra.description) || '';
    const status = stats.docs > 0
      ? '<span class="badge badge-success"><span class="dot"></span>İndekslendi</span>'
      : '<span class="badge badge-neutral"><span class="dot"></span>Boş</span>';
    const chunker = kb.chunker && kb.chunker.type;
    return (
      '<div class="card kb-card" data-kb="' + escapeHtml(kb.kb_id) + '">' +
        '<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">' +
          '<div style="min-width:0;">' +
            '<div style="font-size:15px; font-weight:600; overflow-wrap:anywhere;">' + escapeHtml(kb.name) + '</div>' +
            (desc ? '<div style="font-size:12.5px; color:var(--text-2); margin-top:3px;">' + escapeHtml(desc) + '</div>' : '') +
          '</div>' +
          status +
        '</div>' +
        '<div style="display:flex; gap:18px; margin-top:14px; font-size:12.5px; color:var(--text-2);">' +
          '<span><strong style="color:var(--text);">' + stats.docs + '</strong> document' + (stats.docs === 1 ? '' : 's') + '</span>' +
          '<span><strong style="color:var(--text);">' + stats.chunks + '</strong> chunks</span>' +
        '</div>' +
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; padding-top:12px; border-top:1px solid var(--border);">' +
          '<span style="font-size:11.5px; color:var(--text-3);">' +
            (stats.latest ? 'Güncellendi ' + fmtDate(stats.latest) : 'Henüz doküman yok') +
          '</span>' +
          '<span class="badge badge-neutral">' + escapeHtml(chunkerTypeLabel(chunker)) + '</span>' +
        '</div>' +
        '<div style="display:flex; gap:8px; margin-top:12px;">' +
          '<button class="btn btn-secondary btn-sm kb-open">Open</button>' +
          '<button class="btn btn-danger-ghost btn-sm kb-delete">Delete</button>' +
        '</div>' +
      '</div>'
    );
  }).join('');

  container.innerHTML = '<div class="grid-cards">' + cards + '</div>';

  $all('.kb-card', container).forEach((card) => {
    const kbId = card.dataset.kb;
    const kb = kbs.find((k) => k.kb_id === kbId);
    card.addEventListener('click', () => { window.location.href = '/kb/' + encodeURIComponent(kbId); });
    $('.kb-open', card).addEventListener('click', (e) => {
      e.stopPropagation();
      window.location.href = '/kb/' + encodeURIComponent(kbId);
    });
    $('.kb-delete', card).addEventListener('click', async (e) => {
      e.stopPropagation();
      const stats = byKb[kbId] || { docs: 0 };
      const ok = await confirmDialog({
        title: 'Bilgi tabanını sil',
        message: 'Delete "' + kb.name + '"' + (stats.docs ? ' and its ' + stats.docs + ' indexed document' + (stats.docs === 1 ? '' : 's') : '') + '? This removes its search index and cannot be undone.',
        confirmLabel: 'Delete',
        danger: true
      });
      if (!ok) return;
      try {
        await api('/api/kb/' + encodeURIComponent(kbId), { method: 'DELETE' });
        toast('Bilgi tabanı silindi', 'success');
        loadKbPage();
      } catch (err) {
        toast('Silme başarısız: ' + err.message, 'error');
      }
    });
  });
}

function setupCreateKb() {
  $('#newKbBtn').addEventListener('click', () => {
    $('#kbNameInput').value = '';
    $('#kbDescInput').value = '';
    $('#createKbError').innerHTML = '';
    openModal('createKbModal');
    $('#kbNameInput').focus();
  });

  $('#createKbSubmit').addEventListener('click', async () => {
    const name = $('#kbNameInput').value.trim();
    const description = $('#kbDescInput').value.trim();
    const errorBox = $('#createKbError');
    errorBox.innerHTML = '';
    if (!name) {
      errorBox.innerHTML = '<div class="error-state">A name is required.</div>';
      return;
    }
    const btn = $('#createKbSubmit');
    btn.disabled = true;
    try {
      // Product default: structure-first chunking. Advanced knobs
      // (embedding model, vector store, retrieval method) keep their
      // backend defaults and stay out of the product surface.
      const data = await api('/api/kb', {
        method: 'POST',
        json: {
          name: name,
          chunker: { type: 'structure_first' },
          extra: description ? { description: description } : {}
        }
      });
      closeModal('createKbModal');
      toast('Bilgi tabanı oluşturuldu', 'success');
      window.location.href = '/kb/' + encodeURIComponent(data.kb.kb_id);
    } catch (e) {
      errorBox.innerHTML = '<div class="error-state">' + escapeHtml(e.message) + '</div>';
    } finally {
      btn.disabled = false;
    }
  });

  $('#kbNameInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('#createKbSubmit').click();
  });
}

setupCreateKb();
loadKbPage();
