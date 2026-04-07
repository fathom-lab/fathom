// FATHOM Dashboard — vanilla JS, zero dependencies

const API = window.location.origin;

// ── Tab switching ───────────────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById('tab-score').style.display = tab === 'score' ? 'block' : 'none';
  document.getElementById('tab-compare').style.display = tab === 'compare' ? 'block' : 'none';
}

// ── Grade color mapping ─────────────────────────────────────────────

function gradeColor(grade) {
  const map = {
    S: '#ffd700', A: '#00ff00', B: '#00e5ff',
    C: '#ffeb3b', D: '#ff9800', F: '#ff3d00'
  };
  return map[grade] || '#666';
}

// ── API call helper ─────────────────────────────────────────────────

async function apiCall(endpoint, body) {
  const headers = { 'Content-Type': 'application/json' };
  const key = document.querySelector('meta[name="fathom-key"]');
  if (key) headers['x-api-key'] = key.content;

  const resp = await fetch(API + endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ── Score ────────────────────────────────────────────────────────────

async function runScore() {
  const input = document.getElementById('score-input');
  const prompt = input.value.trim();
  if (!prompt) return;

  const btn = document.getElementById('score-btn');
  const loading = document.getElementById('score-loading');
  const result = document.getElementById('score-result');

  btn.disabled = true;
  loading.classList.add('active');
  result.classList.remove('active');

  try {
    const resp = await apiCall('/v2/fathom_score', { prompt });
    const d = resp.data;
    const meta = resp.meta;

    // Grade badge
    const badge = document.getElementById('grade-badge');
    badge.textContent = d.grade;
    badge.className = 'grade-badge grade-' + d.grade;

    // Score
    document.getElementById('score-value').textContent = Math.round(d.fathom_score);

    // Interpretation
    document.getElementById('score-interpretation').textContent = d.interpretation;

    // Depth bar (scale 7.0 to 9.5)
    const depthRaw = d.depth_result.depth_score;
    const depthPct = Math.max(0, Math.min(100, ((depthRaw - 7.0) / 2.5) * 100));
    const bar = document.getElementById('depth-bar');
    bar.style.width = depthPct + '%';
    bar.style.background = `linear-gradient(90deg, ${gradeColor(d.grade)}, ${gradeColor(d.grade)}88)`;
    document.getElementById('depth-value').textContent = depthRaw != null ? depthRaw.toFixed(3) : 'N/A';
    document.getElementById('depth-value').style.color = gradeColor(d.grade);

    // Layer bars
    const dr = d.depth_result;
    const maxRatio = Math.max(dr.early_ratio, dr.mid_ratio, dr.late_ratio, 0.01);
    setBar('bar-early', 'val-early', dr.early_ratio, maxRatio);
    setBar('bar-mid', 'val-mid', dr.mid_ratio, maxRatio);
    setBar('bar-late', 'val-late', dr.late_ratio, maxRatio);

    // Stats
    document.getElementById('stat-entropy').textContent =
      dr.layer_entropy != null ? dr.layer_entropy.toFixed(3) : '--';
    document.getElementById('stat-features').textContent = dr.feature_count || '--';
    document.getElementById('stat-wdepth').textContent =
      dr.weighted_depth_score != null ? dr.weighted_depth_score.toFixed(3) : '--';

    // Compute time
    document.getElementById('compute-time').textContent =
      `${meta.compute_time_ms.toFixed(0)}ms | ${meta.model}`;

    result.classList.add('active');
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    loading.classList.remove('active');
  }
}

function setBar(barId, valId, ratio, maxRatio) {
  const height = Math.max(2, (ratio / maxRatio) * 55);
  document.getElementById(barId).style.height = height + 'px';
  document.getElementById(valId).textContent = (ratio * 100).toFixed(0) + '%';
}

// ── Compare ─────────────────────────────────────────────────────────

async function runCompare() {
  const surface = document.getElementById('compare-surface').value.trim();
  const insight = document.getElementById('compare-insight').value.trim();
  if (!surface || !insight) return;

  const btn = document.getElementById('compare-btn');
  const loading = document.getElementById('compare-loading');
  const result = document.getElementById('compare-result');

  btn.disabled = true;
  loading.classList.add('active');
  result.classList.remove('active');

  try {
    // Score both through fathom_score endpoint
    const [respS, respI] = await Promise.all([
      apiCall('/v2/fathom_score', { prompt: surface }),
      apiCall('/v2/fathom_score', { prompt: insight }),
    ]);

    const dS = respS.data;
    const dI = respI.data;

    // Surface card
    const gS = document.getElementById('cmp-grade-surface');
    gS.textContent = dS.grade;
    gS.style.color = gradeColor(dS.grade);
    document.getElementById('cmp-score-surface').textContent =
      Math.round(dS.fathom_score) + '/100';

    // Insight card
    const gI = document.getElementById('cmp-grade-insight');
    gI.textContent = dI.grade;
    gI.style.color = gradeColor(dI.grade);
    document.getElementById('cmp-score-insight').textContent =
      Math.round(dI.fathom_score) + '/100';

    // Delta
    const delta = dI.fathom_score - dS.fathom_score;
    const deltaEl = document.getElementById('cmp-delta');
    deltaEl.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(1);
    deltaEl.className = 'delta-value ' + (delta >= 0 ? 'delta-positive' : 'delta-negative');

    const winner = delta > 0 ? 'Insight' : delta < 0 ? 'Surface' : 'Equal';
    document.getElementById('cmp-delta-text').textContent =
      delta !== 0
        ? `${winner} prompt scores ${Math.abs(delta).toFixed(1)} points ${delta > 0 ? 'higher' : 'lower'}`
        : 'Both prompts score equally';

    // Compute time
    const totalMs = respS.meta.compute_time_ms + respI.meta.compute_time_ms;
    document.getElementById('compare-compute-time').textContent =
      `${totalMs.toFixed(0)}ms total | ${respS.meta.model}`;

    result.classList.add('active');
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    loading.classList.remove('active');
  }
}

// ── Keyboard shortcut: Ctrl+Enter to analyze ────────────────────────

document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    const scoreTab = document.getElementById('tab-score');
    if (scoreTab.style.display !== 'none') {
      runScore();
    } else {
      runCompare();
    }
  }
});
