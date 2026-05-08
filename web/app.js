/**
 * Reroute web app.
 *
 * Two operating modes:
 *   1. Static mode — loads scenarios_for_demo.json from same directory.
 *      Slider re-solving is disabled; all numbers come from pre-computed JSON.
 *   2. Live mode — detects API at /api/health and uses POST /api/solve and
 *      POST /api/generate for live solving with custom coefficients.
 *
 * The app probes for the API on load and falls back gracefully to static mode.
 */

(function () {
  'use strict';

  // ============================================================
  // STATE
  // ============================================================

  const state = {
    apiBase: null,          // null = static mode, else origin
    apiAvailable: false,
    scenarios: [],          // loaded scenarios
    currentIdx: 0,
    currentMode: 'lp',      // 'manual' | 'lp' | 'side'
    coefficients: {
      alpha_yield: 1.0,
      beta_spill: 0.85,
      delta_harm: 1.5,
      lambda_miss: 1.0,
      miss_fixed_cost_usd: 250.0,
    },
    solving: false,
  };

  // ============================================================
  // UTILITIES
  // ============================================================

  const $ = (id) => document.getElementById(id);

  function fmtMoney(v) {
    if (v === null || v === undefined) return '—';
    return '$' + Math.round(v).toLocaleString();
  }

  function debounce(fn, wait) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  // ============================================================
  // API DETECTION
  //
  // Order of precedence:
  //   1. window.REROUTE_API_URL — set in index.html for production deploys
  //      (e.g., set to your Render URL after deploying)
  //   2. Same-origin /api — for when the API serves the static files itself
  //   3. http://127.0.0.1:8000 — for local dev with `reroute serve`
  // ============================================================

  async function probeApi() {
    const candidates = [];
    if (typeof window !== 'undefined' && window.REROUTE_API_URL) {
      candidates.push(window.REROUTE_API_URL.replace(/\/$/, ''));
    }
    candidates.push('');                              // same origin
    candidates.push('http://127.0.0.1:8000');         // local dev
    candidates.push('http://localhost:8000');

    for (const base of candidates) {
      try {
        const res = await fetch(`${base}/api/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(3000),  // longer timeout for cold-starting hosted APIs
        });
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'ok') {
            state.apiBase = base;
            state.apiAvailable = true;
            return true;
          }
        }
      } catch (e) { /* try next */ }
    }
    return false;
  }

  function updateApiBadge() {
    const badge = $('api-status');
    const text = $('api-status-text');
    if (state.apiAvailable) {
      badge.classList.add('live');
      badge.classList.remove('offline');
      text.textContent = 'Live API connected';
    } else {
      badge.classList.add('offline');
      badge.classList.remove('live');
      text.textContent = 'Static mode';
    }
  }

  // ============================================================
  // DATA LOADING
  // ============================================================

  async function loadScenarios() {
    if (state.apiAvailable) {
      try {
        const list = await fetch(`${state.apiBase}/api/scenarios`).then(r => r.json());
        // Hydrate full detail for each
        const full = await Promise.all(list.map(async (item) => {
          const detail = await fetch(`${state.apiBase}/api/scenarios/${item.scenario_id}`).then(r => r.json());
          return detail;
        }));
        state.scenarios = full;
        return;
      } catch (e) {
        console.warn('Live API failed, falling back to static:', e);
      }
    }
    // Static fallback
    try {
      const res = await fetch('scenarios_for_demo.json');
      if (res.ok) state.scenarios = await res.json();
    } catch (e) {
      console.error('Failed to load static scenarios:', e);
      state.scenarios = [];
    }
  }

  // ============================================================
  // SCENARIO PICKER
  // ============================================================

  function populateSelect() {
    const sel = $('scenario-select');
    sel.innerHTML = '';
    state.scenarios.forEach((s, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `${s.scenario_id} · ${s.n_passengers} pax · ${s.total_open_seats} seats · ${s.inbound.delay_min}min delay`;
      sel.appendChild(opt);
    });
  }

  function renderCurrent() {
    if (!state.scenarios.length) {
      showEmpty();
      return;
    }
    const idx = ((state.currentIdx % state.scenarios.length) + state.scenarios.length) % state.scenarios.length;
    state.currentIdx = idx;
    $('scenario-select').value = idx;
    render(state.scenarios[idx]);
  }

  function showEmpty() {
    $('pax-list').innerHTML = '<div class="empty-state">No scenario data loaded.</div>';
    $('flights-list').innerHTML = '';
    $('stats').innerHTML = '';
    $('scenario-summary').innerHTML = '<em>Run <code>reroute export-demo</code> to generate scenarios, or start the API server.</em>';
  }

  // ============================================================
  // RENDERING
  // ============================================================

  function statusForAssignment(a) {
    if (a.flight === 'MISCONNECT') return { cls: 'misconnect', label: '✗ Misconnect' };
    return { cls: 'assigned-good', label: `✓ ${a.flight} ${a.cabin}` };
  }

  function paxRowHtml(p, mode) {
    const a = mode === 'manual' ? p.manual : p.lp;
    const status = statusForAssignment(a);
    const probPct = Math.round(p.misconnect_prob * 100);
    const ssrTag = p.has_ssr ? '<span style="color:var(--amber);margin-left:4px" data-tip="Special service request">⚠</span>' : '';
    const umTag = p.is_um ? '<span style="color:var(--red);margin-left:4px" data-tip="Unaccompanied minor">UM</span>' : '';
    return `
      <div class="pax ${status.cls}">
        <span class="tier-badge tier-${p.tier}">${p.tier}</span>
        <div>
          <div class="pax-name">${p.name}${ssrTag}${umTag}</div>
          <div class="pax-meta">${p.cabin} · ${p.buffer_min}min buffer · ${probPct}% miss prob</div>
        </div>
        <div class="pax-yield">$${p.yield_usd.toLocaleString()}</div>
        <div class="pax-status">${status.label}</div>
      </div>
    `;
  }

  function renderFlights(flights) {
    return flights.map(f => `
      <div class="flight">
        <div class="flight-head">
          <span class="flight-num">${f.flight}</span>
          <span class="flight-route">→ ${f.destination} · +${f.minutes_after_arrival}min</span>
        </div>
        <div class="flight-cabins">
          <span>F: <strong>${f.open_F}</strong></span>
          <span>Y+: <strong>${f.open_Yplus}</strong></span>
          <span>Y: <strong>${f.open_Y}</strong></span>
          <span style="margin-left:auto;color:var(--text-2)">total: <strong>${f.open_total}</strong></span>
        </div>
      </div>
    `).join('');
  }

  function renderStats(scenario) {
    const r = scenario.results;
    const html = state.currentMode === 'side' ? `
      <div class="stat bad">
        <div class="stat-label">Manual loss</div>
        <div class="stat-value">${fmtMoney(r.manual.total_loss)}</div>
        <div class="stat-sub">${r.manual.n_misconnects} misconnects</div>
      </div>
      <div class="stat good">
        <div class="stat-label">Reroute loss</div>
        <div class="stat-value">${fmtMoney(r.lp.total_loss)}</div>
        <div class="stat-sub">${r.lp.n_misconnects} misconnects</div>
      </div>
      <div class="stat delta">
        <div class="stat-label">Saved</div>
        <div class="stat-value">${fmtMoney(r.delta_dollars)}</div>
        <div class="stat-sub">${r.delta_pct}% reduction</div>
      </div>
      <div class="stat neutral">
        <div class="stat-label">Solve time</div>
        <div class="stat-value">${r.lp.solve_ms.toFixed(1)} ms</div>
        <div class="stat-sub">LP cohort solve</div>
      </div>
    ` : state.currentMode === 'manual' ? `
      <div class="stat bad">
        <div class="stat-label">Expected loss</div>
        <div class="stat-value">${fmtMoney(r.manual.total_loss)}</div>
        <div class="stat-sub">manual triage</div>
      </div>
      <div class="stat bad">
        <div class="stat-label">Misconnects</div>
        <div class="stat-value">${r.manual.n_misconnects}</div>
        <div class="stat-sub">of ${scenario.n_passengers} pax</div>
      </div>
      <div class="stat neutral">
        <div class="stat-label">Spill cost</div>
        <div class="stat-value">${fmtMoney(r.manual.breakdown.spill || 0)}</div>
        <div class="stat-sub">cabin downgrades</div>
      </div>
      <div class="stat neutral">
        <div class="stat-label">Yield dilution</div>
        <div class="stat-value">${fmtMoney(r.manual.breakdown.yield_dilution || 0)}</div>
        <div class="stat-sub">across cohort</div>
      </div>
    ` : `
      <div class="stat good">
        <div class="stat-label">Expected loss</div>
        <div class="stat-value">${fmtMoney(r.lp.total_loss)}</div>
        <div class="stat-sub">Reroute LP</div>
      </div>
      <div class="stat ${r.lp.n_misconnects <= r.manual.n_misconnects ? 'good' : 'neutral'}">
        <div class="stat-label">Misconnects</div>
        <div class="stat-value">${r.lp.n_misconnects}</div>
        <div class="stat-sub">of ${scenario.n_passengers} pax</div>
      </div>
      <div class="stat good">
        <div class="stat-label">vs baseline</div>
        <div class="stat-value">−${r.delta_pct}%</div>
        <div class="stat-sub">${fmtMoney(r.delta_dollars)} saved</div>
      </div>
      <div class="stat neutral">
        <div class="stat-label">Solve time</div>
        <div class="stat-value">${r.lp.solve_ms.toFixed(1)} ms</div>
        <div class="stat-sub">LP cohort solve</div>
      </div>
    `;
    $('stats').innerHTML = html;
  }

  function render(scenario) {
    // Summary
    const demand_excess = Math.max(0, scenario.n_passengers - scenario.total_open_seats);
    $('scenario-summary').innerHTML = `
      Inbound <strong>${scenario.inbound.flight}</strong> from <strong>${scenario.inbound.origin}</strong>
      arrived <strong>${scenario.inbound.delay_min} minutes late</strong>.
      <strong>${scenario.n_passengers} connecting passengers</strong> need recovery across
      <strong>${scenario.recovery_flights.length} downstream flights</strong>
      with <strong>${scenario.total_open_seats} open seats</strong> total.
      ${demand_excess > 0 ? `Demand exceeds supply by <strong style="color:var(--red)">${demand_excess} seats</strong>.` : 'Supply exceeds demand — the optimization plays out in cabin assignments.'}
      <div class="scenario-meta-pills">
        <span class="pill">Supply/demand: <strong>${scenario.supply_demand_ratio.toFixed(2)}</strong></span>
        <span class="pill">Tier mix: <strong>${countTiers(scenario.passengers)}</strong></span>
        <span class="pill">SSR/UM: <strong>${countSpecial(scenario.passengers)}</strong></span>
      </div>
    `;

    renderStats(scenario);

    if (state.currentMode === 'side') {
      $('single-view').style.display = 'none';
      $('side-view').style.display = 'block';
      $('side-manual-meta').textContent = `${scenario.results.manual.n_misconnects} misconnects · ${fmtMoney(scenario.results.manual.total_loss)}`;
      $('side-lp-meta').textContent = `${scenario.results.lp.n_misconnects} misconnects · ${fmtMoney(scenario.results.lp.total_loss)}`;
      $('side-manual-list').innerHTML = scenario.passengers.map(p => paxRowHtml(p, 'manual')).join('');
      $('side-lp-list').innerHTML = scenario.passengers.map(p => paxRowHtml(p, 'lp')).join('');
    } else {
      $('single-view').style.display = 'block';
      $('side-view').style.display = 'none';
      const r = state.currentMode === 'manual' ? scenario.results.manual : scenario.results.lp;
      $('pax-panel-title').textContent = state.currentMode === 'manual'
        ? 'Baseline · manual triage assignments'
        : 'Reroute · LP allocation';
      $('pax-panel-meta').textContent = `${r.n_misconnects} misconnects · ${fmtMoney(r.total_loss)} loss`;
      $('flights-meta').textContent = `${scenario.total_open_seats} open seats`;
      $('pax-list').innerHTML = scenario.passengers.map(p => paxRowHtml(p, state.currentMode)).join('');
      $('flights-list').innerHTML = renderFlights(scenario.recovery_flights);
    }
  }

  function countTiers(pax) {
    const c = { EXP: 0, PLT: 0, GLD: 0, REG: 0 };
    pax.forEach(p => c[p.tier]++);
    return Object.entries(c).filter(([_, v]) => v > 0).map(([k, v]) => `${k}:${v}`).join(' ');
  }

  function countSpecial(pax) {
    const ssr = pax.filter(p => p.has_ssr).length;
    const um = pax.filter(p => p.is_um).length;
    if (!ssr && !um) return 'none';
    const parts = [];
    if (ssr) parts.push(`${ssr} SSR`);
    if (um) parts.push(`${um} UM`);
    return parts.join(', ');
  }

  // ============================================================
  // LIVE SOLVING (API)
  // ============================================================

  async function resolveCurrent() {
    if (!state.apiAvailable) {
      showToast('Live re-solving requires the API server. Start it with: reroute serve');
      return;
    }
    if (!state.scenarios.length) return;
    const scenario = state.scenarios[state.currentIdx];
    if (!scenario.scenario_full) {
      // Old static scenarios won't have the full body — skip
      showToast('This scenario was loaded in static mode and cannot be re-solved.');
      return;
    }
    state.solving = true;
    $('btn-resolve').disabled = true;
    $('btn-resolve').innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="20" stroke-dashoffset="0"><animateTransform attributeName="transform" type="rotate" from="0 8 8" to="360 8 8" dur="1s" repeatCount="indefinite"/></circle></svg> Solving...';
    try {
      const res = await fetch(`${state.apiBase}/api/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: scenario.scenario_full,
          coefficients: state.coefficients,
        }),
        signal: AbortSignal.timeout(60000),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const updated = await res.json();
      // Splice in the updated result
      state.scenarios[state.currentIdx] = updated;
      render(updated);
      showToast(`Solved in ${updated.results.lp.solve_ms.toFixed(1)} ms with new weights`);
    } catch (e) {
      console.error(e);
      showToast('Re-solve failed: ' + e.message);
    } finally {
      state.solving = false;
      $('btn-resolve').disabled = false;
      $('btn-resolve').innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 8C2 4.7 4.7 2 8 2C10.4 2 12.5 3.4 13.4 5.4M14 8C14 11.3 11.3 14 8 14C5.6 14 3.5 12.6 2.6 10.6M14 2V5.4H10.6M2 14V10.6H5.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Re-solve with these weights';
    }
  }

  async function generateNew() {
    if (!state.apiAvailable) {
      showToast('Generating new scenarios requires the API server. Start it with: reroute serve');
      return;
    }
    state.solving = true;
    $('btn-generate').disabled = true;
    showToast('Generating new scenario… (first request after idle may take 20–40s while the free-tier API wakes up)');
    try {
      const res = await fetch(`${state.apiBase}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_passengers: 15 + Math.floor(Math.random() * 25),
          n_recovery_flights: 3 + Math.floor(Math.random() * 3),
          delay_min: 80 + Math.floor(Math.random() * 120),
        }),
        signal: AbortSignal.timeout(60000),  // 60s for cold start
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const newScn = await res.json();
      state.scenarios.unshift(newScn);
      state.currentIdx = 0;
      populateSelect();
      renderCurrent();
      showToast(`Generated ${newScn.scenario_id} — solved in ${newScn.results.lp.solve_ms.toFixed(1)} ms`);
    } catch (e) {
      showToast('Generate failed: ' + e.message);
    } finally {
      state.solving = false;
      $('btn-generate').disabled = false;
    }
  }

  // ============================================================
  // TOAST
  // ============================================================

  let toastTimer;
  function showToast(msg) {
    const root = $('onboarding-root');
    root.innerHTML = `
      <div class="onboarding" style="background:var(--bg-3);">
        <div class="onboarding-text" style="margin-bottom:0;color:var(--text-1)">${msg}</div>
      </div>
    `;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      if (root.firstChild) root.innerHTML = '';
    }, 3500);
  }

  // ============================================================
  // ONBOARDING
  // ============================================================

  function maybeShowOnboarding() {
    if (sessionStorage.getItem('reroute-onboarded')) return;
    const root = $('onboarding-root');
    root.innerHTML = `
      <div class="onboarding">
        <button class="onboarding-close" onclick="document.getElementById('onboarding-root').innerHTML='';sessionStorage.setItem('reroute-onboarded','1');">×</button>
        <div class="onboarding-step">Quick tour · 1 of 1</div>
        <div class="onboarding-title">Welcome to Reroute</div>
        <div class="onboarding-text">
          Use <strong>Compare</strong> to see both strategies side-by-side. Open <strong>Tune cost coefficients</strong> below to change the optimizer's trade-offs.
          ${state.apiAvailable ? '<br><br>Live API connected — you can re-solve with custom weights.' : ''}
        </div>
        <div class="onboarding-buttons">
          <button class="onboarding-btn primary" onclick="document.querySelector('[data-view=side]').click();document.getElementById('onboarding-root').innerHTML='';sessionStorage.setItem('reroute-onboarded','1');">Show side-by-side</button>
          <button class="onboarding-btn secondary" onclick="document.getElementById('onboarding-root').innerHTML='';sessionStorage.setItem('reroute-onboarded','1');">Got it</button>
        </div>
      </div>
    `;
  }

  // ============================================================
  // HERO ANIMATION
  // ============================================================

  function buildHeroGrid(elId, n) {
    const grid = $(elId);
    grid.innerHTML = '';
    for (let i = 0; i < n; i++) {
      const cell = document.createElement('div');
      cell.className = 'demo-pax-cell';
      cell.textContent = i + 1;
      grid.appendChild(cell);
    }
  }

  async function runHeroAnimation() {
    buildHeroGrid('hero-manual-grid', 18);
    buildHeroGrid('hero-lp-grid', 18);
    let cycleCount = 0;
    while (true) {
      // Wait for scenarios to load
      if (!state.scenarios.length) {
        await new Promise(r => setTimeout(r, 500));
        continue;
      }
      cycleCount++;
      const scn = state.scenarios[cycleCount % state.scenarios.length];
      $('hero-status').textContent = `Scenario ${scn.scenario_id}`;

      // Reset
      const manualCells = $('hero-manual-grid').children;
      const lpCells = $('hero-lp-grid').children;
      [...manualCells].forEach(c => { c.className = 'demo-pax-cell processing'; });
      [...lpCells].forEach(c => { c.className = 'demo-pax-cell processing'; });
      $('hero-manual-loss').textContent = '$0';
      $('hero-manual-miss').textContent = '0';
      $('hero-lp-loss').textContent = '$0';
      $('hero-lp-time').textContent = '—';
      $('hero-delta').textContent = '$0';

      await sleep(700);

      // Manual side: serial fill
      const manualPax = scn.passengers.slice(0, 18);
      let manualLoss = 0, manualMiss = 0;
      for (let i = 0; i < manualPax.length; i++) {
        const cell = manualCells[i];
        if (!cell) break;
        const a = manualPax[i].manual;
        if (a.flight === 'MISCONNECT') {
          cell.className = 'demo-pax-cell misconnect';
          manualMiss++;
        } else {
          cell.className = 'demo-pax-cell assigned';
        }
        manualLoss += a.cost;
        $('hero-manual-loss').textContent = fmtMoney(manualLoss);
        $('hero-manual-miss').textContent = String(manualMiss);
        await sleep(60);
      }

      await sleep(400);

      // LP side: parallel fill
      const lpPax = scn.passengers.slice(0, 18);
      [...lpCells].forEach(c => { c.className = 'demo-pax-cell processing'; });
      await sleep(300);
      let lpLoss = 0;
      for (let i = 0; i < lpPax.length; i++) {
        const cell = lpCells[i];
        if (!cell) break;
        const a = lpPax[i].lp;
        if (a.flight === 'MISCONNECT') cell.className = 'demo-pax-cell misconnect';
        else cell.className = 'demo-pax-cell assigned';
        lpLoss += a.cost;
      }
      // Update simultaneously for the "instant solve" effect
      $('hero-lp-loss').textContent = fmtMoney(lpLoss);
      $('hero-lp-time').textContent = scn.results.lp.solve_ms.toFixed(1) + ' ms';
      const delta = scn.results.delta_dollars;
      $('hero-delta').textContent = fmtMoney(delta);
      $('hero-delta-sub').textContent = `${scn.scenario_id} · ${scn.results.delta_pct.toFixed(1)}% reduction · cycling next in 4s`;

      await sleep(4000);
    }
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ============================================================
  // SLIDERS
  // ============================================================

  function bindSliders() {
    const handlers = [
      { id: 'slide-alpha', valId: 'val-alpha', key: 'alpha_yield' },
      { id: 'slide-beta', valId: 'val-beta', key: 'beta_spill' },
      { id: 'slide-delta', valId: 'val-delta', key: 'delta_harm' },
      { id: 'slide-lambda', valId: 'val-lambda', key: 'lambda_miss' },
    ];
    handlers.forEach(({ id, valId, key }) => {
      const slider = $(id);
      slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        state.coefficients[key] = v;
        $(valId).textContent = v.toFixed(2);
      });
    });
    $('btn-resolve').addEventListener('click', resolveCurrent);
    $('btn-reset-coefs').addEventListener('click', () => {
      state.coefficients = {
        alpha_yield: 1.0, beta_spill: 0.85, delta_harm: 1.5,
        lambda_miss: 1.0, miss_fixed_cost_usd: 250.0,
      };
      $('slide-alpha').value = 1.0; $('val-alpha').textContent = '1.00';
      $('slide-beta').value = 0.85; $('val-beta').textContent = '0.85';
      $('slide-delta').value = 1.5; $('val-delta').textContent = '1.50';
      $('slide-lambda').value = 1.0; $('val-lambda').textContent = '1.00';
    });
  }

  // ============================================================
  // EVENT BINDING
  // ============================================================

  function bindEvents() {
    $('scenario-select').addEventListener('change', (e) => {
      state.currentIdx = parseInt(e.target.value);
      renderCurrent();
    });
    $('btn-prev').addEventListener('click', () => {
      state.currentIdx -= 1;
      renderCurrent();
    });
    $('btn-next').addEventListener('click', () => {
      state.currentIdx += 1;
      renderCurrent();
    });
    $('btn-generate').addEventListener('click', generateNew);

    document.querySelectorAll('.view-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.currentMode = tab.dataset.view;
        renderCurrent();
      });
    });

    bindSliders();

    // Update API help text
    if (!state.apiAvailable) {
      const help = $('advanced-help');
      help.innerHTML = `
        <strong>How this works:</strong> Slide weights to see how each component shifts the LP's trade-offs.
        <strong style="color:var(--amber)">Live re-solving requires the API server</strong> — start it with
        <code style="background:var(--bg-1);padding:1px 5px;border-radius:3px;">reroute serve</code>
        to enable click-to-resolve.
      `;
    }
  }

  // ============================================================
  // BOOT
  // ============================================================

  async function boot() {
    await probeApi();
    updateApiBadge();
    await loadScenarios();
    populateSelect();
    bindEvents();
    renderCurrent();
    runHeroAnimation();
    setTimeout(maybeShowOnboarding, 2500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
