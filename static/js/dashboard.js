/**
 * CITS — dashboard.js
 * Frontend logic for admin.html
 * Fetches model metrics, drift, handles retrain with background polling.
 * All values are dynamic — no hardcoded statistics.
 */

document.addEventListener('DOMContentLoaded', () => {
    loadModelMetrics();
    loadDrift();
    setupRetrain();
    setupSeed();
});


// ============================================================
// MODEL METRICS
// ============================================================

async function loadModelMetrics() {
    try {
        const res = await fetch('/api/model-metrics');
        if (!res.ok) throw new Error('Failed to load metrics');
        const data = await res.json();

        // Metric cards
        document.getElementById('model-version').textContent = data.model_version;
        document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
        document.getElementById('f1-score').textContent = (data.f1_score * 100).toFixed(1) + '%';
        document.getElementById('last-retrain').textContent = data.last_retrain;

        // Charts
        renderBarChart('sentiment-chart', data.sentiment_distribution, ['1★', '2★', '3★', '4★', '5★']);
        renderChartLabels('sentiment-chart-labels', ['1★', '2★', '3★', '4★', '5★']);

        const fakeBinLabels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'];
        renderBarChart('fake-prob-chart', data.fake_probability, fakeBinLabels);
        renderChartLabels('fake-prob-chart-labels', fakeBinLabels);

        // Comparison table
        renderComparisonTable(data.comparison);

        // Processing summary — uses real data from backend, no hardcoding
        renderProcessingSummary(data.total_reviews, data.dataset_cleaned, data.dataset_rejected);
    } catch (err) {
        document.getElementById('model-version').textContent = 'Error';
        console.error(err);
    }
}


// ============================================================
// DRIFT STATUS
// ============================================================

async function loadDrift() {
    try {
        const res = await fetch('/api/drift-status');
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();

        // Drift badge — maps "Stable" / "Moderate Drift" / "High Drift"
        const driftEl = document.getElementById('drift-status');
        const level = data.drift_level;

        let color = 'green';
        let label = level;
        if (level === 'High Drift') {
            color = 'red';
        } else if (level === 'Moderate Drift') {
            color = 'yellow';
        }

        driftEl.innerHTML = `
      <span class="badge badge-${color}">
        <span class="status-dot status-dot-${color}"></span>
        ${label}
      </span>
    `;

        // Dataset health ring
        updateHealthRing(data.dataset_health);
    } catch (err) {
        console.error('Drift load failed:', err);
    }
}

function updateHealthRing(pct) {
    const circumference = 2 * Math.PI * 58; // r=58 from SVG
    const offset = circumference - (pct / 100) * circumference;

    const ring = document.querySelector('.health-ring-fill');
    const numberEl = document.querySelector('.health-ring-number');
    const labelEl = document.querySelector('.health-ring-label');

    if (ring) ring.style.strokeDashoffset = offset;
    if (numberEl) numberEl.textContent = Math.round(pct) + '%';
    if (labelEl) labelEl.textContent = pct >= 80 ? 'Healthy' : pct >= 50 ? 'Fair' : 'Poor';
}


// ============================================================
// BAR CHARTS
// ============================================================

function renderBarChart(containerId, values, labels) {
    const container = document.getElementById(containerId);
    if (!container || !values || values.length === 0) {
        if (container) container.innerHTML = '<p class="text-sm text-muted">No data available.</p>';
        return;
    }

    container.innerHTML = '';
    const maxVal = Math.max(...values) || 1;

    values.forEach((val, i) => {
        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        const height = Math.max((val / maxVal) * 100, 2);
        bar.style.height = height + '%';
        bar.title = labels ? `${labels[i]}: ${(val * 100).toFixed(0)}%` : `Bin ${i + 1}: ${(val * 100).toFixed(0)}%`;
        container.appendChild(bar);
    });
}

function renderChartLabels(containerId, labels) {
    const container = document.getElementById(containerId);
    if (!container || !labels) return;
    container.innerHTML = '';
    labels.forEach(label => {
        const span = document.createElement('span');
        span.textContent = label;
        container.appendChild(span);
    });
}


// ============================================================
// COMPARISON TABLE
// ============================================================

function renderComparisonTable(comparison) {
    const tbody = document.getElementById('comparison-tbody');
    if (!tbody) return;

    if (!comparison || comparison.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading-text" style="padding: 24px 32px;">No comparison data.</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    comparison.forEach(row => {
        const tr = document.createElement('tr');
        const deltaClass = row.delta.includes('▲') ? 'delta-positive' :
            row.delta.includes('▼') ? 'delta-negative' : '';
        tr.innerHTML = `
      <td class="font-medium">${row.metric}</td>
      <td class="mono">${row.current}</td>
      <td class="mono">${row.candidate}</td>
      <td class="text-right"><span class="${deltaClass}">${row.delta}</span></td>
    `;
        tbody.appendChild(tr);
    });
}


// ============================================================
// PROCESSING SUMMARY — dynamic, no hardcoded values
// ============================================================

function renderProcessingSummary(total, cleaned, rejected) {
    const container = document.getElementById('processing-summary');
    if (!container) return;

    // Use backend-provided values; fallback to computing from total only if both are 0
    const displayCleaned = (cleaned > 0 || rejected > 0) ? cleaned : total;
    const displayRejected = (cleaned > 0 || rejected > 0) ? rejected : 0;

    container.innerHTML = `
    <div class="stat-row">
      <span class="stat-label">Total</span>
      <span class="stat-value">${total.toLocaleString()}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Cleaned</span>
      <span class="stat-value">${displayCleaned.toLocaleString()}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Rejected</span>
      <span class="stat-value">${displayRejected.toLocaleString()}</span>
    </div>
  `;
}


// ============================================================
// RETRAIN — background task with polling
// ============================================================

function setupRetrain() {
    const btn = document.getElementById('retrain-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        if (btn.disabled) return;

        const confirmed = confirm('Retrain the ML model on all current review data?');
        if (!confirmed) return;

        btn.disabled = true;
        btn.textContent = 'Retraining…';

        try {
            const res = await fetch('/api/retrain', { method: 'POST' });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Retrain failed');
            }

            const data = await res.json();

            if (data.status === 'running') {
                alert('⏳ A retraining is already in progress. Please wait.');
                btn.disabled = false;
                btn.textContent = 'Retrain Model';
                return;
            }

            // Poll for completion
            pollRetrainStatus(btn);
        } catch (err) {
            alert('❌ Retrain failed: ' + err.message);
            console.error(err);
            btn.disabled = false;
            btn.textContent = 'Retrain Model';
        }
    });
}


async function pollRetrainStatus(btn) {
    const maxAttempts = 60;  // 60 * 2s = 2 minutes max
    let attempts = 0;

    const interval = setInterval(async () => {
        attempts++;
        try {
            const res = await fetch('/api/retrain-status');
            const data = await res.json();

            if (!data.running && data.last_result) {
                clearInterval(interval);
                const result = data.last_result;

                if (result.status === 'success') {
                    alert(`✅ ${result.message}\nAccuracy: ${(result.accuracy * 100).toFixed(1)}%\nF1: ${(result.f1_score * 100).toFixed(1)}%`);
                } else if (result.status === 'rejected') {
                    alert(`⚠️ ${result.message}`);
                } else if (result.status === 'skipped') {
                    alert(`ℹ️ ${result.message}`);
                } else {
                    alert(`❌ ${result.message}`);
                }

                // Refresh all dashboard data
                loadModelMetrics();
                loadDrift();

                btn.disabled = false;
                btn.textContent = 'Retrain Model';
            }

            if (attempts >= maxAttempts) {
                clearInterval(interval);
                alert('⏱️ Retraining is taking longer than expected. Check back later.');
                btn.disabled = false;
                btn.textContent = 'Retrain Model';
            }
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 2000);
}


// ============================================================
// DATABASE SEEDING
// ============================================================

function setupSeed() {
    const btn = document.getElementById('seed-btn');
    const msg = document.getElementById('seed-status-msg');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        const confirmed = confirm('This will populate the database with initial products and reviews from the CSV. It takes a few minutes. Continue?');
        if (!confirmed) return;

        btn.disabled = true;
        btn.textContent = 'Seeding...';
        msg.style.display = 'block';
        msg.textContent = '📦 Seeding started in background...';

        try {
            const res = await fetch('/api/seed', { method: 'POST' });
            if (!res.ok) throw new Error('Seeding request failed');

            // Logic: we don't need highly complex polling here, just alert.
            // Seeding 4k reviews might take 1-2 mins on free tier.
            alert('✅ Seeding has been triggered in the background. Please wait 1-2 minutes and refresh the dashboard.');

            // Re-enable after a bit to prevent double clicks
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = 'Seed Database';
            }, 5000);
        } catch (err) {
            alert('❌ Seed failed: ' + err.message);
            btn.disabled = false;
            btn.textContent = 'Seed Database';
        }
    });
}
