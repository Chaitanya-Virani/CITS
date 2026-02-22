/**
 * CITS — reviews.js
 * Admin reviews page — product-wise sentiment breakdown
 * with expandable negative reviews.
 */

let allProducts = [];
let currentSort = 'name';

document.addEventListener('DOMContentLoaded', () => {
    loadAdminReviews();
    setupSearch();
    setupSort();
});


// ============================================================
// LOAD DATA
// ============================================================

async function loadAdminReviews() {
    try {
        const res = await fetch('/api/admin/reviews');
        if (!res.ok) throw new Error('Failed to load');
        allProducts = await res.json();

        // Summary stats
        const totalProducts = allProducts.length;
        const totalReviews = allProducts.reduce((s, p) => s + p.total, 0);
        const totalPositive = allProducts.reduce((s, p) => s + p.positive_count, 0);
        const totalNegative = allProducts.reduce((s, p) => s + p.negative_count, 0);

        document.getElementById('stat-products').textContent = totalProducts.toLocaleString();
        document.getElementById('stat-reviews').textContent = totalReviews.toLocaleString();
        document.getElementById('stat-positive').textContent = totalPositive.toLocaleString();
        document.getElementById('stat-negative').textContent = totalNegative.toLocaleString();

        applyFilters();
    } catch (err) {
        document.getElementById('product-review-list').innerHTML =
            '<div class="card p-8"><p class="error-text">Failed to load reviews.</p></div>';
        console.error(err);
    }
}


// ============================================================
// SEARCH & SORT
// ============================================================

function setupSearch() {
    const input = document.getElementById('product-search');
    if (!input) return;
    input.addEventListener('input', () => applyFilters());
}

function setupSort() {
    const select = document.getElementById('sort-select');
    if (!select) return;
    select.addEventListener('change', (e) => {
        currentSort = e.target.value;
        applyFilters();
    });
}

function applyFilters() {
    const query = (document.getElementById('product-search')?.value || '').toLowerCase().trim();
    let filtered = allProducts;

    if (query) {
        filtered = filtered.filter(p =>
            p.product_name.toLowerCase().includes(query) ||
            p.asin.toLowerCase().includes(query)
        );
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
        switch (currentSort) {
            case 'positive-desc': return b.positive_count - a.positive_count;
            case 'positive-asc': return a.positive_count - b.positive_count;
            case 'negative-desc': return b.negative_count - a.negative_count;
            case 'negative-asc': return a.negative_count - b.negative_count;
            case 'total-desc': return b.total - a.total;
            default: return a.product_name.localeCompare(b.product_name);
        }
    });

    renderProducts(filtered);
}


// ============================================================
// RENDER PRODUCT LIST
// ============================================================

function renderProducts(products) {
    const container = document.getElementById('product-review-list');

    if (products.length === 0) {
        container.innerHTML = '<div class="card p-8"><p class="text-sm text-muted">No products found.</p></div>';
        return;
    }

    container.innerHTML = '';
    products.forEach(product => {
        container.appendChild(createProductCard(product));
    });
}

function createProductCard(product) {
    const card = document.createElement('div');
    card.className = 'card mb-4';
    card.style.overflow = 'hidden';

    const posPct = product.total > 0 ? (product.positive_count / product.total * 100).toFixed(0) : 0;
    const negPct = product.total > 0 ? (product.negative_count / product.total * 100).toFixed(0) : 0;
    const hasNegative = product.negative_count > 0;

    card.innerHTML = `
        <div class="product-review-header" style="padding: 20px 24px; cursor: ${hasNegative ? 'pointer' : 'default'}; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;"
             ${hasNegative ? `onclick="toggleNegativeReviews(this)"` : ''}>
            <div style="flex: 1; min-width: 200px;">
                <h3 style="font-weight: 600; font-size: 15px; margin-bottom: 4px;">${escapeHTML(product.product_name)}</h3>
                <p style="font-size: 12px; color: var(--color-muted);">${product.asin} · ${product.total} reviews</p>
            </div>
            <div style="display: flex; gap: 16px; align-items: center;">
                <div style="text-align: center;">
                    <span style="display: block; font-size: 20px; font-weight: 700; color: var(--color-green);">${product.positive_count}</span>
                    <span style="font-size: 11px; color: var(--color-muted);">Positive (${posPct}%)</span>
                </div>
                <div style="text-align: center;">
                    <span style="display: block; font-size: 20px; font-weight: 700; color: var(--color-red);">${product.negative_count}</span>
                    <span style="font-size: 11px; color: var(--color-muted);">Negative (${negPct}%)</span>
                </div>
                <!-- Sentiment bar -->
                <div style="width: 120px; height: 8px; background: var(--color-red); border-radius: 4px; overflow: hidden;">
                    <div style="width: ${posPct}%; height: 100%; background: var(--color-green); border-radius: 4px 0 0 4px;"></div>
                </div>
                ${hasNegative ? '<span class="expand-icon material-symbols-outlined" style="font-size: 20px; color: var(--color-muted); transition: transform 0.2s;">expand_more</span>' : ''}
            </div>
        </div>
        ${hasNegative ? `<div class="negative-reviews-panel" style="display: none; border-top: 1px solid var(--color-border); padding: 0 24px; max-height: 400px; overflow-y: auto;">
            ${product.negative_reviews.map(r => `
                <div style="padding: 16px 0; border-bottom: 1px solid var(--color-border);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div>
                            <span style="font-weight: 600; font-size: 13px;">${escapeHTML(r.author)}</span>
                            <span style="margin-left: 8px; color: #f59e0b;">${'★'.repeat(r.rating)}${'☆'.repeat(5 - r.rating)}</span>
                        </div>
                        <span style="font-size: 12px; color: var(--color-muted);">${r.date || ''}</span>
                    </div>
                    <p style="font-size: 13px; line-height: 1.6; color: var(--color-text-secondary);">${escapeHTML(r.text)}</p>
                    <div style="margin-top: 6px; display: flex; gap: 6px;">
                        <span class="badge badge-red" style="font-size: 11px;">Negative</span>
                        ${r.priority === 'High' ? '<span class="badge badge-yellow" style="font-size: 11px;">⚠ Urgent</span>' : ''}
                    </div>
                </div>
            `).join('')}
        </div>` : ''}
    `;

    return card;
}


// ============================================================
// EXPAND/COLLAPSE
// ============================================================

function toggleNegativeReviews(headerEl) {
    const card = headerEl.closest('.card');
    const panel = card.querySelector('.negative-reviews-panel');
    const icon = headerEl.querySelector('.expand-icon');

    if (!panel) return;

    const isOpen = panel.style.display !== 'none';
    panel.style.display = isOpen ? 'none' : 'block';
    if (icon) icon.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
}


// ============================================================
// UTILS
// ============================================================

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
