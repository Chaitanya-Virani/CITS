/**
 * CITS — main.js
 * Frontend logic for user.html
 * Handles product selection, data fetching, review submission.
 */

let currentProductId = null;

document.addEventListener('DOMContentLoaded', () => {
    loadProducts();
    setupReviewForm();
});


// ============================================================
// PRODUCT SELECTOR
// ============================================================

async function loadProducts() {
    const select = document.getElementById('product-selector');
    try {
        const res = await fetch('/api/products');
        const products = await res.json();

        products.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            select.appendChild(opt);
        });

        select.addEventListener('change', (e) => {
            const id = e.target.value;
            if (id) {
                currentProductId = parseInt(id);
                loadProduct(currentProductId);
                loadReviews(currentProductId);
                loadSummary(currentProductId);
            }
        });

        // Auto-select first product if available
        if (products.length > 0) {
            select.value = products[0].id;
            currentProductId = products[0].id;
            loadProduct(currentProductId);
            loadReviews(currentProductId);
            loadSummary(currentProductId);
        }
    } catch (err) {
        console.error('Failed to load products:', err);
    }
}


// ============================================================
// PRODUCT DETAILS
// ============================================================

async function loadProduct(productId) {
    try {
        const res = await fetch(`/api/product/${productId}`);
        if (!res.ok) throw new Error('Product not found');
        const data = await res.json();

        document.getElementById('product-name').textContent = data.name;
        document.getElementById('product-category').textContent = data.category;
        document.getElementById('overall-rating').textContent = data.overall_rating.toFixed(1);
        document.getElementById('total-reviews').textContent = data.total_reviews.toLocaleString();

        // Trust score ring
        updateScoreRing(data.trust_score);
    } catch (err) {
        document.getElementById('product-name').textContent = 'Error loading product';
        console.error(err);
    }
}

function updateScoreRing(score) {
    const circumference = 2 * Math.PI * 45; // r=45 from SVG
    const offset = circumference - (score / 100) * circumference;

    const ring = document.querySelector('.score-ring-fill');
    const numberEl = document.querySelector('.score-ring-number');

    if (ring) ring.style.strokeDashoffset = offset;
    if (numberEl) numberEl.textContent = Math.round(score);
}


// ============================================================
// REVIEWS + RATING BREAKDOWN
// ============================================================

async function loadReviews(productId) {
    const list = document.getElementById('review-list');
    const breakdown = document.getElementById('rating-breakdown');

    list.innerHTML = '<p class="loading-text text-sm">Loading reviews…</p>';
    breakdown.innerHTML = '<p class="loading-text text-sm">Loading…</p>';

    try {
        const res = await fetch(`/api/reviews/${productId}`);
        if (!res.ok) throw new Error('Failed to load reviews');
        const data = await res.json();

        // Render rating breakdown
        renderBreakdown(data.rating_breakdown, data.total);

        // Render reviews
        if (data.reviews.length === 0) {
            list.innerHTML = '<p class="text-sm text-muted">No reviews yet. Be the first!</p>';
            return;
        }

        list.innerHTML = '';
        data.reviews.forEach(review => {
            list.appendChild(createReviewCard(review));
        });
    } catch (err) {
        list.innerHTML = '<p class="error-text">Could not load reviews.</p>';
        console.error(err);
    }
}

function renderBreakdown(ratingBreakdown, total) {
    const container = document.getElementById('rating-breakdown');
    container.innerHTML = '';

    for (let star = 5; star >= 1; star--) {
        const count = ratingBreakdown[String(star)] || 0;
        const pct = total > 0 ? (count / total * 100) : 0;

        const row = document.createElement('div');
        row.className = 'rating-row';
        row.innerHTML = `
      <span class="rating-label">${star} star</span>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${pct}%"></div>
      </div>
      <span class="rating-count">${count}</span>
    `;
        container.appendChild(row);
    }
}

function createReviewCard(review) {
    const card = document.createElement('div');
    card.className = 'review-card';

    const sentimentClass = review.sentiment === 'Positive' ? 'positive' : 'negative';
    const stars = renderStarsHTML(review.rating);
    const dateStr = review.date ? new Date(review.date).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' }) : '';

    card.innerHTML = `
    <div class="review-header">
      <div>
        <span class="review-author">${escapeHTML(review.author)}</span>
        <span class="ml-2">${stars}</span>
      </div>
      <span class="review-date">${dateStr}</span>
    </div>
    <p class="review-text">${escapeHTML(review.text)}</p>
    <span class="review-sentiment ${sentimentClass}">${review.sentiment}</span>
    ${review.priority === 'High' ? '<span class="review-sentiment negative ml-1" style="margin-left:6px;">⚠ Urgent</span>' : ''}
  `;
    return card;
}

function renderStarsHTML(rating) {
    let html = '<span class="stars">';
    for (let i = 1; i <= 5; i++) {
        if (i <= rating) {
            html += '<span class="material-symbols-outlined">star</span>';
        } else {
            html += '<span class="material-symbols-outlined empty">star</span>';
        }
    }
    html += '</span>';
    return html;
}


// ============================================================
// AI SUMMARY
// ============================================================

async function loadSummary(productId) {
    const el = document.getElementById('ai-summary');
    el.textContent = 'Generating summary…';

    try {
        const res = await fetch(`/api/summary/${productId}`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        el.textContent = data.summary;
    } catch (err) {
        el.textContent = 'Could not load AI summary.';
        el.classList.add('error-text');
        console.error(err);
    }
}


// ============================================================
// REVIEW SUBMISSION
// ============================================================

function setupReviewForm() {
    const form = document.getElementById('review-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentProductId) {
            showResult('Please select a product first.', 'error');
            return;
        }

        const author = document.getElementById('review-author').value.trim() || 'Anonymous';
        const rating = parseInt(document.getElementById('review-rating').value);
        const text = document.getElementById('review-text').value.trim();

        if (text.length < 5) {
            showResult('Review must be at least 5 characters.', 'error');
            return;
        }

        showResult('Submitting…', '');

        try {
            const res = await fetch('/api/reviews', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_id: currentProductId,
                    author: author,
                    rating: rating,
                    text: text,
                }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Submission failed');
            }

            const data = await res.json();
            showResult(`${data.sentiment} review submitted (${data.priority} priority)`, 'success');

            // Clear form
            document.getElementById('review-text').value = '';
            document.getElementById('review-author').value = '';

            // Reload reviews and product data
            loadProduct(currentProductId);
            loadReviews(currentProductId);
            loadSummary(currentProductId);
        } catch (err) {
            showResult('Error: ' + err.message, 'error');
            console.error(err);
        }
    });
}

function showResult(msg, type) {
    const el = document.getElementById('review-result');
    if (!el) return;
    el.textContent = msg;
    el.style.color = type === 'error' ? 'var(--color-red)' :
        type === 'success' ? 'var(--color-green)' : 'var(--color-muted)';
}


// ============================================================
// UTILS
// ============================================================

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
