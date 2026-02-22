"""
CITS — Drift Detection Module
Multi-signal data drift detection comparing recent reviews vs. the
baseline distribution saved at last retrain.

Signals:
1. Sentiment distribution shift — % positive reviews
2. Fake probability mean shift — average fake score
3. Vocabulary shift — new token frequency in recent data

Returns: Stable / Moderate Drift / High Drift
"""
import string
from collections import Counter
from datetime import datetime

from app.database import SessionLocal
from app.models import MLMetadata


# ════════════════════════════════════════════════════════════════
# BASELINE PERSISTENCE (via DB)
# ════════════════════════════════════════════════════════════════

def save_baseline(all_reviews) -> None:
    """
    Save the current data distribution as the drift baseline.
    Called after each successful retrain.
    """
    if not all_reviews:
        return

    total = len(all_reviews)
    pos_count = sum(1 for r in all_reviews if r.sentiment == "Positive")
    probs = [r.is_fake_prob for r in all_reviews if r.is_fake_prob is not None]

    baseline = {
        "positive_pct": round(pos_count / total * 100, 1) if total else 0,
        "fake_prob_mean": round(sum(probs) / len(probs), 4) if probs else 0,
        "total_reviews": total,
        "vocab_snapshot": _build_vocab_snapshot(all_reviews),
    }

    db = SessionLocal()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "drift_baseline").first()
        if row:
            row.value = baseline
            row.updated_at = datetime.utcnow()
        else:
            row = MLMetadata(key="drift_baseline", value=baseline)
            db.add(row)
        db.commit()
    finally:
        db.close()


def load_baseline() -> dict | None:
    """Load the saved baseline, or None if not saved yet."""
    db = SessionLocal()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "drift_baseline").first()
        return row.value if row else None
    finally:
        db.close()


def _build_vocab_snapshot(reviews, max_tokens=2000) -> dict:
    """Save top token counts as a vocab baseline."""
    counter = Counter()
    for r in reviews:
        counter.update(_tokenize(r.text))
    return dict(counter.most_common(max_tokens))


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with basic cleaning."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 2]


# ════════════════════════════════════════════════════════════════
# INDIVIDUAL DRIFT SIGNALS
# ════════════════════════════════════════════════════════════════

def _sentiment_shift(recent_reviews, baseline_positive_pct: float) -> dict:
    """Compare recent positive % against baseline."""
    if not recent_reviews:
        return {"shift": 0.0, "recent_pct": baseline_positive_pct, "overall_pct": baseline_positive_pct}

    recent_pos = sum(1 for r in recent_reviews if r.sentiment == "Positive") / len(recent_reviews)
    recent_pct = round(recent_pos * 100, 1)
    shift = abs(recent_pct - baseline_positive_pct)

    return {
        "shift": round(shift, 2),
        "recent_pct": recent_pct,
        "overall_pct": baseline_positive_pct,
    }


def _fake_prob_shift(recent_reviews, baseline_fake_mean: float) -> dict:
    """Compare recent fake probability mean against baseline."""
    recent_probs = [r.is_fake_prob for r in recent_reviews if r.is_fake_prob is not None]

    if not recent_probs:
        return {"shift": 0.0, "recent_mean": baseline_fake_mean, "overall_mean": baseline_fake_mean}

    recent_mean = sum(recent_probs) / len(recent_probs)
    shift = abs(recent_mean - baseline_fake_mean)

    return {
        "shift": round(shift, 4),
        "recent_mean": round(recent_mean, 4),
        "overall_mean": round(baseline_fake_mean, 4),
    }


def _vocabulary_shift(recent_reviews, baseline_vocab: dict) -> dict:
    """Check for new vocabulary in recent reviews not in the baseline vocab."""
    recent_tokens = Counter()
    for r in recent_reviews:
        recent_tokens.update(_tokenize(r.text))

    if not recent_tokens:
        return {"shift": 0.0, "new_token_count": 0, "total_recent_tokens": 0}

    new_tokens = sum(1 for t in recent_tokens if baseline_vocab.get(t, 0) < 2)
    total_unique = len(recent_tokens)
    shift = new_tokens / total_unique if total_unique > 0 else 0.0

    return {
        "shift": round(shift, 4),
        "new_token_count": new_tokens,
        "total_recent_tokens": total_unique,
    }


# ════════════════════════════════════════════════════════════════
# MAIN DRIFT DETECTION
# ════════════════════════════════════════════════════════════════

def detect_drift(recent_reviews, all_reviews) -> dict:
    """
    Multi-signal drift detection.
    Compares recent reviews against the saved baseline (from last retrain).
    Falls back to all_reviews as baseline if no saved baseline exists.
    """
    baseline = load_baseline()

    if baseline:
        baseline_pos_pct = baseline["positive_pct"]
        baseline_fake_mean = baseline["fake_prob_mean"]
        baseline_vocab = baseline.get("vocab_snapshot", {})
    else:
        total = len(all_reviews) or 1
        baseline_pos_pct = round(
            sum(1 for r in all_reviews if r.sentiment == "Positive") / total * 100, 1
        )
        probs = [r.is_fake_prob for r in all_reviews if r.is_fake_prob is not None]
        baseline_fake_mean = round(sum(probs) / len(probs), 4) if probs else 0
        baseline_vocab = _build_vocab_snapshot(all_reviews)

    sentiment = _sentiment_shift(recent_reviews, baseline_pos_pct)
    fake_prob = _fake_prob_shift(recent_reviews, baseline_fake_mean)
    vocabulary = _vocabulary_shift(recent_reviews, baseline_vocab)

    score = (
        min(sentiment["shift"], 50) * 0.4
        + min(fake_prob["shift"] * 50, 50) * 0.3
        + min(vocabulary["shift"] * 50, 50) * 0.3
    )
    score = round(min(score, 100), 1)

    if score > 12:
        level = "High Drift"
    elif score > 5:
        level = "Moderate Drift"
    else:
        level = "Stable"

    return {
        "drift_level": level,
        "score": score,
        "sentiment": sentiment,
        "fake_probability": fake_prob,
        "vocabulary": vocabulary,
    }
