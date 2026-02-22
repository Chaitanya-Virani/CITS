"""
CITS — Drift Detection Module
Multi-signal data drift detection comparing recent vs. overall review data.

Signals:
1. Sentiment distribution shift — % positive reviews
2. Fake probability mean shift — average fake score
3. Vocabulary shift — new token frequency in recent data

Returns: Stable / Moderate Drift / High Drift
"""
import re
import string
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with basic cleaning."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 2]


def _sentiment_shift(recent_reviews, all_reviews) -> dict:
    """Compare positive sentiment percentage: recent vs overall."""
    if not all_reviews:
        return {"shift": 0.0, "recent_pct": 0.0, "overall_pct": 0.0}

    overall_pos = sum(1 for r in all_reviews if r.sentiment == "Positive") / len(all_reviews)

    if not recent_reviews:
        pct = round(overall_pos * 100, 1)
        return {"shift": 0.0, "recent_pct": pct, "overall_pct": pct}

    recent_pos = sum(1 for r in recent_reviews if r.sentiment == "Positive") / len(recent_reviews)
    shift = abs(recent_pos - overall_pos) * 100

    return {
        "shift": round(shift, 2),
        "recent_pct": round(recent_pos * 100, 1),
        "overall_pct": round(overall_pos * 100, 1),
    }


def _fake_prob_shift(recent_reviews, all_reviews) -> dict:
    """Compare mean fake probability: recent vs overall."""
    all_probs = [r.is_fake_prob for r in all_reviews if r.is_fake_prob is not None]
    recent_probs = [r.is_fake_prob for r in recent_reviews if r.is_fake_prob is not None]

    if not all_probs:
        return {"shift": 0.0, "recent_mean": 0.0, "overall_mean": 0.0}

    overall_mean = sum(all_probs) / len(all_probs)
    recent_mean = sum(recent_probs) / len(recent_probs) if recent_probs else overall_mean
    shift = abs(recent_mean - overall_mean)

    return {
        "shift": round(shift, 4),
        "recent_mean": round(recent_mean, 4),
        "overall_mean": round(overall_mean, 4),
    }


def _vocabulary_shift(recent_reviews, all_reviews) -> dict:
    """
    Check for new vocabulary in recent reviews not seen in the broader dataset.
    Returns the fraction of unique recent tokens that are 'new'.
    """
    all_tokens = Counter()
    for r in all_reviews:
        all_tokens.update(_tokenize(r.text))

    recent_tokens = Counter()
    for r in recent_reviews:
        recent_tokens.update(_tokenize(r.text))

    if not recent_tokens:
        return {"shift": 0.0, "new_token_count": 0, "total_recent_tokens": 0}

    # Tokens in recent that appear < 2 times in overall corpus (effectively "new")
    new_tokens = sum(1 for t in recent_tokens if all_tokens.get(t, 0) < 2)
    total_unique = len(recent_tokens)
    shift = new_tokens / total_unique if total_unique > 0 else 0.0

    return {
        "shift": round(shift, 4),
        "new_token_count": new_tokens,
        "total_recent_tokens": total_unique,
    }


def detect_drift(recent_reviews, all_reviews) -> dict:
    """
    Multi-signal drift detection.

    Returns:
        {
            "drift_level": "Stable" | "Moderate Drift" | "High Drift",
            "sentiment": {...},
            "fake_probability": {...},
            "vocabulary": {...},
            "score": float  # 0-100 composite drift score
        }
    """
    sentiment = _sentiment_shift(recent_reviews, all_reviews)
    fake_prob = _fake_prob_shift(recent_reviews, all_reviews)
    vocabulary = _vocabulary_shift(recent_reviews, all_reviews)

    # Composite score: weighted sum of individual shifts
    # Sentiment shift (0-50 range) → weight 0.4
    # Fake prob shift (0-1 range, scaled to 0-50) → weight 0.3
    # Vocabulary shift (0-1 range, scaled to 0-50) → weight 0.3
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
