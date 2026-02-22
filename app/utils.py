"""
CITS — Utility Functions
Text cleaning, urgency detection, summary generation, trust scoring, metrics, drift.
"""
import re
import string
from collections import Counter


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Clean review text for ML model input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# URGENCY DETECTION (Adaptive Bilingual)
# ============================================================

def check_urgency(text: str) -> dict:
    """
    Check if review text is urgent using adaptive bilingual keyword matching.
    Supports English + Hinglish with learned keywords from retraining.

    Returns:
        {
            "is_urgent": bool,
            "matched_keywords": list[str],
            "urgency_score": int,  # count of matched keywords
            "language": "en" | "hinglish"
        }
    """
    from ml.urgency_extraction import check_urgency_adaptive
    return check_urgency_adaptive(text)


# ============================================================
# FAKE REVIEW PROBABILITY SCORING
# ============================================================

def compute_fake_score(text: str, rating: int, sentiment: str,
                       model=None, cleaned_text: str = "") -> float:
    """
    Multi-signal fake review probability (0.0 = genuine, 1.0 = fake).

    Signals:
    1. Text length — very short or generic reviews are suspicious
    2. Model uncertainty — prediction probability near 0.5
    3. Rating-sentiment mismatch — 5★ but model says Negative
    4. Word repetition — high repetition ratio is suspicious
    5. Extreme rating + vague text — 1★ or 5★ with < 30 chars
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 0.5  # Unknown → moderate suspicion

    score = 0.0
    clean = cleaned_text or clean_text(text)
    words = clean.split()
    word_count = len(words)

    # Signal 1: Text length (very short reviews are suspicious)
    if word_count < 5:
        score += 0.25
    elif word_count < 10:
        score += 0.10

    # Signal 2: Model uncertainty (probability near 0.5 = confused = suspicious)
    if model is not None:
        try:
            proba = model.predict_proba([clean])[0]
            confidence = max(proba)
            if confidence < 0.6:
                score += 0.25  # Very uncertain
            elif confidence < 0.75:
                score += 0.12
        except Exception:
            pass

    # Signal 3: Rating-sentiment mismatch
    if rating >= 4 and sentiment == "Negative":
        score += 0.30  # High rating but negative text = suspicious
    elif rating <= 2 and sentiment == "Positive":
        score += 0.20  # Low rating but positive text = somewhat suspicious

    # Signal 4: Word repetition ratio
    if word_count > 3:
        unique_ratio = len(set(words)) / word_count
        if unique_ratio < 0.4:
            score += 0.20  # Very repetitive
        elif unique_ratio < 0.6:
            score += 0.08

    # Signal 5: Extreme rating + vague (short) text
    if rating in (1, 5) and word_count < 8:
        score += 0.15

    return round(min(score, 1.0), 4)


# ============================================================
# TRUST SCORE COMPUTATION
# ============================================================

def compute_trust_score(reviews) -> float:
    """
    Compute trust score (0-100) from a list of review ORM objects.
    Formula: (positive% * 0.6) + (avg_rating/5 * 0.3) + ((1 - avg_fake) * 0.1) * 100
    """
    if not reviews:
        return 0.0

    total = len(reviews)
    positive_count = sum(1 for r in reviews if r.sentiment == "Positive")
    positive_pct = positive_count / total

    avg_rating = sum(r.rating for r in reviews) / total
    avg_fake = sum(r.is_fake_prob for r in reviews) / total

    score = (positive_pct * 0.6 + (avg_rating / 5) * 0.3 + (1 - avg_fake) * 0.1) * 100
    return round(min(max(score, 0), 100), 1)


# ============================================================
# SUMMARY GENERATION (rule-based)
# ============================================================

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "a", "an", "the", "and", "but", "or", "for", "not", "no", "so", "if", "of",
    "to", "in", "on", "at", "by", "from", "with", "as", "into", "about", "after",
    "its", "very", "just", "also", "than", "more", "only", "can", "all", "am",
    "up", "out", "some", "what", "which", "when", "how", "too", "any", "each",
    "get", "got", "use", "used", "using", "one", "two", "even", "much", "like",
    "really", "dont", "im", "ive", "product", "review", "bought", "buy", "good",
    "bad", "well", "still", "lot", "many", "thing", "make", "made", "going",
}


def generate_summary(reviews) -> dict:
    """
    Generate a rule-based summary from a list of review ORM objects.
    Returns: {summary: str, positive_count: int, negative_count: int, top_keywords: list}
    """
    if not reviews:
        return {
            "summary": "No reviews available for this product yet.",
            "positive_count": 0,
            "negative_count": 0,
            "top_keywords": [],
        }

    total = len(reviews)
    positive = [r for r in reviews if r.sentiment == "Positive"]
    negative = [r for r in reviews if r.sentiment == "Negative"]
    pos_count = len(positive)
    neg_count = len(negative)
    avg_rating = sum(r.rating for r in reviews) / total
    urgent_count = sum(1 for r in reviews if r.priority == "High")

    # Extract top keywords
    all_words = []
    for r in reviews:
        words = clean_text(r.text).split()
        all_words.extend(w for w in words if len(w) > 3 and w not in STOP_WORDS)

    top_keywords = [word for word, _ in Counter(all_words).most_common(8)]

    # Build summary text
    pos_pct = round(pos_count / total * 100, 1)
    neg_pct = round(neg_count / total * 100, 1)

    parts = []
    parts.append(
        f"Based on {total} reviews, this product has an average rating of "
        f"{avg_rating:.1f}/5 stars."
    )

    if pos_pct > 70:
        parts.append(
            f"The overall sentiment is strongly positive ({pos_pct}% positive reviews), "
            f"indicating high customer satisfaction."
        )
    elif pos_pct > 50:
        parts.append(
            f"The sentiment is mostly positive ({pos_pct}% positive, {neg_pct}% negative), "
            f"with some areas for improvement."
        )
    else:
        parts.append(
            f"The sentiment leans negative ({neg_pct}% negative reviews), "
            f"suggesting significant customer concerns."
        )

    if urgent_count > 0:
        parts.append(
            f"{urgent_count} reviews were flagged as urgent, mentioning issues like "
            f"potential fraud, damage, or quality problems."
        )

    if top_keywords:
        parts.append(
            f"Frequently mentioned topics include: {', '.join(top_keywords[:5])}."
        )

    return {
        "summary": " ".join(parts),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "top_keywords": top_keywords,
    }


# ============================================================
# DRIFT DETECTION
# ============================================================

def detect_drift(recent_reviews, all_reviews) -> dict:
    """
    Compare sentiment distribution of recent reviews vs all reviews.
    Returns: {drift_level: str, recent_positive_pct: float, overall_positive_pct: float}
    """
    if not all_reviews:
        return {"drift_level": "low", "recent_positive_pct": 0, "overall_positive_pct": 0}

    overall_pos = sum(1 for r in all_reviews if r.sentiment == "Positive") / len(all_reviews)

    if not recent_reviews:
        return {
            "drift_level": "low",
            "recent_positive_pct": round(overall_pos * 100, 1),
            "overall_positive_pct": round(overall_pos * 100, 1),
        }

    recent_pos = sum(1 for r in recent_reviews if r.sentiment == "Positive") / len(recent_reviews)
    delta = abs(recent_pos - overall_pos) * 100

    if delta > 10:
        level = "high"
    elif delta > 5:
        level = "medium"
    else:
        level = "low"

    return {
        "drift_level": level,
        "recent_positive_pct": round(recent_pos * 100, 1),
        "overall_positive_pct": round(overall_pos * 100, 1),
    }


# ============================================================
# DATASET HEALTH
# ============================================================

def compute_dataset_health(reviews) -> float:
    """Percentage of reviews that have valid, non-empty text."""
    if not reviews:
        return 0.0
    valid = sum(1 for r in reviews if r.text and len(r.text.strip()) > 10)
    return round(valid / len(reviews) * 100, 1)