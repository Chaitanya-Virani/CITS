"""
CITS — Bilingual Adaptive Urgency Detection
Language detection, Hinglish normalization, TF-IDF keyword extraction,
urgency classifier, and persistent keyword management.
"""
import os
import re
import json
import string
import threading
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYWORDS_PATH = os.path.join(BASE_DIR, "database", "urgency_keywords.json")

_lock = threading.Lock()

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

# Protected base keywords — NEVER removed
BASE_KEYWORDS = {
    "english": [
        "fake", "fraud", "scam", "cheat", "duplicate", "used product",
        "broken", "damaged", "return", "refund", "waste", "useless",
        "worst", "bad quality", "leak", "defective", "counterfeit",
        "misleading", "dangerous", "hazard",
    ],
    "hinglish": [
        "ghatiya", "nakli", "chor", "tuta", "bekaar", "bakwas",
        "dhoka", "paisa barbaad", "kharab", "bekar", "wahiyat",
        "ganda", "jhoota", "makaar", "lootere",
    ],
}

# Generic terms to exclude from learned keywords
BLOCKED_TERMS = {
    "product", "delivery", "item", "seller", "order", "price", "time",
    "day", "good", "nice", "amazon", "buy", "bought", "bought", "review",
    "star", "stars", "one", "two", "three", "four", "five", "would",
    "could", "also", "really", "very", "just", "like", "much", "even",
    "still", "well", "make", "made", "thing", "things", "way", "got",
    "get", "use", "used", "using", "come", "came", "work", "working",
}

# Hinglish stopwords (grammatical, not complaint-related)
HINGLISH_STOPWORDS = {
    "hai", "tha", "the", "ka", "ki", "ke", "yeh", "woh", "kya",
    "nahi", "mat", "aur", "par", "bhi", "toh", "mein", "se", "ko",
    "hum", "tum", "aap", "mera", "tera", "uska", "isko", "usko",
    "kuch", "sab", "bahut", "jaise", "phir", "liye", "wala", "wali",
}

# Known Hindi romanized tokens for language detection
HINDI_MARKERS = {
    "hai", "tha", "nahi", "mat", "bahut", "bekaar", "ghatiya", "chor",
    "nakli", "accha", "acha", "kharab", "paisa", "paise", "kaam",
    "cheez", "saman", "dukan", "lena", "dena", "milta", "milti",
    "dikha", "dikhta", "lagta", "lagti", "hota", "hoti", "karo",
    "karna", "raha", "rahi", "wala", "wali", "bola", "boli",
    "dekho", "dekha", "achha", "bura", "sasta", "mehenga", "mehnga",
    "theek", "thik", "bilkul", "ekdum", "sachhi", "jhoot", "jhoota",
}

# Hinglish transliteration normalization map
HINGLISH_NORMALIZATION_MAP = {
    "gatiya": "ghatiya",
    "ghatiyaa": "ghatiya",
    "ghatiyaaa": "ghatiya",
    "naklee": "nakli",
    "nakali": "nakli",
    "fraaud": "fraud",
    "defected": "defective",
    "bekaar": "bekar",
    "wahiyaat": "wahiyat",
    "bakwaas": "bakwas",
    "bakwaass": "bakwas",
    "kharab": "kharab",
    "gandaa": "ganda",
    "gandha": "ganda",
    "tootaa": "tuta",
    "toota": "tuta",
    "lootere": "lootere",
    "makar": "makaar",
    "paisaa": "paisa",
    "barbad": "barbaad",
    "dhokha": "dhoka",
    "jhooti": "jhoota",
}

MAX_LEARNED_PER_LANGUAGE = 30
MIN_WORD_LENGTH = 3


# ════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ════════════════════════════════════════════════════════════════

def normalize_hinglish(text: str) -> str:
    """
    Normalize Hinglish text:
    1. Collapse repeated letters (ghatiyaaa → ghatiya)
    2. Apply transliteration mapping
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Collapse repeated letters (3+ consecutive same char → 1 or 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # aaa→aa
    text = re.sub(r'(.)\1{1,}', r'\1', text)     # aa→a (for single-char words)

    # Apply token-level normalization
    tokens = text.split()
    normalized = []
    for token in tokens:
        clean_tok = token.strip(string.punctuation)
        mapped = HINGLISH_NORMALIZATION_MAP.get(clean_tok, clean_tok)
        normalized.append(mapped)

    return " ".join(normalized)


def _clean_for_extraction(text: str) -> str:
    """Clean text for keyword extraction (lower, remove HTML, punctuation)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return normalize_hinglish(text)


# ════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION (heuristic)
# ════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Classify text as 'en' or 'hinglish' using token-based heuristics.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "en"

    tokens = set(text.lower().split())
    if not tokens:
        return "en"

    # Count Hindi marker overlap
    hindi_overlap = len(tokens & HINDI_MARKERS)
    hindi_ratio = hindi_overlap / len(tokens)

    # If ≥15% of tokens are Hindi markers → Hinglish
    if hindi_ratio >= 0.15 or hindi_overlap >= 3:
        return "hinglish"

    return "en"


# ════════════════════════════════════════════════════════════════
# KEYWORD PERSISTENCE
# ════════════════════════════════════════════════════════════════

def _default_keywords() -> dict:
    """Return the default keywords structure."""
    return {
        "base_keywords": {
            "english": list(BASE_KEYWORDS["english"]),
            "hinglish": list(BASE_KEYWORDS["hinglish"]),
        },
        "learned_keywords": {
            "english": [],
            "hinglish": [],
        },
        "last_updated": datetime.utcnow().isoformat(),
        "total_keywords": len(BASE_KEYWORDS["english"]) + len(BASE_KEYWORDS["hinglish"]),
    }


def load_urgency_keywords() -> dict:
    """Load urgency keywords from JSON file."""
    with _lock:
        if not os.path.exists(KEYWORDS_PATH):
            data = _default_keywords()
            save_urgency_keywords(data)
            return data
        try:
            with open(KEYWORDS_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_keywords()


def save_urgency_keywords(data: dict) -> None:
    """Save urgency keywords to JSON file."""
    os.makedirs(os.path.dirname(KEYWORDS_PATH), exist_ok=True)
    with _lock:
        with open(KEYWORDS_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)


def get_all_keywords() -> dict:
    """Get combined base + learned keywords per language."""
    data = load_urgency_keywords()
    return {
        "english": list(set(
            data["base_keywords"]["english"] + data["learned_keywords"]["english"]
        )),
        "hinglish": list(set(
            data["base_keywords"]["hinglish"] + data["learned_keywords"]["hinglish"]
        )),
    }


# ════════════════════════════════════════════════════════════════
# BILINGUAL KEYWORD EXTRACTION (called during retrain)
# ════════════════════════════════════════════════════════════════

def _compute_urgency_scores(negative_texts: list[str], positive_texts: list[str],
                            stopwords: set, top_n: int) -> list[str]:
    """
    Extract top urgent keywords using TF-IDF + frequency ratio.
    urgency_score = freq_in_negative / (freq_in_positive + 1)
    """
    if not negative_texts:
        return []

    # Tokenize and count
    neg_counter = Counter()
    for text in negative_texts:
        tokens = [w for w in text.split() if len(w) >= MIN_WORD_LENGTH
                  and w not in stopwords and w not in BLOCKED_TERMS]
        neg_counter.update(tokens)

    pos_counter = Counter()
    for text in positive_texts:
        tokens = [w for w in text.split() if len(w) >= MIN_WORD_LENGTH]
        pos_counter.update(tokens)

    # Compute urgency scores
    scores = {}
    for word, neg_freq in neg_counter.items():
        if neg_freq < 3:  # Need at least 3 occurrences
            continue
        pos_freq = pos_counter.get(word, 0)
        scores[word] = neg_freq / (pos_freq + 1)

    # Sort by urgency score, take top N
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def extract_and_update_keywords(reviews, drift_level: str = "Stable") -> dict:
    """
    Extract urgent keywords from reviews and update persistent storage.

    Args:
        reviews: list of Review ORM objects
        drift_level: current drift status — affects extraction size

    Returns:
        Updated keywords dict
    """
    # Drift-aware extraction size
    top_n = 40 if drift_level == "High Drift" else 20

    # ── 1. Filter negative reviews ────────────────────────────
    negative_reviews = [
        r for r in reviews
        if r.rating <= 2
        or r.sentiment == "Negative"
        or (r.is_fake_prob is not None and r.is_fake_prob > 0.5)
    ]
    positive_reviews = [
        r for r in reviews
        if r.rating >= 4 and r.sentiment == "Positive"
    ]

    if len(negative_reviews) < 10:
        return load_urgency_keywords()

    # ── 2. Clean and detect language ──────────────────────────
    en_neg_texts, hi_neg_texts = [], []
    en_pos_texts, hi_pos_texts = [], []

    for r in negative_reviews:
        cleaned = _clean_for_extraction(r.text)
        lang = detect_language(r.text)
        if lang == "hinglish":
            hi_neg_texts.append(cleaned)
        else:
            en_neg_texts.append(cleaned)

    for r in positive_reviews:
        cleaned = _clean_for_extraction(r.text)
        lang = detect_language(r.text)
        if lang == "hinglish":
            hi_pos_texts.append(cleaned)
        else:
            en_pos_texts.append(cleaned)

    # ── 3. Extract keywords per language ──────────────────────
    # English stopwords = BLOCKED_TERMS (already filtered in _compute)
    en_learned = _compute_urgency_scores(
        en_neg_texts, en_pos_texts, BLOCKED_TERMS, top_n
    )

    # Hinglish — use separate stopword list
    hi_learned = _compute_urgency_scores(
        hi_neg_texts, hi_pos_texts, HINGLISH_STOPWORDS | BLOCKED_TERMS, top_n
    )

    # ── 4. Cap and merge ──────────────────────────────────────
    # Remove any base keywords (they're already protected)
    base_en = set(w.lower() for w in BASE_KEYWORDS["english"])
    base_hi = set(w.lower() for w in BASE_KEYWORDS["hinglish"])

    en_learned = [w for w in en_learned if w not in base_en][:MAX_LEARNED_PER_LANGUAGE]
    hi_learned = [w for w in hi_learned if w not in base_hi][:MAX_LEARNED_PER_LANGUAGE]

    # ── 5. Update persistent storage ──────────────────────────
    data = load_urgency_keywords()
    data["learned_keywords"]["english"] = en_learned
    data["learned_keywords"]["hinglish"] = hi_learned
    data["last_updated"] = datetime.utcnow().isoformat()
    data["total_keywords"] = (
        len(data["base_keywords"]["english"]) + len(data["base_keywords"]["hinglish"])
        + len(en_learned) + len(hi_learned)
    )
    save_urgency_keywords(data)

    return data


# ════════════════════════════════════════════════════════════════
# URGENCY CHECK (replaces old static check_urgency)
# ════════════════════════════════════════════════════════════════

def check_urgency_adaptive(text: str) -> dict:
    """
    Check if review text is urgent using adaptive bilingual keyword matching.

    Returns:
        {
            "is_urgent": bool,
            "matched_keywords": list[str],
            "urgency_score": int,
            "language": "en" | "hinglish"
        }
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"is_urgent": False, "matched_keywords": [], "urgency_score": 0, "language": "en"}

    # Detect language
    language = detect_language(text)

    # Normalize text
    normalized = _clean_for_extraction(text)
    tokens = set(normalized.split())

    # Load all keywords
    all_kw = get_all_keywords()

    # Match against both English and Hinglish keywords
    matched = []

    for kw in all_kw["english"]:
        # Multi-word keywords: check substring
        if " " in kw:
            if kw in normalized:
                matched.append(kw)
        elif kw in tokens:
            matched.append(kw)

    for kw in all_kw["hinglish"]:
        if " " in kw:
            if kw in normalized:
                matched.append(kw)
        elif kw in tokens:
            matched.append(kw)

    matched = list(set(matched))
    urgency_score = len(matched)

    return {
        "is_urgent": urgency_score > 0,
        "matched_keywords": matched,
        "urgency_score": urgency_score,
        "language": language,
    }
