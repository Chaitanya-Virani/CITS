"""
CITS — Bilingual Adaptive Urgency Detection
Language detection, Hinglish normalization, TF-IDF keyword extraction,
urgency classifier, and persistent keyword management via DB.
"""
import re
import string
from collections import Counter
from datetime import datetime

from app.database import SessionLocal
from app.models import MLMetadata

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
    "day", "good", "nice", "amazon", "buy", "bought", "review",
    "star", "stars", "one", "two", "three", "four", "five", "would",
    "could", "also", "really", "very", "just", "like", "much", "even",
    "still", "well", "make", "made", "thing", "things", "way", "got",
    "get", "use", "used", "using", "come", "came", "work", "working",
}

# Hinglish stopwords
HINGLISH_STOPWORDS = {
    "hai", "tha", "the", "ka", "ki", "ke", "yeh", "woh", "kya",
    "nahi", "mat", "aur", "par", "bhi", "toh", "mein", "se", "ko",
    "hum", "tum", "aap", "mera", "tera", "uska", "isko", "usko",
    "kuch", "sab", "bahut", "jaise", "phir", "liye", "wala", "wali",
}

# Hindi markers for language detection
HINDI_MARKERS = {
    "hai", "tha", "nahi", "mat", "bahut", "bekaar", "ghatiya", "chor",
    "nakli", "accha", "acha", "kharab", "paisa", "paise", "kaam",
    "cheez", "saman", "dukan", "lena", "dena", "milta", "milti",
    "dikha", "dikhta", "lagta", "lagti", "hota", "hoti", "karo",
    "karna", "raha", "rahi", "wala", "wali", "bola", "boli",
    "dekho", "dekha", "achha", "bura", "sasta", "mehenga", "mehnga",
    "theek", "thik", "bilkul", "ekdum", "sachhi", "jhoot", "jhoota",
}

# Transliteration normalization
HINGLISH_NORMALIZATION_MAP = {
    "gatiya": "ghatiya", "ghatiyaa": "ghatiya", "ghatiyaaa": "ghatiya",
    "naklee": "nakli", "nakali": "nakli", "fraaud": "fraud",
    "defected": "defective", "bekaar": "bekar", "wahiyaat": "wahiyat",
    "bakwaas": "bakwas", "bakwaass": "bakwas", "kharab": "kharab",
    "gandaa": "ganda", "gandha": "ganda", "tootaa": "tuta",
    "toota": "tuta", "lootere": "lootere", "makar": "makaar",
    "paisaa": "paisa", "barbad": "barbaad", "dhokha": "dhoka", "jhooti": "jhoota",
}

MAX_LEARNED_PER_LANGUAGE = 30
MIN_WORD_LENGTH = 3


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def normalize_hinglish(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'(.)\1{1,}', r'\1', text)
    tokens = text.split()
    normalized = [HINGLISH_NORMALIZATION_MAP.get(t.strip(string.punctuation), t.strip(string.punctuation)) for t in tokens]
    return " ".join(normalized)


def _clean_for_extraction(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return normalize_hinglish(text)


def detect_language(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 5: return "en"
    tokens = set(text.lower().split())
    if not tokens: return "en"
    hindi_overlap = len(tokens & HINDI_MARKERS)
    if (hindi_overlap / len(tokens)) >= 0.15 or hindi_overlap >= 3:
        return "hinglish"
    return "en"


# ════════════════════════════════════════════════════════════════
# KEYWORD PERSISTENCE (DB)
# ════════════════════════════════════════════════════════════════

def _default_keywords() -> dict:
    return {
        "base_keywords": {
            "english": list(BASE_KEYWORDS["english"]),
            "hinglish": list(BASE_KEYWORDS["hinglish"]),
        },
        "learned_keywords": {"english": [], "hinglish": []},
        "last_updated": datetime.utcnow().isoformat(),
        "total_keywords": len(BASE_KEYWORDS["english"]) + len(BASE_KEYWORDS["hinglish"]),
    }


def load_urgency_keywords() -> dict:
    """Load urgency keywords from the MLMetadata table."""
    db = SessionLocal()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "urgency_keywords").first()
        if not row:
            data = _default_keywords()
            save_urgency_keywords(data)
            return data
        return row.value
    finally:
        db.close()


def save_urgency_keywords(data: dict) -> None:
    """Save urgency keywords to the MLMetadata table."""
    db = SessionLocal()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "urgency_keywords").first()
        if row:
            row.value = data
            row.updated_at = datetime.utcnow()
        else:
            row = MLMetadata(key="urgency_keywords", value=data)
            db.add(row)
        db.commit()
    finally:
        db.close()


def get_all_keywords() -> dict:
    data = load_urgency_keywords()
    return {
        "english": list(set(data["base_keywords"]["english"] + data["learned_keywords"]["english"])),
        "hinglish": list(set(data["base_keywords"]["hinglish"] + data["learned_keywords"]["hinglish"])),
    }


# ════════════════════════════════════════════════════════════════
# EXTRACTION & CHECK
# ════════════════════════════════════════════════════════════════

def _compute_urgency_scores(negative_texts: list[str], positive_texts: list[str], stopwords: set, top_n: int) -> list[str]:
    if not negative_texts: return []
    neg_counter = Counter()
    for text in negative_texts:
        neg_counter.update([w for w in text.split() if len(w) >= MIN_WORD_LENGTH and w not in stopwords and w not in BLOCKED_TERMS])
    pos_counter = Counter()
    for text in positive_texts:
        pos_counter.update([w for w in text.split() if len(w) >= MIN_WORD_LENGTH])
    scores = {word: neg_freq / (pos_counter.get(word, 0) + 1) for word, neg_freq in neg_counter.items() if neg_freq >= 3}
    return [word for word, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def extract_and_update_keywords(reviews, drift_level: str = "Stable") -> dict:
    top_n = 40 if drift_level == "High Drift" else 20
    neg_revs = [r for r in reviews if r.rating <= 2 or r.sentiment == "Negative" or (r.is_fake_prob and r.is_fake_prob > 0.5)]
    pos_revs = [r for r in reviews if r.rating >= 4 and r.sentiment == "Positive"]
    if len(neg_revs) < 10: return load_urgency_keywords()

    en_neg, hi_neg, en_pos, hi_pos = [], [], [], []
    for r in neg_revs:
        c, l = _clean_for_extraction(r.text), detect_language(r.text)
        (hi_neg if l == "hinglish" else en_neg).append(c)
    for r in pos_revs:
        c, l = _clean_for_extraction(r.text), detect_language(r.text)
        (hi_pos if l == "hinglish" else en_pos).append(c)

    en_learned = _compute_urgency_scores(en_neg, en_pos, BLOCKED_TERMS, top_n)
    hi_learned = _compute_urgency_scores(hi_neg, hi_pos, HINGLISH_STOPWORDS | BLOCKED_TERMS, top_n)

    base_en = set(w.lower() for w in BASE_KEYWORDS["english"])
    base_hi = set(w.lower() for w in BASE_KEYWORDS["hinglish"])
    en_learned = [w for w in en_learned if w not in base_en][:MAX_LEARNED_PER_LANGUAGE]
    hi_learned = [w for w in hi_learned if w not in base_hi][:MAX_LEARNED_PER_LANGUAGE]

    data = load_urgency_keywords()
    data.update({
        "learned_keywords": {"english": en_learned, "hinglish": hi_learned},
        "last_updated": datetime.utcnow().isoformat(),
        "total_keywords": len(data["base_keywords"]["english"]) + len(data["base_keywords"]["hinglish"]) + len(en_learned) + len(hi_learned)
    })
    save_urgency_keywords(data)
    return data


def check_urgency_adaptive(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"is_urgent": False, "matched_keywords": [], "urgency_score": 0, "language": "en"}
    lang = detect_language(text)
    norm = _clean_for_extraction(text)
    tokens = set(norm.split())
    all_kw = get_all_keywords()
    matched = []
    for kw in all_kw["english"] + all_kw["hinglish"]:
        if (" " in kw and kw in norm) or (kw in tokens):
            matched.append(kw)
    matched = list(set(matched))
    return {"is_urgent": len(matched) > 0, "matched_keywords": matched, "urgency_score": len(matched), "language": lang}
