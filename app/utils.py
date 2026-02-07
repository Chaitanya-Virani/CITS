import re
import string

# 1. CLEANING FUNCTION
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. URGENCY LIST
URGENT_KEYWORDS = [
    "fake", "fraud", "scam", "cheat", "chor", "duplicate", "used product",
    "broken", "damaged", "tuta", "return", "refund", "waste", "useless",
    "worst", "ghatiya", "bad quality", "leak"
]

# 3. URGENCY CHECK
def check_urgency(text: str) -> bool:
    text = str(text).lower()
    for word in URGENT_KEYWORDS:
        if word in text:
            return True
    return False