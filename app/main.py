import pickle
import os
from fastapi import FastAPI
from app.schemas import ReviewRequest, ReviewResponse
from app.utils import clean_text, check_urgency

app = FastAPI(title="Customer Insight Triage System", version="1.0")

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UPDATED: Changed filename to 'cits.pkl'
MODEL_PATH = os.path.join(BASE_DIR, "model", "cits.pkl")

print(f"Loading model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model Loaded Successfully!")
except FileNotFoundError:
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    model = None

# --- API ENDPOINT ---
@app.post("/predict", response_model=ReviewResponse)
def predict_review(request: ReviewRequest):
    if not model:
        return {"sentiment": "Error", "priority": "Error", "flag": "Model Missing"}

    # 1. Clean Input
    cleaned_input = clean_text(request.text)

    # 2. Predict Sentiment
    prediction = model.predict([cleaned_input])[0]
    sentiment_label = "Positive" if prediction == 1 else "Negative"

    # 3. Check Urgency
    is_urgent = check_urgency(request.text)
    priority_label = "High" if is_urgent else "Normal"
    flag_label = "🚨 URGENT" if is_urgent else "✅ OK"

    return {
        "sentiment": sentiment_label,
        "priority": priority_label,
        "flag": flag_label
    }

# --- HOME ---
@app.get("/")
def home():
    return {"status": "Online", "model": "cits.pkl"}