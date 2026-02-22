"""
CITS — Admin Routes
GET  /api/model-metrics   — model accuracy, F1, distributions (from metadata + live)
GET  /api/drift-status    — multi-signal drift detection  
GET  /api/drift           — legacy alias
POST /api/retrain         — retrain model (background task)
GET  /api/retrain-status  — poll retrain completion
GET  /api/admin/reviews   — product-wise review breakdown
"""
import os
import pickle
import numpy as np
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sklearn.model_selection import train_test_split

from app.database import get_db, SessionLocal
from app.models import Review, Product
from app.schemas import (
    ModelMetricsResponse, ComparisonRow,
    DriftResponse, DriftSignal,
    RetrainResponse, RetrainStatusResponse,
    UrgencyKeywordsResponse, UrgencyKeywordSet,
)
from app.utils import clean_text, compute_dataset_health
from ml.evaluate import evaluate_model
from ml.metrics_service import load_metadata, save_metadata
from ml.drift_detection import detect_drift
from ml.retrain_pipeline import run_retrain, get_retrain_status
from ml.urgency_extraction import load_urgency_keywords

router = APIRouter(prefix="/api", tags=["Admin"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cits.pkl")


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# GET /api/model-metrics
# ─────────────────────────────────────────────────────────────

@router.get("/model-metrics", response_model=ModelMetricsResponse)
def get_model_metrics(db: Session = Depends(get_db)):
    """Get model version, accuracy, F1, distributions, and comparison table."""
    model = _load_model()
    reviews = db.query(Review).all()
    total = len(reviews)

    # Load persistent metadata
    metadata = load_metadata()

    # Live evaluation on current data
    if total >= 20:
        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        live_metrics = evaluate_model(model, X_test, y_test)
    else:
        live_metrics = {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    # Use metadata metrics if they exist (set by retrain), else use live
    accuracy = metadata.get("accuracy") or live_metrics["accuracy"]
    f1 = metadata.get("f1_score") or live_metrics["f1"]
    precision = metadata.get("precision") or live_metrics["precision"]
    recall = metadata.get("recall") or live_metrics["recall"]

    # If metadata has zero values, update with live metrics
    if accuracy == 0 and live_metrics["accuracy"] > 0:
        accuracy = live_metrics["accuracy"]
        f1 = live_metrics["f1"]
        precision = live_metrics["precision"]
        recall = live_metrics["recall"]
        # Persist the computed metrics
        valid_count = sum(1 for r in reviews if r.text and len(r.text.strip()) > 10)
        metadata.update({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "dataset_size": total,
            "dataset_cleaned": valid_count,
            "dataset_rejected": total - valid_count,
        })
        save_metadata(metadata)

    # Version and last retrain from metadata
    model_version = metadata.get("current_version", "v1.0")
    last_retrain = metadata.get("last_retrained", "Unknown")
    # Format datetime if ISO format
    try:
        dt = datetime.fromisoformat(last_retrain)
        last_retrain = dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        pass

    # Dataset stats from metadata
    dataset_cleaned = metadata.get("dataset_cleaned", 0)
    dataset_rejected = metadata.get("dataset_rejected", 0)

    # Sentiment distribution — histogram by star rating
    rating_counts = [0] * 5
    for r in reviews:
        idx = min(max(r.rating - 1, 0), 4)
        rating_counts[idx] += 1
    max_rc = max(rating_counts) or 1
    sentiment_dist = [round(c / max_rc, 3) for c in rating_counts]

    # Fake probability distribution
    fake_probs = [r.is_fake_prob for r in reviews if r.is_fake_prob is not None]
    if fake_probs:
        hist, _ = np.histogram(fake_probs, bins=10, range=(0, 1))
        max_h = max(hist) or 1
        fake_dist = [round(int(h) / max_h, 3) for h in hist]
    else:
        fake_dist = []

    # Comparison table
    comparison = [
        ComparisonRow(metric="Accuracy", current=f"{accuracy * 100:.1f}%", candidate="—", delta="—"),
        ComparisonRow(metric="Precision", current=f"{precision * 100:.1f}%", candidate="—", delta="—"),
        ComparisonRow(metric="Recall", current=f"{recall * 100:.1f}%", candidate="—", delta="—"),
        ComparisonRow(metric="F1 Score", current=f"{f1 * 100:.1f}%", candidate="—", delta="—"),
    ]

    return ModelMetricsResponse(
        model_version=model_version,
        accuracy=accuracy,
        f1_score=f1,
        precision=precision,
        recall=recall,
        last_retrain=last_retrain,
        total_reviews=total,
        dataset_cleaned=dataset_cleaned,
        dataset_rejected=dataset_rejected,
        sentiment_distribution=sentiment_dist,
        fake_probability=fake_dist,
        comparison=comparison,
    )


# ─────────────────────────────────────────────────────────────
# GET /api/drift-status  (+ legacy /api/drift alias)
# ─────────────────────────────────────────────────────────────

@router.get("/drift-status", response_model=DriftResponse)
def get_drift_status(db: Session = Depends(get_db)):
    """Multi-signal drift detection: sentiment, fake probability, vocabulary."""
    all_reviews = db.query(Review).all()
    recent_reviews = (
        db.query(Review)
        .order_by(desc(Review.created_at))
        .limit(100)
        .all()
    )

    drift = detect_drift(recent_reviews, all_reviews)
    health = compute_dataset_health(all_reviews)

    return DriftResponse(
        drift_level=drift["drift_level"],
        score=drift["score"],
        dataset_health=health,
        sentiment=DriftSignal(**drift["sentiment"]),
        fake_probability=DriftSignal(**drift["fake_probability"]),
        vocabulary=DriftSignal(**drift["vocabulary"]),
        recent_positive_pct=drift["sentiment"]["recent_pct"],
        overall_positive_pct=drift["sentiment"]["overall_pct"],
    )


@router.get("/drift", response_model=DriftResponse)
def get_drift_legacy(db: Session = Depends(get_db)):
    """Legacy alias for /api/drift-status."""
    return get_drift_status(db)


# ─────────────────────────────────────────────────────────────
# POST /api/retrain  (background task)
# ─────────────────────────────────────────────────────────────

def _run_retrain_background():
    """Run retrain in background with its own DB session."""
    db = SessionLocal()
    try:
        run_retrain(db)
    finally:
        db.close()


@router.post("/retrain", response_model=RetrainResponse)
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining as a background task."""
    status = get_retrain_status()
    if status["running"]:
        return RetrainResponse(
            message="Retraining already in progress.",
            status="running",
        )

    background_tasks.add_task(_run_retrain_background)

    return RetrainResponse(
        message="Retraining started in background. Poll /api/retrain-status for progress.",
        status="retraining_started",
    )


# ─────────────────────────────────────────────────────────────
# GET /api/retrain-status
# ─────────────────────────────────────────────────────────────

@router.get("/retrain-status", response_model=RetrainStatusResponse)
def retrain_status():
    """Poll the status of an ongoing or completed retrain."""
    status = get_retrain_status()
    return RetrainStatusResponse(
        running=status["running"],
        last_result=status["last_result"],
    )


# ─────────────────────────────────────────────────────────────
# GET /api/admin/reviews
# ─────────────────────────────────────────────────────────────

@router.get("/admin/reviews")
def get_admin_reviews(db: Session = Depends(get_db)):
    """Get all products with positive/negative counts and negative review details."""
    products = db.query(Product).order_by(Product.name).all()
    result = []

    for p in products:
        reviews = db.query(Review).filter(Review.product_id == p.id).all()
        if not reviews:
            continue

        positive = [r for r in reviews if r.sentiment == "Positive"]
        negative = [r for r in reviews if r.sentiment == "Negative"]

        neg_details = []
        for r in negative:
            neg_details.append({
                "id": r.id,
                "author": r.author,
                "rating": r.rating,
                "text": r.text,
                "priority": r.priority,
                "flag": r.flag,
                "date": str(r.date) if r.date else None,
            })

        result.append({
            "product_id": p.id,
            "product_name": p.name,
            "asin": p.asin,
            "total": len(reviews),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "negative_reviews": neg_details,
        })

    return result


# ─────────────────────────────────────────────────────────────
# GET /api/urgency-keywords
# ─────────────────────────────────────────────────────────────

@router.get("/urgency-keywords", response_model=UrgencyKeywordsResponse)
def get_urgency_keywords():
    """Get the current urgency keyword sets (base + learned)."""
    data = load_urgency_keywords()
    return UrgencyKeywordsResponse(
        base_keywords=UrgencyKeywordSet(**data["base_keywords"]),
        learned_keywords=UrgencyKeywordSet(**data["learned_keywords"]),
        last_updated=data.get("last_updated", "—"),
        total_keywords=data.get("total_keywords", 0),
    )
