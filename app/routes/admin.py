"""
CITS — Admin Routes
Admin endpoints for monitoring model performance, drift, and triggering retrains.
All persistence now uses PostgreSQL via SQLAlchemy.
"""
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db, SessionLocal
from app.models import Review, Product
from app.schemas import (
    ModelMetricsResponse, ComparisonRow,
    DriftResponse, DriftSignal,
    RetrainResponse, RetrainStatusResponse,
    UrgencyKeywordsResponse, UrgencyKeywordSet,
    ReviewResponse, 
)
from app.utils import clean_text, compute_dataset_health
from ml.evaluate import evaluate_model
from ml.metrics_service import load_metadata, save_metadata, load_retrain_history
from ml.drift_detection import detect_drift
from ml.retrain_pipeline import run_retrain, get_retrain_status
from ml.urgency_extraction import load_urgency_keywords
from app.routes.reviews import get_model
from app.seed import seed as run_seed_logic

router = APIRouter(prefix="/api", tags=["Admin"])

@router.get("/model-metrics", response_model=ModelMetricsResponse)
def get_model_metrics(db: Session = Depends(get_db)):
    model = get_model()
    reviews = db.query(Review).all()
    total = len(reviews)
    metadata = load_metadata()

    if total >= 20 and model:
        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2)
        live = evaluate_model(model, X_test, y_test)
    else:
        live = {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    accuracy = metadata.get("accuracy") or live["accuracy"]
    f1 = metadata.get("f1_score") or live["f1"]
    precision = metadata.get("precision") or live["precision"]
    recall = metadata.get("recall") or live["recall"]

    if accuracy == 0 and live["accuracy"] > 0:
        accuracy, f1, precision, recall = live["accuracy"], live["f1"], live["precision"], live["recall"]
        valid_count = sum(1 for r in reviews if r.text and len(r.text.strip()) > 10)
        metadata.update({"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall, "dataset_size": total, "dataset_cleaned": valid_count, "dataset_rejected": total-valid_count})
        save_metadata(metadata)

    last_retrain = metadata.get("last_retrained", "Unknown")
    try:
        last_retrain = datetime.fromisoformat(last_retrain).strftime("%Y-%m-%d %H:%M")
    except: pass

    rating_counts = [0] * 5
    for r in reviews: rating_counts[min(max(r.rating - 1, 0), 4)] += 1
    max_rc = max(rating_counts) or 1
    sentiment_dist = [round(c / max_rc, 3) for c in rating_counts]

    fake_probs = [r.is_fake_prob for r in reviews if r.is_fake_prob is not None]
    if fake_probs:
        hist, _ = np.histogram(fake_probs, bins=10, range=(0, 1))
        max_h = max(hist) or 1
        fake_dist = [round(int(h) / max_h, 3) for h in hist]
    else:
        fake_dist = []

    history = load_retrain_history()
    last_success = next((e for e in reversed(history) if e.get("status") == "success" and e.get("old_metrics")), None)

    def _delta(ov, nv):
        d = (nv - ov) * 100
        return f"▲ +{d:.1f}%" if d > 0 else f"▼ {d:.1f}%" if d < 0 else "—"

    if last_success:
        om, nm = last_success["old_metrics"], last_success["new_metrics"]
        comparison = [
            ComparisonRow(metric="Accuracy", current=f"{accuracy*100:.1f}%", candidate=f"{om.get('accuracy',0)*100:.1f}%", delta=_delta(om.get('accuracy',0), nm.get('accuracy',0))),
            ComparisonRow(metric="Precision", current=f"{precision*100:.1f}%", candidate=f"{om.get('precision',0)*100:.1f}%", delta=_delta(om.get('precision',0), nm.get('precision',0))),
            ComparisonRow(metric="Recall", current=f"{recall*100:.1f}%", candidate=f"{om.get('recall',0)*100:.1f}%", delta=_delta(om.get('recall',0), nm.get('recall',0))),
            ComparisonRow(metric="F1 Score", current=f"{f1*100:.1f}%", candidate=f"{om.get('f1',0)*100:.1f}%", delta=_delta(om.get('f1',0), nm.get('f1',0))),
        ]
    else:
        comparison = [ComparisonRow(metric=m, current=f"{v*100:.1f}%", candidate="—", delta="—") for m,v in [("Accuracy", accuracy), ("Precision", precision), ("Recall", recall), ("F1 Score", f1)]]

    return ModelMetricsResponse(model_version=metadata.get("current_version", "v1.0"), accuracy=accuracy, f1_score=f1, precision=precision, recall=recall, last_retrain=last_retrain, total_reviews=total, dataset_cleaned=metadata.get("dataset_cleaned", 0), dataset_rejected=metadata.get("dataset_rejected", 0), sentiment_distribution=sentiment_dist, fake_probability=fake_dist, comparison=comparison)

@router.get("/drift-status", response_model=DriftResponse)
def get_drift_status(db: Session = Depends(get_db)):
    all_revs = db.query(Review).all()
    recent = db.query(Review).order_by(desc(Review.created_at)).limit(100).all()
    drift = detect_drift(recent, all_revs)
    return DriftResponse(drift_level=drift["drift_level"], score=drift["score"], dataset_health=compute_dataset_health(all_revs), sentiment=DriftSignal(**drift["sentiment"]), fake_probability=DriftSignal(**drift["fake_probability"]), vocabulary=DriftSignal(**drift["vocabulary"]), recent_positive_pct=drift["sentiment"]["recent_pct"], overall_positive_pct=drift["sentiment"]["overall_pct"])

@router.get("/drift", response_model=DriftResponse)
def get_drift_legacy(db: Session = Depends(get_db)): return get_drift_status(db)

def _retrain_task():
    db = SessionLocal()
    try: run_retrain(db)
    finally: db.close()

def _seed_task():
    """Run seeder in background."""
    try: run_seed_logic()
    except Exception as e: print(f"❌ Initial seed failed: {e}")

@router.post("/seed")
def seed_database(background_tasks: BackgroundTasks):
    """Trigger the record seeder as a background task (for Render free tier)."""
    background_tasks.add_task(_seed_task)
    return {"message": "Database seeding started in background.", "status": "seeding_started"}

@router.post("/retrain", response_model=RetrainResponse)
def retrain_model(background_tasks: BackgroundTasks):
    if get_retrain_status()["running"]: return RetrainResponse(message="Retraining already in progress.", status="running")
    background_tasks.add_task(_retrain_task)
    return RetrainResponse(message="Retraining started. Poll /api/retrain-status for progress.", status="retraining_started")

@router.get("/retrain-status", response_model=RetrainStatusResponse)
def retrain_status_poll():
    s = get_retrain_status()
    return RetrainStatusResponse(running=s["running"], last_result=s["last_result"])

@router.get("/admin/reviews")
def get_admin_reviews(db: Session = Depends(get_db)):
    res = []
    for p in db.query(Product).order_by(Product.name).all():
        revs = db.query(Review).filter(Review.product_id == p.id).all()
        if not revs: continue
        neg = [r for r in revs if r.sentiment == "Negative"]
        res.append({"product_id": p.id, "product_name": p.name, "asin": p.asin, "total": len(revs), "positive_count": len(revs)-len(neg), "negative_count": len(neg), "negative_reviews": [{"id": r.id, "author": r.author, "rating": r.rating, "text": r.text, "priority": r.priority, "flag": r.flag, "date": str(r.date)} for r in neg]})
    return res

@router.get("/urgency-keywords", response_model=UrgencyKeywordsResponse)
def get_urgency_keywords():
    d = load_urgency_keywords()
    return UrgencyKeywordsResponse(base_keywords=UrgencyKeywordSet(**d["base_keywords"]), learned_keywords=UrgencyKeywordSet(**d["learned_keywords"]), last_updated=d.get("last_updated", "—"), total_keywords=d.get("total_keywords", 0))
