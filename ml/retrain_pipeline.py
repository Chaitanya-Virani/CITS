"""
CITS — Retraining Pipeline
Fully separated retraining module with PostgreSQL / ModelStore support.
"""
import pickle
import numpy as np
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.database import SessionLocal
from app.models import ModelStore, Review
from app.utils import clean_text
from ml.evaluate import evaluate_model
from ml.metrics_service import (
    load_metadata,
    save_metadata,
    get_next_version,
    log_retrain_attempt,
)
from ml.urgency_extraction import extract_and_update_keywords
from ml.drift_detection import detect_drift, save_baseline

# Minimum new reviews required since last retrain to allow retraining
MIN_NEW_REVIEWS = 50

# Global status tracker for async retrain polling
_retrain_status = {"running": False, "last_result": None}

def get_retrain_status() -> dict:
    return _retrain_status.copy()

def _load_current_model_binary():
    """Load the current active model binary from the database."""
    db = SessionLocal()
    try:
        row = db.query(ModelStore).filter(ModelStore.is_active == 1).order_by(ModelStore.created_at.desc()).first()
        if row:
            return pickle.loads(row.model_data)
        return None
    finally:
        db.close()

def _save_model_to_db(model, version, metrics):
    """Save model to the ModelStore table and set as active."""
    db = SessionLocal()
    try:
        # Deactivate previous active models
        db.query(ModelStore).filter(ModelStore.is_active == 1).update({"is_active": 0})
        
        # Add new model
        record = ModelStore(
            version=version,
            model_data=pickle.dumps(model),
            is_active=1,
            accuracy=metrics.get("accuracy"),
            f1_score=metrics.get("f1")
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

def _cross_validate_f1(model, texts, labels, n_folds=5) -> dict:
    X, y = np.array(texts), np.array(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    f_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}
    for _, test_idx in skf.split(X, y):
        m = evaluate_model(model, X[test_idx].tolist(), y[test_idx].tolist())
        for k in f_metrics: f_metrics[k].append(m[k])
    return {k: round(float(np.mean(v)), 4) for k, v in f_metrics.items()} | {"f1_std": round(float(np.std(f_metrics["f1"])), 4)}

def run_retrain(db_session) -> dict:
    global _retrain_status
    _retrain_status["running"] = True
    try:
        reviews = db_session.query(Review).all()
        total = len(reviews)
        if total < 50: return {"status": "error", "message": f"Need \u226550 reviews, have {total}."}

        metadata = load_metadata()
        last_size = metadata.get("dataset_size", 0)
        new_revs = total - last_size
        if new_revs < MIN_NEW_REVIEWS and last_size > 0:
            return {"status": "skipped", "message": f"Need \u226550 new reviews, only {new_revs} added."}

        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]
        
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.2)

        candidate = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        candidate.fit(X_train, y_train)

        new_metrics = _cross_validate_f1(candidate, texts, labels)
        current_model = _load_current_model_binary()
        old_metrics = _cross_validate_f1(current_model, texts, labels) if current_model else {"f1": 0.0}
        
        old_version = metadata.get("current_version", "v1.0")
        if new_metrics["f1"] >= old_metrics.get("f1", 0):
            new_version = get_next_version(old_version)
            _save_model_to_db(candidate, new_version, new_metrics)
            
            # Reload model in-memory for the API
            try:
                from app.routes.reviews import reload_model
                reload_model()
            except: pass

            metadata.update({
                "current_version": new_version, "accuracy": new_metrics["accuracy"],
                "f1_score": new_metrics["f1"], "precision": new_metrics["precision"],
                "recall": new_metrics["recall"], "last_retrained": datetime.utcnow().isoformat(),
                "dataset_size": total, "dataset_cleaned": sum(1 for r in reviews if len(str(r.text)) > 10),
                "dataset_rejected": total - sum(1 for r in reviews if len(str(r.text)) > 10),
            })
            save_metadata(metadata)
            
            # Drift & Keywords
            from sqlalchemy import desc
            recent = db_session.query(Review).order_by(desc(Review.created_at)).limit(100).all()
            drift_level = detect_drift(recent, reviews).get("drift_level", "Stable")
            extract_and_update_keywords(reviews, drift_level)
            save_baseline(reviews)

            res = {"status": "success", "message": f"Upgraded {old_version} \u2192 {new_version}. F1: {old_metrics.get('f1',0):.4f} \u2192 {new_metrics['f1']:.4f}"}
        else:
            res = {"status": "rejected", "message": f"F1 did not improve: current {old_metrics.get('f1',0):.4f} vs {new_metrics['f1']:.4f}"}

        log_retrain_attempt(res | {"old_version": old_version, "new_version": res.get("version"), "old_metrics": old_metrics, "new_metrics": new_metrics, "dataset_size": total, "new_reviews_since_last": new_revs})
        _retrain_status["last_result"] = res | {"new_metrics": new_metrics, "old_metrics": old_metrics, "accuracy": new_metrics["accuracy"], "f1_score": new_metrics["f1"], "version": res.get("version", old_version)}
        return _retrain_status["last_result"]
    except Exception as e:
        err = {"status": "error", "message": f"Retrain failed: {str(e)}"}
        log_retrain_attempt(err)
        _retrain_status["last_result"] = err
        return err
    finally:
        _retrain_status["running"] = False
