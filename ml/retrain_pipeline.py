"""
CITS — Retraining Pipeline
Fully separated retraining module.

Workflow:
1. Load reviews from database
2. Preprocess + vectorize (TF-IDF)
3. Train new LogisticRegression pipeline
4. Evaluate candidate vs. current model
5. Replace model ONLY if candidate F1 ≥ current F1
6. Backup old model, save new model, update metadata
7. Log attempt to retrain history
"""
import os
import pickle
import shutil
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.utils import clean_text
from ml.evaluate import evaluate_model
from ml.metrics_service import (
    load_metadata,
    save_metadata,
    get_next_version,
    log_retrain_attempt,
)
from ml.urgency_extraction import extract_and_update_keywords

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cits.pkl")

# Global status tracker for async retrain polling
_retrain_status = {
    "running": False,
    "last_result": None,
}


def get_retrain_status() -> dict:
    """Get the current retraining status for polling."""
    return _retrain_status.copy()


def _load_current_model():
    """Load the current production model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _backup_model(version: str) -> str:
    """Backup the current model as model_vX.Y.pkl. Returns backup path."""
    backup_name = f"cits_{version}.pkl"
    backup_path = os.path.join(MODEL_DIR, backup_name)
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, backup_path)
    return backup_path


def _save_model(model) -> None:
    """Save model to the production path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def run_retrain(db_session) -> dict:
    """
    Full retraining pipeline.
    Returns a dict with: status, message, old_metrics, new_metrics, version.
    """
    global _retrain_status
    _retrain_status["running"] = True
    _retrain_status["last_result"] = None

    try:
        from app.models import Review

        # ── 1. Load data from DB ──────────────────────────────────
        reviews = db_session.query(Review).all()
        total = len(reviews)

        if total < 50:
            result = {
                "status": "error",
                "message": f"Not enough data to retrain. Need ≥50 reviews, have {total}.",
            }
            _retrain_status["last_result"] = result
            return result

        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]

        # Dataset health stats
        valid_count = sum(1 for r in reviews if r.text and len(r.text.strip()) > 10)
        rejected_count = total - valid_count

        # ── 2. Train/test split ───────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # ── 3. Train candidate model ─────────────────────────────
        candidate = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        candidate.fit(X_train, y_train)

        # ── 4. Evaluate candidate ─────────────────────────────────
        new_metrics = evaluate_model(candidate, X_test, y_test)

        # ── 5. Evaluate current model on same test set ────────────
        current_model = _load_current_model()
        old_metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        if current_model is not None:
            try:
                old_metrics = evaluate_model(current_model, X_test, y_test)
            except Exception:
                pass  # Current model incompatible, treat as baseline 0

        # ── 6. Compare — only replace if F1 improves ──────────────
        metadata = load_metadata()
        old_version = metadata.get("current_version", "v1.0")

        if new_metrics["f1"] >= old_metrics["f1"]:
            # Backup old model
            _backup_model(old_version)

            # Save new model
            _save_model(candidate)

            # Update version
            new_version = get_next_version(old_version)

            # Reload model in reviews module
            try:
                from app.routes.reviews import reload_model
                reload_model()
            except Exception:
                pass

            # ── 7. Update metadata ────────────────────────────────
            metadata.update({
                "current_version": new_version,
                "accuracy": new_metrics["accuracy"],
                "f1_score": new_metrics["f1"],
                "precision": new_metrics["precision"],
                "recall": new_metrics["recall"],
                "last_retrained": datetime.utcnow().isoformat(),
                "dataset_size": total,
                "dataset_cleaned": valid_count,
                "dataset_rejected": rejected_count,
            })
            save_metadata(metadata)

            # ── 7b. Update urgency keywords ───────────────────────
            try:
                from ml.drift_detection import detect_drift
                from sqlalchemy import desc
                recent = db_session.query(Review).order_by(desc(Review.created_at)).limit(100).all()
                drift = detect_drift(recent, reviews)
                drift_level = drift["drift_level"]
            except Exception:
                drift_level = "Stable"

            extract_and_update_keywords(reviews, drift_level)

            status = "success"
            message = (
                f"Model upgraded {old_version} → {new_version}. "
                f"F1: {old_metrics['f1']:.4f} → {new_metrics['f1']:.4f}"
            )
        else:
            # Reject — keep old model
            status = "rejected"
            new_version = old_version
            message = (
                f"Candidate rejected. F1 did not improve: "
                f"current {old_metrics['f1']:.4f} vs candidate {new_metrics['f1']:.4f}"
            )

        # ── 8. Log retrain attempt ────────────────────────────────
        log_retrain_attempt({
            "status": status,
            "old_version": old_version,
            "new_version": new_version if status == "success" else None,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "dataset_size": total,
            "message": message,
        })

        result = {
            "status": status,
            "message": message,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "version": new_version if status == "success" else old_version,
            "accuracy": new_metrics["accuracy"],
            "f1_score": new_metrics["f1"],
        }
        _retrain_status["last_result"] = result
        return result

    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Retraining failed: {str(e)}",
        }
        log_retrain_attempt({
            "status": "error",
            "message": str(e),
        })
        _retrain_status["last_result"] = error_result
        return error_result

    finally:
        _retrain_status["running"] = False
