"""
CITS — Metrics Service
Persistent model metadata storage using PostgreSQL / SQLite via SQLAlchemy.
Single source of truth for model version, accuracy, f1, timestamps, etc.
"""
from datetime import datetime
from app.database import SessionLocal
from app.models import MLMetadata, RetrainHistory


# ── Default metadata template ──────────────────────────────────
DEFAULT_METADATA = {
    "current_version": "v1.0",
    "accuracy": 0.0,
    "f1_score": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "last_retrained": "2026-02-16T23:17:00",
    "dataset_size": 0,
    "dataset_cleaned": 0,
    "dataset_rejected": 0,
}


def _get_session():
    """Get a fresh DB session."""
    return SessionLocal()


def load_metadata() -> dict:
    """Load model metadata from the ml_metadata table."""
    db = _get_session()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "model_metadata").first()
        if row is None:
            save_metadata(DEFAULT_METADATA)
            return DEFAULT_METADATA.copy()
        return row.value
    finally:
        db.close()


def save_metadata(data: dict) -> None:
    """Save model metadata to the ml_metadata table."""
    db = _get_session()
    try:
        row = db.query(MLMetadata).filter(MLMetadata.key == "model_metadata").first()
        if row:
            row.value = data
            row.updated_at = datetime.utcnow()
        else:
            row = MLMetadata(key="model_metadata", value=data)
            db.add(row)
        db.commit()
    finally:
        db.close()


def get_next_version(current_version: str) -> str:
    """
    Auto-increment version string.
    v1.0 → v1.1, v1.9 → v1.10, v2.3 → v2.4
    """
    try:
        version = current_version.lstrip("v")
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return f"v{major}.{minor + 1}"
    except (ValueError, IndexError):
        return "v1.1"


def log_retrain_attempt(entry: dict) -> None:
    """Append a retrain attempt to the retrain_history table."""
    db = _get_session()
    try:
        record = RetrainHistory(
            status=entry.get("status", "unknown"),
            old_version=entry.get("old_version"),
            new_version=entry.get("new_version"),
            old_metrics=entry.get("old_metrics"),
            new_metrics=entry.get("new_metrics"),
            dataset_size=entry.get("dataset_size"),
            new_reviews_since_last=entry.get("new_reviews_since_last"),
            message=entry.get("message"),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def load_retrain_history() -> list:
    """Load the retrain history log from the DB."""
    db = _get_session()
    try:
        rows = db.query(RetrainHistory).order_by(RetrainHistory.created_at.asc()).all()
        return [
            {
                "status": r.status,
                "old_version": r.old_version,
                "new_version": r.new_version,
                "old_metrics": r.old_metrics,
                "new_metrics": r.new_metrics,
                "dataset_size": r.dataset_size,
                "new_reviews_since_last": r.new_reviews_since_last,
                "message": r.message,
                "timestamp": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()
