"""
CITS — Metrics Service
Persistent model metadata storage using JSON.
Single source of truth for model version, accuracy, f1, timestamps, etc.
"""
import os
import json
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "database", "model_metadata.json")
HISTORY_PATH = os.path.join(BASE_DIR, "database", "retrain_history.json")

_lock = threading.Lock()

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


def load_metadata() -> dict:
    """Load model metadata from JSON file."""
    with _lock:
        if not os.path.exists(METADATA_PATH):
            save_metadata(DEFAULT_METADATA)
            return DEFAULT_METADATA.copy()
        with open(METADATA_PATH, "r") as f:
            return json.load(f)


def save_metadata(data: dict) -> None:
    """Save model metadata to JSON file."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with _lock:
        with open(METADATA_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)


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
    """Append a retrain attempt to the history log."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with _lock:
        history = []
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []

        entry["timestamp"] = datetime.utcnow().isoformat()
        history.append(entry)

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2, default=str)


def load_retrain_history() -> list:
    """Load the retrain history log."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
