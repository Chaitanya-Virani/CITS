"""
CITS — Model Evaluation
Computes sklearn metrics (accuracy, f1, precision, recall) on a test set.
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test: list[str], y_test: list[int]) -> dict:
    """
    Evaluate a trained model on a test set.
    Returns dict with accuracy, f1, precision, recall (all rounded to 4 decimals).
    """
    if not X_test or not y_test:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    predictions = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
    }
