"""
CITS — Customer Intelligent Trust Scoring
FastAPI Application Entry Point
"""
import os
import pickle
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.model_selection import train_test_split

from app.database import init_db, SessionLocal
from app.routes import products, reviews, summary, admin
from app.utils import clean_text
from ml.metrics_service import load_metadata, save_metadata
from ml.evaluate import evaluate_model

# --- App ---
app = FastAPI(
    title="Customer Intelligent Trust Scoring",
    version="1.0",
    description="AI-powered review analysis and trust scoring system"
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- Static Files ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Include Routers ---
app.include_router(products.router)
app.include_router(reviews.router)
app.include_router(summary.router)
app.include_router(admin.router)


# --- Startup ---
@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ Database initialized")

    # Initialize metadata with real metrics if needed
    _initialize_metadata()
    print("✅ Model metadata ready")
    print("✅ CITS API ready")


def _initialize_metadata():
    """Compute and persist initial model metrics if metadata has zero values."""
    metadata = load_metadata()
    if metadata.get("accuracy", 0) > 0:
        return  # Already initialized

    model_path = os.path.join(BASE_DIR, "model", "cits.pkl")
    if not os.path.exists(model_path):
        return

    db = SessionLocal()
    try:
        from app.models import Review
        reviews = db.query(Review).all()
        if len(reviews) < 20:
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        metrics = evaluate_model(model, X_test, y_test)
        valid_count = sum(1 for r in reviews if r.text and len(r.text.strip()) > 10)

        metadata.update({
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "dataset_size": len(reviews),
            "dataset_cleaned": valid_count,
            "dataset_rejected": len(reviews) - valid_count,
        })
        save_metadata(metadata)
        print(f"   📊 Initial metrics: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
    except Exception as e:
        print(f"   ⚠️ Could not initialize metrics: {e}")
    finally:
        db.close()


# --- Template Serving ---
def _serve_template(filename: str) -> HTMLResponse:
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the user-facing product page."""
    return _serve_template("user.html")


@app.get("/user", response_class=HTMLResponse)
def user_page():
    """Serve the user-facing product page."""
    return _serve_template("user.html")


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """Serve the admin dashboard."""
    return _serve_template("admin.html")


@app.get("/admin/reviews", response_class=HTMLResponse)
def admin_reviews_page():
    """Serve the admin reviews page."""
    return _serve_template("reviews.html")


# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "Online", "version": "1.0"}