import os
import pickle
import threading
import urllib.request
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.model_selection import train_test_split

from app.database import init_db, SessionLocal
from app.routes import products, reviews, summary, admin
from app.utils import clean_text
from ml.metrics_service import load_metadata, save_metadata
from ml.evaluate import evaluate_model


# --- Keep-Alive Self-Ping (prevents Render free-tier sleep) ---
def _keep_alive():
    """Ping own /health endpoint every 10 minutes to prevent Render sleep."""
    import time
    url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not url:
        print("ℹ️ RENDER_EXTERNAL_URL not set, skipping keep-alive")
        return
    
    health_url = f"{url}/health"
    print(f"🏓 Keep-alive started → target: {health_url}")
    
    # Ping once immediately to verify
    try:
        with urllib.request.urlopen(health_url, timeout=15) as r:
            print(f"✅ Initial keep-alive ping successful (Status: {r.getcode()})")
    except Exception as e:
        print(f"⚠️ Initial keep-alive ping failed: {e}")

    while True:
        time.sleep(600)  # 10 minutes
        try:
            # Bypass cache with timestamp
            ts_url = f"{health_url}?t={int(time.time())}"
            with urllib.request.urlopen(ts_url, timeout=15) as r:
                if r.getcode() == 200:
                    print(f"🏓 Periodic keep-alive successful ({time.strftime('%H:%M:%S')})")
        except Exception as e:
            print(f"⚠️ Periodic keep-alive failed: {e}")


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

    # Important: Model initialization for first-run or cloud environments
    _initialize_model_store()

    # Initialize metadata & drift
    _initialize_metadata()
    _initialize_drift_baseline()

    # Start keep-alive background thread
    threading.Thread(target=_keep_alive, daemon=True).start()

    print("✅ CITS API ready")


def _initialize_model_store():
    """Ensure the ModelStore has an active model. Seeks from disk if DB is empty."""
    db = SessionLocal()
    try:
        from app.models import ModelStore
        if db.query(ModelStore).count() == 0:
            model_path = os.path.join(BASE_DIR, "model", "cits.pkl")
            if os.path.exists(model_path):
                print("   🤖 Seeding initial model from disk to DB...")
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                
                initial_model = ModelStore(
                    version="v1.0",
                    model_data=model_bytes,
                    is_active=1,
                    accuracy=0.94,
                    f1_score=0.95
                )
                db.add(initial_model)
                db.commit()
                print("   ✅ Initial model ready")
    except Exception as e:
        print(f"   ⚠️ Could not initialize ModelStore: {e}")
    finally:
        db.close()


def _initialize_drift_baseline():
    """Save initial drift baseline if none exists."""
    from ml.drift_detection import load_baseline, save_baseline
    if load_baseline() is not None:
        return
    db = SessionLocal()
    try:
        from app.models import Review
        reviews = db.query(Review).all()
        if reviews:
            save_baseline(reviews)
            print("   📈 Drift baseline initialized")
    except Exception:
        pass
    finally:
        db.close()


def _initialize_metadata():
    """Compute and persist initial model metrics if metadata has zero values."""
    metadata = load_metadata()
    if metadata.get("accuracy", 0) > 0:
        return

    from app.routes.reviews import get_model
    model = get_model()
    if not model:
        return

    db = SessionLocal()
    try:
        from app.models import Review
        reviews = db.query(Review).all()
        if len(reviews) < 20:
            return

        texts = [clean_text(r.text) for r in reviews]
        labels = [1 if r.rating > 3 else 0 for r in reviews]
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2)
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
        print(f"   📊 Initial stats: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
    except Exception as e:
        print(f"   ⚠️ Metadata init error: {e}")
    finally:
        db.close()


# --- Template Serving ---
def _serve_template(filename: str) -> HTMLResponse:
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/", response_class=HTMLResponse)
def home(): return _serve_template("user.html")

@app.get("/user", response_class=HTMLResponse)
def user_page(): return _serve_template("user.html")

@app.get("/admin", response_class=HTMLResponse)
def admin_page(): return _serve_template("admin.html")

@app.get("/admin/reviews", response_class=HTMLResponse)
def admin_reviews_page(): return _serve_template("reviews.html")

@app.get("/health")
def health(): return {"status": "Online", "version": "1.0"}
