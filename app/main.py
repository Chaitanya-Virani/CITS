"""
CITS — Customer Intelligent Trust Scoring
FastAPI Application Entry Point
"""
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.database import init_db
from app.routes import products, reviews, summary, admin

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
    print("✅ CITS API ready")


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