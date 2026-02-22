"""
CITS — Review Routes
GET  /api/reviews/{product_id} — list reviews for a product + rating breakdown
POST /api/reviews              — submit a new review (runs ML model)
"""
import pickle
import os
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import Review, Product
from app.schemas import ReviewSubmit, ReviewOut, ReviewsResponse, ReviewResponse
from app.utils import clean_text, check_urgency

router = APIRouter(prefix="/api", tags=["Reviews"])

# Load model at module level
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cits.pkl")
_model = None


def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def reload_model():
    """Force reload model from disk (called after retrain)."""
    global _model
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    return _model


@router.get("/reviews/{product_id}", response_model=ReviewsResponse)
def get_reviews(product_id: int, db: Session = Depends(get_db)):
    """Get all reviews for a product with rating breakdown."""
    # Verify product exists
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    reviews = (
        db.query(Review)
        .filter(Review.product_id == product_id)
        .order_by(desc(Review.created_at))
        .all()
    )

    # Rating breakdown: count per star
    breakdown = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    for r in reviews:
        key = str(min(max(r.rating, 1), 5))
        breakdown[key] += 1

    review_items = []
    for r in reviews:
        review_items.append(ReviewOut(
            id=r.id,
            author=r.author,
            rating=r.rating,
            text=r.text,
            sentiment=r.sentiment,
            priority=r.priority,
            flag=r.flag,
            is_fake_prob=r.is_fake_prob or 0.0,
            date=str(r.date) if r.date else None,
        ))

    return ReviewsResponse(
        reviews=review_items,
        rating_breakdown=breakdown,
        total=len(reviews),
    )


@router.post("/reviews", response_model=ReviewResponse)
def submit_review(data: ReviewSubmit, db: Session = Depends(get_db)):
    """Submit a new review — runs through ML model, saves to DB."""
    # Verify product exists
    product = db.query(Product).filter(Product.id == data.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    model = get_model()
    if not model:
        raise HTTPException(status_code=500, detail="ML model not available")

    # ML inference
    cleaned = clean_text(data.text)
    prediction = model.predict([cleaned])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Fake probability
    try:
        proba = model.predict_proba([cleaned])[0]
        fake_prob = round(float(min(proba)), 4)
    except Exception:
        fake_prob = 0.0

    # Urgency check
    urgency = check_urgency(data.text)
    is_urgent = urgency["is_urgent"]
    priority = "High" if is_urgent else "Normal"
    flag = "🚨 URGENT" if is_urgent else "✅ OK"

    # Save to DB
    review = Review(
        product_id=data.product_id,
        author=data.author,
        rating=data.rating,
        text=data.text,
        sentiment=sentiment,
        priority=priority,
        flag=flag,
        is_fake_prob=fake_prob,
        date=date.today(),
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    return ReviewResponse(
        sentiment=sentiment,
        priority=priority,
        flag=flag,
    )
