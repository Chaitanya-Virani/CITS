"""
CITS — Review Routes
GET  /api/reviews/{product_id} — list reviews for a product + rating breakdown
POST /api/reviews              — submit a new review (runs ML model)
"""
import pickle
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db, SessionLocal
from app.models import Review, Product, ModelStore
from app.schemas import ReviewSubmit, ReviewOut, ReviewsResponse, ReviewResponse
from app.utils import clean_text, compute_fake_score
from ml.urgency_extraction import check_urgency_adaptive

router = APIRouter(prefix="/api", tags=["Reviews"])

# Use a singleton pattern for the model in memory
_model = None

def get_model():
    """Load the current active model from the DB if not in memory."""
    global _model
    if _model is not None:
        return _model
    
    db = SessionLocal()
    try:
        active_row = db.query(ModelStore).filter(ModelStore.is_active == 1).order_by(ModelStore.created_at.desc()).first()
        if not active_row:
            return None
        _model = pickle.loads(active_row.model_data)
        return _model
    finally:
        db.close()

def reload_model():
    """Force reload the model from the DB (e.g., after successful retraining)."""
    global _model
    _model = None
    return get_model()

@router.get("/reviews/{product_id}", response_model=ReviewsResponse)
def get_reviews(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    reviews = db.query(Review).filter(Review.product_id == product_id).order_by(desc(Review.created_at)).all()

    breakdown = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    items = []
    for r in reviews:
        breakdown[str(min(max(r.rating, 1), 5))] += 1
        items.append(ReviewOut(
            id=r.id, author=r.author, rating=r.rating, text=r.text,
            sentiment=r.sentiment, priority=r.priority, flag=r.flag,
            is_fake_prob=r.is_fake_prob or 0.0,
            date=str(r.date) if r.date else None,
        ))

    return ReviewsResponse(reviews=items, rating_breakdown=breakdown, total=len(reviews))

@router.post("/reviews", response_model=ReviewResponse)
def submit_review(data: ReviewSubmit, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == data.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    model = get_model()
    if not model:
        raise HTTPException(status_code=500, detail="ML model not found in database. Please run /api/retrain or seed the DB.")

    cleaned = clean_text(data.text)
    sentiment = "Positive" if model.predict([cleaned])[0] == 1 else "Negative"
    fake_prob = compute_fake_score(data.text, data.rating, sentiment, model, cleaned)

    # Use adaptive urgency detection
    urgency = check_urgency_adaptive(data.text)
    is_urgent = urgency["is_urgent"]
    priority = "High" if is_urgent else "Normal"
    flag = "🚨 URGENT" if is_urgent else "✅ OK"

    review = Review(
        product_id=data.product_id, author=data.author, rating=data.rating,
        text=data.text, sentiment=sentiment, priority=priority, flag=flag,
        is_fake_prob=fake_prob, date=date.today()
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    return ReviewResponse(sentiment=sentiment, priority=priority, flag=flag)
