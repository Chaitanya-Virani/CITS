"""
CITS — Summary Route
GET /api/summary/{product_id} — AI-generated summary of product reviews
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, Review
from app.schemas import SummaryResponse
from app.utils import generate_summary

router = APIRouter(prefix="/api", tags=["Summary"])


@router.get("/summary/{product_id}", response_model=SummaryResponse)
def get_summary(product_id: int, db: Session = Depends(get_db)):
    """Generate a rule-based AI summary for a product's reviews."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    reviews = db.query(Review).filter(Review.product_id == product_id).all()
    result = generate_summary(reviews)

    return SummaryResponse(
        summary=result["summary"],
        positive_count=result["positive_count"],
        negative_count=result["negative_count"],
        top_keywords=result["top_keywords"],
    )
