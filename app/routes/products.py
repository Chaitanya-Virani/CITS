"""
CITS — Product Routes
GET /api/products — list all products
GET /api/product/{product_id} — single product with computed trust score
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, Review
from app.schemas import ProductListItem, ProductResponse
from app.utils import compute_trust_score

router = APIRouter(prefix="/api", tags=["Products"])


@router.get("/products", response_model=list[ProductListItem])
def list_products(db: Session = Depends(get_db)):
    """List all products for dropdown/selector."""
    products = db.query(Product).order_by(Product.name).all()
    return products


@router.get("/product/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get single product with computed trust score, avg rating, total reviews."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    reviews = db.query(Review).filter(Review.product_id == product_id).all()
    total = len(reviews)
    avg_rating = sum(r.rating for r in reviews) / total if total > 0 else 0.0
    trust = compute_trust_score(reviews)

    return ProductResponse(
        id=product.id,
        asin=product.asin,
        name=product.name,
        category=product.category,
        trust_score=trust,
        overall_rating=round(avg_rating, 1),
        total_reviews=total,
    )
