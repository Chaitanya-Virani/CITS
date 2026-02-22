"""
CITS — ORM Models
Product and Review tables
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Text, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), default="General")
    created_at = Column(DateTime, default=datetime.utcnow)

    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product {self.asin}: {self.name}>"


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    author = Column(String(100), default="Anonymous")
    rating = Column(Integer, nullable=False)  # 1-5
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), default="Unknown")  # Positive / Negative
    priority = Column(String(20), default="Normal")     # High / Normal
    flag = Column(String(20), default="OK")             # URGENT / OK
    is_fake_prob = Column(Float, default=0.0)           # 0.0 to 1.0
    date = Column(Date, default=date.today)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="reviews")

    def __repr__(self):
        return f"<Review {self.id}: {self.sentiment} ({self.rating}★)>"
