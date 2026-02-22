"""
CITS — ORM Models
Product, Review, and ML storage tables
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Text, Date, DateTime, ForeignKey, LargeBinary, JSON
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


# ════════════════════════════════════════════════════════════════
# ML STORAGE TABLES
# ════════════════════════════════════════════════════════════════

class MLMetadata(Base):
    """Key-value store for ML metadata (model info, urgency keywords, drift baseline)."""
    __tablename__ = "ml_metadata"

    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False)  # Stores any JSON-serializable data
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MLMetadata {self.key}>"


class RetrainHistory(Base):
    """Structured log of all retrain attempts."""
    __tablename__ = "retrain_history"

    id = Column(Integer, primary_key=True)
    status = Column(String(20), nullable=False)   # success / rejected / error / skipped
    old_version = Column(String(20))
    new_version = Column(String(20))
    old_metrics = Column(JSON)
    new_metrics = Column(JSON)
    dataset_size = Column(Integer)
    new_reviews_since_last = Column(Integer)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RetrainHistory {self.id}: {self.status}>"


class ModelStore(Base):
    """Binary storage for trained ML models."""
    __tablename__ = "model_store"

    id = Column(Integer, primary_key=True)
    version = Column(String(20), nullable=False, index=True)
    model_data = Column(LargeBinary, nullable=False)  # Pickled model bytes
    is_active = Column(Integer, default=1)  # 1 = current production model
    accuracy = Column(Float)
    f1_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelStore {self.version} active={self.is_active}>"
