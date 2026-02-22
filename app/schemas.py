"""
CITS — Pydantic Schemas
Request/response models for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, Union
from datetime import date


# === Legacy (kept for backwards compat) ===

class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    sentiment: str
    priority: str
    flag: str


# === Product ===

class ProductListItem(BaseModel):
    id: int
    asin: str
    name: str
    category: str

    class Config:
        from_attributes = True

class ProductResponse(BaseModel):
    id: int
    asin: str
    name: str
    category: str
    trust_score: float = 0.0
    overall_rating: float = 0.0
    total_reviews: int = 0


# === Reviews ===

class ReviewSubmit(BaseModel):
    product_id: int
    author: str = Field(default="Anonymous", max_length=100)
    rating: int = Field(ge=1, le=5)
    text: str = Field(min_length=5)

class ReviewOut(BaseModel):
    id: int
    author: str
    rating: int
    text: str
    sentiment: str
    priority: str
    flag: str
    is_fake_prob: float
    date: Union[str, None] = None

    class Config:
        from_attributes = True

class ReviewsResponse(BaseModel):
    reviews: list[ReviewOut]
    rating_breakdown: dict[str, int]  # {"5": 120, "4": 80, ...}
    total: int


# === Summary ===

class SummaryResponse(BaseModel):
    summary: str
    positive_count: int = 0
    negative_count: int = 0
    top_keywords: list[str] = []


# === Admin — Model Metrics ===

class ComparisonRow(BaseModel):
    metric: str
    current: str
    candidate: str
    delta: str

class ModelMetricsResponse(BaseModel):
    model_version: str = "v1.0"
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    last_retrain: str = "—"
    total_reviews: int = 0
    dataset_cleaned: int = 0
    dataset_rejected: int = 0
    sentiment_distribution: list[float] = []
    fake_probability: list[float] = []
    comparison: list[ComparisonRow] = []


# === Admin — Drift ===

class DriftSignal(BaseModel):
    shift: float = 0.0
    recent_pct: Optional[float] = None
    overall_pct: Optional[float] = None
    recent_mean: Optional[float] = None
    overall_mean: Optional[float] = None
    new_token_count: Optional[int] = None
    total_recent_tokens: Optional[int] = None

class DriftResponse(BaseModel):
    drift_level: str = "Stable"  # Stable / Moderate Drift / High Drift
    score: float = 0.0
    dataset_health: float = 0.0
    sentiment: Optional[DriftSignal] = None
    fake_probability: Optional[DriftSignal] = None
    vocabulary: Optional[DriftSignal] = None
    # Legacy compat
    recent_positive_pct: float = 0.0
    overall_positive_pct: float = 0.0


# === Admin — Retrain ===

class RetrainResponse(BaseModel):
    message: str
    status: str = "retraining_started"  # retraining_started / success / rejected / error
    accuracy: float = 0.0
    f1_score: float = 0.0

class RetrainStatusResponse(BaseModel):
    running: bool = False
    last_result: Optional[dict] = None


# === Admin — Urgency Keywords ===

class UrgencyKeywordSet(BaseModel):
    english: list[str] = []
    hinglish: list[str] = []

class UrgencyKeywordsResponse(BaseModel):
    base_keywords: UrgencyKeywordSet
    learned_keywords: UrgencyKeywordSet
    last_updated: str = "—"
    total_keywords: int = 0