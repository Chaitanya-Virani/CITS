"""
CITS — Database Configuration
Supports PostgreSQL (production/Render) and SQLite (local development).

Set DATABASE_URL environment variable for PostgreSQL:
    DATABASE_URL=postgresql://user:pass@host:5432/cits_db

If not set, falls back to SQLite at data/cits.db
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ── Connection URL ────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")

if DATABASE_URL:
    # Render uses "postgres://" but SQLAlchemy needs "postgresql://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Local development — SQLite
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    SQLITE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'cits.db')}"
    engine = create_engine(
        SQLITE_URL, 
        connect_args={"check_same_thread": False, "timeout": 30}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, auto-closes after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
