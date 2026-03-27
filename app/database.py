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

# ── Load .env for local development (no-op on Render) ────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Connection URL ────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")

if DATABASE_URL:
    # Render uses "postgres://" but SQLAlchemy needs "postgresql://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,           # Max persistent connections
        max_overflow=10,       # Extra connections under load
        pool_timeout=30,       # Seconds to wait for a connection
        pool_recycle=300,      # Recycle connections every 5 min (prevents stale)
    )
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

    # Enable WAL mode for better concurrency with background tasks
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

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
