"""
Database connection and session management.

Driver strategy
───────────────
We use psycopg2 (v2) everywhere because psycopg3's binary wheel is blocked
by Application Control policies on this machine and on most Render instances
that don't ship a pre-installed libpq.

URL rewriting rules applied at startup:
  postgres://...          → postgresql+psycopg2://...
  postgresql://...        → postgresql+psycopg2://...
  postgresql+psycopg://... → postgresql+psycopg2://...   (psycopg3 → psycopg2)

Neon-specific parameter stripping:
  Neon connection strings sometimes include channel_binding=require, which
  psycopg2 does not understand and will raise ProgrammingError on connect.
  We strip it (and any other unsupported query params) before building the engine.

Pool settings:
  Neon's serverless PostgreSQL uses a connection proxy that idles out quickly.
  pool_pre_ping=True ensures stale connections are detected and recycled rather
  than causing 500 errors on first use after a period of inactivity.
  pool_recycle=300 drops connections before Neon's ~5-minute idle timeout.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Unsupported query-string parameters for psycopg2 ─────────────────────────
# psycopg2 passes unknown params to libpq. Some are fine; these are not.
_STRIP_PARAMS = {"channel_binding"}


def _build_database_url() -> str:
    """
    Read DATABASE_URL, normalise the driver scheme, and strip any query-string
    parameters that psycopg2 / libpq do not understand.
    """
    raw = os.getenv("DATABASE_URL", "sqlite:///./ragchatbot.db")

    # Normalise scheme → psycopg2
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw.startswith("postgresql+psycopg://"):
        raw = raw.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+psycopg2://", 1)
    # sqlite or already correct scheme → leave as-is

    # Strip unsupported query-string parameters (psycopg2 / Neon incompatibility)
    if "postgresql" in raw and "?" in raw:
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in qs.items() if k not in _STRIP_PARAMS}
        clean_query = urlencode(filtered, doseq=True)
        parsed = parsed._replace(query=clean_query)
        raw = urlunparse(parsed)

    return raw


DATABASE_URL = _build_database_url()

# ── Engine ────────────────────────────────────────────────────────────────────
_is_sqlite = "sqlite" in DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    # SQLite requires this flag; PostgreSQL must not have it
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # Keep connections alive across Neon's idle-timeout window
    pool_pre_ping=True,
    pool_recycle=300,
    # Neon free tier: cap the pool so we don't exhaust the 10-connection limit
    pool_size=3 if not _is_sqlite else 5,
    max_overflow=2 if not _is_sqlite else 10,
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all SQLAlchemy-mapped tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a database session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    """Called on application startup to ensure all tables exist."""
    create_tables()
    print(f"Database initialised  ({DATABASE_URL.split('@')[-1].split('/')[0] if '@' in DATABASE_URL else 'sqlite'})")
