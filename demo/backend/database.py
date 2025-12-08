"""Postgres-backed storage for demo interactions.

This uses SQLAlchemy and reads `DATABASE_URL` from the environment.
If `DATABASE_URL` is not set, it falls back to a local SQLite file
`sqlite:///./demo.db` to keep the demo runnable without Postgres.
"""
from datetime import datetime
import os
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./demo.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    # Attempt to create tables, retrying if the database service is not
    # yet accepting connections (common when containers start together).
    from sqlalchemy.exc import OperationalError
    import time
    import logging

    logger = logging.getLogger(__name__)

    max_retries = int(os.environ.get("DB_INIT_RETRIES", "20"))
    delay = float(os.environ.get("DB_INIT_DELAY", "1"))

    for attempt in range(1, max_retries + 1):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database initialized (tables created)")
            return
        except OperationalError as e:
            logger.warning(
                "Database not ready (attempt %d/%d): %s",
                attempt,
                max_retries,
                e,
            )
            time.sleep(delay)
            # backoff slightly
            delay = min(delay * 1.5, 5.0)

    # If we reach here, initialization failed
    logger.error("Could not initialize database after %d attempts", max_retries)
    raise RuntimeError("Failed to initialize database")


def add_interaction(prompt: str, response: str) -> dict:
    db: Session = SessionLocal()
    try:
        rec = Interaction(prompt=prompt, response=response)
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return {"id": rec.id, "prompt": rec.prompt, "response": rec.response, "created_at": rec.created_at.isoformat()}
    finally:
        db.close()


def get_interactions(limit: int = 100) -> list:
    db: Session = SessionLocal()
    try:
        rows = db.query(Interaction).order_by(Interaction.id.desc()).limit(limit).all()
        return [{"id": r.id, "prompt": r.prompt, "response": r.response, "created_at": r.created_at.isoformat()} for r in rows]
    finally:
        db.close()


# Initialize DB tables on import
init_db()

