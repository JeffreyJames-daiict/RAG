"""Database migration helper — ensures all ORM tables, schemas, and columns exist on PostgreSQL / SQLite."""

from __future__ import annotations
import logging
from sqlalchemy import event, text

from rag.core.database import DatabaseManager
from rag.core.db_models import Base

logger = logging.getLogger(__name__)


def _prepare_schemas(engine) -> None:
    """Create the 'raw' schema/database and ensure column migrations depending on the dialect.

    - **PostgreSQL**: ``CREATE SCHEMA IF NOT EXISTS raw;`` + column alterations.
    - **SQLite**: ``ATTACH DATABASE ':memory:' AS raw;``
    """
    dialect = engine.dialect.name

    if dialect == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))
            # Column migrations for existing tables
            conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'document_versions') THEN
                        ALTER TABLE public.document_versions ADD COLUMN IF NOT EXISTS metadata JSON NOT NULL DEFAULT '{}'::json;
                    END IF;
                END $$;
            """))
            conn.commit()
        logger.info("Ensured PostgreSQL schema 'raw' and document_versions columns exist.")

    elif dialect == "sqlite":
        # Ensure raw database is attached on existing raw connection
        raw_conn = engine.raw_connection()
        try:
            raw_conn.execute("ATTACH DATABASE ':memory:' AS raw")
        except Exception:
            pass  # Already attached
        finally:
            raw_conn.close()


def ensure_raw_schema(engine) -> None:
    """Public helper to prepare schemas independently."""
    _prepare_schemas(engine)


def run_migrations(db_manager: DatabaseManager) -> None:
    """Create all schemas and tables defined in the ORM Base metadata.

    This is a lightweight, idempotent 'create if not exists' migration.

    Args:
        db_manager: Initialized DatabaseManager with a live engine.
    """
    logger.info("Running schema migrations (create_all)...")
    _prepare_schemas(db_manager.engine)
    Base.metadata.create_all(bind=db_manager.engine)
    table_names = list(Base.metadata.tables.keys())
    logger.info("Schema up-to-date. Tables: %s", table_names)
    print(f"✅ Schema migration complete. Tables ensured: {table_names}")
