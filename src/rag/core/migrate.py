"""Database migration helper — ensures all ORM tables exist on the target PostgreSQL instance."""

from __future__ import annotations
import logging

from rag.core.database import DatabaseManager
from rag.core.db_models import Base

logger = logging.getLogger(__name__)


def run_migrations(db_manager: DatabaseManager) -> None:
    """Create all tables defined in the ORM Base metadata if they do not exist.

    This is a lightweight, idempotent 'create if not exists' migration. It will
    not modify or drop existing tables. For production schema migrations use
    Alembic instead.

    Args:
        db_manager: Initialized DatabaseManager with a live engine.
    """
    logger.info("Running schema migrations (create_all)...")
    Base.metadata.create_all(bind=db_manager.engine)
    table_names = list(Base.metadata.tables.keys())
    logger.info("Schema up-to-date. Tables: %s", table_names)
    print(f"✅ Schema migration complete. Tables ensured: {table_names}")
