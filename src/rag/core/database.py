"""Database connection and session factory management."""

from __future__ import annotations

import contextlib
import logging
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from rag.core.db_models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLAlchemy engine, schema creation, and session lifecycles."""

    def __init__(
        self,
        db_url: str = "sqlite:///:memory:",
        echo: bool = False,
        pool_size: Optional[int] = None,
        max_overflow: Optional[int] = None,
        **engine_kwargs,
    ):
        self.db_url = db_url
        engine_args = {"echo": echo, **engine_kwargs}

        if self.db_url.startswith("sqlite"):
            if ":memory:" in self.db_url:
                engine_args.setdefault("poolclass", StaticPool)
                engine_args.setdefault("connect_args", {"check_same_thread": False})
        else:
            if pool_size is not None:
                engine_args["pool_size"] = pool_size
            if max_overflow is not None:
                engine_args["max_overflow"] = max_overflow

        self.engine: Engine = create_engine(self.db_url, **engine_args)

        # Attach 'raw' database for SQLite automatically on every connection
        if self.db_url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _attach_sqlite_raw(dbapi_conn, connection_record):
                try:
                    dbapi_conn.execute("ATTACH DATABASE ':memory:' AS raw")
                except Exception:
                    pass

        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def create_tables(self) -> None:
        """Create all defined ORM schemas and tables in the target database."""
        from rag.core.migrate import run_migrations
        run_migrations(self)

    def drop_tables(self) -> None:
        """Drop all tables in the target database."""
        Base.metadata.drop_all(self.engine)

    @contextlib.contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        sess = self.session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()
