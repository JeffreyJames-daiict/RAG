"""Database connection and session factory management."""

import contextlib
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from rag.core.db_models import Base


class DatabaseManager:
    """Manages SQLAlchemy engine, schema creation, and session lifecycles."""

    def __init__(self, db_url: str = "sqlite:///:memory:", echo: bool = False):
        self.db_url = db_url
        self.engine: Engine = create_engine(self.db_url, echo=echo)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def create_tables(self) -> None:
        """Create all defined ORM tables in the target database."""
        Base.metadata.create_all(self.engine)

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
