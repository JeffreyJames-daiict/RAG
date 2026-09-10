"""Repository factory and exports for vector retrieval backends."""

from __future__ import annotations

import logging
from typing import Optional

from rag.core.config import RetrievalConfig, get_settings
from rag.core.database import DatabaseManager
from rag.retrieval.exceptions import BackendUnavailableError
from rag.retrieval.repositories.base import BaseVectorRepository
from rag.retrieval.repositories.python_repository import PythonVectorRepository

logger = logging.getLogger(__name__)

__all__ = [
    "BaseVectorRepository",
    "PythonVectorRepository",
    "get_vector_repository",
]


def _try_pgvector(
    db_manager: DatabaseManager,
    cfg: RetrievalConfig,
) -> Optional[BaseVectorRepository]:
    """Attempt to instantiate PgVectorRepository; return None if unavailable."""
    try:
        from rag.retrieval.repositories.pgvector_repository import PgVectorRepository

        repo = PgVectorRepository(
            db_manager=db_manager,
            distance_metric=cfg.distance_metric,
            candidate_limit=cfg.candidate_limit,
        )
        if repo.is_available():
            logger.info(
                "pgvector backend available — using native HNSW similarity search "
                "(metric=%s).",
                cfg.distance_metric,
            )
            return repo
        else:
            logger.info(
                "pgvector extension or embedding_vector column not ready — "
                "falling back."
            )
            return None
    except Exception as e:
        logger.warning("Could not instantiate PgVectorRepository: %s", e)
        return None


def get_vector_repository(
    db_manager: DatabaseManager,
    config: Optional[RetrievalConfig] = None,
) -> BaseVectorRepository:
    """Factory: create the appropriate vector repository backend.

    Args:
        db_manager: Database connection manager.
        config:     Optional retrieval configuration override.

    Returns:
        A concrete BaseVectorRepository implementation.

    Raises:
        BackendUnavailableError: If the requested backend is not available.
    """
    cfg = config or get_settings().retrieval
    backend = cfg.backend.lower()

    if backend == "python":
        return PythonVectorRepository(
            db_manager=db_manager,
            candidate_limit=cfg.candidate_limit,
            distance_metric=cfg.distance_metric,
        )

    if backend == "pgvector":
        repo = _try_pgvector(db_manager, cfg)
        if repo is not None:
            return repo
        if cfg.allow_python_fallback:
            logger.warning(
                "pgvector backend explicitly requested but unavailable — "
                "falling back to Python in-memory."
            )
            return PythonVectorRepository(
                db_manager=db_manager,
                candidate_limit=cfg.candidate_limit,
                distance_metric=cfg.distance_metric,
            )
        raise BackendUnavailableError(
            "pgvector backend requested but is not available and "
            "allow_python_fallback=False."
        )

    if backend == "auto":
        # Auto-detect: try pgvector first, fall back to Python
        repo = _try_pgvector(db_manager, cfg)
        if repo is not None:
            return repo

        if cfg.allow_python_fallback:
            logger.info(
                "Auto backend: pgvector not available, using Python in-memory vector search."
            )
            return PythonVectorRepository(
                db_manager=db_manager,
                candidate_limit=cfg.candidate_limit,
                distance_metric=cfg.distance_metric,
            )
        raise BackendUnavailableError(
            "No vector backend available and allow_python_fallback=False."
        )

    raise BackendUnavailableError(f"Unknown retrieval backend: '{backend}'.")
