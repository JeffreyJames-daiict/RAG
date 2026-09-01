"""Embedding Service: Orchestrates chunk validation, batching, retries, vector validation,

and transactional database persistence.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set, Tuple
import tiktoken
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.core.config import EmbeddingConfig, get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInputTooLargeError,
    EmbeddingPersistenceError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingValidationError,
)
from rag.embeddings.models import (
    BatchEmbeddingResult,
    ChunkEmbeddingFailure,
    EmbeddingResult,
    EmbeddingStatus,
    EmbeddingSummary,
    EmbeddingTask,
)
from rag.embeddings.providers import get_embedding_provider
from rag.embeddings.validator import EmbeddingValidator

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Production service for embedding generation and database persistence."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        provider: Optional[BaseEmbeddingProvider] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.db_manager = db_manager
        self.config = config or get_settings().embedding
        self.provider = provider or get_embedding_provider(self.config)
        self.validator = EmbeddingValidator(expected_dimension=self.provider.dimension)

        # Initialize tokenizer for token validation and safe truncation
        try:
            self._tokenizer = tiktoken.get_encoding(get_settings().chunking.encoding_model)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def model_version(self) -> str:
        return self.provider.model_version

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    # ------------------------------------------------------------------
    # Input Validation & Preprocessing
    # ------------------------------------------------------------------

    def validate_and_prepare_chunk(self, chunk: Chunk) -> Optional[EmbeddingTask]:
        """Validate chunk text, handle whitespace/empty blocks, and check token limits.

        Returns:
            EmbeddingTask if eligible, or None if skipped (e.g. empty/whitespace).
        """
        raw_text = chunk.text or ""
        stripped = raw_text.strip()
        if not stripped:
            logger.debug("Skipping empty/whitespace-only chunk: %s", chunk.chunk_id)
            return None

        # Token count calculation
        tokens = self._tokenizer.encode(stripped)
        token_count = len(tokens)

        max_limit = self.provider.max_input_tokens
        if token_count > max_limit:
            strategy = self.config.oversized_chunk_strategy.lower()
            if strategy == "truncate":
                logger.warning(
                    "Oversized chunk %s (%d tokens > limit %d): truncating text to fit model context.",
                    chunk.chunk_id, token_count, max_limit,
                )
                truncated_tokens = tokens[:max_limit]
                stripped = self._tokenizer.decode(truncated_tokens)
                token_count = len(truncated_tokens)
            elif strategy == "reject":
                logger.warning(
                    "Oversized chunk %s (%d tokens > limit %d): rejected by policy.",
                    chunk.chunk_id, token_count, max_limit,
                )
                return None
            else:
                raise EmbeddingInputTooLargeError(
                    f"Chunk {chunk.chunk_id} has {token_count} tokens exceeding limit {max_limit}."
                )

        return EmbeddingTask(
            chunk_id=chunk.chunk_id,
            text=stripped,
            document_id=chunk.document_id,
            version=chunk.version,
            token_count=token_count,
            metadata=chunk.chunk_metadata or {},
        )

    # ------------------------------------------------------------------
    # Idempotency: Check Existing Embeddings
    # ------------------------------------------------------------------

    def get_existing_embedded_chunk_ids(
        self,
        chunk_ids: List[str],
        session: Session,
    ) -> Set[str]:
        """Query DB to find chunk IDs that already have a valid embedding for this model+version."""
        if not chunk_ids:
            return set()

        stmt = select(Embedding.chunk_id).where(
            Embedding.chunk_id.in_(chunk_ids),
            Embedding.embedding_model == self.model_name,
            Embedding.embedding_version == self.model_version,
        )
        existing = session.execute(stmt).scalars().all()
        return set(existing)

    # ------------------------------------------------------------------
    # Provider Execution with Retries
    # ------------------------------------------------------------------

    def _call_provider_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call provider.embed_batch with exponential backoff for transient errors."""
        max_retries = self.config.max_retries
        backoff = self.config.retry_backoff_sec

        for attempt in range(max_retries + 1):
            try:
                return self.provider.embed_batch(texts)
            except (EmbeddingTimeoutError, EmbeddingRateLimitError) as e:
                if attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(
                        "Transient embedding error (attempt %d/%d): %s. Retrying in %.2fs...",
                        attempt + 1, max_retries, e, sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Embedding provider failed after %d retries: %s", max_retries, e)
                    raise
            except (EmbeddingValidationError, EmbeddingDimensionMismatchError):
                # Never retry validation or dimension mismatch errors
                raise
            except Exception as e:
                if attempt < max_retries and "connection" in str(e).lower():
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning("Connection error on attempt %d: %s. Retrying...", attempt + 1, e)
                    time.sleep(sleep_time)
                else:
                    raise

        raise EmbeddingProviderError("Exhausted retries calling embedding provider.")

    # ------------------------------------------------------------------
    # Batch Processing & Error Isolation
    # ------------------------------------------------------------------

    def process_batch(self, tasks: List[EmbeddingTask]) -> BatchEmbeddingResult:
        """Process a single batch of embedding tasks with validation and fallback isolation."""
        start_time = time.time()
        result = BatchEmbeddingResult(
            status=EmbeddingStatus.SUCCESS,
            provider_name=self.provider_name,
            model_name=self.model_name,
            model_version=self.model_version,
            dimension=self.dimension,
        )

        if not tasks:
            result.status = EmbeddingStatus.SKIPPED
            return result

        texts = [t.text for t in tasks]
        chunk_ids = [t.chunk_id for t in tasks]

        try:
            vectors = self._call_provider_with_retry(texts)
            self.validator.validate_batch(vectors, chunk_ids)

            for task, vec in zip(tasks, vectors):
                result.successful_results.append(
                    EmbeddingResult(
                        chunk_id=task.chunk_id,
                        vector=vec,
                        model_name=self.model_name,
                        model_version=self.model_version,
                        dimension=self.dimension,
                        provider_name=self.provider_name,
                        prompt_tokens=task.token_count,
                    )
                )
                result.total_tokens += task.token_count

        except Exception as batch_err:
            logger.warning(
                "Batch embedding of %d chunks failed: %s. Attempting chunk-by-chunk fallback...",
                len(tasks), batch_err,
            )
            # Fallback to chunk-by-chunk to isolate individual failures
            for task in tasks:
                try:
                    vec = self._call_provider_with_retry([task.text])[0]
                    self.validator.validate_vector(vec, chunk_id=task.chunk_id)
                    result.successful_results.append(
                        EmbeddingResult(
                            chunk_id=task.chunk_id,
                            vector=vec,
                            model_name=self.model_name,
                            model_version=self.model_version,
                            dimension=self.dimension,
                            provider_name=self.provider_name,
                            prompt_tokens=task.token_count,
                        )
                    )
                    result.total_tokens += task.token_count
                except Exception as single_err:
                    logger.error("Failed to embed chunk %s: %s", task.chunk_id, single_err)
                    result.failed_chunks.append(
                        ChunkEmbeddingFailure(
                            chunk_id=task.chunk_id,
                            error_type=type(single_err).__name__,
                            error_message=str(single_err),
                        )
                    )

        if result.failed_chunks:
            result.status = (
                EmbeddingStatus.PARTIAL_SUCCESS
                if result.successful_results
                else EmbeddingStatus.FAILED
            )

        result.duration_ms = (time.time() - start_time) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Transactional Persistence
    # ------------------------------------------------------------------

    def persist_results(
        self,
        results: List[EmbeddingResult],
        session: Session,
    ) -> int:
        """Persist generated embedding results to the database."""
        if not results:
            return 0

        persisted = 0
        for res in results:
            db_emb = Embedding(
                chunk_id=res.chunk_id,
                embedding_model=res.model_name,
                embedding_version=res.model_version,
                dimension=res.dimension,
                provider=res.provider_name,
                vector=res.vector,
            )
            session.merge(db_emb)
            persisted += 1

        session.flush()
        return persisted

    # ------------------------------------------------------------------
    # High-Level Orchestration APIs
    # ------------------------------------------------------------------

    def embed_document(
        self,
        document_id: str,
        version: Optional[int] = None,
        force_reembed: bool = False,
        session: Optional[Session] = None,
    ) -> EmbeddingSummary:
        """Generate and persist embeddings for all chunks of a document."""
        start_time = time.time()
        summary = EmbeddingSummary(
            document_id=document_id,
            model_name=self.model_name,
            model_version=self.model_version,
            provider_name=self.provider_name,
        )

        def _do_embed(sess: Session) -> None:
            # Query chunks for this document
            stmt = select(Chunk).where(Chunk.document_id == document_id)
            if version is not None:
                stmt = stmt.where(Chunk.version == version)
            stmt = stmt.order_by(Chunk.chunk_index.asc())

            chunks: List[Chunk] = sess.execute(stmt).scalars().all()
            summary.total_chunks = len(chunks)

            if not chunks:
                logger.info("No chunks found for document %s (version=%s)", document_id, version)
                return

            # Prepare tasks
            tasks: List[EmbeddingTask] = []
            for c in chunks:
                task = self.validate_and_prepare_chunk(c)
                if task:
                    tasks.append(task)
                else:
                    summary.skipped_count += 1

            if not tasks:
                return

            # Idempotency check: skip already embedded chunks unless force_reembed
            if not force_reembed:
                existing_ids = self.get_existing_embedded_chunk_ids(
                    [t.chunk_id for t in tasks],
                    session=sess,
                )
                if existing_ids:
                    summary.skipped_count += len(existing_ids)
                    tasks = [t for t in tasks if t.chunk_id not in existing_ids]

            if not tasks:
                logger.info(
                    "All %d chunks for doc %s already have up-to-date embeddings (model=%s, ver=%s).",
                    summary.total_chunks, document_id, self.model_name, self.model_version,
                )
                return

            # Process in configured batch sizes
            batch_size = self.config.batch_size
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_res = self.process_batch(batch_tasks)

                if batch_res.successful_results:
                    self.persist_results(batch_res.successful_results, session=sess)
                    summary.embedded_count += len(batch_res.successful_results)

                if batch_res.failed_chunks:
                    summary.failed_count += len(batch_res.failed_chunks)
                    for f in batch_res.failed_chunks:
                        summary.errors.append(f"Chunk {f.chunk_id}: {f.error_message}")

        if session is not None:
            _do_embed(session)
        else:
            with self.db_manager.session() as sess:
                _do_embed(sess)

        summary.duration_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "Embedded document %s: %d embedded, %d skipped, %d failed (%.2f ms)",
            document_id, summary.embedded_count, summary.skipped_count, summary.failed_count, summary.duration_ms,
        )
        return summary

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query text string (utility for future retrieval)."""
        stripped = query.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or whitespace.")
        vector = self.provider.embed(stripped)
        self.validator.validate_vector(vector, chunk_id="query")
        return vector
