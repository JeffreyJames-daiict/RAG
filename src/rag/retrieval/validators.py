"""Validation layer for queries, query embeddings, and retrieved results."""

from __future__ import annotations

import math
from typing import List

from rag.embeddings.validator import EmbeddingValidator
from rag.retrieval.exceptions import (
    EmptyQueryError,
    IncompatibleEmbeddingError,
    RetrievalError,
    TenantIsolationError,
)
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest, RetrievedChunkResult


class QueryValidator:
    """Validates user retrieval request parameters."""

    @staticmethod
    def validate_request(request: RetrievalRequest, max_top_k: int = 100) -> None:
        """Validate request query, top_k, and tenant."""
        if not request.query or not request.query.strip():
            raise EmptyQueryError("Retrieval query must not be empty or whitespace-only.")

        if not request.tenant_id or not request.tenant_id.strip():
            raise TenantIsolationError("Tenant ID must be specified for tenant-isolated retrieval.")

        if request.top_k < 1:
            raise RetrievalError(f"top_k must be at least 1, got {request.top_k}")

        if request.top_k > max_top_k:
            raise RetrievalError(
                f"Requested top_k ({request.top_k}) exceeds allowed ceiling ({max_top_k})."
            )

        if request.similarity_threshold is not None:
            if not (-1.0 <= request.similarity_threshold <= 1.0):
                raise RetrievalError(
                    f"similarity_threshold must be between -1.0 and 1.0, got {request.similarity_threshold}"
                )


class EmbeddingCompatibilityValidator:
    """Validates query vector mathematics and compatibility against indexed model."""

    @staticmethod
    def validate_query_vector(
        vector: List[float],
        expected_dimension: int,
        expected_model: str,
    ) -> None:
        """Ensure vector is finite, non-zero, non-constant, and matches target dimension."""
        if not vector:
            raise IncompatibleEmbeddingError("Generated query embedding vector is empty.")

        if len(vector) != expected_dimension:
            raise IncompatibleEmbeddingError(
                f"Query embedding dimension mismatch for model '{expected_model}': "
                f"expected {expected_dimension}, got {len(vector)}."
            )

        # Finiteness & numeric check
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise IncompatibleEmbeddingError(f"Query vector element at index {i} is non-numeric: {val}")
            if math.isnan(val):
                raise IncompatibleEmbeddingError(f"Query vector element at index {i} is NaN.")
            if math.isinf(val):
                raise IncompatibleEmbeddingError(f"Query vector element at index {i} is Infinity.")

        # Zero and constant vector checks
        if EmbeddingValidator.is_zero_vector(vector):
            raise IncompatibleEmbeddingError("Query embedding is an invalid zero-vector.")

        if EmbeddingValidator.is_constant_vector(vector):
            raise IncompatibleEmbeddingError("Query embedding is an invalid constant-vector.")


class ResultValidator:
    """Validates retrieved candidates before returning to caller."""

    @staticmethod
    def validate_candidate(candidate: RetrievedCandidate, expected_tenant_id: str) -> None:
        """Validate candidate attributes, score bounds, and sanity."""
        if not math.isfinite(candidate.similarity_score):
            raise RetrievalError(
                f"Invalid non-finite similarity score for chunk {candidate.chunk_id}: {candidate.similarity_score}"
            )

        if candidate.embedding_model != "bm25" and not (-1.0001 <= candidate.similarity_score <= 1.0001):
            raise RetrievalError(
                f"Similarity score out of bounds for chunk {candidate.chunk_id}: {candidate.similarity_score}"
            )

        if not candidate.text or not candidate.text.strip():
            # Flag empty text
            candidate.text = "(empty chunk)"
