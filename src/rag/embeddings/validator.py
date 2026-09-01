"""Embedding Vector Validation."""

from __future__ import annotations

import math
from typing import List
from rag.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingValidationError,
)


class EmbeddingValidator:
    """Validates embedding vectors for dimensionality, finiteness, and schema integrity."""

    def __init__(self, expected_dimension: int):
        if expected_dimension <= 0:
            raise ValueError(f"expected_dimension must be positive, got {expected_dimension}")
        self.expected_dimension = expected_dimension

    def validate_vector(self, vector: List[float], chunk_id: str = "unknown") -> None:
        """Validate a single embedding vector.

        Raises:
            EmbeddingValidationError: If vector is None, empty, contains non-numeric, NaN, or Inf values.
            EmbeddingDimensionMismatchError: If vector dimension does not match expected_dimension.
        """
        if vector is None:
            raise EmbeddingValidationError(f"Embedding vector for chunk '{chunk_id}' is None.")

        if not isinstance(vector, (list, tuple)):
            raise EmbeddingValidationError(
                f"Embedding vector for chunk '{chunk_id}' must be a list/tuple of floats, got {type(vector).__name__}."
            )

        if len(vector) == 0:
            raise EmbeddingValidationError(f"Embedding vector for chunk '{chunk_id}' is empty.")

        if len(vector) != self.expected_dimension:
            raise EmbeddingDimensionMismatchError(
                f"Dimension mismatch for chunk '{chunk_id}': expected {self.expected_dimension}, got {len(vector)}."
            )

        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                raise EmbeddingValidationError(
                    f"Non-numeric element at index {i} in embedding vector for chunk '{chunk_id}': {type(val).__name__}."
                )
            if math.isnan(val):
                raise EmbeddingValidationError(
                    f"NaN value detected at index {i} in embedding vector for chunk '{chunk_id}'."
                )
            if math.isinf(val):
                raise EmbeddingValidationError(
                    f"Infinity value detected at index {i} in embedding vector for chunk '{chunk_id}'."
                )

    def validate_batch(
        self,
        vectors: List[List[float]],
        chunk_ids: List[str],
    ) -> None:
        """Validate a batch of vectors against corresponding chunk IDs."""
        if len(vectors) != len(chunk_ids):
            raise EmbeddingValidationError(
                f"Batch count mismatch: {len(chunk_ids)} chunk IDs provided but provider returned {len(vectors)} vectors."
            )

        for vec, cid in zip(vectors, chunk_ids):
            self.validate_vector(vec, chunk_id=cid)
