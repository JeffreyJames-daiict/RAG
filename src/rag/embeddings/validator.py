"""Embedding Vector Validation."""

from __future__ import annotations

import logging
import math
from typing import List, Tuple
from rag.embeddings.exceptions import (
    EmbeddingConstantVectorError,
    EmbeddingDimensionMismatchError,
    EmbeddingValidationError,
    EmbeddingZeroVectorError,
)

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """Validates embedding vectors for dimensionality, finiteness, schema integrity, and vector mathematics."""

    def __init__(
        self,
        expected_dimension: int,
        allow_zero: bool = False,
        allow_constant: bool = False,
    ):
        if expected_dimension <= 0:
            raise ValueError(f"expected_dimension must be positive, got {expected_dimension}")
        self.expected_dimension = expected_dimension
        self.allow_zero = allow_zero
        self.allow_constant = allow_constant

    @staticmethod
    def calculate_l2_norm(vector: List[float]) -> float:
        """Calculate Euclidean (L2) norm of vector."""
        return math.sqrt(sum(x * x for x in vector))

    @staticmethod
    def is_zero_vector(vector: List[float], eps: float = 1e-12) -> bool:
        """Check if all elements in the vector are effectively zero."""
        return all(abs(x) <= eps for x in vector)

    @staticmethod
    def is_constant_vector(vector: List[float], eps: float = 1e-8) -> bool:
        """Check if all elements in the vector are identical/constant."""
        if len(vector) <= 1:
            return False
        first = vector[0]
        return all(abs(x - first) <= eps for x in vector)

    def validate_vector(
        self,
        vector: List[float],
        chunk_id: str = "unknown",
        allow_zero: bool | None = None,
        allow_constant: bool | None = None,
    ) -> None:
        """Validate a single embedding vector.

        Raises:
            EmbeddingValidationError: If vector is None, empty, contains non-numeric, NaN, Inf, or invalid norm.
            EmbeddingDimensionMismatchError: If vector dimension does not match expected_dimension.
            EmbeddingZeroVectorError: If vector is all zeros (and not explicitly allowed).
            EmbeddingConstantVectorError: If vector has constant values (and not explicitly allowed).
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
            if not isinstance(val, (int, float)) or isinstance(val, bool):
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

        # Check for zero vector
        check_zero = not (self.allow_zero if allow_zero is None else allow_zero)
        if check_zero and self.is_zero_vector(vector):
            raise EmbeddingZeroVectorError(
                f"Zero vector detected for chunk '{chunk_id}': all {len(vector)} elements are 0.0."
            )

        # Check for constant vector
        check_constant = not (self.allow_constant if allow_constant is None else allow_constant)
        if check_constant and self.is_constant_vector(vector):
            raise EmbeddingConstantVectorError(
                f"Constant vector detected for chunk '{chunk_id}': all elements equal {vector[0]}."
            )

        # Norm verification
        norm = self.calculate_l2_norm(vector)
        if not math.isfinite(norm):
            raise EmbeddingValidationError(
                f"Non-finite L2 norm ({norm}) detected for chunk '{chunk_id}'."
            )

    def validate_batch(
        self,
        vectors: List[List[float]],
        chunk_ids: List[str],
        allow_zero: bool | None = None,
        allow_constant: bool | None = None,
    ) -> None:
        """Validate a batch of vectors against corresponding chunk IDs."""
        if len(vectors) != len(chunk_ids):
            raise EmbeddingValidationError(
                f"Batch count mismatch: {len(chunk_ids)} chunk IDs provided but provider returned {len(vectors)} vectors."
            )

        for vec, cid in zip(vectors, chunk_ids):
            self.validate_vector(vec, chunk_id=cid, allow_zero=allow_zero, allow_constant=allow_constant)

    @staticmethod
    def calculate_duplicate_stats(vectors: List[List[float]]) -> Tuple[int, float]:
        """Calculate unique vs duplicate vector counts and duplicate ratio in a list of vectors.

        Returns:
            Tuple of (duplicate_count, duplicate_ratio)
        """
        if not vectors:
            return 0, 0.0
        # Convert to rounded tuples for hashable comparison
        serialized = [tuple(round(x, 7) for x in v) for v in vectors]
        unique_count = len(set(serialized))
        total_count = len(serialized)
        duplicate_count = total_count - unique_count
        duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0.0
        return duplicate_count, duplicate_ratio
