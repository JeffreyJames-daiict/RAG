"""Unit tests for EmbeddingValidator."""

import pytest
from rag.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingValidationError,
)
from rag.embeddings.validator import EmbeddingValidator


def test_validator_valid_vector():
    validator = EmbeddingValidator(expected_dimension=4)
    validator.validate_vector([0.1, -0.2, 0.3, 0.0], chunk_id="chk_1")


def test_validator_invalid_expected_dim():
    with pytest.raises(ValueError):
        EmbeddingValidator(expected_dimension=0)


def test_validator_none_or_empty():
    validator = EmbeddingValidator(expected_dimension=3)
    with pytest.raises(EmbeddingValidationError, match="is None"):
        validator.validate_vector(None)  # type: ignore

    with pytest.raises(EmbeddingValidationError, match="is empty"):
        validator.validate_vector([])


def test_validator_dimension_mismatch():
    validator = EmbeddingValidator(expected_dimension=3)
    with pytest.raises(EmbeddingDimensionMismatchError, match="expected 3, got 2"):
        validator.validate_vector([0.1, 0.2], chunk_id="chk_dim")


def test_validator_nan_and_inf():
    validator = EmbeddingValidator(expected_dimension=3)
    with pytest.raises(EmbeddingValidationError, match="NaN value detected"):
        validator.validate_vector([0.1, float("nan"), 0.3])

    with pytest.raises(EmbeddingValidationError, match="Infinity value detected"):
        validator.validate_vector([0.1, float("inf"), 0.3])


def test_validator_non_numeric():
    validator = EmbeddingValidator(expected_dimension=2)
    with pytest.raises(EmbeddingValidationError, match="Non-numeric element"):
        validator.validate_vector(["0.1", 0.2])  # type: ignore


def test_validator_batch_validation():
    validator = EmbeddingValidator(expected_dimension=2)
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    chunk_ids = ["chk_a", "chk_b"]
    validator.validate_batch(vectors, chunk_ids)

    # Count mismatch
    with pytest.raises(EmbeddingValidationError, match="Batch count mismatch"):
        validator.validate_batch(vectors, ["chk_a"])
