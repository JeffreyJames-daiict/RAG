"""Comprehensive tests for embedding authenticity, vector mathematics, semantic sensitivity,
mock detection, input mapping, and persistence round-trip fidelity.
"""

import hashlib
import math
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select

from rag.core.config import EmbeddingConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.authenticity import (
    EmbeddingAuthenticityChecker,
    EmbeddingAuthenticityReport,
    cosine_similarity,
)
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingConstantVectorError,
    EmbeddingDimensionMismatchError,
    EmbeddingValidationError,
    EmbeddingZeroVectorError,
)
from rag.embeddings.providers import MockEmbeddingProvider, OpenAIEmbeddingProvider, get_embedding_provider
from rag.embeddings.validator import EmbeddingValidator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Synthetic Providers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def in_memory_db():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


class SyntheticSemanticProvider(BaseEmbeddingProvider):
    """Simulated authentic provider with realistic semantic sensitivity."""

    def __init__(self, dimension: int = 128):
        super().__init__(model_name="synthetic-semantic-v1", dimension=dimension)

    @property
    def provider_name(self) -> str:
        return "synthetic_real"

    def validate_connection(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Vector design:
        # database texts share a strong signal in the first quadrant
        # quantum texts share a signal in the second quadrant
        results = []
        d = self.dimension
        for t in texts:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            vec = [0.001 * ((seed >> (i % 60)) & 0x7) for i in range(d)]
            if "database" in t.lower() or "postgresql" in t.lower():
                for i in range(0, min(d, max(1, d // 3))):
                    vec[i] += 0.8
            if "persistence" in t.lower():
                for i in range(min(d, max(1, d // 6)), min(d, max(2, d // 2))):
                    vec[i] += 0.7
            if "quantum" in t.lower():
                for i in range(min(d, max(2, d // 2)), d):
                    vec[i] += 0.9

            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([round(x / norm, 6) for x in vec])
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Mathematics Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_math_valid_vector():
    validator = EmbeddingValidator(expected_dimension=4)
    validator.validate_vector([0.2, 0.5, -0.4, 0.1])
    assert validator.calculate_l2_norm([3.0, 4.0]) == 5.0


def test_vector_math_zero_vector_rejection():
    validator = EmbeddingValidator(expected_dimension=4)
    with pytest.raises(EmbeddingZeroVectorError, match="Zero vector detected"):
        validator.validate_vector([0.0, 0.0, 0.0, 0.0])


def test_vector_math_constant_vector_rejection():
    validator = EmbeddingValidator(expected_dimension=4)
    with pytest.raises(EmbeddingConstantVectorError, match="Constant vector detected"):
        validator.validate_vector([0.25, 0.25, 0.25, 0.25])


def test_vector_math_nan_inf_rejection():
    validator = EmbeddingValidator(expected_dimension=3)
    with pytest.raises(EmbeddingValidationError, match="NaN"):
        validator.validate_vector([0.1, float("nan"), 0.5])

    with pytest.raises(EmbeddingValidationError, match="Infinity"):
        validator.validate_vector([0.1, float("inf"), 0.5])


def test_vector_math_dimension_mismatch():
    validator = EmbeddingValidator(expected_dimension=4)
    with pytest.raises(EmbeddingDimensionMismatchError):
        validator.validate_vector([0.1, 0.2, 0.3])


def test_vector_math_non_numeric_and_boolean():
    validator = EmbeddingValidator(expected_dimension=3)
    with pytest.raises(EmbeddingValidationError, match="Non-numeric"):
        validator.validate_vector(["0.1", 0.2, 0.3])

    with pytest.raises(EmbeddingValidationError, match="Non-numeric"):
        validator.validate_vector([True, 0.2, 0.3])  # Python bool is subclass of int


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cosine Similarity & Stability Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cosine_similarity_edge_cases():
    # Identical vectors
    v1 = [1.0, 2.0, 3.0]
    assert pytest.approx(cosine_similarity(v1, v1), 1e-6) == 1.0

    # Orthogonal vectors
    v_a = [1.0, 0.0]
    v_b = [0.0, 1.0]
    assert pytest.approx(cosine_similarity(v_a, v_b), 1e-6) == 0.0

    # Opposite vectors
    v_c = [-1.0, 0.0]
    assert pytest.approx(cosine_similarity(v_a, v_c), 1e-6) == -1.0

    # Zero vector handles safely without ZeroDivisionError
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_same_input_stability_check():
    prov = MockEmbeddingProvider(dimension=64)
    checker = EmbeddingAuthenticityChecker(prov)
    stab = checker.check_same_input_stability("Enterprise RAG deterministic query.")
    assert stab["exact_match"] is True
    assert stab["cosine_similarity"] == 1.0
    assert stab["maximum_absolute_difference"] == 0.0
    assert stab["stable"] is True


def test_different_input_variation():
    prov = MockEmbeddingProvider(dimension=64)
    checker = EmbeddingAuthenticityChecker(prov)
    var = checker.check_different_input_variation()
    assert var["total_count"] == 10
    assert var["unique_count"] == 10
    assert var["duplicate_count"] == 0
    assert var["duplicate_ratio"] == 0.0
    assert var["is_suspiciously_identical"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic Sensitivity & Mock vs Real Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_provider_avalanche_effect_detected():
    """Verify that SHA-256 hash-based mock provider is detected as is_mock_like."""
    prov = MockEmbeddingProvider(dimension=128)
    checker = EmbeddingAuthenticityChecker(prov)
    sem = checker.check_semantic_sensitivity()

    # In SHA-256, similar text and unrelated text are both effectively random orthogonal vectors
    assert pytest.approx(sem["same_text_similarity"], 1e-6) == 1.0
    assert abs(sem["similar_text_similarity"]) < 0.2
    assert sem["is_mock_like"] is True


def test_synthetic_semantic_provider_passes_sensitivity():
    """Verify that an authentic semantic provider is recognized as authentic."""
    prov = SyntheticSemanticProvider(dimension=128)
    checker = EmbeddingAuthenticityChecker(prov)
    sem = checker.check_semantic_sensitivity()

    # Similar text has high similarity, unrelated text has lower similarity
    assert sem["same_text_similarity"] > 0.99
    assert sem["similar_text_similarity"] > 0.70
    assert sem["unrelated_text_similarity"] < 0.35
    assert sem["is_mock_like"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Input-to-Embedding Mapping Integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_input_mapping_integrity():
    prov = MockEmbeddingProvider(dimension=64)
    checker = EmbeddingAuthenticityChecker(prov)
    mapping = checker.check_input_to_embedding_mapping()
    assert mapping["mapping_verified"] is True
    assert len(mapping["discrepancies"]) == 0


def test_input_mapping_detects_order_shuffle():
    flaky_prov = MagicMock()
    flaky_prov.dimension = 4
    # Batch response reverses the order
    flaky_prov.embed_batch.return_value = [
        [0.3, 0.3, 0.3, 0.1],
        [0.2, 0.2, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.1],
    ]
    flaky_prov.embed.side_effect = [
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.1],
        [0.3, 0.3, 0.3, 0.1],
    ]

    checker = EmbeddingAuthenticityChecker(flaky_prov)
    mapping = checker.check_input_to_embedding_mapping()
    assert mapping["mapping_verified"] is False
    assert len(mapping["discrepancies"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Database Persistence Round-Trip Fidelity
# ─────────────────────────────────────────────────────────────────────────────

def test_persistence_integrity_roundtrip(in_memory_db):
    prov = MockEmbeddingProvider(dimension=128)
    checker = EmbeddingAuthenticityChecker(prov)
    persist_res = checker.check_persistence_integrity(in_memory_db)

    assert persist_res["verified"] is True
    assert persist_res["maximum_absolute_difference"] < 1e-6
    assert persist_res["cosine_similarity"] > 0.999999
    assert persist_res["metadata_match"] is True
    assert persist_res["stored_dimension"] == 128
    assert persist_res["stored_provider"] == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Provider Authenticity & Guard-Rails (Production vs Development)
# ─────────────────────────────────────────────────────────────────────────────

def test_production_mode_strictly_rejects_mock():
    cfg = EmbeddingConfig(provider="mock")
    with pytest.raises(EmbeddingConfigurationError, match="strictly prohibited in production"):
        get_embedding_provider(cfg, is_production=True)


def test_disallowed_mock_in_development_rejected():
    cfg = EmbeddingConfig(provider="mock", allow_mock=False)
    with pytest.raises(EmbeddingConfigurationError, match="disabled"):
        get_embedding_provider(cfg, is_production=False)


def test_allowed_mock_in_development_succeeds():
    cfg = EmbeddingConfig(provider="mock", allow_mock=True)
    prov = get_embedding_provider(cfg, is_production=False)
    assert isinstance(prov, MockEmbeddingProvider)


def test_safe_diagnostic_dict_never_exposes_api_key():
    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key="sk-secret-confidential-api-key-12345",
    )
    safe = cfg.safe_diagnostic_dict(is_production=True)
    assert "sk-secret" not in str(safe)
    assert safe["embedding_api_key_configured"] is True
    assert safe["embedding_provider"] == "openai"
    assert safe["mode"] == "production"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Authenticity Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_full_authenticity_report_with_mock_provider(in_memory_db):
    prov = MockEmbeddingProvider(dimension=64)
    checker = EmbeddingAuthenticityChecker(prov)
    report: EmbeddingAuthenticityReport = checker.generate_full_report(
        db_manager=in_memory_db,
        is_production=False,
    )

    assert report.is_valid is True
    assert report.provider_verified is True
    assert report.dimension_valid is True
    assert report.norm_valid is True
    assert report.is_zero_vector is False
    assert report.is_constant_vector is False
    assert report.is_repeated_vector is False
    assert report.persistence_verified is True
    assert report.is_mock_like is True
    assert report.overall_status == "WARN"  # Mock in dev produces WARN
    assert len(report.warnings) > 0


def test_full_authenticity_report_with_synthetic_real_provider(in_memory_db):
    prov = SyntheticSemanticProvider(dimension=64)
    checker = EmbeddingAuthenticityChecker(prov)
    report: EmbeddingAuthenticityReport = checker.generate_full_report(
        db_manager=in_memory_db,
        is_production=False,
    )

    assert report.is_valid is True
    assert report.is_mock_like is False
    assert report.overall_status == "PASS"
    assert report.similar_text_similarity is not None
    assert report.similar_text_similarity > 0.65
