"""Opt-in real provider integration test for live external embedding endpoints.

Run only when explicitly enabled:
    RUN_EMBEDDING_PROVIDER_INTEGRATION=true pytest tests/test_embeddings_real_provider_integration.py
"""

import os
import pytest
from rag.core.config import EmbeddingConfig
from rag.embeddings.authenticity import EmbeddingAuthenticityChecker, cosine_similarity
from rag.embeddings.providers import OpenAIEmbeddingProvider

ENABLE_INTEGRATION = os.getenv("RUN_EMBEDDING_PROVIDER_INTEGRATION", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@pytest.mark.skipif(
    not ENABLE_INTEGRATION,
    reason="Live integration tests disabled. Set RUN_EMBEDDING_PROVIDER_INTEGRATION=true to enable.",
)
def test_live_openai_embedding_provider_integration():
    """Verify live OpenAI provider with authentic model dimensions and semantic sensitivity."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY environment variable is not configured.")

    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        api_key=OPENAI_API_KEY,
    )

    provider = OpenAIEmbeddingProvider(
        model_name=cfg.model,
        dimension=cfg.dimension,
        api_key=cfg.api_key,
    )

    # 1. Connection check
    assert provider.validate_connection() is True

    # 2. Authenticity checks
    checker = EmbeddingAuthenticityChecker(provider)

    # Mathematical checks on single embed
    probe = provider.embed("Enterprise RAG verification text.")
    math_res = checker.check_vector_math(probe)
    assert math_res["valid"] is True
    assert math_res["dimension"] == 1536
    assert math_res["is_zero"] is False
    assert math_res["is_constant"] is False

    # Stability
    stab = checker.check_same_input_stability("Enterprise RAG verification text.")
    assert stab["stable"] is True
    assert stab["cosine_similarity"] > 0.9999

    # Variation across inputs
    var = checker.check_different_input_variation()
    assert var["duplicate_count"] == 0
    assert var["unique_count"] == 10

    # Semantic sensitivity
    sem = checker.check_semantic_sensitivity()
    assert sem["same_text_similarity"] > 0.99
    # Real semantic model has much higher similarity for similar texts than unrelated
    assert sem["similar_text_similarity"] > sem["unrelated_text_similarity"]
    assert sem["is_mock_like"] is False
