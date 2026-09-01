"""Unit tests for BaseEmbeddingProvider, MockEmbeddingProvider, and OpenAIEmbeddingProvider."""

from unittest.mock import MagicMock
import pytest
import requests
from rag.core.config import EmbeddingConfig
from rag.embeddings.exceptions import (
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)
from rag.embeddings.providers import (
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


def test_mock_embedding_provider_single_and_batch():
    provider = MockEmbeddingProvider(dimension=128)
    assert provider.provider_name == "mock"
    assert provider.dimension == 128
    assert provider.validate_connection() is True

    # Single embed
    vec1 = provider.embed("hello world")
    assert len(vec1) == 128
    assert isinstance(vec1[0], float)

    # Determinism
    vec1_again = provider.embed("hello world")
    assert vec1 == vec1_again

    # Distinct text produces distinct vector
    vec2 = provider.embed("different text query")
    assert vec1 != vec2

    # Batch embed
    batch = provider.embed_batch(["text one", "text two", "text three"])
    assert len(batch) == 3
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128
    assert len(batch[2]) == 128


def test_mock_embedding_provider_empty_batch():
    provider = MockEmbeddingProvider(dimension=64)
    assert provider.embed_batch([]) == []


def test_mock_embedding_provider_simulated_failure():
    provider = MockEmbeddingProvider(simulate_failure=True)
    with pytest.raises(EmbeddingProviderError):
        provider.validate_connection()
    with pytest.raises(EmbeddingProviderError):
        provider.embed("test")


def test_openai_embedding_provider_success():
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
    }
    mock_session.post.return_value = mock_response

    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small",
        dimension=3,
        api_key="sk-test-key",
        session=mock_session,
    )
    assert provider.provider_name == "openai"

    results = provider.embed_batch(["first text", "second text"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.4, 0.5, 0.6]

    # Verify headers and payload
    mock_session.post.assert_called_once()
    _, kwargs = mock_session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
    assert kwargs["json"]["model"] == "text-embedding-3-small"
    assert kwargs["json"]["dimensions"] == 3


def test_openai_embedding_provider_rate_limit():
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"
    mock_session.post.return_value = mock_response

    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        session=mock_session,
    )

    with pytest.raises(EmbeddingRateLimitError):
        provider.embed("text")


def test_openai_embedding_provider_timeout():
    mock_session = MagicMock()
    mock_session.post.side_effect = requests.Timeout("Request timed out")

    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        session=mock_session,
    )

    with pytest.raises(EmbeddingTimeoutError):
        provider.embed("text")


def test_provider_factory():
    cfg_mock = EmbeddingConfig(provider="mock", dimension=256)
    p_mock = get_embedding_provider(cfg_mock)
    assert isinstance(p_mock, MockEmbeddingProvider)
    assert p_mock.dimension == 256

    cfg_openai = EmbeddingConfig(provider="openai", model="text-embedding-3-large", dimension=3072)
    p_openai = get_embedding_provider(cfg_openai)
    assert isinstance(p_openai, OpenAIEmbeddingProvider)
    assert p_openai.dimension == 3072

    cfg_invalid = EmbeddingConfig(provider="unsupported_provider")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider(cfg_invalid)
