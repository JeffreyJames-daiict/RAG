"""Embedding Providers registry and factory."""

from __future__ import annotations

from typing import Dict, Type
from rag.core.config import EmbeddingConfig
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.providers.mock_provider import MockEmbeddingProvider
from rag.embeddings.providers.openai_provider import OpenAIEmbeddingProvider

_PROVIDER_REGISTRY: Dict[str, Type[BaseEmbeddingProvider]] = {
    "mock": MockEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Factory creating configured embedding provider instance."""
    provider_key = config.provider.lower()
    if provider_key not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown embedding provider '{config.provider}'. Available: {list(_PROVIDER_REGISTRY.keys())}"
        )

    provider_cls = _PROVIDER_REGISTRY[provider_key]
    if provider_key == "mock":
        return MockEmbeddingProvider(
            model_name=config.model,
            model_version=config.model_version,
            dimension=config.dimension,
            timeout_sec=config.timeout_sec,
            max_batch_size=config.batch_size,
            max_input_tokens=config.max_input_tokens,
        )
    elif provider_key == "openai":
        return OpenAIEmbeddingProvider(
            model_name=config.model,
            model_version=config.model_version,
            dimension=config.dimension,
            api_key=config.api_key,
            endpoint_url=config.endpoint_url,
            timeout_sec=config.timeout_sec,
            max_batch_size=config.batch_size,
            max_input_tokens=config.max_input_tokens,
        )
    else:
        return provider_cls(
            model_name=config.model,
            model_version=config.model_version,
            dimension=config.dimension,
            timeout_sec=config.timeout_sec,
            max_batch_size=config.batch_size,
            max_input_tokens=config.max_input_tokens,
        )


__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]
