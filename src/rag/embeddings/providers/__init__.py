import logging
from typing import Dict, Optional, Type
from rag.core.config import EmbeddingConfig, get_settings
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import EmbeddingConfigurationError
from rag.embeddings.providers.mock_provider import MockEmbeddingProvider
from rag.embeddings.providers.openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Type[BaseEmbeddingProvider]] = {
    "mock": MockEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(
    config: EmbeddingConfig,
    is_production: Optional[bool] = None,
) -> BaseEmbeddingProvider:
    """Factory creating configured embedding provider instance with production guard-rails."""
    provider_key = config.provider.lower()
    if provider_key not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown embedding provider '{config.provider}'. Available: {list(_PROVIDER_REGISTRY.keys())}"
        )

    # Determine environment mode
    if is_production is None:
        try:
            is_production = get_settings().app.is_production
        except Exception:
            is_production = False

    if provider_key == "mock":
        if is_production:
            raise EmbeddingConfigurationError(
                "Mock embedding provider is strictly prohibited in production mode. "
                "Configure a real embedding provider (e.g. 'openai') with a valid API key."
            )
        if not getattr(config, "allow_mock", True):
            raise EmbeddingConfigurationError(
                "Mock embedding provider is disabled (allow_mock=False). "
                "Set EMBEDDING_ALLOW_MOCK=true or configure a real provider."
            )
        logger.warning(
            "[MOCK MODE] Using MockEmbeddingProvider (model='%s', dim=%d). "
            "Embeddings are synthetic SHA-256 hash vectors, not real semantic embeddings.",
            config.model,
            config.dimension,
        )
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
