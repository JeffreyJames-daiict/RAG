"""Vision provider registry and factory."""

from __future__ import annotations

import logging
from typing import Optional

from rag.assets.providers.base import BaseVisionProvider
from rag.assets.providers.mock_provider import MockVisionProvider
from rag.assets.providers.openai_vision import OpenAIVisionProvider
from rag.assets.providers.openrouter_vision import (
    FREE_VISION_MODELS,
    OpenRouterVisionProvider,
)
from rag.core.config import VisionConfig

logger = logging.getLogger(__name__)


def get_vision_provider(config: Optional[VisionConfig] = None) -> BaseVisionProvider:
    """Instantiate and return the configured vision provider."""
    cfg = config or VisionConfig()
    provider_type = (cfg.provider or "openrouter").strip().lower()

    if provider_type in ("openrouter", "openrouter/free", "free", "auto"):
        return OpenRouterVisionProvider(
            api_key=cfg.api_key,
            model_name=cfg.model if cfg.model and "free" in cfg.model else FREE_VISION_MODELS[0],
            timeout_sec=cfg.timeout_sec,
        )
    elif provider_type == "mock":
        return MockVisionProvider(model_name=cfg.model or "mock-vision-v1")
    elif provider_type in ("openai", "azure_openai", "gpt4o"):
        return OpenAIVisionProvider(
            api_key=cfg.api_key,
            model_name=cfg.model or "gpt-4o",
            endpoint_url=cfg.endpoint_url,
            timeout_sec=cfg.timeout_sec,
        )
    else:
        logger.warning(
            "Unknown vision provider '%s', using OpenRouterVisionProvider (free model auto-selection)",
            provider_type,
        )
        return OpenRouterVisionProvider(
            api_key=cfg.api_key,
            model_name=cfg.model or FREE_VISION_MODELS[0],
            timeout_sec=cfg.timeout_sec,
        )


__all__ = [
    "BaseVisionProvider",
    "MockVisionProvider",
    "OpenAIVisionProvider",
    "OpenRouterVisionProvider",
    "FREE_VISION_MODELS",
    "get_vision_provider",
]
