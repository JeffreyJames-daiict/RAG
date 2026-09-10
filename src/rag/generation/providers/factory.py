"""Provider factory for instantiating LLMProvider implementations."""

from __future__ import annotations

from typing import Optional

from rag.core.config import GenerationConfig
from rag.generation.interfaces import LLMProvider
from rag.generation.openrouter_client import OpenRouterProvider
from rag.generation.providers.mock_provider import MockLLMProvider


def get_llm_provider(config: Optional[GenerationConfig] = None) -> LLMProvider:
    """Factory to instantiate the configured LLMProvider."""
    cfg = config or GenerationConfig()
    provider_type = (cfg.provider or "openrouter").lower().strip()

    if provider_type in ("openrouter", "openai"):
        return OpenRouterProvider(
            candidate_models=[cfg.model, cfg.fallback_model] + cfg.fallback_models,
            max_retries_per_model=cfg.max_retries,
            timeout_sec=cfg.timeout_sec,
        )
    elif provider_type == "mock":
        return MockLLMProvider()

    raise ValueError(f"Unsupported LLM provider: '{cfg.provider}'")
