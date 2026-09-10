"""Provider protocols and interfaces for LLM Answer Generation."""

from __future__ import annotations

from typing import Generator, Protocol, Sequence, runtime_checkable

from rag.generation.models import ChatMessage, LLMResponse, LLMStreamChunk


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for provider-independent LLM inference (OpenRouter, OpenAI-compatible, Mock)."""

    @property
    def provider_name(self) -> str:
        """Identifier for the provider (e.g. 'openrouter', 'mock', 'openai')."""
        ...

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> LLMResponse:
        """Asynchronously generate a complete text completion."""
        ...

    def generate_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> LLMResponse:
        """Synchronously generate a complete text completion."""
        ...

    def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> Generator[LLMStreamChunk, None, None]:
        """Stream completion tokens as generator chunks."""
        ...
