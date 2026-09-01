"""Abstract Base Class for Embedding Providers."""

from __future__ import annotations

import abc
from typing import List, Optional


class BaseEmbeddingProvider(abc.ABC):
    """Abstract interface for embedding generation providers.

    All concrete providers (e.g. Mock, OpenAI, FastEmbed) must implement this interface.
    The ingestion pipeline and embedding service depend exclusively on this contract.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str = "v1.0",
        dimension: int = 1536,
        timeout_sec: float = 30.0,
        max_batch_size: int = 64,
        max_input_tokens: int = 8191,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.dimension = dimension
        self.timeout_sec = timeout_sec
        self.max_batch_size = max_batch_size
        self.max_input_tokens = max_input_tokens

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Unique identifier of the provider implementation (e.g. 'openai', 'mock')."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_connection(self) -> bool:
        """Verify provider reachability, credentials, and model availability."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate a single embedding vector for a text string.

        Args:
            text: Normalized text to embed.

        Returns:
            List of floats representing the dense embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of text strings in batch.

        Args:
            texts: List of normalized text strings.

        Returns:
            List of dense embedding vectors in the exact order of the input texts.
        """
        raise NotImplementedError
