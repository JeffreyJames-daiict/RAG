"""Deterministic Mock Embedding Provider for fast unit and integration testing."""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import EmbeddingProviderError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic, unit-normalized float vectors using SHA-256 seed hashing.

    Guarantees:
    - Identical text always yields identical vectors.
    - Different text yields distinct vectors.
    - Vectors have exact specified dimension and unit L2 norm.
    - Fast, zero external API calls, offline compatible.
    """

    def __init__(
        self,
        model_name: str = "mock-embedding-v1",
        model_version: str = "v1.0",
        dimension: int = 1536,
        timeout_sec: float = 30.0,
        max_batch_size: int = 64,
        max_input_tokens: int = 8191,
        simulate_failure: bool = False,
        simulate_dimension_error: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            timeout_sec=timeout_sec,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
        )
        self.simulate_failure = simulate_failure
        self.simulate_dimension_error = simulate_dimension_error

    @property
    def provider_name(self) -> str:
        return "mock"

    def validate_connection(self) -> bool:
        if self.simulate_failure:
            raise EmbeddingProviderError("Mock provider connection failed by simulation flag.")
        return True

    def _generate_vector(self, text: str) -> List[float]:
        if self.simulate_failure:
            raise EmbeddingProviderError("Simulated mock provider failure during embedding generation.")

        dim = self.dimension
        if self.simulate_dimension_error:
            dim = self.dimension - 10  # deliberately incorrect dimension

        # Generate deterministic floats from multi-stage SHA-256 hashing
        vector: List[float] = []
        salt = 0
        while len(vector) < dim:
            seed_bytes = f"{text}:{salt}".encode("utf-8")
            digest = hashlib.sha256(seed_bytes).digest()
            # Convert 4-byte chunks to floats between -1.0 and 1.0
            for i in range(0, len(digest) - 3, 4):
                if len(vector) >= dim:
                    break
                int_val = int.from_bytes(digest[i:i+4], byteorder="big", signed=True)
                vector.append(int_val / 2147483648.0)
            salt += 1

        # L2 normalize vector
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [round(x / norm, 6) for x in vector]

    def embed(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self._generate_vector(t) for t in texts]
