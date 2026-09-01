"""OpenAI-compatible HTTP Embedding Provider."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import requests

from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import (
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI or any OpenAI-compatible HTTP endpoint.

    Compatible with OpenAI, Azure OpenAI, vLLM, Ollama, and LiteLLM endpoints.
    """

    DEFAULT_ENDPOINT = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        model_version: str = "v1.0",
        dimension: int = 1536,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 30.0,
        max_batch_size: int = 128,
        max_input_tokens: int = 8191,
        session: Optional[requests.Session] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            timeout_sec=timeout_sec,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.endpoint_url = endpoint_url or self.DEFAULT_ENDPOINT
        self.session = session or requests.Session()

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def validate_connection(self) -> bool:
        """Validate credentials and endpoint by embedding a probe string."""
        if not self.api_key and "openai.com" in self.endpoint_url:
            raise EmbeddingProviderError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key in config."
            )
        try:
            self.embed("test probe connection")
            return True
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI connection validation failed: {e}") from e

    def _call_api(self, input_data: str | List[str]) -> List[List[float]]:
        headers = self._get_headers()
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_data,
        }
        if self.dimension and "text-embedding-3" in self.model_name:
            payload["dimensions"] = self.dimension

        try:
            response = self.session.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )

            if response.status_code == 429:
                raise EmbeddingRateLimitError(
                    f"OpenAI rate limit exceeded (429): {response.text}"
                )

            if response.status_code >= 500:
                raise EmbeddingProviderError(
                    f"OpenAI server error ({response.status_code}): {response.text}"
                )

            if response.status_code >= 400:
                raise EmbeddingProviderError(
                    f"OpenAI client error ({response.status_code}): {response.text}"
                )

            data = response.json()
            items = data.get("data", [])
            # Sort by index to ensure strictly guaranteed input/output ordering
            items_sorted = sorted(items, key=lambda x: x.get("index", 0))
            vectors = [item["embedding"] for item in items_sorted]
            return vectors

        except requests.Timeout as e:
            raise EmbeddingTimeoutError(
                f"OpenAI request timed out after {self.timeout_sec}s: {e}"
            ) from e
        except (EmbeddingRateLimitError, EmbeddingTimeoutError, EmbeddingProviderError):
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI request failed: {e}") from e

    def embed(self, text: str) -> List[float]:
        results = self._call_api(text)
        if not results:
            raise EmbeddingProviderError("OpenAI returned empty embedding list.")
        return results[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._call_api(texts)
