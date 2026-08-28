"""Base abstraction and registry for pluggable OCR providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from PIL import Image

from rag.ocr.models import OCRPageResult

logger = logging.getLogger(__name__)


class BaseOCRProvider(ABC):
    """Abstract interface for all OCR engine backends."""

    def __init__(self, name: str = "base", **kwargs: Any):
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def process_page(
        self,
        image: Image.Image,
        page_number: int,
        language: str = "eng",
        options: Optional[Dict[str, Any]] = None,
    ) -> OCRPageResult:
        """Execute OCR recognition on a rendered PIL Image.

        Args:
            image: Preprocessed PIL Image of the PDF page.
            page_number: 1-indexed page number.
            language: Tesseract / OCR language code (default: 'eng').
            options: Optional dictionary of engine-specific parameters.

        Returns:
            OCRPageResult containing recognized text, blocks, confidence, and status.
        """
        pass

    def is_available(self) -> bool:
        """Check whether the underlying OCR binary / engine is installed and operational."""
        return True


class OCRProviderRegistry:
    """Registry allowing dynamic registration and retrieval of OCR providers."""

    _providers: Dict[str, Type[BaseOCRProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[BaseOCRProvider]) -> None:
        """Register a provider class with an identifier."""
        cls._providers[name.lower()] = provider_cls
        logger.debug("Registered OCR provider: '%s' (%s)", name, provider_cls.__name__)

    @classmethod
    def get_provider(cls, name: str, **kwargs: Any) -> BaseOCRProvider:
        """Instantiate and return a registered OCR provider."""
        normalized = name.lower()
        if normalized not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"OCR Provider '{name}' is not registered. Available providers: {available}"
            )
        provider_cls = cls._providers[normalized]
        return provider_cls(name=normalized, **kwargs)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List names of all registered providers."""
        return list(cls._providers.keys())
