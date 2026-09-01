"""Domain exceptions for visual asset and figure processing subsystem."""

from __future__ import annotations
from typing import Optional


class AssetError(Exception):
    """Base exception for all asset processing errors."""

    def __init__(self, message: str, asset_id: Optional[str] = None, page_number: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.asset_id = asset_id
        self.page_number = page_number

    def __str__(self) -> str:
        prefix = f"[Asset {self.asset_id}] " if self.asset_id else ""
        page_str = f"(Page {self.page_number}) " if self.page_number else ""
        return f"{prefix}{page_str}{self.message}"


class AssetExtractionError(AssetError):
    """Raised when extracting an image from PDF or scanned page fails."""


class AssetClassificationError(AssetError):
    """Raised when asset classification encounters an unrecoverable failure."""


class AssetOCRError(AssetError):
    """Raised when running OCR on a figure fails."""


class VisionAnalysisError(AssetError):
    """Raised when vision/multimodal diagram analysis fails."""


class AssetStorageError(AssetError):
    """Raised when persisting an asset image to object storage fails."""


class AssetValidationError(AssetError):
    """Raised when an asset fails validation (e.g. invalid dimensions, corrupt bytes)."""
