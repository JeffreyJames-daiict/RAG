"""Abstract interface for multimodal vision and figure analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from rag.assets.models import FigureAnalysisResult
from rag.core.models import AssetType


class BaseVisionProvider(ABC):
    """Abstract interface for multimodal vision providers analyzing document figures/diagrams."""

    def __init__(self, model_name: str = "default", **kwargs):
        self.model_name = model_name

    @abstractmethod
    def analyze_figure(
        self,
        image_bytes: bytes,
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        asset_type: AssetType = AssetType.FIGURE,
        mime_type: str = "image/png",
    ) -> FigureAnalysisResult:
        """Analyze a visual figure/diagram and return structured semantic descriptions,

        entities, and relationships.
        """
