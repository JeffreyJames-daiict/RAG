"""Figure semantic analyzer orchestrating vision providers with failure isolation."""

from __future__ import annotations

import logging
import time
from typing import Optional

from rag.assets.exceptions import VisionAnalysisError
from rag.assets.models import FigureAnalysisResult, FigureEntityRelationship
from rag.assets.providers import get_vision_provider
from rag.assets.providers.base import BaseVisionProvider
from rag.core.config import AssetConfig
from rag.core.models import AssetType, IndexingPolicy

logger = logging.getLogger(__name__)


class FigureAnalyzer:
    """Orchestrates multimodal semantic analysis for diagrams, charts, and figures."""

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        vision_provider: Optional[BaseVisionProvider] = None,
    ):
        self.config = config or AssetConfig()
        self.vision_config = self.config.vision
        self.provider = vision_provider or get_vision_provider(self.vision_config)

    def should_analyze_with_vision(
        self,
        asset_type: AssetType,
        indexing_policy: Optional[IndexingPolicy] = None,
    ) -> bool:
        """Production check: Determine if expensive vision analysis should run."""
        if not self.vision_config.enabled:
            return False

        if indexing_policy is not None:
            return indexing_policy in (IndexingPolicy.VISION_ANALYSIS, IndexingPolicy.FULL_MULTIMODAL)

        # Fallback type-based evaluation when policy is not provided
        valuable_types = {
            AssetType.ARCHITECTURE_DIAGRAM,
            AssetType.DIAGRAM,
            AssetType.FLOWCHART,
            AssetType.CHART,
            AssetType.GRAPH,
            AssetType.TECHNICAL_DRAWING,
            AssetType.FIGURE,
            AssetType.SCREENSHOT,
            AssetType.TABLE_IMAGE,
            AssetType.EQUATION_IMAGE,
        }
        return asset_type in valuable_types

    def analyze(
        self,
        image_bytes: bytes,
        asset_type: AssetType,
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        mime_type: str = "image/png",
        context: Optional[Any] = None,
        indexing_policy: Optional[IndexingPolicy] = None,
    ) -> FigureAnalysisResult:
        """Run vision analysis with retry logic and error isolation."""
        if not self.should_analyze_with_vision(asset_type, indexing_policy=indexing_policy):
            return FigureAnalysisResult(
                description=caption or (ocr_text[:200] if ocr_text else ""),
                entities=[],
                relationships=[],
                summary=caption or "",
                provider_name="skipped",
                status="SKIPPED",
            )

        max_retries = self.vision_config.max_retries
        backoff = self.vision_config.retry_backoff_sec
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                res = self.provider.analyze_figure(
                    image_bytes=image_bytes,
                    caption=caption,
                    ocr_text=ocr_text,
                    asset_type=asset_type,
                    mime_type=mime_type,
                    context=context,
                )
                return res

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Vision analysis attempt %d/%d failed for asset type %s: %s",
                    attempt, max_retries, asset_type.value, exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0

        # Failure isolation: return structured fallback rather than crashing document ingestion
        logger.error("Vision analysis failed after %d retries: %s. Preserving asset without vision metadata.", max_retries, last_error)
        return FigureAnalysisResult(
            description=caption or (f"Text inside figure: {ocr_text}" if ocr_text else ""),
            entities=[],
            relationships=[],
            summary=caption or "Visual figure",
            provider_name=getattr(self.provider, "model_name", "unknown"),
            status="FAILED",
            error_message=str(last_error),
        )
