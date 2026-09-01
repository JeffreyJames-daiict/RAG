"""Asset classifier evaluating visual properties, dimensions, and semantic cues."""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, Optional
from PIL import Image

from rag.assets.models import AssetClassificationResult, ExtractedVisualCandidate
from rag.core.config import AssetConfig
from rag.core.models import AssetType

logger = logging.getLogger(__name__)


class AssetClassifier:
    """Classifies visual candidates into asset types and filters decorative artifacts."""

    def __init__(self, config: Optional[AssetConfig] = None):
        self.config = config or AssetConfig()

    def classify(
        self,
        candidate: ExtractedVisualCandidate,
        caption: Optional[str] = None,
    ) -> AssetClassificationResult:
        """Classify candidate based on image metrics and contextual text."""
        w, h = candidate.width, candidate.height
        if w <= 0 or h <= 0:
            try:
                img = Image.open(io.BytesIO(candidate.image_bytes))
                w, h = img.size
                candidate.width, candidate.height = w, h
            except Exception:
                pass

        area = w * h
        aspect_ratio = (w / max(h, 1)) if w >= h else (h / max(w, 1))

        features = {
            "width": w,
            "height": h,
            "area": area,
            "aspect_ratio": round(aspect_ratio, 2),
            "source_type": candidate.source_type,
        }

        # 1. Check for decorative elements (tiny icons, thin lines, banner bars)
        if self.config.filter_decorative:
            if w < self.config.min_image_width or h < self.config.min_image_height:
                return AssetClassificationResult(
                    asset_type=AssetType.DECORATIVE,
                    asset_subtype="tiny_icon",
                    confidence=0.95,
                    is_decorative=True,
                    reason=f"Dimensions ({w}x{h}) below minimum threshold ({self.config.min_image_width}x{self.config.min_image_height})",
                    features=features,
                )

            if area < self.config.min_image_area:
                return AssetClassificationResult(
                    asset_type=AssetType.DECORATIVE,
                    asset_subtype="small_artifact",
                    confidence=0.90,
                    is_decorative=True,
                    reason=f"Area ({area}px) below minimum threshold ({self.config.min_image_area}px)",
                    features=features,
                )

            if aspect_ratio > self.config.max_aspect_ratio:
                return AssetClassificationResult(
                    asset_type=AssetType.DECORATIVE,
                    asset_subtype="banner_line",
                    confidence=0.85,
                    is_decorative=True,
                    reason=f"Aspect ratio ({aspect_ratio:.1f}) exceeds maximum threshold ({self.config.max_aspect_ratio})",
                    features=features,
                )

        # 2. Textual context analysis (caption and nearby text)
        context_text = ""
        if caption:
            context_text += " " + caption
        if candidate.nearby_text:
            context_text += " " + candidate.nearby_text
        context_lower = context_text.lower()

        # Classification rule evaluation
        if any(kw in context_lower for kw in ["architecture", "pipeline", "system design", "infrastructure", "microservice"]):
            return AssetClassificationResult(
                asset_type=AssetType.ARCHITECTURE_DIAGRAM,
                asset_subtype="system_architecture",
                confidence=0.92,
                is_decorative=False,
                reason="Context indicates system architecture / pipeline",
                features=features,
            )

        if any(kw in context_lower for kw in ["flowchart", "workflow", "decision tree", "flow chart", "sequence diagram"]):
            return AssetClassificationResult(
                asset_type=AssetType.FLOWCHART,
                asset_subtype="process_flow",
                confidence=0.90,
                is_decorative=False,
                reason="Context indicates flowchart or workflow process",
                features=features,
            )

        if any(kw in context_lower for kw in ["diagram", "network topology", "schema", "state machine", "graph"]):
            return AssetClassificationResult(
                asset_type=AssetType.DIAGRAM,
                asset_subtype="technical_diagram",
                confidence=0.88,
                is_decorative=False,
                reason="Context indicates technical diagram",
                features=features,
            )

        if any(kw in context_lower for kw in ["chart", "plot", "histogram", "bar chart", "scatter plot", "trend", "graph of"]):
            return AssetClassificationResult(
                asset_type=AssetType.CHART,
                asset_subtype="data_chart",
                confidence=0.88,
                is_decorative=False,
                reason="Context indicates quantitative chart or graph",
                features=features,
            )

        if any(kw in context_lower for kw in ["table", "matrix", "grid", "comparison table"]):
            return AssetClassificationResult(
                asset_type=AssetType.TABLE_IMAGE,
                asset_subtype="embedded_table",
                confidence=0.85,
                is_decorative=False,
                reason="Context indicates table image",
                features=features,
            )

        if any(kw in context_lower for kw in ["schematic", "blueprint", "cad", "technical drawing"]):
            return AssetClassificationResult(
                asset_type=AssetType.TECHNICAL_DRAWING,
                asset_subtype="schematic",
                confidence=0.85,
                is_decorative=False,
                reason="Context indicates technical schematic/drawing",
                features=features,
            )

        if any(kw in context_lower for kw in ["photo", "photograph", "snapshot", "portrait"]):
            return AssetClassificationResult(
                asset_type=AssetType.PHOTO,
                asset_subtype="photograph",
                confidence=0.80,
                is_decorative=False,
                reason="Context indicates real-world photograph",
                features=features,
            )

        # 3. Default fallback based on size
        return AssetClassificationResult(
            asset_type=AssetType.FIGURE,
            asset_subtype="general_figure",
            confidence=0.75,
            is_decorative=False,
            reason="Standard document visual figure",
            features=features,
        )
