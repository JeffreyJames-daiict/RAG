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
        context: Optional[Any] = None,
        frequency: int = 1,
    ) -> AssetClassificationResult:
        """Classify candidate based on image metrics, surrounding text, and layout signals."""
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
            "frequency": frequency,
            "page_coverage": round(candidate.page_coverage, 3),
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

        # 2. Repeated asset across pages -> Likely Logo or Header/Footer icon
        if frequency >= 3 and not caption:
            if area < 150000:
                return AssetClassificationResult(
                    asset_type=AssetType.LOGO,
                    asset_subtype="repeating_header_logo",
                    confidence=0.90,
                    is_decorative=True,
                    reason=f"Asset repeats across {frequency} pages without unique caption (likely branding/logo)",
                    features=features,
                )

        # 3. Textual context analysis (caption, section, nearby text)
        context_parts = []
        if caption:
            context_parts.append(caption)
        if candidate.nearby_text:
            context_parts.append(candidate.nearby_text)
        if candidate.section_heading:
            context_parts.append(candidate.section_heading)
        if context:
            if hasattr(context, "section_heading") and context.section_heading:
                context_parts.append(context.section_heading)
            if hasattr(context, "nearby_text") and context.nearby_text:
                context_parts.append(context.nearby_text)

        context_text = " ".join(context_parts).lower()

        # Check for logos / watermarks / icons
        if any(kw in context_text for kw in ["logo", "trademark", "brandmark", "corporate identity"]):
            return AssetClassificationResult(
                asset_type=AssetType.LOGO,
                asset_subtype="brand_logo",
                confidence=0.88,
                is_decorative=True,
                reason="Context indicates company logo or branding asset",
                features=features,
            )

        if any(kw in context_text for kw in ["watermark", "draft", "confidential watermark", "background stamp"]):
            return AssetClassificationResult(
                asset_type=AssetType.WATERMARK,
                asset_subtype="document_watermark",
                confidence=0.92,
                is_decorative=True,
                reason="Context indicates document watermark",
                features=features,
            )

        if any(kw in context_text for kw in ["signature", "signed by", "authorized signature", "initials"]):
            return AssetClassificationResult(
                asset_type=AssetType.SIGNATURE,
                asset_subtype="handwritten_signature",
                confidence=0.90,
                is_decorative=False,
                reason="Context indicates signature image",
                features=features,
            )

        # Technical architecture / system diagram
        if any(kw in context_text for kw in ["architecture", "pipeline", "system design", "infrastructure", "microservice", "component diagram"]):
            return AssetClassificationResult(
                asset_type=AssetType.ARCHITECTURE_DIAGRAM,
                asset_subtype="system_architecture",
                confidence=0.92,
                is_decorative=False,
                reason="Context indicates system architecture / pipeline",
                features=features,
            )

        # Flowchart / workflow
        if any(kw in context_text for kw in ["flowchart", "workflow", "decision tree", "flow chart", "sequence diagram", "activity diagram", "state machine"]):
            return AssetClassificationResult(
                asset_type=AssetType.FLOWCHART,
                asset_subtype="process_flow",
                confidence=0.90,
                is_decorative=False,
                reason="Context indicates flowchart or workflow process",
                features=features,
            )

        # Technical drawing / schematics
        if any(kw in context_text for kw in ["schematic", "blueprint", "cad", "technical drawing", "circuit diagram", "wiring diagram"]):
            return AssetClassificationResult(
                asset_type=AssetType.TECHNICAL_DRAWING,
                asset_subtype="schematic",
                confidence=0.88,
                is_decorative=False,
                reason="Context indicates technical schematic/drawing",
                features=features,
            )

        # Chart / Plot / Graph
        if any(kw in context_text for kw in ["chart", "plot", "histogram", "bar chart", "scatter plot", "trend", "pie chart", "box plot", "heat map"]):
            return AssetClassificationResult(
                asset_type=AssetType.CHART,
                asset_subtype="data_chart",
                confidence=0.88,
                is_decorative=False,
                reason="Context indicates quantitative chart or graph",
                features=features,
            )

        if any(kw in context_text for kw in ["network topology", "dependency graph", "graph", "knowledge graph"]):
            return AssetClassificationResult(
                asset_type=AssetType.GRAPH,
                asset_subtype="network_graph",
                confidence=0.86,
                is_decorative=False,
                reason="Context indicates network or relational graph",
                features=features,
            )

        # Table image
        if any(kw in context_text for kw in ["table", "matrix", "grid", "comparison table"]):
            return AssetClassificationResult(
                asset_type=AssetType.TABLE_IMAGE,
                asset_subtype="embedded_table",
                confidence=0.85,
                is_decorative=False,
                reason="Context indicates table image",
                features=features,
            )

        # Screenshot / UI
        if any(kw in context_text for kw in ["screenshot", "screen capture", "user interface", "dashboard", "gui", "webpage"]):
            return AssetClassificationResult(
                asset_type=AssetType.SCREENSHOT,
                asset_subtype="ui_screenshot",
                confidence=0.85,
                is_decorative=False,
                reason="Context indicates UI screenshot / dashboard",
                features=features,
            )

        # Equation / formula
        if any(kw in context_text for kw in ["equation", "formula", "theorem", "math", "derivation"]):
            return AssetClassificationResult(
                asset_type=AssetType.EQUATION_IMAGE,
                asset_subtype="math_equation",
                confidence=0.85,
                is_decorative=False,
                reason="Context indicates mathematical equation / formula",
                features=features,
            )

        # Portrait / Author photo
        if any(kw in context_text for kw in ["portrait", "author photo", "headshot", "biography", "speaker"]):
            return AssetClassificationResult(
                asset_type=AssetType.PORTRAIT,
                asset_subtype="author_portrait",
                confidence=0.82,
                is_decorative=False,
                reason="Context indicates author portrait / headshot",
                features=features,
            )

        # General photo
        if any(kw in context_text for kw in ["photo", "photograph", "snapshot", "microscopy"]):
            return AssetClassificationResult(
                asset_type=AssetType.PHOTO,
                asset_subtype="photograph",
                confidence=0.80,
                is_decorative=False,
                reason="Context indicates real-world photograph",
                features=features,
            )

        # Illustration
        if any(kw in context_text for kw in ["illustration", "artwork", "cartoon", "sketch"]):
            return AssetClassificationResult(
                asset_type=AssetType.ILLUSTRATION,
                asset_subtype="conceptual_illustration",
                confidence=0.80,
                is_decorative=False,
                reason="Context indicates conceptual illustration",
                features=features,
            )

        # Diagram general
        if any(kw in context_text for kw in ["diagram", "schema", "overview", "layout"]):
            return AssetClassificationResult(
                asset_type=AssetType.DIAGRAM,
                asset_subtype="technical_diagram",
                confidence=0.82,
                is_decorative=False,
                reason="Context indicates technical diagram",
                features=features,
            )

        # 4. Default fallback based on size
        return AssetClassificationResult(
            asset_type=AssetType.FIGURE,
            asset_subtype="general_figure",
            confidence=0.75,
            is_decorative=False,
            reason="Standard document visual figure",
            features=features,
        )
