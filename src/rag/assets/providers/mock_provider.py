"""Deterministic Mock Vision Provider for unit and integration testing."""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

from rag.assets.models import FigureAnalysisResult, FigureEntityRelationship
from rag.assets.providers.base import BaseVisionProvider
from rag.core.models import AssetType

logger = logging.getLogger(__name__)


class MockVisionProvider(BaseVisionProvider):
    """Deterministic mock vision provider for testing architecture and diagram analysis."""

    def __init__(self, model_name: str = "mock-vision-v1", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def analyze_figure(
        self,
        image_bytes: bytes,
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        asset_type: AssetType = AssetType.FIGURE,
        mime_type: str = "image/png",
        context: Optional[Any] = None,
    ) -> FigureAnalysisResult:
        """Generate structured deterministic semantic analysis."""
        title = caption or f"{asset_type.value} illustration"
        entities: List[str] = []
        relationships: List[FigureEntityRelationship] = []

        # Extract entities from OCR text or caption words
        all_words = []
        if ocr_text:
            all_words.extend([w.strip(".,;:()") for w in ocr_text.split() if len(w) > 2])
        if caption:
            all_words.extend([w.strip(".,;:()") for w in caption.split() if len(w) > 2])

        # De-duplicate while preserving order
        seen = set()
        for w in all_words:
            cap_w = w.capitalize()
            if cap_w not in seen and len(entities) < 8:
                seen.add(cap_w)
                entities.append(cap_w)

        if not entities:
            # Fallback default entities for diagrams
            if asset_type in (AssetType.ARCHITECTURE_DIAGRAM, AssetType.DIAGRAM, AssetType.FLOWCHART):
                entities = ["Client", "API Gateway", "Application Service", "Database"]
            else:
                entities = ["Visual Component", "Data Node"]

        # Build synthetic directed relationships between consecutive entities
        for i in range(len(entities) - 1):
            src = entities[i]
            tgt = entities[i + 1]
            rel = "connects_to"
            if "database" in tgt.lower() or "store" in tgt.lower():
                rel = "persists_data_to"
            elif "service" in tgt.lower() or "gateway" in tgt.lower():
                rel = "routes_request_to"
            elif "user" in src.lower() or "client" in src.lower():
                rel = "sends_request_to"

            relationships.append(FigureEntityRelationship(source=src, relationship=rel, target=tgt))

        description = (
            f"The {asset_type.value.lower().replace('_', ' ')} '{title}' illustrates "
            f"the interaction and data flow across {', '.join(entities)}. "
            f"Key relationships include " +
            ", ".join(f"{r.source} {r.relationship.replace('_', ' ')} {r.target}" for r in relationships) + "."
            if relationships else f"The visual asset depicts {title}."
        )

        return FigureAnalysisResult(
            description=description,
            entities=entities,
            relationships=relationships,
            summary=f"Analysis of {title} comprising {len(entities)} entities.",
            provider_name=self.model_name,
            status="SUCCESS",
        )
