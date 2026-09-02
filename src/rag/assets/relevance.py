"""Heuristic relevance scoring and indexing policy selection engine for visual assets."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rag.assets.models import (
    AssetClassificationResult,
    AssetContextInfo,
    AssetRelevanceResult,
    ExtractedVisualCandidate,
    FigureOCRResult,
)
from rag.core.config import AssetConfig
from rag.core.models import AssetType, IndexingPolicy

logger = logging.getLogger(__name__)

# Common technical and analytical keywords indicating high informational value
_TECHNICAL_TERMS = frozenset([
    "architecture", "pipeline", "service", "gateway", "database", "server", "client",
    "retrieval", "embedding", "vector", "encoder", "decoder", "transformer", "attention",
    "cluster", "network", "node", "layer", "interface", "protocol", "workflow", "process",
    "algorithm", "model", "accuracy", "loss", "precision", "recall", "latency", "throughput",
    "benchmark", "dataset", "evaluation", "schema", "table", "column", "row", "query",
    "flow", "input", "output", "step", "state", "system", "component", "module", "api",
    "request", "response", "cloud", "aws", "azure", "docker", "kubernetes", "cpu", "gpu",
])

_FIGURE_LABEL_PATTERN = re.compile(
    r"^(?:Figure|Fig\.?|Diagram|Chart|Exhibit|Table)\s*\d+",
    re.IGNORECASE,
)


class AssetRelevanceScorer:
    """Calculates a normalized relevance score [0.0 - 1.0] for an extracted visual asset

    and assigns an IndexingPolicy (IGNORE, STORE_ONLY, OCR_ONLY, VISION_ANALYSIS).
    """

    def __init__(self, config: Optional[AssetConfig] = None):
        self.config = config or AssetConfig()
        self.weights = self.config.weights

    def evaluate(
        self,
        candidate: ExtractedVisualCandidate,
        classification: AssetClassificationResult,
        caption: Optional[str] = None,
        ocr_result: Optional[FigureOCRResult] = None,
        context: Optional[AssetContextInfo] = None,
        frequency: int = 1,
    ) -> AssetRelevanceResult:
        """Compute composite relevance score and determine indexing policy."""
        if not self.config.relevance_enabled:
            # When disabled, default to vision for informative types, store_only for decorative
            if classification.is_decorative:
                return AssetRelevanceResult(
                    relevance_score=0.10,
                    relevance_reason="Decorative element (relevance evaluation disabled)",
                    indexing_policy=IndexingPolicy.STORE_ONLY,
                )
            return AssetRelevanceResult(
                relevance_score=0.80,
                relevance_reason="Default active policy (relevance evaluation disabled)",
                indexing_policy=IndexingPolicy.VISION_ANALYSIS,
            )

        features: Dict[str, Any] = {}
        reason_parts: List[str] = []

        # 1. Base score by asset type
        base_scores = {
            AssetType.ARCHITECTURE_DIAGRAM: 0.65,
            AssetType.FLOWCHART: 0.60,
            AssetType.TECHNICAL_DRAWING: 0.60,
            AssetType.CHART: 0.60,
            AssetType.GRAPH: 0.60,
            AssetType.DIAGRAM: 0.55,
            AssetType.TABLE_IMAGE: 0.50,
            AssetType.SCREENSHOT: 0.50,
            AssetType.EQUATION_IMAGE: 0.50,
            AssetType.FIGURE: 0.40,
            AssetType.PHOTO: 0.35,
            AssetType.ILLUSTRATION: 0.35,
            AssetType.PORTRAIT: 0.20,
            AssetType.SIGNATURE: 0.15,
            AssetType.LOGO: 0.10,
            AssetType.WATERMARK: 0.05,
            AssetType.ICON: 0.05,
            AssetType.DECORATIVE: 0.05,
            AssetType.UNKNOWN: 0.30,
        }
        score = base_scores.get(classification.asset_type, 0.40)
        reason_parts.append(f"Base score {score:.2f} for {classification.asset_type.value}")
        features["base_score"] = score

        # 2. Caption scoring
        if caption and caption.strip():
            score += self.weights.caption
            reason_parts.append(f"+{self.weights.caption:.2f} (caption detected)")
            features["has_caption"] = True

            # Explicit figure label (e.g. "Figure 3: System Overview")
            if _FIGURE_LABEL_PATTERN.search(caption.strip()):
                score += self.weights.figure_label
                reason_parts.append(f"+{self.weights.figure_label:.2f} (figure label)")
                features["has_figure_label"] = True
        else:
            features["has_caption"] = False

        # 3. OCR text scoring
        ocr_text = (ocr_result.ocr_text if ocr_result else "") or ""
        ocr_words = [w.lower().strip(".,;:()[]{}") for w in ocr_text.split() if w.strip()]
        features["ocr_word_count"] = len(ocr_words)

        if len(ocr_words) >= 4:
            score += self.weights.meaningful_ocr_length
            reason_parts.append(f"+{self.weights.meaningful_ocr_length:.2f} ({len(ocr_words)} OCR words)")
            features["meaningful_ocr"] = True

            # Check technical keyword presence
            tech_matches = [w for w in ocr_words if w in _TECHNICAL_TERMS]
            if tech_matches:
                score += self.weights.technical_ocr
                reason_parts.append(f"+{self.weights.technical_ocr:.2f} ({len(tech_matches)} tech terms in OCR)")
                features["tech_terms_count"] = len(tech_matches)
        elif ocr_result and ocr_result.ocr_applied and len(ocr_words) == 0 and not caption:
            score += self.weights.empty_ocr_penalty
            reason_parts.append(f"{self.weights.empty_ocr_penalty:.2f} (empty OCR)")

        # 4. Context scoring (surrounding text and section heading)
        context_str = ""
        if context:
            if context.section_heading:
                context_str += " " + context.section_heading
            if context.nearby_text:
                context_str += " " + context.nearby_text
            if context.document_title:
                context_str += " " + context.document_title
        elif candidate.nearby_text or candidate.section_heading:
            context_str = f"{candidate.section_heading or ''} {candidate.nearby_text or ''}"

        if context_str:
            ctx_words = [w.lower().strip(".,;:()[]{}") for w in context_str.split()]
            ctx_tech_matches = [w for w in ctx_words if w in _TECHNICAL_TERMS]
            if len(ctx_tech_matches) >= 2:
                score += self.weights.technical_context
                reason_parts.append(f"+{self.weights.technical_context:.2f} (technical section context)")
                features["context_tech_matches"] = len(ctx_tech_matches)

        # 5. Visual geometry and layout properties
        area = candidate.width * candidate.height
        aspect = (candidate.width / max(candidate.height, 1)) if candidate.width >= candidate.height else (candidate.height / max(candidate.width, 1))
        if area >= 30000 and aspect <= 3.5:
            score += self.weights.informative_aspect_size
            features["good_geometry"] = True

        if candidate.page_coverage >= 0.15:
            score += self.weights.large_page_coverage
            reason_parts.append(f"+{self.weights.large_page_coverage:.2f} (large page coverage {candidate.page_coverage:.1%})")
            features["large_coverage"] = True

        # 6. Penalties for noise / decorative / repeated assets
        if frequency >= 2 and not caption:
            # Recurring logo/header penalty scaled by frequency
            rep_pen = self.weights.repeated_asset_penalty * min(frequency - 1, 3)
            score += rep_pen
            reason_parts.append(f"{rep_pen:.2f} (repeated across {frequency} pages)")
            features["repeated_penalty"] = rep_pen

        if classification.asset_type == AssetType.LOGO or classification.asset_subtype == "brand_logo":
            score += self.weights.logo_penalty
            reason_parts.append(f"{self.weights.logo_penalty:.2f} (logo penalty)")

        if classification.asset_type == AssetType.WATERMARK:
            score += self.weights.watermark_penalty
            reason_parts.append(f"{self.weights.watermark_penalty:.2f} (watermark penalty)")

        if classification.asset_type in (AssetType.ICON, AssetType.DECORATIVE):
            score += self.weights.tiny_icon_penalty
            reason_parts.append(f"{self.weights.tiny_icon_penalty:.2f} (decorative icon penalty)")

        # 7. Clamp final score to [0.0, 1.0]
        final_score = max(0.0, min(1.0, round(score, 3)))
        features["final_score"] = final_score

        # 8. Policy selection based on thresholds
        if classification.is_decorative or final_score < self.config.store_only_threshold:
            policy = IndexingPolicy.IGNORE if (candidate.width <= 32 and candidate.height <= 32) else IndexingPolicy.STORE_ONLY
            reason_parts.append(f"→ Selected policy: {policy.value} (score < {self.config.store_only_threshold:.2f})")
        elif final_score >= self.config.analyze_threshold:
            policy = IndexingPolicy.VISION_ANALYSIS
            reason_parts.append(f"→ Selected policy: {policy.value} (score >= {self.config.analyze_threshold:.2f})")
        else:
            # Middle band [store_only_threshold, analyze_threshold)
            # If OCR text contains meaningful substance, use OCR_ONLY indexing, otherwise STORE_ONLY
            if len(ocr_words) >= 5:
                policy = IndexingPolicy.OCR_ONLY
                reason_parts.append(f"→ Selected policy: {policy.value} (moderate score with OCR text)")
            else:
                policy = IndexingPolicy.STORE_ONLY
                reason_parts.append(f"→ Selected policy: {policy.value} (moderate score without substantive text)")

        full_reason = "; ".join(reason_parts)
        return AssetRelevanceResult(
            relevance_score=final_score,
            relevance_reason=full_reason,
            indexing_policy=policy,
            features=features,
        )
