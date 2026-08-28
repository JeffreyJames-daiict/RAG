"""Multi-signal page quality evaluator and layout classifier."""

from __future__ import annotations

import logging
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz

from rag.ocr.language_detector import LanguageDetector
from rag.ocr.models import ContentSource, PageClassificationResult

logger = logging.getLogger(__name__)

# Common printable ASCII + standard typographic characters
_PRINTABLE_SET = set(string.printable).union({
    "—", "–", "“", "”", "‘", "’", "•", "✓", "✗", "×", "±", "€", "£", "¥", "§", "©", "®", "™", "°",
})

# Regular expression matching normal words (alphabetic or alphanumeric with basic hyphens)
_WORD_RE = re.compile(r"^[A-Za-z0-9\u00C0-\u024F]+(?:[-'][A-Za-z0-9\u00C0-\u024F]+)*$")


class PageQualityAnalyzer:
    """Evaluates native text extraction quality, language, and classifies pages as native, scanned, or mixed."""

    def __init__(
        self,
        min_native_text_quality: float = 0.70,
        min_text_length: int = 40,
        image_coverage_threshold: float = 0.35,
        default_language: str = "eng",
    ):
        self.min_native_text_quality = min_native_text_quality
        self.min_text_length = min_text_length
        self.image_coverage_threshold = image_coverage_threshold
        self.language_detector = LanguageDetector(default_language=default_language)

    def calculate_text_quality(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Compute multi-signal quality score for extracted text.

        Returns:
            Tuple of (overall_quality_score [0.0..1.0], metrics_dict).
        """
        if not text or not text.strip():
            return 0.0, {
                "printable_ratio": 0.0,
                "meaningful_word_ratio": 0.0,
                "replacement_ratio": 0.0,
                "text_length": 0,
            }

        clean_text = text.strip()
        total_chars = len(clean_text)

        # 1. Printable Character Ratio
        printable_count = sum(1 for ch in clean_text if ch in _PRINTABLE_SET or ch.isalnum() or ch.isspace())
        printable_ratio = printable_count / total_chars if total_chars else 0.0

        # 2. Replacement / Garbled Control Character Ratio & Unexpected Symbols
        replacement_count = clean_text.count("\ufffd") + sum(1 for ch in clean_text if ord(ch) < 32 and ch not in "\n\r\t")
        unexpected_symbols = (total_chars - printable_count) + replacement_count
        replacement_ratio = replacement_count / total_chars if total_chars else 0.0
        unexpected_symbol_ratio = unexpected_symbols / total_chars if total_chars else 0.0

        # 3. Meaningful Word Ratio
        words = clean_text.split()
        if words:
            valid_word_count = sum(1 for w in words if _WORD_RE.match(w.strip(".,;:!?()[]\"'")) and len(w) >= 2)
            meaningful_word_ratio = valid_word_count / len(words)
        else:
            meaningful_word_ratio = 0.0

        # Weighted composite score
        quality_score = (
            (printable_ratio * 0.40)
            + (meaningful_word_ratio * 0.45)
            + (max(0.0, 1.0 - (replacement_ratio * 5.0)) * 0.15)
        )
        quality_score = max(0.0, min(1.0, quality_score))

        metrics = {
            "printable_ratio": round(printable_ratio, 4),
            "meaningful_word_ratio": round(meaningful_word_ratio, 4),
            "replacement_ratio": round(replacement_ratio, 4),
            "unexpected_symbol_count": unexpected_symbols,
            "unexpected_symbol_ratio": round(unexpected_symbol_ratio, 4),
            "text_length": total_chars,
        }
        return quality_score, metrics

    def analyze_page(
        self,
        page: fitz.Page,
        page_number: int,
        native_text: Optional[str] = None,
        native_blocks: Optional[List[Any]] = None,
    ) -> PageClassificationResult:
        """Classify a PDF page as digital (NATIVE), SCANNED, or MIXED."""
        if native_text is None:
            native_text = page.get_text("text")

        text_quality, metrics = self.calculate_text_quality(native_text)
        text_length = len(native_text.strip())

        # Perform automatic language detection
        lang_res = self.language_detector.detect_language(native_text)

        # Inspect images and calculate page image coverage
        images = page.get_images()
        image_count = len(images)
        image_coverage = 0.0

        page_rect = page.rect
        page_area = (page_rect.width * page_rect.height) if page_rect.width and page_rect.height else 1.0

        # Approximate total image area on page
        total_img_area = 0.0
        for img_info in images:
            xref = img_info[0]
            try:
                img_rects = page.get_image_rects(xref)
                for r in img_rects:
                    total_img_area += (r.width * r.height)
            except Exception:
                pass

        image_coverage = min(1.0, total_img_area / page_area) if page_area > 0 else 0.0

        # ── Decision Tree ────────────────────────────────────────────────────
        # Case 1: Empty or extremely short text (< 15 chars)
        if text_length < 15:
            if image_count > 0 or image_coverage > 0.2:
                source_type = ContentSource.SCANNED
                ocr_required = True
                reason = "No usable native text detected on image-based page"
            else:
                source_type = ContentSource.NATIVE
                ocr_required = False
                reason = "Blank or minimal text page with no images"

        # Case 2: Poor text quality / garbled characters
        elif text_quality < self.min_native_text_quality:
            source_type = ContentSource.SCANNED
            ocr_required = True
            reason = (
                f"Corrupted or low-quality native text (quality {text_quality:.2f} "
                f"< {self.min_native_text_quality})"
            )

        # Case 3: Mixed page (native text exists, but large scanned diagram/image region present)
        elif image_count >= 1 and image_coverage >= self.image_coverage_threshold and text_length < 500:
            source_type = ContentSource.MIXED
            ocr_required = True
            reason = (
                f"Native text exists but page has significant image coverage ({image_coverage:.1%})"
            )

        # Case 4: High quality digital native page
        else:
            source_type = ContentSource.NATIVE
            ocr_required = False
            reason = "Native text extraction passed quality checks"

        logger.debug(
            "Page %d classification: %s (quality=%.2f, lang=%s [%.2f], ocr_required=%s, img_cov=%.1f%%)",
            page_number, source_type.value, text_quality, lang_res.language, lang_res.confidence,
            ocr_required, image_coverage * 100,
        )

        return PageClassificationResult(
            page_number=page_number,
            source_type=source_type,
            native_text_quality=round(text_quality, 4),
            ocr_required=ocr_required,
            reason=reason,
            image_count=image_count,
            text_length=text_length,
            printable_ratio=metrics["printable_ratio"],
            meaningful_word_ratio=metrics["meaningful_word_ratio"],
            image_coverage=round(image_coverage, 4),
            detected_language=lang_res.language,
            language_confidence=lang_res.confidence,
            unexpected_symbol_count=metrics.get("unexpected_symbol_count", 0),
            unexpected_symbol_ratio=metrics.get("unexpected_symbol_ratio", 0.0),
            metrics=metrics,
        )
