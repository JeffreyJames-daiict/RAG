"""OCR result quality evaluator and multi-attempt retry manager."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from rag.ocr.base import BaseOCRProvider
from rag.ocr.models import OCRPageResult, OCRStatus
from rag.ocr.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"^[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*$")


class OCRQualityEvaluator:
    """Evaluates recognized OCR text quality and executes controlled retries."""

    def __init__(
        self,
        min_confidence: float = 0.60,
        max_retries: int = 2,
        default_preprocessing: str = "standard",
    ):
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.default_preprocessing = default_preprocessing

    def evaluate_result(self, result: OCRPageResult) -> OCRPageResult:
        """Calculate quality metrics on an OCR result and assign operational status."""
        text = result.text.strip()
        if not text:
            result.quality_score = 0.0
            result.status = OCRStatus.FAILED
            return result

        total_chars = len(text)
        printable_count = sum(1 for ch in text if ch.isprintable() or ch.isspace())
        printable_ratio = printable_count / total_chars if total_chars else 0.0

        words = text.split()
        if words:
            valid_word_count = sum(1 for w in words if _WORD_RE.match(w.strip(".,;:!?()[]\"'")) and len(w) >= 2)
            word_ratio = valid_word_count / len(words)
        else:
            word_ratio = 0.0

        # Weighted quality score
        quality = (result.confidence * 0.45) + (printable_ratio * 0.30) + (word_ratio * 0.25)
        quality = max(0.0, min(1.0, quality))
        result.quality_score = round(quality, 4)

        if quality >= self.min_confidence:
            result.status = OCRStatus.PASS
        else:
            result.status = OCRStatus.LOW_QUALITY

        return result

    def process_with_retry(
        self,
        provider: BaseOCRProvider,
        raw_image: Image.Image,
        page_number: int,
        language: str = "eng",
        options: Optional[Dict[str, Any]] = None,
    ) -> OCRPageResult:
        """Run OCR with progressive preprocessing retries on low-quality recognition."""
        # Progressive profile chain for retries
        profiles = [self.default_preprocessing, "aggressive", "minimal"]
        max_attempts = min(self.max_retries + 1, len(profiles))

        best_result: Optional[OCRPageResult] = None

        for attempt_idx in range(max_attempts):
            profile = profiles[attempt_idx]
            logger.debug(
                "OCR page %d attempt %d/%d (profile='%s', provider='%s')",
                page_number, attempt_idx + 1, max_attempts, profile, provider.name,
            )

            # Preprocess image
            processed_img = ImagePreprocessor.preprocess(raw_image, profile=profile)

            # Execute OCR
            result = provider.process_page(
                image=processed_img,
                page_number=page_number,
                language=language,
                options=options,
            )
            result.attempt = attempt_idx + 1
            result.preprocessing_profile = profile

            # Evaluate output quality
            self.evaluate_result(result)

            if best_result is None or result.quality_score > best_result.quality_score:
                best_result = result

            # If provider is unavailable or skipped, do not retry
            if result.status == OCRStatus.SKIPPED:
                return result

            # If quality passed threshold, break early
            if result.status == OCRStatus.PASS:
                logger.debug(
                    "OCR page %d passed on attempt %d with quality %.2f",
                    page_number, attempt_idx + 1, result.quality_score,
                )
                return result
            else:
                if attempt_idx + 1 < max_attempts:
                    logger.info(
                        "OCR page %d low quality on attempt %d (score=%.2f < min=%.2f) -> retrying",
                        page_number, attempt_idx + 1, result.quality_score, self.min_confidence,
                    )

        return best_result or OCRPageResult(
            page_number=page_number,
            text="",
            blocks=[],
            confidence=0.0,
            quality_score=0.0,
            status=OCRStatus.FAILED,
            error_message="All OCR retry attempts exhausted.",
        )
