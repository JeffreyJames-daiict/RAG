"""Mock and fallback OCR Provider for deterministic testing and fallbacks."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from rag.ocr.base import BaseOCRProvider
from rag.ocr.models import OCRBlock, OCRPageResult, OCRStatus

logger = logging.getLogger(__name__)


class MockOCRProvider(BaseOCRProvider):
    """Mock OCR provider for fast, deterministic unit and integration testing."""

    def __init__(self, name: str = "mock", default_text: Optional[str] = None, **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.default_text = default_text
        self.mock_blocks: List[OCRBlock] = []

    def is_available(self) -> bool:
        return True

    def set_mock_response(self, text: str, blocks: Optional[List[OCRBlock]] = None) -> None:
        """Configure the response returned on subsequent calls."""
        self.default_text = text
        if blocks is not None:
            self.mock_blocks = blocks

    def process_page(
        self,
        image: Image.Image,
        page_number: int,
        language: str = "eng",
        options: Optional[Dict[str, Any]] = None,
    ) -> OCRPageResult:
        start_time = time.perf_counter()
        w, h = image.size

        text = self.default_text if self.default_text is not None else f"Mock OCR recognized text on page {page_number}."
        if self.mock_blocks:
            blocks = self.mock_blocks
        else:
            blocks = [
                OCRBlock(
                    text=text,
                    bbox=(50.0, 50.0, float(w - 50), float(h - 50)),
                    confidence=0.95,
                    block_type=0,
                    block_no=1,
                    lines=[text],
                )
            ]

        return OCRPageResult(
            page_number=page_number,
            text=text,
            blocks=blocks,
            confidence=0.95,
            quality_score=0.95,
            status=OCRStatus.PASS,
            provider_name=self.name,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
