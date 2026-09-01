"""Figure-specific OCR runner reusing Tesseract OCR and preprocessing."""

from __future__ import annotations

import io
import logging
from typing import Optional
from PIL import Image

from rag.assets.models import FigureOCRResult
from rag.core.config import AssetConfig
from rag.ocr.preprocessor import ImagePreprocessor
from rag.ocr.providers.tesseract import TesseractOCRProvider

logger = logging.getLogger(__name__)


class FigureOCRRunner:
    """Runs targeted OCR on extracted figure and diagram images."""

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        ocr_provider: Optional[TesseractOCRProvider] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.config = config or AssetConfig()
        self.ocr_provider = ocr_provider or TesseractOCRProvider()
        self.preprocessor = preprocessor or ImagePreprocessor()

    def process_figure_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        language: str = "eng",
    ) -> FigureOCRResult:
        """Run OCR on image bytes and return clean extracted text and confidence."""
        if not self.config.figure_ocr_enabled:
            return FigureOCRResult(ocr_applied=False)

        if not self.ocr_provider.is_available():
            logger.debug("Tesseract is not available for figure OCR.")
            return FigureOCRResult(ocr_applied=False)

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Preprocess for clean diagram text extraction
            processed_img, _ = self.preprocessor.process(pil_img, profile_name="standard")

            # Execute OCR with sparse text / diagram PSM options (PSM 11 or 3)
            page_res = self.ocr_provider.process_page(
                image=processed_img,
                page_number=page_number,
                language=language,
                options={"psm": 11},
            )

            # Filter words by minimum confidence
            valid_words = []
            for b in page_res.blocks:
                for line in b.lines:
                    valid_words.extend(line.split())

            raw_text = page_res.text.strip()
            if not raw_text and valid_words:
                raw_text = " ".join(valid_words)

            return FigureOCRResult(
                ocr_text=raw_text,
                confidence=page_res.confidence,
                words=valid_words,
                tokens_count=len(valid_words),
                ocr_applied=True,
            )

        except Exception as exc:
            logger.warning("Figure OCR warning on page %d: %s", page_number, exc)
            return FigureOCRResult(
                ocr_text="",
                confidence=0.0,
                words=[],
                tokens_count=0,
                ocr_applied=True,
            )
