"""Unit tests for figure-specific OCR extraction."""

import io
from PIL import Image, ImageDraw
import pytest

from rag.assets.ocr import FigureOCRRunner
from rag.core.config import AssetConfig
from rag.ocr.providers.tesseract import TesseractOCRProvider


def test_figure_ocr_extraction():
    img = Image.new("RGB", (300, 150), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Database", fill="black")
    draw.text((20, 70), "API Gateway", fill="black")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    runner = FigureOCRRunner(config=AssetConfig(figure_ocr_enabled=True))
    result = runner.process_figure_image(img_bytes, page_number=1)

    # When tesseract is installed in test environment, text should be extracted
    assert isinstance(result.ocr_text, str)
    assert isinstance(result.words, list)
    if runner.ocr_provider.is_available():
        assert result.ocr_applied is True
