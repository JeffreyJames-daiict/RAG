"""OCR providers package."""

from rag.ocr.base import OCRProviderRegistry
from rag.ocr.providers.mock_ocr import MockOCRProvider
from rag.ocr.providers.tesseract import TesseractOCRProvider

# Register standard providers in global registry
OCRProviderRegistry.register("tesseract", TesseractOCRProvider)
OCRProviderRegistry.register("mock", MockOCRProvider)

__all__ = [
    "TesseractOCRProvider",
    "MockOCRProvider",
]
