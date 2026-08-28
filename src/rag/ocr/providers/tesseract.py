"""Tesseract OCR Provider implementation using pytesseract."""

from __future__ import annotations

import logging
import shutil
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from rag.ocr.base import BaseOCRProvider
from rag.ocr.models import OCRBlock, OCRPageResult, OCRStatus

logger = logging.getLogger(__name__)


class TesseractOCRProvider(BaseOCRProvider):
    """OCR provider wrapping the Google Tesseract OCR engine via pytesseract."""

    def __init__(self, name: str = "tesseract", tesseract_cmd: Optional[str] = None, **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.tesseract_cmd = tesseract_cmd
        if self.tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            except Exception as e:
                logger.warning("Failed to configure custom tesseract_cmd '%s': %s", self.tesseract_cmd, e)

    _warned_missing: bool = False

    def is_available(self) -> bool:
        """Verify pytesseract import and binary in PATH."""
        try:
            import pytesseract
            cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
            return bool(shutil.which(cmd))
        except Exception:
            return False

    def process_page(
        self,
        image: Image.Image,
        page_number: int,
        language: str = "eng",
        options: Optional[Dict[str, Any]] = None,
    ) -> OCRPageResult:
        """Run Tesseract recognition on image and build structured bounding-box blocks."""
        start_time = time.perf_counter()
        opts = options or {}
        psm = opts.get("psm", 3)
        oem = opts.get("oem", 3)
        custom_config = opts.get("config", f"--oem {oem} --psm {psm}")

        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as e:
            return OCRPageResult(
                page_number=page_number,
                text="",
                blocks=[],
                confidence=0.0,
                quality_score=0.0,
                status=OCRStatus.SKIPPED,
                error_message=f"pytesseract library is not installed: {e}",
                provider_name=self.name,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not self.is_available():
            if not TesseractOCRProvider._warned_missing:
                logger.warning(
                    "Tesseract binary is not installed in PATH. OCR processing skipped. "
                    "Install tesseract (e.g. 'sudo apt-get install tesseract-ocr') to enable OCR on scanned pages."
                )
                TesseractOCRProvider._warned_missing = True
            return OCRPageResult(
                page_number=page_number,
                text="",
                blocks=[],
                confidence=0.0,
                quality_score=0.0,
                status=OCRStatus.SKIPPED,
                error_message="Tesseract binary not found in PATH.",
                provider_name=self.name,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            # Extract detailed word-level dictionary data
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=custom_config,
                output_type=Output.DICT,
            )

            blocks_map: Dict[int, List[Dict[str, Any]]] = {}
            confidences: List[float] = []

            n_boxes = len(data.get("text", []))
            for i in range(n_boxes):
                text_word = str(data["text"][i]).strip()
                conf = float(data["conf"][i])
                if not text_word or conf < 0:
                    continue

                block_num = data["block_num"][i]
                if block_num not in blocks_map:
                    blocks_map[block_num] = []

                blocks_map[block_num].append({
                    "text": text_word,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": conf,
                    "line_num": data["line_num"][i],
                })
                confidences.append(conf)

            # Assemble consolidated OCRBlock instances per paragraph/block
            ocr_blocks: List[OCRBlock] = []
            full_text_parts: List[str] = []

            for b_idx, (b_num, words) in enumerate(sorted(blocks_map.items()), start=1):
                if not words:
                    continue

                min_x = min(w["x"] for w in words)
                min_y = min(w["y"] for w in words)
                max_x = max(w["x"] + w["w"] for w in words)
                max_y = max(w["y"] + w["h"] for w in words)
                block_bbox = (float(min_x), float(min_y), float(max_x), float(max_y))

                # Group words into lines
                lines_map: Dict[int, List[str]] = {}
                for w in words:
                    lines_map.setdefault(w["line_num"], []).append(w["text"])

                block_lines = [" ".join(l_words) for _, l_words in sorted(lines_map.items())]
                block_text = "\n".join(block_lines).strip()
                block_conf = sum(w["conf"] for w in words) / (len(words) * 100.0) if words else 0.0

                if block_text:
                    ocr_blocks.append(
                        OCRBlock(
                            text=block_text,
                            bbox=block_bbox,
                            confidence=block_conf,
                            block_type=0,
                            block_no=b_idx,
                            lines=block_lines,
                        )
                    )
                    full_text_parts.append(block_text)

            avg_conf = (sum(confidences) / (len(confidences) * 100.0)) if confidences else 0.0
            full_text = "\n\n".join(full_text_parts)

            status = OCRStatus.PASS if full_text and avg_conf >= 0.5 else (
                OCRStatus.LOW_QUALITY if full_text else OCRStatus.FAILED
            )

            return OCRPageResult(
                page_number=page_number,
                text=full_text,
                blocks=ocr_blocks,
                confidence=avg_conf,
                quality_score=avg_conf,
                status=status,
                raw_response=None,
                provider_name=self.name,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Tesseract OCR exception on page %d: %s", page_number, e, exc_info=True)
            return OCRPageResult(
                page_number=page_number,
                text="",
                blocks=[],
                confidence=0.0,
                quality_score=0.0,
                status=OCRStatus.FAILED,
                error_message=f"Tesseract OCR error: {e}",
                provider_name=self.name,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
