"""Image preprocessing profiles and pipeline for OCR optimization."""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Applies configurable image enhancement profiles before OCR execution."""

    PROFILES = ("minimal", "standard", "aggressive")

    @classmethod
    def preprocess(cls, image: Image.Image, profile: str = "standard") -> Image.Image:
        """Apply image transformations according to specified profile.

        Profiles:
            minimal:    Converts to RGB and ensures consistent color mode.
            standard:   Grayscale conversion, contrast auto-leveling, and sharpening.
            aggressive: Grayscale, high-contrast enhancement, median denoising, and binarization.
        """
        norm_profile = profile.lower()
        if norm_profile not in cls.PROFILES:
            logger.warning("Unknown preprocessing profile '%s'; defaulting to 'standard'", profile)
            norm_profile = "standard"

        if norm_profile == "minimal":
            return cls._apply_minimal(image)
        elif norm_profile == "standard":
            return cls._apply_standard(image)
        elif norm_profile == "aggressive":
            return cls._apply_aggressive(image)
        return image

    @staticmethod
    def _apply_minimal(image: Image.Image) -> Image.Image:
        """Minimal processing: clean RGB or L mode."""
        if image.mode not in ("RGB", "L"):
            return image.convert("RGB")
        return image

    @staticmethod
    def _apply_standard(image: Image.Image) -> Image.Image:
        """Standard processing: grayscale, auto-contrast, mild unsharp mask."""
        gray = image.convert("L")
        # Auto-level contrast to stretch histogram
        autocontrast = ImageOps.autocontrast(gray, cutoff=1)
        # Mild sharpening
        enhancer = ImageEnhance.Sharpness(autocontrast)
        sharpened = enhancer.enhance(1.5)
        return sharpened

    @staticmethod
    def _apply_aggressive(image: Image.Image) -> Image.Image:
        """Aggressive processing: noise filtering, high contrast, Otsu-like thresholding."""
        gray = image.convert("L")
        # Median filter to remove salt-and-pepper noise
        denoised = gray.filter(ImageFilter.MedianFilter(size=3))
        # High contrast boost
        contrast_enhancer = ImageEnhance.Contrast(denoised)
        high_contrast = contrast_enhancer.enhance(2.0)
        # Auto-threshold to black and white
        autocontrast = ImageOps.autocontrast(high_contrast, cutoff=2)
        # Point binarization threshold at mid-gray
        threshold = 140
        binarized = autocontrast.point(lambda p: 255 if p > threshold else 0)
        return binarized
