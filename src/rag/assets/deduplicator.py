"""Exact SHA-256 and Perceptual Hash deduplication engine for visual assets."""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

from rag.assets.models import FigureAnalysisResult, FigureOCRResult

logger = logging.getLogger(__name__)


def compute_sha256(image_bytes: bytes) -> str:
    """Compute deterministic SHA-256 hash of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def compute_phash(image_bytes: bytes, hash_size: int = 8) -> str:
    """Compute deterministic 64-bit perceptual difference hash (dHash) using PIL.

    dHash tracks horizontal gradients, making it highly invariant to scaling,
    compression artifacts, subtle contrast shifts, and minor rendering variations.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            resample_filter = getattr(Image, "Resampling", Image).BILINEAR
            gray = img.convert("L").resize((hash_size + 1, hash_size), resample=resample_filter)
            pixels = list(gray.tobytes())

            # Compare adjacent pixels row by row
            diff = []
            width = hash_size + 1
            for row in range(hash_size):
                row_start = row * width
                for col in range(hash_size):
                    left = pixels[row_start + col]
                    right = pixels[row_start + col + 1]
                    diff.append(1 if left > right else 0)

            # Convert 64 booleans into 16-character hexadecimal string
            decimal_val = 0
            for bit in diff:
                decimal_val = (decimal_val << 1) | bit
            return f"{decimal_val:016x}"

    except Exception as exc:
        logger.debug("Failed to compute perceptual hash, falling back to sha256 prefix: %s", exc)
        return compute_sha256(image_bytes)[:16]


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate hamming distance (number of differing bits) between two hex hashes."""
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count("1")
    except Exception:
        return 64 if hash1 != hash2 else 0


class AssetDeduplicator:
    """Tracks visual assets across document pages to detect exact & near-identical duplicates,

    maintain page occurrence frequencies, and cache/reuse semantic analyses.
    """

    def __init__(
        self,
        dedup_enabled: bool = True,
        phash_enabled: bool = True,
        phash_distance_threshold: int = 5,
    ):
        self.dedup_enabled = dedup_enabled
        self.phash_enabled = phash_enabled
        self.phash_distance_threshold = phash_distance_threshold

        # sha256 -> first seen asset_id
        self._sha_to_asset_id: Dict[str, str] = {}
        # sha256 -> occurrence count
        self._sha_frequencies: Dict[str, int] = {}
        # sha256 -> list of page numbers where it appeared
        self._sha_pages: Dict[str, List[int]] = {}

        # list of (phash_str, asset_id, sha256)
        self._phash_records: List[Tuple[str, str, str]] = []

        # Analysis cache: sha256 / phash -> (FigureAnalysisResult, FigureOCRResult)
        self._analysis_cache: Dict[str, Tuple[Optional[FigureAnalysisResult], Optional[FigureOCRResult]]] = {}

    def reset(self) -> None:
        """Clear session state for new document processing."""
        self._sha_to_asset_id.clear()
        self._sha_frequencies.clear()
        self._sha_pages.clear()
        self._phash_records.clear()
        self._analysis_cache.clear()

    def evaluate_asset(
        self,
        image_bytes: bytes,
        asset_id: str,
        page_number: int,
    ) -> Tuple[str, str, bool, Optional[str], int]:
        """Evaluate an asset candidate for exact and perceptual duplication.

        Returns:
            Tuple of:
            - content_hash (SHA-256)
            - perceptual_hash (dHash hex)
            - is_duplicate (True if exact or perceptual match found)
            - canonical_asset_id (asset_id of first occurrence, or None if new)
            - frequency (total occurrences seen so far)
        """
        sha = compute_sha256(image_bytes)
        phash = compute_phash(image_bytes) if self.phash_enabled else sha[:16]

        # Track frequency and pages
        self._sha_frequencies[sha] = self._sha_frequencies.get(sha, 0) + 1
        freq = self._sha_frequencies[sha]
        if sha not in self._sha_pages:
            self._sha_pages[sha] = []
        self._sha_pages[sha].append(page_number)

        if not self.dedup_enabled:
            return sha, phash, False, None, freq

        # 1. Exact SHA-256 match
        if sha in self._sha_to_asset_id:
            canonical_id = self._sha_to_asset_id[sha]
            logger.debug(
                "Exact duplicate asset detected (SHA=%s, page=%d, original_asset=%s, freq=%d)",
                sha[:12], page_number, canonical_id, freq,
            )
            return sha, phash, True, canonical_id, freq

        # 2. Perceptual match (near-identical image)
        if self.phash_enabled:
            for existing_phash, existing_asset_id, existing_sha in self._phash_records:
                dist = hamming_distance(phash, existing_phash)
                if dist <= self.phash_distance_threshold:
                    logger.debug(
                        "Perceptual duplicate asset detected (dist=%d <= %d, page=%d, original_asset=%s)",
                        dist, self.phash_distance_threshold, page_number, existing_asset_id,
                    )
                    # Register this sha mapping too
                    self._sha_to_asset_id[sha] = existing_asset_id
                    return sha, phash, True, existing_asset_id, freq

        # 3. New unique asset
        self._sha_to_asset_id[sha] = asset_id
        if self.phash_enabled:
            self._phash_records.append((phash, asset_id, sha))

        return sha, phash, False, None, freq

    def get_frequency(self, content_hash: str) -> int:
        """Get occurrence frequency of an image hash across document pages."""
        return self._sha_frequencies.get(content_hash, 1)

    def cache_analysis(
        self,
        content_hash: str,
        perceptual_hash: Optional[str],
        analysis: Optional[FigureAnalysisResult],
        ocr: Optional[FigureOCRResult],
    ) -> None:
        """Store analysis results in cache for reuse across duplicate occurrences."""
        if content_hash:
            self._analysis_cache[content_hash] = (analysis, ocr)
        if perceptual_hash:
            self._analysis_cache[perceptual_hash] = (analysis, ocr)

    def get_cached_analysis(
        self,
        content_hash: str,
        perceptual_hash: Optional[str] = None,
    ) -> Optional[Tuple[Optional[FigureAnalysisResult], Optional[FigureOCRResult]]]:
        """Retrieve previously executed analysis for an exact or near-duplicate hash."""
        if content_hash in self._analysis_cache:
            return self._analysis_cache[content_hash]

        if perceptual_hash and self.phash_enabled:
            if perceptual_hash in self._analysis_cache:
                return self._analysis_cache[perceptual_hash]
            # Check near matches in cache
            for cached_key, result in self._analysis_cache.items():
                if len(cached_key) == 16:  # phash length
                    if hamming_distance(perceptual_hash, cached_key) <= self.phash_distance_threshold:
                        return result

        return None
