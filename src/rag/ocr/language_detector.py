"""Multi-strategy language detector for PDF pages and OCR blocks.

Provides:
- Fast dictionary & stopword frequency analysis for European languages (eng, deu, fra, spa, ita, nld, etc.)
- Unicode script recognition (Latin, Cyrillic, CJK, Arabic, Devanagari)
- ISO 639-1 <-> ISO 639-2 / Tesseract language code normalization
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Stopword profiles for common languages (ISO 639-2 / Tesseract format)
_STOPWORD_PROFILES: Dict[str, Set[str]] = {
    "eng": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
        "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
        "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
        "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
        "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
        "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    },
    "deu": {
        "der", "die", "das", "und", "in", "den", "von", "zu", "mit", "ist", "des", "nicht",
        "eine", "ein", "dem", "sich", "auf", "f\u00fcr", "fuer", "dass", "im", "als", "auch",
        "es", "an", "werden", "aus", "er", "hat", "einer", "bei", "ihrem", "wie", "nach",
        "vor", "durch", "noch", "sie", "ihr", "oder", "wenn", "aber", "kann", "um", "wir",
        "sind", "nur", "einem", "haben", "zum", "zur", "unter", "dieser", "diese", "dieses",
        "ueber", "\u00fcber", "seite", "jahr", "alle", "immer", "ohne", "wird", "wurde", "dann",
    },
    "fra": {
        "le", "la", "les", "et", "de", "des", "en", "un", "une", "du", "est", "que", "qui",
        "dans", "a", "\u00e0", "pour", "par", "sur", "pas", "plus", "avec", "ce", "au", "aux",
        "ne", "se", "ou", "mais", "comme", "tout", "nous", "vous", "ils", "son", "sa", "ses",
        "on", "fait", "elle", "si", "leur", "bien", "peut", "faire", "ces", "sans", "été",
    },
    "spa": {
        "el", "la", "los", "las", "de", "en", "y", "a", "un", "una", "es", "que", "del",
        "por", "para", "con", "no", "se", "su", "al", "lo", "como", "mas", "m\u00e1s", "pero",
        "sus", "le", "ya", "o", "este", "esta", "estos", "estas", "sin", "sobre", "todo",
        "tambien", "tambi\u00e9n", "entre", "cuando", "era", "muy", "hasta", "desde",
    },
    "ita": {
        "il", "la", "lo", "i", "gli", "le", "di", "e", "in", "un", "una", "uno", "per",
        "con", "su", "da", "non", "si", "sono", "del", "della", "dei", "delle", "che",
        "al", "alla", "ai", "alle", "come", "ma", "anche", "questo", "questa", "questi",
    },
    "nld": {
        "de", "het", "een", "en", "van", "ik", "te", "dat", "die", "in", "is", "voor",
        "niet", "met", "zijn", "op", "aan", "er", "om", "ook", "als", "omdat", "maar",
        "over", "kunnen", "jaar", "door", "bij", "uit", "naar", "hebben", "worden",
    },
}

# ISO 639-1 to ISO 639-2 / Tesseract code mapping
_ISO_MAP = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "nl": "nld",
    "pt": "por",
    "ru": "rus",
    "zh": "chi_sim",
    "ja": "jpn",
    "ar": "ara",
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection on a text snippet or page."""
    language: str              # ISO 639-2 code (e.g. 'eng', 'deu', 'fra')
    confidence: float          # 0.0 to 1.0
    detected_script: str       # 'latin', 'cyrillic', 'cjk', 'arabic', etc.
    word_count: int = 0
    scores: Dict[str, float] = None

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


class LanguageDetector:
    """Fast, deterministic statistical language detector for documents and OCR text."""

    def __init__(self, default_language: str = "eng"):
        self.default_language = self.normalize_code(default_language)

    @staticmethod
    def normalize_code(code: str) -> str:
        """Convert 2-letter or 3-letter code to normalized 3-letter Tesseract/ISO-639-2."""
        cleaned = code.lower().strip()
        return _ISO_MAP.get(cleaned, cleaned)

    @staticmethod
    def detect_script(text: str) -> str:
        """Identify primary Unicode script family from text characters."""
        if not text:
            return "latin"

        counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "arabic": 0, "devanagari": 0}
        for ch in text:
            code = ord(ch)
            if (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
                counts["latin"] += 1
            elif 0x0400 <= code <= 0x04FF:
                counts["cyrillic"] += 1
            elif (0x4E00 <= code <= 0x9FFF) or (0x3040 <= code <= 0x30FF):
                counts["cjk"] += 1
            elif 0x0600 <= code <= 0x06FF:
                counts["arabic"] += 1
            elif 0x0900 <= code <= 0x097F:
                counts["devanagari"] += 1

        top_script = max(counts.items(), key=lambda x: x[1])
        return top_script[0] if top_script[1] > 0 else "latin"

    def detect_language(self, text: str, min_words: int = 4) -> LanguageDetectionResult:
        """Identify language from word frequency distribution and stopword overlap."""
        if not text or not text.strip():
            return LanguageDetectionResult(
                language=self.default_language,
                confidence=0.0,
                detected_script="latin",
                word_count=0,
                scores={},
            )

        script = self.detect_script(text)
        if script == "cyrillic":
            return LanguageDetectionResult(language="rus", confidence=0.85, detected_script="cyrillic")
        elif script == "cjk":
            return LanguageDetectionResult(language="chi_sim", confidence=0.85, detected_script="cjk")
        elif script == "arabic":
            return LanguageDetectionResult(language="ara", confidence=0.85, detected_script="arabic")

        # Clean words for Latin script stopword matching
        raw_words = re.findall(r"\b[a-zA-Z\u00C0-\u024F]{2,}\b", text.lower())
        total_words = len(raw_words)

        if total_words < min_words:
            return LanguageDetectionResult(
                language=self.default_language,
                confidence=0.30,
                detected_script="latin",
                word_count=total_words,
                scores={self.default_language: 0.30},
            )

        # Count stopword hits per language
        lang_hits: Dict[str, int] = {}
        for lang_code, stopwords in _STOPWORD_PROFILES.items():
            hits = sum(1 for w in raw_words if w in stopwords)
            lang_hits[lang_code] = hits

        # Calculate overlap ratios
        scores: Dict[str, float] = {}
        for lang_code, hits in lang_hits.items():
            scores[lang_code] = round(hits / total_words, 4)

        best_lang, best_hits = max(lang_hits.items(), key=lambda x: x[1])

        if best_hits > 0:
            # Confidence based on score margin over second-best
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0]
            confidence = min(1.0, 0.50 + (margin * 2.0))
            return LanguageDetectionResult(
                language=best_lang,
                confidence=round(confidence, 4),
                detected_script="latin",
                word_count=total_words,
                scores=scores,
            )

        return LanguageDetectionResult(
            language=self.default_language,
            confidence=0.50,
            detected_script="latin",
            word_count=total_words,
            scores=scores,
        )
