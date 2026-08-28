"""Tests for LanguageDetector and language-aware parsing/chunking."""

import pytest
from rag.ocr.language_detector import LanguageDetector, LanguageDetectionResult
from rag.parsing.pdf_parser import PDFParser
from rag.parsing.chunker import RecursiveTextChunker
from rag.core.config import OCRConfig


class TestLanguageDetector:
    """Unit tests for LanguageDetector stopword and script heuristics."""

    def setup_method(self):
        self.detector = LanguageDetector(default_language="eng")

    def test_english_text_detection(self):
        text = (
            "This document describes the architecture of the enterprise retrieval augmented "
            "generation pipeline with multi-table extraction and vector search capabilities."
        )
        res = self.detector.detect_language(text)
        assert res.language == "eng"
        assert res.confidence >= 0.50
        assert res.detected_script == "latin"

    def test_german_text_detection(self):
        text = (
            "Dieses Dokument beschreibt die Anforderungen für das neue Produkt. "
            "Die Kalender werden mit mattem oder glänzendem Papier gedruckt und "
            "sind in verschiedenen Formaten für alle Kunden verfügbar."
        )
        res = self.detector.detect_language(text)
        assert res.language == "deu"
        assert res.confidence >= 0.50
        assert res.detected_script == "latin"

    def test_french_text_detection(self):
        text = (
            "Ce document présente les spécifications techniques pour le nouveau système. "
            "Les utilisateurs peuvent sélectionner les options de configuration dans le menu principal."
        )
        res = self.detector.detect_language(text)
        assert res.language == "fra"
        assert res.confidence >= 0.50
        assert res.detected_script == "latin"

    def test_short_text_returns_default_with_low_confidence(self):
        text = "Hello"
        res = self.detector.detect_language(text)
        assert res.language == "eng"
        assert res.confidence <= 0.40

    def test_empty_text_returns_default(self):
        res = self.detector.detect_language("")
        assert res.language == "eng"
        assert res.confidence == 0.0

    def test_cyrillic_script_detection(self):
        text = "Это документ на русском языке для тестирования системы."
        res = self.detector.detect_language(text)
        assert res.language == "rus"
        assert res.detected_script == "cyrillic"

    def test_cjk_script_detection(self):
        text = "这是一个用于测试多语言处理的中文文档。"
        res = self.detector.detect_language(text)
        assert res.language == "chi_sim"
        assert res.detected_script == "cjk"

    def test_iso_normalization(self):
        assert self.detector.normalize_code("en") == "eng"
        assert self.detector.normalize_code("de") == "deu"
        assert self.detector.normalize_code("fr") == "fra"
        assert self.detector.normalize_code("eng") == "eng"
