"""Comprehensive unit and integration test suite for the Content Normalization Layer.

Covers:
 1. Multiple spaces normalization
 2. Tabs normalization
 3. Excessive blank lines
 4. Unicode normalization (NFC)
 5. Zero-width character removal
 6. Broken PDF line wrapping
 7. Hyphenated words dehyphenation
 8. Legitimate hyphenated words preservation
 9. Paragraph preservation
10. Repeated header detection & removal
11. Repeated footer detection & removal
12. Standalone page number removal
13. OCR low-confidence content flagging
14. OCR artifacts (excessive repetition, broken words)
15. Table preservation (structured table integrity)
16. Duplicate native text + OCR resolution
17. Empty content validation
18. Garbage / high symbol density content flagging
19. Multilingual text safety
20. Mixed-content page normalization & pipeline integration
"""

import pytest
from rag.core.config import NormalizationConfig
from rag.normalization import (
    ContentNormalizer,
    DuplicateResolver,
    ExtractionMethod,
    HeaderFooterDetector,
    NormalizedContentBlock,
    NormalizedContentType,
    NormalizedDocument,
    OCRNormalizer,
    TextNormalizer,
)
from rag.parsing.models import (
    HeadingElement,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    TableElement,
    TextElement,
)


class TestTextNormalization:
    """Tests 1–9: Pure text normalization, Unicode, hyphens, and whitespace."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer(
            unicode_form="NFC",
            repair_soft_hyphens=True,
            remove_standalone_page_numbers=True,
        )

    def test_01_multiple_spaces(self, normalizer):
        raw = "This    is   a    document   with   irregular     spacing."
        cleaned, actions = normalizer.normalize(raw)
        assert cleaned == "This is a document with irregular spacing."
        assert "whitespace_normalization" in actions

    def test_02_tabs(self, normalizer):
        raw = "This    is   a    document.\n\nThis\tis\tanother\tparagraph."
        cleaned, actions = normalizer.normalize(raw)
        assert "\t" not in cleaned
        assert cleaned == "This is a document.\n\nThis is another paragraph."
        assert "whitespace_normalization" in actions

    def test_03_excessive_blank_lines(self, normalizer):
        raw = "Paragraph One.\n\n\n\n\n\nParagraph Two."
        cleaned, _ = normalizer.normalize(raw)
        assert cleaned == "Paragraph One.\n\nParagraph Two."

    def test_04_unicode_normalization(self, normalizer):
        # Combining accents vs precomposed
        decomposed_e = "e\u0301"  # é in NFD
        raw = f"Caf{decomposed_e} and fa\uFB01le"  # includes ligature ﬁ (\uFB01)
        cleaned, actions = normalizer.normalize(raw)
        assert cleaned == "Café and faﬁle" or "Café" in cleaned
        assert "unicode_normalization" in actions

    def test_05_zero_width_characters(self, normalizer):
        raw = "Sec\u200Bure\u200CData\uFEFFTransmission\u00ADPacket"
        cleaned, actions = normalizer.normalize(raw)
        assert cleaned == "SecureDataTransmissionPacket"

    def test_06_broken_pdf_line_wrapping(self, normalizer):
        raw = "This is a sentence that was\nbroken across multiple lines\nby the PDF parser."
        cleaned, _ = normalizer.normalize(raw)
        assert cleaned == "This is a sentence that was broken across multiple lines by the PDF parser."

    def test_07_hyphenated_words_dehyphenation(self, normalizer):
        raw = "This is a docu-\nment containing informa-\ntion about machine learning."
        cleaned, actions = normalizer.normalize(raw)
        assert "document" in cleaned
        assert "information" in cleaned
        assert "dehyphenation" in actions

    def test_08_legitimate_hyphenated_words(self, normalizer):
        raw = "We deployed a state-of-the-\nart, real-\ntime, end-to-\nend architecture."
        cleaned, _ = normalizer.normalize(raw)
        assert "state-of-the-art" in cleaned
        assert "real-time" in cleaned
        assert "end-to-end" in cleaned

    def test_09_paragraph_preservation(self, normalizer):
        raw = (
            "First paragraph line one\n"
            "continues on line two.\n\n"
            "Second paragraph starts here\n"
            "and finishes here."
        )
        cleaned, _ = normalizer.normalize(raw)
        assert cleaned == (
            "First paragraph line one continues on line two.\n\n"
            "Second paragraph starts here and finishes here."
        )


class TestHeaderFooterAndPageNumbers:
    """Tests 10–12: Repetitive headers, footers, and page number removal."""

    def test_10_repeated_header_detection(self):
        detector = HeaderFooterDetector(header_repeat_threshold=3, footer_repeat_threshold=3)
        pages = [
            "ACME CORP CONFIDENTIAL\nPage content 1\nFooter info",
            "ACME CORP CONFIDENTIAL\nPage content 2\nFooter info",
            "ACME CORP CONFIDENTIAL\nPage content 3\nFooter info",
            "ACME CORP CONFIDENTIAL\nPage content 4\nDifferent footer",
        ]
        result = detector.analyze_document(pages)
        assert "acme corp confidential" in result.detected_headers

        cleaned, rem_h, rem_f = detector.strip_headers_and_footers(pages[0], result)
        assert "ACME CORP CONFIDENTIAL" not in cleaned
        assert "Page content 1" in cleaned
        assert len(rem_h) >= 1

    def test_11_repeated_footer_detection(self):
        detector = HeaderFooterDetector(header_repeat_threshold=3, footer_repeat_threshold=3)
        pages = [
            "Title 1\nBody 1\nInternal Use Only — Confidential",
            "Title 2\nBody 2\nInternal Use Only — Confidential",
            "Title 3\nBody 3\nInternal Use Only — Confidential",
        ]
        result = detector.analyze_document(pages)
        assert "internal use only — confidential" in result.detected_footers

        cleaned, _, rem_f = detector.strip_headers_and_footers(pages[0], result)
        assert "Internal Use Only" not in cleaned
        assert "Body 1" in cleaned
        assert len(rem_f) == 1

    def test_12_standalone_page_numbers(self):
        normalizer = TextNormalizer(remove_standalone_page_numbers=True)
        raw = "Some important header\n\nPage 4 of 20\n\nBody text starts here.\n\n- 4 -\n"
        cleaned, actions = normalizer.normalize(raw)
        assert "Page 4 of 20" not in cleaned
        assert "- 4 -" not in cleaned
        assert "Body text starts here." in cleaned
        assert "page_numbers_removed" in actions


class TestOCRNormalization:
    """Tests 13–14: OCR low-confidence, repetition, and spacing fixes."""

    @pytest.fixture
    def ocr_norm(self):
        return OCRNormalizer(confidence_threshold=0.60, max_symbol_ratio=0.30)

    def test_13_ocr_low_confidence_flagging(self, ocr_norm):
        block = NormalizedContentBlock(
            normalized_text="Low quality extracted text snippet",
            confidence_score=0.42,
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "low_confidence" in processed.quality_flags

    def test_14_ocr_artifacts(self, ocr_norm):
        # Repeated noise & broken spacing
        block = NormalizedContentBlock(
            normalized_text="Notice: t e s t pattern and aaaaaaaaa noise -------",
            confidence_score=0.85,
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "test" in processed.normalized_text
        assert "excessive_character_repetition" in processed.quality_flags
        assert "repeated_symbol_artifacts" in processed.quality_flags


class TestTablePreservationAndDuplicates:
    """Tests 15–16: Table block preservation and cross-source duplicate resolution."""

    def test_15_table_preservation(self):
        table = TableElement(
            table_id="table_p1_1",
            headers=["Product", "Revenue", "Year"],
            rows=[["Widget A", "$10,000", "2024"], ["Widget B", "$20,000", "2025"]],
            raw_markdown="| Product | Revenue | Year |\n|---|---|---|\n| Widget A | $10,000 | 2024 |\n| Widget B | $20,000 | 2025 |",
        )
        parsed_doc = ParsedDocument(
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Introduction to annual sales report.",
                    tables=[table],
                    elements=[
                        TextElement(element_id="txt_1", page_number=1, content="Introduction to annual sales report."),
                        table,
                    ],
                )
            ],
            tables=[table],
        )
        normalizer = ContentNormalizer()
        norm_doc = normalizer.normalize(parsed_doc)

        table_blocks = [b for b in norm_doc.all_active_blocks if b.content_type == NormalizedContentType.TABLE]
        assert len(table_blocks) == 1
        assert table_blocks[0].table_element == table
        assert "| Widget A | $10,000 | 2024 |" in table_blocks[0].normalized_text

    def test_16_duplicate_native_and_ocr(self):
        resolver = DuplicateResolver(similarity_threshold=0.80)
        b_native = NormalizedContentBlock(
            block_id="b_native",
            normalized_text="Enterprise AI Platform Architecture and Ingestion Pipeline Overview.",
            extraction_method=ExtractionMethod.NATIVE_PDF,
            confidence_score=0.95,
        )
        b_ocr = NormalizedContentBlock(
            block_id="b_ocr",
            normalized_text="Enterprise AI Platform Architecture and Ingestion Pipeline Overview.",
            extraction_method=ExtractionMethod.OCR,
            confidence_score=0.88,
        )
        blocks, decisions = resolver.resolve_page_duplicates([b_native, b_ocr])
        assert len(decisions) == 1
        assert decisions[0].selected_source == "native_pdf"
        assert decisions[0].suppressed_source == "ocr"
        assert b_native.is_suppressed is False
        assert b_ocr.is_suppressed is True


class TestQualityValidationAndMultilingual:
    """Tests 17–20: Content quality, multilingual integrity, and end-to-end normalization."""

    def test_17_empty_content_validation(self):
        parsed_doc = ParsedDocument(
            pages=[
                ParsedPage(
                    page_number=1,
                    text="   \n\n\t  \n  ",
                    elements=[TextElement(element_id="txt_1", page_number=1, content="   \n\t  ")],
                )
            ]
        )
        normalizer = ContentNormalizer(NormalizationConfig(min_text_length=5))
        norm_doc = normalizer.normalize(parsed_doc)
        assert len(norm_doc.all_active_blocks) == 0

    def test_18_garbage_content_flagging(self):
        ocr_norm = OCRNormalizer(max_symbol_ratio=0.30)
        block = NormalizedContentBlock(
            normalized_text="!!@@##$$%%^^&&**((__++==||\\\\//",
            confidence_score=0.45,
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "high_symbol_density" in processed.quality_flags
        assert "suspected_corruption" in processed.quality_flags

    def test_19_multilingual_text_safety(self):
        normalizer = TextNormalizer()
        # Japanese, German, Spanish, French
        multilingual_text = (
            "日本語のテキストと機械学習モデル。\n"
            "Übermäßige Verzögerungen müssen vermieden werden.\n"
            "El niño comió piña en la mañana y dijo: ¡delicioso!"
        )
        cleaned, _ = normalizer.normalize(multilingual_text)
        assert "日本語のテキスト" in cleaned
        assert "Übermäßige" in cleaned
        assert "niño comió piña" in cleaned

    def test_20_mixed_content_page_normalization(self):
        table = TableElement(
            table_id="tab_1",
            headers=["Metrics", "Score"],
            rows=[["Accuracy", "0.98"], ["Latency", "12ms"]],
        )
        p1 = ParsedPage(
            page_number=1,
            text="CONFIDENTIAL COMPANY REPORT\nExecutive Overview: Model Performance.\nPage 1 of 5",
            headings=["Executive Overview: Model Performance."],
            tables=[table],
            elements=[
                HeadingElement(element_id="h1", page_number=1, content="Executive Overview: Model Performance."),
                TextElement(element_id="t1", page_number=1, content="Overall accuracy reached 98% with 12ms p99 latency."),
                table,
            ],
        )
        parsed_doc = ParsedDocument(pages=[p1], tables=[table])
        normalizer = ContentNormalizer()
        norm_doc = normalizer.normalize(parsed_doc, document_id="doc-123", filename="report.pdf")

        assert norm_doc.num_pages == 1
        active = norm_doc.all_active_blocks
        assert len(active) >= 3  # Heading, text, and table
        assert any(b.content_type == NormalizedContentType.HEADING for b in active)
        assert any(b.content_type == NormalizedContentType.TABLE for b in active)
        assert norm_doc.normalization_diagnostics["blocks_processed"] >= 3

    def test_21_ocr_graphical_divider_stripping(self):
        ocr_norm = OCRNormalizer()
        block = NormalizedContentBlock(
            normalized_text="Section Heading\n————-_-—»}\nNext paragraph content follows here.",
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "————-_-—»}" not in processed.normalized_text
        assert "Section Heading" in processed.normalized_text
        assert "Next paragraph content follows here." in processed.normalized_text
        assert "graphical_dividers_stripped" in processed.normalization_actions

    def test_22_ocr_url_protocol_repair(self):
        ocr_norm = OCRNormalizer()
        block = NormalizedContentBlock(
            normalized_text="Audited Property: hitps://inheritx-ai-alpha.vercel.app and htps://example.com/api",
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "https://inheritx-ai-alpha.vercel.app" in processed.normalized_text
        assert "https://example.com/api" in processed.normalized_text
        assert "url_protocols_repaired" in processed.normalization_actions

    def test_23_ocr_bullet_marker_standardization(self):
        ocr_norm = OCRNormalizer()
        block = NormalizedContentBlock(
            normalized_text="° Item Alpha\n• Item Beta\n* Item Gamma",
            extraction_method=ExtractionMethod.OCR,
        )
        processed = ocr_norm.process_block(block)
        assert "- Item Alpha" in processed.normalized_text
        assert "- Item Beta" in processed.normalized_text
        assert "- Item Gamma" in processed.normalized_text
        assert "bullet_markers_standardized" in processed.normalization_actions

