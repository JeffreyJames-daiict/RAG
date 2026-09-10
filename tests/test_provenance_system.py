"""Comprehensive unit and integration tests for Citation and Provenance subsystem.

Covers:
1. Valid citations and high confidence grounding
2. Missing citations on substantive claims
3. Invalid / hallucinated citation IDs
4. Cross-tenant citation security violations
5. Wrong document version mismatches
6. Citation mismatch (claim contradicts/fails to overlap with cited evidence)
7. Asset citation (figures, diagrams, captions, OCR)
8. Table citation (structured tabular data)
9. Empty evidence handling
10. Unsupported answer detection
11. Citation completeness (unused citations, duplicate citations)
12. Registry immutability and freeze protection
13. Sanitization of hallucinated citations
"""

import pytest
from rag.provenance import (
    Citation,
    CitationProvenanceValidator,
    CitationRegistry,
    CitationRegistryError,
    CitationValidationResult,
    GroundedAnswerResponse,
    ProvenanceService,
    ProvenanceType,
    RetrievedEvidenceItem,
)
from rag.retrieval.models import RetrievedChunkResult


@pytest.fixture
def sample_evidence() -> list[RetrievedEvidenceItem]:
    return [
        RetrievedEvidenceItem(
            item_id="chunk_1",
            document_id="doc_apollo",
            document_version_id="v2",
            tenant_id="tenant_aerospace",
            text="The Apollo 11 Lunar Module Eagle touched down on the Moon on July 20, 1969 with Neil Armstrong.",
            title="Apollo Flight Journal",
            page_number=42,
            section_path=["Flight Operations", "Lunar Descent"],
            retrieval_score=0.95,
            retrieval_method="hybrid_rrf",
            provenance_type=ProvenanceType.DIRECT_CHUNK,
        ),
        RetrievedEvidenceItem(
            item_id="chunk_2",
            document_id="doc_saturn",
            document_version_id="v1",
            tenant_id="tenant_aerospace",
            text="Saturn V used RP-1 kerosene and liquid oxygen in its five F-1 first stage booster engines.",
            title="Saturn V Launch Vehicle Spec",
            page_number=15,
            section_path=["Propulsion", "Stage 1"],
            retrieval_score=0.88,
            retrieval_method="dense_vector",
            provenance_type=ProvenanceType.DIRECT_CHUNK,
        ),
    ]


@pytest.fixture
def populated_registry(sample_evidence) -> CitationRegistry:
    reg = CitationRegistry(
        tenant_id="tenant_aerospace",
        expected_document_versions={"doc_apollo": "v2", "doc_saturn": "v1"},
    )
    for ev in sample_evidence:
        reg.register_evidence(ev)
    reg.freeze()
    return reg


@pytest.fixture
def service() -> ProvenanceService:
    return ProvenanceService()


# ---------------------------------------------------------------------------
# Test 1: Valid citations & High Confidence
# ---------------------------------------------------------------------------
def test_valid_citations(populated_registry: CitationRegistry, service: ProvenanceService):
    answer = (
        "The Lunar Module Eagle touched down on the Moon in July 1969 [C1]. "
        "The Saturn V first stage was powered by five F-1 booster engines burning kerosene [C2]."
    )

    resp: GroundedAnswerResponse = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert resp.citation_validation.is_valid is True
    assert set(resp.citation_validation.valid_citations) == {"[C1]", "[C2]"}
    assert len(resp.citation_validation.invalid_citations) == 0
    assert len(resp.citation_validation.mismatched_citations) == 0
    assert len(resp.citation_validation.cross_tenant_citations) == 0
    assert len(resp.unsupported_claims) == 0
    assert resp.confidence > 0.85
    assert len(resp.citations) == 2


# ---------------------------------------------------------------------------
# Test 2: Missing citation on substantive claims
# ---------------------------------------------------------------------------
def test_missing_citation(populated_registry: CitationRegistry, service: ProvenanceService):
    answer = (
        "The Lunar Module Eagle touched down on the Moon on July 20, 1969 [C1]. "
        "The Apollo 13 mission suffered an oxygen tank explosion during translunar coast."
    )

    resp = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert resp.citation_validation.is_valid is False
    assert len(resp.citation_validation.missing_citations) == 1
    assert "oxygen tank explosion" in resp.citation_validation.missing_citations[0]
    assert len(resp.unsupported_claims) == 1
    assert resp.confidence < 0.80


# ---------------------------------------------------------------------------
# Test 3: Invalid citation ID (hallucinated marker)
# ---------------------------------------------------------------------------
def test_invalid_citation_id(populated_registry: CitationRegistry, service: ProvenanceService):
    answer = "The Apollo mission landed successfully [C1], reaching tranquility base safely [C99]."

    resp = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert resp.citation_validation.is_valid is False
    assert "[C99]" in resp.citation_validation.invalid_citations
    assert "[C1]" in resp.citation_validation.valid_citations
    assert any("Marker '[C99]' does not exist" in w for w in resp.warnings)


# ---------------------------------------------------------------------------
# Test 4: Cross-tenant citation attempt
# ---------------------------------------------------------------------------
def test_cross_tenant_citation_attempt(service: ProvenanceService):
    reg = CitationRegistry(tenant_id="tenant_bank")
    # Register an evidence chunk belonging to tenant_healthcare
    foreign_chunk = RetrievedEvidenceItem(
        item_id="chk_secret",
        document_id="doc_patient",
        tenant_id="tenant_healthcare",
        text="Patient heart rate was 72 beats per minute.",
    )
    reg.register_evidence(foreign_chunk)
    reg.freeze()

    answer = "Patient heart rate was 72 bpm [C1]."
    resp = service.validate_answer(
        answer=answer,
        registry=reg,
        expected_tenant_id="tenant_bank",  # Mismatched expected tenant
    )

    assert resp.citation_validation.is_valid is False
    assert "[C1]" in resp.citation_validation.cross_tenant_citations
    assert resp.confidence == 0.0  # Security violation drops confidence to zero
    assert any("Cross-tenant violation" in w for w in resp.warnings)


# ---------------------------------------------------------------------------
# Test 5: Wrong document version
# ---------------------------------------------------------------------------
def test_wrong_document_version(service: ProvenanceService):
    reg = CitationRegistry(
        tenant_id="tenant_tech",
        expected_document_versions={"doc_firmware": "v3"},  # Active version is v3
    )
    outdated_chunk = RetrievedEvidenceItem(
        item_id="chk_old",
        document_id="doc_firmware",
        document_version_id="v1",  # Outdated version v1
        tenant_id="tenant_tech",
        text="Firmware baud rate default is configured to 9600 bps.",
    )
    reg.register_evidence(outdated_chunk)
    reg.freeze()

    answer = "The default firmware baud rate is configured to 9600 bps [C1]."
    resp = service.validate_answer(
        answer=answer,
        registry=reg,
        expected_tenant_id="tenant_tech",
        expected_document_versions={"doc_firmware": "v3"},
    )

    assert resp.citation_validation.is_valid is False
    assert "[C1]" in resp.citation_validation.version_mismatch_citations
    assert any("references version 'v1', expected 'v3'" in w for w in resp.warnings)


# ---------------------------------------------------------------------------
# Test 6: Citation mismatch (claim does not overlap with evidence text)
# ---------------------------------------------------------------------------
def test_citation_mismatch(populated_registry: CitationRegistry, service: ProvenanceService):
    # C1 is about Apollo 11 Lunar Module, but sentence claims quantum cryptographic keys
    answer = "Quantum key distribution ensures unbreakable cryptographic ciphers across optical fibers [C1]."

    resp = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert resp.citation_validation.is_valid is False
    assert "[C1]" in resp.citation_validation.mismatched_citations
    assert len(resp.unsupported_claims) == 1
    assert any("Citation mismatch" in w for w in resp.warnings)


# ---------------------------------------------------------------------------
# Test 7: Visual Asset Citation (Figure, diagram, OCR)
# ---------------------------------------------------------------------------
def test_asset_citation(service: ProvenanceService):
    reg = CitationRegistry(tenant_id="tenant_hardware")
    cit = reg.register_asset(
        asset_id="fig_block_diagram",
        document_id="doc_schematic",
        description="Block diagram displaying DDR5 memory controller interfaced with PCIe Gen 5 lanes.",
        asset_type="DIAGRAM",
        caption="Figure 3.1: Memory Controller Architecture",
        ocr_text="PCIe 5.0 x16 -> DDR5-4800 Controller",
        page_number=8,
    )
    reg.freeze()

    assert cit.citation_id == "[C1]"
    assert cit.provenance_type == ProvenanceType.FIGURE
    assert cit.asset_id == "fig_block_diagram"

    answer = "The block diagram details the DDR5 memory controller communicating over PCIe Gen 5 lanes [C1]."
    resp = service.validate_answer(answer=answer, registry=reg, expected_tenant_id="tenant_hardware")

    assert resp.citation_validation.is_valid is True
    assert "[C1]" in resp.citation_validation.valid_citations


# ---------------------------------------------------------------------------
# Test 8: Table Citation
# ---------------------------------------------------------------------------
def test_table_citation(service: ProvenanceService):
    reg = CitationRegistry(tenant_id="tenant_finance")
    markdown_table = (
        "| Quarter | Revenue ($M) | Operating Margin |\n"
        "| Q1 2024 | $4,520       | 28.5%            |\n"
        "| Q2 2024 | $4,890       | 30.2%            |"
    )
    cit = reg.register_table(
        table_id="tbl_financial_results",
        document_id="doc_q2_earnings",
        formatted_table=markdown_table,
        title="Consolidated Financial Performance",
        page_number=12,
        score=0.92,
    )
    reg.freeze()

    assert cit.citation_id == "[C1]"
    assert cit.provenance_type == ProvenanceType.TABLE
    assert cit.table_id == "tbl_financial_results"

    answer = "In Q2 2024, revenue climbed to $4,890M with an operating margin of 30.2% [C1]."
    resp = service.validate_answer(answer=answer, registry=reg, expected_tenant_id="tenant_finance")

    assert resp.citation_validation.is_valid is True
    assert "[C1]" in resp.citation_validation.valid_citations


# ---------------------------------------------------------------------------
# Test 9: Empty evidence handling
# ---------------------------------------------------------------------------
def test_empty_evidence(service: ProvenanceService):
    reg = CitationRegistry(tenant_id="tenant_empty")
    reg.freeze()

    answer = "The boiling point of nitrogen is -195.8 degrees Celsius [C1]."
    resp = service.validate_answer(answer=answer, registry=reg, expected_tenant_id="tenant_empty")

    assert resp.citation_validation.is_valid is False
    assert "[C1]" in resp.citation_validation.invalid_citations
    assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 10: Unsupported answer
# ---------------------------------------------------------------------------
def test_unsupported_answer(populated_registry: CitationRegistry, service: ProvenanceService):
    answer = (
        "Mars has two small moons named Phobos and Deimos. "
        "Liquid water was discovered in vast underground aquifers."
    )

    resp = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert resp.citation_validation.is_valid is False
    assert len(resp.unsupported_claims) == 2
    assert resp.confidence < 0.5


# ---------------------------------------------------------------------------
# Test 11: Citation completeness (Unused citations and Duplicates)
# ---------------------------------------------------------------------------
def test_citation_completeness(populated_registry: CitationRegistry, service: ProvenanceService):
    # Answer only uses C1, leaving C2 unused; also repeats [C1] twice in one sentence
    answer = "The Lunar Module Eagle landed in July 1969 [C1], piloted by Armstrong [C1]."

    resp = service.validate_answer(
        answer=answer,
        registry=populated_registry,
        expected_tenant_id="tenant_aerospace",
    )

    assert "[C2]" in resp.citation_validation.unused_citations
    assert "[C1]" in resp.citation_validation.duplicate_citations
    assert any("Duplicate citation '[C1]'" in w for w in resp.warnings)


# ---------------------------------------------------------------------------
# Test 12: Registry freeze protection
# ---------------------------------------------------------------------------
def test_registry_freeze_protection():
    reg = CitationRegistry()
    reg.register_chunk({"chunk_id": "c1", "text": "Valid evidence."})
    reg.freeze()

    with pytest.raises(CitationRegistryError):
        reg.register_chunk({"chunk_id": "c2", "text": "Unverified post-hoc chunk."})


# ---------------------------------------------------------------------------
# Test 13: Sanitize hallucinated citations
# ---------------------------------------------------------------------------
def test_sanitize_answer(populated_registry: CitationRegistry, service: ProvenanceService):
    raw_answer = "The Eagle touched down [C1] and returned to lunar orbit [C99]."
    clean = service.sanitize_answer(raw_answer, populated_registry)
    assert "[C1]" in clean
    assert "[C99]" not in clean
    assert clean == "The Eagle touched down [C1] and returned to lunar orbit."


# ---------------------------------------------------------------------------
# Test 14: Registration from RetrievedChunkResult
# ---------------------------------------------------------------------------
def test_register_from_retrieved_chunk_result(service: ProvenanceService):
    chunk = RetrievedChunkResult(
        chunk_id="chk_hybrid",
        document_id="doc_spacecraft",
        document_version_id="v4",
        tenant_id="tenant_nasa",
        text="Thermal protection tiles absorb re-entry heat up to 1650 degrees C.",
        title="Thermal Protection Spec",
        page_number=22,
        section_path=["TPS", "Tiles"],
        source_uri="s3://nasa-docs/tps.pdf",
        raw_score=0.91,
    )
    reg = service.build_registry([chunk], tenant_id="tenant_nasa")
    assert reg.is_frozen is True
    assert len(reg.all_citations()) == 1
    cit = reg.get("[C1]")
    assert cit is not None
    assert cit.document_title == "Thermal Protection Spec"
    assert cit.page_number == 22
    assert cit.section_path == ["TPS", "Tiles"]
    assert cit.tenant_id == "tenant_nasa"
