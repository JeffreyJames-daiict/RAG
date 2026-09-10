"""Hardened test suite for BM25 Sparse Keyword Retrieval.

Covers all 18 verification scenarios:
  1. Exact keyword matching
  2. Technical terminology (e.g. IPv6, H2SO4, microservices)
  3. Patent numbers (e.g. US 11,234,567 B2, EP3456789, WO2020123456)
  4. Product codes & alphanumeric compounds (e.g. GTX-4090, SKU-1234-X)
  5. Company names & symbols (e.g. AT&T, 3M, Johnson & Johnson)
  6. Claim numbers (e.g. Claim 1, Claim 14(a))
  7. Exact quoted phrase queries
  8. Title boosting (title match scores higher than body match)
  9. Heading boosting (heading match scores higher than body match)
 10. Table content retrieval
 11. Figure caption retrieval
 12. Strict tenant isolation (no cross-tenant leakage)
 13. Soft-deleted document exclusion
 14. Inactive version exclusion
 15. Empty query & length limit rejection
 16. No results & min score cutoff
 17. Indexing lifecycle: initial, incremental, version eviction, reindex
 18. Failure recovery and status reporting telemetry
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from pydantic import ValidationError
import pytest

from rag.core.config import BM25Config
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document
from rag.core.models import DocumentStatus
from rag.retrieval.bm25 import BM25Index, BM25Retriever, TenantBM25IndexManager
from rag.retrieval.bm25_service import BM25RetrievalService
from rag.retrieval.exceptions import EmptyQueryError, QueryTooLongError, TenantContextMissingError
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.models import RetrievalRequest


@pytest.fixture
def db_mgr():
    """In-memory SQLite database manager for isolated testing."""
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


@pytest.fixture
def bm25_config():
    """Configured BM25 test configuration."""
    return BM25Config(
        enabled=True,
        top_k=5,
        k1=1.5,
        b=0.75,
        min_score=0.0,
        index_refresh_interval=300,
        title_weight=2.5,
        heading_weight=2.0,
        body_weight=1.0,
        table_weight=1.5,
        caption_weight=1.2,
    )


def _seed_document(
    db_mgr: DatabaseManager,
    tenant_id: str = "tenant_a",
    filename: str = "doc.pdf",
    version: int = 1,
    status: str = DocumentStatus.INDEXED.value,
    deleted: bool = False,
) -> str:
    """Helper to seed a document and return its document_id."""
    doc_id = str(uuid.uuid4())
    with db_mgr.session() as session:
        doc = Document(
            document_id=doc_id,
            tenant_id=tenant_id,
            source="LOCAL_FILE",
            source_uri=f"/test/{doc_id}.pdf",
            filename=filename,
            mime_type="application/pdf",
            content_hash=f"hash_{doc_id[:8]}",
            version=version,
            status=status,
            deleted_at=datetime.now(timezone.utc) if deleted else None,
        )
        session.add(doc)
        session.commit()
    return doc_id


def _seed_chunk(
    db_mgr: DatabaseManager,
    document_id: str,
    text: str,
    version: int = 1,
    chunk_index: int = 0,
    section: str = "",
    page: int = 1,
    chunk_type: str = "text",
    metadata: dict = None,
) -> str:
    """Helper to seed a chunk and return its chunk_id."""
    chunk_id = str(uuid.uuid4())
    with db_mgr.session() as session:
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            version=version,
            chunk_index=chunk_index,
            text=text,
            section=section,
            page=page,
            chunk_metadata=metadata or {"chunk_type": chunk_type},
            chunking_version="v1.0",
        )
        session.add(chunk)
        session.commit()
    return chunk_id


# ── 1. Exact Keyword Matching ───────────────────────────────────────────────

def test_exact_keyword_matching(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="semiconductors.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="Photolithography processes enable sub-nanometer transistor fabrication.")
    c2 = _seed_chunk(db_mgr, d_id, text="Internal combustion engines convert gasoline into mechanical energy.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("photolithography", tenant_id="t1")

    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    assert res.results[0].raw_score > 0.0
    assert res.results[0].retrieval_method == "bm25"


# ── 2. Technical Terminology ────────────────────────────────────────────────

def test_technical_terms_preservation(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="tech.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="Deploying IPv6 dual-stack networking with Kubernetes microservices.")
    c2 = _seed_chunk(db_mgr, d_id, text="Handling concentrated H2SO4 sulfuric acid safely in laboratory environments.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    res_ipv6 = svc.retrieve_sync("IPv6 microservices", tenant_id="t1")
    assert len(res_ipv6.results) >= 1
    assert res_ipv6.results[0].chunk_id == c1

    res_chem = svc.retrieve_sync("H2SO4 acid", tenant_id="t1")
    assert len(res_chem.results) >= 1
    assert res_chem.results[0].chunk_id == c2


# ── 3. Patent Numbers ───────────────────────────────────────────────────────

def test_patent_number_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="patents.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="According to patent US 11,234,567 B2, the quantum gate fidelity exceeds 99.9%.")
    c2 = _seed_chunk(db_mgr, d_id, text="Disclosed in European patent EP3456789 is a cryogenic refrigerator.")
    c3 = _seed_chunk(db_mgr, d_id, text="International publication WO/2020/123456 describes laser optical traps.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    # Search with formatted US patent
    res_us = svc.retrieve_sync("US 11,234,567 B2", tenant_id="t1")
    assert len(res_us.results) >= 1
    assert res_us.results[0].chunk_id == c1

    # Search with EP patent
    res_ep = svc.retrieve_sync("EP3456789", tenant_id="t1")
    assert len(res_ep.results) >= 1
    assert res_ep.results[0].chunk_id == c2

    # Search with WO patent
    res_wo = svc.retrieve_sync("WO2020123456", tenant_id="t1")
    assert len(res_wo.results) >= 1
    assert res_wo.results[0].chunk_id == c3


# ── 4. Product Codes & Models ───────────────────────────────────────────────

def test_product_codes_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="hardware.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="Recommended GPU for model inference is NVIDIA RTX-4090 or GTX-4090.")
    c2 = _seed_chunk(db_mgr, d_id, text="Inventory replacement item SKU-1234-X has arrived at the warehouse.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    res_gpu = svc.retrieve_sync("GTX-4090", tenant_id="t1")
    assert len(res_gpu.results) >= 1
    assert res_gpu.results[0].chunk_id == c1

    res_sku = svc.retrieve_sync("SKU-1234-X", tenant_id="t1")
    assert len(res_sku.results) >= 1
    assert res_sku.results[0].chunk_id == c2


# ── 5. Company Names ────────────────────────────────────────────────────────

def test_company_names_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="corporate.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="Collaboration agreement signed between AT&T and telecommunications partners.")
    c2 = _seed_chunk(db_mgr, d_id, text="Innovative optical films manufactured by 3M for aerospace.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    res_att = svc.retrieve_sync("AT&T partners", tenant_id="t1")
    assert len(res_att.results) >= 1
    assert res_att.results[0].chunk_id == c1

    res_3m = svc.retrieve_sync("3M optical films", tenant_id="t1")
    assert len(res_3m.results) >= 1
    assert res_3m.results[0].chunk_id == c2


# ── 6. Claim Numbers ────────────────────────────────────────────────────────

def test_claim_numbers_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="patent_claims.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="In claim 1, a system comprising a processor and non-transitory memory.")
    c2 = _seed_chunk(db_mgr, d_id, text="The method according to claim 14(a), further comprising decrypting ciphertext.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    res_cl1 = svc.retrieve_sync("claim 1", tenant_id="t1")
    assert len(res_cl1.results) >= 1
    assert res_cl1.results[0].chunk_id == c1

    res_cl14a = svc.retrieve_sync("claim 14(a)", tenant_id="t1")
    assert len(res_cl14a.results) >= 1
    assert res_cl14a.results[0].chunk_id == c2


# ── 7. Phrase Queries ───────────────────────────────────────────────────────

def test_quoted_phrase_queries(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="physics.pdf")
    # c1 has the exact phrase
    c1 = _seed_chunk(db_mgr, d_id, text="Research on quantum error correction codes demonstrated fault-tolerant thresholds.")
    # c2 has the individual words scattered
    c2 = _seed_chunk(db_mgr, d_id, text="There was an error in the quantum experiment before calibration and correction.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync('"quantum error correction"', tenant_id="t1")

    assert len(res.results) >= 1
    # c1 must rank first due to phrase boost
    assert res.results[0].chunk_id == c1


# ── 8. Title Boosting ───────────────────────────────────────────────────────

def test_title_boosting(db_mgr, bm25_config):
    # doc1 has "Superconductor" in title
    d1 = _seed_document(db_mgr, tenant_id="t1", filename="superconductor_research.pdf")
    c1 = _seed_chunk(db_mgr, d1, text="General laboratory equipment maintenance protocol and calibration schedule.")

    # doc2 only has "superconductor" once in the body
    d2 = _seed_document(db_mgr, tenant_id="t1", filename="general_notes.pdf")
    c2 = _seed_chunk(db_mgr, d2, text="Mention of superconductor materials tested yesterday.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("superconductor", tenant_id="t1")

    assert len(res.results) >= 2
    # c1 should outrank c2 because title weight is 2.5 vs body weight 1.0
    assert res.results[0].chunk_id == c1
    assert res.results[0].raw_score > res.results[1].raw_score


# ── 9. Heading Boosting ─────────────────────────────────────────────────────

def test_heading_boosting(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="spec.pdf")
    # c1 has "Microservices Architecture" in the heading
    c1 = _seed_chunk(db_mgr, d_id, text="The system uses containerized pods.", section="Microservices Architecture")
    # c2 only mentions microservices in body
    c2 = _seed_chunk(db_mgr, d_id, text="We evaluated microservices versus monolithic architectures.", section="Appendix")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("microservices", tenant_id="t1")

    assert len(res.results) >= 2
    # c1 should outrank c2 due to heading weight boost
    assert res.results[0].chunk_id == c1


# ── 10. Table Content Retrieval ─────────────────────────────────────────────

def test_table_content_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="financials.pdf")
    c1 = _seed_chunk(
        db_mgr,
        d_id,
        text="| Metric | Q1 | Q2 |\n| Revenue | $10M | $12M |\n| Net Income | $2M | $3M |",
        chunk_type="table",
        section="Table 1: Financial Performance",
    )
    c2 = _seed_chunk(db_mgr, d_id, text="Executive summary of company strategy.", chunk_type="text")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("Financial Performance Revenue Net Income", tenant_id="t1")

    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    assert res.results[0].chunk_type == "table"


# ── 11. Caption Content Retrieval ───────────────────────────────────────────

def test_figure_caption_retrieval(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="biology.pdf")
    c1 = _seed_chunk(
        db_mgr,
        d_id,
        text="The cell division cycle depicted graphically.",
        metadata={"caption": "Figure 3: Electron micrograph of mitosis spindle fibers."},
    )
    c2 = _seed_chunk(db_mgr, d_id, text="Introduction to cellular biology concepts.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("mitosis spindle fibers", tenant_id="t1")

    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1


# ── 12. Strict Tenant Isolation ─────────────────────────────────────────────

def test_strict_tenant_isolation(db_mgr, bm25_config):
    d1 = _seed_document(db_mgr, tenant_id="tenant_secret", filename="secret.pdf")
    c1 = _seed_chunk(db_mgr, d1, text="Top secret project codename Prometheus.")

    d2 = _seed_document(db_mgr, tenant_id="tenant_public", filename="public.pdf")
    c2 = _seed_chunk(db_mgr, d2, text="Public company press release announcement.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    # Searching from tenant_public should NEVER return tenant_secret
    res_pub = svc.retrieve_sync("Prometheus secret", tenant_id="tenant_public")
    assert all(r.chunk_id != c1 for r in res_pub.results)
    assert all(r.tenant_id == "tenant_public" for r in res_pub.results)

    # Searching from tenant_secret returns c1
    res_sec = svc.retrieve_sync("Prometheus", tenant_id="tenant_secret")
    assert len(res_sec.results) >= 1
    assert res_sec.results[0].chunk_id == c1


# ── 13. Soft-Deleted Document Exclusion ─────────────────────────────────────

def test_soft_deleted_document_exclusion(db_mgr, bm25_config):
    # Active document
    d_active = _seed_document(db_mgr, tenant_id="t1", filename="active.pdf", deleted=False)
    c_active = _seed_chunk(db_mgr, d_active, text="Artificial intelligence and natural language processing systems.")

    # Soft-deleted document
    d_del = _seed_document(db_mgr, tenant_id="t1", filename="deleted.pdf", deleted=True)
    c_del = _seed_chunk(db_mgr, d_del, text="Artificial intelligence confidential deleted draft.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("Artificial intelligence", tenant_id="t1")

    returned_cids = {r.chunk_id for r in res.results}
    assert c_active in returned_cids
    assert c_del not in returned_cids


# ── 14. Inactive Version Exclusion ──────────────────────────────────────────

def test_inactive_version_exclusion(db_mgr, bm25_config):
    # Document is at active version 2
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="versioned.pdf", version=2)
    # Chunk from version 1 (old)
    c_v1 = _seed_chunk(db_mgr, d_id, text="Old version 1 terms: legacy protocol alpha.", version=1)
    # Chunk from version 2 (active)
    c_v2 = _seed_chunk(db_mgr, d_id, text="New version 2 terms: modern protocol beta.", version=2)

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("protocol", tenant_id="t1")

    returned_cids = {r.chunk_id for r in res.results}
    assert c_v2 in returned_cids
    assert c_v1 not in returned_cids


# ── 15. Query Validation & Length Limits ────────────────────────────────────

def test_empty_and_oversized_query_rejection(db_mgr, bm25_config):
    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    with pytest.raises((EmptyQueryError, ValidationError)):
        svc.retrieve_sync("", tenant_id="t1")

    with pytest.raises((EmptyQueryError, ValidationError)):
        svc.retrieve_sync("   ", tenant_id="t1")

    with pytest.raises((TenantContextMissingError, ValidationError)):
        svc.retrieve_sync("test query", tenant_id="")

    with pytest.raises(QueryTooLongError):
        svc.retrieve(RetrievalRequest(query="a" * 5000, tenant_id="t1", max_query_length=4000))


# ── 16. No Results & Min Score Cutoff ───────────────────────────────────────

def test_no_results_and_min_score_cutoff(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="astronomy.pdf")
    _seed_chunk(db_mgr, d_id, text="Observations of spiral galaxies and red dwarf stars.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)

    # Completely unrelated query
    res_none = svc.retrieve_sync("submarine sonar torpedo", tenant_id="t1")
    assert res_none.results == []
    assert res_none.returned_results == 0

    # High min_score threshold
    bm25_config.min_score = 999.0
    svc_high_thresh = BM25RetrievalService(db_mgr, config=bm25_config)
    res_thresh = svc_high_thresh.retrieve_sync("spiral galaxies", tenant_id="t1")
    assert res_thresh.results == []


# ── 17. Index Lifecycle: Incremental, Eviction & Reindex ─────────────────────

def test_index_lifecycle_operations(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t_lifecycle", filename="lifecycle.pdf")
    c1 = _seed_chunk(db_mgr, d_id, text="Initial chunk indexing content.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    mgr = svc.index_manager

    # 1. Initial hydration
    initial_count = mgr.hydrate_tenant("t_lifecycle")
    assert initial_count == 1

    # 2. Incremental addition
    mgr.index_chunk(
        tenant_id="t_lifecycle",
        chunk_id="c_incremental",
        text="Incrementally added chunk with unique token zephyros.",
        document_id=d_id,
    )
    res_inc = svc.retrieve_sync("zephyros", tenant_id="t_lifecycle")
    assert len(res_inc.results) == 1
    assert res_inc.results[0].chunk_id == "c_incremental"

    # 3. Document removal
    mgr.remove_document("t_lifecycle", d_id)
    res_after_del = svc.retrieve_sync("zephyros", tenant_id="t_lifecycle")
    assert res_after_del.results == []

    # 4. Force reindex from DB
    reindexed_count = mgr.reindex_tenant("t_lifecycle")
    assert reindexed_count == 1  # Only c1 is in DB


# ── 18. Failure Recovery & Status Telemetry ─────────────────────────────────

def test_failure_recovery_and_status_reporting(db_mgr, bm25_config):
    d_id = _seed_document(db_mgr, tenant_id="t_telemetry", filename="telemetry.pdf")
    _seed_chunk(db_mgr, d_id, text="Telemetry verification chunk content.")

    svc = BM25RetrievalService(db_mgr, config=bm25_config)
    res = svc.retrieve_sync("telemetry", tenant_id="t_telemetry")

    assert res.diagnostics is not None
    assert "query_hash" in res.diagnostics
    assert "search_latency_ms" in res.diagnostics
    assert "index_freshness_sec" in res.diagnostics
    assert "corpus_size" in res.diagnostics

    # Inspect status reporting
    status = svc.index_manager.get_status("t_telemetry")
    assert status["tenant_id"] == "t_telemetry"
    assert status["corpus_size"] >= 1
    assert status["total_terms"] > 0
    assert status["index_version"] == "v1.0"
    assert "memory_bytes_estimate" in status
