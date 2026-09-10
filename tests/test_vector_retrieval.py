"""Unit and integration tests for Vector Retrieval and Python Vector Repository."""

import uuid
import pytest
from sqlalchemy import select

from rag.core.config import RetrievalConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.embeddings.service import EmbeddingService
from rag.retrieval.exceptions import (
    EmptyQueryError,
    IncompatibleEmbeddingError,
    TenantIsolationError,
)
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.models import RetrievalRequest
from rag.retrieval.repositories.python_repository import (
    PythonVectorRepository,
    _cosine_similarity,
    _inner_product,
    _l2_distance,
)
from rag.retrieval.vector_service import VectorRetrievalService


@pytest.fixture
def db_mgr():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


def test_similarity_math_functions():
    """Verify precision and edge cases of vector similarity helpers."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]
    vec_neg = [-1.0, 0.0, 0.0]

    # Cosine similarity
    assert abs(_cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6
    assert abs(_cosine_similarity(vec_a, vec_c) - 0.0) < 1e-6
    assert abs(_cosine_similarity(vec_a, vec_neg) - (-1.0)) < 1e-6

    # L2 distance
    assert abs(_l2_distance(vec_a, vec_b) - 0.0) < 1e-6
    assert abs(_l2_distance(vec_a, vec_neg) - 2.0) < 1e-6

    # Inner product
    assert abs(_inner_product(vec_a, vec_b) - 1.0) < 1e-6
    assert abs(_inner_product(vec_a, vec_c) - 0.0) < 1e-6


def test_python_repository_cosine_search(db_mgr):
    """Verify in-memory cosine similarity search and ranking in PythonVectorRepository."""
    doc_id = str(uuid.uuid4())
    chunk1_id = str(uuid.uuid4())
    chunk2_id = str(uuid.uuid4())

    with db_mgr.session() as session:
        doc = Document(
            document_id=doc_id,
            tenant_id="tenant_alpha",
            source="local",
            source_uri="s3://bucket/patent.pdf",
            filename="patent.pdf",
            mime_type="application/pdf",
            content_hash="hash_alpha",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        session.add(doc)

        c1 = Chunk(
            chunk_id=chunk1_id,
            document_id=doc_id,
            version=1,
            chunk_index=0,
            text="Autonomous vehicle navigation system with LiDAR sensors.",
            section="1. Introduction > 1.1 LiDAR",
            page=1,
            chunk_metadata={"chunk_type": "text"},
        )
        c2 = Chunk(
            chunk_id=chunk2_id,
            document_id=doc_id,
            version=1,
            chunk_index=1,
            text="Chocolate cake recipe with dark cocoa and fudge frosting.",
            section="2. Recipes",
            page=2,
            chunk_metadata={"chunk_type": "text"},
        )
        session.add_all([c1, c2])

        # Orthogonal embedding vectors for testing
        e1 = Embedding(
            chunk_id=chunk1_id,
            embedding_model="test-model",
            embedding_version="v1.0",
            dimension=3,
            provider="mock",
            vector=[1.0, 0.0, 0.0],
        )
        e2 = Embedding(
            chunk_id=chunk2_id,
            embedding_model="test-model",
            embedding_version="v1.0",
            dimension=3,
            provider="mock",
            vector=[0.0, 1.0, 0.0],
        )
        session.add_all([e1, e2])

    repo = PythonVectorRepository(db_manager=db_mgr)

    # Query vector pointing along LiDAR direction [1, 0, 0]
    req = RetrievalRequest(query="LiDAR autonomous car", tenant_id="tenant_alpha", top_k=2)
    candidates, total = repo.search_vectors(
        query_vector=[1.0, 0.0, 0.0],
        request=req,
        model_name="test-model",
        model_version="v1.0",
    )

    assert total == 2
    assert len(candidates) == 2
    # chunk1 must rank first with similarity = 1.0
    assert candidates[0].chunk_id == chunk1_id
    assert abs(candidates[0].similarity_score - 1.0) < 1e-4
    # chunk2 must rank second with similarity = 0.0
    assert candidates[1].chunk_id == chunk2_id
    assert abs(candidates[1].similarity_score - 0.0) < 1e-4


def test_tenant_isolation(db_mgr):
    """Verify that retrieval strictly isolates candidate chunks by tenant_id."""
    with db_mgr.session() as session:
        # Tenant A document and chunk
        doc_a = Document(
            document_id="doc_a",
            tenant_id="tenant_A",
            source="local",
            source_uri="s3://a.pdf",
            filename="a.pdf",
            mime_type="application/pdf",
            content_hash="hash_a",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c_a = Chunk(
            chunk_id="chunk_a",
            document_id="doc_a",
            version=1,
            chunk_index=0,
            text="Secret financial statement of Tenant A",
        )
        e_a = Embedding(
            chunk_id="chunk_a",
            embedding_model="test-model",
            embedding_version="v1.0",
            dimension=2,
            provider="mock",
            vector=[1.0, 0.0],
        )

        # Tenant B document and chunk
        doc_b = Document(
            document_id="doc_b",
            tenant_id="tenant_B",
            source="local",
            source_uri="s3://b.pdf",
            filename="b.pdf",
            mime_type="application/pdf",
            content_hash="hash_b",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c_b = Chunk(
            chunk_id="chunk_b",
            document_id="doc_b",
            version=1,
            chunk_index=0,
            text="Public press release of Tenant B",
        )
        e_b = Embedding(
            chunk_id="chunk_b",
            embedding_model="test-model",
            embedding_version="v1.0",
            dimension=2,
            provider="mock",
            vector=[1.0, 0.0],
        )

        session.add_all([doc_a, c_a, e_a, doc_b, c_b, e_b])

    repo = PythonVectorRepository(db_manager=db_mgr)

    # Searching with tenant_B must NEVER return tenant_A's chunk
    req_b = RetrievalRequest(query="Secret statement", tenant_id="tenant_B", top_k=5)
    cands_b, total_b = repo.search_vectors(
        query_vector=[1.0, 0.0],
        request=req_b,
        model_name="test-model",
        model_version="v1.0",
    )

    assert total_b == 1
    assert len(cands_b) == 1
    assert cands_b[0].chunk_id == "chunk_b"
    assert all(c.chunk_id != "chunk_a" for c in cands_b)


def test_similarity_threshold_filtering(db_mgr):
    """Verify that chunks below similarity_threshold are excluded."""
    with db_mgr.session() as session:
        doc = Document(
            document_id="doc_thresh",
            tenant_id="tenant_thresh",
            source="local",
            source_uri="s3://doc.pdf",
            filename="doc.pdf",
            mime_type="application/pdf",
            content_hash="hash_thresh",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c1 = Chunk(chunk_id="ch_high", document_id="doc_thresh", version=1, chunk_index=0, text="High sim")
        c2 = Chunk(chunk_id="ch_low", document_id="doc_thresh", version=1, chunk_index=1, text="Low sim")
        e1 = Embedding(
            chunk_id="ch_high",
            embedding_model="m",
            embedding_version="v1",
            dimension=2,
            provider="mock",
            vector=[1.0, 0.0],
        )
        e2 = Embedding(
            chunk_id="ch_low",
            embedding_model="m",
            embedding_version="v1",
            dimension=2,
            provider="mock",
            vector=[0.0, 1.0],
        )
        session.add_all([doc, c1, c2, e1, e2])

    repo = PythonVectorRepository(db_manager=db_mgr)

    # Request with threshold 0.5 (query is [1, 0], so ch_high is 1.0, ch_low is 0.0)
    req = RetrievalRequest(
        query="High",
        tenant_id="tenant_thresh",
        top_k=5,
        similarity_threshold=0.5,
    )
    cands, _ = repo.search_vectors(
        query_vector=[1.0, 0.0],
        request=req,
        model_name="m",
        model_version="v1",
    )
    assert len(cands) == 1
    assert cands[0].chunk_id == "ch_high"


def test_vector_retrieval_service_full_pipeline(db_mgr):
    """Test VectorRetrievalService end-to-end with EmbeddingService."""
    emb_svc = EmbeddingService(db_manager=db_mgr)
    dim = emb_svc.dimension

    # Create document, chunk, and matching embedding
    with db_mgr.session() as session:
        doc = Document(
            document_id="doc_svc",
            tenant_id="acme",
            source="local",
            source_uri="s3://acme/handbook.pdf",
            filename="handbook.pdf",
            mime_type="application/pdf",
            content_hash="hash_svc",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c = Chunk(
            chunk_id="ch_svc",
            document_id="doc_svc",
            version=1,
            chunk_index=0,
            text="Employee health insurance and dental plan details.",
            section="Benefits > Health Insurance",
            page=4,
            chunk_metadata={"chunk_type": "text", "section_path": ["Benefits", "Health Insurance"]},
        )
        # Real/mock embedding matching service model
        mock_vec = [0.05] * dim
        e = Embedding(
            chunk_id="ch_svc",
            embedding_model=emb_svc.model_name,
            embedding_version=emb_svc.model_version,
            dimension=dim,
            provider=emb_svc.provider_name,
            vector=mock_vec,
        )
        session.add_all([doc, c, e])

    service = VectorRetrievalService(db_manager=db_mgr, embedding_service=emb_svc)

    req = RetrievalRequest(query="health insurance", tenant_id="acme", top_k=3)
    response = service.retrieve(req)

    assert response.query == "health insurance"
    assert response.tenant_id == "acme"
    assert len(response.results) == 1
    result = response.results[0]
    assert result.chunk_id == "ch_svc"
    assert result.rank == 1
    assert result.section_path == ["Benefits", "Health Insurance"]
    assert result.page == 4
    assert result.retrieval_method == "vector"
    assert response.duration_ms > 0
    assert response.search_latency_ms >= 0


def test_invalid_query_validation():
    """Verify request validation rejects empty queries or missing tenant."""
    with pytest.raises(EmptyQueryError):
        RetrievalRequest(query="   ", tenant_id="acme")

    with pytest.raises(TenantIsolationError):
        RetrievalRequest(query="test", tenant_id="   ")
