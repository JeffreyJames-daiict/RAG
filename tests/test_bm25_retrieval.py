"""Unit tests for BM25 Lexical Keyword Search and BM25Retriever."""

import pytest

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document
from rag.core.models import DocumentStatus
from rag.retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from rag.retrieval.models import RetrievalRequest


@pytest.fixture
def db_mgr():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


def test_bm25_tokenize():
    """Verify tokenizer correctly lowercases, removes punctuation and stopwords."""
    text = "The quick BROWN fox jumps over the lazy dog!"
    tokens = tokenize(text)
    # Stopwords like 'the', 'over' should be removed
    assert "the" not in tokens
    assert "over" not in tokens
    # Informative terms kept and lowercased
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens


def test_bm25_index_scoring():
    """Verify BM25 index scoring ranks documents by term relevance."""
    docs = [
        ("doc1", "Quantum computing and superconducting qubits"),
        ("doc2", "Deep learning neural network architectures and transformers"),
        ("doc3", "Quantum error correction codes in superconducting circuits"),
    ]
    index = BM25Index(k1=1.5, b=0.75)
    index.index_documents(docs)

    # Query for 'quantum'
    scores = index.score("quantum")
    assert len(scores) == 2  # doc1 and doc3 contain 'quantum'
    matched_ids = [doc_id for doc_id, s in scores]
    assert "doc1" in matched_ids
    assert "doc3" in matched_ids
    assert "doc2" not in matched_ids

    # Query with specific term
    scores_transformers = index.score("transformers")
    assert len(scores_transformers) == 1
    assert scores_transformers[0][0] == "doc2"
    assert scores_transformers[0][1] > 0.0


def test_bm25_retriever_database(db_mgr):
    """Verify BM25Retriever querying from database with tenant isolation."""
    with db_mgr.session() as session:
        # Tenant 1 doc
        d1 = Document(
            document_id="d1",
            tenant_id="tenant_x",
            source="local",
            filename="financials.pdf",
            source_uri="s3://financials.pdf",
            mime_type="application/pdf",
            content_hash="hash_x",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c1 = Chunk(
            chunk_id="c1",
            document_id="d1",
            version=1,
            chunk_index=0,
            text="Quarterly revenue reached twelve million dollars with twenty percent operating margin.",
            section="Financial Highlights",
            page=1,
        )
        c2 = Chunk(
            chunk_id="c2",
            document_id="d1",
            version=1,
            chunk_index=1,
            text="Our corporate headquarters relocated to Seattle Washington.",
            section="Company Info",
            page=2,
        )

        # Tenant 2 doc (must not be returned)
        d2 = Document(
            document_id="d2",
            tenant_id="tenant_y",
            source="local",
            filename="y_report.pdf",
            source_uri="s3://y.pdf",
            mime_type="application/pdf",
            content_hash="hash_y",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        c3 = Chunk(
            chunk_id="c3",
            document_id="d2",
            version=1,
            chunk_index=0,
            text="Revenue in tenant Y was twenty million dollars.",
            section="Summary",
            page=1,
        )

        session.add_all([d1, d2, c1, c2, c3])

    retriever = BM25Retriever(db_manager=db_mgr)

    # Search for revenue in tenant_x
    req = RetrievalRequest(query="quarterly revenue margin", tenant_id="tenant_x", top_k=5)
    candidates, total = retriever.search(req)

    assert total == 2
    assert len(candidates) >= 1
    # c1 must be the top candidate
    assert candidates[0].chunk_id == "c1"
    assert candidates[0].similarity_score > 0.0
    # No leak from tenant_y
    assert all(c.chunk_id != "c3" for c in candidates)
