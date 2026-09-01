"""Unit tests for EmbeddingService."""

import uuid
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from rag.core.config import EmbeddingConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.exceptions import EmbeddingInputTooLargeError
from rag.embeddings.models import EmbeddingStatus, EmbeddingTask
from rag.embeddings.providers.mock_provider import MockEmbeddingProvider
from rag.embeddings.service import EmbeddingService


@pytest.fixture
def db_mgr():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


def test_service_validate_and_prepare_chunk(db_mgr):
    service = EmbeddingService(db_manager=db_mgr)

    # Empty chunk
    c_empty = Chunk(chunk_id="c0", document_id="d1", version=1, chunk_index=0, text="   ")
    assert service.validate_and_prepare_chunk(c_empty) is None

    # Valid chunk
    c_valid = Chunk(chunk_id="c1", document_id="d1", version=1, chunk_index=1, text="Hello enterprise RAG")
    task = service.validate_and_prepare_chunk(c_valid)
    assert task is not None
    assert task.chunk_id == "c1"
    assert task.text == "Hello enterprise RAG"
    assert task.token_count > 0


def test_service_oversized_chunk_handling(db_mgr):
    provider = MockEmbeddingProvider(max_input_tokens=10)

    # Truncate strategy
    cfg_truncate = EmbeddingConfig(oversized_chunk_strategy="truncate")
    service_trunc = EmbeddingService(db_manager=db_mgr, provider=provider, config=cfg_truncate)
    c_large = Chunk(chunk_id="c_lg", document_id="d1", version=1, chunk_index=0, text="Word " * 50)
    task_trunc = service_trunc.validate_and_prepare_chunk(c_large)
    assert task_trunc is not None
    assert task_trunc.token_count <= 10

    # Reject strategy
    cfg_reject = EmbeddingConfig(oversized_chunk_strategy="reject")
    service_reject = EmbeddingService(db_manager=db_mgr, provider=provider, config=cfg_reject)
    task_reject = service_reject.validate_and_prepare_chunk(c_large)
    assert task_reject is None

    # Error strategy
    cfg_error = EmbeddingConfig(oversized_chunk_strategy="error")
    service_error = EmbeddingService(db_manager=db_mgr, provider=provider, config=cfg_error)
    with pytest.raises(EmbeddingInputTooLargeError):
        service_error.validate_and_prepare_chunk(c_large)


def test_service_batch_processing_and_persistence(db_mgr):
    provider = MockEmbeddingProvider(dimension=64)
    service = EmbeddingService(db_manager=db_mgr, provider=provider)

    tasks = [
        EmbeddingTask(chunk_id=f"chk_{i}", text=f"Chunk text content {i}", document_id="doc_1", version=1)
        for i in range(5)
    ]

    batch_res = service.process_batch(tasks)
    assert batch_res.status == EmbeddingStatus.SUCCESS
    assert len(batch_res.successful_results) == 5
    assert len(batch_res.failed_chunks) == 0

    with db_mgr.session() as session:
        persisted = service.persist_results(batch_res.successful_results, session=session)
        assert persisted == 5

    with db_mgr.session() as session:
        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) == 5
        assert len(embs[0].vector) == 64
        assert embs[0].dimension == 64
        assert embs[0].provider == "mock"


def test_service_batch_fallback_on_single_chunk_failure(db_mgr):
    mock_provider = MagicMock()
    mock_provider.dimension = 32
    mock_provider.provider_name = "mock_flaky"
    mock_provider.model_name = "mock_model"
    mock_provider.model_version = "v1.0"
    mock_provider.max_input_tokens = 8191

    # Batch call raises exception, but individual calls succeed except for one bad chunk
    def fake_embed_batch(texts):
        if len(texts) > 1:
            raise RuntimeError("Batch request failed!")
        if "bad" in texts[0]:
            raise ValueError("Corrupt text input")
        return [[0.1] * 32]

    mock_provider.embed_batch.side_effect = fake_embed_batch

    service = EmbeddingService(db_manager=db_mgr, provider=mock_provider)
    tasks = [
        EmbeddingTask(chunk_id="good_1", text="Good text 1", document_id="doc_1", version=1),
        EmbeddingTask(chunk_id="bad_1", text="This is bad text", document_id="doc_1", version=1),
        EmbeddingTask(chunk_id="good_2", text="Good text 2", document_id="doc_1", version=1),
    ]

    batch_res = service.process_batch(tasks)
    assert batch_res.status == EmbeddingStatus.PARTIAL_SUCCESS
    assert len(batch_res.successful_results) == 2
    assert len(batch_res.failed_chunks) == 1
    assert batch_res.failed_chunks[0].chunk_id == "bad_1"


def test_service_embed_query(db_mgr):
    service = EmbeddingService(db_manager=db_mgr)
    vec = service.embed_query("Enterprise search query")
    assert len(vec) == service.dimension

    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_query("   ")
