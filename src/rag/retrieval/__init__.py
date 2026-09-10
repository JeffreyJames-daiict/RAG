"""Retrieval package — semantic vector search, BM25 keyword search, hybrid RRF fusion, quality metrics, and table-aware retrieval."""

from rag.retrieval.bm25 import BM25Index, BM25Retriever, TenantBM25IndexManager
from rag.retrieval.bm25_service import BM25RetrievalService
from rag.retrieval.distribution import (
    ChunkDistributionAnalyzer,
    ChunkDistributionReport,
)
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.fusion import FusedCandidateInfo, FusionDiagnostics, FusionStrategy, RRFFusion
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.interfaces import HybridRetriever, VectorRetriever
from rag.retrieval.metrics import RetrievalEvaluator, ScoreDistribution
from rag.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResult,
)
from rag.retrieval.repositories import (
    BaseVectorRepository,
    PythonVectorRepository,
    get_vector_repository,
)
from rag.retrieval.reranking import (
    MockReranker,
    RerankResponse,
    RerankResult,
    Reranker,
    RerankingService,
    get_reranker,
)
from rag.retrieval.retriever import (
    RetrievalFilter,
    RetrievedChunk,
    TableAwareRetriever,
)
from rag.retrieval.tokenizer import extract_phrases, extract_search_terms, tokenize
from rag.retrieval.vector_service import VectorRetrievalService

__all__ = [
    # Legacy / Table-aware retriever
    "TableAwareRetriever",
    "RetrievalFilter",
    "RetrievedChunk",
    # Vector retrieval
    "VectorRetrievalService",
    "VectorRetriever",
    "BaseVectorRepository",
    "PythonVectorRepository",
    "get_vector_repository",
    # BM25 & Hybrid
    "BM25Index",
    "BM25Retriever",
    "BM25RetrievalService",
    "TenantBM25IndexManager",
    "tokenize",
    "extract_search_terms",
    "extract_phrases",
    "RRFFusion",
    "FusedCandidateInfo",
    "FusionDiagnostics",
    "FusionStrategy",
    "HybridRetrievalService",
    "HybridRetriever",
    # Reranking
    "Reranker",
    "RerankResult",
    "RerankResponse",
    "MockReranker",
    "RerankingService",
    "get_reranker",
    # Evaluation & Analytics
    "RetrievalEvaluator",
    "ScoreDistribution",
    "ChunkDistributionAnalyzer",
    "ChunkDistributionReport",
    # Models & Filters
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedChunkResult",
    "RetrievalFilters",
]

