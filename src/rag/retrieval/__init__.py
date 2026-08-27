"""Retrieval package — semantic, metadata-filtered, and table-aware document retrieval."""

from rag.retrieval.retriever import (
    RetrievalFilter,
    RetrievedChunk,
    TableAwareRetriever,
)

__all__ = [
    "TableAwareRetriever",
    "RetrievalFilter",
    "RetrievedChunk",
]
