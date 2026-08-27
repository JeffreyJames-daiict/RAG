"""Table-aware retrieval engine with metadata filtering and parent-child table expansion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFilter:
    """Metadata filters applied during retrieval."""
    chunk_type: Optional[str] = None       # "text" | "table" | "table_summary" | None (all)
    document_id: Optional[str] = None
    table_id: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    min_tokens: Optional[int] = None


@dataclass
class RetrievedChunk:
    """A scored result chunk returned by the retriever."""
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_type: str
    text: str
    score: float = 1.0
    section: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    table_id: Optional[str] = None
    parent_id: Optional[str] = None


class TableAwareRetriever:
    """Retrieves text and table chunks with chunk-type filtering and parent-child table expansion.

    Usage::

        retriever = TableAwareRetriever(db_manager)
        results = retriever.search(query="patent status", filter=RetrievalFilter(chunk_type="table"))
        for res in results:
            if res.table_id:
                expanded = retriever.expand_parent_table(res.table_id, res.document_id)
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def search(
        self,
        query: str,
        retrieval_filter: Optional[RetrievalFilter] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Search chunks with keyword/exact matching and metadata filtering.

        Args:
            query:            Search query string.
            retrieval_filter: Optional chunk_type, document_id, table_id filters.
            top_k:            Max results to return.

        Returns:
            List of scored RetrievedChunk objects.
        """
        results: List[RetrievedChunk] = []
        rf = retrieval_filter or RetrievalFilter()

        with self.db_manager.session() as session:
            stmt = select(Chunk)
            clauses = []

            if rf.document_id:
                clauses.append(Chunk.document_id == rf.document_id)
            if rf.page is not None:
                clauses.append(Chunk.page == rf.page)
            if rf.section:
                clauses.append(Chunk.section.ilike(f"%{rf.section}%"))

            if clauses:
                stmt = stmt.where(and_(*clauses))

            chunks = session.execute(stmt).scalars().all()

            # Filter and score in-memory (extensible to vector cosine search)
            query_terms = [t.lower() for t in query.split() if len(t) > 1]

            scored_candidates = []
            for ch in chunks:
                meta = ch.chunk_metadata or {}
                ch_type = meta.get("chunk_type", "text")

                # Filter by chunk_type if requested
                if rf.chunk_type and ch_type != rf.chunk_type:
                    continue

                if rf.table_id and meta.get("table_id") != rf.table_id:
                    continue

                # Term frequency scoring
                text_lower = ch.text.lower()
                matched_terms = sum(1 for term in query_terms if term in text_lower)

                if query_terms:
                    score = matched_terms / len(query_terms)
                else:
                    score = 1.0

                if score > 0 or not query_terms:
                    # Boost table chunks slightly if query looks tabular
                    is_table = ch_type in ("table", "table_summary")
                    final_score = score * (1.1 if is_table and any(k in query.lower() for k in ["table", "status", "price", "count", "cost"]) else 1.0)
                    scored_candidates.append((final_score, ch))

            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            for score, ch in scored_candidates[:top_k]:
                meta = ch.chunk_metadata or {}
                results.append(
                    RetrievedChunk(
                        chunk_id=ch.chunk_id,
                        document_id=ch.document_id,
                        chunk_index=ch.chunk_index,
                        chunk_type=meta.get("chunk_type", "text"),
                        text=ch.text,
                        score=round(score, 4),
                        section=ch.section,
                        page=ch.page,
                        metadata=meta,
                        table_id=meta.get("table_id"),
                        parent_id=meta.get("parent_id"),
                    )
                )

        return results

    def expand_parent_table(
        self,
        table_id: str,
        document_id: str,
        target_chunk_index: Optional[int] = None,
        neighbor_window: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve sibling child chunks for a table to reconstitute bounded table context.

        Args:
            table_id:           The unique table identifier (e.g. "report_p2_t1").
            document_id:        UUID of the parent document.
            target_chunk_index: Optional chunk_index of the matched chunk for localized expansion.
            neighbor_window:    Max neighboring row chunks to include before and after target.
            max_rows:           Hard ceiling on total table rows returned.

        Returns:
            Sorted list of relevant table chunks (summary + bounded row chunks).
        """
        all_table_chunks: List[RetrievedChunk] = []
        with self.db_manager.session() as session:
            stmt = (
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            )
            chunks = session.execute(stmt).scalars().all()

            for ch in chunks:
                meta = ch.chunk_metadata or {}
                # Match either table_id or parent_id
                if meta.get("table_id") == table_id or meta.get("parent_id") == table_id:
                    all_table_chunks.append(
                        RetrievedChunk(
                            chunk_id=ch.chunk_id,
                            document_id=ch.document_id,
                            chunk_index=ch.chunk_index,
                            chunk_type=meta.get("chunk_type", "table"),
                            text=ch.text,
                            score=1.0,
                            section=ch.section,
                            page=ch.page,
                            metadata=meta,
                            table_id=meta.get("table_id"),
                            parent_id=meta.get("parent_id"),
                        )
                    )

        if not all_table_chunks:
            return []

        # If bounded neighbor window is requested and target_chunk_index is specified
        if target_chunk_index is not None and neighbor_window is not None:
            # Find target index in all_table_chunks
            target_pos = None
            for idx, c in enumerate(all_table_chunks):
                if c.chunk_index == target_chunk_index:
                    target_pos = idx
                    break

            if target_pos is not None:
                start_pos = max(0, target_pos - neighbor_window)
                end_pos = min(len(all_table_chunks), target_pos + neighbor_window + 1)
                selected_chunks = all_table_chunks[start_pos:end_pos]

                # Always include summary chunk if available and not already in slice
                summary_chunk = next((c for c in all_table_chunks if c.chunk_type == "table_summary"), None)
                if summary_chunk and summary_chunk not in selected_chunks:
                    selected_chunks.insert(0, summary_chunk)

                all_table_chunks = selected_chunks

        # Apply max_rows budget ceiling if specified
        if max_rows is not None:
            filtered_chunks: List[RetrievedChunk] = []
            accumulated_rows = 0
            for c in all_table_chunks:
                if c.chunk_type == "table_summary":
                    filtered_chunks.append(c)
                    continue

                r_start = c.metadata.get("row_start", 1)
                r_end = c.metadata.get("row_end", r_start)
                row_span = (r_end - r_start + 1)

                if accumulated_rows + row_span <= max_rows or not filtered_chunks:
                    filtered_chunks.append(c)
                    accumulated_rows += row_span
                else:
                    break
            all_table_chunks = filtered_chunks

        logger.info(
            "Expanded table '%s' for doc %s -> %d bounded chunk(s)",
            table_id, document_id, len(all_table_chunks),
        )
        return all_table_chunks

