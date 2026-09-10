"""Parent and neighbor chunk expansion engine.

Enriches retrieved focal passages with surrounding narrative or parent structural
context while strictly respecting expansion caps and document boundaries.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from rag.core.config import ContextBuilderConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk
from rag.retrieval.models import RetrievedChunkResult

logger = logging.getLogger(__name__)


class ContextExpander:
    """Expands seed retrieval chunks with previous, next, or parent context."""

    def __init__(
        self,
        config: Optional[ContextBuilderConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        custom_resolver: Optional[Callable[[str, str, int, Optional[str]], List[RetrievedChunkResult]]] = None,
    ):
        self.config = config or ContextBuilderConfig()
        self.db_manager = db_manager
        self.custom_resolver = custom_resolver

    def _fetch_from_db(
        self,
        document_id: str,
        version: int,
        target_indices: List[int],
        parent_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Fetch neighbor or parent chunks directly from the database."""
        if not self.db_manager:
            return []

        expanded_chunks: List[Chunk] = []
        try:
            with self.db_manager.session() as session:
                if target_indices:
                    query = (
                        session.query(Chunk)
                        .filter(
                            Chunk.document_id == document_id,
                            Chunk.version == version,
                            Chunk.chunk_index.in_(target_indices),
                        )
                        .order_by(Chunk.chunk_index.asc())
                    )
                    expanded_chunks.extend(query.all())

                if parent_id:
                    parent_query = (
                        session.query(Chunk)
                        .filter(
                            Chunk.document_id == document_id,
                            Chunk.chunk_id == parent_id,
                        )
                    )
                    parent_res = parent_query.first()
                    if parent_res and parent_res not in expanded_chunks:
                        expanded_chunks.append(parent_res)
        except Exception as e:
            logger.warning("Error querying database for chunk expansion: %s", e)

        return expanded_chunks

    def expand(
        self,
        seed_chunks: List[RetrievedChunkResult],
    ) -> List[RetrievedChunkResult]:
        """Expand seed chunks with adjacent previous/next and parent context.

        Returns:
            List containing original seed chunks and newly admitted expanded chunks,
            annotated with expansion metadata.
        """
        if not (self.config.expand_neighbors or self.config.expand_parents):
            return list(seed_chunks)

        existing_ids: Set[str] = {c.chunk_id for c in seed_chunks}
        doc_chunk_counts: Dict[str, int] = {}
        for c in seed_chunks:
            doc_chunk_counts[c.document_id] = doc_chunk_counts.get(c.document_id, 0) + 1

        all_results: List[RetrievedChunkResult] = list(seed_chunks)

        for seed in seed_chunks:
            doc_id = seed.document_id
            version = seed.document_version
            chunk_idx = seed.chunk_index
            max_doc = self.config.max_chunks_per_document

            if doc_chunk_counts.get(doc_id, 0) >= max_doc:
                continue

            indices_to_fetch: List[int] = []
            if self.config.expand_neighbors:
                if chunk_idx > 0:
                    indices_to_fetch.append(chunk_idx - 1)
                indices_to_fetch.append(chunk_idx + 1)

            parent_id: Optional[str] = None
            if self.config.expand_parents:
                meta = seed.metadata or {}
                parent_id = meta.get("parent_chunk_id") or meta.get("parent_id")
                if parent_id in existing_ids:
                    parent_id = None

            # Resolve candidates via custom resolver or DB
            new_candidates: List[RetrievedChunkResult] = []
            if self.custom_resolver:
                resolved = self.custom_resolver(doc_id, seed.chunk_id, chunk_idx, parent_id)
                new_candidates.extend(resolved)
            elif self.db_manager:
                db_chunks = self._fetch_from_db(doc_id, version, indices_to_fetch, parent_id)
                for d_chunk in db_chunks:
                    if d_chunk.chunk_id in existing_ids:
                        continue
                    exp_type = "parent" if d_chunk.chunk_id == parent_id else (
                        "neighbor_prev" if d_chunk.chunk_index < chunk_idx else "neighbor_next"
                    )
                    chunk_res = RetrievedChunkResult(
                        chunk_id=d_chunk.chunk_id,
                        document_id=d_chunk.document_id,
                        document_version=d_chunk.version,
                        document_version_id=f"v{d_chunk.version}",
                        tenant_id=seed.tenant_id,
                        text=d_chunk.text,
                        rank=seed.rank,  # Derived from seed
                        section=d_chunk.section or seed.section,
                        page=d_chunk.page or seed.page,
                        page_number=d_chunk.page or seed.page,
                        filename=seed.filename,
                        chunk_type=d_chunk.chunk_metadata.get("chunk_type", "text") if d_chunk.chunk_metadata else "text",
                        chunk_index=d_chunk.chunk_index,
                        similarity_score=seed.similarity_score * 0.9,  # slightly decayed
                        raw_score=seed.raw_score * 0.9,
                        metadata={
                            **(d_chunk.chunk_metadata or {}),
                            "is_expanded": True,
                            "expansion_type": exp_type,
                            "parent_seed_id": seed.chunk_id,
                        },
                    )
                    new_candidates.append(chunk_res)

            # Enforce max expansion count per seed
            added_for_seed = 0
            for cand in new_candidates:
                if cand.chunk_id in existing_ids:
                    continue
                if doc_chunk_counts.get(doc_id, 0) >= max_doc:
                    break
                if added_for_seed >= self.config.max_expansion_count:
                    break

                existing_ids.add(cand.chunk_id)
                doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
                added_for_seed += 1
                all_results.append(cand)

        return all_results
