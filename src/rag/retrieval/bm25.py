"""BM25 (Best Matching 25) Sparse Keyword Retrieval Subsystem.

Implements production-grade lexical and sparse keyword retrieval featuring:
- Okapi BM25 scoring with Robertson-Spärck Jones IDF (Lucene +1 smoothing)
- Weighted searchable fields: title, headings, keywords, body, tables, captions, OCR, asset descriptions
- Technical identifier preservation: patent numbers, product codes, claim IDs, company names
- Quoted exact phrase matching with boost
- Inverted index with thread-safe tenant isolation (TenantBM25IndexManager)
- Complete indexing lifecycle: initial, incremental, version eviction, refresh, and status reporting
- Pre-filtering: strict tenant isolation, non-deleted status, active version matching, and metadata filters
- High-level candidate extraction compatible with Vector Retrieval and RRF Fusion
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import Select, and_, select

from rag.core.config import BM25Config, get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document
from rag.core.models import DocumentStatus
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest
from rag.retrieval.tokenizer import (
    extract_phrases,
    extract_search_terms,
    normalize_text,
    tokenize,
)

logger = logging.getLogger(__name__)


@dataclass
class BM25ChunkRecord:
    """Internal metadata and token representation of an indexed chunk."""

    chunk_id: str
    document_id: str
    version: int
    chunk_index: int
    text: str
    title: str = ""
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_type: str = "text"
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    chunking_version: str = "v1.0"
    source_uri: str = ""
    filename: str = ""
    mime_type: str = ""
    content_hash: str = ""
    # Searchable text segments for phrase verification
    full_searchable_text: str = ""
    # Weighted term frequencies: term -> weighted_frequency
    term_frequencies: Dict[str, float] = field(default_factory=dict)
    # Effective weighted document length
    effective_length: float = 0.0


class BM25Index:
    """Inverted index implementing weighted multi-field Okapi BM25 scoring."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 2.5,
        heading_weight: float = 2.0,
        keywords_weight: float = 1.8,
        table_weight: float = 1.5,
        caption_weight: float = 1.2,
        ocr_weight: float = 1.0,
        asset_weight: float = 1.0,
        body_weight: float = 1.0,
    ):
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self.heading_weight = heading_weight
        self.keywords_weight = keywords_weight
        self.table_weight = table_weight
        self.caption_weight = caption_weight
        self.ocr_weight = ocr_weight
        self.asset_weight = asset_weight
        self.body_weight = body_weight

        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.total_doc_length: float = 0.0

        # Chunk storage: chunk_id -> BM25ChunkRecord
        self.chunks: Dict[str, BM25ChunkRecord] = {}
        # Inverted index: term -> (chunk_id -> weighted_frequency)
        self.inverted_index: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        # Document frequency: term -> number of chunks containing term
        self.doc_freqs: Dict[str, int] = collections.defaultdict(int)
        # Precomputed Robertson-Spärck Jones IDF: term -> idf
        self.idf: Dict[str, float] = {}

    def clear(self) -> None:
        """Clear the index completely."""
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.total_doc_length = 0.0
        self.chunks.clear()
        self.inverted_index.clear()
        self.doc_freqs.clear()
        self.idf.clear()

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        document_id: str = "",
        version: int = 1,
        chunk_index: int = 0,
        title: str = "",
        section: Optional[str] = None,
        page: Optional[int] = None,
        chunk_type: str = "text",
        chunk_metadata: Optional[Dict[str, Any]] = None,
        chunking_version: str = "v1.0",
        source_uri: str = "",
        filename: str = "",
        mime_type: str = "",
        content_hash: str = "",
    ) -> None:
        """Add or idempotently replace a chunk in the inverted index."""
        # If chunk already exists, remove it first to preserve invariant idempotency
        if chunk_id in self.chunks:
            self.remove_chunk(chunk_id)

        meta = chunk_metadata or {}
        # Extract weighted field texts
        clean_text = text or ""
        clean_title = title or filename or meta.get("title", "")
        clean_section = section or meta.get("heading", "") or meta.get("section", "")
        clean_keywords = str(meta.get("keywords", "")) + " " + str(meta.get("entities", ""))
        
        # Table text
        is_table = chunk_type == "table" or "table" in str(clean_section).lower()
        clean_table = clean_text if is_table else meta.get("table_summary", "")
        clean_caption = meta.get("caption", "") or meta.get("figure_caption", "")
        clean_ocr = meta.get("ocr_text", "")
        clean_asset = meta.get("asset_description", "") or meta.get("description", "")

        # Compute weighted term frequencies
        term_freqs: Dict[str, float] = collections.defaultdict(float)

        def _accumulate(field_str: str, weight: float) -> None:
            if not field_str or weight <= 0.0:
                return
            field_tokens = tokenize(field_str, remove_stopwords=True, preserve_identifiers=True)
            for tok in field_tokens:
                term_freqs[tok] += weight

        _accumulate(clean_text, self.body_weight)
        _accumulate(clean_title, self.title_weight)
        _accumulate(clean_section, self.heading_weight)
        _accumulate(clean_keywords, self.keywords_weight)
        _accumulate(clean_table, self.table_weight)
        _accumulate(clean_caption, self.caption_weight)
        _accumulate(clean_ocr, self.ocr_weight)
        _accumulate(clean_asset, self.asset_weight)

        effective_len = sum(term_freqs.values())
        if effective_len == 0.0:
            effective_len = 1.0  # Avoid division by zero

        full_searchable = f"{clean_title} {clean_section} {clean_text} {clean_caption} {clean_ocr}".lower()

        record = BM25ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            version=version,
            chunk_index=chunk_index,
            text=clean_text,
            title=clean_title,
            section=clean_section,
            page=page,
            chunk_type=chunk_type,
            chunk_metadata=meta,
            chunking_version=chunking_version,
            source_uri=source_uri,
            filename=filename,
            mime_type=mime_type,
            content_hash=content_hash,
            full_searchable_text=full_searchable,
            term_frequencies=dict(term_freqs),
            effective_length=effective_len,
        )

        self.chunks[chunk_id] = record
        self.corpus_size += 1
        self.total_doc_length += effective_len

        # Update inverted index and document frequencies
        for term, freq in term_freqs.items():
            self.inverted_index[term][chunk_id] = freq
            self.doc_freqs[term] += 1

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a single chunk from the inverted index."""
        record = self.chunks.pop(chunk_id, None)
        if not record:
            return

        self.corpus_size = max(0, self.corpus_size - 1)
        self.total_doc_length = max(0.0, self.total_doc_length - record.effective_length)

        for term in record.term_frequencies:
            if term in self.inverted_index:
                self.inverted_index[term].pop(chunk_id, None)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

            if term in self.doc_freqs:
                self.doc_freqs[term] = max(0, self.doc_freqs[term] - 1)
                if self.doc_freqs[term] == 0:
                    del self.doc_freqs[term]
                    self.idf.pop(term, None)

    def remove_document(self, document_id: str, version: Optional[int] = None) -> int:
        """Remove all chunks associated with a document (or specific document version).
        
        Returns:
            Number of chunks removed.
        """
        to_remove = [
            cid for cid, ch in self.chunks.items()
            if ch.document_id == document_id and (version is None or ch.version == version)
        ]
        for cid in to_remove:
            self.remove_chunk(cid)
        return len(to_remove)

    def recompute_idf(self) -> None:
        """Recompute Robertson-Spärck Jones IDF for all terms in the corpus."""
        self.idf.clear()
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        self.avg_doc_len = self.total_doc_length / self.corpus_size

        # Robertson-Spärck Jones BM25 IDF formula with +1 smoothing (Lucene standard)
        # IDF(q) = ln(1.0 + (N - n + 0.5) / (n + 0.5))
        n_docs = self.corpus_size
        for term, df in self.doc_freqs.items():
            idf_val = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = idf_val

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Legacy helper: index a list of (doc_id, text) pairs and compute IDF."""
        self.clear()
        for doc_id, text in documents:
            self.add_chunk(chunk_id=doc_id, text=text)
        self.recompute_idf()

    def score(
        self,
        query: str,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Score all indexed chunks against query, returning sorted (chunk_id, score) descending."""
        if not query or not query.strip() or self.corpus_size == 0 or self.avg_doc_len == 0.0:
            return []

        search_info = extract_search_terms(query)
        query_tokens = search_info["tokens"]
        phrases = search_info["phrases"]
        identifiers = search_info["identifiers"]

        if not query_tokens:
            return []

        scores: Dict[str, float] = collections.defaultdict(float)

        # Okapi BM25 scoring across terms
        for term in query_tokens:
            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0.0:
                continue

            posting_list = self.inverted_index.get(term, {})
            for chunk_id, freq in posting_list.items():
                record = self.chunks.get(chunk_id)
                if not record:
                    continue

                doc_len = record.effective_length
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf_val * (freq * (self.k1 + 1.0)) / denom
                scores[chunk_id] += term_score

        # Apply exact phrase match bonus
        if phrases:
            for phrase in phrases:
                for chunk_id in list(scores.keys()):
                    record = self.chunks.get(chunk_id)
                    if record and phrase in record.full_searchable_text:
                        scores[chunk_id] += 2.0  # Significant boost for exact phrase match

        # Apply technical identifier match bonus
        if identifiers:
            for ident in identifiers:
                norm_ident = normalize_text(ident).lower()
                for chunk_id in list(scores.keys()):
                    record = self.chunks.get(chunk_id)
                    if record and norm_ident in record.full_searchable_text:
                        scores[chunk_id] += 3.0  # Major boost for exact identifier match

        # Filter by min_score
        results: List[Tuple[str, float]] = [
            (cid, sc) for cid, sc in scores.items() if sc >= min_score
        ]

        # Deterministic sorting: score DESC, chunk_index ASC, chunk_id ASC
        def _sort_key(item: Tuple[str, float]) -> Tuple[float, int, str]:
            cid, sc = item
            rec = self.chunks.get(cid)
            c_idx = rec.chunk_index if rec else 0
            return (-sc, c_idx, cid)

        results.sort(key=_sort_key)
        return results


class TenantBM25IndexManager:
    """Thread-safe manager for tenant-isolated in-memory BM25 indices."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: Optional[BM25Config] = None,
    ):
        self.db_manager = db_manager
        self.config = config or get_settings().bm25
        self._lock = threading.RLock()
        self._indices: Dict[str, BM25Index] = {}
        self._last_refresh: Dict[str, float] = {}
        self._status: Dict[str, Dict[str, Any]] = {}

    def get_index(self, tenant_id: str, auto_hydrate: bool = True) -> BM25Index:
        """Get or initialize the BM25 index for a specific tenant."""
        with self._lock:
            # Check refresh interval
            now = time.time()
            refresh_interval = self.config.index_refresh_interval
            if tenant_id in self._indices and refresh_interval > 0:
                last_time = self._last_refresh.get(tenant_id, 0.0)
                if now - last_time > refresh_interval:
                    logger.info("BM25 index for tenant '%s' expired (>%ds); refreshing.", tenant_id, refresh_interval)
                    self.hydrate_tenant(tenant_id, force=True)

            if tenant_id not in self._indices:
                idx = BM25Index(
                    k1=self.config.k1,
                    b=self.config.b,
                    title_weight=self.config.title_weight,
                    heading_weight=self.config.heading_weight,
                    body_weight=self.config.body_weight,
                    table_weight=self.config.table_weight,
                    caption_weight=self.config.caption_weight,
                )
                self._indices[tenant_id] = idx
                if auto_hydrate:
                    self.hydrate_tenant(tenant_id, force=False)

            return self._indices[tenant_id]

    def hydrate_tenant(self, tenant_id: str, force: bool = False) -> int:
        """Hydrate tenant index from persistent database chunks.
        
        Returns:
            Count of indexed chunks.
        """
        with self._lock:
            if tenant_id not in self._indices:
                self._indices[tenant_id] = BM25Index(
                    k1=self.config.k1,
                    b=self.config.b,
                    title_weight=self.config.title_weight,
                    heading_weight=self.config.heading_weight,
                    body_weight=self.config.body_weight,
                    table_weight=self.config.table_weight,
                    caption_weight=self.config.caption_weight,
                )

            idx = self._indices[tenant_id]
            if force:
                idx.clear()

            start_t = time.time()
            failed_count = 0

            # Query only active, indexed, non-deleted chunks for the tenant
            stmt = (
                select(
                    Chunk.chunk_id,
                    Chunk.document_id,
                    Chunk.version,
                    Chunk.chunk_index,
                    Chunk.text,
                    Chunk.section,
                    Chunk.page,
                    Chunk.chunk_metadata,
                    Chunk.chunking_version,
                    Document.source_uri,
                    Document.filename,
                    Document.mime_type,
                    Document.content_hash,
                )
                .join(Document, Chunk.document_id == Document.document_id)
                .where(
                    Document.tenant_id == tenant_id,
                    Document.deleted_at.is_(None),
                    Document.status == DocumentStatus.INDEXED.value,
                    Chunk.version == Document.version,
                )
            )

            with self.db_manager.session() as session:
                rows = session.execute(stmt).all()

            for row in rows:
                try:
                    idx.add_chunk(
                        chunk_id=row.chunk_id,
                        document_id=row.document_id,
                        version=row.version,
                        chunk_index=row.chunk_index,
                        text=row.text,
                        title=row.filename or "",
                        section=row.section,
                        page=row.page,
                        chunk_type=(row.chunk_metadata or {}).get("chunk_type", "text"),
                        chunk_metadata=row.chunk_metadata or {},
                        chunking_version=row.chunking_version,
                        source_uri=row.source_uri or "",
                        filename=row.filename or "",
                        mime_type=row.mime_type or "",
                        content_hash=row.content_hash or "",
                    )
                except Exception as e:
                    failed_count += 1
                    logger.warning("Failed to index chunk %s for tenant %s: %s", row.chunk_id, tenant_id, e)

            idx.recompute_idf()
            duration_ms = (time.time() - start_t) * 1000.0
            self._last_refresh[tenant_id] = time.time()

            # Record telemetry
            self._status[tenant_id] = {
                "tenant_id": tenant_id,
                "corpus_size": idx.corpus_size,
                "total_terms": len(idx.doc_freqs),
                "avg_doc_len": round(idx.avg_doc_len, 2),
                "indexed_chunks": idx.corpus_size,
                "failed_chunks": failed_count,
                "indexing_duration_ms": round(duration_ms, 2),
                "last_refreshed_at": datetime.now(timezone.utc).isoformat(),
                "index_version": "v1.0",
                "memory_bytes_estimate": sys.getsizeof(idx.inverted_index) + sys.getsizeof(idx.chunks),
            }

            logger.info(
                "Tenant '%s' BM25 index hydrated: %d chunks (%d failed) in %.2f ms",
                tenant_id, idx.corpus_size, failed_count, duration_ms
            )
            return idx.corpus_size

    def index_chunk(
        self,
        tenant_id: str,
        chunk_id: str,
        text: str,
        document_id: str = "",
        version: int = 1,
        chunk_index: int = 0,
        title: str = "",
        section: Optional[str] = None,
        page: Optional[int] = None,
        chunk_type: str = "text",
        chunk_metadata: Optional[Dict[str, Any]] = None,
        chunking_version: str = "v1.0",
        source_uri: str = "",
        filename: str = "",
        mime_type: str = "",
        content_hash: str = "",
    ) -> None:
        """Incrementally add or update a chunk in the tenant's index."""
        with self._lock:
            idx = self.get_index(tenant_id, auto_hydrate=False)
            idx.add_chunk(
                chunk_id=chunk_id,
                text=text,
                document_id=document_id,
                version=version,
                chunk_index=chunk_index,
                title=title,
                section=section,
                page=page,
                chunk_type=chunk_type,
                chunk_metadata=chunk_metadata,
                chunking_version=chunking_version,
                source_uri=source_uri,
                filename=filename,
                mime_type=mime_type,
                content_hash=content_hash,
            )
            idx.recompute_idf()

    def remove_document(self, tenant_id: str, document_id: str, version: Optional[int] = None) -> int:
        """Incrementally remove document chunks from tenant index."""
        with self._lock:
            if tenant_id in self._indices:
                removed = self._indices[tenant_id].remove_document(document_id, version=version)
                self._indices[tenant_id].recompute_idf()
                return removed
            return 0

    def reindex_tenant(self, tenant_id: str) -> int:
        """Force full reindexing of tenant from database."""
        return self.hydrate_tenant(tenant_id, force=True)

    def delete_tenant_index(self, tenant_id: str) -> None:
        """Evict tenant index from memory."""
        with self._lock:
            self._indices.pop(tenant_id, None)
            self._last_refresh.pop(tenant_id, None)
            self._status.pop(tenant_id, None)

    def get_status(self, tenant_id: str) -> Dict[str, Any]:
        """Return status and telemetry for a tenant index."""
        with self._lock:
            if tenant_id not in self._status:
                idx = self.get_index(tenant_id, auto_hydrate=True)
                return self._status.get(tenant_id, {
                    "tenant_id": tenant_id,
                    "corpus_size": idx.corpus_size,
                    "index_version": "v1.0",
                })
            status = dict(self._status[tenant_id])
            last_t = self._last_refresh.get(tenant_id, 0.0)
            status["index_freshness_sec"] = round(time.time() - last_t, 2) if last_t else 0.0
            return status


class BM25Retriever:
    """Production BM25 Lexical Keyword Retriever for Chunk documents."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        k1: float = 1.5,
        b: float = 0.75,
        candidate_limit: int = 1000,
        config: Optional[BM25Config] = None,
        index_manager: Optional[TenantBM25IndexManager] = None,
    ):
        self.db_manager = db_manager
        self.config = config or get_settings().bm25
        self.k1 = k1 or self.config.k1
        self.b = b or self.config.b
        self.candidate_limit = candidate_limit
        self.index_manager = index_manager or TenantBM25IndexManager(db_manager, self.config)

    def build_chunk_select_stmt(self, request: RetrievalRequest) -> Select:
        """Construct SELECT statement fetching chunks for candidate BM25 search."""
        stmt = (
            select(
                Chunk.chunk_id,
                Chunk.document_id,
                Chunk.version,
                Chunk.chunk_index,
                Chunk.text,
                Chunk.section,
                Chunk.page,
                Chunk.chunk_metadata,
                Chunk.chunking_version,
                Document.source_uri,
                Document.filename,
                Document.mime_type,
                Document.content_hash,
            )
            .join(Document, Chunk.document_id == Document.document_id)
            .where(Document.tenant_id == request.tenant_id)
        )

        if not request.include_inactive:
            stmt = stmt.where(
                Document.deleted_at.is_(None),
                Document.status == DocumentStatus.INDEXED.value,
                Chunk.version == Document.version,
            )

        if request.filters:
            clauses = request.filters.build_sql_clauses()
            if clauses:
                stmt = stmt.where(and_(*clauses))

        stmt = stmt.limit(self.candidate_limit)
        return stmt

    def search(
        self,
        request: RetrievalRequest,
    ) -> Tuple[List[RetrievedCandidate], int]:
        """Execute BM25 keyword retrieval against persisted chunks with strict tenant isolation.

        Args:
            request: Validated retrieval request.

        Returns:
            Tuple of (list of RetrievedCandidate sorted by BM25 score desc, total candidates evaluated).
        """
        # When request has custom filters (e.g. specific document_ids or chunk_types),
        # query the filtered candidate chunks from database directly to construct an isolated scoped index.
        has_custom_filters = bool(request.filters and request.filters.build_sql_clauses())
        
        if has_custom_filters or request.include_inactive:
            # Build scoped ad-hoc index
            stmt = self.build_chunk_select_stmt(request)
            with self.db_manager.session() as session:
                rows = session.execute(stmt).all()

            total_candidates = len(rows)
            if total_candidates == 0:
                return [], 0

            scoped_index = BM25Index(
                k1=self.k1,
                b=self.b,
                title_weight=self.config.title_weight,
                heading_weight=self.config.heading_weight,
                body_weight=self.config.body_weight,
                table_weight=self.config.table_weight,
                caption_weight=self.config.caption_weight,
            )
            for row in rows:
                scoped_index.add_chunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    version=row.version,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    title=row.filename or "",
                    section=row.section,
                    page=row.page,
                    chunk_type=(row.chunk_metadata or {}).get("chunk_type", "text"),
                    chunk_metadata=row.chunk_metadata or {},
                    chunking_version=row.chunking_version,
                    source_uri=row.source_uri or "",
                    filename=row.filename or "",
                    mime_type=row.mime_type or "",
                    content_hash=row.content_hash or "",
                )
            scoped_index.recompute_idf()
            scored_pairs = scoped_index.score(request.query, min_score=self.config.min_score)
            active_index = scoped_index
        else:
            # Use tenant index manager cache
            tenant_index = self.index_manager.get_index(request.tenant_id, auto_hydrate=True)
            total_candidates = tenant_index.corpus_size
            if total_candidates == 0:
                return [], 0
            scored_pairs = tenant_index.score(request.query, min_score=self.config.min_score)
            active_index = tenant_index

        candidates: List[RetrievedCandidate] = []
        for chunk_id, score in scored_pairs:
            rec = active_index.chunks.get(chunk_id)
            if not rec:
                continue

            # Normalized score: sigmoid mapping of BM25 score into [0, 1]
            norm_score = 1.0 / (1.0 + math.exp(-score / 5.0)) if score > 0 else 0.0

            cand = RetrievedCandidate(
                chunk_id=rec.chunk_id,
                document_id=rec.document_id,
                document_version=rec.version,
                chunk_index=rec.chunk_index,
                text=rec.text,
                similarity_score=float(score),
                raw_score=float(score),
                normalized_score=float(norm_score),
                distance=max(0.0, 1.0 / (1.0 + score)),
                title=rec.title,
                section=rec.section,
                page=rec.page,
                chunk_metadata=rec.chunk_metadata,
                chunking_version=rec.chunking_version,
                embedding_model="bm25",
                embedding_provider="bm25",
                embedding_dimension=0,
                source_uri=rec.source_uri,
                filename=rec.filename,
                mime_type=rec.mime_type,
                content_hash=rec.content_hash,
            )
            candidates.append(cand)

        return candidates[: request.top_k], total_candidates
