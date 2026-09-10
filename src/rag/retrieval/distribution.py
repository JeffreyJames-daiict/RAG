"""Chunk Metadata Distribution Analyzer.

Computes comprehensive statistical distributions and health metrics across
ingested document chunks for a tenant or specific document:
- Chunk type frequencies (text, table, visual, etc.)
- Token count statistics (min, max, mean, median, p95, histogram)
- Section hierarchy and breadcrumb coverage
- Content source and OCR distribution (native vs scanned vs OCR)
- Page coverage and density
- Language distribution
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.core.db_models import Chunk, ChunkMetadata, Document


class TokenStatistics(BaseModel):
    """Token length statistical distribution."""

    count: int = 0
    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stddev: float = 0.0
    histogram: Dict[str, int] = Field(
        default_factory=dict,
        description="Bucket frequencies (e.g. '<100', '100-250', '250-500', '500-1000', '>1000')",
    )


class PageCoverage(BaseModel):
    """Page distribution metrics across chunks."""

    total_pages_covered: int = 0
    min_page: Optional[int] = None
    max_page: Optional[int] = None
    mean_chunks_per_page: float = 0.0
    max_chunks_per_page: int = 0


class SectionDistribution(BaseModel):
    """Section header distribution metrics."""

    chunks_with_section: int = 0
    chunks_without_section: int = 0
    section_coverage_ratio: float = 0.0
    distinct_sections_count: int = 0
    top_sections: Dict[str, int] = Field(default_factory=dict)


class ModalityDistribution(BaseModel):
    """Content modality and OCR statistics."""

    content_sources: Dict[str, int] = Field(default_factory=dict)
    ocr_applied_count: int = 0
    ocr_ratio: float = 0.0
    mean_ocr_confidence: Optional[float] = None
    mean_quality_score: Optional[float] = None


class ChunkDistributionReport(BaseModel):
    """Complete distribution report for chunk metadata."""

    tenant_id: str
    document_id: Optional[str] = None
    total_chunks: int = 0
    chunk_types: Dict[str, int] = Field(default_factory=dict)
    token_stats: TokenStatistics = Field(default_factory=TokenStatistics)
    page_coverage: PageCoverage = Field(default_factory=PageCoverage)
    section_distribution: SectionDistribution = Field(default_factory=SectionDistribution)
    modality_distribution: ModalityDistribution = Field(default_factory=ModalityDistribution)
    languages: Dict[str, int] = Field(default_factory=dict)


class ChunkDistributionAnalyzer:
    """Computes distribution statistics over document chunks."""

    @classmethod
    def analyze_records(
        cls,
        records: Sequence[Dict[str, Any]],
        tenant_id: str = "default",
        document_id: Optional[str] = None,
    ) -> ChunkDistributionReport:
        """Compute distribution report from raw chunk metadata dictionaries or objects.

        Each record dictionary can contain keys:
            - chunk_type (str)
            - token_count (int) or text (str)
            - section (str)
            - page (int)
            - language (str)
            - content_source (str)
            - ocr_applied (bool)
            - ocr_confidence (float)
            - quality_score (float)
        """
        total_chunks = len(records)
        if total_chunks == 0:
            return ChunkDistributionReport(
                tenant_id=tenant_id,
                document_id=document_id,
                total_chunks=0,
            )

        # 1. Chunk types
        chunk_types: Dict[str, int] = collections.defaultdict(int)
        token_lengths: List[int] = []
        pages: List[int] = []
        page_chunk_counts: Dict[int, int] = collections.defaultdict(int)
        sections: Dict[str, int] = collections.defaultdict(int)
        chunks_with_section = 0
        content_sources: Dict[str, int] = collections.defaultdict(int)
        ocr_count = 0
        ocr_confidences: List[float] = []
        quality_scores: List[float] = []
        languages: Dict[str, int] = collections.defaultdict(int)

        for rec in records:
            # Chunk type
            c_type = rec.get("chunk_type") or (rec.get("metadata", {}).get("chunk_type") if isinstance(rec.get("metadata"), dict) else None) or "text"
            chunk_types[str(c_type)] += 1

            # Token count
            tok_count = rec.get("token_count")
            if tok_count is None and isinstance(rec.get("metadata"), dict):
                tok_count = rec["metadata"].get("token_count")
            if tok_count is None:
                # Estimate from text words
                txt = rec.get("text", "")
                tok_count = len(txt.split()) if txt else 0
            token_lengths.append(int(tok_count))

            # Page
            p = rec.get("page")
            if p is not None:
                try:
                    p_int = int(p)
                    pages.append(p_int)
                    page_chunk_counts[p_int] += 1
                except (ValueError, TypeError):
                    pass

            # Section
            sec = rec.get("section")
            if sec:
                chunks_with_section += 1
                sections[str(sec)] += 1

            # Modality / Source
            src = rec.get("content_source") or (rec.get("metadata", {}).get("content_source") if isinstance(rec.get("metadata"), dict) else None) or "native"
            content_sources[str(src)] += 1

            # OCR
            ocr = rec.get("ocr_applied")
            if ocr is None and isinstance(rec.get("metadata"), dict):
                ocr = rec["metadata"].get("ocr_applied")
            if ocr:
                ocr_count += 1
                conf = rec.get("ocr_confidence")
                if conf is None and isinstance(rec.get("metadata"), dict):
                    conf = rec["metadata"].get("ocr_confidence")
                if conf is not None:
                    try:
                        ocr_confidences.append(float(conf))
                    except (ValueError, TypeError):
                        pass

            # Quality
            q_score = rec.get("quality_score")
            if q_score is None and isinstance(rec.get("metadata"), dict):
                q_score = rec["metadata"].get("quality_score")
            if q_score is not None:
                try:
                    quality_scores.append(float(q_score))
                except (ValueError, TypeError):
                    pass

            # Language
            lang = rec.get("language") or (rec.get("metadata", {}).get("language") if isinstance(rec.get("metadata"), dict) else None) or "eng"
            languages[str(lang)] += 1

        # Token statistics computation
        token_lengths.sort()
        n = len(token_lengths)
        mean_tok = sum(token_lengths) / n
        min_tok = token_lengths[0]
        max_tok = token_lengths[-1]

        def percentile(arr: List[int], p: float) -> float:
            if not arr:
                return 0.0
            k = (len(arr) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return float(arr[int(k)])
            return float(arr[f] * (c - k) + arr[c] * (k - f))

        median_tok = percentile(token_lengths, 0.50)
        p90_tok = percentile(token_lengths, 0.90)
        p95_tok = percentile(token_lengths, 0.95)
        p99_tok = percentile(token_lengths, 0.99)

        variance = sum((x - mean_tok) ** 2 for x in token_lengths) / (n - 1) if n > 1 else 0.0
        stddev_tok = math.sqrt(max(0.0, variance))

        # Histogram buckets
        histogram = {
            "<100": sum(1 for x in token_lengths if x < 100),
            "100-250": sum(1 for x in token_lengths if 100 <= x < 250),
            "250-500": sum(1 for x in token_lengths if 250 <= x < 500),
            "500-1000": sum(1 for x in token_lengths if 500 <= x < 1000),
            ">1000": sum(1 for x in token_lengths if x >= 1000),
        }

        tok_stats = TokenStatistics(
            count=n,
            min=min_tok,
            max=max_tok,
            mean=round(mean_tok, 2),
            median=round(median_tok, 2),
            p90=round(p90_tok, 2),
            p95=round(p95_tok, 2),
            p99=round(p99_tok, 2),
            stddev=round(stddev_tok, 2),
            histogram=histogram,
        )

        # Page coverage
        distinct_pages = set(pages)
        page_cov = PageCoverage(
            total_pages_covered=len(distinct_pages),
            min_page=min(distinct_pages) if distinct_pages else None,
            max_page=max(distinct_pages) if distinct_pages else None,
            mean_chunks_per_page=round(total_chunks / len(distinct_pages), 2) if distinct_pages else 0.0,
            max_chunks_per_page=max(page_chunk_counts.values()) if page_chunk_counts else 0,
        )

        # Section distribution
        sorted_sections = dict(sorted(sections.items(), key=lambda item: item[1], reverse=True)[:10])
        sec_dist = SectionDistribution(
            chunks_with_section=chunks_with_section,
            chunks_without_section=total_chunks - chunks_with_section,
            section_coverage_ratio=round(chunks_with_section / total_chunks, 4),
            distinct_sections_count=len(sections),
            top_sections=sorted_sections,
        )

        # Modality distribution
        mod_dist = ModalityDistribution(
            content_sources=dict(content_sources),
            ocr_applied_count=ocr_count,
            ocr_ratio=round(ocr_count / total_chunks, 4),
            mean_ocr_confidence=round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else None,
            mean_quality_score=round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
        )

        return ChunkDistributionReport(
            tenant_id=tenant_id,
            document_id=document_id,
            total_chunks=total_chunks,
            chunk_types=dict(chunk_types),
            token_stats=tok_stats,
            page_coverage=page_cov,
            section_distribution=sec_dist,
            modality_distribution=mod_dist,
            languages=dict(languages),
        )

    @classmethod
    def analyze_db(
        cls,
        session: Session,
        tenant_id: str,
        document_id: Optional[str] = None,
    ) -> ChunkDistributionReport:
        """Compute distribution report by querying database chunks and chunk_metadata tables directly."""
        stmt = (
            select(
                Chunk.chunk_id,
                Chunk.text,
                Chunk.section,
                Chunk.page,
                Chunk.chunk_metadata,
                ChunkMetadata.chunk_type,
                ChunkMetadata.token_count,
                ChunkMetadata.language,
                ChunkMetadata.content_source,
                ChunkMetadata.ocr_applied,
                ChunkMetadata.ocr_confidence,
                ChunkMetadata.quality_score,
            )
            .join(Document, Chunk.document_id == Document.document_id)
            .outerjoin(ChunkMetadata, Chunk.chunk_id == ChunkMetadata.chunk_id)
            .where(Document.tenant_id == tenant_id)
        )

        if document_id:
            stmt = stmt.where(Chunk.document_id == document_id)

        rows = session.execute(stmt).all()

        records = []
        for row in rows:
            meta = row.chunk_metadata or {}
            rec = {
                "chunk_id": row.chunk_id,
                "text": row.text,
                "section": row.section,
                "page": row.page,
                "chunk_type": row.chunk_type or meta.get("chunk_type", "text"),
                "token_count": row.token_count if row.token_count is not None else meta.get("token_count"),
                "language": row.language or meta.get("language", "eng"),
                "content_source": row.content_source or meta.get("content_source", "native"),
                "ocr_applied": row.ocr_applied if row.ocr_applied is not None else meta.get("ocr_applied", False),
                "ocr_confidence": row.ocr_confidence if row.ocr_confidence is not None else meta.get("ocr_confidence"),
                "quality_score": row.quality_score if row.quality_score is not None else meta.get("quality_score"),
                "metadata": meta,
            }
            records.append(rec)

        return cls.analyze_records(records, tenant_id=tenant_id, document_id=document_id)
