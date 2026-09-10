"""Reciprocal Rank Fusion (RRF) for multi-source retrieval ranking.

Implements production-grade weighted RRF fusion (Cormack et al., 2009):
    RRF_score(d) = Σ weight_i / (k + rank_i(d))
over an arbitrary number of named ranked candidate lists (e.g., dense vector,
sparse BM25, table retriever, metadata retriever, multimodal retriever).

Features:
- Generic ``FusionStrategy`` protocol — swap RRF for any future fusion algorithm.
- Per-source configurable weights with default fallback to 1.0.
- Lossless provenance tracking: source ranks, raw scores, retrieval methods.
- 5-tier deterministic tie-breaking:
    1. RRF score DESC
    2. Best individual rank ASC
    3. Dense raw score DESC (if present)
    4. Sparse raw score DESC (if present)
    5. Chunk ID ASC
- ``FusionDiagnostics`` dataclass for full observability.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from rag.retrieval.interfaces import RetrievedCandidate

logger = logging.getLogger(__name__)


# ── Diagnostics ────────────────────────────────────────────────────────────────

@dataclass
class FusionDiagnostics:
    """Observable metadata emitted by a fusion pass for telemetry and debugging."""

    fusion_strategy: str = "rrf"
    rrf_k: int = 60
    source_weights: Dict[str, float] = field(default_factory=dict)
    input_result_counts: Dict[str, int] = field(default_factory=dict)
    unique_result_count: int = 0
    final_result_count: int = 0


# ── Fused Result Container ─────────────────────────────────────────────────────

@dataclass
class FusedCandidateInfo:
    """Rich container holding a fused candidate with per-source provenance."""

    candidate: RetrievedCandidate
    rrf_score: float
    source_ranks: Dict[str, int] = field(default_factory=dict)
    source_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_methods: List[str] = field(default_factory=list)


# ── Generic Fusion Protocol ────────────────────────────────────────────────────

@runtime_checkable
class FusionStrategy(Protocol):
    """Generic interface for ranked-list fusion strategies.

    Any fusion algorithm (RRF, CombSUM, CombMNZ, learned models) can implement
    this protocol to be used interchangeably by the hybrid retrieval service.
    """

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[RetrievedCandidate]],
        *,
        top_k: int,
    ) -> List[FusedCandidateInfo]:
        """Combine multiple named ranked lists into a single fused ranking.

        Args:
            ranked_lists: Mapping of source name (e.g. 'vector', 'bm25',
                'table_retriever') to a rank-ordered sequence of candidates
                where index 0 is rank 1.
            top_k: Maximum number of fused results to return.

        Returns:
            List of FusedCandidateInfo sorted by fused score descending.
        """
        ...

    def diagnostics(self) -> FusionDiagnostics:
        """Return the diagnostics from the most recent fusion pass."""
        ...


# ── Weighted RRF Implementation ────────────────────────────────────────────────

class RRFFusion:
    """Reciprocal Rank Fusion combiner with per-source weights and full diagnostics.

    Formula:
        RRF_score(d) = Σ weight_i / (k + rank_i(d))

    When ``weights`` is not provided or a source is missing from the map,
    the default weight of 1.0 is applied.
    """

    def __init__(
        self,
        k: int = 60,
        weights: Optional[Dict[str, float]] = None,
        default_weight: float = 1.0,
    ):
        """Initialize RRF with smoothing constant k and per-source weights.

        Args:
            k: Smoothing constant (must be >= 1). Higher k reduces the impact of
                high-rank positions. Standard value is 60 (Cormack et al., 2009).
            weights: Optional mapping of source name to non-negative weight.
                Sources not listed receive ``default_weight``.
            default_weight: Fallback weight for sources not in ``weights``.

        Raises:
            ValueError: If k < 1 or any weight is negative.
        """
        if k < 1:
            raise ValueError(f"RRF smoothing constant k must be >= 1, got {k}")
        if default_weight < 0:
            raise ValueError(f"default_weight must be >= 0, got {default_weight}")

        self.k = k
        self._weights: Dict[str, float] = dict(weights) if weights else {}
        self._default_weight = default_weight
        self._last_diagnostics: Optional[FusionDiagnostics] = None

        # Validate provided weights
        for name, w in self._weights.items():
            if w < 0:
                raise ValueError(f"RRF weight for source '{name}' must be >= 0, got {w}")

    # ── Public API ─────────────────────────────────────────────────────────

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[RetrievedCandidate]],
        *,
        top_k: int = 0,
    ) -> List[FusedCandidateInfo]:
        """Combine multiple named ranked lists using weighted RRF.

        This is the primary entry point that satisfies the ``FusionStrategy``
        protocol. Backward-compatible with the previous positional list API:
        callers using ``List[List[RetrievedCandidate]]`` should migrate to
        ``fuse_sources()`` or pass a dict.

        Args:
            ranked_lists: Mapping of source name to ranked candidate list.
            top_k: Maximum number of fused results (0 or None = unlimited).

        Returns:
            List of FusedCandidateInfo sorted by weighted RRF score descending.
        """
        return self.fuse_sources(
            dict(ranked_lists),
            top_k=top_k if top_k and top_k > 0 else None,
        )

    def fuse_positional(
        self,
        ranked_lists: List[List[RetrievedCandidate]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[RetrievedCandidate, float]]:
        """Legacy backward-compatible API accepting positional lists.

        Auto-names each list as ``source_0``, ``source_1``, etc.

        Returns:
            List of tuples (candidate, rrf_score) sorted by rrf_score descending.
        """
        if not ranked_lists:
            self._last_diagnostics = FusionDiagnostics(
                fusion_strategy="rrf",
                rrf_k=self.k,
                source_weights={},
                input_result_counts={},
                unique_result_count=0,
                final_result_count=0,
            )
            return []

        named_lists = {f"source_{i}": lst for i, lst in enumerate(ranked_lists)}
        fused_infos = self.fuse_sources(named_lists, top_k=top_k)
        return [(info.candidate, info.rrf_score) for info in fused_infos]

    def fuse_sources(
        self,
        named_lists: Dict[str, List[RetrievedCandidate]],
        top_k: Optional[int] = None,
    ) -> List[FusedCandidateInfo]:
        """Combine multiple named candidate lists (e.g. 'vector' and 'bm25').

        Tracks individual ranks, raw scores, merges candidate metadata,
        applies per-source weights, and produces full diagnostics.

        Args:
            named_lists: Mapping of source name (e.g. 'vector', 'bm25',
                'table_retriever') to ranked list.
            top_k: Optional limit on the number of returned fused candidates.

        Returns:
            List of FusedCandidateInfo instances sorted by weighted RRF score descending.
        """
        input_counts: Dict[str, int] = {}

        if not named_lists:
            self._last_diagnostics = FusionDiagnostics(
                fusion_strategy="rrf",
                rrf_k=self.k,
                source_weights=dict(self._weights),
                input_result_counts=input_counts,
                unique_result_count=0,
                final_result_count=0,
            )
            return []

        # Build effective weight map for this call
        effective_weights: Dict[str, float] = {}
        for source_name in named_lists:
            effective_weights[source_name] = self._weights.get(source_name, self._default_weight)

        candidate_map: Dict[str, RetrievedCandidate] = {}
        rrf_scores: Dict[str, float] = collections.defaultdict(float)
        source_ranks: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
        source_scores: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        source_methods: Dict[str, List[str]] = collections.defaultdict(list)

        for source_name, ranked_list in named_lists.items():
            input_counts[source_name] = len(ranked_list)
            weight = effective_weights[source_name]

            for rank_0, candidate in enumerate(ranked_list):
                rank_1 = rank_0 + 1
                chunk_id = candidate.chunk_id

                if chunk_id not in candidate_map:
                    candidate_map[chunk_id] = candidate
                else:
                    # Merge metadata without overwriting existing non-empty values
                    existing_meta = candidate_map[chunk_id].chunk_metadata
                    for k, v in candidate.chunk_metadata.items():
                        if k not in existing_meta and v is not None:
                            existing_meta[k] = v

                # Weighted RRF score: weight / (k + rank)
                rrf_scores[chunk_id] += weight / (self.k + rank_1)
                source_ranks[chunk_id][source_name] = rank_1
                source_scores[chunk_id][source_name] = candidate.similarity_score
                if source_name not in source_methods[chunk_id]:
                    source_methods[chunk_id].append(source_name)

        # 5-tier deterministic tie-breaking:
        # 1. RRF score DESC
        # 2. Best individual rank ASC (min rank across sources)
        # 3. Dense raw score DESC (if present, else 0)
        # 4. Sparse raw score DESC (if present, else 0)
        # 5. Chunk ID ASC (lexicographic)
        def _sort_key(item: Tuple[str, float]) -> Tuple[float, int, float, float, str]:
            cid, score = item
            ranks = source_ranks[cid].values()
            min_rank = min(ranks) if ranks else 999999
            scores = source_scores[cid]
            dense_score = scores.get("vector", 0.0)
            sparse_score = scores.get("bm25", 0.0)
            return (-score, min_rank, -dense_score, -sparse_score, cid)

        sorted_items = sorted(rrf_scores.items(), key=_sort_key)

        if top_k is not None and top_k > 0:
            sorted_items = sorted_items[:top_k]

        fused_infos: List[FusedCandidateInfo] = []
        for chunk_id, score in sorted_items:
            fused_infos.append(
                FusedCandidateInfo(
                    candidate=candidate_map[chunk_id],
                    rrf_score=score,
                    source_ranks=source_ranks[chunk_id],
                    source_scores=source_scores[chunk_id],
                    retrieval_methods=source_methods[chunk_id],
                )
            )

        # Emit diagnostics
        self._last_diagnostics = FusionDiagnostics(
            fusion_strategy="rrf",
            rrf_k=self.k,
            source_weights=effective_weights,
            input_result_counts=input_counts,
            unique_result_count=len(candidate_map),
            final_result_count=len(fused_infos),
        )

        logger.debug(
            "RRF fusion: k=%d, sources=%s, weights=%s, unique=%d, returned=%d",
            self.k,
            list(named_lists.keys()),
            effective_weights,
            len(candidate_map),
            len(fused_infos),
        )

        return fused_infos

    def diagnostics(self) -> FusionDiagnostics:
        """Return the diagnostics from the most recent fusion pass.

        Returns:
            FusionDiagnostics instance, or a default empty instance if no
            fusion has been performed yet.
        """
        if self._last_diagnostics is None:
            return FusionDiagnostics(
                fusion_strategy="rrf",
                rrf_k=self.k,
                source_weights=dict(self._weights),
            )
        return self._last_diagnostics
