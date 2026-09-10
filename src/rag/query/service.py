"""Query rewrite orchestration service.

Coordinates query analysis, strategy selection, rewrite execution, validation,
deduplication, and fallback handling across the query rewriting pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from rag.query.analyzer import QueryAnalyzer
from rag.query.models import (
    ConversationTurn,
    QueryAnalysis,
    QueryRewriteResult,
    QueryType,
    RewriteStrategyType,
    RewrittenQuery,
)
from rag.query.strategies import (
    RewriteLLMCaller,
    MockRewriteLLMCaller,
    get_strategy,
)

logger = logging.getLogger(__name__)


class QueryRewriteService:
    """Production query rewriting orchestrator.

    Pipeline:
    1. Analyze query (deterministic: type, identifiers, keywords)
    2. Select strategies based on analysis + config
    3. Execute the primary strategy
    4. Validate outputs (non-empty, identifiers preserved)
    5. Deduplicate generated queries
    6. Cap at max_queries
    7. Prepend original query
    8. Return QueryRewriteResult

    Falls back to original query on any failure.
    """

    def __init__(
        self,
        llm_caller: Optional[RewriteLLMCaller] = None,
        analyzer: Optional[QueryAnalyzer] = None,
        *,
        enabled: bool = True,
        max_queries: int = 5,
        confidence_threshold: float = 0.3,
        fallback_enabled: bool = True,
        enable_expansion: bool = True,
        enable_multi_query: bool = True,
        enable_hyde: bool = False,
        enable_decomposition: bool = True,
        enable_conversation: bool = True,
        timeout_sec: float = 5.0,
    ):
        self._llm_caller = llm_caller
        self._analyzer = analyzer or QueryAnalyzer()
        self._enabled = enabled
        self._max_queries = max(1, min(max_queries, 10))
        self._confidence_threshold = confidence_threshold
        self._fallback_enabled = fallback_enabled
        self._timeout_sec = timeout_sec

        # Feature flags
        self._enable_expansion = enable_expansion
        self._enable_multi_query = enable_multi_query
        self._enable_hyde = enable_hyde
        self._enable_decomposition = enable_decomposition
        self._enable_conversation = enable_conversation

    def rewrite(
        self,
        query: str,
        *,
        conversation_history: Optional[List[ConversationTurn]] = None,
        tenant_id: Optional[str] = None,
    ) -> QueryRewriteResult:
        """Rewrite a query through the full pipeline.

        Args:
            query: The user's original query.
            conversation_history: Optional prior conversation turns.
            tenant_id: Tenant identifier for observability.

        Returns:
            QueryRewriteResult containing original + rewritten query variants.
        """
        start_time = time.perf_counter()
        original = query.strip()
        warnings: List[str] = []

        # ── Bypass if disabled ────────────────────────────────────────────
        if not self._enabled:
            return self._original_only_result(original, latency_ms=0.0)

        # ── 1. Analyze ────────────────────────────────────────────────────
        try:
            analysis = self._analyzer.analyze(
                original,
                conversation_history=conversation_history,
                enable_expansion=self._enable_expansion,
                enable_multi_query=self._enable_multi_query,
                enable_hyde=self._enable_hyde,
                enable_decomposition=self._enable_decomposition,
                enable_conversation=self._enable_conversation,
            )
        except Exception as exc:
            logger.warning("Query analysis failed: %s. Falling back to original.", exc)
            latency = (time.perf_counter() - start_time) * 1000
            return self._original_only_result(original, latency_ms=latency, fallback=True,
                                              warnings=[f"Analysis failed: {exc}"])

        # ── 2. Select strategies ──────────────────────────────────────────
        strategies = analysis.suggested_strategies
        if not strategies:
            strategies = [RewriteStrategyType.ORIGINAL_ONLY]

        # ── 3. Execute strategies ─────────────────────────────────────────
        all_rewrites: List[RewrittenQuery] = []
        total_tokens = 0
        total_cost = 0.0
        primary_strategy = strategies[0]
        fallback_used = False

        for strat_type in strategies:
            try:
                strategy = get_strategy(
                    strat_type,
                    llm_caller=self._llm_caller,
                    max_queries=self._max_queries,
                )
                rewrites = strategy.rewrite(
                    original,
                    analysis,
                    context=conversation_history,
                )

                for rw in rewrites:
                    # Accumulate token usage
                    if "token_usage" in rw.metadata:
                        total_tokens += rw.metadata["token_usage"]
                    if "cost_usd" in rw.metadata:
                        total_cost += rw.metadata["cost_usd"]

                    # Filter by confidence threshold
                    if rw.confidence < self._confidence_threshold:
                        warnings.append(
                            f"Dropped low-confidence rewrite (conf={rw.confidence:.2f}, "
                            f"threshold={self._confidence_threshold:.2f}): {rw.text[:50]}..."
                        )
                        continue

                    all_rewrites.append(rw)

            except Exception as exc:
                logger.warning("Strategy '%s' failed: %s", strat_type.value, exc)
                warnings.append(f"Strategy '{strat_type.value}' failed: {exc}")
                if self._fallback_enabled:
                    fallback_used = True
                continue

            # Stop once the primary strategy successfully produced rewrites
            if all_rewrites:
                primary_strategy = strat_type
                break

        # ── 4. Validate & deduplicate ─────────────────────────────────────
        validated = self._validate_rewrites(all_rewrites, analysis, warnings)
        deduped = self._deduplicate(validated, original)

        # ── 5. Cap at max_queries ─────────────────────────────────────────
        capped = deduped[: self._max_queries - 1]  # Reserve slot for original

        # ── 6. Prepend original ───────────────────────────────────────────
        original_rw = RewrittenQuery(
            text=original,
            strategy=RewriteStrategyType.ORIGINAL_ONLY,
            confidence=1.0,
            is_original=True,
            identifiers_preserved=True,
        )
        final_queries = [original_rw] + capped

        # ── 7. Build result ───────────────────────────────────────────────
        latency = (time.perf_counter() - start_time) * 1000

        is_fallback = (not capped and primary_strategy != RewriteStrategyType.ORIGINAL_ONLY) or (fallback_used and not capped)
        if is_fallback:
            warnings.append("All strategies failed or produced no valid rewrites. Using original query only.")

        return QueryRewriteResult(
            original_query=original,
            rewritten_queries=final_queries,
            selected_strategy=primary_strategy,
            analysis=analysis,
            fallback_used=is_fallback,
            latency_ms=round(latency, 2),
            token_usage=total_tokens,
            cost_usd=total_cost,
            warnings=warnings,
        )

    def _validate_rewrites(
        self,
        rewrites: List[RewrittenQuery],
        analysis: QueryAnalysis,
        warnings: List[str],
    ) -> List[RewrittenQuery]:
        """Validate rewritten queries: non-empty, identifiers preserved."""
        valid: List[RewrittenQuery] = []
        for rw in rewrites:
            # Skip empty
            if not rw.text or not rw.text.strip():
                warnings.append(f"Dropped empty rewrite from strategy '{rw.strategy.value}'.")
                continue

            # Skip if too short (likely garbage)
            if len(rw.text.strip()) < 3:
                warnings.append(f"Dropped very short rewrite: '{rw.text[:20]}'")
                continue

            # Check identifier preservation and reinject if needed
            if analysis.detected_identifiers:
                missing = [ident for ident in analysis.detected_identifiers if ident.lower() not in rw.text.lower()]
                if missing:
                    rw.text = f"{rw.text} {' '.join(missing)}"
                    rw.identifiers_preserved = True
                    warnings.append(
                        f"Rewrite from '{rw.strategy.value}' had missing identifiers. "
                        f"Identifiers were reinjected."
                    )
                else:
                    rw.identifiers_preserved = True

            valid.append(rw)
        return valid

    def _deduplicate(
        self,
        rewrites: List[RewrittenQuery],
        original: str,
    ) -> List[RewrittenQuery]:
        """Remove duplicate query texts, preserving order."""
        seen = {original.lower().strip()}
        unique: List[RewrittenQuery] = []
        for rw in rewrites:
            normalized = rw.text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(rw)
        return unique

    def _original_only_result(
        self,
        query: str,
        latency_ms: float = 0.0,
        fallback: bool = False,
        warnings: Optional[List[str]] = None,
    ) -> QueryRewriteResult:
        """Build a result containing only the original query."""
        return QueryRewriteResult(
            original_query=query,
            rewritten_queries=[
                RewrittenQuery(
                    text=query,
                    strategy=RewriteStrategyType.ORIGINAL_ONLY,
                    confidence=1.0,
                    is_original=True,
                    identifiers_preserved=True,
                )
            ],
            selected_strategy=RewriteStrategyType.ORIGINAL_ONLY,
            fallback_used=fallback,
            latency_ms=round(latency_ms, 2),
            warnings=warnings or [],
        )
