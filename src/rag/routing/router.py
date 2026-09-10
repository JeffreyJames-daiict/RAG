"""Query router orchestrator coordinating rule-based, LLM, and hybrid routing."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from rag.routing.llm_classifier import LLMQueryClassifier, LLMRouterCaller
from rag.routing.metrics import RoutingMetricsCollector
from rag.routing.models import (
    QueryType,
    RouterType,
    RouteTarget,
    RoutingDecision,
)
from rag.routing.rule_classifier import RuleBasedQueryClassifier

logger = logging.getLogger(__name__)


class QueryRouter:
    """Production-grade query router selecting retrieval and answering strategies.

    Features:
    - Deterministic rule-based classification by default (0 ms overhead)
    - Optional LLM-based classification with cost limits
    - Hybrid mode: fast rules with LLM disambiguation on low confidence
    - Confidence threshold gating against default fallback route
    - Tenant-safe metadata and telemetry recording
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        router_type: RouterType = RouterType.RULE_BASED,
        confidence_threshold: float = 0.6,
        default_route: RouteTarget = RouteTarget.HYBRID_RRF,
        max_cost: float = 0.01,
        llm_caller: Optional[LLMRouterCaller] = None,
        rule_classifier: Optional[RuleBasedQueryClassifier] = None,
        metrics_collector: Optional[RoutingMetricsCollector] = None,
    ):
        self._enabled = enabled
        self._router_type = router_type
        self._confidence_threshold = confidence_threshold
        self._default_route = default_route
        self._max_cost = max_cost

        self._rule_classifier = rule_classifier or RuleBasedQueryClassifier()
        self._llm_classifier = (
            LLMQueryClassifier(
                llm_caller=llm_caller,
                max_cost=max_cost,
                fallback_classifier=self._rule_classifier,
            )
            if llm_caller is not None or router_type in (RouterType.LLM, RouterType.HYBRID)
            else None
        )
        self.metrics = metrics_collector or RoutingMetricsCollector()

    def route(
        self,
        query: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a user query to its target retrieval and answering strategy.

        Args:
            query: The user query string.
            tenant_id: Tenant identifier for tenant-safe logging and metadata.

        Returns:
            RoutingDecision containing selected RouteTarget, QueryType, and metadata.
        """
        start_time = time.perf_counter()
        clean_query = query.strip()

        # ── 1. Bypass if disabled ─────────────────────────────────────────────
        if not self._enabled:
            latency = (time.perf_counter() - start_time) * 1000
            decision = RoutingDecision(
                query=clean_query,
                route=self._default_route,
                query_type=QueryType.GENERAL_SEMANTIC,
                confidence=1.0,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                latency_ms=round(latency, 2),
                metadata={
                    "reason": "Router disabled; routed directly to default route",
                    "tenant_safe_id": f"tenant_{tenant_id}" if tenant_id else "global",
                },
            )
            self.metrics.record(decision)
            return decision

        # ── 2. Execute selected router ────────────────────────────────────────
        decision: RoutingDecision
        try:
            if self._router_type == RouterType.LLM and self._llm_classifier:
                decision = self._llm_classifier.classify(clean_query)
            elif self._router_type == RouterType.HYBRID and self._llm_classifier:
                rule_decision = self._rule_classifier.classify(clean_query)
                # If rule confidence is high enough, accept it
                if rule_decision.confidence >= self._confidence_threshold:
                    decision = rule_decision
                else:
                    # Ambiguous query: consult LLM classifier
                    decision = self._llm_classifier.classify(clean_query)
                    decision.router_type = RouterType.HYBRID
            else:
                # Default: purely rule-based
                decision = self._rule_classifier.classify(clean_query)

        except Exception as exc:
            logger.error("Router classification raised error: %s. Using default fallback.", exc)
            latency = (time.perf_counter() - start_time) * 1000
            decision = RoutingDecision(
                query=clean_query,
                route=self._default_route,
                query_type=QueryType.GENERAL_SEMANTIC,
                confidence=0.0,
                fallback_used=True,
                router_type=self._router_type,
                latency_ms=round(latency, 2),
                metadata={"error": str(exc), "reason": "Classification threw unhandled exception"},
            )

        # ── 3. Confidence threshold enforcement & Fallback ────────────────────
        if decision.confidence < self._confidence_threshold and not decision.fallback_used:
            logger.info(
                "Routing confidence %.2f below threshold %.2f. Falling back to %s.",
                decision.confidence,
                self._confidence_threshold,
                self._default_route.value,
            )
            decision.route = self._default_route
            decision.fallback_used = True
            decision.metadata["original_query_type"] = decision.query_type.value
            decision.metadata["fallback_reason"] = (
                f"Confidence {decision.confidence:.2f} below threshold {self._confidence_threshold:.2f}"
            )

        # ── 4. Tenant safety sanitization ─────────────────────────────────────
        if tenant_id:
            decision.metadata["tenant_safe_id"] = f"tenant_{tenant_id}"

        # ── 5. Telemetry recording ────────────────────────────────────────────
        self.metrics.record(decision)
        return decision
