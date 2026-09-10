"""Telemetry and metrics tracking for query routing decisions."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, List, Optional

from rag.routing.models import QueryType, RouterType, RouteTarget, RoutingDecision


class RoutingMetricsCollector:
    """Thread-safe telemetry collector for routing distribution, confidence, and latency."""

    def __init__(self):
        self._lock = threading.RLock()
        self.total_routed = 0
        self.route_counts: Counter[str] = Counter()
        self.query_type_counts: Counter[str] = Counter()
        self.router_type_counts: Counter[str] = Counter()
        self.fallback_count = 0
        self.total_confidence = 0.0
        self.total_latency_ms = 0.0
        self.total_cost_usd = 0.0

    def record(self, decision: RoutingDecision) -> None:
        """Record a single routing decision event."""
        with self._lock:
            self.total_routed += 1
            self.route_counts[decision.route.value] += 1
            self.query_type_counts[decision.query_type.value] += 1
            self.router_type_counts[decision.router_type.value] += 1
            if decision.fallback_used:
                self.fallback_count += 1
            self.total_confidence += decision.confidence
            self.total_latency_ms += decision.latency_ms
            self.total_cost_usd += decision.cost_usd

    @property
    def fallback_rate(self) -> float:
        """Fraction of routing decisions that fell back to the default route."""
        with self._lock:
            if self.total_routed == 0:
                return 0.0
            return self.fallback_count / self.total_routed

    @property
    def avg_confidence(self) -> float:
        """Average classification confidence score."""
        with self._lock:
            if self.total_routed == 0:
                return 0.0
            return self.total_confidence / self.total_routed

    @property
    def avg_latency_ms(self) -> float:
        """Average routing computation time in ms."""
        with self._lock:
            if self.total_routed == 0:
                return 0.0
            return self.total_latency_ms / self.total_routed

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_routed = 0
            self.route_counts.clear()
            self.query_type_counts.clear()
            self.router_type_counts.clear()
            self.fallback_count = 0
            self.total_confidence = 0.0
            self.total_latency_ms = 0.0
            self.total_cost_usd = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Export snapshot of routing performance telemetry."""
        with self._lock:
            return {
                "total_routed": self.total_routed,
                "fallback_count": self.fallback_count,
                "fallback_rate": round(self.fallback_rate, 4),
                "avg_confidence": round(self.avg_confidence, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "routes": dict(self.route_counts),
                "query_types": dict(self.query_type_counts),
                "router_types": dict(self.router_type_counts),
            }
