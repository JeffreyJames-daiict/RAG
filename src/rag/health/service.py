"""Health and readiness service monitoring system dependencies across the RAG pipeline."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from rag.health.models import ComponentHealth, HealthReport, HealthStatus

logger = logging.getLogger(__name__)

_PROCESS_START_TIME = time.time()


class HealthCheckService:
    """Probes and aggregates health across database, vector store, search indices, and model providers."""

    def __init__(self, environment: str = "production", version: str = "1.0.0"):
        self.environment = environment
        self.version = version
        self._probes: Dict[str, Tuple[Callable[[], Any], bool]] = {}  # name -> (callable, is_critical)

        # Register default simulated/pluggable probes
        self._register_default_probes()

    def _register_default_probes(self) -> None:
        """Initialize standard RAG dependency probes."""
        self.register_probe("database", self._probe_database, is_critical=True)
        self.register_probe("vector_index", self._probe_vector_index, is_critical=True)
        self.register_probe("sparse_index", self._probe_sparse_index, is_critical=False)
        self.register_probe("embedding_provider", self._probe_embedding_provider, is_critical=True)
        self.register_probe("reranker_provider", self._probe_reranker_provider, is_critical=False)
        self.register_probe("llm_provider", self._probe_llm_provider, is_critical=True)
        self.register_probe("storage", self._probe_storage, is_critical=False)

    def register_probe(
        self,
        name: str,
        probe_fn: Callable[[], Any],
        is_critical: bool = True,
    ) -> None:
        """Register a health check probe for a specific dependency."""
        self._probes[name] = (probe_fn, is_critical)

    # -----------------------------------------------------------------------
    # Default Probe Implementations (can be overridden with real connections)
    # -----------------------------------------------------------------------
    def _probe_database(self) -> Dict[str, Any]:
        return {"connection": "ok", "pool_size": 10}

    def _probe_vector_index(self) -> Dict[str, Any]:
        return {"index_type": "pgvector_hnsw", "status": "ready"}

    def _probe_sparse_index(self) -> Dict[str, Any]:
        return {"engine": "bm25", "status": "ready"}

    def _probe_embedding_provider(self) -> Dict[str, Any]:
        return {"provider": "active", "status": "ready"}

    def _probe_reranker_provider(self) -> Dict[str, Any]:
        return {"provider": "active", "status": "ready"}

    def _probe_llm_provider(self) -> Dict[str, Any]:
        return {"provider": "openrouter", "status": "ready"}

    def _probe_storage(self) -> Dict[str, Any]:
        return {"storage": "local_filesystem", "status": "accessible"}

    def check_liveness(self) -> bool:
        """Liveness probe: verifies the service process is alive and responsive."""
        return True

    def check_readiness(self) -> bool:
        """Readiness probe: verifies all critical dependencies are healthy."""
        report = self.check_health()
        return report.readiness

    def check_health(self) -> HealthReport:
        """Execute all registered dependency probes and compile comprehensive health report."""
        dependencies: Dict[str, ComponentHealth] = {}
        has_critical_failure = False
        has_non_critical_failure = False

        for name, (probe_fn, is_critical) in self._probes.items():
            start = time.perf_counter()
            status = HealthStatus.HEALTHY
            message: Optional[str] = None
            details: Dict[str, Any] = {}

            try:
                if inspect.iscoroutinefunction(probe_fn):
                    # Run sync in existing loop or fallback
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            res = "<async probe>"
                        else:
                            res = loop.run_until_complete(probe_fn())
                    except Exception:
                        res = asyncio.run(probe_fn())
                else:
                    res = probe_fn()

                if isinstance(res, dict):
                    details = res
                latency_ms = (time.perf_counter() - start) * 1000.0

            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                status = HealthStatus.UNHEALTHY
                message = str(exc)
                details = {"error": type(exc).__name__}
                if is_critical:
                    has_critical_failure = True
                else:
                    has_non_critical_failure = True

            dependencies[name] = ComponentHealth(
                name=name,
                status=status,
                latency_ms=round(latency_ms, 2),
                is_critical=is_critical,
                message=message,
                details=details,
            )

        if has_critical_failure:
            overall_status = HealthStatus.UNHEALTHY
            readiness = False
        elif has_non_critical_failure:
            overall_status = HealthStatus.DEGRADED
            readiness = True  # Can still serve traffic even if optional reranker/storage is degraded
        else:
            overall_status = HealthStatus.HEALTHY
            readiness = True

        uptime = time.time() - _PROCESS_START_TIME

        return HealthReport(
            status=overall_status,
            liveness=self.check_liveness(),
            readiness=readiness,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=round(uptime, 1),
            dependencies=dependencies,
            version=self.version,
            environment=self.environment,
        )


_GLOBAL_HEALTH_SERVICE: HealthCheckService = HealthCheckService()


def get_health_service() -> HealthCheckService:
    """Get the global health service instance."""
    return _GLOBAL_HEALTH_SERVICE
