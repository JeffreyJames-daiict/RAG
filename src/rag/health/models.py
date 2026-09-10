"""Data models for system health, dependency status, liveness, and readiness endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health classification for dependencies and overall system state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Detailed health check result for an individual pipeline dependency."""

    name: str = Field(..., description="Dependency component name")
    status: HealthStatus = Field(default=HealthStatus.HEALTHY, description="Component health status")
    latency_ms: float = Field(default=0.0, description="Health check probe duration in milliseconds")
    is_critical: bool = Field(default=True, description="Whether failure of this dependency renders the pipeline unavailable")
    message: Optional[str] = Field(default=None, description="Diagnostic note or error summary")
    details: Dict[str, Any] = Field(default_factory=dict, description="Component-specific diagnostics")


class HealthReport(BaseModel):
    """Aggregated health report containing liveness, readiness, and all dependency probes."""

    status: HealthStatus = Field(..., description="Overall system health status")
    liveness: bool = Field(default=True, description="True if process is alive and responsive")
    readiness: bool = Field(default=True, description="True if critical dependencies are healthy and ready to serve traffic")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")
    uptime_seconds: float = Field(default=0.0, description="Process running duration in seconds")
    dependencies: Dict[str, ComponentHealth] = Field(default_factory=dict, description="Dependency status breakdown")
    version: str = Field(default="1.0.0", description="RAG service software version")
    environment: str = Field(default="production", description="Runtime environment")
