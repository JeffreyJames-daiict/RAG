"""Multi-tenant cache service with TTL expiration, LRU eviction, and targeted invalidation."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from rag.caching.models import CacheDomain, CacheKey

logger = logging.getLogger(__name__)


class CacheEntry:
    """Internal container for cached payload and expiration timestamp."""

    def __init__(self, key: str, value: Any, ttl_sec: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.perf_counter()
        self.expires_at = (self.created_at + ttl_sec) if ttl_sec and ttl_sec > 0 else None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.perf_counter() >= self.expires_at


class MultiTenantCacheService:
    """Thread-safe, tenant-isolated in-memory cache supporting TTL, LRU pruning, and selective invalidation."""

    def __init__(
        self,
        max_entries: int = 5000,
        default_ttl_sec: float = 3600.0,
    ):
        self.max_entries = max(10, max_entries)
        self.default_ttl_sec = default_ttl_sec

        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Telemetry
        self._hits = 0
        self._misses = 0

    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total > 0 else 0.0

    def get(self, key: CacheKey) -> Optional[Any]:
        """Fetch cached item for a specific tenant and artifact key."""
        key_str = key.to_key_string()

        with self._lock:
            entry = self._store.get(key_str)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._store[key_str]
                self._misses += 1
                return None

            # Move to end for LRU
            self._store.move_to_end(key_str)
            self._hits += 1
            return entry.value

    def set(
        self,
        key: CacheKey,
        value: Any,
        ttl_sec: Optional[float] = None,
    ) -> None:
        """Store item in cache strictly partitioned under key's tenant."""
        key_str = key.to_key_string()
        ttl = ttl_sec if ttl_sec is not None else self.default_ttl_sec

        with self._lock:
            if key_str in self._store:
                self._store.move_to_end(key_str)

            self._store[key_str] = CacheEntry(key_str, value, ttl_sec=ttl)

            # Enforce LRU eviction
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def has(self, key: CacheKey) -> bool:
        """Check if active, non-expired entry exists."""
        return self.get(key) is not None

    def invalidate_tenant(self, tenant_id: str) -> int:
        """Purge all cached artifacts belonging to a given tenant."""
        prefix = f"t={tenant_id}:"
        with self._lock:
            keys_to_remove = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            logger.info("Invalidated %d cache entries for tenant '%s'.", len(keys_to_remove), tenant_id)
            return len(keys_to_remove)

    def invalidate_document_version(self, tenant_id: str, document_index_version: str) -> int:
        """Purge cached retrieval/reranking results for a tenant when documents are reindexed."""
        prefix = f"t={tenant_id}:"
        token = f":div={document_index_version}"
        with self._lock:
            keys_to_remove = [k for k in self._store.keys() if k.startswith(prefix) and token in k]
            for k in keys_to_remove:
                del self._store[k]
            logger.info(
                "Invalidated %d cache entries for tenant '%s' and document version '%s'.",
                len(keys_to_remove),
                tenant_id,
                document_index_version,
            )
            return len(keys_to_remove)

    def invalidate_domain(self, tenant_id: str, domain: CacheDomain) -> int:
        """Purge artifacts within a specific domain for a tenant."""
        target = f"t={tenant_id}:d={domain.value}:"
        with self._lock:
            keys_to_remove = [k for k in self._store.keys() if k.startswith(target)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all entries across all tenants."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Diagnostic summary of cache usage."""
        with self._lock:
            return {
                "total_entries": len(self._store),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self.hit_ratio,
            }


_GLOBAL_CACHE: MultiTenantCacheService = MultiTenantCacheService()


def get_cache_service() -> MultiTenantCacheService:
    """Get the global multi-tenant cache instance."""
    return _GLOBAL_CACHE
