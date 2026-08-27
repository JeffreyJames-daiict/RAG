"""Raw storage package — immutable preservation of original document bytes.

Exports:
    RawStorageBackend       — abstract interface for all storage backends
    LocalRawStorageBackend  — local filesystem implementation
"""

from rag.storage.base import RawStorageBackend
from rag.storage.local_storage import LocalRawStorageBackend

__all__ = [
    "RawStorageBackend",
    "LocalRawStorageBackend",
]
