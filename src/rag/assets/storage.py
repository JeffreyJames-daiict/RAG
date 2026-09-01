"""Asset binary storage manager integrating with RawStorageBackend."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from rag.assets.exceptions import AssetStorageError
from rag.core.models import AssetType
from rag.storage.base import RawStorageBackend

logger = logging.getLogger(__name__)


class AssetStorage:
    """Manages immutable persistence of visual asset images in object/raw storage."""

    def __init__(self, storage_backend: Optional[RawStorageBackend] = None):
        self.storage_backend = storage_backend

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        """Compute deterministic SHA-256 hash of image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def format_storage_key(
        document_id: str,
        version: int,
        asset_type: AssetType,
        asset_id: str,
        mime_type: str = "image/png",
    ) -> str:
        """Construct deterministic storage key for an asset (no tenant prefix)."""
        ext = "png"
        if "jpeg" in mime_type or "jpg" in mime_type:
            ext = "jpg"
        elif "webp" in mime_type:
            ext = "webp"
        elif "svg" in mime_type:
            ext = "svg"

        # Organize by asset category folder
        folder_map = {
            AssetType.FIGURE: "figures",
            AssetType.DIAGRAM: "diagrams",
            AssetType.ARCHITECTURE_DIAGRAM: "architecture",
            AssetType.FLOWCHART: "flowcharts",
            AssetType.CHART: "charts",
            AssetType.TABLE_IMAGE: "table_images",
            AssetType.PHOTO: "photos",
            AssetType.ILLUSTRATION: "illustrations",
            AssetType.TECHNICAL_DRAWING: "technical_drawings",
            AssetType.DECORATIVE: "decorative",
            AssetType.UNKNOWN: "misc",
        }
        subfolder = folder_map.get(asset_type, "figures")
        filename = f"{asset_id}.{ext}"

        # No tenant prefix — path: {doc_id}/v{version}/assets/{subfolder}/{filename}
        return f"{document_id}/v{version}/assets/{subfolder}/{filename}"

    def store_asset(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        asset_type: AssetType,
        asset_id: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        force: bool = False,
    ) -> str:
        """Persist asset image bytes into storage and return the storage URI."""
        if not image_bytes:
            raise AssetStorageError("Cannot store empty asset image bytes", asset_id=asset_id)

        storage_key = self.format_storage_key(
            document_id=document_id,
            version=version,
            asset_type=asset_type,
            asset_id=asset_id,
            mime_type=mime_type,
        )

        if self.storage_backend is None:
            # Fallback in-memory / direct URI reference
            return f"in-memory://{storage_key}"

        try:
            # subpath_filename is the part after v{version}/
            # e.g. "assets/figures/{asset_id}.png"
            subpath_filename = storage_key.split(f"v{version}/", 1)[-1]
            return self.storage_backend.write(
                tenant_id=tenant_id,
                document_id=document_id,
                version=version,
                filename=subpath_filename,
                content=image_bytes,
                force=force,
            )
        except FileExistsError:
            logger.debug("Asset %s already exists in storage key '%s'", asset_id, storage_key)
            return storage_key
        except Exception as exc:
            raise AssetStorageError(f"Failed to persist asset image: {exc}", asset_id=asset_id) from exc

    def read_asset(self, storage_uri: str) -> bytes:
        """Retrieve raw image bytes from storage backend."""
        if self.storage_backend is None:
            raise AssetStorageError(f"No storage backend configured to read '{storage_uri}'")
        return self.storage_backend.read(storage_uri)
