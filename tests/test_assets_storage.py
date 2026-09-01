"""Unit tests for asset binary storage management."""

from pathlib import Path
from PIL import Image
import io
import pytest

from rag.assets.storage import AssetStorage
from rag.core.models import AssetType
from rag.storage.local_storage import LocalRawStorageBackend


def test_asset_storage_key_formatting():
    key = AssetStorage.format_storage_key(
        document_id="doc_123",
        version=2,
        asset_type=AssetType.ARCHITECTURE_DIAGRAM,
        asset_id="asset_456",
        mime_type="image/png",
    )
    assert key == "doc_123/v2/assets/architecture/asset_456.png"


def test_asset_storage_write_and_read(tmp_path: Path):
    backend = LocalRawStorageBackend(base_dir=tmp_path / "raw_storage")
    storage = AssetStorage(storage_backend=backend)

    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    key = storage.store_asset(
        tenant_id="default",
        document_id="doc_789",
        version=1,
        asset_type=AssetType.FIGURE,
        asset_id="fig_999",
        image_bytes=data,
        mime_type="image/png",
    )

    assert "assets/figures/fig_999.png" in key
    read_data = storage.read_asset(key)
    assert read_data == data
    assert storage.compute_hash(read_data) == storage.compute_hash(data)
