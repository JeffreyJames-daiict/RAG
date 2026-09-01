"""Formats visual asset analysis and OCR text into searchable semantic chunks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from rag.assets.models import ProcessedAsset
from rag.core.models import ChunkSchema

logger = logging.getLogger(__name__)


class AssetChunker:
    """Transforms processed visual assets into searchable semantic chunks for vector embedding."""

    def __init__(self, chunking_version: str = "asset-v1.0"):
        self.chunking_version = chunking_version

    def build_semantic_text(self, processed_asset: ProcessedAsset) -> str:
        """Construct structured, high-density textual representation of figure/diagram."""
        asset = processed_asset.schema
        title = asset.caption or f"{asset.asset_type.value.replace('_', ' ').title()}"
        lines = [
            f"Figure: {title}",
            f"Asset Type: {asset.asset_type.value} (Page {asset.page_number})",
        ]

        if asset.caption:
            lines.append(f"Caption: {asset.caption}")

        if asset.semantic_description:
            lines.append(f"\nDescription:\n{asset.semantic_description}")

        if asset.ocr_text and asset.ocr_text.strip():
            lines.append(f"\nText Detected Inside Figure:\n{asset.ocr_text.strip()}")

        if asset.entities:
            lines.append(f"\nComponents / Entities:\n" + ", ".join(asset.entities))

        if asset.relationships:
            rel_lines = []
            for r in asset.relationships:
                src = r.get("source", "")
                rel = r.get("relationship", "").replace("_", " ")
                tgt = r.get("target", "")
                rel_lines.append(f"- {src} {rel} {tgt}")
            lines.append(f"\nComponent Relationships:\n" + "\n".join(rel_lines))

        return "\n".join(lines).strip()

    def create_chunk(
        self,
        processed_asset: ProcessedAsset,
        chunk_index: int,
    ) -> ChunkSchema:
        """Build a ChunkSchema instance populated with asset metadata."""
        asset = processed_asset.schema
        text_rep = self.build_semantic_text(processed_asset)
        processed_asset.semantic_text_representation = text_rep

        meta: Dict[str, Any] = {
            "chunk_type": "figure",
            "asset_id": str(asset.asset_id),
            "asset_type": asset.asset_type.value,
            "asset_subtype": asset.asset_subtype,
            "storage_uri": asset.storage_uri,
            "content_hash": asset.content_hash,
            "bbox": asset.bbox,
            "width": asset.width,
            "height": asset.height,
            "caption": asset.caption,
            "has_ocr": bool(asset.ocr_text),
            "has_vision": bool(asset.semantic_description),
            "entities": asset.entities,
            "processing_status": asset.processing_status,
            **asset.asset_metadata,
        }

        return ChunkSchema(
            chunk_id=uuid4(),
            document_id=asset.document_id,
            version=asset.version,
            chunk_index=chunk_index,
            text=text_rep,
            section=asset.caption or f"Figure {asset.asset_type.value} on Page {asset.page_number}",
            page=asset.page_number,
            metadata=meta,
            chunking_version=self.chunking_version,
        )
