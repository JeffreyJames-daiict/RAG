"""Document metadata lookup tool for agentic workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from rag.agentic.tools.base import BaseTool


class MetadataLookupInput(BaseModel):
    """Input for document metadata inquiry."""

    document_id: Optional[str] = Field(default=None, description="Document unique identifier")
    filename: Optional[str] = Field(default=None, description="Filename of the document")
    field: Optional[str] = Field(default=None, description="Specific metadata field (e.g. 'author', 'created_at')")


class MetadataLookupOutput(BaseModel):
    """Result of metadata query."""

    found: bool = Field(default=False, description="Whether metadata was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata key-value pairs")
    value: Optional[Any] = Field(default=None, description="Specific field value if requested")


class MetadataLookupTool(BaseTool):
    """Tool for querying document metadata."""

    name = "metadata_lookup"
    description = "Queries document metadata such as author, creation date, page count, and document type."
    input_schema = MetadataLookupInput
    output_schema = MetadataLookupOutput

    def __init__(self, metadata_store: Optional[Dict[str, Dict[str, Any]]] = None):
        self._store = metadata_store or {}

    def execute(self, inputs: MetadataLookupInput) -> MetadataLookupOutput:
        """Lookup document metadata."""
        key = (inputs.document_id or inputs.filename or "").lower().strip()
        data = self._store.get(key, {})

        if not data:
            # Fallback search across keys
            for k, v in self._store.items():
                if key and (key in k.lower() or k.lower() in key):
                    data = v
                    break

        if not data:
            return MetadataLookupOutput(found=False, metadata={})

        val = data.get(inputs.field) if inputs.field else None
        return MetadataLookupOutput(
            found=True,
            metadata=data,
            value=val,
        )
