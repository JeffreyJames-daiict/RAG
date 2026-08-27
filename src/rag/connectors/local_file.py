"""Local filesystem source connector."""

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set
from rag.connectors.base import BaseSourceConnector
from rag.core.exceptions import ConnectionValidationError, DiscoveryError, FetchError
from rag.core.models import DocumentDescriptor, RawDocumentPayload, SourceType


class LocalFileConnector(BaseSourceConnector):
    """Connector for ingesting files from a local directory."""

    DEFAULT_ALLOWED_EXTENSIONS = {
        ".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv", ".json", ".html", ".xml"
    }

    def __init__(
        self,
        base_dir: str | Path,
        allowed_extensions: Optional[Set[str]] = None,
        recursive: bool = True,
        tenant_id: str = "default",
    ):
        super().__init__(tenant_id=tenant_id)
        self.base_dir = Path(base_dir).resolve()
        self.allowed_extensions = {ext.lower() for ext in (allowed_extensions or self.DEFAULT_ALLOWED_EXTENSIONS)}
        self.recursive = recursive

    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL_FILE

    def validate_connection(self) -> bool:
        """Ensure base directory exists and is readable."""
        if not self.base_dir.exists():
            raise ConnectionValidationError(f"Base directory does not exist: {self.base_dir}")
        if not self.base_dir.is_dir():
            raise ConnectionValidationError(f"Base path is not a directory: {self.base_dir}")
        if not os.access(self.base_dir, os.R_OK):
            raise ConnectionValidationError(f"Permission denied to read base directory: {self.base_dir}")
        return True

    def _get_mime_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path.name)
        if mime:
            return mime
        ext = path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".json": "application/json",
            ".html": "text/html",
        }
        return mime_map.get(ext, "application/octet-stream")

    def discover(self, since: Optional[datetime] = None) -> Iterator[DocumentDescriptor]:
        """Traverse directory and discover matching files."""
        self.validate_connection()
        try:
            pattern = "**/*" if self.recursive else "*"
            for file_path in self.base_dir.glob(pattern):
                if file_path.is_file():
                    if file_path.suffix.lower() in self.allowed_extensions:
                        stat = file_path.stat()
                        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                        
                        if since and modified_time <= since:
                            continue

                        yield DocumentDescriptor(
                            source=self.source_type,
                            source_uri=str(file_path.resolve()),
                            filename=file_path.name,
                            mime_type=self._get_mime_type(file_path),
                            tenant_id=self.tenant_id,
                            last_modified_at=modified_time,
                            custom_metadata={
                                "file_size_bytes": stat.st_size,
                                "relative_path": str(file_path.relative_to(self.base_dir)),
                                "file_extension": file_path.suffix.lower(),
                            },
                        )
        except Exception as e:
            raise DiscoveryError(f"Error discovering files in {self.base_dir}: {e}") from e

    def fetch(self, descriptor: DocumentDescriptor) -> RawDocumentPayload:
        """Read file content bytes from disk and calculate SHA-256 hash."""
        file_path = Path(descriptor.source_uri)
        if not file_path.exists() or not file_path.is_file():
            raise FetchError(f"File not found: {descriptor.source_uri}")

        try:
            content = file_path.read_bytes()
            content_hash = self.compute_hash(content)
            return RawDocumentPayload(
                descriptor=descriptor,
                content=content,
                content_hash=content_hash,
            )
        except Exception as e:
            raise FetchError(f"Error reading file {descriptor.source_uri}: {e}") from e
