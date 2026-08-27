"""Web and HTTP source connector."""

import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse
import requests
from rag.connectors.base import BaseSourceConnector
from rag.core.exceptions import (
    ConnectionValidationError,
    DiscoveryError,
    FetchError,
    RateLimitError,
)
from rag.core.models import DocumentDescriptor, RawDocumentPayload, SourceType

class WebConnector(BaseSourceConnector):
    """Connector for discovering and fetching web pages / online documents via HTTP."""

    def __init__(
        self,
        seed_urls: List[str],
        user_agent: str = "EnterpriseRAG-Crawler/1.0",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
        session: Optional[requests.Session] = None,
    ):
        super().__init__(tenant_id=tenant_id)
        self.seed_urls = seed_urls
        self.timeout = timeout
        self.headers = headers or {}
        self.headers.setdefault("User-Agent", user_agent)
        self.session = session or requests.Session()
        self.session.headers.update(self.headers)

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB

    def validate_connection(self) -> bool:
        """Verify network reachability of at least one seed URL."""
        if not self.seed_urls:
            raise ConnectionValidationError("No seed URLs provided to WebConnector.")
        test_url = self.seed_urls[0]
        try:
            resp = self.session.head(test_url, timeout=self.timeout, allow_redirects=True)
            if resp.status_code >= 500:
                raise ConnectionValidationError(f"Seed URL returned server error {resp.status_code}: {test_url}")
            return True
        except requests.RequestException as e:
            raise ConnectionValidationError(f"Could not reach seed URL {test_url}: {e}") from e

    def _extract_filename(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path:
            name = path.split("/")[-1]
            if "." in name:
                return name
            return f"{name}.html"
        netloc = parsed.netloc.replace(".", "_")
        return f"{netloc}_index.html"

    def discover(self, since: Optional[datetime] = None) -> Iterator[DocumentDescriptor]:
        """Discover descriptors for configured seed URLs."""
        for url in self.seed_urls:
            try:
                # Perform lightweight HEAD request to inspect headers
                resp = self.session.head(url, timeout=self.timeout, allow_redirects=True)
                if resp.status_code == 429:
                    raise RateLimitError(f"Rate limited while discovering {url}")
                if resp.status_code >= 400:
                    continue

                content_type = resp.headers.get("Content-Type", "text/html").split(";")[0].strip()
                last_modified_str = resp.headers.get("Last-Modified")
                last_modified: Optional[datetime] = None
                if last_modified_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        last_modified = parsedate_to_datetime(last_modified_str)
                    except Exception:
                        pass

                if since and last_modified and last_modified <= since:
                    continue

                filename = self._extract_filename(url)
                etag = resp.headers.get("ETag", "").strip('"')

                yield DocumentDescriptor(
                    source=self.source_type,
                    source_uri=url,
                    filename=filename,
                    mime_type=content_type,
                    tenant_id=self.tenant_id,
                    last_modified_at=last_modified,
                    expected_hash=etag if etag else None,
                    custom_metadata={
                        "url": url,
                        "status_code": resp.status_code,
                        "etag": etag,
                        "content_length": resp.headers.get("Content-Length"),
                    },
                )
            except RateLimitError:
                raise
            except Exception as e:
                raise DiscoveryError(f"Failed to discover URL {url}: {e}") from e

    def fetch(self, descriptor: DocumentDescriptor) -> RawDocumentPayload:
        """Download web page or document payload via GET request."""
        url = descriptor.source_uri
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limited when fetching {url}")
            resp.raise_for_status()

            content = resp.content
            content_hash = self.compute_hash(content)
            
            # Enrich metadata
            descriptor.custom_metadata["fetch_status_code"] = resp.status_code
            descriptor.custom_metadata["response_headers"] = dict(resp.headers)

            return RawDocumentPayload(
                descriptor=descriptor,
                content=content,
                content_hash=content_hash,
            )
        except RateLimitError:
            raise
        except Exception as e:
            raise FetchError(f"Failed to fetch content from {url}: {e}") from e
