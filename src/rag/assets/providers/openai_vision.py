"""OpenAI GPT-4o / Vision API provider for diagram and figure semantic analysis."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional
import urllib.error
import urllib.request

from rag.assets.exceptions import VisionAnalysisError
from rag.assets.models import FigureAnalysisResult, FigureEntityRelationship
from rag.assets.providers.base import BaseVisionProvider
from rag.core.models import AssetType

logger = logging.getLogger(__name__)


class OpenAIVisionProvider(BaseVisionProvider):
    """Multimodal vision provider using OpenAI GPT-4o or compatible vision endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 45.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.endpoint_url = endpoint_url or "https://api.openai.com/v1/chat/completions"
        self.timeout_sec = timeout_sec

    def analyze_figure(
        self,
        image_bytes: bytes,
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        asset_type: AssetType = AssetType.FIGURE,
        mime_type: str = "image/png",
    ) -> FigureAnalysisResult:
        """Call OpenAI Vision API to analyze image and return structured semantic representation."""
        if not self.api_key:
            raise VisionAnalysisError("OpenAI API key is required for OpenAIVisionProvider")

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_img}"

        system_prompt = (
            "You are an expert technical document analysis system specializing in architecture diagrams, "
            "flowcharts, and technical figures. Analyze the provided image and output a strict JSON object with: \n"
            "- 'description': Comprehensive explanation of what the diagram/figure depicts and its workflow.\n"
            "- 'entities': List of distinct components, systems, services, databases, actors, or labels.\n"
            "- 'relationships': List of objects {'source': '...', 'relationship': '...', 'target': '...'}.\n"
            "- 'summary': A concise 1-sentence summary.\n"
            "Respond ONLY with valid JSON."
        )

        user_content = [
            {"type": "text", "text": f"Asset Type: {asset_type.value}\nCaption: {caption or 'None'}\nOCR Text Detected: {ocr_text or 'None'}\nPlease analyze this figure in detail."},
            {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=req_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data["choices"][0]["message"]["content"]
                parsed = json.loads(choice)

                relationships = []
                for r in parsed.get("relationships", []):
                    relationships.append(
                        FigureEntityRelationship(
                            source=str(r.get("source", "")),
                            relationship=str(r.get("relationship", "connects_to")),
                            target=str(r.get("target", "")),
                        )
                    )

                return FigureAnalysisResult(
                    description=parsed.get("description", ""),
                    entities=parsed.get("entities", []),
                    relationships=relationships,
                    summary=parsed.get("summary", ""),
                    provider_name=self.model_name,
                    status="SUCCESS",
                )

        except urllib.error.HTTPError as http_err:
            body = http_err.read().decode("utf-8", errors="ignore")
            raise VisionAnalysisError(f"OpenAI Vision API HTTP error {http_err.code}: {body}") from http_err
        except Exception as exc:
            raise VisionAnalysisError(f"OpenAI Vision analysis failed: {exc}") from exc
