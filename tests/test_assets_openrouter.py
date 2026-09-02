"""Unit tests for OpenRouter Free Vision Provider and auto model selection."""

import io
from PIL import Image

from rag.assets.models import FigureAnalysisResult
from rag.assets.providers.openrouter_vision import FREE_VISION_MODELS, OpenRouterVisionProvider
from rag.assets.providers import get_vision_provider
from rag.core.config import VisionConfig
from rag.core.models import AssetType


def test_openrouter_free_vision_provider_init():
    provider = OpenRouterVisionProvider(api_key="test_key")
    assert provider.model_name in FREE_VISION_MODELS
    assert len(provider.candidate_models) >= len(FREE_VISION_MODELS)


def test_openrouter_factory_selection():
    cfg = VisionConfig(provider="openrouter/free", api_key="sk-test")
    provider = get_vision_provider(cfg)
    assert isinstance(provider, OpenRouterVisionProvider)


def test_openrouter_smart_contextual_fallback_without_key():
    provider = OpenRouterVisionProvider(api_key="")
    img = Image.new("RGB", (300, 200), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    result = provider.analyze_figure(
        image_bytes=data,
        caption="Architecture Overview for Payment Gateway",
        ocr_text="Payment Service -> Database",
        asset_type=AssetType.ARCHITECTURE_DIAGRAM,
    )

    assert isinstance(result, FigureAnalysisResult)
    assert result.status == "SUCCESS"
    assert "300×200" in result.description or "300x200" in result.description or "Architecture Overview" in result.description
    assert len(result.entities) > 0
