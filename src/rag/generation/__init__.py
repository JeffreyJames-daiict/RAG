"""Generation package — grounded LLM answer synthesis with OpenRouter model rotation and retries."""

from rag.generation.models import (
    GenerationContextChunk,
    GenerationRequest,
    GenerationResponse,
    ModelAttemptRecord,
)
from rag.generation.openrouter_client import DEFAULT_MODELS, OpenRouterGenerator
from rag.generation.service import LLMGenerationService

__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "GenerationContextChunk",
    "ModelAttemptRecord",
    "OpenRouterGenerator",
    "LLMGenerationService",
    "DEFAULT_MODELS",
]
