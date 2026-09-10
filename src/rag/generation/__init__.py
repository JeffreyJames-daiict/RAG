"""Generation package — grounded LLM answer synthesis with model failover, validation, and cost tracking."""

from rag.generation.cost import CostTracker
from rag.generation.interfaces import LLMProvider
from rag.generation.models import (
    AnswerResponse,
    ChatMessage,
    GenerationContextChunk,
    GenerationRequest,
    GenerationResponse,
    LLMResponse,
    LLMStreamChunk,
    ModelAttemptRecord,
    ValidationResult,
)
from rag.generation.openrouter_client import (
    DEFAULT_MODELS,
    OpenRouterGenerator,
    OpenRouterProvider,
)
from rag.generation.prompt_builder import PromptBuilder
from rag.generation.providers.factory import get_llm_provider
from rag.generation.providers.mock_provider import MockLLMProvider
from rag.generation.service import LLMGenerationService
from rag.generation.validator import ResponseValidator

__all__ = [
    "AnswerResponse",
    "ChatMessage",
    "CostTracker",
    "DEFAULT_MODELS",
    "GenerationContextChunk",
    "GenerationRequest",
    "GenerationResponse",
    "LLMGenerationService",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "MockLLMProvider",
    "ModelAttemptRecord",
    "OpenRouterGenerator",
    "OpenRouterProvider",
    "PromptBuilder",
    "ResponseValidator",
    "ValidationResult",
    "get_llm_provider",
]
