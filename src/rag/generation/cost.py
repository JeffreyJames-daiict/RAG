"""Usage and cost tracking for LLM Answer Generation."""

from __future__ import annotations

from typing import Dict, Tuple

# Pricing registry: (cost_per_million_input_tokens, cost_per_million_output_tokens)
MODEL_PRICING_USD_PER_M: Dict[str, Tuple[float, float]] = {
    # Gemini
    "google/gemini-2.5-flash": (0.075, 0.30),
    "google/gemini-2.0-flash": (0.10, 0.40),
    "google/gemini-pro": (1.25, 5.00),
    "google/gemini-1.5-flash": (0.075, 0.30),
    # Anthropic
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    # OpenAI
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4-turbo": (10.00, 30.00),
    # Open source / Meta / DeepSeek
    "meta-llama/llama-3.3-70b-instruct": (0.40, 0.40),
    "meta-llama/llama-3.1-8b-instruct": (0.05, 0.05),
    "deepseek/deepseek-chat": (0.14, 0.28),
    "mistralai/mistral-large": (2.00, 6.00),
    # Mock / Local
    "mock-model": (0.0, 0.0),
    "mock": (0.0, 0.0),
}

DEFAULT_PRICING: Tuple[float, float] = (1.00, 3.00)


class CostTracker:
    """Calculates estimated financial cost in USD from token usage telemetry."""

    @classmethod
    def get_rates(cls, model_name: str) -> Tuple[float, float]:
        """Find matching rates per 1M tokens by exact match or model prefix."""
        norm = model_name.lower().strip()
        if norm in MODEL_PRICING_USD_PER_M:
            return MODEL_PRICING_USD_PER_M[norm]

        for prefix, rates in MODEL_PRICING_USD_PER_M.items():
            if prefix in norm or norm in prefix:
                return rates

        return DEFAULT_PRICING

    @classmethod
    def calculate_cost(
        cls,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate total USD cost for a generation invocation."""
        if input_tokens <= 0 and output_tokens <= 0:
            return 0.0

        rate_in, rate_out = cls.get_rates(model_name)
        cost_in = (input_tokens / 1_000_000.0) * rate_in
        cost_out = (output_tokens / 1_000_000.0) * rate_out
        return round(cost_in + cost_out, 6)
