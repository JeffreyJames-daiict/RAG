"""Hardened test suite for the LLM Answer Generation Layer.

Covers:
- LLMProvider protocol compliance (sync, async, streaming)
- Versioned prompt construction (grounding, citations, abstention, injection isolation)
- Cost calculation and token usage tracking across model tiers
- Response validation and citation cross-checking against ContextPackage
- Resilient model failover and transient error retries
- Streaming token delivery and error handling
- End-to-end answer synthesis with ContextPackage integration
"""

import asyncio
import pytest

from rag.context.models import Citation, ContextChunk, ContextPackage
from rag.core.config import GenerationConfig
from rag.generation.cost import CostTracker
from rag.generation.interfaces import LLMProvider
from rag.generation.models import (
    AnswerResponse,
    ChatMessage,
    GenerationResponse,
    LLMResponse,
    LLMStreamChunk,
)
from rag.generation.prompt_builder import PromptBuilder
from rag.generation.providers.factory import get_llm_provider
from rag.generation.providers.mock_provider import MockLLMProvider
from rag.generation.service import LLMGenerationService
from rag.generation.validator import ResponseValidator
from rag.retrieval.models import RetrievedChunkResult


# ── Test Helpers ───────────────────────────────────────────────────────────────

def _make_context_package(
    chunks_text: list[str] = None,
    citations_data: list[tuple[str, str, int]] = None,
) -> ContextPackage:
    """Construct a mock ContextPackage for generation tests."""
    texts = chunks_text or ["Autonomous vehicle perception uses LiDAR [C1]."]
    cites = citations_data or [("[C1]", "c1", 1)]

    selected: list[ContextChunk] = []
    citations: list[Citation] = []

    formatted_passages = []
    for idx, (cid, chunk_id, page) in enumerate(cites):
        t = texts[idx] if idx < len(texts) else "Evidence text"
        selected.append(
            ContextChunk(
                chunk_id=chunk_id,
                document_id="doc_test",
                text=t,
                rank=idx + 1,
                page=page,
                citation_id=cid,
                token_count=len(t.split()),
            )
        )
        citations.append(
            Citation(
                citation_id=cid,
                chunk_id=chunk_id,
                document_id="doc_test",
                page=page,
                snippet=t[:60],
            )
        )
        formatted_passages.append(f'<passage citation="{cid}" page="{page}">\n{t}\n</passage>')

    formatted = (
        "<retrieved_evidence>\n"
        "<document id=\"doc_test\">\n"
        + "\n".join(formatted_passages)
        + "\n</document>\n"
        "</retrieved_evidence>"
    )

    return ContextPackage(
        query="Test query",
        selected_chunks=selected,
        formatted_context=formatted,
        citations=citations,
        token_count=len(formatted.split()),
        token_budget=2000,
    )


# ── 1. MockLLMProvider & Protocol Compliance Tests ─────────────────────────────

class TestMockLLMProvider:
    """Verify MockLLMProvider fulfills LLMProvider protocol in sync, async, and streaming modes."""

    def test_protocol_compliance(self):
        """MockLLMProvider must satisfy LLMProvider runtime protocol."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)
        assert provider.provider_name == "mock"

    def test_generate_sync(self):
        """Synchronous generation produces complete LLMResponse."""
        provider = MockLLMProvider(default_answer="Grounded answer text [C1].")
        msgs = [ChatMessage(role="user", content="Explain perception")]
        resp = provider.generate_sync(msgs, model="test-model")

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Grounded answer text [C1]."
        assert resp.model == "test-model"
        assert resp.total_tokens > 0
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_async(self):
        """Asynchronous generation matches synchronous results."""
        provider = MockLLMProvider(default_answer="Async answer.")
        msgs = [ChatMessage(role="user", content="Query")]
        resp = await provider.generate(msgs, model="test-model")

        assert resp.text == "Async answer."
        assert resp.model == "test-model"

    def test_generate_stream(self):
        """Streaming yields delta chunks and terminates with is_finished=True."""
        provider = MockLLMProvider(default_answer="One two three.")
        msgs = [ChatMessage(role="user", content="Count")]
        chunks = list(provider.generate_stream(msgs, model="stream-model"))

        assert len(chunks) >= 3
        deltas = [c.delta for c in chunks if not c.is_finished]
        combined = "".join(deltas).strip()
        assert combined == "One two three."
        assert chunks[-1].is_finished is True
        assert chunks[-1].finish_reason == "stop"


# ── 2. Versioned PromptBuilder Tests ───────────────────────────────────────────

class TestPromptBuilder:
    """Verify versioned prompt messages, citation directives, and abstention policy."""

    def test_prompt_includes_grounding_and_abstention(self):
        """System prompt contains strict evidence rules and abstention instructions."""
        builder = PromptBuilder()
        pkg = _make_context_package()
        msgs = builder.build_messages("What is LiDAR?", pkg)

        assert len(msgs) == 2
        sys_msg = msgs[0]
        user_msg = msgs[1]

        assert sys_msg.role == "system"
        assert "Grounding" in sys_msg.content
        assert "Citations" in sys_msg.content
        assert "Abstention Policy" in sys_msg.content
        assert "Security & Prompt Injection" in sys_msg.content

        assert user_msg.role == "user"
        assert "What is LiDAR?" in user_msg.content
        assert "<retrieved_evidence>" in user_msg.content

    def test_custom_system_prompt_override(self):
        """Custom system prompt overrides default versioned prompt when passed."""
        builder = PromptBuilder()
        custom = "Custom strict research guidelines."
        msgs = builder.build_messages("Query", custom_system_prompt=custom)

        assert msgs[0].content == custom


# ── 3. Cost & Usage Tracking Tests ─────────────────────────────────────────────

class TestCostTracker:
    """Verify pricing calculations for various model families."""

    def test_cost_calculation(self):
        """Calculates accurate USD cost based on token counts and model pricing."""
        # 1,000,000 input tokens and 1,000,000 output tokens on claude-3.5-sonnet ($3 in, $15 out)
        cost = CostTracker.calculate_cost("anthropic/claude-3.5-sonnet", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0)

        # Zero tokens
        assert CostTracker.calculate_cost("openai/gpt-4o", 0, 0) == 0.0

        # Gemini Flash pricing ($0.075 in, $0.30 out)
        flash_cost = CostTracker.calculate_cost("google/gemini-2.5-flash", 1_000_000, 1_000_000)
        assert flash_cost == pytest.approx(0.375)


# ── 4. Response Validator & Citation Cross-Checking Tests ──────────────────────

class TestResponseValidator:
    """Verify response validation, refusal detection, and citation integrity checks."""

    def test_empty_response_fails_validation(self):
        """Empty or whitespace-only response fails validation."""
        val = ResponseValidator()
        res = val.validate("   ")
        assert res.is_valid is False
        assert any("empty" in w.lower() for w in res.warnings)

    def test_refusal_detection(self):
        """Safety refusal strings are detected and marked invalid."""
        val = ResponseValidator()
        res = val.validate("I am sorry, but I cannot assist with this request.")
        assert res.is_valid is False
        assert any("refusal" in w.lower() for w in res.warnings)

    def test_valid_and_hallucinated_citations(self):
        """Citations are cross-checked against ContextPackage registry."""
        val = ResponseValidator(validate_citations=True)
        pkg = _make_context_package(citations_data=[("[C1]", "c1", 1), ("[C2]", "c2", 2)])

        # Text references [C1] (valid) and [C99] (hallucinated)
        text = "Finding one [C1] is verified, but speculative claim [C99] was invented."
        res = val.validate(text, context_package=pkg)

        assert res.is_valid is True
        assert "[C1]" in res.valid_citations
        assert "[C99]" in res.hallucinated_citations
        assert any("hallucinated" in w.lower() for w in res.warnings)


# ── 5. Resilient Failover & Retry Tests ────────────────────────────────────────

class TestFailoverAndRetries:
    """Verify automatic retries and model failover when primary model fails."""

    def test_model_failover_on_primary_failure(self):
        """When primary model fails, service automatically fails over to fallback model."""
        cfg = GenerationConfig(
            model="failing-primary-model",
            fallback_model="working-fallback-model",
            max_retries=1,
        )
        provider = MockLLMProvider(
            fail_on_models={"failing-primary-model"},
            default_answer="Answer produced by working fallback model [C1].",
        )
        svc = LLMGenerationService(provider=provider, config=cfg)
        pkg = _make_context_package()

        resp = svc.generate_answer("query", pkg)

        assert resp.fallback_occurred is True
        assert resp.model_used == "working-fallback-model"
        assert len(resp.attempted_models) == 2
        assert resp.attempted_models[0].model == "failing-primary-model"
        assert resp.attempted_models[0].success is False
        assert resp.attempted_models[1].model == "working-fallback-model"
        assert resp.attempted_models[1].success is True

    def test_exhausted_failover_returns_clean_response(self):
        """When all models in roster fail, returns graceful exhausted response without crashing."""
        cfg = GenerationConfig(
            model="model-1",
            fallback_model="model-2",
            fallback_models=[],
            max_retries=0,
        )
        provider = MockLLMProvider(should_fail=True)
        svc = LLMGenerationService(provider=provider, config=cfg)

        resp = svc.generate_answer("query", _make_context_package())

        assert resp.model_used == "exhausted"
        assert resp.fallback_occurred is True
        assert "All candidate generation models failed" in resp.answer


# ── 6. Streaming Generation Tests ──────────────────────────────────────────────

class TestStreamingGeneration:
    """Verify streaming token delivery and chunk events."""

    def test_streaming_emits_tokens_and_done_event(self):
        """stream_answer yields model selection, tokens, and done event with citations."""
        provider = MockLLMProvider(default_answer="Streaming answer text [C1].")
        svc = LLMGenerationService(provider=provider)
        pkg = _make_context_package()

        events = list(svc.stream_answer("query", pkg))
        event_types = [e["type"] for e in events]

        assert "model_selected" in event_types
        assert "token" in event_types
        assert "done" in event_types

        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["model_used"] == svc.config.model
        assert "citations" in done_event


# ── 7. End-to-End Generation Tests ─────────────────────────────────────────────

class TestEndToEndGeneration:
    """Verify end-to-end integration between ContextBuilder and LLMGenerationService."""

    def test_generate_from_raw_chunks_auto_builds_package(self):
        """Raw RetrievedChunkResults automatically trigger ContextBuilder."""
        provider = MockLLMProvider()
        svc = LLMGenerationService(provider=provider)

        raw_chunk = RetrievedChunkResult(
            chunk_id="raw_1",
            document_id="doc_raw",
            document_version=1,
            document_version_id="v1",
            tenant_id="default",
            text="Autonomous vehicle sensors and camera perception.",
            rank=1,
            page=2,
            section="Sensors",
        )

        resp = svc.generate_answer("Explain sensors", [raw_chunk])

        assert isinstance(resp, AnswerResponse)
        assert resp.context_package is not None
        assert len(resp.context_package.selected_chunks) == 1
        assert len(resp.citations) == 1
        assert resp.citations[0].citation_id == "[C1]"
        assert resp.total_tokens is not None
        assert resp.total_tokens > 0

    def test_provider_factory_instantiates_mock(self):
        """Provider factory creates MockLLMProvider when configured."""
        cfg = GenerationConfig(provider="mock")
        provider = get_llm_provider(cfg)
        assert isinstance(provider, MockLLMProvider)
        assert provider.provider_name == "mock"
