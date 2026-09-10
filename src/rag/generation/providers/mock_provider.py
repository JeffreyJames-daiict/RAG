"""Deterministic mock LLM provider for unit tests, offline development, and failover validation."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Generator, List, Optional, Sequence, Set

from rag.generation.cost import CostTracker
from rag.generation.interfaces import LLMProvider
from rag.generation.models import ChatMessage, LLMResponse, LLMStreamChunk


class MockLLMProvider:
    """Mock LLM provider supporting deterministic answers, citation simulation, and streaming."""

    def __init__(
        self,
        default_answer: Optional[str] = None,
        should_fail: bool = False,
        should_timeout: bool = False,
        should_refuse: bool = False,
        fail_on_models: Optional[Set[str]] = None,
        simulated_latency_sec: float = 0.0,
        provider_name: str = "mock",
    ):
        self.default_answer = default_answer
        self.should_fail = should_fail
        self.should_timeout = should_timeout
        self.should_refuse = should_refuse
        self.fail_on_models = set(fail_on_models or [])
        self.simulated_latency_sec = simulated_latency_sec
        self._provider_name = provider_name
        self.call_count = 0
        self.attempted_models: List[str] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _synthesize_answer(self, messages: Sequence[ChatMessage], model: str) -> str:
        if self.should_refuse:
            return "I am sorry, but I cannot fulfill this request as per safety guidelines."

        if self.default_answer is not None:
            return self.default_answer

        # Look for user query and citations in context
        full_text = " ".join(m.content for m in messages)
        citations = re.findall(r"\[C\d+\]", full_text)
        if citations:
            unique_citations = list(dict.fromkeys(citations))
            cites_str = " ".join(unique_citations[:2])
            return f"Based on the provided documentation, the findings are confirmed by technical evidence {cites_str}."

        if "<!-- No evidence passages retrieved -->" in full_text or "<retrieved_evidence>" not in full_text:
            return "The provided context does not contain sufficient information to answer the question."

        return "The requested information was analyzed based on the retrieved context."

    def _execute(
        self,
        messages: Sequence[ChatMessage],
        model: str,
    ) -> LLMResponse:
        self.call_count += 1
        self.attempted_models.append(model)
        start = time.perf_counter()

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        if self.should_timeout:
            raise TimeoutError(f"Mock provider timed out querying {model}")

        if self.should_fail or model in self.fail_on_models:
            raise RuntimeError(f"Mock provider error querying {model}")

        answer = self._synthesize_answer(messages, model)
        latency_ms = (time.perf_counter() - start) * 1000.0

        in_tok = sum(len(m.content.split()) for m in messages)
        out_tok = len(answer.split())
        tot_tok = in_tok + out_tok
        cost = CostTracker.calculate_cost(model, in_tok, out_tok)

        return LLMResponse(
            text=answer,
            model=model,
            provider=self._provider_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
            finish_reason="stop",
            raw={"simulated": True},
        )

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> LLMResponse:
        if self.simulated_latency_sec > 0:
            await asyncio.sleep(self.simulated_latency_sec)
        return self._execute(messages, model)

    def generate_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> LLMResponse:
        return self._execute(messages, model)

    def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: float = 30.0,
    ) -> Generator[LLMStreamChunk, None, None]:
        self.call_count += 1
        self.attempted_models.append(model)

        if self.should_timeout:
            raise TimeoutError(f"Mock provider stream timed out for {model}")

        if self.should_fail or model in self.fail_on_models:
            raise RuntimeError(f"Mock provider stream failed for {model}")

        answer = self._synthesize_answer(messages, model)
        words = answer.split()

        for idx, w in enumerate(words):
            if self.simulated_latency_sec > 0:
                time.sleep(self.simulated_latency_sec / max(1, len(words)))
            yield LLMStreamChunk(
                delta=w + (" " if idx < len(words) - 1 else ""),
                is_finished=False,
                model=model,
            )

        yield LLMStreamChunk(
            delta="",
            is_finished=True,
            model=model,
            finish_reason="stop",
            token_usage={"total_tokens": len(words)},
        )
