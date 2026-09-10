"""Production LLM Answer Generation Service — orchestrates prompt building, provider routing, retries, failover, validation, and cost tracking."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from rag.context import ContextBuilder, ContextPackage
from rag.core.config import GenerationConfig, get_settings
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
)
from rag.generation.prompt_builder import PromptBuilder
from rag.generation.providers.factory import get_llm_provider
from rag.generation.validator import ResponseValidator
from rag.retrieval.models import RetrievalResponse, RetrievedChunkResult

logger = logging.getLogger(__name__)


class LLMGenerationService:
    """High-level service orchestrating retrieval context grounding and LLM answer generation."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[GenerationConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[ResponseValidator] = None,
        context_builder: Optional[ContextBuilder] = None,
        # Legacy backwards compatibility arguments
        generator: Optional[Any] = None,
        default_model: Optional[str] = None,
    ):
        self.config = config or getattr(get_settings(), "generation", GenerationConfig())
        self.legacy_generator = None
        if generator is not None and not hasattr(generator, "generate_sync"):
            self.legacy_generator = generator
            self.provider = None
        else:
            self.provider = provider or generator or get_llm_provider(self.config)
        self.generator = generator or self.provider
        self.prompt_builder = prompt_builder or PromptBuilder(version=self.config.prompt_version)
        self.validator = validator or ResponseValidator(validate_citations=self.config.validate_citations)
        self.context_builder = context_builder or ContextBuilder()
        self.default_model = default_model or self.config.model

    def _prepare_context_package(
        self,
        query: str,
        retrieval_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
    ) -> ContextPackage:
        """Coerce arbitrary retrieval input into a fully structured ContextPackage."""
        if isinstance(retrieval_source, ContextPackage):
            return retrieval_source
        if self.context_builder:
            return self.context_builder.build(query, retrieval_source)

        # Fallback empty context package if context builder is disabled
        return ContextPackage(query=query)

    def _prepare_context_chunks(
        self,
        chunks_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
    ) -> List[GenerationContextChunk]:
        """Convert input sources into legacy GenerationContextChunk objects for backwards compatibility."""
        context_chunks: List[GenerationContextChunk] = []

        if isinstance(chunks_source, ContextPackage):
            for ch in chunks_source.selected_chunks:
                context_chunks.append(
                    GenerationContextChunk(
                        rank=ch.rank,
                        text=ch.text,
                        chunk_id=ch.chunk_id,
                        filename=ch.filename,
                        page=ch.page,
                        section=ch.section,
                        chunk_type=ch.chunk_type,
                        metadata={**ch.metadata, "citation_id": ch.citation_id},
                    )
                )
            return context_chunks

        if isinstance(chunks_source, RetrievalResponse):
            raw_items = chunks_source.results
        elif isinstance(chunks_source, list):
            raw_items = chunks_source
        else:
            raw_items = []

        for idx, item in enumerate(raw_items, start=1):
            if isinstance(item, RetrievedChunkResult):
                section_str = " › ".join(item.section_path) if item.section_path else (item.heading or item.section or "")
                context_chunks.append(
                    GenerationContextChunk(
                        rank=item.rank or idx,
                        text=item.text,
                        chunk_id=item.chunk_id,
                        filename=item.filename,
                        page=item.page,
                        section=section_str,
                        chunk_type=item.chunk_type,
                        metadata=item.metadata,
                    )
                )
            elif isinstance(item, dict):
                context_chunks.append(
                    GenerationContextChunk(
                        rank=item.get("rank", idx),
                        text=item.get("text", ""),
                        chunk_id=item.get("chunk_id", ""),
                        filename=item.get("filename", ""),
                        page=item.get("page"),
                        section=item.get("section"),
                        chunk_type=item.get("chunk_type", "text"),
                        metadata=item.get("metadata", {}),
                    )
                )

        return context_chunks

    def _get_candidate_models(self, requested_model: Optional[str] = None) -> List[str]:
        """Build ordered model failover roster."""
        primary = requested_model or self.config.model or self.default_model
        models = [primary]

        if self.config.fallback_model and self.config.fallback_model != primary:
            models.append(self.config.fallback_model)

        for m in self.config.fallback_models:
            if m not in models:
                models.append(m)

        return models

    def generate_answer(
        self,
        query: str,
        retrieval_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
        primary_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AnswerResponse:
        """Generate a grounded, cited answer using retrieved context with automatic retries and failover."""
        start_time = time.perf_counter()
        target_temp = temperature if temperature is not None else self.config.temperature
        target_tokens = max_tokens or self.config.max_tokens

        # Build structured ContextPackage
        context_package = self._prepare_context_package(query, retrieval_source)

        # Legacy generator backward-compatibility path
        if self.legacy_generator is not None:
            chunks = self._prepare_context_chunks(context_package)
            req = GenerationRequest(
                query=query,
                context_chunks=chunks,
                context_package=context_package,
                primary_model=primary_model or self.default_model,
                temperature=target_temp,
                max_tokens=target_tokens,
                system_prompt=system_prompt,
            )
            resp = self.legacy_generator.generate(req)
            if getattr(resp, "context_package", None) is None:
                resp.context_package = context_package
            return resp

        # Build versioned prompt messages
        messages = self.prompt_builder.build_messages(
            query=query,
            context_package=context_package,
            custom_system_prompt=system_prompt,
        )

        candidate_models = self._get_candidate_models(primary_model)
        attempts: List[ModelAttemptRecord] = []
        last_error: Optional[str] = None

        total_retries = 0

        for model_idx, model in enumerate(candidate_models):
            m_start = time.perf_counter()
            retries = 0
            model_error: Optional[str] = None

            while retries <= self.config.max_retries:
                try:
                    # Execute synchronous generation
                    llm_resp = self.provider.generate_sync(
                        messages=messages,
                        model=model,
                        temperature=target_temp,
                        max_tokens=target_tokens,
                        timeout_sec=self.config.timeout_sec,
                    )

                    # Validate response
                    val_result = self.validator.validate(
                        raw_text=llm_resp.text,
                        context_package=context_package,
                    )

                    total_ms = (time.perf_counter() - start_time) * 1000.0
                    m_ms = (time.perf_counter() - m_start) * 1000.0

                    attempts.append(
                        ModelAttemptRecord(
                            model=model,
                            success=True,
                            retries_attempted=retries,
                            duration_ms=round(m_ms, 2),
                        )
                    )

                    # Calculate or extract cost
                    cost = llm_resp.cost_usd or CostTracker.calculate_cost(
                        model, llm_resp.input_tokens, llm_resp.output_tokens
                    )

                    # Extract cited numeric ranks for backwards compatibility
                    cited_ranks: List[int] = []
                    for c_str in val_result.valid_citations:
                        digits = "".join(filter(str.isdigit, c_str))
                        if digits:
                            cited_ranks.append(int(digits))

                    p_name = getattr(self.provider, "provider_name", "unknown_provider")
                    return AnswerResponse(
                        query=query,
                        answer=val_result.clean_text,
                        model_used=model,
                        provider_used=p_name,
                        fallback_occurred=(model_idx > 0),
                        attempted_models=attempts,
                        cited_ranks=sorted(list(set(cited_ranks))),
                        cited_markers=val_result.valid_citations,
                        hallucinated_citations=val_result.hallucinated_citations,
                        citations=context_package.citations if context_package else [],
                        context_package=context_package,
                        latency_ms=round(total_ms, 2),
                        input_tokens=llm_resp.input_tokens,
                        output_tokens=llm_resp.output_tokens,
                        total_tokens=llm_resp.total_tokens,
                        estimated_cost_usd=cost,
                        retry_count=total_retries,
                        warnings=val_result.warnings + context_package.warnings,
                        raw_response=llm_resp.raw,
                    )

                except Exception as exc:
                    model_error = str(exc)
                    retries += 1
                    total_retries += 1
                    logger.warning(
                        "Model '%s' failed (attempt %d/%d): %s",
                        model,
                        retries,
                        self.config.max_retries + 1,
                        exc,
                    )
                    if retries <= self.config.max_retries:
                        time.sleep(0.3 * (2 ** (retries - 1)))

            # Model failed across all retries; record and failover to next model
            m_ms = (time.perf_counter() - m_start) * 1000.0
            attempts.append(
                ModelAttemptRecord(
                    model=model,
                    success=False,
                    error=model_error,
                    retries_attempted=retries,
                    duration_ms=round(m_ms, 2),
                )
            )
            last_error = model_error
            logger.info("Failing over from model '%s' to next candidate in roster.", model)

        # All models exhausted
        total_ms = (time.perf_counter() - start_time) * 1000.0
        p_name = getattr(self.provider, "provider_name", "unknown_provider")
        return AnswerResponse(
            query=query,
            answer=f"⚠️ All candidate generation models failed to produce an answer. Last error: {last_error}",
            model_used="exhausted",
            provider_used=p_name,
            fallback_occurred=True,
            attempted_models=attempts,
            latency_ms=round(total_ms, 2),
            retry_count=total_retries,
            context_package=context_package,
            warnings=[f"Generation exhausted: {last_error}"],
        )

    def stream_answer(
        self,
        query: str,
        retrieval_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
        primary_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated answer with real-time model failover events."""
        target_temp = temperature if temperature is not None else self.config.temperature
        target_tokens = max_tokens or self.config.max_tokens

        context_package = self._prepare_context_package(query, retrieval_source)
        if self.legacy_generator is not None and hasattr(self.legacy_generator, "generate_stream"):
            chunks = self._prepare_context_chunks(context_package)
            req = GenerationRequest(
                query=query,
                context_chunks=chunks,
                context_package=context_package,
                primary_model=primary_model or self.default_model,
                temperature=target_temp,
                max_tokens=target_tokens,
                system_prompt=system_prompt,
            )
            return self.legacy_generator.generate_stream(req)

        messages = self.prompt_builder.build_messages(
            query=query,
            context_package=context_package,
            custom_system_prompt=system_prompt,
        )

        candidate_models = self._get_candidate_models(primary_model)

        for model_idx, model in enumerate(candidate_models):
            yield {
                "type": "model_selected",
                "model": model,
                "fallback": (model_idx > 0),
            }

            try:
                chunk_gen = self.provider.generate_stream(
                    messages=messages,
                    model=model,
                    temperature=target_temp,
                    max_tokens=target_tokens,
                    timeout_sec=self.config.timeout_sec,
                )

                for chunk in chunk_gen:
                    if chunk.error:
                        yield {"type": "error", "message": chunk.error}
                        break
                    if chunk.delta:
                        yield {"type": "token", "content": chunk.delta}
                    if chunk.is_finished:
                        yield {
                            "type": "done",
                            "model_used": model,
                            "fallback": (model_idx > 0),
                            "citations": [c.model_dump() for c in context_package.citations],
                        }
                        return

            except Exception as exc:
                logger.warning("Stream failed on model '%s': %s. Failing over...", model, exc)
                continue

        yield {
            "type": "error",
            "message": "All candidate models failed during streaming.",
        }
