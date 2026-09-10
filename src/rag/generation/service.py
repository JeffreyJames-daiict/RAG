"""LLM Generation Service — orchestrates prompt formatting, citations, and OpenRouter generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Union

from rag.context import ContextBuilder, ContextPackage
from rag.generation.models import (
    GenerationContextChunk,
    GenerationRequest,
    GenerationResponse,
)
from rag.generation.openrouter_client import DEFAULT_MODELS, OpenRouterGenerator
from rag.retrieval.models import RetrievalResponse, RetrievedChunkResult

logger = logging.getLogger(__name__)


class LLMGenerationService:
    """High-level service orchestrating retrieval context grounding and LLM answer generation."""

    def __init__(
        self,
        generator: Optional[OpenRouterGenerator] = None,
        default_model: Optional[str] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.generator = generator or OpenRouterGenerator()
        self.default_model = default_model or DEFAULT_MODELS[0]
        self.context_builder = context_builder or ContextBuilder()

    def _prepare_context_chunks(
        self,
        chunks_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
    ) -> List[GenerationContextChunk]:
        """Convert various input types into standardized GenerationContextChunk objects."""
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
                section_str = " › ".join(item.section_path) if item.section_path else (item.heading or "")
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

    def generate_answer(
        self,
        query: str,
        retrieval_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
        primary_model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate a grounded, cited answer using retrieved context passages with model failover."""
        context_pkg: Optional[ContextPackage] = None
        if isinstance(retrieval_source, ContextPackage):
            context_pkg = retrieval_source
            chunks = self._prepare_context_chunks(context_pkg)
        elif self.context_builder:
            context_pkg = self.context_builder.build(query, retrieval_source)
            chunks = self._prepare_context_chunks(context_pkg)
        else:
            chunks = self._prepare_context_chunks(retrieval_source)

        request = GenerationRequest(
            query=query,
            context_chunks=chunks,
            context_package=context_pkg,
            primary_model=primary_model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        return self.generator.generate(request)

    def stream_answer(
        self,
        query: str,
        retrieval_source: Union[ContextPackage, RetrievalResponse, List[RetrievedChunkResult], List[Dict[str, Any]]],
        primary_model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated answer with real-time model selection and failover events."""
        context_pkg: Optional[ContextPackage] = None
        if isinstance(retrieval_source, ContextPackage):
            context_pkg = retrieval_source
            chunks = self._prepare_context_chunks(context_pkg)
        elif self.context_builder:
            context_pkg = self.context_builder.build(query, retrieval_source)
            chunks = self._prepare_context_chunks(context_pkg)
        else:
            chunks = self._prepare_context_chunks(retrieval_source)

        request = GenerationRequest(
            query=query,
            context_chunks=chunks,
            context_package=context_pkg,
            primary_model=primary_model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        return self.generator.generate_stream(request)
