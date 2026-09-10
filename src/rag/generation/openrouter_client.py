"""OpenRouter Client with automatic retries and model rotation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from rag.generation.models import (
    GenerationRequest,
    GenerationResponse,
    ModelAttemptRecord,
)

logger = logging.getLogger(__name__)

# Curated roster of active free models on OpenRouter (openrouter/free dynamically routes to best available)
DEFAULT_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
]


class OpenRouterGenerator:
    """Production client for OpenRouter LLM generation with automatic retries and model rotation.

    Features:
    - Automatic exponential backoff retries on 429 rate limits, 502/503/504 errors, or timeouts.
    - Automatic model rotation across candidate models when a model fails or quota is exhausted.
    - Transparent streaming and non-streaming modes.
    - Citation extraction ([1], [2]).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        candidate_models: Optional[List[str]] = None,
        max_retries_per_model: int = 2,
        timeout_sec: float = 40.0,
        endpoint_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ):
        if not api_key and "OPENROUTER_API_KEY" not in os.environ:
            from pathlib import Path
            from dotenv import load_dotenv
            # Attempt to locate RAG/.env
            for cand in [Path.cwd() / "RAG" / ".env", Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
                if cand.exists():
                    load_dotenv(cand)
                    break

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.candidate_models = list(candidate_models or DEFAULT_MODELS)
        self.max_retries = max_retries_per_model
        self.timeout_sec = timeout_sec
        self.endpoint_url = endpoint_url

    def _get_ordered_models(self, primary_model: Optional[str] = None) -> List[str]:
        """Return candidate models ordered with preferred primary model first."""
        if not primary_model:
            return list(self.candidate_models)

        ordered = [primary_model]
        for m in self.candidate_models:
            if m != primary_model and m not in ordered:
                ordered.append(m)
        return ordered

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, str]]:
        """Construct the prompt messages formatted with grounded context passages."""
        default_sys = (
            "You are an expert AI research assistant. Answer the user's question accurately, "
            "synthesizing the key findings based strictly on the provided context passages.\n"
            "Guidelines:\n"
            "1. Ground every claim in the provided passages.\n"
            "2. Include inline citation tags like [1], [2] corresponding to the source ranks provided.\n"
            "3. If the context does not contain the answer, state clearly that the information "
            "is not present in the retrieved passages without hallucinating.\n"
            "4. Organize your answer clearly using bullet points or concise sections where appropriate."
        )
        system_prompt = request.system_prompt or default_sys

        if request.context_package and getattr(request.context_package, "formatted_context", None):
            user_content = f"User Question: {request.query}\n\n{request.context_package.formatted_context}"
        else:
            # Format context chunks with provenance
            context_blocks = []
            for ch in request.context_chunks:
                meta_parts = [f"[{ch.rank}]"]
                if ch.filename:
                    meta_parts.append(f"Source: {ch.filename}")
                if ch.page:
                    meta_parts.append(f"Page {ch.page}")
                if ch.section:
                    meta_parts.append(f"Section: {ch.section}")
                header = " • ".join(meta_parts)
                context_blocks.append(f"{header}\n{ch.text.strip()}")

            context_str = "\n\n---\n\n".join(context_blocks)
            user_content = f"User Question: {request.query}\n\nRetrieved Passages:\n{context_str}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _extract_citations(self, text: str) -> List[int]:
        """Extract all cited numbers like [1], [2], [C1], [C2] from the generated text."""
        citations = set()
        # Standard numeric citations: [1], [2]
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
        for m in matches:
            for num in m.split(","):
                try:
                    citations.add(int(num.strip()))
                except ValueError:
                    pass

        # Context builder citations: [C1], [C2], [C1, C2]
        c_matches = re.findall(r"\[C(\d+(?:,\s*C?\d+)*)\]", text, re.IGNORECASE)
        for m in c_matches:
            for num in re.split(r"[,\s]+", m):
                cleaned = re.sub(r"[^0-9]", "", num)
                if cleaned:
                    try:
                        citations.add(int(cleaned))
                    except ValueError:
                        pass

        return sorted(citations)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate complete answer with automatic retry and model rotation."""
        if not self.api_key:
            return GenerationResponse(
                query=request.query,
                answer="⚠️ **OpenRouter API Key missing.** Please set `OPENROUTER_API_KEY` in `RAG/.env`.",
                model_used="none",
                fallback_occurred=False,
            )

        start_time = time.time()
        models = self._get_ordered_models(request.primary_model)
        messages = self._build_messages(request)
        attempts: List[ModelAttemptRecord] = []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/inx/enterprise-rag",
            "X-Title": "Enterprise RAG Pipeline",
        }

        last_error = None

        for model_idx, model in enumerate(models):
            m_start = time.time()
            retries = 0
            model_succeeded = False
            model_error_msg = None

            while retries <= self.max_retries:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                }

                try:
                    resp = requests.post(
                        self.endpoint_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_sec,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data["choices"][0]["message"]["content"].strip()
                        m_duration = (time.time() - m_start) * 1000.0

                        attempts.append(
                            ModelAttemptRecord(
                                model=model,
                                success=True,
                                retries_attempted=retries,
                                duration_ms=round(m_duration, 1),
                            )
                        )

                        cited = self._extract_citations(answer_text)
                        total_ms = (time.time() - start_time) * 1000.0

                        return GenerationResponse(
                            query=request.query,
                            answer=answer_text,
                            model_used=model,
                            fallback_occurred=(model_idx > 0),
                            attempted_models=attempts,
                            cited_ranks=cited,
                            citations=getattr(request.context_package, "citations", []) if request.context_package else [],
                            context_package=request.context_package,
                            latency_ms=round(total_ms, 1),
                            total_tokens=data.get("usage", {}).get("total_tokens"),
                            raw_response=data,
                        )

                    elif resp.status_code in (429, 500, 502, 503, 504):
                        model_error_msg = f"HTTP {resp.status_code}: {resp.text[:150]}"
                        logger.warning(
                            "OpenRouter model '%s' returned transient %s (attempt %d/%d). Retrying...",
                            model, resp.status_code, retries + 1, self.max_retries + 1
                        )
                        retries += 1
                        if retries <= self.max_retries:
                            time.sleep(0.6 * (2 ** (retries - 1)))
                        continue
                    else:
                        model_error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        logger.warning("OpenRouter model '%s' failed with permanent error: %s", model, model_error_msg)
                        break

                except (requests.Timeout, requests.ConnectionError) as req_err:
                    model_error_msg = f"Network/Timeout: {req_err}"
                    logger.warning("OpenRouter model '%s' network failure: %s (attempt %d/%d)", model, req_err, retries + 1, self.max_retries + 1)
                    retries += 1
                    if retries <= self.max_retries:
                        time.sleep(0.5 * retries)
                    continue

                except Exception as exc:
                    model_error_msg = f"Unexpected error: {exc}"
                    logger.error("Unexpected error querying OpenRouter model '%s': %s", model, exc)
                    break

            # Model failed after retries
            m_duration = (time.time() - m_start) * 1000.0
            attempts.append(
                ModelAttemptRecord(
                    model=model,
                    success=False,
                    error=model_error_msg,
                    retries_attempted=retries,
                    duration_ms=round(m_duration, 1),
                )
            )
            last_error = model_error_msg
            logger.info("Switching to next candidate model after '%s' failed.", model)

        total_ms = (time.time() - start_time) * 1000.0
        return GenerationResponse(
            query=request.query,
            answer=f"❌ **All candidate models failed to generate an answer.** Last error: {last_error}",
            model_used="exhausted",
            fallback_occurred=True,
            attempted_models=attempts,
            latency_ms=round(total_ms, 1),
        )

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream answer tokens with automatic model rotation if initial connection fails.

        Yields dictionary events:
        - `{"type": "model_selected", "model": str, "fallback": bool}`
        - `{"type": "token", "content": str}`
        - `{"type": "done", "model_used": str, "fallback": bool}`
        - `{"type": "error", "message": str}`
        """
        if not self.api_key:
            yield {"type": "token", "content": "⚠️ **OpenRouter API Key missing.** Set `OPENROUTER_API_KEY` in `RAG/.env`."}
            yield {"type": "done", "model_used": "none", "fallback": False}
            return

        models = self._get_ordered_models(request.primary_model)
        messages = self._build_messages(request)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/inx/enterprise-rag",
            "X-Title": "Enterprise RAG Pipeline",
        }

        for model_idx, model in enumerate(models):
            retries = 0
            connected = False

            while retries <= self.max_retries:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": True,
                }

                try:
                    resp = requests.post(
                        self.endpoint_url,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=self.timeout_sec,
                    )

                    if resp.status_code == 200:
                        connected = True
                        yield {
                            "type": "model_selected",
                            "model": model,
                            "fallback": (model_idx > 0),
                        }

                        # Stream response lines
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    token_part = line_str[6:].strip()
                                    if token_part == "[DONE]":
                                        break
                                    try:
                                        chunk_data = json.loads(token_part)
                                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content")
                                        if content:
                                            yield {"type": "token", "content": content}
                                    except Exception:
                                        continue

                        yield {"type": "done", "model_used": model, "fallback": (model_idx > 0)}
                        return

                    elif resp.status_code in (429, 500, 502, 503, 504):
                        retries += 1
                        logger.warning("Stream: model '%s' error %d. Retrying...", model, resp.status_code)
                        if retries <= self.max_retries:
                            time.sleep(0.5 * retries)
                        continue
                    else:
                        logger.warning("Stream: model '%s' error %d. Rotating...", model, resp.status_code)
                        break

                except Exception as exc:
                    logger.warning("Stream: exception with model '%s' (%s). Retrying...", model, exc)
                    retries += 1
                    if retries <= self.max_retries:
                        time.sleep(0.5 * retries)
                    continue

            # If not connected, rotate to next model
            logger.info("Stream: model '%s' exhausted candidate retries. Rotating...", model)

        yield {
            "type": "error",
            "message": "❌ All candidate models failed to connect or were rate limited.",
        }
