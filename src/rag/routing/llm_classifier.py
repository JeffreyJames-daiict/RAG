"""LLM-based query classifier for query routing."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from rag.routing.models import (
    ExtractedFeatures,
    QueryType,
    RouterType,
    RouteTarget,
    RoutingDecision,
)
from rag.routing.rule_classifier import RuleBasedQueryClassifier

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMRouterCaller(Protocol):
    """Protocol for LLM router callers."""

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> Any:
        ...


class MockLLMRouterCaller:
    """Mock LLM caller for router testing and offline development."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}
        self.call_count = 0

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> Any:
        self.call_count += 1
        for key, resp in self._responses.items():
            if key.lower() in user_prompt.lower():
                return type(
                    "CallResult",
                    (),
                    {
                        "text": resp,
                        "total_tokens": 40,
                        "cost_usd": 0.0002,
                    },
                )()

        # Default classification response
        default_resp = json.dumps({
            "query_type": "general_semantic",
            "route": "hybrid_rrf",
            "confidence": 0.85,
            "reasoning": "Standard semantic inquiry",
            "target_documents": [],
        })
        return type(
            "CallResult",
            (),
            {
                "text": default_resp,
                "total_tokens": 35,
                "cost_usd": 0.0001,
            },
        )()


class LLMQueryClassifier:
    """LLM-assisted query classifier with structured JSON output and cost gating."""

    _SYSTEM_PROMPT = (
        "You are an enterprise query routing classifier for an information retrieval system.\n"
        "Classify the user's query into one query type and its corresponding optimal route.\n\n"
        "Allowed Query Types:\n"
        "- general_semantic -> route: hybrid_rrf\n"
        "- exact_identifier -> route: bm25_first\n"
        "- metadata -> route: metadata_lookup\n"
        "- table -> route: table_aware\n"
        "- figure -> route: multimodal\n"
        "- summarization -> route: summarization\n"
        "- document_specific -> route: document_filtered\n"
        "- multi_hop -> route: agentic_workflow\n"
        "- unanswerable -> route: direct_abstain\n\n"
        "Respond ONLY with a valid JSON object in this exact schema:\n"
        "{\n"
        '  "query_type": "general_semantic",\n'
        '  "route": "hybrid_rrf",\n'
        '  "confidence": 0.9,\n'
        '  "reasoning": "short justification",\n'
        '  "target_documents": ["optional_doc.pdf"]\n'
        "}"
    )

    def __init__(
        self,
        llm_caller: Optional[LLMRouterCaller] = None,
        max_cost: float = 0.01,
        fallback_classifier: Optional[RuleBasedQueryClassifier] = None,
    ):
        self._llm = llm_caller
        self._max_cost = max_cost
        self._fallback = fallback_classifier or RuleBasedQueryClassifier()

    def classify(self, query: str) -> RoutingDecision:
        """Classify query using LLM, with fallback to rule classifier on error or cost overage."""
        start_time = time.perf_counter()
        clean_query = query.strip()

        if self._llm is None:
            # Fall back to rule-based directly
            return self._fallback.classify(clean_query)

        user_prompt = f"Query: \"{clean_query}\""

        try:
            call_res = self._llm.call(
                self._SYSTEM_PROMPT,
                user_prompt,
                max_tokens=150,
                temperature=0.0,
            )
            cost = getattr(call_res, "cost_usd", 0.0)
            if cost > self._max_cost:
                logger.warning("LLM router call cost $%.4f exceeded max cost $%.4f", cost, self._max_cost)
                decision = self._fallback.classify(clean_query)
                decision.fallback_used = True
                decision.metadata["warning"] = "LLM cost budget exceeded, fell back to rules"
                return decision

            text = getattr(call_res, "text", "").strip()
            decision = self._parse_response(clean_query, text, cost, start_time)
            return decision

        except Exception as exc:
            logger.warning("LLM routing failed: %s. Falling back to rule-based.", exc)
            decision = self._fallback.classify(clean_query)
            decision.fallback_used = True
            decision.metadata["warning"] = f"LLM routing failed: {exc}"
            return decision

    def _parse_response(
        self,
        query: str,
        raw_text: str,
        cost: float,
        start_time: float,
    ) -> RoutingDecision:
        """Parse LLM JSON response into validated RoutingDecision."""
        latency = (time.perf_counter() - start_time) * 1000

        # Try extract JSON
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            decision = self._fallback.classify(query)
            decision.fallback_used = True
            decision.metadata["warning"] = "LLM output did not contain valid JSON"
            return decision

        try:
            data = json.loads(json_match.group(0))
        except Exception:
            decision = self._fallback.classify(query)
            decision.fallback_used = True
            decision.metadata["warning"] = "Malformed JSON from LLM router"
            return decision

        raw_qtype = str(data.get("query_type", "")).lower()
        raw_route = str(data.get("route", "")).lower()
        confidence = float(data.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(data.get("reasoning", "Classified by LLM"))
        target_docs = [str(d) for d in data.get("target_documents", []) if d]

        # Validate enums
        try:
            query_type = QueryType(raw_qtype)
        except ValueError:
            query_type = QueryType.GENERAL_SEMANTIC

        try:
            route = RouteTarget(raw_route)
        except ValueError:
            route = RouteTarget.HYBRID_RRF

        features = self._fallback.extract_features(query)

        return RoutingDecision(
            query=query,
            route=route,
            query_type=query_type,
            confidence=round(confidence, 2),
            fallback_used=False,
            router_type=RouterType.LLM,
            features=features,
            target_document_ids=target_docs or features.target_documents,
            latency_ms=round(latency, 2),
            cost_usd=round(cost, 6),
            metadata={"reason": reasoning},
        )
