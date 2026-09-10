"""Query rewrite strategy implementations — both rule-based and LLM-based.

Rule-based strategies (no LLM cost):
- OriginalOnlyStrategy
- KeywordExtractionStrategy
- IdentifierExtractionStrategy

LLM-based strategies (bounded by token/timeout/cost):
- ExpansionStrategy
- MultiQueryStrategy
- HyDEStrategy
- DecompositionStrategy
- ConversationAwareStrategy
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from rag.query.models import (
    ConversationTurn,
    QueryAnalysis,
    RewriteStrategyType,
    RewrittenQuery,
)

logger = logging.getLogger(__name__)


# ── Strategy Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class RewriteStrategy(Protocol):
    """Protocol for query rewrite strategies."""

    strategy_type: RewriteStrategyType

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        """Generate rewritten query variants."""
        ...


# ── LLM Caller Protocol (minimal) ───────────────────────────────────────────

@runtime_checkable
class RewriteLLMCaller(Protocol):
    """Minimal protocol for LLM calls used by rewrite strategies."""

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> "LLMCallResult":
        ...


class LLMCallResult:
    """Result from an LLM call for query rewriting."""

    def __init__(
        self,
        text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms


class MockRewriteLLMCaller:
    """Mock LLM caller for testing and offline development."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}
        self.call_count = 0

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMCallResult:
        self.call_count += 1

        # Check for custom response matching
        for key, response in self._responses.items():
            if key.lower() in user_prompt.lower():
                return LLMCallResult(text=response, input_tokens=50, output_tokens=30)

        # Default: generate mock expansion
        return LLMCallResult(
            text=json.dumps({
                "queries": [
                    f"expanded: {user_prompt[:80]}",
                    f"rephrased: {user_prompt[:80]}",
                ]
            }),
            input_tokens=50,
            output_tokens=40,
        )


# ═══════════════════════════════════════════════════════════════════════════
# RULE-BASED STRATEGIES (no LLM cost)
# ═══════════════════════════════════════════════════════════════════════════

class OriginalOnlyStrategy:
    """Returns the original query unchanged. Used as default or fallback."""

    strategy_type = RewriteStrategyType.ORIGINAL_ONLY

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        return [
            RewrittenQuery(
                text=query,
                strategy=RewriteStrategyType.ORIGINAL_ONLY,
                confidence=1.0,
                is_original=True,
            )
        ]


class KeywordExtractionStrategy:
    """Builds a keyword-focused query variant from extracted terms.

    Produces a condensed query of just the top keywords, useful for
    BM25 matching where verbose natural language hurts precision.
    """

    strategy_type = RewriteStrategyType.KEYWORD_EXTRACTION

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        keywords = analysis.detected_keywords
        if not keywords:
            return []

        # Build keyword query
        keyword_query = " ".join(keywords[:8])

        # Only return if meaningfully different from original
        if keyword_query.lower().strip() == query.lower().strip():
            return []

        return [
            RewrittenQuery(
                text=keyword_query,
                strategy=RewriteStrategyType.KEYWORD_EXTRACTION,
                confidence=0.7,
                is_original=False,
                metadata={"keywords": keywords[:8]},
            )
        ]


class IdentifierExtractionStrategy:
    """Builds an exact-match query from extracted identifiers.

    Ensures identifiers like UUIDs, version strings, and codes
    are searched as exact terms, complementing the semantic search.
    """

    strategy_type = RewriteStrategyType.IDENTIFIER_EXTRACTION

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        identifiers = analysis.detected_identifiers
        if not identifiers:
            return []

        # Build identifier-focused query
        id_query = " ".join(identifiers)

        if id_query.lower().strip() == query.lower().strip():
            return []

        return [
            RewrittenQuery(
                text=id_query,
                strategy=RewriteStrategyType.IDENTIFIER_EXTRACTION,
                confidence=0.9,
                is_original=False,
                identifiers_preserved=True,
                metadata={"identifiers": identifiers},
            )
        ]


# ═══════════════════════════════════════════════════════════════════════════
# LLM-BASED STRATEGIES (bounded by token/timeout budgets)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_queries(raw_text: str, max_queries: int = 5) -> List[str]:
    """Parse JSON response from LLM containing query list.

    Supports formats:
    - {"queries": ["q1", "q2"]}
    - ["q1", "q2"]
    - Plain text with numbered lines
    """
    text = raw_text.strip()

    # Try JSON object with "queries" key
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("queries", "questions", "variants", "rewrites"):
                if key in obj and isinstance(obj[key], list):
                    return [str(q).strip() for q in obj[key] if str(q).strip()][:max_queries]
        if isinstance(obj, list):
            return [str(q).strip() for q in obj if str(q).strip()][:max_queries]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON array from mixed text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(q).strip() for q in arr if str(q).strip()][:max_queries]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to line-based parsing ONLY if lines look like numbered/bulleted lists
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^(?:\d+[\.\)]\s+|[-*•]\s+)", line):
            cleaned = re.sub(r"^(?:\d+[\.\)]\s+|[-*•]\s+)", "", line)
            cleaned = cleaned.strip().strip('"').strip("'")
            if cleaned and len(cleaned) > 5:
                lines.append(cleaned)

    return lines[:max_queries]


def _verify_identifiers(query: str, identifiers: List[str]) -> bool:
    """Check that all extracted identifiers appear in the rewritten query."""
    if not identifiers:
        return True
    lower_query = query.lower()
    return all(ident.lower() in lower_query for ident in identifiers)


def _reinject_identifiers(query: str, identifiers: List[str]) -> str:
    """Append missing identifiers to a rewritten query to prevent loss."""
    if not identifiers:
        return query
    lower_query = query.lower()
    missing = [ident for ident in identifiers if ident.lower() not in lower_query]
    if missing:
        return f"{query} {' '.join(missing)}"
    return query


class ExpansionStrategy:
    """Generates a single expanded query with synonyms, related terms, and clarifications."""

    strategy_type = RewriteStrategyType.EXPANSION

    _SYSTEM_PROMPT = (
        "You are a search query expansion expert. Given a user query, generate an expanded version "
        "that includes synonyms, related technical terms, and alternate phrasings to improve "
        "document retrieval. Do NOT change the meaning or add unsupported assumptions.\n\n"
        "Respond with JSON: {\"queries\": [\"expanded query\"]}"
    )

    def __init__(self, llm_caller: RewriteLLMCaller, max_tokens: int = 300):
        self._llm = llm_caller
        self._max_tokens = max_tokens

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        user_prompt = f"Expand this search query with related terms and synonyms:\n\n\"{query}\""

        try:
            result = self._llm.call(
                self._SYSTEM_PROMPT, user_prompt,
                max_tokens=self._max_tokens, temperature=0.3,
            )
            queries = _parse_json_queries(result.text, max_queries=2)
        except Exception as exc:
            logger.warning("Expansion strategy failed: %s", exc)
            return []

        rewrites: List[RewrittenQuery] = []
        for q_text in queries:
            if q_text.lower().strip() == query.lower().strip():
                continue
            # Reinject identifiers if lost
            q_text = _reinject_identifiers(q_text, analysis.detected_identifiers)
            rewrites.append(
                RewrittenQuery(
                    text=q_text,
                    strategy=RewriteStrategyType.EXPANSION,
                    confidence=0.7,
                    identifiers_preserved=_verify_identifiers(q_text, analysis.detected_identifiers),
                    metadata={"token_usage": result.total_tokens, "cost_usd": result.cost_usd},
                )
            )

        return rewrites


class MultiQueryStrategy:
    """Generates multiple semantically diverse query variants covering different angles."""

    strategy_type = RewriteStrategyType.MULTI_QUERY

    _SYSTEM_PROMPT = (
        "You are a search query diversification expert. Given a user query, generate 3 semantically "
        "diverse query variants that approach the topic from different angles. Each variant should "
        "retrieve complementary documents. Do NOT fabricate facts or introduce assumptions.\n\n"
        "Respond with JSON: {\"queries\": [\"variant 1\", \"variant 2\", \"variant 3\"]}"
    )

    def __init__(self, llm_caller: RewriteLLMCaller, max_queries: int = 3, max_tokens: int = 400):
        self._llm = llm_caller
        self._max_queries = max_queries
        self._max_tokens = max_tokens

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        user_prompt = (
            f"Generate {self._max_queries} diverse search query variants for:\n\n\"{query}\""
        )

        try:
            result = self._llm.call(
                self._SYSTEM_PROMPT, user_prompt,
                max_tokens=self._max_tokens, temperature=0.5,
            )
            queries = _parse_json_queries(result.text, max_queries=self._max_queries)
        except Exception as exc:
            logger.warning("Multi-query strategy failed: %s", exc)
            return []

        rewrites: List[RewrittenQuery] = []
        for q_text in queries:
            if q_text.lower().strip() == query.lower().strip():
                continue
            q_text = _reinject_identifiers(q_text, analysis.detected_identifiers)
            rewrites.append(
                RewrittenQuery(
                    text=q_text,
                    strategy=RewriteStrategyType.MULTI_QUERY,
                    confidence=0.65,
                    identifiers_preserved=_verify_identifiers(q_text, analysis.detected_identifiers),
                    metadata={"token_usage": result.total_tokens, "cost_usd": result.cost_usd},
                )
            )

        return rewrites


class HyDEStrategy:
    """Hypothetical Document Embedding — generates a hypothetical answer passage.

    The generated passage is used as a retrieval query, leveraging the insight
    that relevant documents resemble the expected answer more than the question.
    """

    strategy_type = RewriteStrategyType.HYDE

    _SYSTEM_PROMPT = (
        "You are a technical documentation expert. Given a user question, write a short hypothetical "
        "passage (2-3 sentences) that directly answers the question as if it were extracted from "
        "an authoritative document. Write factually and concisely. Do NOT include disclaimers.\n\n"
        "Respond with JSON: {\"queries\": [\"hypothetical passage\"]}"
    )

    def __init__(self, llm_caller: RewriteLLMCaller, max_tokens: int = 300):
        self._llm = llm_caller
        self._max_tokens = max_tokens

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        user_prompt = f"Write a hypothetical document passage that answers:\n\n\"{query}\""

        try:
            result = self._llm.call(
                self._SYSTEM_PROMPT, user_prompt,
                max_tokens=self._max_tokens, temperature=0.4,
            )
            passages = _parse_json_queries(result.text, max_queries=1)
        except Exception as exc:
            logger.warning("HyDE strategy failed: %s", exc)
            return []

        rewrites: List[RewrittenQuery] = []
        for passage in passages:
            if len(passage) < 20:
                continue
            passage = _reinject_identifiers(passage, analysis.detected_identifiers)
            rewrites.append(
                RewrittenQuery(
                    text=passage,
                    strategy=RewriteStrategyType.HYDE,
                    confidence=0.6,
                    identifiers_preserved=_verify_identifiers(passage, analysis.detected_identifiers),
                    metadata={
                        "type": "hypothetical_passage",
                        "token_usage": result.total_tokens,
                        "cost_usd": result.cost_usd,
                    },
                )
            )

        return rewrites


class DecompositionStrategy:
    """Decomposes complex multi-part questions into atomic sub-questions."""

    strategy_type = RewriteStrategyType.DECOMPOSITION

    _SYSTEM_PROMPT = (
        "You are a question decomposition expert. Given a complex question, break it down into "
        "2-3 simpler, self-contained sub-questions that together fully cover the original question. "
        "Each sub-question should be independently searchable.\n\n"
        "Respond with JSON: {\"queries\": [\"sub-question 1\", \"sub-question 2\"]}"
    )

    def __init__(self, llm_caller: RewriteLLMCaller, max_queries: int = 3, max_tokens: int = 400):
        self._llm = llm_caller
        self._max_queries = max_queries
        self._max_tokens = max_tokens

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        if not analysis.is_complex:
            return []

        user_prompt = f"Decompose this complex question into simpler sub-questions:\n\n\"{query}\""

        try:
            result = self._llm.call(
                self._SYSTEM_PROMPT, user_prompt,
                max_tokens=self._max_tokens, temperature=0.3,
            )
            queries = _parse_json_queries(result.text, max_queries=self._max_queries)
        except Exception as exc:
            logger.warning("Decomposition strategy failed: %s", exc)
            return []

        rewrites: List[RewrittenQuery] = []
        for q_text in queries:
            if q_text.lower().strip() == query.lower().strip():
                continue
            q_text = _reinject_identifiers(q_text, analysis.detected_identifiers)
            rewrites.append(
                RewrittenQuery(
                    text=q_text,
                    strategy=RewriteStrategyType.DECOMPOSITION,
                    confidence=0.7,
                    identifiers_preserved=_verify_identifiers(q_text, analysis.detected_identifiers),
                    metadata={"token_usage": result.total_tokens, "cost_usd": result.cost_usd},
                )
            )

        return rewrites


class ConversationAwareStrategy:
    """Resolves coreferences and implicit context from conversation history.

    Rewrites queries like "What about the second one?" into self-contained
    queries using prior conversation turns for context.
    """

    strategy_type = RewriteStrategyType.CONVERSATION_AWARE

    _SYSTEM_PROMPT = (
        "You are a conversation context resolver. Given a user's latest query and their "
        "conversation history, rewrite the query to be fully self-contained by resolving "
        "pronouns, references, and implicit context. The rewritten query should make sense "
        "without the conversation history.\n\n"
        "Respond with JSON: {\"queries\": [\"resolved query\"]}"
    )

    def __init__(self, llm_caller: RewriteLLMCaller, max_history_turns: int = 6, max_tokens: int = 300):
        self._llm = llm_caller
        self._max_history_turns = max_history_turns
        self._max_tokens = max_tokens

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[List[ConversationTurn]] = None,
    ) -> List[RewrittenQuery]:
        if not context:
            return []

        # Trim to recent history
        recent = context[-self._max_history_turns:]
        history_str = "\n".join(f"[{t.role}]: {t.content}" for t in recent)

        user_prompt = (
            f"Conversation history:\n{history_str}\n\n"
            f"Latest query: \"{query}\"\n\n"
            f"Rewrite the latest query to be self-contained:"
        )

        try:
            result = self._llm.call(
                self._SYSTEM_PROMPT, user_prompt,
                max_tokens=self._max_tokens, temperature=0.2,
            )
            queries = _parse_json_queries(result.text, max_queries=1)
        except Exception as exc:
            logger.warning("Conversation-aware strategy failed: %s", exc)
            return []

        rewrites: List[RewrittenQuery] = []
        for q_text in queries:
            if q_text.lower().strip() == query.lower().strip():
                continue
            q_text = _reinject_identifiers(q_text, analysis.detected_identifiers)
            rewrites.append(
                RewrittenQuery(
                    text=q_text,
                    strategy=RewriteStrategyType.CONVERSATION_AWARE,
                    confidence=0.75,
                    identifiers_preserved=_verify_identifiers(q_text, analysis.detected_identifiers),
                    metadata={
                        "history_turns": len(recent),
                        "token_usage": result.total_tokens,
                        "cost_usd": result.cost_usd,
                    },
                )
            )

        return rewrites


# ── Strategy Factory ─────────────────────────────────────────────────────────

def get_strategy(
    strategy_type: RewriteStrategyType,
    llm_caller: Optional[RewriteLLMCaller] = None,
    max_queries: int = 3,
) -> RewriteStrategy:
    """Create a strategy instance by type.

    Rule-based strategies don't require an LLM caller.
    LLM-based strategies require a valid caller or return OriginalOnlyStrategy as fallback.
    """
    if strategy_type == RewriteStrategyType.ORIGINAL_ONLY:
        return OriginalOnlyStrategy()

    if strategy_type == RewriteStrategyType.KEYWORD_EXTRACTION:
        return KeywordExtractionStrategy()

    if strategy_type == RewriteStrategyType.IDENTIFIER_EXTRACTION:
        return IdentifierExtractionStrategy()

    if llm_caller is None:
        logger.warning(
            "LLM-based strategy '%s' requested but no LLM caller available. Falling back to ORIGINAL_ONLY.",
            strategy_type.value,
        )
        return OriginalOnlyStrategy()

    if strategy_type == RewriteStrategyType.EXPANSION:
        return ExpansionStrategy(llm_caller)

    if strategy_type == RewriteStrategyType.MULTI_QUERY:
        return MultiQueryStrategy(llm_caller, max_queries=max_queries)

    if strategy_type == RewriteStrategyType.HYDE:
        return HyDEStrategy(llm_caller)

    if strategy_type == RewriteStrategyType.DECOMPOSITION:
        return DecompositionStrategy(llm_caller, max_queries=max_queries)

    if strategy_type == RewriteStrategyType.CONVERSATION_AWARE:
        return ConversationAwareStrategy(llm_caller)

    logger.warning("Unknown strategy type '%s'. Falling back to ORIGINAL_ONLY.", strategy_type.value)
    return OriginalOnlyStrategy()
