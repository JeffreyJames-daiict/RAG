"""Rule-based query analyzer for type detection, identifier extraction, keyword extraction, and strategy routing.

This module performs deterministic analysis of user queries without LLM calls.
All classification, extraction, and strategy selection is based on regex patterns and heuristics.
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from rag.query.models import (
    ConversationTurn,
    QueryAnalysis,
    QueryType,
    RewriteStrategyType,
)

logger = logging.getLogger(__name__)

# ── Identifier Patterns ──────────────────────────────────────────────────────

# UUID variants (standard and hex-only)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Version strings: v1.2.3, v2, V1.0
_VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE)

# File paths: /path/to/file.ext, ./relative/path
_FILE_PATH_PATTERN = re.compile(r"(?:(?:/|\./)[\w\-./]+\.\w+)")

# URLs
_URL_PATTERN = re.compile(r"https?://[^\s,\)]+")

# Numeric codes: PROJ-1234, ERR-500, DOC_001
_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{1,10}[-_]\d{2,}\b")

# Model names: gpt-4o, claude-3.5-sonnet, gemini-2.5-flash, llama-3.3-70b
_MODEL_NAME_PATTERN = re.compile(
    r"\b(?:gpt-[\w.-]+|claude-[\w.-]+|gemini-[\w.-]+|llama-[\w.-]+|"
    r"mistral-[\w.-]+|deepseek-[\w.-]+|text-embedding-[\w.-]+)\b",
    re.IGNORECASE,
)

# Hex hashes: sha256, md5, commit shas
_HEX_HASH_PATTERN = re.compile(r"\b[0-9a-f]{32,64}\b")

ALL_IDENTIFIER_PATTERNS = [
    _UUID_PATTERN,
    _VERSION_PATTERN,
    _FILE_PATH_PATTERN,
    _URL_PATTERN,
    _CODE_PATTERN,
    _MODEL_NAME_PATTERN,
    _HEX_HASH_PATTERN,
]

# ── Query Type Detection Patterns ────────────────────────────────────────────

_FACTUAL_STARTERS = re.compile(
    r"^(?:what|which|who|where|when|name|define|list)\b", re.IGNORECASE
)

_ANALYTICAL_STARTERS = re.compile(
    r"^(?:why|how does|how do|how can|explain|analyze|describe|what causes)\b",
    re.IGNORECASE,
)

_COMPARISON_MARKERS = re.compile(
    r"\b(?:compare|versus|vs\.?|differ(?:ence|ent)|between|better|worse|"
    r"advantage|disadvantage|pros and cons|trade-?off)\b",
    re.IGNORECASE,
)

_PROCEDURAL_STARTERS = re.compile(
    r"^(?:how to|how do I|steps to|guide|tutorial|setup|install|configure|"
    r"create|build|implement|deploy|migrate)\b",
    re.IGNORECASE,
)

_NAVIGATIONAL_MARKERS = re.compile(
    r"\b(?:find|locate|show me|where is|where are|go to|navigate|look up|lookup|search for|get|fetch)\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_MARKERS = re.compile(
    r"\b(?:you said|you mentioned|earlier|previous|above|that|this|it|they|"
    r"the same|follow up|more about|can you also|what about)\b",
    re.IGNORECASE,
)

_COMPLEX_MARKERS = re.compile(
    r"\b(?:and also|additionally|furthermore|in addition|as well as|"
    r"both .+ and|first .+ then|not only .+ but)\b",
    re.IGNORECASE,
)

_UNANSWERABLE_MARKERS = re.compile(
    r"\b(?:predict the future|what will happen in 20\d{2}|"
    r"personal opinion|your favorite|how do you feel)\b",
    re.IGNORECASE,
)

# ── Stopwords ────────────────────────────────────────────────────────────────

ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "because", "until", "while",
    "about", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "it", "its", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "i",
}


class QueryAnalyzer:
    """Deterministic rule-based query analysis engine.

    Classifies query type, extracts identifiers and keywords, detects conversation
    context, and routes to appropriate rewrite strategies. No LLM calls.
    """

    def __init__(
        self,
        max_keywords: int = 10,
        min_keyword_length: int = 3,
    ):
        self.max_keywords = max_keywords
        self.min_keyword_length = min_keyword_length

    def analyze(
        self,
        query: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
        *,
        enable_expansion: bool = True,
        enable_multi_query: bool = True,
        enable_hyde: bool = False,
        enable_decomposition: bool = True,
        enable_conversation: bool = True,
    ) -> QueryAnalysis:
        """Analyze a query and produce routing recommendations.

        Args:
            query: The user's raw query text.
            conversation_history: Optional prior conversation turns.
            enable_*: Feature flags from config controlling which strategies are eligible.

        Returns:
            QueryAnalysis with type, identifiers, keywords, and strategy suggestions.
        """
        stripped = query.strip()
        words = stripped.split()
        word_count = len(words)

        # 1. Extract identifiers
        identifiers = self._extract_identifiers(stripped)

        # 2. Extract keywords
        keywords = self._extract_keywords(stripped)

        # 3. Detect query type
        query_type, confidence = self._classify_type(stripped, identifiers, conversation_history)

        # 4. Detect conversational context
        is_conversational = self._is_conversational(stripped, conversation_history)

        # 5. Detect complexity
        is_complex = bool(_COMPLEX_MARKERS.search(stripped)) or stripped.count("?") > 1

        # 6. Detect language (basic heuristic)
        language = self._detect_language(stripped)

        # 7. Route to strategies
        strategies = self._select_strategies(
            query_type=query_type,
            is_conversational=is_conversational,
            is_complex=is_complex,
            has_identifiers=bool(identifiers),
            word_count=word_count,
            enable_expansion=enable_expansion,
            enable_multi_query=enable_multi_query,
            enable_hyde=enable_hyde,
            enable_decomposition=enable_decomposition,
            enable_conversation=enable_conversation,
        )

        return QueryAnalysis(
            original_query=stripped,
            query_type=query_type,
            detected_identifiers=identifiers,
            detected_keywords=keywords,
            language=language,
            is_conversational=is_conversational,
            confidence=confidence,
            suggested_strategies=strategies,
            is_complex=is_complex,
            word_count=word_count,
        )

    def _extract_identifiers(self, query: str) -> List[str]:
        """Extract exact identifiers from the query (UUIDs, versions, paths, codes, model names)."""
        found: List[str] = []
        for pattern in ALL_IDENTIFIER_PATTERNS:
            matches = pattern.findall(query)
            found.extend(matches)
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for ident in found:
            if ident.lower() not in seen:
                seen.add(ident.lower())
                deduped.append(ident)
        return deduped

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract top-k domain keywords using stopword filtering and frequency counting."""
        # Tokenize: split on non-alphanumeric, keep hyphens in compound words
        tokens = re.findall(r"[a-zA-Z][\w-]*[a-zA-Z]|[a-zA-Z]{2,}", query.lower())

        # Filter stopwords and short tokens
        meaningful = [
            t for t in tokens
            if t not in ENGLISH_STOPWORDS and len(t) >= self.min_keyword_length
        ]

        # Count and rank
        counter = Counter(meaningful)
        top_keywords = [kw for kw, _ in counter.most_common(self.max_keywords)]
        return top_keywords

    def _classify_type(
        self,
        query: str,
        identifiers: List[str],
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> Tuple[QueryType, float]:
        """Classify query type using pattern matching."""
        # Priority-ordered classification
        if _UNANSWERABLE_MARKERS.search(query):
            return QueryType.UNANSWERABLE, 0.8

        alpha_chars = [c for c in query.lower() if c.isalpha()]
        if not alpha_chars or re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", query.lower()):
            return QueryType.UNANSWERABLE, 0.9

        if identifiers:
            if len(identifiers) >= 2 or _NAVIGATIONAL_MARKERS.search(query) or re.search(r"\b(?:lookup|fetch|get|status|doc|code|file)\b", query, re.IGNORECASE) or len(query.split()) <= 6:
                return QueryType.IDENTIFIER_LOOKUP, 0.9

        if self._is_conversational(query, conversation_history):
            return QueryType.CONVERSATIONAL, 0.7

        if _COMPARISON_MARKERS.search(query):
            return QueryType.COMPARISON, 0.8

        if _PROCEDURAL_STARTERS.search(query):
            return QueryType.PROCEDURAL, 0.8

        if _ANALYTICAL_STARTERS.search(query):
            return QueryType.ANALYTICAL, 0.75

        if _NAVIGATIONAL_MARKERS.search(query):
            return QueryType.NAVIGATIONAL, 0.7

        if _FACTUAL_STARTERS.search(query):
            return QueryType.FACTUAL, 0.7

        # Default: factual with lower confidence
        return QueryType.FACTUAL, 0.5

    def _is_conversational(
        self,
        query: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> bool:
        """Detect if query references conversation history."""
        if conversation_history and len(conversation_history) > 0:
            if _CONVERSATIONAL_MARKERS.search(query):
                return True
            # Very short queries with history are likely follow-ups
            if len(query.split()) <= 5 and len(conversation_history) >= 2:
                return True
        return False

    def _detect_language(self, query: str) -> str:
        """Basic language detection heuristic."""
        ascii_chars = sum(1 for c in query if c.isascii())
        ratio = ascii_chars / max(len(query), 1)
        if ratio < 0.7:
            return "unknown"
        return "en"

    def _select_strategies(
        self,
        query_type: QueryType,
        is_conversational: bool,
        is_complex: bool,
        has_identifiers: bool,
        word_count: int,
        enable_expansion: bool,
        enable_multi_query: bool,
        enable_hyde: bool,
        enable_decomposition: bool,
        enable_conversation: bool,
    ) -> List[RewriteStrategyType]:
        """Select appropriate strategies based on query analysis and config flags."""
        strategies: List[RewriteStrategyType] = []

        # Conversational queries need coreference resolution first
        if is_conversational and enable_conversation:
            strategies.append(RewriteStrategyType.CONVERSATION_AWARE)

        # Complex queries benefit from decomposition
        if is_complex and enable_decomposition:
            strategies.append(RewriteStrategyType.DECOMPOSITION)

        # Strategy routing based on query type
        if query_type == QueryType.UNANSWERABLE:
            # Don't expand unanswerable queries
            return [RewriteStrategyType.ORIGINAL_ONLY]

        if query_type == QueryType.IDENTIFIER_LOOKUP:
            strategies.append(RewriteStrategyType.IDENTIFIER_EXTRACTION)
            if enable_expansion:
                strategies.append(RewriteStrategyType.KEYWORD_EXTRACTION)
            return strategies if strategies else [RewriteStrategyType.ORIGINAL_ONLY]

        if query_type == QueryType.COMPARISON:
            if enable_decomposition:
                strategies.append(RewriteStrategyType.DECOMPOSITION)
            if enable_multi_query:
                strategies.append(RewriteStrategyType.MULTI_QUERY)
            if enable_expansion:
                strategies.append(RewriteStrategyType.EXPANSION)

        elif query_type == QueryType.ANALYTICAL:
            if enable_multi_query:
                strategies.append(RewriteStrategyType.MULTI_QUERY)
            if enable_expansion:
                strategies.append(RewriteStrategyType.EXPANSION)
            if enable_hyde:
                strategies.append(RewriteStrategyType.HYDE)

        elif query_type == QueryType.PROCEDURAL:
            if enable_expansion:
                strategies.append(RewriteStrategyType.EXPANSION)
            if enable_multi_query:
                strategies.append(RewriteStrategyType.MULTI_QUERY)

        elif query_type == QueryType.NAVIGATIONAL:
            strategies.append(RewriteStrategyType.KEYWORD_EXTRACTION)
            if enable_expansion:
                strategies.append(RewriteStrategyType.EXPANSION)

        else:
            # FACTUAL, CONVERSATIONAL (when not handled above)
            if enable_multi_query:
                strategies.append(RewriteStrategyType.MULTI_QUERY)
            if enable_expansion:
                strategies.append(RewriteStrategyType.EXPANSION)
            if enable_hyde:
                strategies.append(RewriteStrategyType.HYDE)

        # Short factual queries can optionally have keyword extraction appended at the end
        if word_count <= 4 and not is_conversational and query_type not in (QueryType.IDENTIFIER_LOOKUP, QueryType.UNANSWERABLE):
            if RewriteStrategyType.KEYWORD_EXTRACTION not in strategies:
                strategies.append(RewriteStrategyType.KEYWORD_EXTRACTION)

        return strategies if strategies else [RewriteStrategyType.ORIGINAL_ONLY]
