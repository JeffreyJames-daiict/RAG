"""Comprehensive test suite for query rewriting and expansion layer.

Covers all 15 required test areas:
1. Original query preservation
2. Exact identifier preservation
3. Rewrite failure handling & fallback
4. Empty rewrite handling
5. Multiple queries generation
6. Duplicate queries deduplication
7. Cost & max_queries limits
8. Timeout resilience
9. Conversation context & coreference resolution
10. Unanswerable queries handling
11. Strategy routing per query type
12. Identifier extraction regexes
13. Keyword extraction & stopword filtering
14. Config disabled bypass
15. Confidence threshold filtering
"""

import json
import pytest
from typing import List, Optional

from rag.core.config import QueryRewriteConfig, Settings
from rag.query.analyzer import QueryAnalyzer
from rag.query.models import (
    ConversationTurn,
    QueryAnalysis,
    QueryRewriteResult,
    QueryType,
    RewriteStrategyType,
    RewrittenQuery,
)
from rag.query.service import QueryRewriteService
from rag.query.strategies import (
    ConversationAwareStrategy,
    DecompositionStrategy,
    ExpansionStrategy,
    HyDEStrategy,
    IdentifierExtractionStrategy,
    KeywordExtractionStrategy,
    LLMCallResult,
    MockRewriteLLMCaller,
    MultiQueryStrategy,
    OriginalOnlyStrategy,
    get_strategy,
)


class ErrorThrowingLLMCaller:
    """Mock LLM caller that deliberately throws an exception."""

    def __init__(self, exc: Optional[Exception] = None):
        self.exc = exc or RuntimeError("Simulated upstream LLM failure")

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMCallResult:
        raise self.exc


class EmptyResponseLLMCaller:
    """Mock LLM caller that returns empty text or invalid format."""

    def __init__(self, text: str = ""):
        self.text = text

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMCallResult:
        return LLMCallResult(text=self.text, input_tokens=10, output_tokens=0)


class CustomResponseLLMCaller:
    """Mock LLM caller that returns a pre-configured response."""

    def __init__(self, response_text: str, total_tokens: int = 50, cost_usd: float = 0.001):
        self.response_text = response_text
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMCallResult:
        return LLMCallResult(
            text=self.response_text,
            input_tokens=25,
            output_tokens=25,
            cost_usd=self.cost_usd,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1. ORIGINAL QUERY PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════

class TestOriginalQueryPreservation:
    def test_original_always_first_and_preserved(self):
        caller = MockRewriteLLMCaller()
        service = QueryRewriteService(llm_caller=caller, enabled=True)
        query = "What is the capital of France?"

        result = service.rewrite(query)

        assert len(result.rewritten_queries) >= 1
        first_q = result.rewritten_queries[0]
        assert first_q.text == query
        assert first_q.is_original is True
        assert first_q.strategy == RewriteStrategyType.ORIGINAL_ONLY
        assert first_q.confidence == 1.0
        assert first_q.identifiers_preserved is True

    def test_original_only_strategy_returns_single_original(self):
        strategy = OriginalOnlyStrategy()
        analysis = QueryAnalysis(original_query="hello world")
        res = strategy.rewrite("hello world", analysis)
        assert len(res) == 1
        assert res[0].text == "hello world"
        assert res[0].is_original is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXACT IDENTIFIER PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════

class TestExactIdentifierPreservation:
    def test_extract_and_preserve_uuid_and_version(self):
        uuid_val = "123e4567-e89b-12d3-a456-426614174000"
        query = f"Find the status of document {uuid_val} in release v2.4.1"

        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(query)

        assert uuid_val in analysis.detected_identifiers
        assert "v2.4.1" in analysis.detected_identifiers
        assert analysis.query_type == QueryType.IDENTIFIER_LOOKUP

        # Ensure identifier extraction strategy includes them
        strat = IdentifierExtractionStrategy()
        rewrites = strat.rewrite(query, analysis)
        assert len(rewrites) >= 1
        assert uuid_val in rewrites[0].text
        assert "v2.4.1" in rewrites[0].text

    def test_llm_rewrite_reinjects_missing_identifiers(self):
        caller = CustomResponseLLMCaller(
            json.dumps({"queries": ["paraphrased query with no ids"]})
        )
        service = QueryRewriteService(llm_caller=caller, enabled=True)
        query = "What is error ERR-404 in gemini-2.5-flash?"

        result = service.rewrite(query)

        # All rewrites should preserve or reinject identifiers
        for rw in result.rewritten_queries:
            if not rw.is_original:
                assert "ERR-404" in rw.text
                assert "gemini-2.5-flash" in rw.text


# ═══════════════════════════════════════════════════════════════════════════
# 3. REWRITE FAILURE HANDLING & FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class TestRewriteFailureFallback:
    def test_llm_exception_falls_back_to_original(self):
        caller = ErrorThrowingLLMCaller(RuntimeError("Connection timeout"))
        service = QueryRewriteService(llm_caller=caller, fallback_enabled=True)
        query = "How to install PostgreSQL on Ubuntu?"

        result = service.rewrite(query)

        assert result.fallback_used is True
        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].text == query
        assert result.rewritten_queries[0].is_original is True
        assert any("failed" in w.lower() for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# 4. EMPTY REWRITE HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyRewriteHandling:
    def test_empty_string_llm_response(self):
        caller = EmptyResponseLLMCaller("")
        service = QueryRewriteService(llm_caller=caller)
        query = "Explain backpropagation in neural networks"

        result = service.rewrite(query)

        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].text == query
        assert result.rewritten_queries[0].is_original is True

    def test_invalid_json_llm_response_falls_back(self):
        caller = EmptyResponseLLMCaller("This is not valid JSON at all")
        service = QueryRewriteService(llm_caller=caller)
        query = "Explain transformer self-attention"

        result = service.rewrite(query)

        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].text == query


# ═══════════════════════════════════════════════════════════════════════════
# 5. MULTIPLE QUERIES GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMultipleQueriesGeneration:
    def test_multi_query_strategy_produces_variants(self):
        caller = CustomResponseLLMCaller(
            json.dumps({
                "queries": [
                    "Kubernetes architecture overview",
                    "How Kubernetes components communicate",
                    "Kubernetes master and worker nodes explanation",
                ]
            })
        )
        service = QueryRewriteService(llm_caller=caller, max_queries=4)
        query = "Explain Kubernetes architecture"

        result = service.rewrite(query)

        assert len(result.rewritten_queries) == 4
        assert result.rewritten_queries[0].is_original is True
        assert len(result.query_texts) == 4


# ═══════════════════════════════════════════════════════════════════════════
# 6. DUPLICATE QUERIES DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDuplicateQueriesDeduplication:
    def test_exact_and_case_insensitive_duplicates_removed(self):
        caller = CustomResponseLLMCaller(
            json.dumps({
                "queries": [
                    "What is docker?",  # duplicate of original case-insensitive
                    "WHAT IS DOCKER?",  # duplicate
                    "Docker container basics",
                    "docker container basics",  # duplicate of above
                ]
            })
        )
        service = QueryRewriteService(llm_caller=caller, max_queries=5)
        query = "What is Docker?"

        result = service.rewrite(query)

        texts = [q.text.lower().strip() for q in result.rewritten_queries]
        assert len(texts) == len(set(texts))  # All unique
        assert len(result.rewritten_queries) == 2  # Original + 1 unique variant


# ═══════════════════════════════════════════════════════════════════════════
# 7. COST & MAX_QUERIES LIMITS
# ═══════════════════════════════════════════════════════════════════════════

class TestCostAndMaxQueriesLimits:
    def test_max_queries_cap_enforced(self):
        caller = CustomResponseLLMCaller(
            json.dumps({
                "queries": [f"Variant {i}" for i in range(10)]
            })
        )
        service = QueryRewriteService(llm_caller=caller, max_queries=3)
        query = "Test query for cap"

        result = service.rewrite(query)

        assert len(result.rewritten_queries) == 3
        assert result.rewritten_queries[0].is_original is True

    def test_token_and_cost_aggregation(self):
        caller = CustomResponseLLMCaller(
            json.dumps({"queries": ["Variant 1"]}),
            total_tokens=120,
            cost_usd=0.0024,
        )
        service = QueryRewriteService(llm_caller=caller)
        query = "Compare Python and Rust"

        result = service.rewrite(query)

        assert result.token_usage > 0
        assert result.cost_usd > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 8. TIMEOUT RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeoutResilience:
    def test_timeout_fallback_to_original(self):
        caller = ErrorThrowingLLMCaller(TimeoutError("Rewrite request exceeded 5.0s"))
        service = QueryRewriteService(llm_caller=caller, timeout_sec=1.0)
        query = "Database index types and performance"

        result = service.rewrite(query)

        assert result.fallback_used is True
        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].text == query


# ═══════════════════════════════════════════════════════════════════════════
# 9. CONVERSATION CONTEXT & COREFERENCE RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

class TestConversationContext:
    def test_conversation_aware_rewriting(self):
        history = [
            ConversationTurn(role="user", content="How does PostgreSQL handle concurrency?"),
            ConversationTurn(role="assistant", content="PostgreSQL uses Multi-Version Concurrency Control (MVCC)."),
        ]
        caller = CustomResponseLLMCaller(
            json.dumps({"queries": ["What are the limitations of PostgreSQL MVCC concurrency?"]})
        )
        service = QueryRewriteService(llm_caller=caller)
        query = "What are its limitations?"

        result = service.rewrite(query, conversation_history=history)

        assert len(result.rewritten_queries) >= 2
        rewritten_variant = result.rewritten_queries[1]
        assert "PostgreSQL" in rewritten_variant.text
        assert rewritten_variant.strategy == RewriteStrategyType.CONVERSATION_AWARE


# ═══════════════════════════════════════════════════════════════════════════
# 10. UNANSWERABLE QUERIES HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestUnanswerableQueries:
    def test_unanswerable_query_routes_to_original_only(self):
        analyzer = QueryAnalyzer()
        queries = [
            "asdfghjk qwerty",
            "???",
            "12345 67890",
        ]
        for q in queries:
            analysis = analyzer.analyze(q)
            assert analysis.query_type == QueryType.UNANSWERABLE
            assert RewriteStrategyType.ORIGINAL_ONLY in analysis.suggested_strategies


# ═══════════════════════════════════════════════════════════════════════════
# 11. STRATEGY ROUTING PER QUERY TYPE
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyRouting:
    @pytest.mark.parametrize(
        "query,expected_type,expected_strat",
        [
            ("What is ACID in database systems?", QueryType.FACTUAL, RewriteStrategyType.EXPANSION),
            ("Why does GIL exist in Python?", QueryType.ANALYTICAL, RewriteStrategyType.MULTI_QUERY),
            ("Compare B-tree vs LSM-tree indexing", QueryType.COMPARISON, RewriteStrategyType.DECOMPOSITION),
            ("How to configure SSL in Nginx?", QueryType.PROCEDURAL, RewriteStrategyType.MULTI_QUERY),
            ("Where is the user configuration file?", QueryType.NAVIGATIONAL, RewriteStrategyType.KEYWORD_EXTRACTION),
            ("Lookup DOC-99482 in system", QueryType.IDENTIFIER_LOOKUP, RewriteStrategyType.IDENTIFIER_EXTRACTION),
        ],
    )
    def test_query_routing_classification(self, query, expected_type, expected_strat):
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(query)
        assert analysis.query_type == expected_type
        assert expected_strat in analysis.suggested_strategies


# ═══════════════════════════════════════════════════════════════════════════
# 12. IDENTIFIER EXTRACTION REGEXES
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentifierExtractionRegexes:
    def test_extract_all_identifier_formats(self):
        analyzer = QueryAnalyzer()
        query = (
            "Check https://example.com/docs/api for commit a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4 "
            "with UUID 550e8400-e29b-41d4-a716-446655440000 and file /etc/nginx/nginx.conf "
            "using claude-3.5-sonnet in v3.2.1 ticket JIRA-5432"
        )
        analysis = analyzer.analyze(query)

        ids = analysis.detected_identifiers
        assert "550e8400-e29b-41d4-a716-446655440000" in ids
        assert "/etc/nginx/nginx.conf" in ids
        assert "claude-3.5-sonnet" in ids
        assert "v3.2.1" in ids
        assert "JIRA-5432" in ids
        assert "https://example.com/docs/api" in ids
        assert "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4" in ids


# ═══════════════════════════════════════════════════════════════════════════
# 13. KEYWORD EXTRACTION & STOPWORD FILTERING
# ═══════════════════════════════════════════════════════════════════════════

class TestKeywordExtraction:
    def test_keyword_extraction_filters_stopwords(self):
        analyzer = QueryAnalyzer()
        query = "What is the best way to optimize database queries in PostgreSQL?"
        analysis = analyzer.analyze(query)

        keywords = analysis.detected_keywords
        assert len(keywords) > 0
        # Stopwords like 'what', 'is', 'the', 'to', 'in' should NOT be keywords
        for stopword in ["what", "is", "the", "to", "in"]:
            assert stopword not in keywords
        # Domain terms should be included
        assert any(k in ["postgresql", "optimize", "database", "queries"] for k in keywords)

    def test_keyword_extraction_strategy(self):
        strat = KeywordExtractionStrategy()
        analysis = QueryAnalysis(
            original_query="What is the speed of vector search in Qdrant?",
            detected_keywords=["speed", "vector", "search", "qdrant"],
        )
        rewrites = strat.rewrite("What is the speed of vector search in Qdrant?", analysis)
        assert len(rewrites) == 1
        assert "vector" in rewrites[0].text
        assert "qdrant" in rewrites[0].text


# ═══════════════════════════════════════════════════════════════════════════
# 14. CONFIG DISABLED BYPASS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigDisabledBypass:
    def test_disabled_service_bypasses_layer(self):
        caller = MockRewriteLLMCaller()
        service = QueryRewriteService(llm_caller=caller, enabled=False)
        query = "What is Raft consensus algorithm?"

        result = service.rewrite(query)

        assert caller.call_count == 0  # No LLM calls
        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].text == query
        assert result.rewritten_queries[0].is_original is True
        assert result.latency_ms == 0.0

    def test_settings_integration(self):
        settings = Settings()
        assert hasattr(settings, "query_rewrite")
        assert isinstance(settings.query_rewrite, QueryRewriteConfig)
        assert settings.query_rewrite.enabled is False
        assert settings.query_rewrite.max_queries == 5


# ═══════════════════════════════════════════════════════════════════════════
# 15. CONFIDENCE THRESHOLD FILTERING
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceThresholdFiltering:
    def test_low_confidence_rewrites_dropped(self):
        # We test strategy output filtering by setting confidence threshold higher than confidence
        caller = CustomResponseLLMCaller(
            json.dumps({"queries": ["low confidence candidate"]})
        )
        # ExpansionStrategy sets confidence=0.7 by default in strategies.py
        # If threshold is 0.85, it should be dropped
        service = QueryRewriteService(
            llm_caller=caller,
            confidence_threshold=0.85,
        )
        query = "Explain linear regression"

        result = service.rewrite(query)

        # Only original query should remain
        assert len(result.rewritten_queries) == 1
        assert result.rewritten_queries[0].is_original is True
        assert any("dropped low-confidence" in w.lower() for w in result.warnings)
