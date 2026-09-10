"""Comprehensive test suite for query routing layer.

Covers:
- Exact identifiers (patents, UUIDs, version strings, codes)
- Table queries
- Figure and diagram queries
- Metadata queries
- General semantic queries
- Summarization queries
- Document-specific queries
- Multi-hop queries
- Unanswerable / gibberish queries
- Ambiguous queries and low-confidence fallback
- Router failure handling and fallback
- LLM router and hybrid router execution
- Tenant-safe metadata
- Telemetry metrics collection
- Golden evaluation dataset benchmark
"""

import json
from pathlib import Path
import pytest

from rag.core.config import QueryRoutingConfig, Settings
from rag.routing import (
    ExtractedFeatures,
    LLMQueryClassifier,
    MockLLMRouterCaller,
    QueryRouter,
    QueryType,
    RouterType,
    RouteTarget,
    RoutingDecision,
    RoutingMetricsCollector,
    RuleBasedQueryClassifier,
)


class ErrorThrowingClassifier(RuleBasedQueryClassifier):
    """Classifier that raises an unhandled exception for failure testing."""

    def classify(self, query: str) -> RoutingDecision:
        raise RuntimeError("Simulated internal classifier crash")


class ErrorThrowingLLMCaller:
    """LLM caller that raises an exception for failure testing."""

    def call(self, system_prompt: str, user_prompt: str, **kwargs):
        raise TimeoutError("Upstream LLM timeout exceeded")


# ═══════════════════════════════════════════════════════════════════════════
# 1. EXACT IDENTIFIERS ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestExactIdentifierRouting:
    @pytest.mark.parametrize(
        "query,expected_identifier",
        [
            ("Lookup patent US9876543A in repository", "US9876543A"),
            ("Patent EP1234567 details", "EP1234567"),
            ("Find record 550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000"),
            ("What causes ERR-500 in auth service?", "ERR-500"),
            ("Status of PROJ-4412", "PROJ-4412"),
            ("Changelog for release v1.8.0", "v1.8.0"),
        ],
    )
    def test_identifiers_route_to_bm25_first(self, query, expected_identifier):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.EXACT_IDENTIFIER
        assert decision.route == RouteTarget.BM25_FIRST
        assert decision.confidence >= 0.8
        assert expected_identifier in decision.features.detected_identifiers
        assert decision.fallback_used is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. TABLE QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestTableQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "What is the total revenue in the Q3 financial table?",
            "Which column lists operating expenses in table 2?",
            "Compare row 4 and row 8 in the balance sheet",
            "What is the metric percentage in the data sheet cells?",
        ],
    )
    def test_table_queries_route_to_table_aware(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.TABLE
        assert decision.route == RouteTarget.TABLE_AWARE
        assert decision.features.has_table_signals is True
        assert decision.confidence >= 0.8


# ═══════════════════════════════════════════════════════════════════════════
# 3. FIGURE AND DIAGRAM QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestFigureQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "Explain what the system architecture diagram illustrates",
            "What is the trend depicted in figure 3 bar chart?",
            "Describe the steps in the deployment flowchart",
            "What does the electrical schematic show on page 10?",
        ],
    )
    def test_figure_queries_route_to_multimodal(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.FIGURE
        assert decision.route == RouteTarget.MULTIMODAL
        assert decision.features.has_figure_signals is True
        assert decision.confidence >= 0.8


# ═══════════════════════════════════════════════════════════════════════════
# 4. METADATA QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "Who is the author of this specification document?",
            "When was this research paper published?",
            "What is the page count and file size of the report?",
            "Who wrote the compliance policy?",
        ],
    )
    def test_metadata_queries_route_to_metadata_lookup(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.METADATA
        assert decision.route == RouteTarget.METADATA_LOOKUP
        assert decision.features.has_metadata_signals is True
        assert decision.confidence >= 0.8


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERAL SEMANTIC QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneralSemanticQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "How does the Raft consensus algorithm handle leader election?",
            "What are the benefits of vector embeddings over keyword search?",
            "Explain how PostgreSQL handles MVCC concurrency",
        ],
    )
    def test_general_semantic_queries_route_to_hybrid_rrf(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.GENERAL_SEMANTIC
        assert decision.route == RouteTarget.HYBRID_RRF
        assert decision.confidence >= 0.6


# ═══════════════════════════════════════════════════════════════════════════
# 6. SUMMARIZATION QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestSummarizationQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "Please summarize the entire audit report",
            "Provide an executive summary of quarterly earnings",
            "Give me a brief overview of key takeaways from this document",
            "TL;DR of the whitepaper",
        ],
    )
    def test_summarization_queries_route_to_summarization(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.SUMMARIZATION
        assert decision.route == RouteTarget.SUMMARIZATION
        assert decision.features.has_summarization_signals is True
        assert decision.confidence >= 0.8


# ═══════════════════════════════════════════════════════════════════════════
# 7. DOCUMENT-SPECIFIC QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestDocumentSpecificQueryRouting:
    def test_explicit_document_filename(self):
        router = QueryRouter()
        query = "In document annual_report_2024.pdf, what was gross margin?"

        decision = router.route(query)

        assert decision.query_type == QueryType.DOCUMENT_SPECIFIC
        assert decision.route == RouteTarget.DOCUMENT_FILTERED
        assert "annual_report_2024.pdf" in decision.target_document_ids

    def test_according_to_phrase(self):
        router = QueryRouter()
        query = "According to security_guidelines.pdf, what are the password rules?"

        decision = router.route(query)

        assert decision.query_type == QueryType.DOCUMENT_SPECIFIC
        assert decision.route == RouteTarget.DOCUMENT_FILTERED
        assert "security_guidelines.pdf" in decision.target_document_ids


# ═══════════════════════════════════════════════════════════════════════════
# 8. MULTI-HOP QUERIES ROUTING
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiHopQueryRouting:
    def test_causal_multi_hop_query(self):
        router = QueryRouter()
        query = "How does inflation affect interest rates which then affects corporate borrowing?"

        decision = router.route(query)

        assert decision.query_type == QueryType.MULTI_HOP
        assert decision.route == RouteTarget.AGENTIC_WORKFLOW
        assert decision.features.has_multi_hop_signals is True

    def test_cross_reference_multi_hop(self):
        router = QueryRouter()
        query = "Trace the connection between supply disruption and chip shortages across both sectors"

        decision = router.route(query)

        assert decision.query_type == QueryType.MULTI_HOP
        assert decision.route == RouteTarget.AGENTIC_WORKFLOW


# ═══════════════════════════════════════════════════════════════════════════
# 9. UNANSWERABLE AND GIBBERISH QUERIES
# ═══════════════════════════════════════════════════════════════════════════

class TestUnanswerableQueryRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "Predict what will happen in the stock market in 2050",
            "asdfghjk qwertyuiop zxcvbnm",
            "??? !!! ...",
            "What is your favorite color and how do you feel today?",
            "Who will win the presidential election in 2064?",
        ],
    )
    def test_unanswerable_queries_route_to_direct_abstain(self, query):
        router = QueryRouter()
        decision = router.route(query)

        assert decision.query_type == QueryType.UNANSWERABLE
        assert decision.route == RouteTarget.DIRECT_ABSTAIN
        assert decision.confidence >= 0.9


# ═══════════════════════════════════════════════════════════════════════════
# 10. AMBIGUOUS QUERIES & LOW-CONFIDENCE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbiguousQueriesAndFallback:
    def test_ambiguous_short_query_falls_back_to_default_route(self):
        router = QueryRouter(
            confidence_threshold=0.6,
            default_route=RouteTarget.HYBRID_RRF,
        )
        # 1-word query has confidence ~0.45, below threshold 0.6
        decision = router.route("apple")

        assert decision.fallback_used is True
        assert decision.route == RouteTarget.HYBRID_RRF
        assert "fallback_reason" in decision.metadata


# ═══════════════════════════════════════════════════════════════════════════
# 11. ROUTER FAILURE HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestRouterFailureHandling:
    def test_classifier_crash_falls_back_gracefully(self):
        failing_classifier = ErrorThrowingClassifier()
        router = QueryRouter(
            rule_classifier=failing_classifier,
            default_route=RouteTarget.HYBRID_RRF,
        )

        decision = router.route("What is ACID in database systems?")

        assert decision.fallback_used is True
        assert decision.route == RouteTarget.HYBRID_RRF
        assert decision.confidence == 0.0
        assert "error" in decision.metadata

    def test_llm_router_timeout_falls_back_to_rule_classifier(self):
        failing_caller = ErrorThrowingLLMCaller()
        router = QueryRouter(
            router_type=RouterType.LLM,
            llm_caller=failing_caller,
        )

        # Query has table signal, rule-based fallback should route to table_aware
        decision = router.route("What is the value in row 3 of table 1?")

        assert decision.fallback_used is True
        assert decision.route == RouteTarget.TABLE_AWARE
        assert "warning" in decision.metadata


# ═══════════════════════════════════════════════════════════════════════════
# 12. LLM & HYBRID ROUTER MODES
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMAndHybridRouterModes:
    def test_llm_classifier_routes_correctly(self):
        mock_resp = json.dumps({
            "query_type": "table",
            "route": "table_aware",
            "confidence": 0.95,
            "reasoning": "Tabular metrics query",
            "target_documents": [],
        })
        caller = MockLLMRouterCaller({"revenue": mock_resp})
        router = QueryRouter(
            router_type=RouterType.LLM,
            llm_caller=caller,
        )

        decision = router.route("Show revenue summary")

        assert caller.call_count == 1
        assert decision.route == RouteTarget.TABLE_AWARE
        assert decision.router_type == RouterType.LLM
        assert decision.cost_usd > 0

    def test_hybrid_router_uses_rules_for_high_confidence(self):
        caller = MockLLMRouterCaller()
        router = QueryRouter(
            router_type=RouterType.HYBRID,
            llm_caller=caller,
            confidence_threshold=0.6,
        )

        # High-confidence exact identifier query
        decision = router.route("Lookup patent US9876543A")

        assert caller.call_count == 0  # Resolved by rules without calling LLM
        assert decision.route == RouteTarget.BM25_FIRST
        assert decision.router_type == RouterType.RULE_BASED

    def test_hybrid_router_uses_llm_for_ambiguous_query(self):
        mock_resp = json.dumps({
            "query_type": "general_semantic",
            "route": "hybrid_rrf",
            "confidence": 0.88,
            "reasoning": "Semantic search for product entity",
            "target_documents": [],
        })
        caller = MockLLMRouterCaller({"apple": mock_resp})
        router = QueryRouter(
            router_type=RouterType.HYBRID,
            llm_caller=caller,
            confidence_threshold=0.6,
        )

        # Short ambiguous query has rule confidence 0.45 < 0.6 -> activates LLM
        decision = router.route("apple")

        assert caller.call_count == 1
        assert decision.router_type == RouterType.HYBRID
        assert decision.confidence == 0.88


# ═══════════════════════════════════════════════════════════════════════════
# 13. TENANT SAFETY & ROUTER DISABLED BYPASS
# ═══════════════════════════════════════════════════════════════════════════

class TestTenantSafetyAndConfigBypass:
    def test_tenant_id_sanitized_in_metadata(self):
        router = QueryRouter()
        decision = router.route("What is gradient descent?", tenant_id="tenant_abc_123")

        assert decision.metadata.get("tenant_safe_id") == "tenant_tenant_abc_123"

    def test_disabled_router_bypasses_layer(self):
        router = QueryRouter(enabled=False, default_route=RouteTarget.HYBRID_RRF)
        decision = router.route("Lookup patent US9876543A")

        assert decision.route == RouteTarget.HYBRID_RRF
        assert decision.confidence == 1.0
        assert decision.metadata["reason"] == "Router disabled; routed directly to default route"

    def test_settings_integration(self):
        settings = Settings()
        assert hasattr(settings, "routing")
        assert isinstance(settings.routing, QueryRoutingConfig)
        assert settings.routing.enabled is True
        assert settings.routing.router_type == "rule_based"
        assert settings.routing.confidence_threshold == 0.6
        assert settings.routing.default_route == "hybrid_rrf"


# ═══════════════════════════════════════════════════════════════════════════
# 14. TELEMETRY METRICS COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestRoutingTelemetryMetrics:
    def test_metrics_collection_and_summary(self):
        metrics = RoutingMetricsCollector()
        router = QueryRouter(metrics_collector=metrics)

        router.route("Lookup patent US9876543A")
        router.route("What is in table 1?")
        router.route("apple")  # triggers fallback

        summary = metrics.get_summary()

        assert summary["total_routed"] == 3
        assert summary["fallback_count"] == 1
        assert summary["fallback_rate"] == pytest.approx(1 / 3, 0.01)
        assert summary["avg_confidence"] > 0.5
        assert summary["avg_latency_ms"] >= 0.0
        assert "bm25_first" in summary["routes"]
        assert "table_aware" in summary["routes"]


# ═══════════════════════════════════════════════════════════════════════════
# 15. GOLDEN EVALUATION DATASET BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

class TestRoutingEvaluationBenchmark:
    def test_golden_dataset_accuracy_benchmark(self):
        dataset_path = Path(__file__).parent / "routing" / "routing_golden_dataset.json"
        assert dataset_path.exists(), f"Golden dataset missing at {dataset_path}"

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        assert len(cases) >= 25, "Dataset must contain at least 25 evaluation cases"

        router = QueryRouter()
        correct_route = 0
        correct_type = 0

        for case in cases:
            decision = router.route(case["query"])
            if decision.route.value == case["expected_route"]:
                correct_route += 1
            if decision.query_type.value == case["expected_type"]:
                correct_type += 1

        route_accuracy = correct_route / len(cases)
        type_accuracy = correct_type / len(cases)

        # High benchmark requirement: > 90% routing accuracy
        assert route_accuracy >= 0.90, f"Route accuracy {route_accuracy:.2%} below required 90%"
        assert type_accuracy >= 0.90, f"Type accuracy {type_accuracy:.2%} below required 90%"
