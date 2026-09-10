"""Deterministic rule-based query classifier for query routing."""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from rag.routing.models import (
    ExtractedFeatures,
    QueryType,
    RouterType,
    RouteTarget,
    RoutingDecision,
)

# ── Identifier Patterns ───────────────────────────────────────────────────────

# Patent numbers: US9876543A, US10123456B2, EP1234567, WO2020123456
_PATENT_PATTERN = re.compile(
    r"\b(?:US|EP|WO|JP|CN|GB|FR|DE)\s?\d{6,12}[A-Z\d]*\b",
    re.IGNORECASE,
)

# UUIDs
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Version strings: v1.2.3, v2.0
_VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)

# Standard project / ticket / standard codes: PROJ-1234, ERR-404, ISO-9001, RFC-7540, DOC-001
_CODE_PATTERN = re.compile(r"\b[A-Z]{2,10}[-_]\d{2,}\b")

# AI / ML Model names: gpt-4o, claude-3.5-sonnet, gemini-2.5-flash
_MODEL_NAME_PATTERN = re.compile(
    r"\b(?:gpt-[\w.-]+|claude-[\w.-]+|gemini-[\w.-]+|llama-[\w.-]+|mistral-[\w.-]+|deepseek-[\w.-]+)\b",
    re.IGNORECASE,
)

# Document filename patterns: report.pdf, data_2024.xlsx
_FILE_EXT_PATTERN = re.compile(
    r"\b[\w\-./]+\.(?:pdf|docx|txt|csv|xlsx|json|md|html)\b",
    re.IGNORECASE,
)

# Explicit document target phrases: "in document X", "in file Y", "according to Z"
_DOC_TARGET_PHRASE = re.compile(
    r"(?:in\s+document|in\s+file|according\s+to|from\s+document|within\s+file)\s+['\"]?([A-Za-z0-9_\-./]+\.[a-z0-9]+|['\"][^'\"]+['\"])",
    re.IGNORECASE,
)

# ── Category Patterns ─────────────────────────────────────────────────────────

# Table signals
_TABLE_PATTERN = re.compile(
    r"\b(?:tables?|tabular|rows?|columns?|cells?|data\s*sheet|spread\s*sheet|balance\s*sheet|"
    r"income\s*statement|financial\s*metrics?|table\s*\d+|in\s+the\s+table)\b",
    re.IGNORECASE,
)

# Figure / visual signals
_FIGURE_PATTERN = re.compile(
    r"\b(?:figures?|diagrams?|charts?|flowcharts?|schematics?|screenshots?|graphs?|"
    r"plots?|illustrations?|architecture\s*diagram|figure\s*\d+|bar\s*chart|pie\s*chart|"
    r"scatter\s*plot|visual|image)\b",
    re.IGNORECASE,
)

# Metadata signals
_METADATA_PATTERN = re.compile(
    r"\b(?:who\s+(?:is\s+the\s+author|wrote|authored|created|published)|author\s+of|"
    r"publication\s+date|when\s+was\s+.*?\s*(?:published|created|written|modified|released)|"
    r"page\s+count|how\s+many\s+pages|file\s+size|document\s+type|file\s+extension|creation\s+date|"
    r"last\s+modified|copyright|metadata)\b",
    re.IGNORECASE,
)

# Summarization signals
_SUMMARIZATION_PATTERN = re.compile(
    r"\b(?:summarize|summarise|summary\s+of|executive\s+summary|overview\s+of\s+the\s+(?:entire\s+)?document|"
    r"tldr|tl;dr|key\s+takeaways|recap\s+the|brief\s+overview\s+of|give\s+me\s+a\s+summary)\b",
    re.IGNORECASE,
)

# Multi-hop / complex reasoning signals
_MULTI_HOP_PATTERN = re.compile(
    r"\b(?:first\s+.*and\s+then|how\s+does\s+.*affect\s+.*which\s+then|"
    r"trace\s+the\s+connection\s+between|relationship\s+between\s+.*and\s+.*and|"
    r"cross-reference|across\s+both\s+.*and|step\s*1\s+.*step\s*2|"
    r"what\s+is\s+the\s+impact\s+of\s+.*on\s+.*leading\s+to)\b",
    re.IGNORECASE,
)

# Unanswerable / out-of-scope / opinion signals
_UNANSWERABLE_PATTERN = re.compile(
    r"\b(?:predict\s+(?:the\s+future|what\s+will\s+happen)|what\s+will\s+happen\s+.*?\s*20\d{2}|"
    r"personal\s+opinion|your\s+favorite|how\s+do\s+you\s+feel\s+about|"
    r"are\s+you\s+conscious|who\s+will\s+win\s+.*?\s*election)\b",
    re.IGNORECASE,
)


class RuleBasedQueryClassifier:
    """High-speed deterministic query classifier using regex and linguistic rules."""

    def extract_features(self, query: str) -> ExtractedFeatures:
        """Extract structural, syntactic, and keyword features from the query."""
        stripped = query.strip()
        words = stripped.split()

        # 1. Identifier extraction
        identifiers: List[str] = []
        for pat in (_PATENT_PATTERN, _UUID_PATTERN, _VERSION_PATTERN, _CODE_PATTERN, _MODEL_NAME_PATTERN):
            for match in pat.findall(stripped):
                if match not in identifiers:
                    identifiers.append(match)

        # 2. Target document extraction
        target_docs: List[str] = []
        for match in _FILE_EXT_PATTERN.findall(stripped):
            if match not in target_docs:
                target_docs.append(match)
        for match in _DOC_TARGET_PHRASE.findall(stripped):
            clean_doc = match.strip("'\"")
            if clean_doc not in target_docs:
                target_docs.append(clean_doc)

        # 3. Structural checks
        has_table = bool(_TABLE_PATTERN.search(stripped))
        has_figure = bool(_FIGURE_PATTERN.search(stripped))
        has_metadata = bool(_METADATA_PATTERN.search(stripped))
        has_summary = bool(_SUMMARIZATION_PATTERN.search(stripped))
        has_multi_hop = bool(_MULTI_HOP_PATTERN.search(stripped))

        # Check for multiple question marks or chained question clauses
        if stripped.count("?") > 1 and (" and " in stripped.lower() or " compared to " in stripped.lower()):
            has_multi_hop = True

        # Check for gibberish or empty
        alpha_chars = [c for c in stripped.lower() if c.isalpha()]
        is_empty_or_gibberish = (
            len(stripped) == 0
            or not any(c.isalnum() for c in stripped)
            or bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", stripped.lower()))
        )

        return ExtractedFeatures(
            detected_identifiers=identifiers,
            target_documents=target_docs,
            has_table_signals=has_table,
            has_figure_signals=has_figure,
            has_metadata_signals=has_metadata,
            has_summarization_signals=has_summary,
            has_multi_hop_signals=has_multi_hop,
            is_empty_or_gibberish=is_empty_or_gibberish,
            word_count=len(words),
        )

    def classify(self, query: str) -> RoutingDecision:
        """Classify a query into QueryType and RouteTarget with confidence and reasoning."""
        start_time = time.perf_counter()
        clean_query = query.strip()
        features = self.extract_features(clean_query)

        # Priority 1: Empty or Gibberish or Unsupported
        if features.is_empty_or_gibberish or _UNANSWERABLE_PATTERN.search(clean_query):
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.DIRECT_ABSTAIN,
                query_type=QueryType.UNANSWERABLE,
                confidence=0.95,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Query is unanswerable, gibberish, or unsupported prediction"},
            )

        # Priority 2: Exact Identifier (Patent, UUID, error code, version, part number)
        # If query has identifier and is concise (focused on lookup)
        if features.detected_identifiers:
            # Check if query is dominated by identifier search
            id_len = sum(len(i) for i in features.detected_identifiers)
            if features.word_count <= 8 or id_len / max(len(clean_query), 1) > 0.25:
                latency = (time.perf_counter() - start_time) * 1000
                return RoutingDecision(
                    query=clean_query,
                    route=RouteTarget.BM25_FIRST,
                    query_type=QueryType.EXACT_IDENTIFIER,
                    confidence=0.95,
                    fallback_used=False,
                    router_type=RouterType.RULE_BASED,
                    features=features,
                    latency_ms=round(latency, 2),
                    metadata={
                        "reason": "Exact identifier detected, prioritized for lexical BM25 retrieval",
                        "identifiers": features.detected_identifiers,
                    },
                )

        # Priority 3: Summarization
        if features.has_summarization_signals:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.SUMMARIZATION,
                query_type=QueryType.SUMMARIZATION,
                confidence=0.90,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Document or section summarization requested"},
            )

        # Priority 4: Metadata Query
        if features.has_metadata_signals:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.METADATA_LOOKUP,
                query_type=QueryType.METADATA,
                confidence=0.90,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Document metadata query detected"},
            )

        # Priority 5: Document-specific query
        if features.target_documents:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.DOCUMENT_FILTERED,
                query_type=QueryType.DOCUMENT_SPECIFIC,
                confidence=0.90,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                target_document_ids=features.target_documents,
                latency_ms=round(latency, 2),
                metadata={
                    "reason": "Explicit document scope detected",
                    "target_documents": features.target_documents,
                },
            )

        # Priority 6: Table query
        if features.has_table_signals:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.TABLE_AWARE,
                query_type=QueryType.TABLE,
                confidence=0.85,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Tabular structure or metric query detected"},
            )

        # Priority 7: Figure or diagram query
        if features.has_figure_signals:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.MULTIMODAL,
                query_type=QueryType.FIGURE,
                confidence=0.85,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Figure, diagram, or visual asset query detected"},
            )

        # Priority 8: Multi-hop reasoning query
        if features.has_multi_hop_signals:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.AGENTIC_WORKFLOW,
                query_type=QueryType.MULTI_HOP,
                confidence=0.80,
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Complex multi-step or cross-document reasoning detected"},
            )

        # Ambiguity check: very short query without clear intent
        if features.word_count <= 2 and not features.detected_identifiers:
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingDecision(
                query=clean_query,
                route=RouteTarget.HYBRID_RRF,
                query_type=QueryType.GENERAL_SEMANTIC,
                confidence=0.45,  # Low confidence triggers fallback checking
                fallback_used=False,
                router_type=RouterType.RULE_BASED,
                features=features,
                latency_ms=round(latency, 2),
                metadata={"reason": "Short ambiguous query with low classification confidence"},
            )

        # Default: General semantic question
        latency = (time.perf_counter() - start_time) * 1000
        return RoutingDecision(
            query=clean_query,
            route=RouteTarget.HYBRID_RRF,
            query_type=QueryType.GENERAL_SEMANTIC,
            confidence=0.75,
            fallback_used=False,
            router_type=RouterType.RULE_BASED,
            features=features,
            latency_ms=round(latency, 2),
            metadata={"reason": "General conceptual or factual inquiry routed to hybrid search with RRF"},
        )
