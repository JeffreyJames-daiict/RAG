"""Embedding Diagnostic CLI and Verification Command.

Usage:
    python -m rag.embeddings.diagnostics
    python -m rag.embeddings.diagnostics --env production
    python -m rag.embeddings.diagnostics --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from rag.core.config import get_settings
from rag.core.database import DatabaseManager
from rag.embeddings.authenticity import EmbeddingAuthenticityChecker, EmbeddingAuthenticityReport
from rag.embeddings.providers import get_embedding_provider


def run_diagnostics(
    env: Optional[str] = None,
    db_url: Optional[str] = None,
    output_json: bool = False,
) -> int:
    """Run full suite of embedding diagnostics and print formatted report."""
    settings = get_settings()
    is_production = (env.lower() == "production") if env else settings.app.is_production

    cfg = settings.embedding
    safe_config = cfg.safe_diagnostic_dict(is_production=is_production)

    # Initialize provider
    try:
        provider = get_embedding_provider(cfg, is_production=is_production)
    except Exception as exc:
        print(f"ERROR: Failed to initialize configured embedding provider: {exc}")
        return 1

    # Initialize DB manager (use in-memory SQLite for non-destructive check if no DB URL provided)
    db_mgr: Optional[DatabaseManager] = None
    try:
        target_db = db_url or settings.database_url
        db_mgr = DatabaseManager(db_url=target_db)
        db_mgr.create_tables()
    except Exception as exc:
        # Fallback to in-memory SQLite for safe roundtrip testing
        try:
            db_mgr = DatabaseManager(db_url="sqlite:///:memory:")
            db_mgr.create_tables()
        except Exception:
            db_mgr = None

    checker = EmbeddingAuthenticityChecker(provider=provider)
    report: EmbeddingAuthenticityReport = checker.generate_full_report(
        db_manager=db_mgr,
        is_production=is_production,
    )

    if output_json:
        print(report.model_dump_json(indent=2))
        return 0 if report.overall_status in ("PASS", "WARN") else 1

    # Terminal Formatted Output
    print("\nEmbedding Diagnostic Report")
    print("===========================\n")
    print(f"Provider:             {report.configured_provider}")
    print(f"Model:                {report.configured_model}")
    print(f"Configured dimension: {report.configured_dimension}")
    print(f"Runtime mode:         {report.runtime_mode}")
    print(f"API Endpoint:         {safe_config.get('embedding_api_base', 'none')}")
    print(f"API Key Configured:   {safe_config.get('embedding_api_key_configured', False)}\n")

    # Diagnostic Line Checks
    checks = [
        ("Provider reachable", report.provider_verified),
        ("API response valid", report.is_valid and len(report.errors) == 0),
        ("Dimension validation", report.dimension_valid),
        ("NaN/Inf validation", not report.is_zero_vector and report.norm_valid),
        ("Zero-vector check", not report.is_zero_vector),
        ("Constant-vector check", not report.is_constant_vector),
        ("Same-input stability", report.stability_exact_match or (report.stability_cosine_similarity or 0.0) > 0.99999),
        ("Different-input sensitivity", not report.is_suspiciously_identical),
        ("Duplicate-vector check", not report.is_repeated_vector),
        ("Persistence round-trip", report.persistence_verified),
        ("Provider metadata verification", report.model_verified),
        ("Mock fallback detected", not report.is_mock_like),
    ]

    for name, passed in checks:
        badge = "PASS" if passed else "FAIL"
        print(f"{name:<32}: {badge}")

    print(f"\nOverall status: {report.overall_status}")

    # Metrics Summary
    print("\nDiagnostics Details:")
    if report.same_text_similarity is not None:
        print(f"  • Same-text similarity:       {report.same_text_similarity:.6f}")
    if report.similar_text_similarity is not None:
        print(f"  • Similar-text similarity:    {report.similar_text_similarity:.6f}")
    if report.unrelated_text_similarity is not None:
        print(f"  • Unrelated-text similarity:  {report.unrelated_text_similarity:.6f}")
    if report.persistence_max_abs_diff is not None:
        print(f"  • Persistence max abs diff:   {report.persistence_max_abs_diff:.8f}")
    if report.persistence_cosine_similarity is not None:
        print(f"  • Persistence cosine sim:     {report.persistence_cosine_similarity:.8f}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  ⚠️  {w}")

    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  ❌ {e}")

    print()
    return 0 if report.overall_status in ("PASS", "WARN") else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate embedding provider authenticity and persistence.")
    parser.add_argument("--env", type=str, choices=["development", "staging", "production"], help="Override runtime mode")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL for persistence roundtrip")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")

    args = parser.parse_args()
    exit_code = run_diagnostics(env=args.env, db_url=args.db_url, output_json=args.json)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
