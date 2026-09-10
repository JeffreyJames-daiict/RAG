"""Multi-format report generators (JSON, CSV, HTML) for RAG benchmark evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from rag.evaluation.models import EvaluationReport


class JsonReporter:
    """Generates comprehensive JSON evaluation reports."""

    @staticmethod
    def render(report: EvaluationReport) -> str:
        return report.model_dump_json(indent=2)

    @staticmethod
    def save(report: EvaluationReport, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(JsonReporter.render(report))


class CsvReporter:
    """Exports tabular CSV reports containing case-level and configuration-level metrics."""

    @staticmethod
    def save(report: EvaluationReport, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 1. Write Configuration Summary Section
            writer.writerow(["=== CONFIGURATION SUMMARY ==="])
            writer.writerow([
                "Configuration", "Total Cases", "Recall@5", "Recall@10",
                "Precision@5", "MRR", "NDCG@5", "Exact ID Acc", "Parent Doc Acc",
                "Citation Validity", "Unsupported Rate", "Abstention Acc",
                "Mean Latency (ms)", "Error Rate",
            ])

            for cfg_name, agg in report.configurations.items():
                writer.writerow([
                    cfg_name,
                    agg.total_cases,
                    f"{agg.mean_recall_at_5:.4f}",
                    f"{agg.mean_recall_at_10:.4f}",
                    f"{agg.mean_precision_at_5:.4f}",
                    f"{agg.mean_mrr:.4f}",
                    f"{agg.mean_ndcg_at_5:.4f}",
                    f"{agg.mean_exact_id_accuracy:.4f}",
                    f"{agg.mean_parent_doc_accuracy:.4f}",
                    f"{agg.mean_citation_validity:.4f}",
                    f"{agg.mean_unsupported_citation_rate:.4f}",
                    f"{agg.abstention_accuracy:.4f}",
                    f"{agg.mean_latency_ms:.2f}",
                    f"{agg.error_rate:.4f}",
                ])

            writer.writerow([])
            writer.writerow(["=== PER-CASE RESULTS ==="])
            writer.writerow([
                "Case ID", "Configuration", "Query", "Recall@5", "MRR", "NDCG@5",
                "Exact ID Hit", "Retrieved Chunks", "Citations Used", "Citation Validity",
                "Latency (ms)", "Error",
            ])

            for cr in report.case_results:
                r_m = cr.retrieval_metrics
                g_m = cr.generation_metrics
                writer.writerow([
                    cr.case_id,
                    cr.config_name,
                    cr.query[:60],
                    f"{r_m.recall_at_k.get(5, 0.0):.4f}",
                    f"{r_m.mrr:.4f}",
                    f"{r_m.ndcg_at_k.get(5, 0.0):.4f}",
                    f"{r_m.exact_id_accuracy:.1f}",
                    " ".join(cr.retrieved_chunk_ids[:5]),
                    " ".join(cr.citations_used),
                    f"{g_m.citation_validity:.4f}" if g_m else "N/A",
                    f"{r_m.latency_ms:.2f}",
                    cr.error or "",
                ])


class HtmlReporter:
    """Generates self-contained, responsive HTML dashboards for visual analysis and CI/CD artifacts."""

    @staticmethod
    def render(report: EvaluationReport) -> str:
        # Determine overall status badge
        status_badge = (
            '<span style="background:#10b981;color:#fff;padding:6px 12px;border-radius:20px;font-weight:bold;">PASSED</span>'
            if report.overall_passed
            else '<span style="background:#ef4444;color:#fff;padding:6px 12px;border-radius:20px;font-weight:bold;">FAILED REGRESSION</span>'
        )

        # Build configuration table rows
        config_rows = []
        for name, agg in report.configurations.items():
            config_rows.append(f"""
            <tr>
                <td style="font-weight:600;padding:12px;border-bottom:1px solid #374151;">{name}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_recall_at_5:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_mrr:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_ndcg_at_5:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_exact_id_accuracy:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_parent_doc_accuracy:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_citation_validity:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.abstention_accuracy:.4f}</td>
                <td style="padding:12px;border-bottom:1px solid #374151;">{agg.mean_latency_ms:.1f} ms</td>
            </tr>
            """)

        # Build case table rows
        case_rows = []
        for cr in report.case_results[:100]:
            r_m = cr.retrieval_metrics
            g_m = cr.generation_metrics
            r5 = r_m.recall_at_k.get(5, 0.0)
            status_color = "#10b981" if (r5 > 0 or cr.error is None) else "#ef4444"

            case_rows.append(f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #374151;font-family:monospace;">{cr.case_id}</td>
                <td style="padding:10px;border-bottom:1px solid #374151;"><span style="background:#1f2937;padding:3px 8px;border-radius:4px;">{cr.config_name}</span></td>
                <td style="padding:10px;border-bottom:1px solid #374151;">{cr.query}</td>
                <td style="padding:10px;border-bottom:1px solid #374151;color:{status_color};font-weight:bold;">{r5:.2f}</td>
                <td style="padding:10px;border-bottom:1px solid #374151;">{r_m.mrr:.2f}</td>
                <td style="padding:10px;border-bottom:1px solid #374151;">{g_m.citation_validity if g_m else 'N/A'}</td>
                <td style="padding:10px;border-bottom:1px solid #374151;">{r_m.latency_ms:.1f}ms</td>
            </tr>
            """)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report — {report.run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #111827;
            color: #f3f4f6;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        .card {{
            background-color: #1f2937;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            border: 1px solid #374151;
        }}
        .header-flex {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background-color: #374151;
            padding: 12px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .stat-card {{
            background: #111827;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #374151;
        }}
        .stat-title {{
            font-size: 12px;
            color: #9ca3af;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #60a5fa;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="card">
            <div class="header-flex">
                <div>
                    <h1 style="margin:0 0 8px 0;font-size:28px;">RAG Evaluation Benchmark Report</h1>
                    <p style="margin:0;color:#9ca3af;">Run ID: <code>{report.run_id}</code> | Dataset: <strong>{report.dataset_id}</strong> | Timestamp: {report.timestamp}</p>
                </div>
                <div>{status_badge}</div>
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-title">Total Cases</div>
                    <div class="stat-value">{report.total_cases}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Configurations Compared</div>
                    <div class="stat-value">{len(report.configurations)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Evaluation Status</div>
                    <div class="stat-value" style="color:{'#10b981' if report.overall_passed else '#ef4444'};">{'PASS' if report.overall_passed else 'FAIL'}</div>
                </div>
            </div>
        </div>

        <!-- Configuration Comparison -->
        <div class="card">
            <h2 style="margin-top:0;">Configuration Comparison Benchmark</h2>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>Recall@5</th>
                            <th>MRR</th>
                            <th>NDCG@5</th>
                            <th>Exact ID Acc</th>
                            <th>Parent Doc Acc</th>
                            <th>Citation Validity</th>
                            <th>Abstention Acc</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(config_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Per-Case Breakdown -->
        <div class="card">
            <h2 style="margin-top:0;">Case Breakdown (Top 100)</h2>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Case ID</th>
                            <th>Config</th>
                            <th>Query</th>
                            <th>Recall@5</th>
                            <th>MRR</th>
                            <th>Citation Validity</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(case_rows)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html

    @staticmethod
    def save(report: EvaluationReport, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(HtmlReporter.render(report))
