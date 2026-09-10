"""Production Context Builder — converts retrieval results into structured, token-budgeted, citation-ready context packages."""

from __future__ import annotations

from collections import defaultdict
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from rag.core.config import ContextBuilderConfig, get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import DocumentAsset
from rag.context.deduplicator import ContextDeduplicator
from rag.context.expander import ContextExpander
from rag.context.isolation import PromptIsolationEngine
from rag.context.models import (
    AssetContext,
    Citation,
    ContextChunk,
    ContextPackage,
    GroupedDocumentContext,
    TableContext,
)
from rag.context.tokenizer import ContextTokenizer
from rag.retrieval.models import RetrievalResponse, RetrievedChunkResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Orchestrates deduplication, expansion, table/asset association, token budgeting, and citation assignment."""

    def __init__(
        self,
        config: Optional[ContextBuilderConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        custom_resolver: Optional[Callable[[str, str, int, Optional[str]], List[RetrievedChunkResult]]] = None,
    ):
        self.config = config or getattr(get_settings(), "context", ContextBuilderConfig())
        self.db_manager = db_manager
        self.tokenizer = ContextTokenizer(encoding_name=self.config.tokenizer_encoding)
        self.deduplicator = ContextDeduplicator(overlap_threshold=self.config.overlap_threshold)
        self.expander = ContextExpander(
            config=self.config,
            db_manager=db_manager,
            custom_resolver=custom_resolver,
        )
        self.isolation_engine = PromptIsolationEngine(enabled=self.config.injection_defense_enabled)

    def _coerce_to_results(
        self,
        candidates: Union[RetrievalResponse, Sequence[Any]],
    ) -> List[RetrievedChunkResult]:
        """Normalize various input candidate representations into RetrievedChunkResult list."""
        if isinstance(candidates, RetrievalResponse):
            return list(candidates.results)

        results: List[RetrievedChunkResult] = []
        for idx, item in enumerate(candidates, start=1):
            if isinstance(item, RetrievedChunkResult):
                results.append(item)
            elif hasattr(item, "chunk_id") and hasattr(item, "text"):
                # Duck-typed object
                results.append(
                    RetrievedChunkResult(
                        chunk_id=getattr(item, "chunk_id"),
                        document_id=getattr(item, "document_id", f"doc_{idx}"),
                        document_version=getattr(item, "document_version", 1),
                        document_version_id=f"v{getattr(item, 'document_version', 1)}",
                        tenant_id=getattr(item, "tenant_id", "default"),
                        text=getattr(item, "text"),
                        rank=getattr(item, "rank", idx),
                        similarity_score=getattr(item, "similarity_score", 1.0),
                        raw_score=getattr(item, "raw_score", 1.0),
                        section=getattr(item, "section", None),
                        page=getattr(item, "page", None),
                        filename=getattr(item, "filename", ""),
                        chunk_type=getattr(item, "chunk_type", "text"),
                        metadata=getattr(item, "metadata", {}) or {},
                    )
                )
            elif isinstance(item, dict):
                results.append(
                    RetrievedChunkResult(
                        chunk_id=item.get("chunk_id", f"chunk_{idx}"),
                        document_id=item.get("document_id", f"doc_{idx}"),
                        document_version=item.get("document_version", 1),
                        document_version_id=f"v{item.get('document_version', 1)}",
                        tenant_id=item.get("tenant_id", "default"),
                        text=item.get("text", ""),
                        rank=item.get("rank", idx),
                        similarity_score=item.get("similarity_score", 1.0),
                        raw_score=item.get("raw_score", 1.0),
                        section=item.get("section"),
                        page=item.get("page"),
                        filename=item.get("filename", ""),
                        chunk_type=item.get("chunk_type", "text"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return results

    def _resolve_assets_for_chunks(
        self,
        chunks: List[RetrievedChunkResult],
        query: str,
    ) -> Dict[str, List[AssetContext]]:
        """Find and attach relevant visual assets (figures, diagrams) matching chunks or query."""
        assets_by_doc: Dict[str, List[AssetContext]] = defaultdict(list)
        if not self.db_manager:
            return assets_by_doc

        doc_ids = {c.document_id for c in chunks}
        asset_ids_in_meta = {
            c.metadata.get("asset_id") for c in chunks if c.metadata and c.metadata.get("asset_id")
        }

        # Check if query specifically asks for figures/diagrams
        query_wants_visuals = any(
            w in query.lower() for w in ("figure", "diagram", "chart", "architecture", "flowchart", "image")
        )

        try:
            with self.db_manager.session() as session:
                q = session.query(DocumentAsset).filter(DocumentAsset.document_id.in_(doc_ids))
                db_assets = q.all()

                for a in db_assets:
                    # Include if explicitly referenced or if query explicitly asks for visuals on that page
                    is_relevant = (
                        a.asset_id in asset_ids_in_meta
                        or (query_wants_visuals and any(c.document_id == a.document_id and c.page == a.page_number for c in chunks))
                    )
                    if is_relevant:
                        assets_by_doc[a.document_id].append(
                            AssetContext(
                                asset_id=a.asset_id,
                                asset_type=a.asset_type,
                                caption=a.caption,
                                ocr_text=a.ocr_text,
                                semantic_description=a.semantic_description,
                                page=a.page_number,
                                storage_uri=a.storage_uri,
                                entities=a.entities or [],
                            )
                        )
        except Exception as e:
            logger.debug("Could not resolve assets from DB: %s", e)

        return assets_by_doc

    def _format_table_if_needed(self, chunk: RetrievedChunkResult) -> Optional[TableContext]:
        """Detect and reconstruct table format with preserved headers and markup."""
        meta = chunk.metadata or {}
        chunk_type = chunk.chunk_type or meta.get("chunk_type", "text")
        table_id = chunk.table_id or meta.get("table_id")

        if chunk_type not in ("table", "table_summary") and not table_id and "|---" not in chunk.text:
            return None

        # Format markdown table
        lines = chunk.text.strip().split("\n")
        caption = meta.get("caption") or meta.get("title") or chunk.section or "Structured Table"
        page = chunk.page

        # Reconstructed table banner
        formatted = f"[TABLE: {caption} | Page {page or 'N/A'}]\n{chunk.text.strip()}"

        return TableContext(
            table_id=table_id or f"table_{chunk.chunk_id}",
            caption=caption,
            page=page,
            formatted_table=formatted,
            is_reconstructed=meta.get("is_continuation", False),
            columns=meta.get("columns", []),
            total_rows=meta.get("total_rows", len(lines)),
        )

    def build(
        self,
        query: str,
        retrieval_candidates: Union[RetrievalResponse, Sequence[Any]],
        *,
        token_budget: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
        max_tokens_per_section: Optional[int] = None,
        tenant_id: str = "default",
    ) -> ContextPackage:
        """Build the complete, token-budgeted, citation-ready ContextPackage."""
        warnings: List[str] = []
        budget = token_budget or self.config.token_budget
        doc_budget = max_tokens_per_doc or self.config.max_tokens_per_document
        sec_budget = max_tokens_per_section or self.config.max_tokens_per_section

        raw_candidates = self._coerce_to_results(retrieval_candidates)
        if not raw_candidates:
            return ContextPackage(
                query=query,
                selected_chunks=[],
                grouped_documents=[],
                formatted_context=self.isolation_engine.wrap_evidence_boundary("<!-- No evidence passages retrieved -->"),
                citations=[],
                token_count=self.tokenizer.count_tokens("<!-- No evidence passages retrieved -->"),
                token_budget=budget,
                truncated=False,
                context_strategy="hierarchical_xml",
                warnings=["No candidates provided for context construction"],
            )

        # 1. Deduplication
        deduped, dedup_warnings = self.deduplicator.deduplicate(raw_candidates)
        warnings.extend(dedup_warnings)

        # 2. Expansion (neighbors and parents)
        expanded = self.expander.expand(deduped)

        # 3. Associate visual assets
        assets_by_doc = self._resolve_assets_for_chunks(expanded, query)

        # 4. Token Budget Allocation & Chunk Selection
        # Group candidates by doc & section to evaluate hierarchical budgets
        doc_token_counts: Dict[str, int] = defaultdict(int)
        sec_token_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        total_tokens = 0

        admitted_chunks: List[ContextChunk] = []
        truncated = False
        truncation_reasons: List[str] = []

        citation_counter = 1
        prefix = self.config.citation_prefix

        # Preserve top-ranked chunk even if large (up to min_content_preservation_tokens)
        for idx, cand in enumerate(expanded):
            text = cand.text
            is_seed = not cand.metadata.get("is_expanded", False)
            exp_type = cand.metadata.get("expansion_type", "seed" if is_seed else "neighbor")

            doc_id = cand.document_id
            sec_name = cand.section or "General"
            sec_key = (doc_id, sec_name)

            chunk_tokens = self.tokenizer.count_tokens(text)

            min_preservation = min(
                self.config.min_content_preservation_tokens,
                max(16, doc_budget // 3),
                max(16, sec_budget // 3),
                max(16, budget // 3),
            )

            # Check per-doc budget
            if doc_token_counts[doc_id] + chunk_tokens > doc_budget:
                remaining_doc = doc_budget - doc_token_counts[doc_id]
                if remaining_doc < min_preservation:
                    truncated = True
                    truncation_reasons.append(f"Document {doc_id} exceeded token limit ({doc_budget})")
                    continue
                # Truncate chunk text to fit remaining doc budget
                text, chunk_tokens = self.tokenizer.truncate_to_tokens(text, remaining_doc)
                truncated = True
                truncation_reasons.append(f"Chunk {cand.chunk_id} trimmed to fit document budget ({remaining_doc} tokens)")
                warnings.append(f"Chunk {cand.chunk_id} truncated to fit document budget ({remaining_doc} tokens)")

            # Check per-section budget
            if sec_token_counts[sec_key] + chunk_tokens > sec_budget:
                remaining_sec = sec_budget - sec_token_counts[sec_key]
                if remaining_sec < min_preservation:
                    truncated = True
                    truncation_reasons.append(f"Section '{sec_name}' exceeded token limit ({sec_budget})")
                    continue
                text, chunk_tokens = self.tokenizer.truncate_to_tokens(text, remaining_sec)
                truncated = True
                truncation_reasons.append(f"Chunk {cand.chunk_id} trimmed to fit section budget")
                warnings.append(f"Chunk {cand.chunk_id} truncated to fit section budget")

            # Check global token budget (reserving overhead for XML envelope)
            xml_envelope_reserve = min(80, max(20, budget // 5))
            effective_global_budget = max(20, budget - xml_envelope_reserve)

            if total_tokens + chunk_tokens > effective_global_budget:
                remaining_global = effective_global_budget - total_tokens
                if remaining_global >= min_preservation:
                    text, chunk_tokens = self.tokenizer.truncate_to_tokens(text, remaining_global)
                    truncated = True
                    truncation_reasons.append(f"Chunk {cand.chunk_id} trimmed to fit global context budget")
                    warnings.append(f"Chunk {cand.chunk_id} truncated to fit global context budget")
                else:
                    truncated = True
                    truncation_reasons.append(f"Exceeded total context token budget ({budget})")
                    break

            # Sanitize text against prompt injection
            sanitized_text, injection_warnings = self.isolation_engine.sanitize_text(text)
            warnings.extend(injection_warnings)

            # Assign Citation
            cit_id = f"[{prefix}{citation_counter}]"
            citation_counter += 1

            table_ctx = self._format_table_if_needed(cand)

            context_chunk = ContextChunk(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=sanitized_text,
                rank=cand.rank,
                page=cand.page,
                section=sec_name,
                chunk_type=cand.chunk_type,
                is_expanded=not is_seed,
                expansion_type=exp_type,
                citation_id=cit_id,
                token_count=chunk_tokens,
                score=cand.similarity_score,
                table_id=table_ctx.table_id if table_ctx else None,
                asset_id=cand.metadata.get("asset_id"),
                filename=cand.filename,
                metadata=cand.metadata,
            )

            admitted_chunks.append(context_chunk)
            doc_token_counts[doc_id] += chunk_tokens
            sec_token_counts[sec_key] += chunk_tokens
            total_tokens += chunk_tokens

        # 5. Build Citations Registry
        citations: List[Citation] = []
        for ch in admitted_chunks:
            snippet = ch.text[:140].replace("\n", " ") + ("..." if len(ch.text) > 140 else "")
            citations.append(
                Citation(
                    citation_id=ch.citation_id,
                    chunk_id=ch.chunk_id,
                    document_id=ch.document_id,
                    filename=ch.filename,
                    page=ch.page,
                    section=ch.section,
                    asset_id=ch.asset_id,
                    table_id=ch.table_id,
                    snippet=snippet,
                    score=ch.score,
                )
            )

        # 6. Group by Document & Section
        grouped_docs: List[GroupedDocumentContext] = []
        doc_chunk_map: Dict[str, List[ContextChunk]] = defaultdict(list)
        for ch in admitted_chunks:
            doc_chunk_map[ch.document_id].append(ch)

        for doc_id, doc_chunks in doc_chunk_map.items():
            first_chunk = doc_chunks[0]
            sections_map: Dict[str, List[ContextChunk]] = defaultdict(list)
            doc_tables: List[TableContext] = []

            # Sort chunks sequentially by page and chunk rank to preserve narrative order
            doc_chunks.sort(key=lambda c: (c.page or 0, c.rank))

            for c in doc_chunks:
                sec_title = c.section or "Content"
                sections_map[sec_title].append(c)
                tbl = self._format_table_if_needed(
                    RetrievedChunkResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        text=c.text,
                        page=c.page,
                        section=c.section,
                        chunk_type=c.chunk_type,
                        metadata=c.metadata,
                    )
                )
                if tbl and not any(t.table_id == tbl.table_id for t in doc_tables):
                    doc_tables.append(tbl)

            grouped_docs.append(
                GroupedDocumentContext(
                    document_id=doc_id,
                    filename=first_chunk.filename,
                    version=1,
                    sections=dict(sections_map),
                    tables=doc_tables,
                    assets=assets_by_doc.get(doc_id, []),
                    total_tokens=doc_token_counts[doc_id],
                )
            )

        # 7. Render Formatted Context (Hierarchical XML with strict security boundary)
        xml_blocks: List[str] = []
        for g_doc in grouped_docs:
            doc_meta_attr = f'id="{g_doc.document_id}"'
            if g_doc.filename:
                doc_meta_attr += f' filename="{g_doc.filename}"'
            xml_blocks.append(f'<document {doc_meta_attr}>')

            # Render tables if present
            for tbl in g_doc.tables:
                xml_blocks.append(
                    f'  <table id="{tbl.table_id}" page="{tbl.page or "N/A"}">\n'
                    f"    {tbl.formatted_table}\n"
                    f"  </table>"
                )

            # Render visual assets if present
            for asset in g_doc.assets:
                caption_part = f"Caption: {asset.caption} | " if asset.caption else ""
                desc_part = f"Description: {asset.semantic_description}" if asset.semantic_description else ""
                xml_blocks.append(
                    f'  <visual_asset id="{asset.asset_id}" type="{asset.asset_type}" page="{asset.page or "N/A"}">\n'
                    f"    [{caption_part}{desc_part}]\n"
                    f"  </visual_asset>"
                )

            # Render sections and passages
            for sec_name, s_chunks in g_doc.sections.items():
                xml_blocks.append(f'  <section title="{sec_name}">')
                for c in s_chunks:
                    page_attr = f' page="{c.page}"' if c.page else ""
                    xml_blocks.append(
                        f'    <passage citation="{c.citation_id}"{page_attr}>\n'
                        f"      {c.text.strip()}\n"
                        f"    </passage>"
                    )
                xml_blocks.append("  </section>")

            xml_blocks.append("</document>")

        inner_context = "\n".join(xml_blocks)
        formatted_context = self.isolation_engine.wrap_evidence_boundary(inner_context)
        final_token_count = self.tokenizer.count_tokens(formatted_context)

        return ContextPackage(
            query=query,
            selected_chunks=admitted_chunks,
            grouped_documents=grouped_docs,
            formatted_context=formatted_context,
            citations=citations,
            token_count=final_token_count,
            token_budget=budget,
            truncated=truncated,
            truncation_reason="; ".join(truncation_reasons) if truncation_reasons else None,
            context_strategy="hierarchical_xml",
            warnings=warnings,
        )
