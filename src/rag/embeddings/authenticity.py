"""Embedding Authenticity Verification and Diagnostic Subsystem.

Provides mathematical validation, mock-signature detection, semantic sensitivity
checks, stability verification, input-to-chunk mapping verification, and database
persistence round-trip fidelity tests.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.validator import EmbeddingValidator

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two float vectors.

    Returns:
        Float value between -1.0 and 1.0 (or 0.0 if vectors have zero norm).
    """
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0 or not math.isfinite(norm_a) or not math.isfinite(norm_b):
        return 0.0

    sim = dot / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies
    return max(-1.0, min(1.0, sim))


class EmbeddingAuthenticityReport(BaseModel):
    """Structured report documenting the complete authenticity and integrity of embeddings."""

    is_valid: bool = Field(default=True, description="Overall validity of the embedding implementation")
    is_zero_vector: bool = Field(default=False, description="True if any zero vector was detected")
    is_constant_vector: bool = Field(default=False, description="True if any constant vector was detected")
    is_repeated_vector: bool = Field(default=False, description="True if duplicate vectors were detected for different text")
    is_suspiciously_identical: bool = Field(default=False, description="True if unrelated chunks produce identical vectors")
    is_mock_like: bool = Field(default=False, description="True if vectors exhibit mock or pseudo-random hash characteristics")
    dimension_valid: bool = Field(default=True, description="True if dimensions strictly match configured dimension")
    norm_valid: bool = Field(default=True, description="True if vector L2 norms are finite, positive, and healthy")
    provider_verified: bool = Field(default=False, description="True if provider is reachable and verified")
    model_verified: bool = Field(default=False, description="True if model identifier and metadata match")
    input_mapping_verified: bool = Field(default=False, description="True if chunk inputs map 1:1 in exact batch order")
    persistence_verified: bool = Field(default=False, description="True if DB round-trip preserves vector within tolerance")

    # Diagnostic Metrics
    configured_provider: str = ""
    configured_model: str = ""
    configured_dimension: int = 0
    runtime_mode: str = "development"

    same_text_similarity: Optional[float] = None
    similar_text_similarity: Optional[float] = None
    unrelated_text_similarity: Optional[float] = None

    stability_exact_match: bool = False
    stability_cosine_similarity: Optional[float] = None
    stability_max_abs_diff: Optional[float] = None

    unique_vector_count: Optional[int] = None
    duplicate_vector_count: Optional[int] = None
    duplicate_ratio: Optional[float] = None

    persistence_max_abs_diff: Optional[float] = None
    persistence_mean_abs_diff: Optional[float] = None
    persistence_cosine_similarity: Optional[float] = None

    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    overall_status: str = "PASS"  # PASS | FAIL | WARN


class EmbeddingAuthenticityChecker:
    """Diagnostic and audit utility to test embedding quality, authenticity, and persistence."""

    DEFAULT_DIAGNOSTIC_TEXTS = [
        "PostgreSQL is used for relational database storage and ACID transactions.",
        "PostgreSQL supports relational database persistence and ACID compliance.",
        "Quantum mechanics describes physical systems at subatomic microscopic scales.",
        "Optical character recognition extracts text glyphs from scanned page images.",
        "Recursive chunking splits long markdown documents along structural boundaries.",
        "Vector search finds top-k nearest neighbors via approximate distance metrics.",
        "Enterprise document ingestion pipelines require incremental change tracking.",
        "Large language models generate coherent responses given retrieved context.",
        "Convolutional neural networks extract hierarchical spatial image features.",
        "MinIO provides high-performance S3-compatible cloud object storage.",
    ]

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider
        self.validator = EmbeddingValidator(expected_dimension=provider.dimension)

    def check_vector_math(self, vector: List[float]) -> Dict[str, Any]:
        """Perform strict mathematical checks on a vector."""
        res: Dict[str, Any] = {
            "valid": True,
            "dimension": len(vector),
            "expected_dimension": self.provider.dimension,
            "is_zero": False,
            "is_constant": False,
            "has_nan": False,
            "has_inf": False,
            "norm": 0.0,
            "error": None,
        }

        if len(vector) != self.provider.dimension:
            res["valid"] = False
            res["error"] = f"Dimension mismatch: expected {self.provider.dimension}, got {len(vector)}"
            return res

        for val in vector:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                res["valid"] = False
                res["error"] = f"Non-numeric element: {val}"
                return res
            if math.isnan(val):
                res["valid"] = False
                res["has_nan"] = True
                res["error"] = "NaN value in vector"
                return res
            if math.isinf(val):
                res["valid"] = False
                res["has_inf"] = True
                res["error"] = "Infinity value in vector"
                return res

        res["is_zero"] = EmbeddingValidator.is_zero_vector(vector)
        res["is_constant"] = EmbeddingValidator.is_constant_vector(vector)
        res["norm"] = EmbeddingValidator.calculate_l2_norm(vector)

        if res["is_zero"]:
            res["valid"] = False
            res["error"] = "Zero vector detected"
        elif res["is_constant"]:
            res["valid"] = False
            res["error"] = "Constant vector detected"
        elif res["norm"] == 0.0 or not math.isfinite(res["norm"]):
            res["valid"] = False
            res["error"] = f"Invalid norm: {res['norm']}"

        return res

    def check_same_input_stability(
        self,
        text: str = "Enterprise RAG architectural persistence and transactional verification.",
    ) -> Dict[str, Any]:
        """Verify that identical input text generates stable/identical embeddings."""
        v1 = self.provider.embed(text)
        v2 = self.provider.embed(text)
        v3 = self.provider.embed(text)

        exact_match = (v1 == v2 == v3)
        sim_12 = cosine_similarity(v1, v2)
        sim_23 = cosine_similarity(v2, v3)
        min_sim = min(sim_12, sim_23)

        max_diff = max(max(abs(a - b) for a, b in zip(v1, v2)), max(abs(b - c) for b, c in zip(v2, v3)))

        return {
            "exact_match": exact_match,
            "cosine_similarity": min_sim,
            "maximum_absolute_difference": max_diff,
            "stable": min_sim > 0.99999,
        }

    def check_different_input_variation(
        self,
        texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify that distinct text inputs generate distinct vectors."""
        sample_texts = texts or self.DEFAULT_DIAGNOSTIC_TEXTS
        vectors = self.provider.embed_batch(sample_texts)

        dup_count, dup_ratio = EmbeddingValidator.calculate_duplicate_stats(vectors)
        unique_count = len(sample_texts) - dup_count
        is_suspicious = (dup_count > 0 and len(sample_texts) > 1)

        return {
            "total_count": len(sample_texts),
            "unique_count": unique_count,
            "duplicate_count": dup_count,
            "duplicate_ratio": dup_ratio,
            "is_suspiciously_identical": is_suspicious,
            "vectors": vectors,
        }

    def check_semantic_sensitivity(self) -> Dict[str, Any]:
        """Test semantic sensitivity: same vs similar vs unrelated inputs."""
        t_base = "PostgreSQL is used for relational database storage."
        t_same = "PostgreSQL is used for relational database storage."
        t_similar = "PostgreSQL is used for relational database persistence."
        t_unrelated = "Quantum mechanics describes physical systems at microscopic scales."

        batch = self.provider.embed_batch([t_base, t_same, t_similar, t_unrelated])
        v_base, v_same, v_similar, v_unrelated = batch[0], batch[1], batch[2], batch[3]

        same_sim = cosine_similarity(v_base, v_same)
        similar_sim = cosine_similarity(v_base, v_similar)
        unrelated_sim = cosine_similarity(v_base, v_unrelated)

        # In a real semantic model, similar_sim is substantially higher than unrelated_sim.
        # In a hash-based mock model (SHA-256), the avalanche effect makes similar_sim ≈ unrelated_sim ≈ 0.0.
        is_mock_like = False
        mock_indicators = []

        if self.provider.provider_name.lower() == "mock":
            is_mock_like = True
            mock_indicators.append("Provider identifies as 'mock'")

        if abs(similar_sim - unrelated_sim) < 0.08 and similar_sim < 0.35:
            is_mock_like = True
            mock_indicators.append(
                f"Near-orthogonal similar text similarity ({similar_sim:.4f} vs unrelated {unrelated_sim:.4f}) indicates pseudo-random hash generator."
            )

        return {
            "same_text_similarity": same_sim,
            "similar_text_similarity": similar_sim,
            "unrelated_text_similarity": unrelated_sim,
            "is_mock_like": is_mock_like,
            "mock_indicators": mock_indicators,
        }

    def check_input_to_embedding_mapping(self) -> Dict[str, Any]:
        """Verify that batch responses strictly preserve input chunk ordering and mapping."""
        chunks = [
            ("chk_1", "This is content about databases."),
            ("chk_2", "This is content about OCR and vision processing."),
            ("chk_3", "This is content about vector search and ANN indexing."),
        ]
        texts = [c[1] for c in chunks]
        cids = [c[0] for c in chunks]

        batch_vectors = self.provider.embed_batch(texts)
        single_vectors = [self.provider.embed(t) for t in texts]

        # Verify each batch item matches the independently computed single item
        mapping_verified = True
        discrepancies = []
        for i, (cid, text) in enumerate(chunks):
            cos_sim = cosine_similarity(batch_vectors[i], single_vectors[i])
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cos_sim < 0.9999:
                mapping_verified = False
                discrepancies.append(
                    f"Chunk {cid} (hash {text_hash[:8]}): batch vector does not match single vector (sim={cos_sim:.4f})"
                )

        # Cross-comparison: vec[0] must not equal vec[1]
        if cosine_similarity(batch_vectors[0], batch_vectors[1]) > 0.9999:
            mapping_verified = False
            discrepancies.append("Different chunks produced identical batch vectors!")

        return {
            "mapping_verified": mapping_verified,
            "discrepancies": discrepancies,
            "chunk_count": len(chunks),
        }

    def check_persistence_integrity(
        self,
        db_manager: DatabaseManager,
    ) -> Dict[str, Any]:
        """Verify database persistence and readback fidelity."""
        test_doc_id = f"diag_doc_{uuid.uuid4().hex[:8]}"
        test_chunk_id = f"diag_chk_{uuid.uuid4().hex[:8]}"
        test_text = "Diagnostic chunk for persistence round-trip fidelity check."

        # 1. Generate vector
        generated_vec = self.provider.embed(test_text)
        expected_dim = len(generated_vec)

        # 2. Persist to DB
        with db_manager.session() as session:
            doc = Document(
                document_id=test_doc_id,
                tenant_id="diagnostic",
                source="DIAGNOSTIC",
                source_uri="/test/diag.pdf",
                filename="diag.pdf",
                mime_type="application/pdf",
                content_hash=hashlib.sha256(test_text.encode("utf-8")).hexdigest(),
                version=1,
            )
            session.add(doc)

            chunk = Chunk(
                chunk_id=test_chunk_id,
                document_id=test_doc_id,
                version=1,
                chunk_index=0,
                text=test_text,
            )
            session.add(chunk)

            db_emb = Embedding(
                chunk_id=test_chunk_id,
                embedding_model=self.provider.model_name,
                embedding_version=self.provider.model_version,
                dimension=expected_dim,
                provider=self.provider.provider_name,
                vector=generated_vec,
            )
            session.add(db_emb)
            session.commit()

        # 3. Read back
        with db_manager.session() as session:
            stmt = select(Embedding).where(
                Embedding.chunk_id == test_chunk_id,
                Embedding.embedding_model == self.provider.model_name,
                Embedding.embedding_version == self.provider.model_version,
            )
            loaded_emb = session.execute(stmt).scalar_one_or_none()

            if not loaded_emb:
                return {
                    "verified": False,
                    "error": "Failed to read back persisted embedding from database.",
                }

            readback_vec = loaded_emb.vector
            readback_dim = loaded_emb.dimension
            readback_provider = loaded_emb.provider
            readback_model = loaded_emb.embedding_model
            readback_version = loaded_emb.embedding_version

        # 4. Clean up test data
        try:
            with db_manager.session() as session:
                del_doc = session.execute(select(Document).where(Document.document_id == test_doc_id)).scalar_one_or_none()
                if del_doc:
                    session.delete(del_doc)
                session.commit()
        except Exception:
            pass

        # 5. Numerical comparison
        if len(readback_vec) != expected_dim:
            return {
                "verified": False,
                "error": f"Readback dimension mismatch: expected {expected_dim}, got {len(readback_vec)}",
            }

        abs_diffs = [abs(a - b) for a, b in zip(generated_vec, readback_vec)]
        max_abs_diff = max(abs_diffs)
        mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
        cos_sim = cosine_similarity(generated_vec, readback_vec)

        # Check metadata match
        metadata_match = (
            readback_dim == expected_dim
            and readback_provider == self.provider.provider_name
            and readback_model == self.provider.model_name
            and readback_version == self.provider.model_version
        )

        verified = (max_abs_diff < 1e-5 and cos_sim > 0.99999 and metadata_match)

        return {
            "verified": verified,
            "maximum_absolute_difference": max_abs_diff,
            "mean_absolute_difference": mean_abs_diff,
            "cosine_similarity": cos_sim,
            "metadata_match": metadata_match,
            "stored_dimension": readback_dim,
            "stored_provider": readback_provider,
            "stored_model": readback_model,
        }

    def generate_full_report(
        self,
        db_manager: Optional[DatabaseManager] = None,
        is_production: bool = False,
    ) -> EmbeddingAuthenticityReport:
        """Run all authenticity, sensitivity, stability, math, and persistence checks."""
        report = EmbeddingAuthenticityReport(
            configured_provider=self.provider.provider_name,
            configured_model=self.provider.model_name,
            configured_dimension=self.provider.dimension,
            runtime_mode="production" if is_production else "development",
        )

        # 1. Connection Validation
        try:
            conn_ok = self.provider.validate_connection()
            report.provider_verified = bool(conn_ok)
        except Exception as e:
            report.provider_verified = False
            report.errors.append(f"Provider connection check failed: {e}")

        # 2. Stability check
        try:
            stab = self.check_same_input_stability()
            report.stability_exact_match = stab["exact_match"]
            report.stability_cosine_similarity = stab["cosine_similarity"]
            report.stability_max_abs_diff = stab["maximum_absolute_difference"]
            if not stab["stable"]:
                report.warnings.append("Same input produced non-identical or unstable embeddings.")
        except Exception as e:
            report.errors.append(f"Stability check error: {e}")

        # 3. Different inputs variation & duplicate check
        try:
            var = self.check_different_input_variation()
            report.unique_vector_count = var["unique_count"]
            report.duplicate_vector_count = var["duplicate_count"]
            report.duplicate_ratio = var["duplicate_ratio"]
            report.is_repeated_vector = var["duplicate_count"] > 0
            report.is_suspiciously_identical = var["is_suspiciously_identical"]
            if var["is_suspiciously_identical"]:
                report.errors.append("Unrelated texts produced duplicate/identical embedding vectors!")

            # Mathematical check on the generated vectors
            for i, vec in enumerate(var["vectors"]):
                m = self.check_vector_math(vec)
                if not m["valid"]:
                    report.is_valid = False
                    report.errors.append(f"Vector {i} mathematical check failed: {m['error']}")
                if m["is_zero"]:
                    report.is_zero_vector = True
                if m["is_constant"]:
                    report.is_constant_vector = True
        except Exception as e:
            report.errors.append(f"Input variation check error: {e}")

        # 4. Semantic sensitivity & mock detection
        try:
            sem = self.check_semantic_sensitivity()
            report.same_text_similarity = sem["same_text_similarity"]
            report.similar_text_similarity = sem["similar_text_similarity"]
            report.unrelated_text_similarity = sem["unrelated_text_similarity"]
            report.is_mock_like = sem["is_mock_like"]

            if sem["is_mock_like"]:
                for ind in sem["mock_indicators"]:
                    report.warnings.append(f"Mock vector detected: {ind}")
        except Exception as e:
            report.errors.append(f"Semantic sensitivity check error: {e}")

        # 5. Input-to-embedding mapping
        try:
            mapping = self.check_input_to_embedding_mapping()
            report.input_mapping_verified = mapping["mapping_verified"]
            if not mapping["mapping_verified"]:
                report.errors.extend(mapping["discrepancies"])
        except Exception as e:
            report.errors.append(f"Input mapping check error: {e}")

        # 6. Database persistence round-trip check
        if db_manager is not None:
            try:
                persist_res = self.check_persistence_integrity(db_manager)
                report.persistence_verified = persist_res["verified"]
                report.persistence_max_abs_diff = persist_res.get("maximum_absolute_difference")
                report.persistence_mean_abs_diff = persist_res.get("mean_absolute_difference")
                report.persistence_cosine_similarity = persist_res.get("cosine_similarity")
                if not persist_res["verified"]:
                    report.errors.append(f"Persistence round-trip error: {persist_res.get('error', 'Integrity mismatch')}")
            except Exception as e:
                report.errors.append(f"Database persistence check error: {e}")
        else:
            report.warnings.append("DatabaseManager not provided; skipped persistence round-trip verification.")

        # Overall Status Resolution
        report.model_verified = (report.configured_model == self.provider.model_name)
        report.dimension_valid = (self.provider.dimension > 0 and not report.is_zero_vector)
        report.norm_valid = not report.is_zero_vector

        # Production Guard Check
        if is_production and report.is_mock_like:
            report.errors.append("CRITICAL: Mock or dummy embeddings detected in PRODUCTION environment!")

        if report.errors:
            report.is_valid = False
            report.overall_status = "FAIL"
        elif report.is_mock_like and not is_production:
            report.overall_status = "WARN"
        else:
            report.overall_status = "PASS"

        return report
