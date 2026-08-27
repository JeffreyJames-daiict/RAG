"""Tests for the centralized config.json configuration system."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag.core.config import (
    AppConfig,
    ChunkingConfigModel,
    DatabaseConfig,
    LoggingConfig,
    ParserConfig,
    PipelineConfig,
    Settings,
    StorageConfig,
    _expand_env_vars,
    generate_json_schema,
    get_settings,
    load_settings,
)


class TestEnvVarExpansion:
    """Test environment variable expansion logic."""

    def test_expand_simple_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello_world")
        data = {"key": "${TEST_VAR}"}
        result = _expand_env_vars(data)
        assert result["key"] == "hello_world"

    def test_expand_var_with_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("UNSET_VAR", raising=False)
        data = {"key": "${UNSET_VAR:-default_val}"}
        result = _expand_env_vars(data)
        assert result["key"] == "default_val"

    def test_expand_var_with_default_when_set(self, monkeypatch):
        monkeypatch.setenv("SET_VAR", "actual_val")
        data = {"key": "${SET_VAR:-default_val}"}
        result = _expand_env_vars(data)
        assert result["key"] == "actual_val"

    def test_expand_nested_structures(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "postgres.local")
        data = {
            "database": {
                "url": "postgresql://user:pass@${DB_HOST}:5432/rag",
                "list": ["${DB_HOST}", "static"]
            }
        }
        result = _expand_env_vars(data)
        assert result["database"]["url"] == "postgresql://user:pass@postgres.local:5432/rag"
        assert result["database"]["list"] == ["postgres.local", "static"]


class TestConfigValidation:
    """Test validation constraints on configuration models."""

    def test_valid_settings_instantiation(self):
        settings = Settings()
        assert settings.app.name == "Enterprise RAG Pipeline"
        assert settings.chunking.chunk_size == 512
        assert settings.chunking.chunk_overlap == 64

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap.*must be < chunk_size"):
            ChunkingConfigModel(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap.*must be < chunk_size"):
            ChunkingConfigModel(chunk_size=100, chunk_overlap=150)

    def test_safe_url_masks_password(self):
        db_cfg = DatabaseConfig(url="postgresql://user:secret123@localhost:5432/rag")
        assert "secret123" not in db_cfg.safe_url()
        assert "user:***@localhost:5432/rag" in db_cfg.safe_url()

    def test_safe_url_unaffected_when_no_credentials(self):
        db_cfg = DatabaseConfig(url="sqlite:///rag.db")
        assert db_cfg.safe_url() == "sqlite:///rag.db"


class TestConfigLoading:
    """Test loading settings from config.json."""

    def test_load_default_config_json(self):
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.app.name == "Enterprise RAG Pipeline"
        assert settings.database.pool_size >= 1
        assert "application/pdf" in settings.parser.supported_mime_types

    def test_load_custom_config_file(self, tmp_path):
        custom_file = tmp_path / "custom_config.json"
        custom_data = {
            "app": {"name": "Custom RAG", "version": "2.0.0", "env": "production"},
            "chunking": {"chunk_size": 256, "chunk_overlap": 32}
        }
        custom_file.write_text(json.dumps(custom_data))

        settings = load_settings(config_path=custom_file)
        assert settings.app.name == "Custom RAG"
        assert settings.app.version == "2.0.0"
        assert settings.app.env == "production"
        assert settings.chunking.chunk_size == 256
        assert settings.chunking.chunk_overlap == 32

    def test_generate_json_schema(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema = generate_json_schema(output_path=schema_file)
        assert "$defs" in schema or "properties" in schema
        assert schema_file.exists()
        loaded = json.loads(schema_file.read_text())
        assert "properties" in loaded

    def test_summary_does_not_leak_secrets(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://admin:super_secret_pw@127.0.0.1:5432/rag")
        get_settings.cache_clear()
        settings = get_settings()
        summary = settings.summary()
        assert "super_secret_pw" not in summary
        assert "admin:***@" in summary
