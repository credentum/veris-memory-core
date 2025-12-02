"""
Test suite for embedding service config loading functionality.

Tests cover:
- Config file discovery from multiple locations
- YAML parsing and validation
- Model name mapping
- Error handling scenarios
- Service initialization with loaded config
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.embedding.service import (
    _load_config_from_file,
    _MODEL_NAME_MAP,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingService,
    get_embedding_service,
)


class TestConfigFileDiscovery:
    """Test config file discovery across multiple search paths."""

    def test_load_from_env_var_path(self, tmp_path):
        """Test loading config from CTX_CONFIG_PATH environment variable."""
        config_file = tmp_path / "custom_config.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384,
                'max_retries': 5,
                'timeout': 60.0,
                'batch_size': 50
            }
        }
        config_file.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {'CTX_CONFIG_PATH': str(config_file)}):
            config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2
        assert config.target_dimensions == 384
        assert config.max_retries == 5
        assert config.timeout_seconds == 60.0
        assert config.batch_size == 50

    def test_load_from_config_dir(self, tmp_path, monkeypatch):
        """Test loading config from config/.ctxrc.yaml."""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / ".ctxrc.yaml"

        config_data = {
            'embeddings': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimensions': 384
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Change to tmp_path directory
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2
        assert config.target_dimensions == 384

    def test_load_from_root_dir(self, tmp_path, monkeypatch):
        """Test loading config from .ctxrc.yaml in root directory."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384
            }
        }
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2

    def test_priority_order_env_var_wins(self, tmp_path, monkeypatch):
        """Test that CTX_CONFIG_PATH has highest priority."""
        # Create config in env var path
        env_config = tmp_path / "env_config.yaml"
        env_config.write_text(yaml.dump({
            'embeddings': {'model': 'text-embedding-ada-002', 'dimensions': 1536}
        }))

        # Create config in config dir
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / ".ctxrc.yaml"
        config_file.write_text(yaml.dump({
            'embeddings': {'model': 'all-MiniLM-L6-v2', 'dimensions': 384}
        }))

        monkeypatch.chdir(tmp_path)

        with patch.dict(os.environ, {'CTX_CONFIG_PATH': str(env_config)}):
            config = _load_config_from_file()

        # Should use env var config (OpenAI model)
        assert config.model == EmbeddingModel.OPENAI_ADA_002
        assert config.target_dimensions == 1536

    def test_no_config_file_returns_none(self, tmp_path, monkeypatch):
        """Test that missing config file returns None."""
        monkeypatch.chdir(tmp_path)

        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_file()

        assert config is None


class TestYAMLParsing:
    """Test YAML parsing and validation."""

    def test_parse_valid_yaml(self, tmp_path, monkeypatch):
        """Test parsing valid YAML configuration."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384,
                'max_retries': 3,
                'timeout': 30.0,
                'batch_size': 100
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2

    def test_invalid_yaml_continues_search(self, tmp_path, monkeypatch):
        """Test that invalid YAML doesn't break - continues to next candidate."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_file.write_text("invalid: yaml: content: [[[")
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        # Should return None (no valid config found)
        assert config is None

    def test_non_dict_yaml_continues_search(self, tmp_path, monkeypatch):
        """Test that YAML parsing to non-dict continues search."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_file.write_text("just a string")
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is None

    def test_missing_embeddings_section_continues(self, tmp_path, monkeypatch):
        """Test that missing embeddings section continues search."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {'other_section': {'key': 'value'}}
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is None

    def test_supports_both_embeddings_and_embedding_keys(self, tmp_path, monkeypatch):
        """Test that both 'embeddings' and 'embedding' keys work."""
        # Test with 'embedding' (singular)
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embedding': {  # singular
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2


class TestModelNameMapping:
    """Test model name to EmbeddingModel enum mapping."""

    def test_exact_match_all_models(self, tmp_path, monkeypatch):
        """Test exact matching for all supported models."""
        test_cases = [
            ('all-MiniLM-L6-v2', EmbeddingModel.MINI_LM_L6_V2),
            ('sentence-transformers/all-MiniLM-L6-v2', EmbeddingModel.MINI_LM_L6_V2),
            ('text-embedding-ada-002', EmbeddingModel.OPENAI_ADA_002),
            ('text-embedding-3-small', EmbeddingModel.OPENAI_3_SMALL),
            ('text-embedding-3-large', EmbeddingModel.OPENAI_3_LARGE),
        ]

        for model_name, expected_enum in test_cases:
            config_file = tmp_path / ".ctxrc.yaml"
            config_data = {
                'embeddings': {
                    'model': model_name,
                    'dimensions': 384
                }
            }
            config_file.write_text(yaml.dump(config_data))
            monkeypatch.chdir(tmp_path)

            config = _load_config_from_file()

            assert config is not None, f"Failed to parse config for model: {model_name}"
            assert config.model == expected_enum, f"Wrong enum for model: {model_name}"

    def test_unknown_model_uses_default(self, tmp_path, monkeypatch):
        """Test that unknown model name uses default."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'unknown-model-name',
                'dimensions': 384
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        # Should fall back to default
        assert config.model == EmbeddingModel.MINI_LM_L6_V2

    def test_model_name_map_coverage(self):
        """Test that _MODEL_NAME_MAP covers all expected variants."""
        assert 'all-MiniLM-L6-v2' in _MODEL_NAME_MAP
        assert 'sentence-transformers/all-MiniLM-L6-v2' in _MODEL_NAME_MAP
        assert 'text-embedding-ada-002' in _MODEL_NAME_MAP
        assert 'text-embedding-3-small' in _MODEL_NAME_MAP
        assert 'text-embedding-3-large' in _MODEL_NAME_MAP

        # Verify they map to correct enums
        assert _MODEL_NAME_MAP['all-MiniLM-L6-v2'] == EmbeddingModel.MINI_LM_L6_V2
        assert _MODEL_NAME_MAP['text-embedding-ada-002'] == EmbeddingModel.OPENAI_ADA_002


class TestPartialConfigHandling:
    """Test handling of partial/incomplete config."""

    def test_missing_dimensions_uses_default(self, tmp_path, monkeypatch):
        """Test that missing dimensions uses default value."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2'
                # dimensions missing
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.target_dimensions == 384  # default

    def test_missing_retries_uses_default(self, tmp_path, monkeypatch):
        """Test that missing max_retries uses default value."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384
                # max_retries missing
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.max_retries == 3  # default

    def test_all_fields_explicit(self, tmp_path, monkeypatch):
        """Test config with all fields explicitly set."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 768,
                'max_retries': 5,
                'timeout': 45.0,
                'batch_size': 200
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = _load_config_from_file()

        assert config is not None
        assert config.model == EmbeddingModel.MINI_LM_L6_V2
        assert config.target_dimensions == 768
        assert config.max_retries == 5
        assert config.timeout_seconds == 45.0
        assert config.batch_size == 200


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_io_error_continues_search(self, tmp_path, monkeypatch):
        """Test that IOError during file read continues search."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_file.write_text("embeddings: {model: 'all-MiniLM-L6-v2'}")

        # Make file unreadable
        config_file.chmod(0o000)
        monkeypatch.chdir(tmp_path)

        try:
            config = _load_config_from_file()
            # Should return None (couldn't read file)
            assert config is None
        finally:
            # Restore permissions for cleanup
            config_file.chmod(0o644)

    def test_invalid_type_in_config_continues(self, tmp_path, monkeypatch):
        """Test that invalid types in config continue search."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 'not-a-number',  # Invalid type
                'max_retries': 'also-not-a-number'
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        # Should handle ValueError/TypeError and continue
        config = _load_config_from_file()

        assert config is None


class TestServiceInitializationWithConfig:
    """Test service initialization with loaded config."""

    @pytest.mark.asyncio
    async def test_service_uses_loaded_config(self, tmp_path, monkeypatch):
        """Test that service initialization uses loaded config."""
        config_file = tmp_path / ".ctxrc.yaml"
        config_data = {
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 512,  # Non-default
                'max_retries': 7,    # Non-default
                'timeout': 90.0      # Non-default
            }
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        # Mock the get_embedding_service to avoid actual model loading
        with patch('src.embedding.service._load_config_from_file') as mock_load:
            # Return our test config
            test_config = EmbeddingConfig(
                model=EmbeddingModel.MINI_LM_L6_V2,
                target_dimensions=512,
                max_retries=7,
                timeout_seconds=90.0
            )
            mock_load.return_value = test_config

            # Reset the global service
            import src.embedding.service as svc
            svc._embedding_service = None

            # Mock initialize to avoid actual model loading
            with patch.object(EmbeddingService, 'initialize', return_value=True):
                service = await get_embedding_service()

                assert service.config.target_dimensions == 512
                assert service.config.max_retries == 7
                assert service.config.timeout_seconds == 90.0

    @pytest.mark.asyncio
    async def test_service_fallback_to_defaults_when_no_config(self, tmp_path, monkeypatch):
        """Test that service uses defaults when no config file exists."""
        monkeypatch.chdir(tmp_path)

        with patch('src.embedding.service._load_config_from_file', return_value=None):
            # Reset the global service
            import src.embedding.service as svc
            svc._embedding_service = None

            with patch.object(EmbeddingService, 'initialize', return_value=True):
                service = await get_embedding_service()

                # Should use default config
                assert service.config.target_dimensions == 384
                assert service.config.max_retries == 3
                assert service.config.timeout_seconds == 30.0


class TestEmojiLoggingConfiguration:
    """Test emoji logging configuration via environment variable."""

    def test_emoji_enabled_by_default(self):
        """Test that emoji logging is enabled by default."""
        from src.embedding.service import _USE_EMOJI_LOGGING

        # Default should be True
        assert _USE_EMOJI_LOGGING == True

    def test_emoji_can_be_disabled(self):
        """Test that emoji logging can be disabled via env var."""
        with patch.dict(os.environ, {'EMBEDDING_SERVICE_EMOJI_LOGGING': 'false'}):
            # Reload the module to pick up the env var
            import importlib
            import src.embedding.service as svc
            importlib.reload(svc)

            assert svc._USE_EMOJI_LOGGING == False

    def test_log_prefix_function(self):
        """Test _log_prefix function behavior."""
        from src.embedding.service import _log_prefix

        # When emoji enabled
        with patch('src.embedding.service._USE_EMOJI_LOGGING', True):
            result = _log_prefix('🔧', '[INIT]')
            assert result == '🔧'

        # When emoji disabled
        with patch('src.embedding.service._USE_EMOJI_LOGGING', False):
            result = _log_prefix('🔧', '[INIT]')
            assert result == '[INIT]'
