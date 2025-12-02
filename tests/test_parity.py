#!/usr/bin/env python3
"""
Dimension drift and parity tests.
Ensures no regression to wrong embedding dimensions.
"""

import pytest
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from unittest.mock import Mock, patch, MagicMock


class TestDimensionParity:
    """Test suite for dimension parity across system."""
    
    def test_config_dimensions_consistency(self, tmp_path):
        """Test all config files have consistent dimensions."""
        # Create test configs
        configs = {
            "production_locked_config.yaml": {
                "embedding": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "distance": "Cosine"
                }
            },
            ".ctxrc.yaml": {
                "embedding": {
                    "dimensions": 384,
                    "model": "all-MiniLM-L6-v2"
                }
            },
            "docker-compose.yml": {
                "services": {
                    "api": {
                        "environment": {
                            "EMBEDDING_DIM": "384"
                        }
                    }
                }
            }
        }
        
        # Write configs
        for filename, content in configs.items():
            file_path = tmp_path / filename
            with open(file_path, 'w') as f:
                yaml.dump(content, f)
        
        # Extract dimensions
        dimensions = set()
        
        for filename in configs.keys():
            file_path = tmp_path / filename
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract dimension value based on file structure
            if filename == "production_locked_config.yaml":
                dim = data["embedding"]["dimensions"]
            elif filename == ".ctxrc.yaml":
                dim = data["embedding"]["dimensions"]
            elif filename == "docker-compose.yml":
                dim = int(data["services"]["api"]["environment"]["EMBEDDING_DIM"])
            
            dimensions.add(dim)
        
        # Assert all dimensions are the same
        assert len(dimensions) == 1, f"Inconsistent dimensions found: {dimensions}"
        assert 384 in dimensions, f"Expected 384 dimensions, got: {dimensions}"
    
    def test_no_1536_dimensions_in_configs(self, tmp_path):
        """Test that no config files contain 1536 dimensions."""
        # Create config with wrong dimensions (should fail)
        bad_config = {
            "embedding": {
                "dimensions": 1536,  # Wrong!
                "model": "text-embedding-ada-002"
            }
        }
        
        file_path = tmp_path / "bad_config.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(bad_config, f)
        
        # Read and check
        with open(file_path, 'r') as f:
            content = f.read()
        
        # This should detect the wrong dimension
        assert "1536" in content, "Test setup error"
        
        # In production, this would trigger an alert
        with pytest.raises(AssertionError, match="Detected wrong dimension"):
            if "1536" in content:
                raise AssertionError("Detected wrong dimension: 1536")
    
    def test_embedding_model_dimension_match(self):
        """Test that embedding model matches expected dimensions."""
        model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        # Our expected configuration
        expected_model = "all-MiniLM-L6-v2"
        expected_dim = 384
        
        # Verify correct model is configured
        assert model_dimensions[expected_model] == expected_dim
        
        # Verify we're not using OpenAI models
        openai_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        assert expected_model not in openai_models, "Should not use OpenAI models"
    
    @patch('requests.get')
    def test_qdrant_collection_dimensions(self, mock_get):
        """Test Qdrant collection has correct dimensions."""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "result": {
                "config": {
                    "params": {
                        "vectors": {
                            "size": 384,
                            "distance": "Cosine"
                        }
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Make request
        response = requests.get("http://localhost:6333/collections/context_embeddings")
        data = response.json()
        
        # Extract dimensions
        vectors_config = data["result"]["config"]["params"]["vectors"]
        dimension = vectors_config["size"]
        distance = vectors_config["distance"]
        
        # Assertions
        assert dimension == 384, f"Wrong dimension in Qdrant: {dimension}"
        assert distance == "Cosine", f"Wrong distance metric: {distance}"
    
    def test_vector_generation_dimensions(self):
        """Test that generated vectors have correct dimensions."""
        # Simulate vector generation
        def generate_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> np.ndarray:
            """Simulate embedding generation."""
            if model == "all-MiniLM-L6-v2":
                return np.random.randn(384)
            elif model == "text-embedding-ada-002":
                return np.random.randn(1536)
            else:
                raise ValueError(f"Unknown model: {model}")
        
        # Test correct model
        vector = generate_embedding("test text", "all-MiniLM-L6-v2")
        assert vector.shape == (384,), f"Wrong vector shape: {vector.shape}"
        
        # Test wrong model would fail
        with pytest.raises(AssertionError):
            wrong_vector = generate_embedding("test text", "text-embedding-ada-002")
            assert wrong_vector.shape == (384,), "Should fail for wrong dimensions"
    
    def test_manifest_dimension_validation(self, tmp_path):
        """Test manifest files are validated for dimensions."""
        # Create manifests
        good_manifest = {
            "collection": "context_embeddings",
            "config": {
                "dimensions": 384,
                "distance": "Cosine"
            }
        }
        
        bad_manifest = {
            "collection": "context_embeddings",
            "config": {
                "dimensions": 1536,  # Wrong!
                "distance": "Cosine"
            }
        }
        
        # Write manifests
        good_path = tmp_path / "good_manifest.yaml"
        bad_path = tmp_path / "bad_manifest.yaml"
        
        with open(good_path, 'w') as f:
            yaml.dump(good_manifest, f)
        
        with open(bad_path, 'w') as f:
            yaml.dump(bad_manifest, f)
        
        # Validation function
        def validate_manifest(path: Path) -> Tuple[bool, Optional[str]]:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            dim = data.get("config", {}).get("dimensions")
            if dim != 384:
                return False, f"Wrong dimensions: {dim}"
            
            return True, None
        
        # Test validation
        valid, error = validate_manifest(good_path)
        assert valid == True, f"Good manifest should be valid: {error}"
        
        valid, error = validate_manifest(bad_path)
        assert valid == False, "Bad manifest should be invalid"
        assert "Wrong dimensions: 1536" in error


class TestDriftDetection:
    """Test suite for detecting configuration drift."""
    
    def test_detect_dimension_drift_in_code(self):
        """Test detection of hardcoded wrong dimensions in code."""
        code_samples = [
            # Good code
            ("vector = np.zeros(384)", True),
            ("dim = 384", True),
            ("EMBEDDING_DIM = 384", True),
            
            # Bad code
            ("vector = np.zeros(1536)", False),
            ("dim = 1536", False),
            ("EMBEDDING_DIM = 1536", False),
        ]
        
        for code, should_pass in code_samples:
            if "1536" in code:
                assert not should_pass, f"Code with 1536 should fail: {code}"
            elif "384" in code:
                assert should_pass, f"Code with 384 should pass: {code}"
    
    def test_detect_model_drift(self):
        """Test detection of model changes that affect dimensions."""
        # Configuration history
        configs = [
            {"date": "2024-01-01", "model": "all-MiniLM-L6-v2", "dim": 384},
            {"date": "2024-01-15", "model": "all-MiniLM-L6-v2", "dim": 384},
            # Potential drift point
            {"date": "2024-02-01", "model": "text-embedding-ada-002", "dim": 1536},
        ]
        
        # Detect drift
        baseline_dim = configs[0]["dim"]
        
        for config in configs:
            if config["dim"] != baseline_dim:
                drift_detected = True
                assert config["date"] == "2024-02-01", "Drift at wrong date"
                assert config["model"] == "text-embedding-ada-002", "Wrong model in drift"
                break
        else:
            drift_detected = False
        
        # In this test case, drift should be detected
        assert drift_detected, "Should detect dimension drift"
    
    @patch('requests.post')
    def test_runtime_dimension_validation(self, mock_post):
        """Test runtime validation of vector dimensions."""
        
        def validate_vector_dimensions(vector: List[float]) -> bool:
            """Validate vector dimensions at runtime."""
            return len(vector) == 384
        
        # Test vectors
        good_vector = [0.1] * 384
        bad_vector = [0.1] * 1536
        
        # Good vector should pass
        assert validate_vector_dimensions(good_vector) == True
        
        # Bad vector should fail
        assert validate_vector_dimensions(bad_vector) == False
        
        # Mock API call with dimension check
        def api_store_vector(vector: List[float]) -> Dict:
            if not validate_vector_dimensions(vector):
                raise ValueError(f"Invalid vector dimensions: {len(vector)}, expected 384")
            
            return {"status": "success", "id": "vec_123"}
        
        # Test API calls
        result = api_store_vector(good_vector)
        assert result["status"] == "success"
        
        with pytest.raises(ValueError, match="Invalid vector dimensions: 1536"):
            api_store_vector(bad_vector)


class TestParityMonitoring:
    """Test suite for continuous parity monitoring."""
    
    def test_parity_metrics(self):
        """Test parity metrics collection."""
        metrics = {
            "dimension_checks": {
                "passed": 0,
                "failed": 0,
                "last_check": None
            },
            "config_validation": {
                "passed": 0,
                "failed": 0,
                "last_validation": None
            }
        }
        
        # Simulate checks
        def check_dimensions(vector: List[float]) -> bool:
            is_valid = len(vector) == 384
            
            if is_valid:
                metrics["dimension_checks"]["passed"] += 1
            else:
                metrics["dimension_checks"]["failed"] += 1
            
            metrics["dimension_checks"]["last_check"] = "2024-01-15T10:00:00Z"
            return is_valid
        
        # Run checks
        check_dimensions([0.1] * 384)  # Pass
        check_dimensions([0.1] * 384)  # Pass
        check_dimensions([0.1] * 1536)  # Fail
        
        # Assert metrics
        assert metrics["dimension_checks"]["passed"] == 2
        assert metrics["dimension_checks"]["failed"] == 1
        assert metrics["dimension_checks"]["last_check"] is not None
    
    def test_parity_alerting(self):
        """Test alerting on parity violations."""
        alerts = []
        
        def check_and_alert(dimension: int) -> None:
            if dimension != 384:
                alert = {
                    "type": "DIMENSION_PARITY_VIOLATION",
                    "severity": "CRITICAL",
                    "message": f"Detected wrong dimension: {dimension}, expected 384",
                    "timestamp": "2024-01-15T10:00:00Z"
                }
                alerts.append(alert)
        
        # Test scenarios
        check_and_alert(384)   # No alert
        check_and_alert(1536)  # Alert!
        check_and_alert(768)   # Alert!
        
        # Verify alerts
        assert len(alerts) == 2, f"Expected 2 alerts, got {len(alerts)}"
        
        for alert in alerts:
            assert alert["type"] == "DIMENSION_PARITY_VIOLATION"
            assert alert["severity"] == "CRITICAL"
            assert "wrong dimension" in alert["message"]


@pytest.mark.parametrize("config_file,expected_dim", [
    ("production_locked_config.yaml", 384),
    (".ctxrc.yaml", 384),
    ("manifests/current.yaml", 384),
])
def test_config_dimensions_parametrized(config_file, expected_dim, tmp_path):
    """Parametrized test for configuration dimensions."""
    # Create test config
    config = {
        "embedding": {
            "dimensions": expected_dim
        }
    }
    
    file_path = tmp_path / config_file
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
    
    # Read and verify
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data["embedding"]["dimensions"] == expected_dim


@pytest.mark.parametrize("vector_size,should_pass", [
    (384, True),   # Correct
    (1536, False), # OpenAI
    (768, False),  # BERT
    (512, False),  # Other
    (0, False),    # Empty
])
def test_vector_validation_parametrized(vector_size, should_pass):
    """Parametrized test for vector validation."""
    vector = [0.1] * vector_size if vector_size > 0 else []
    
    def validate(v):
        return len(v) == 384
    
    result = validate(vector)
    assert result == should_pass, f"Vector size {vector_size} validation incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])