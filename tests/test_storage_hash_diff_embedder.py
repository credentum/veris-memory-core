#!/usr/bin/env python3
"""
Comprehensive tests for Hash Diff Embedder - Phase 10 Coverage

This test module provides comprehensive coverage for the hash-based diff embedder
including document hashing, MinHash/SimHash algorithms, and embedding operations.
"""
import pytest
import tempfile
import yaml
import json
import os
import hashlib
import time
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List, Optional

# Import hash diff embedder components
try:
    from src.storage.hash_diff_embedder import (
        HashDiffEmbedder,
        DocumentHash,
        EmbeddingTask
    )
    HASH_DIFF_EMBEDDER_AVAILABLE = True
except ImportError:
    HASH_DIFF_EMBEDDER_AVAILABLE = False


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestDocumentHash:
    """Test DocumentHash dataclass functionality"""
    
    def test_document_hash_creation(self):
        """Test DocumentHash creation"""
        doc_hash = DocumentHash(
            document_id="test_doc_001",
            file_path="/path/to/document.yaml",
            content_hash="abc123def456",
            embedding_hash="def456ghi789",
            last_embedded="2024-01-15T10:30:00",
            vector_id="test_doc_001-def456gh"
        )
        
        assert doc_hash.document_id == "test_doc_001"
        assert doc_hash.file_path == "/path/to/document.yaml"
        assert doc_hash.content_hash == "abc123def456"
        assert doc_hash.embedding_hash == "def456ghi789"
        assert doc_hash.last_embedded == "2024-01-15T10:30:00"
        assert doc_hash.vector_id == "test_doc_001-def456gh"
    
    def test_document_hash_equality(self):
        """Test DocumentHash equality comparison"""
        doc_hash1 = DocumentHash(
            document_id="test_doc_001",
            file_path="/path/to/document.yaml",
            content_hash="abc123def456",
            embedding_hash="def456ghi789",
            last_embedded="2024-01-15T10:30:00",
            vector_id="test_doc_001-def456gh"
        )
        
        doc_hash2 = DocumentHash(
            document_id="test_doc_001",
            file_path="/path/to/document.yaml",
            content_hash="abc123def456",
            embedding_hash="def456ghi789",
            last_embedded="2024-01-15T10:30:00",
            vector_id="test_doc_001-def456gh"
        )
        
        assert doc_hash1 == doc_hash2
        
        # Test inequality
        doc_hash3 = DocumentHash(
            document_id="test_doc_002",
            file_path="/path/to/document.yaml",
            content_hash="abc123def456",
            embedding_hash="def456ghi789",
            last_embedded="2024-01-15T10:30:00",
            vector_id="test_doc_002-def456gh"
        )
        
        assert doc_hash1 != doc_hash3


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestEmbeddingTask:
    """Test EmbeddingTask dataclass functionality"""
    
    def test_embedding_task_creation(self):
        """Test EmbeddingTask creation"""
        task = EmbeddingTask(
            file_path=Path("/test/document.yaml"),
            document_id="test_doc_001",
            content="test content here",
            data={"title": "Test Document", "type": "documentation"}
        )
        
        assert task.file_path == Path("/test/document.yaml")
        assert task.document_id == "test_doc_001"
        assert task.content == "test content here"
        assert task.data["title"] == "Test Document"
        assert task.data["type"] == "documentation"
    
    def test_embedding_task_with_complex_data(self):
        """Test EmbeddingTask with complex data structure"""
        complex_data = {
            "title": "Complex Document",
            "description": "A complex test document",
            "content": "Detailed content here",
            "metadata": {
                "author": "Test Author",
                "created": "2024-01-15",
                "tags": ["test", "documentation", "example"]
            },
            "document_type": "technical_spec"
        }
        
        task = EmbeddingTask(
            file_path=Path("/test/complex.yaml"),
            document_id="complex_001",
            content=yaml.dump(complex_data),
            data=complex_data
        )
        
        assert task.data["metadata"]["author"] == "Test Author"
        assert len(task.data["metadata"]["tags"]) == 3
        assert task.data["document_type"] == "technical_spec"


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestHashDiffEmbedder:
    """Test HashDiffEmbedder basic functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Sample configuration
        self.test_config = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "collection_name": "test_context",
                "embedding_model": "text-embedding-ada-002"
            }
        }
        
        self.test_perf_config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 50,
                    "max_retries": 2,
                    "initial_retry_delay": 0.5,
                    "retry_backoff_factor": 1.5,
                    "request_timeout": 15
                }
            }
        }
        
        # Create embedder with test configuration
        with patch('builtins.open', mock_open(read_data=yaml.dump(self.test_config))):
            with patch('yaml.safe_load', return_value=self.test_config):
                with patch('builtins.open', mock_open(read_data=yaml.dump(self.test_perf_config))):
                    self.embedder = HashDiffEmbedder(
                        config_path="test_config.yaml",
                        perf_config_path="test_perf.yaml",
                        verbose=False,
                        num_hashes=64,
                        embedding_dim=256
                    )
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_embedder_initialization(self):
        """Test embedder initialization"""
        assert self.embedder is not None
        assert self.embedder.config == self.test_config
        assert self.embedder.perf_config == self.test_perf_config
        assert self.embedder.num_hashes == 64
        assert self.embedder.embedding_dim == 256
        assert self.embedder.batch_size == 50
        assert self.embedder.max_retries == 2
        assert len(self.embedder.hash_funcs) == 64
    
    def test_embedder_with_missing_configs(self):
        """Test embedder with missing configuration files"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            embedder = HashDiffEmbedder(
                config_path="missing_config.yaml",
                perf_config_path="missing_perf.yaml",
                verbose=True
            )
            
            # Should use default configurations
            assert isinstance(embedder.config, dict)
            assert isinstance(embedder.perf_config, dict)
            assert embedder.verbose is True
    
    def test_embedder_with_default_parameters(self):
        """Test embedder with default parameters"""
        with patch('builtins.open', mock_open(read_data="{}")):
            with patch('yaml.safe_load', return_value={}):
                embedder = HashDiffEmbedder()
                
                assert embedder.num_hashes == 128  # Default
                assert embedder.embedding_dim == 384  # Default
                assert embedder.batch_size == 100  # Default from empty perf config
                assert embedder.verbose is False  # Default
    
    def test_compute_content_hash(self):
        """Test content hash computation"""
        test_content = "This is test content for hashing"
        
        content_hash = self.embedder._compute_content_hash(test_content)
        
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64  # SHA-256 hex length
        
        # Same content should produce same hash
        content_hash2 = self.embedder._compute_content_hash(test_content)
        assert content_hash == content_hash2
        
        # Different content should produce different hash
        different_content = "This is different test content"
        different_hash = self.embedder._compute_content_hash(different_content)
        assert content_hash != different_hash
    
    def test_compute_embedding_hash(self):
        """Test embedding hash computation"""
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        embedding_hash = self.embedder._compute_embedding_hash(test_embedding)
        
        assert isinstance(embedding_hash, str)
        assert len(embedding_hash) == 64  # SHA-256 hex length
        
        # Same embedding should produce same hash
        embedding_hash2 = self.embedder._compute_embedding_hash(test_embedding)
        assert embedding_hash == embedding_hash2
        
        # Different embedding should produce different hash
        different_embedding = [0.6, 0.7, 0.8, 0.9, 1.0]
        different_hash = self.embedder._compute_embedding_hash(different_embedding)
        assert embedding_hash != different_hash
    
    def test_load_hash_cache(self):
        """Test hash cache loading"""
        # Test with existing cache
        cache_data = {
            "/test/doc1.yaml": {
                "document_id": "doc1",
                "content_hash": "hash1",
                "embedding_hash": "embed_hash1",
                "vector_id": "doc1-12345678",
                "last_embedded": "2024-01-15T10:30:00"
            }
        }
        
        cache_file = self.temp_path / "hash_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock the cache path
        with patch.object(self.embedder, 'hash_cache_path', cache_file):
            loaded_cache = self.embedder._load_hash_cache()
            
            assert isinstance(loaded_cache, dict)
            assert "/test/doc1.yaml" in loaded_cache
            assert loaded_cache["/test/doc1.yaml"]["document_id"] == "doc1"
    
    def test_load_hash_cache_corrupted(self):
        """Test hash cache loading with corrupted file"""
        # Create corrupted cache file
        cache_file = self.temp_path / "corrupted_cache.json"
        with open(cache_file, 'w') as f:
            f.write("invalid json content {")
        
        with patch.object(self.embedder, 'hash_cache_path', cache_file):
            loaded_cache = self.embedder._load_hash_cache()
            
            # Should return empty dict for corrupted cache
            assert loaded_cache == {}


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestMinHashSimHash:
    """Test MinHash and SimHash algorithms"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('builtins.open', mock_open(read_data="{}")):
            with patch('yaml.safe_load', return_value={}):
                self.embedder = HashDiffEmbedder(num_hashes=32)
    
    def test_generate_hash_functions(self):
        """Test hash function generation"""
        hash_funcs = self.embedder._generate_hash_functions(10)
        
        assert len(hash_funcs) == 10
        assert all(callable(func) for func in hash_funcs)
        
        # Test that hash functions work
        test_input = "test_string"
        hash_values = [func(test_input) for func in hash_funcs]
        
        assert len(hash_values) == 10
        assert all(isinstance(val, int) for val in hash_values)
        
        # Hash functions should produce different values
        assert len(set(hash_values)) > 1  # At least some should be different
    
    def test_compute_minhash(self):
        """Test MinHash computation"""
        tokens = ["apple", "banana", "cherry", "date", "elderberry"]
        
        minhash_sig = self.embedder.compute_minhash(tokens)
        
        assert isinstance(minhash_sig, list)
        assert len(minhash_sig) == self.embedder.num_hashes
        assert all(isinstance(val, int) for val in minhash_sig)
        
        # Same tokens should produce same signature
        minhash_sig2 = self.embedder.compute_minhash(tokens)
        assert minhash_sig == minhash_sig2
        
        # Different tokens should produce different signature
        different_tokens = ["grape", "honeydew", "kiwi", "lemon", "mango"]
        different_sig = self.embedder.compute_minhash(different_tokens)
        assert minhash_sig != different_sig
    
    def test_compute_minhash_empty_tokens(self):
        """Test MinHash computation with empty tokens"""
        empty_tokens = []
        
        minhash_sig = self.embedder.compute_minhash(empty_tokens)
        
        assert isinstance(minhash_sig, list)
        assert len(minhash_sig) == self.embedder.num_hashes
        assert all(val == 0 for val in minhash_sig)
    
    def test_compute_simhash(self):
        """Test SimHash computation"""
        tokens = ["machine", "learning", "artificial", "intelligence", "neural"]
        
        simhash_val = self.embedder.compute_simhash(tokens)
        
        assert isinstance(simhash_val, int)
        assert simhash_val >= 0
        
        # Same tokens should produce same hash
        simhash_val2 = self.embedder.compute_simhash(tokens)
        assert simhash_val == simhash_val2
        
        # Different tokens should produce different hash
        different_tokens = ["database", "storage", "query", "indexing", "search"]
        different_hash = self.embedder.compute_simhash(different_tokens)
        assert simhash_val != different_hash
    
    def test_compute_simhash_empty_tokens(self):
        """Test SimHash computation with empty tokens"""
        empty_tokens = []
        
        simhash_val = self.embedder.compute_simhash(empty_tokens)
        
        assert simhash_val == 0
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation"""
        hash1 = 0b11010110  # Binary: 214
        hash2 = 0b11000100  # Binary: 196
        
        distance = self.embedder.hamming_distance(hash1, hash2)
        
        # Expected Hamming distance: 2 (bits 1 and 4 differ)
        assert distance == 2
        
        # Distance to self should be 0
        self_distance = self.embedder.hamming_distance(hash1, hash1)
        assert self_distance == 0
        
        # Distance should be symmetric
        reverse_distance = self.embedder.hamming_distance(hash2, hash1)
        assert distance == reverse_distance
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation"""
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [1, 2, 3, 6, 7]  # 3 matches out of 5
        
        similarity = self.embedder.jaccard_similarity(sig1, sig2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity == 0.6  # 3/5 = 0.6
        
        # Identical signatures should have similarity 1.0
        identical_sim = self.embedder.jaccard_similarity(sig1, sig1)
        assert identical_sim == 1.0
        
        # Completely different signatures should have similarity 0.0
        sig3 = [10, 11, 12, 13, 14]
        different_sim = self.embedder.jaccard_similarity(sig1, sig3)
        assert different_sim == 0.0
    
    def test_jaccard_similarity_different_lengths(self):
        """Test Jaccard similarity with different length signatures"""
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [1, 2, 3]  # Different length
        
        similarity = self.embedder.jaccard_similarity(sig1, sig2)
        
        # Should return 0.0 for different length signatures
        assert similarity == 0.0


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestEmbedderConnection:
    """Test embedder connection functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5
            }
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(self.test_config))):
            with patch('yaml.safe_load', return_value=self.test_config):
                self.embedder = HashDiffEmbedder(config_path="test.yaml")
    
    @patch('src.storage.hash_diff_embedder.QdrantClient')
    @patch('src.storage.hash_diff_embedder.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    def test_connect_success(self, mock_openai, mock_qdrant):
        """Test successful connection to services"""
        # Mock successful Qdrant connection
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_collections.return_value = []
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Mock successful OpenAI connection
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock get_secure_connection_config
        with patch('src.storage.hash_diff_embedder.get_secure_connection_config') as mock_config:
            mock_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5
            }
            
            result = self.embedder.connect()
            
            assert result is True
            assert self.embedder.client == mock_qdrant_instance
            assert self.embedder.openai_client == mock_openai_instance
    
    @patch('src.storage.hash_diff_embedder.QdrantClient')
    @patch.dict(os.environ, {}, clear=True)
    def test_connect_missing_openai_key(self, mock_qdrant):
        """Test connection with missing OpenAI API key"""
        # Mock successful Qdrant connection
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_collections.return_value = []
        mock_qdrant.return_value = mock_qdrant_instance
        
        with patch('src.storage.hash_diff_embedder.get_secure_connection_config') as mock_config:
            mock_config.return_value = {"host": "localhost", "port": 6333}
            
            with patch('click.echo') as mock_echo:
                result = self.embedder.connect()
                
                assert result is False
                mock_echo.assert_called_with("Error: OPENAI_API_KEY not set", err=True)
    
    @patch('src.storage.hash_diff_embedder.QdrantClient')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    def test_connect_qdrant_failure(self, mock_qdrant):
        """Test connection with Qdrant failure"""
        # Mock Qdrant connection failure
        mock_qdrant.side_effect = Exception("Qdrant connection failed")
        
        with patch('src.storage.hash_diff_embedder.get_secure_connection_config') as mock_config:
            mock_config.return_value = {"host": "localhost", "port": 6333}
            
            with patch('click.echo') as mock_echo:
                result = self.embedder.connect()
                
                assert result is False
                mock_echo.assert_called_with("Failed to connect: Qdrant connection failed", err=True)


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestEmbeddingOperations:
    """Test embedding operations"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('builtins.open', mock_open(read_data="{}")):
            with patch('yaml.safe_load', return_value={}):
                self.embedder = HashDiffEmbedder(
                    verbose=False,
                    max_retries=2,
                    initial_retry_delay=0.1
                )
        
        # Mock OpenAI client
        self.mock_openai_client = MagicMock()
        self.embedder.openai_client = self.mock_openai_client
    
    def test_embed_with_retry_success(self):
        """Test successful embedding with retry logic"""
        test_text = "This is test text for embedding"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock successful OpenAI response
        mock_response = MagicMock()
        mock_response.data[0].embedding = expected_embedding
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        result = self.embedder._embed_with_retry(test_text)
        
        assert result == expected_embedding
        self.mock_openai_client.embeddings.create.assert_called_once()
    
    def test_embed_with_retry_rate_limit(self):
        """Test embedding with rate limit retry"""
        test_text = "This is test text for embedding"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock rate limit error then success
        mock_response = MagicMock()
        mock_response.data[0].embedding = expected_embedding
        
        self.mock_openai_client.embeddings.create.side_effect = [
            Exception("rate_limit exceeded"),  # First attempt fails
            mock_response  # Second attempt succeeds
        ]
        
        with patch('time.sleep') as mock_sleep:
            with patch('click.echo') as mock_echo:
                result = self.embedder._embed_with_retry(test_text)
                
                assert result == expected_embedding
                assert self.mock_openai_client.embeddings.create.call_count == 2
                mock_sleep.assert_called_once()
    
    def test_embed_with_retry_max_retries_exceeded(self):
        """Test embedding with max retries exceeded"""
        test_text = "This is test text for embedding"
        
        # Mock persistent failure
        self.mock_openai_client.embeddings.create.side_effect = Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            self.embedder._embed_with_retry(test_text)
        
        assert self.mock_openai_client.embeddings.create.call_count == self.embedder.max_retries
    
    def test_embed_with_retry_no_client(self):
        """Test embedding without initialized client"""
        self.embedder.openai_client = None
        test_text = "This is test text for embedding"
        
        with pytest.raises(Exception, match="OpenAI client not initialized"):
            self.embedder._embed_with_retry(test_text)
    
    def test_process_embedding_task_success(self):
        """Test successful embedding task processing"""
        # Create test task
        task_data = {
            "title": "Test Document",
            "description": "A test document for embedding",
            "content": "This is the main content of the document",
            "document_type": "documentation",
            "created_date": "2024-01-15",
            "last_modified": "2024-01-15"
        }
        
        task = EmbeddingTask(
            file_path=Path("/test/doc.yaml"),
            document_id="test_doc_001",
            content=yaml.dump(task_data),
            data=task_data
        )
        
        # Mock embedding response
        expected_embedding = [0.1] * 256
        mock_response = MagicMock()
        mock_response.data[0].embedding = expected_embedding
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        # Mock Qdrant client
        mock_qdrant_client = MagicMock()
        self.embedder.client = mock_qdrant_client
        
        with patch('click.echo') as mock_echo:
            result = self.embedder._process_embedding_task(task)
            
            assert isinstance(result, str)  # Should return vector ID
            assert result.startswith("test_doc_001-")
            
            # Verify Qdrant upsert was called
            mock_qdrant_client.upsert.assert_called_once()
            
            # Verify cache was updated
            cache_key = str(task.file_path)
            assert cache_key in self.embedder.hash_cache
            assert self.embedder.hash_cache[cache_key]["document_id"] == "test_doc_001"
    
    def test_process_embedding_task_failure(self):
        """Test embedding task processing failure"""
        task = EmbeddingTask(
            file_path=Path("/test/doc.yaml"),
            document_id="test_doc_001",
            content="test content",
            data={"title": "Test"}
        )
        
        # Mock embedding failure
        self.mock_openai_client.embeddings.create.side_effect = Exception("Embedding failed")
        
        with patch('click.echo') as mock_echo:
            result = self.embedder._process_embedding_task(task)
            
            assert result is None
            mock_echo.assert_called()  # Should log error


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestDirectoryEmbedding:
    """Test directory embedding functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory with test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test YAML files
        self.create_test_files()
        
        with patch('builtins.open', mock_open(read_data="{}")):
            with patch('yaml.safe_load', return_value={}):
                self.embedder = HashDiffEmbedder(batch_size=2, verbose=False)
        
        # Mock clients
        self.mock_openai_client = MagicMock()
        self.mock_qdrant_client = MagicMock()
        self.embedder.openai_client = self.mock_openai_client
        self.embedder.client = self.mock_qdrant_client
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test YAML files"""
        # Valid document 1
        doc1_data = {
            "id": "doc1",
            "title": "First Document",
            "description": "This is the first test document",
            "content": "Content of the first document",
            "document_type": "documentation"
        }
        
        doc1_path = self.temp_path / "doc1.yaml"
        with open(doc1_path, 'w') as f:
            yaml.dump(doc1_data, f)
        
        # Valid document 2
        doc2_data = {
            "id": "doc2",
            "title": "Second Document",
            "description": "This is the second test document",
            "content": "Content of the second document",
            "document_type": "tutorial"
        }
        
        doc2_path = self.temp_path / "doc2.yaml"
        with open(doc2_path, 'w') as f:
            yaml.dump(doc2_data, f)
        
        # Invalid YAML file
        invalid_path = self.temp_path / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # Empty YAML file
        empty_path = self.temp_path / "empty.yaml"
        with open(empty_path, 'w') as f:
            f.write("")
        
        # Create subdirectory with file
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        
        doc3_data = {
            "id": "doc3",
            "title": "Third Document",
            "content": "Content in subdirectory"
        }
        
        doc3_path = subdir / "doc3.yaml"
        with open(doc3_path, 'w') as f:
            yaml.dump(doc3_data, f)
        
        # Create schemas directory (should be skipped)
        schemas_dir = self.temp_path / "schemas"
        schemas_dir.mkdir()
        
        schema_path = schemas_dir / "schema.yaml"
        with open(schema_path, 'w') as f:
            yaml.dump({"type": "schema"}, f)
        
        # Create .embeddings_cache directory (should be skipped)
        cache_dir = self.temp_path / ".embeddings_cache"
        cache_dir.mkdir()
        
        cache_path = cache_dir / "cache.yaml"
        with open(cache_path, 'w') as f:
            yaml.dump({"cache": "data"}, f)
    
    def test_embed_directory_success(self):
        """Test successful directory embedding"""
        # Mock successful embedding
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 256
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        with patch('click.echo') as mock_echo:
            embedded_count, total_count = self.embedder.embed_directory(self.temp_path)
            
            # Should process 3 valid documents (doc1, doc2, doc3)
            # Should skip schemas, .embeddings_cache, invalid, and empty files
            assert embedded_count == 3
            assert total_count >= 3  # Total YAML files found
            
            # Verify Qdrant upsert was called for each embedded document
            assert self.mock_qdrant_client.upsert.call_count == 3
    
    def test_embed_directory_with_cache(self):
        """Test directory embedding with existing cache"""
        # Pre-populate cache for doc1
        doc1_path = str(self.temp_path / "doc1.yaml")
        with open(doc1_path, 'r') as f:
            content = f.read()
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        self.embedder.hash_cache[doc1_path] = {
            "document_id": "doc1",
            "content_hash": content_hash,  # Same hash - should skip
            "embedding_hash": "existing_hash",
            "vector_id": "doc1-12345678",
            "last_embedded": "2024-01-15T10:30:00"
        }
        
        # Mock successful embedding for non-cached documents
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 256
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        with patch('click.echo') as mock_echo:
            embedded_count, total_count = self.embedder.embed_directory(self.temp_path)
            
            # Should only embed 2 documents (doc2, doc3) - doc1 is cached
            assert embedded_count == 2
            
            # Verify Qdrant upsert was called only for non-cached documents
            assert self.mock_qdrant_client.upsert.call_count == 2
    
    def test_embed_directory_batch_processing(self):
        """Test directory embedding with batch processing"""
        # Set small batch size
        self.embedder.batch_size = 1
        
        # Mock successful embedding
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 256
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        with patch('click.echo') as mock_echo:
            embedded_count, total_count = self.embedder.embed_directory(self.temp_path)
            
            assert embedded_count == 3
            
            # Should process each document individually due to batch size 1
            assert self.mock_qdrant_client.upsert.call_count == 3
    
    def test_embed_directory_no_valid_documents(self):
        """Test directory embedding with no valid documents"""
        # Create directory with only invalid files
        empty_dir = self.temp_path / "empty_dir"
        empty_dir.mkdir()
        
        embedded_count, total_count = self.embedder.embed_directory(empty_dir)
        
        assert embedded_count == 0
        assert total_count == 0
    
    def test_embed_directory_partial_failures(self):
        """Test directory embedding with partial failures"""
        # Mock embedding that fails for some documents
        def mock_embedding_side_effect(*args, **kwargs):
            if "Second Document" in args[0]:
                raise Exception("Embedding failed for doc2")
            
            mock_response = MagicMock()
            mock_response.data[0].embedding = [0.1] * 256
            return mock_response
        
        self.mock_openai_client.embeddings.create.side_effect = mock_embedding_side_effect
        
        with patch('click.echo') as mock_echo:
            embedded_count, total_count = self.embedder.embed_directory(self.temp_path)
            
            # Should succeed for 2 out of 3 documents
            assert embedded_count == 2
            assert total_count >= 3
    
    def test_embed_directory_cache_save(self):
        """Test that cache is saved after directory embedding"""
        # Mock successful embedding
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 256
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        # Mock cache file creation
        cache_file_content = []
        
        def mock_write_cache(content):
            cache_file_content.append(content)
        
        mock_file = mock_open()
        mock_file.return_value.write = mock_write_cache
        
        with patch('builtins.open', mock_file):
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                embedded_count, total_count = self.embedder.embed_directory(self.temp_path)
                
                assert embedded_count == 3
                
                # Verify cache directory creation
                mock_mkdir.assert_called()
                
                # Verify cache file write
                assert len(cache_file_content) > 0


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestAsyncMainFunction:
    """Test async main function"""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.storage.hash_diff_embedder.HashDiffEmbedder')
    @patch('click.echo')
    def test_main_function_success(self, mock_echo, mock_embedder_class, mock_parse_args):
        """Test successful main function execution"""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.path = Path("test_context")
        mock_args.config = "test_config.yaml"
        mock_args.perf_config = "test_perf.yaml"
        mock_args.verbose = True
        mock_parse_args.return_value = mock_args
        
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.connect.return_value = True
        mock_embedder.embed_directory.return_value = (50, 100)  # embedded, total
        mock_embedder_class.return_value = mock_embedder
        
        # Import and run main
        import asyncio
        from src.storage.hash_diff_embedder import main
        
        with patch('time.time', side_effect=[0, 10]):  # Mock elapsed time
            asyncio.run(main())
        
        # Verify embedder was created and used
        mock_embedder_class.assert_called_once_with(
            config_path="test_config.yaml",
            perf_config_path="test_perf.yaml",
            verbose=True
        )
        mock_embedder.connect.assert_called_once()
        mock_embedder.embed_directory.assert_called_once_with(Path("test_context"))
        
        # Verify output messages
        mock_echo.assert_any_call("=== Async Hash-Diff Embedder ===\n")
        mock_echo.assert_any_call("\nResults:")
        mock_echo.assert_any_call("  Embedded: 50/100")
        mock_echo.assert_any_call("  Time: 10.00s")
        mock_echo.assert_any_call("  Rate: 5.00 docs/sec")
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.storage.hash_diff_embedder.HashDiffEmbedder')
    def test_main_function_connection_failure(self, mock_embedder_class, mock_parse_args):
        """Test main function with connection failure"""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.path = Path("test_context")
        mock_args.config = "test_config.yaml"
        mock_args.perf_config = "test_perf.yaml"
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # Mock embedder with connection failure
        mock_embedder = Mock()
        mock_embedder.connect.return_value = False
        mock_embedder_class.return_value = mock_embedder
        
        # Import and run main
        import asyncio
        from src.storage.hash_diff_embedder import main
        
        # Should return early due to connection failure
        asyncio.run(main())
        
        mock_embedder.connect.assert_called_once()
        # embed_directory should not be called due to connection failure
        mock_embedder.embed_directory.assert_not_called()


@pytest.mark.skipif(not HASH_DIFF_EMBEDDER_AVAILABLE, reason="Hash diff embedder not available")
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('builtins.open', mock_open(read_data="{}")):
            with patch('yaml.safe_load', return_value={}):
                self.embedder = HashDiffEmbedder()
    
    def test_hash_functions_with_different_inputs(self):
        """Test hash functions with various input types"""
        hash_func = self.embedder.hash_funcs[0]
        
        # Test with string
        string_result = hash_func("test_string")
        assert isinstance(string_result, int)
        
        # Test with bytes
        bytes_result = hash_func(b"test_bytes")
        assert isinstance(bytes_result, int)
        
        # Test with integer (should work directly)
        int_result = hash_func(12345)
        assert isinstance(int_result, int)
    
    def test_minhash_with_single_token(self):
        """Test MinHash with single token"""
        single_token = ["only_token"]
        
        minhash_sig = self.embedder.compute_minhash(single_token)
        
        assert len(minhash_sig) == self.embedder.num_hashes
        assert all(isinstance(val, int) for val in minhash_sig)
    
    def test_simhash_with_unicode_tokens(self):
        """Test SimHash with Unicode tokens"""
        unicode_tokens = ["café", "naïve", "résumé", "piñata", "mañana"]
        
        simhash_val = self.embedder.compute_simhash(unicode_tokens)
        
        assert isinstance(simhash_val, int)
        assert simhash_val >= 0
    
    def test_embedding_hash_with_large_embedding(self):
        """Test embedding hash with large embedding vector"""
        large_embedding = [0.1] * 1000  # Large embedding
        
        embedding_hash = self.embedder._compute_embedding_hash(large_embedding)
        
        assert isinstance(embedding_hash, str)
        assert len(embedding_hash) == 64  # SHA-256 hex length
    
    def test_embedding_hash_with_special_values(self):
        """Test embedding hash with special float values"""
        special_embedding = [float('inf'), float('-inf'), float('nan'), 0.0, -0.0]
        
        # Should handle special values without crashing
        try:
            embedding_hash = self.embedder._compute_embedding_hash(special_embedding)
            assert isinstance(embedding_hash, str)
        except (ValueError, TypeError):
            # Acceptable if implementation doesn't support special values
            pass
    
    def test_content_hash_with_large_content(self):
        """Test content hash with very large content"""
        large_content = "x" * 100000  # 100KB of content
        
        content_hash = self.embedder._compute_content_hash(large_content)
        
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64  # SHA-256 hex length
    
    def test_content_hash_with_unicode_content(self):
        """Test content hash with Unicode content"""
        unicode_content = "This contains Unicode: café, naïve, résumé 🚀🎉💯"
        
        content_hash = self.embedder._compute_content_hash(unicode_content)
        
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64  # SHA-256 hex length
    
    def test_hamming_distance_edge_cases(self):
        """Test Hamming distance with edge cases"""
        # Maximum distance (all bits different)
        max_distance = self.embedder.hamming_distance(0, 2**32 - 1)
        assert max_distance == 32
        
        # Large numbers
        large_num1 = 2**31 - 1
        large_num2 = 2**31
        distance = self.embedder.hamming_distance(large_num1, large_num2)
        assert distance == 32  # All bits flip
    
    def test_jaccard_similarity_edge_cases(self):
        """Test Jaccard similarity with edge cases"""
        # Empty signatures
        empty_sim = self.embedder.jaccard_similarity([], [])
        assert empty_sim == 1.0  # Both empty, considered identical
        
        # One empty, one non-empty
        mixed_sim = self.embedder.jaccard_similarity([], [1, 2, 3])
        assert mixed_sim == 0.0
    
    def test_config_loading_with_none_values(self):
        """Test config loading with None values in YAML"""
        config_with_none = {
            "qdrant": {
                "host": None,
                "port": 6333,
                "collection_name": None
            }
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(config_with_none))):
            with patch('yaml.safe_load', return_value=config_with_none):
                embedder = HashDiffEmbedder(config_path="test.yaml")
                
                # Should handle None values gracefully
                assert embedder.config == config_with_none
                assert embedder.embedding_model == "text-embedding-ada-002"  # Default