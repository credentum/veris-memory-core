#!/usr/bin/env python3
"""
Comprehensive tests for AsyncHashDiffEmbedder to achieve full coverage
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.storage.hash_diff_embedder import EmbeddingTask  # noqa: E402


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection",
            "embedding_model": "text-embedding-ada-002",
            "ssl": False,
            "timeout": 5,
        }
    }


@pytest.fixture
def mock_perf_config():
    """Mock performance configuration"""
    return {
        "vector_db": {
            "embedding": {
                "batch_size": 50,
                "max_retries": 2,
                "initial_retry_delay": 0.5,
                "retry_backoff_factor": 1.5,
                "request_timeout": 15,
            }
        }
    }


@pytest.fixture
def embedder(temp_dir, mock_config, mock_perf_config):
    """Create AsyncHashDiffEmbedder instance for testing"""
    # Create config files in a separate config directory to avoid interference
    config_dir = temp_dir / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    perf_file = config_dir / "perf.yaml"

    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)
    with open(perf_file, "w") as f:
        yaml.dump(mock_perf_config, f)

    embedder = AsyncHashDiffEmbedder(
        config_path=str(config_file), perf_config_path=str(perf_file), verbose=True
    )

    # Override cache path to temp directory
    embedder.hash_cache_path = temp_dir / "hash_cache.json"

    return embedder


@pytest.fixture
def embedding_task(temp_dir):
    """Create sample embedding task"""
    file_path = temp_dir / "test_doc.yaml"
    return EmbeddingTask(
        file_path=file_path,
        document_id="test-001",
        content="test content",
        data={
            "id": "test-001",
            "title": "Test Document",
            "description": "A test document",
            "content": "Test content here",
            "document_type": "design",
            "created_date": "2025-07-12",
        },
    )


@pytest.fixture
def docs_dir(temp_dir):
    """Create clean directory for document tests"""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()
    return docs_dir


class TestEmbeddingTask:
    """Test EmbeddingTask dataclass"""

    def test_embedding_task_creation(self, temp_dir) -> None:
        """Test EmbeddingTask creation and attributes"""
        file_path = temp_dir / "test.yaml"
        task = EmbeddingTask(
            file_path=file_path,
            document_id="test-123",
            content="sample content",
            data={"title": "Test", "type": "document"},
        )

        assert task.file_path == file_path
        assert task.document_id == "test-123"
        assert task.content == "sample content"
        assert task.data["title"] == "Test"


class TestAsyncHashDiffEmbedder:
    """Comprehensive tests for AsyncHashDiffEmbedder"""

    def test_init_with_default_paths(self) -> None:
        """Test initialization with default config paths"""
        # Use non-existent paths to test defaults
        with patch.object(Path, "exists", return_value=False):
            embedder = AsyncHashDiffEmbedder(
                config_path="nonexistent.yaml", perf_config_path="nonexistent_perf.yaml"
            )

            assert embedder.config == {}  # Config file doesn't exist
            assert embedder.perf_config == {"vector_db": {"embedding": {}}}
            assert embedder.embedding_model == "text-embedding-ada-002"
            assert embedder.verbose is False
            assert embedder.batch_size == 100  # Default
            assert embedder.max_retries == 3  # Default

    def test_init_with_custom_config(self, embedder, mock_config, mock_perf_config) -> None:
        """Test initialization with custom configuration"""
        assert embedder.config == mock_config
        assert embedder.perf_config == mock_perf_config
        assert embedder.embedding_model == "text-embedding-ada-002"
        assert embedder.verbose is True
        assert embedder.batch_size == 50  # From perf config
        assert embedder.max_retries == 2  # From perf config
        assert embedder.initial_retry_delay == 0.5
        assert embedder.retry_backoff_factor == 1.5
        assert embedder.request_timeout == 15

    def test_load_config_file_not_found(self) -> None:
        """Test loading config when file doesn't exist"""
        embedder = AsyncHashDiffEmbedder()
        config = embedder._load_config("nonexistent.yaml")
        assert config == {}

    def test_load_perf_config_file_not_found(self) -> None:
        """Test loading perf config when file doesn't exist"""
        embedder = AsyncHashDiffEmbedder()
        config = embedder._load_perf_config("nonexistent.yaml")
        assert config == {"vector_db": {"embedding": {}}}

    def test_load_hash_cache_no_file(self, embedder) -> None:
        """Test loading hash cache when file doesn't exist"""
        # Make sure the cache file doesn't exist
        embedder.hash_cache_path = Path("nonexistent_cache.json")
        cache = embedder._load_hash_cache()
        assert cache == {}

    def test_load_hash_cache_with_file(self, embedder, temp_dir) -> None:
        """Test loading hash cache when file exists"""
        cache_data = {"test.yaml": {"document_id": "test", "content_hash": "abc123"}}

        embedder.hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(embedder.hash_cache_path, "w") as f:
            json.dump(cache_data, f)

        cache = embedder._load_hash_cache()
        assert cache == cache_data

    def test_load_hash_cache_invalid_json(self, embedder, temp_dir) -> None:
        """Test loading hash cache with invalid JSON"""
        embedder.hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(embedder.hash_cache_path, "w") as f:
            f.write("invalid json{")

        cache = embedder._load_hash_cache()
        assert cache == {}


class TestAsyncConnection:
    """Test async connection functionality"""

    @pytest.mark.asyncio
    async def test_connect_success(self, embedder) -> None:
        """Test successful connection"""
        with patch("storage.hash_diff_embedder_async.AsyncQdrantClient") as mock_qdrant:
            with patch("storage.hash_diff_embedder_async.AsyncOpenAI") as mock_openai:
                with patch("os.getenv", return_value="test-api-key"):
                    with patch("core.utils.get_secure_connection_config") as mock_get_config:
                        # Mock config
                        mock_get_config.return_value = {
                            "host": "localhost",
                            "port": 6333,
                            "ssl": False,
                            "timeout": 5,
                        }

                        # Mock Qdrant client
                        mock_client = AsyncMock()
                        mock_client.get_collections = AsyncMock(return_value=[])
                        mock_qdrant.return_value = mock_client

                        # Mock OpenAI client
                        mock_openai.return_value = AsyncMock()

                        result = await embedder.connect()

                        assert result is True
                        assert embedder.client == mock_client
                        assert embedder.openai_client is not None
                        mock_client.get_collections.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_connect_no_openai_key(self, embedder) -> None:
        """Test connection failure when OpenAI key is missing"""
        with patch("storage.hash_diff_embedder_async.AsyncQdrantClient") as mock_qdrant:
            with patch("os.getenv", return_value=None):
                with patch("core.utils.get_secure_connection_config") as mock_get_config:
                    mock_get_config.return_value = {"host": "localhost", "port": 6333}

                    # Mock Qdrant client
                    mock_client = AsyncMock()
                    mock_client.get_collections = AsyncMock(return_value=[])
                    mock_qdrant.return_value = mock_client

                    result = await embedder.connect()

                    assert result is False
                    assert embedder.openai_client is None

    @pytest.mark.asyncio
    async def test_connect_qdrant_failure(self, embedder) -> None:
        """Test connection failure when Qdrant fails"""
        with patch("storage.hash_diff_embedder_async.AsyncQdrantClient") as mock_qdrant:
            with patch("core.utils.get_secure_connection_config") as mock_get_config:
                mock_get_config.return_value = {"host": "localhost", "port": 6333}

                # Mock Qdrant client to raise exception
                mock_qdrant.side_effect = Exception("Connection failed")

                result = await embedder.connect()

                assert result is False
                assert embedder.client is None

    @pytest.mark.asyncio
    async def test_connect_qdrant_get_collections_failure(self, embedder) -> None:
        """Test connection failure when get_collections fails"""
        with patch("storage.hash_diff_embedder_async.AsyncQdrantClient") as mock_qdrant:
            with patch("core.utils.get_secure_connection_config") as mock_get_config:
                mock_get_config.return_value = {"host": "localhost", "port": 6333}

                # Mock Qdrant client that fails on get_collections
                mock_client = AsyncMock()
                mock_client.get_collections = AsyncMock(side_effect=Exception("API error"))
                mock_qdrant.return_value = mock_client

                result = await embedder.connect()

                assert result is False


class TestEmbeddingWithRetry:
    """Test embedding with retry logic"""

    @pytest.mark.asyncio
    async def test_embed_with_retry_success(self, embedder) -> None:
        """Test successful embedding on first try"""
        # Mock OpenAI client
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]

        embedder.openai_client = AsyncMock()
        embedder.openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await embedder._embed_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        embedder.openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002", input="test text", timeout=15
        )

    @pytest.mark.asyncio
    async def test_embed_with_retry_rate_limit_success(self, embedder) -> None:
        """Test successful embedding after rate limit retry"""
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]

        embedder.openai_client = AsyncMock()
        embedder.openai_client.embeddings.create = AsyncMock(
            side_effect=[Exception("rate_limit exceeded"), mock_response]
        )

        result = await embedder._embed_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        assert embedder.openai_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_with_retry_max_retries_exceeded(self, embedder) -> None:
        """Test failure when max retries exceeded"""
        embedder.openai_client = AsyncMock()
        embedder.openai_client.embeddings.create = AsyncMock(
            side_effect=Exception("rate_limit exceeded")
        )

        with pytest.raises(Exception, match="rate_limit exceeded"):
            await embedder._embed_with_retry("test text")

        assert embedder.openai_client.embeddings.create.call_count == embedder.max_retries

    @pytest.mark.asyncio
    async def test_embed_with_retry_non_rate_limit_error(self, embedder) -> None:
        """Test immediate failure on non-rate-limit error"""
        embedder.openai_client = AsyncMock()
        embedder.openai_client.embeddings.create = AsyncMock(
            side_effect=Exception("Invalid API key")
        )

        with pytest.raises(Exception, match="Invalid API key"):
            await embedder._embed_with_retry("test text")

        assert embedder.openai_client.embeddings.create.call_count == 1


class TestProcessEmbeddingTask:
    """Test embedding task processing"""

    @pytest.mark.asyncio
    async def test_process_embedding_task_success(self, embedder, embedding_task) -> None:
        """Test successful embedding task processing"""
        # Mock the embedding process
        with patch.object(embedder, "_embed_with_retry", return_value=[0.1, 0.2, 0.3]):
            # Mock Qdrant client
            embedder.client = AsyncMock()
            embedder.client.upsert = AsyncMock()

            result = await embedder._process_embedding_task(embedding_task)

            assert result is not None
            assert result.startswith("test-001-")

            # Check cache was updated
            cache_key = str(embedding_task.file_path)
            assert cache_key in embedder.hash_cache
            assert embedder.hash_cache[cache_key]["document_id"] == "test-001"

            # Verify upsert was called
            embedder.client.upsert.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_process_embedding_task_with_minimal_data(self, embedder, temp_dir) -> None:
        """Test processing task with minimal document data"""
        task = EmbeddingTask(
            file_path=temp_dir / "minimal.yaml",
            document_id="minimal-001",
            content="minimal content",
            data={"id": "minimal-001"},  # Only ID, no title/description/content
        )

        with patch.object(embedder, "_embed_with_retry", return_value=[0.1, 0.2]):
            embedder.client = AsyncMock()
            embedder.client.upsert = AsyncMock()

            result = await embedder._process_embedding_task(task)

            assert result is not None
            assert result.startswith("minimal-001-")

    @pytest.mark.asyncio
    async def test_process_embedding_task_failure(self, embedder, embedding_task) -> None:
        """Test embedding task processing failure"""
        with patch.object(embedder, "_embed_with_retry", side_effect=Exception("Embedding failed")):
            result = await embedder._process_embedding_task(embedding_task)

            assert result is None


class TestEmbedDirectory:
    """Test directory embedding functionality"""

    @pytest.mark.asyncio
    async def test_embed_directory_success(self, embedder, docs_dir) -> None:
        """Test successful directory embedding"""
        # Create test documents
        doc1_path = docs_dir / "doc1.yaml"
        doc2_path = docs_dir / "doc2.yaml"

        doc1_data = {
            "id": "doc-001",
            "title": "Document 1",
            "content": "Content 1",
            "document_type": "design",
        }
        doc2_data = {
            "id": "doc-002",
            "title": "Document 2",
            "content": "Content 2",
            "document_type": "decision",
        }

        with open(doc1_path, "w") as f:
            yaml.dump(doc1_data, f)
        with open(doc2_path, "w") as f:
            yaml.dump(doc2_data, f)

        # Mock embedding process
        with patch.object(embedder, "_process_embedding_task", return_value="test-vector-id"):
            embedded, total = await embedder.embed_directory(docs_dir)

            assert embedded == 2
            assert total == 2

    @pytest.mark.asyncio
    async def test_embed_directory_with_cache_hit(self, embedder, docs_dir) -> None:
        """Test directory embedding with cache hits"""
        # Create test document
        doc_path = docs_dir / "cached_doc.yaml"
        doc_data = {"id": "cached-001", "title": "Cached Document"}

        with open(doc_path, "w") as f:
            yaml.dump(doc_data, f)

        # Pre-populate cache with same content hash
        content = yaml.dump(doc_data)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        embedder.hash_cache[str(doc_path)] = {
            "content_hash": content_hash,
            "document_id": "cached-001",
        }

        embedded, total = await embedder.embed_directory(docs_dir)

        assert embedded == 0  # No new embeddings needed
        assert total == 1

    @pytest.mark.asyncio
    async def test_embed_directory_skips_schemas(self, embedder, docs_dir) -> None:
        """Test that schemas directory is skipped"""
        # Create schemas directory with file
        schemas_dir = docs_dir / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "schema.yaml"

        with open(schema_file, "w") as f:
            yaml.dump({"schema": "test"}, f)

        # Create regular document
        doc_path = docs_dir / "doc.yaml"
        with open(doc_path, "w") as f:
            yaml.dump({"id": "doc-001"}, f)

        with patch.object(embedder, "_process_embedding_task", return_value="test-id"):
            embedded, total = await embedder.embed_directory(docs_dir)

            assert embedded == 1  # Only the regular doc
            assert total == 2  # Both files counted

    @pytest.mark.asyncio
    async def test_embed_directory_skips_archive(self, embedder, docs_dir) -> None:
        """Test that archive directory is skipped"""
        # Create archive directory with file
        archive_dir = docs_dir / "archive"
        archive_dir.mkdir()
        archive_file = archive_dir / "archived.yaml"

        with open(archive_file, "w") as f:
            yaml.dump({"archived": "document"}, f)

        with patch.object(embedder, "_process_embedding_task", return_value="test-id"):
            embedded, total = await embedder.embed_directory(docs_dir)

            assert embedded == 0  # Archive file skipped
            assert total == 1

    @pytest.mark.asyncio
    async def test_embed_directory_invalid_yaml(self, embedder, docs_dir) -> None:
        """Test handling of invalid YAML files"""
        # Create invalid YAML file
        invalid_path = docs_dir / "invalid.yaml"
        with open(invalid_path, "w") as f:
            f.write("invalid: yaml: content: {")

        embedded, total = await embedder.embed_directory(docs_dir)

        assert embedded == 0
        assert total == 1

    @pytest.mark.asyncio
    async def test_embed_directory_empty_yaml(self, embedder, docs_dir) -> None:
        """Test handling of empty YAML files"""
        # Create empty YAML file
        empty_path = docs_dir / "empty.yaml"
        with open(empty_path, "w") as f:
            f.write("")

        embedded, total = await embedder.embed_directory(docs_dir)

        assert embedded == 0
        assert total == 1

    @pytest.mark.asyncio
    async def test_embed_directory_no_yaml_files(self, embedder, docs_dir) -> None:
        """Test directory with no YAML files"""
        # Create non-YAML file
        txt_file = docs_dir / "readme.txt"
        with open(txt_file, "w") as f:
            f.write("This is not a YAML file")

        embedded, total = await embedder.embed_directory(docs_dir)

        assert embedded == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_embed_directory_batch_processing(self, embedder, docs_dir) -> None:
        """Test batch processing with small batch size"""
        embedder.batch_size = 2  # Small batch size for testing

        # Create multiple documents
        for i in range(5):
            doc_path = docs_dir / f"doc{i}.yaml"
            with open(doc_path, "w") as f:
                yaml.dump({"id": f"doc-{i:03d}", "title": f"Document {i}"}, f)

        with patch.object(embedder, "_process_embedding_task", return_value="test-id"):
            embedded, total = await embedder.embed_directory(docs_dir)

            assert embedded == 5
            assert total == 5

    @pytest.mark.asyncio
    async def test_embed_directory_saves_cache(self, embedder, docs_dir) -> None:
        """Test that cache is saved to disk"""
        # Create test document
        doc_path = docs_dir / "test_doc.yaml"
        doc_data = {"id": "test-001", "title": "Test"}
        with open(doc_path, "w") as f:
            yaml.dump(doc_data, f)

        # Mock embedding process but let cache update happen
        async def mock_process_task(task):
            # Simulate successful processing and update cache like real method
            embedder.hash_cache[str(task.file_path)] = {
                "document_id": task.document_id,
                "content_hash": "test-hash",
                "embedding_hash": "test-embedding-hash",
                "vector_id": "test-vector-id",
                "last_embedded": "2025-07-12T00:00:00",
            }
            return "test-vector-id"

        with patch.object(embedder, "_process_embedding_task", side_effect=mock_process_task):
            await embedder.embed_directory(docs_dir)

            # Check cache file was created
            assert embedder.hash_cache_path.exists()

            # Verify cache content
            with open(embedder.hash_cache_path, "r") as f:
                saved_cache = json.load(f)

            assert str(doc_path) in saved_cache


class TestMainFunction:
    """Test main function and CLI"""

    @pytest.mark.asyncio
    async def test_main_function_success(self, temp_dir) -> None:
        """Test main function with successful execution"""
        # Create config files
        config_file = temp_dir / "config.yaml"
        perf_file = temp_dir / "perf.yaml"
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()

        with open(config_file, "w") as f:
            yaml.dump({"qdrant": {"host": "localhost"}}, f)
        with open(perf_file, "w") as f:
            yaml.dump({"vector_db": {"embedding": {}}}, f)

        # Create test document
        doc_path = docs_dir / "test.yaml"
        with open(doc_path, "w") as f:
            yaml.dump({"id": "test", "title": "Test Doc"}, f)

        # Mock sys.argv
        test_args = [
            "test_script.py",
            str(docs_dir),
            "--config",
            str(config_file),
            "--perf-config",
            str(perf_file),
            "--verbose",
        ]

        with patch("sys.argv", test_args):
            with patch(
                "src.storage.hash_diff_embedder_async.AsyncHashDiffEmbedder"
            ) as mock_embedder_class:
                # Mock embedder instance
                mock_embedder = AsyncMock()
                mock_embedder.connect.return_value = True
                mock_embedder.embed_directory.return_value = (1, 1)
                mock_embedder_class.return_value = mock_embedder

                # Import and run main
                from src.storage.hash_diff_embedder_async import main

                await main()

                # Verify embedder was created and used
                mock_embedder_class.assert_called_once_with()
                mock_embedder.connect.assert_called_once_with()
                mock_embedder.embed_directory.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_main_function_connection_failure(self, temp_dir) -> None:
        """Test main function when connection fails"""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()

        test_args = ["test_script.py", str(docs_dir)]

        with patch("sys.argv", test_args):
            with patch(
                "src.storage.hash_diff_embedder_async.AsyncHashDiffEmbedder"
            ) as mock_embedder_class:
                # Mock embedder that fails to connect
                mock_embedder = AsyncMock()
                mock_embedder.connect.return_value = False
                mock_embedder_class.return_value = mock_embedder

                from src.storage.hash_diff_embedder_async import main

                await main()

                # Verify connection was attempted but embedding was not
                mock_embedder.connect.assert_called_once_with()
                mock_embedder.embed_directory.assert_not_called()
