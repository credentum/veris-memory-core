#!/usr/bin/env python3
"""
Extended tests for hash_diff_embedder module to improve coverage
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import yaml

from src.storage.hash_diff_embedder import DocumentHash, HashDiffEmbedder  # noqa: E402


class TestHashDiffEmbedderCoverage:
    """Extended tests to improve coverage"""

    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("click.echo")
    def test_connect_missing_openai_key(self, mock_echo, mock_qdrant_client):
        """Test connection when OPENAI_API_KEY is missing"""
        # Mock successful Qdrant connection
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_qdrant

        embedder = HashDiffEmbedder()
        embedder.config = {"qdrant": {"host": "localhost", "port": 6333}}

        # Remove OPENAI_API_KEY from environment
        with patch.dict(os.environ, {}, clear=True):
            result = embedder.connect()

        assert result is False
        mock_echo.assert_called_with("Error: OPENAI_API_KEY environment variable not set", err=True)

    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("storage.hash_diff_embedder.openai.OpenAI")
    @patch("click.echo")
    def test_connect_openai_failure(self, mock_echo, mock_openai, mock_qdrant_client):
        """Test connection when OpenAI connection fails"""
        # Mock successful Qdrant connection
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_qdrant

        # Mock OpenAI connection failure
        mock_openai.side_effect = Exception("API connection failed")

        embedder = HashDiffEmbedder()
        embedder.config = {"qdrant": {"host": "localhost", "port": 6333}}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            result = embedder.connect()

        assert result is False
        assert any("Failed to connect to OpenAI" in str(call) for call in mock_echo.call_args_list)

    def test_needs_embedding_with_new_file(self):
        """Test needs_embedding for a new file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "new_doc.yaml"
            test_file.write_text("title: New Document\ncontent: Some content")

            embedder = HashDiffEmbedder()
            embedder.hash_cache = {}  # Empty cache

            needs_embed, vector_id = embedder.needs_embedding(test_file)

            assert needs_embed is True
            assert vector_id is None

    def test_needs_embedding_with_cached_unchanged_file(self):
        """Test needs_embedding for cached file with no changes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "cached_doc.yaml"
            content = "title: Cached Document\ncontent: Original content"
            test_file.write_text(content)

            embedder = HashDiffEmbedder()

            # Add to cache
            content_hash = embedder._compute_content_hash(content)
            embedder.hash_cache[str(test_file)] = DocumentHash(
                document_id="cached_doc",
                file_path=str(test_file),
                content_hash=content_hash,
                embedding_hash="embed123",
                last_embedded="2024-01-01T00:00:00",
                vector_id="vec123",
            )

            needs_embed, vector_id = embedder.needs_embedding(test_file)

            assert needs_embed is False
            assert vector_id == "vec123"

    def test_needs_embedding_with_cached_changed_file(self):
        """Test needs_embedding for cached file with changes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "changed_doc.yaml"
            test_file.write_text("title: Changed Document\ncontent: New content")

            embedder = HashDiffEmbedder()

            # Add to cache with different hash
            embedder.hash_cache[str(test_file)] = DocumentHash(
                document_id="changed_doc",
                file_path=str(test_file),
                content_hash="old_hash_value",
                embedding_hash="embed123",
                last_embedded="2024-01-01T00:00:00",
                vector_id="vec123",
            )

            needs_embed, vector_id = embedder.needs_embedding(test_file)

            assert needs_embed is True
            assert vector_id is None

    @patch("click.echo")
    def test_needs_embedding_with_error(self, mock_echo):
        """Test needs_embedding when file reading fails"""
        embedder = HashDiffEmbedder()

        # Try to check non-existent file
        non_existent = Path("/non/existent/file.yaml")
        needs_embed, vector_id = embedder.needs_embedding(non_existent)

        assert needs_embed is False
        assert vector_id is None
        assert any("Error checking" in str(call) for call in mock_echo.call_args_list)

    @patch("storage.hash_diff_embedder.openai.OpenAI")
    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("click.echo")
    def test_embed_document_verbose_skip(self, mock_echo, mock_qdrant, mock_openai):
        """Test embed_document with verbose mode when skipping"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "skip_doc.yaml"
            content = "title: Skip Document"
            test_file.write_text(content)

            embedder = HashDiffEmbedder(verbose=True)
            embedder.client = AsyncMock()

            # Add to cache to make it skip
            content_hash = embedder._compute_content_hash(content)
            embedder.hash_cache[str(test_file)] = DocumentHash(
                document_id="skip_doc",
                file_path=str(test_file),
                content_hash=content_hash,
                embedding_hash="embed123",
                last_embedded="2024-01-01T00:00:00",
                vector_id="existing_vec",
            )

            result = embedder.embed_document(test_file, force=False)

            assert result == "existing_vec"
            assert any(
                "Skipping" in str(call) and "no changes detected" in str(call)
                for call in mock_echo.call_args_list
            )

    @patch("storage.hash_diff_embedder.openai.OpenAI")
    @patch("click.echo")
    def test_embed_document_with_rate_limit_retry(self, mock_echo, mock_openai):
        """Test embed_document with rate limit and successful retry"""
        import openai as openai_module

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "retry_doc.yaml"
            test_file.write_text(
                yaml.dump(
                    {
                        "id": "retry_doc",
                        "title": "Retry Document",
                        "description": "Test description",
                        "content": "Test content",
                        "goals": ["goal1", "goal2"],
                    }
                )
            )

            # Mock OpenAI client with rate limit then success
            mock_client = AsyncMock()
            mock_embeddings = AsyncMock()

            # First call: rate limit error
            # Second call: success
            mock_response = AsyncMock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

            mock_embeddings.create.side_effect = [
                openai_module.RateLimitError("Rate limit", response=Mock(status_code=429), body={}),
                mock_response,
            ]

            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client

            embedder = HashDiffEmbedder(verbose=True)
            embedder.client = AsyncMock()
            embedder.config = {"qdrant": {"collection_name": "test_collection"}}

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                with patch("time.sleep"):  # Skip actual sleep
                    result = embedder.embed_document(test_file)

            assert result is not None
            assert mock_embeddings.create.call_count == 2
            assert any("Rate limit hit, retrying" in str(call) for call in mock_echo.call_args_list)

    @patch("storage.hash_diff_embedder.openai.OpenAI")
    def test_embed_document_with_existing_vector_deletion(self, mock_openai):
        """Test embed_document deletes old vector when updating"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "update_doc.yaml"
            test_file.write_text(yaml.dump({"id": "update_doc", "title": "Updated Document"}))

            # Mock OpenAI
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Mock Qdrant client
            mock_qdrant = AsyncMock()

            embedder = HashDiffEmbedder()
            embedder.client = mock_qdrant
            embedder.config = {"qdrant": {"collection_name": "test_collection"}}

            # Mock the needs_embedding to return the old vector ID
            # This simulates the case where we force re-embed the same content
            with patch.object(embedder, "needs_embedding", return_value=(True, "old_vector_id")):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                    embedder.embed_document(test_file, force=True)

            # Verify old vector was deleted
            mock_qdrant.delete.assert_called_once_with()
            call_args = mock_qdrant.delete.call_args
            # Check positional and keyword arguments
            assert call_args[1]["collection_name"] == "test_collection"
            assert call_args[1]["points_selector"] == ["old_vector_id"]

    def test_embed_directory_with_nested_files(self):
        """Test embed_directory with nested YAML files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            (Path(temp_dir) / "docs").mkdir()
            (Path(temp_dir) / "docs" / "schemas").mkdir()
            (Path(temp_dir) / ".embeddings_cache").mkdir()
            (Path(temp_dir) / "archive").mkdir()

            # Create test files
            files = [
                Path(temp_dir) / "doc1.yaml",
                Path(temp_dir) / "docs" / "doc2.yaml",
                Path(temp_dir) / "docs" / "schemas" / "schema.yaml",  # Should be skipped
                Path(temp_dir) / ".embeddings_cache" / "cache.yaml",  # Should be skipped
                Path(temp_dir) / "archive" / "old.yaml",  # Should be skipped
            ]

            for f in files:
                f.write_text(yaml.dump({"title": f"Document {f.name}"}))

            embedder = HashDiffEmbedder()
            embedder.client = AsyncMock()

            # Mock embed_document to track calls
            embed_calls = []

            def mock_embed(file_path, force=False):
                embed_calls.append(file_path)
                return f"vec_{file_path.name}"

            with patch.object(embedder, "embed_document", side_effect=mock_embed):
                embedded, total = embedder.embed_directory(Path(temp_dir))

                # Should only process doc1.yaml and docs/doc2.yaml
                assert total == 2
                assert embedded == 2
                assert len(embed_calls) == 2
                assert any("doc1.yaml" in str(p) for p in embed_calls)
                assert any("doc2.yaml" in str(p) for p in embed_calls)

    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("click.echo")
    def test_cleanup_orphaned_vectors(self, mock_echo, mock_qdrant_class):
        """Test cleanup_orphaned_vectors functionality"""
        # Create mock points
        mock_point1 = AsyncMock()
        mock_point1.id = "vec1"
        mock_point1.payload = {"file_path": "/non/existent/file1.yaml"}

        mock_point2 = AsyncMock()
        mock_point2.id = "vec2"
        mock_point2.payload = {"file_path": __file__}  # This file exists

        mock_point3 = AsyncMock()
        mock_point3.id = "vec3"
        mock_point3.payload = {"file_path": "/non/existent/file2.yaml"}

        # Mock Qdrant client
        mock_qdrant = AsyncMock()
        mock_qdrant.scroll.return_value = (
            [mock_point1, mock_point2, mock_point3],
            None,
        )

        embedder = HashDiffEmbedder(verbose=True)
        embedder.client = mock_qdrant
        embedder.config = {"qdrant": {"collection_name": "test_collection"}}

        # Add entries to cache
        embedder.hash_cache = {
            "/non/existent/file1.yaml": DocumentHash(
                document_id="doc1",
                file_path="/non/existent/file1.yaml",
                content_hash="hash1",
                embedding_hash="embed1",
                last_embedded="2024-01-01",
                vector_id="vec1",
            ),
            "/non/existent/file2.yaml": DocumentHash(
                document_id="doc3",
                file_path="/non/existent/file2.yaml",
                content_hash="hash3",
                embedding_hash="embed3",
                last_embedded="2024-01-01",
                vector_id="vec3",
            ),
        }

        removed = embedder.cleanup_orphaned_vectors()

        assert removed == 2
        # Verify delete was called for orphaned vectors
        assert mock_qdrant.delete.call_count == 2
        # Verify cache was updated
        assert len(embedder.hash_cache) == 0

    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("click.echo")
    def test_cleanup_orphaned_vectors_error(self, mock_echo, mock_qdrant_class):
        """Test cleanup_orphaned_vectors with error"""
        # Mock Qdrant client that raises error
        mock_qdrant = AsyncMock()
        mock_qdrant.scroll.side_effect = Exception("Connection error")

        embedder = HashDiffEmbedder()
        embedder.client = mock_qdrant
        embedder.config = {"qdrant": {"collection_name": "test_collection"}}

        removed = embedder.cleanup_orphaned_vectors()

        assert removed == 0
        assert any("Error during cleanup" in str(call) for call in mock_echo.call_args_list)

    def test_cleanup_orphaned_vectors_no_client(self):
        """Test cleanup_orphaned_vectors when client is None"""
        embedder = HashDiffEmbedder()
        embedder.client = None

        removed = embedder.cleanup_orphaned_vectors()

        assert removed == 0

    @patch("storage.hash_diff_embedder.HashDiffEmbedder.connect")
    @patch("storage.hash_diff_embedder.HashDiffEmbedder.embed_document")
    def test_main_single_file(self, mock_embed_doc, mock_connect):
        """Test main function with single file"""
        from src.storage.hash_diff_embedder import main

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.yaml"
            test_file.write_text("title: Test")

            mock_connect.return_value = True
            mock_embed_doc.return_value = "vec123"

            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [str(test_file)])

            assert result.exit_code == 0
            assert mock_embed_doc.called
            assert "✓ Embedded:" in result.output

    @patch("storage.hash_diff_embedder.HashDiffEmbedder.connect")
    @patch("storage.hash_diff_embedder.HashDiffEmbedder.embed_document")
    def test_main_single_file_failed(self, mock_embed_doc, mock_connect):
        """Test main function with single file that fails"""
        from src.storage.hash_diff_embedder import main

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.yaml"
            test_file.write_text("title: Test")

            mock_connect.return_value = True
            mock_embed_doc.return_value = None  # Embedding failed

            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [str(test_file)])

            assert result.exit_code == 0
            assert "✗ Failed to embed:" in result.output

    @patch("storage.hash_diff_embedder.HashDiffEmbedder.connect")
    @patch("storage.hash_diff_embedder.HashDiffEmbedder.embed_directory")
    @patch("storage.hash_diff_embedder.HashDiffEmbedder.cleanup_orphaned_vectors")
    def test_main_directory_with_cleanup(self, mock_cleanup, mock_embed_dir, mock_connect):
        """Test main function with directory and cleanup option"""
        from src.storage.hash_diff_embedder import main

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_connect.return_value = True
            mock_embed_dir.return_value = (3, 5)  # 3 embedded out of 5
            mock_cleanup.return_value = 2  # 2 orphaned vectors removed

            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [temp_dir, "--cleanup", "--verbose"])

            assert result.exit_code == 0
            assert mock_cleanup.called
            assert "Removed 2 orphaned vectors" in result.output
            assert "Embedded: 3/5" in result.output
            assert "Skipped: 2" in result.output

    @patch("storage.hash_diff_embedder.HashDiffEmbedder.connect")
    def test_main_connection_failure(self, mock_connect):
        """Test main function when connection fails"""
        from src.storage.hash_diff_embedder import main

        mock_connect.return_value = False

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(main, ["context"])

        assert result.exit_code == 0
        assert "Failed to connect to required services" in result.output
