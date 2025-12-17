#!/usr/bin/env python3
"""
Unit tests for scripts/reindex_embeddings.py

Tests the EmbeddingReindexer class methods with mocked dependencies.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from reindex_embeddings import EmbeddingReindexer, main


class TestEmbeddingReindexerInit:
    """Tests for EmbeddingReindexer initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        reindexer = EmbeddingReindexer()
        assert reindexer.neo4j_host == "neo4j"
        assert reindexer.neo4j_port == 7687
        assert reindexer.neo4j_user == "neo4j"
        assert reindexer.qdrant_host == "qdrant"
        assert reindexer.qdrant_port == 6333

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        reindexer = EmbeddingReindexer(
            neo4j_host="custom-neo4j",
            neo4j_port=7688,
            neo4j_user="admin",
            neo4j_password="secret",
            qdrant_host="custom-qdrant",
            qdrant_port=6334,
        )
        assert reindexer.neo4j_host == "custom-neo4j"
        assert reindexer.neo4j_port == 7688
        assert reindexer.neo4j_password == "secret"


class TestEmbeddingReindexerInitialize:
    """Tests for the initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_connects_to_services(self):
        """Test that initialize connects to Neo4j, Qdrant, and EmbeddingService."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        with patch('reindex_embeddings.GraphDatabase') as mock_neo4j, \
             patch('reindex_embeddings.QdrantClient') as mock_qdrant, \
             patch('reindex_embeddings.EmbeddingService') as mock_embed_svc:

            # Setup mocks
            mock_driver = MagicMock()
            mock_neo4j.driver.return_value = mock_driver

            mock_qdrant_client = MagicMock()
            mock_qdrant.return_value = mock_qdrant_client

            mock_embed_instance = MagicMock()
            mock_embed_instance.initialize = AsyncMock()
            mock_embed_svc.return_value = mock_embed_instance

            await reindexer.initialize()

            # Verify connections
            mock_neo4j.driver.assert_called_once()
            mock_qdrant.assert_called_once()
            mock_embed_instance.initialize.assert_awaited_once()


class TestFetchDocuments:
    """Tests for fetch_documents method."""

    def test_fetch_documents_single_id(self):
        """Test fetching a single document by ID."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_session = MagicMock()
        mock_result = [
            {
                'id': 'test-id-123',
                'title': 'Test Document',
                'type': 'decision',
                'severity': 'Critical',
                'verdict': 'APPROVED',
                'proposal': None,
                'key_principle': None,
                'content_str': None,
                'searchable_text': 'test text'
            }
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        reindexer._neo4j_driver = mock_driver

        documents = reindexer.fetch_documents(doc_id='test-id-123')

        assert len(documents) == 1
        assert documents[0]['id'] == 'test-id-123'
        assert documents[0]['content']['title'] == 'Test Document'
        assert documents[0]['content']['severity'] == 'Critical'

    def test_fetch_documents_parses_content_str(self):
        """Test that content_str JSON is parsed and merged."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_session = MagicMock()
        mock_result = [
            {
                'id': 'test-id-456',
                'title': 'Base Title',
                'type': None,
                'severity': None,
                'verdict': None,
                'proposal': None,
                'key_principle': None,
                'content_str': json.dumps({'extra_field': 'extra_value', 'title': 'Override Title'}),
                'searchable_text': None
            }
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        reindexer._neo4j_driver = mock_driver

        documents = reindexer.fetch_documents()

        # content_str should be parsed and merged
        assert documents[0]['content']['extra_field'] == 'extra_value'
        # content_str values override individual fields
        assert documents[0]['content']['title'] == 'Override Title'


class TestGenerateNewEmbedding:
    """Tests for generate_new_embedding method."""

    @pytest.mark.asyncio
    async def test_generate_embedding_calls_service(self):
        """Test that generate_new_embedding uses the embedding service."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_service = MagicMock()
        mock_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        reindexer._embedding_service = mock_service

        content = {'title': 'Test', 'type': 'log'}
        embedding = await reindexer.generate_new_embedding(content)

        assert embedding == [0.1, 0.2, 0.3]
        mock_service.generate_embedding.assert_awaited_once_with(content)


class TestPreviewTextExtraction:
    """Tests for preview_text_extraction method."""

    def test_preview_uses_public_method(self):
        """Test that preview_text_extraction uses the public method."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_service = MagicMock()
        mock_service.extract_text_for_embedding.return_value = "TITLE: Test\nTYPE: log"
        reindexer._embedding_service = mock_service

        content = {'title': 'Test', 'type': 'log'}
        result = reindexer.preview_text_extraction(content)

        assert result == "TITLE: Test\nTYPE: log"
        mock_service.extract_text_for_embedding.assert_called_once_with(content)


class TestUpdateQdrantEmbedding:
    """Tests for update_qdrant_embedding method."""

    @pytest.mark.asyncio
    async def test_update_uses_update_vectors(self):
        """Test that update uses update_vectors (not upsert) to preserve payload."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_client = MagicMock()
        mock_client.update_vectors = MagicMock()
        reindexer._qdrant_client = mock_client

        result = await reindexer.update_qdrant_embedding(
            doc_id='test-123',
            embedding=[0.1, 0.2, 0.3]
        )

        assert result is True
        mock_client.update_vectors.assert_called_once()
        # Verify it's NOT using upsert (which would overwrite payload)
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_handles_error(self):
        """Test that update returns False on error."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_client = MagicMock()
        mock_client.update_vectors.side_effect = Exception("Connection failed")
        reindexer._qdrant_client = mock_client

        result = await reindexer.update_qdrant_embedding(
            doc_id='test-123',
            embedding=[0.1, 0.2, 0.3]
        )

        assert result is False


class TestReindexDocument:
    """Tests for reindex_document method."""

    @pytest.mark.asyncio
    async def test_reindex_dry_run(self):
        """Test that dry run doesn't update anything."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_service = MagicMock()
        mock_service.extract_text_for_embedding.return_value = "test text"
        mock_service.generate_embedding = AsyncMock()
        reindexer._embedding_service = mock_service

        mock_client = MagicMock()
        reindexer._qdrant_client = mock_client

        doc = {'id': 'test-123', 'content': {'title': 'Test'}}
        result = await reindexer.reindex_document(doc, dry_run=True)

        assert result is True
        # Should NOT generate embedding or update Qdrant in dry run
        mock_service.generate_embedding.assert_not_awaited()
        mock_client.update_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_reindex_full_process(self):
        """Test full reindex process."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        mock_service = MagicMock()
        mock_service.extract_text_for_embedding.return_value = "TITLE: Test"
        mock_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        reindexer._embedding_service = mock_service

        mock_client = MagicMock()
        reindexer._qdrant_client = mock_client

        doc = {'id': 'test-123', 'content': {'title': 'Test'}}
        result = await reindexer.reindex_document(doc, dry_run=False)

        assert result is True
        mock_service.generate_embedding.assert_awaited_once()
        mock_client.update_vectors.assert_called_once()


class TestReindexAll:
    """Tests for reindex_all method."""

    @pytest.mark.asyncio
    async def test_reindex_all_processes_documents(self):
        """Test that reindex_all processes all fetched documents."""
        reindexer = EmbeddingReindexer(neo4j_password="test")

        # Mock fetch_documents
        reindexer.fetch_documents = MagicMock(return_value=[
            {'id': 'doc-1', 'content': {'title': 'Doc 1'}},
            {'id': 'doc-2', 'content': {'title': 'Doc 2'}},
        ])

        # Mock reindex_document
        reindexer.reindex_document = AsyncMock(return_value=True)

        await reindexer.reindex_all(dry_run=True)

        assert reindexer.reindex_document.await_count == 2


class TestMain:
    """Tests for main() function."""

    @pytest.mark.asyncio
    async def test_main_requires_password(self):
        """Test that main exits if NEO4J_PASSWORD not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.argv', ['reindex_embeddings.py']):
                with pytest.raises(SystemExit) as exc_info:
                    await main()
                assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_initializes_and_runs(self):
        """Test that main initializes reindexer and runs reindex_all."""
        with patch.dict(os.environ, {'NEO4J_PASSWORD': 'test'}):
            with patch('sys.argv', ['reindex_embeddings.py', '--dry-run']):
                with patch('reindex_embeddings.EmbeddingReindexer') as mock_class:
                    mock_instance = MagicMock()
                    mock_instance.initialize = AsyncMock()
                    mock_instance.reindex_all = AsyncMock()
                    mock_instance.close = MagicMock()
                    mock_class.return_value = mock_instance

                    await main()

                    mock_instance.initialize.assert_awaited_once()
                    mock_instance.reindex_all.assert_awaited_once()
                    mock_instance.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
