#!/usr/bin/env python3
"""
Re-Index Embeddings Script

Re-generates embeddings for all existing documents using the new contextual
headers extraction method. This is required after updating _extract_text()
to include metadata fields like severity, verdict, type, etc.

Usage:
    # Dry run (show what would be updated)
    python3 scripts/reindex_embeddings.py --dry-run

    # Re-index all documents
    python3 scripts/reindex_embeddings.py

    # Re-index specific document
    python3 scripts/reindex_embeddings.py --id "068dce90-b12d-4a88-80ad-b3ad34402564"

    # Re-index with batch size
    python3 scripts/reindex_embeddings.py --batch-size 50

Environment Variables:
    NEO4J_HOST: Neo4j host (default: neo4j)
    NEO4J_PORT: Neo4j port (default: 7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (required)
    QDRANT_HOST: Qdrant host (default: qdrant)
    QDRANT_PORT: Qdrant port (default: 6333)
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingReindexer:
    """Re-indexes embeddings for existing documents."""

    def __init__(
        self,
        neo4j_host: str = "neo4j",
        neo4j_port: int = 7687,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
    ):
        self.neo4j_host = neo4j_host
        self.neo4j_port = neo4j_port
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        self._neo4j_driver = None
        self._qdrant_client = None
        self._embedding_service = None

    async def initialize(self):
        """Initialize connections to Neo4j, Qdrant, and embedding service."""
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient

        # Neo4j connection
        uri = f"bolt://{self.neo4j_host}:{self.neo4j_port}"
        self._neo4j_driver = GraphDatabase.driver(
            uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        logger.info(f"Connected to Neo4j at {uri}")

        # Qdrant connection
        self._qdrant_client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )
        logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

        # Embedding service
        from embedding.service import EmbeddingService
        self._embedding_service = EmbeddingService()
        await self._embedding_service.initialize()
        logger.info("Embedding service initialized")

    def close(self):
        """Close connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()

    def fetch_documents(self, doc_id: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch documents from Neo4j with their content."""
        with self._neo4j_driver.session() as session:
            if doc_id:
                query = """
                MATCH (c:Context {id: $doc_id})
                RETURN c.id as id, c.title as title, c.type as type,
                       c.severity as severity, c.verdict as verdict,
                       c.proposal as proposal, c.key_principle as key_principle,
                       c.content as content_str, c.searchable_text as searchable_text
                """
                result = session.run(query, doc_id=doc_id)
            else:
                query = """
                MATCH (c:Context)
                RETURN c.id as id, c.title as title, c.type as type,
                       c.severity as severity, c.verdict as verdict,
                       c.proposal as proposal, c.key_principle as key_principle,
                       c.content as content_str, c.searchable_text as searchable_text
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

            documents = []
            for record in result:
                doc = dict(record)
                # Reconstruct content dict from individual fields
                content = {}
                if doc.get('title'):
                    content['title'] = doc['title']
                if doc.get('type'):
                    content['type'] = doc['type']
                if doc.get('severity'):
                    content['severity'] = doc['severity']
                if doc.get('verdict'):
                    content['verdict'] = doc['verdict']
                if doc.get('proposal'):
                    content['proposal'] = doc['proposal']
                if doc.get('key_principle'):
                    content['key_principle'] = doc['key_principle']

                # Try to parse content_str if it exists
                if doc.get('content_str'):
                    try:
                        parsed = json.loads(doc['content_str'])
                        if isinstance(parsed, dict):
                            content.update(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass

                doc['content'] = content
                documents.append(doc)

            return documents

    async def generate_new_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate embedding using the new contextual headers method."""
        embedding = await self._embedding_service.generate_embedding(content)
        return embedding

    def preview_text_extraction(self, content: Dict[str, Any]) -> str:
        """Preview what text will be extracted for embedding."""
        return self._embedding_service._extract_text(content)

    async def update_qdrant_embedding(
        self,
        doc_id: str,
        embedding: List[float],
        collection: str = "veris_memory"
    ) -> bool:
        """Update embedding in Qdrant."""
        from qdrant_client.models import PointStruct

        try:
            # Update the point with new embedding
            self._qdrant_client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={}  # Keep existing payload
                    )
                ],
                wait=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update Qdrant for {doc_id}: {e}")
            return False

    async def reindex_document(
        self,
        doc: Dict[str, Any],
        dry_run: bool = False
    ) -> bool:
        """Re-index a single document."""
        doc_id = doc['id']
        content = doc.get('content', {})

        # Show what text will be extracted
        extracted_text = self.preview_text_extraction(content)
        logger.info(f"\n{'='*60}")
        logger.info(f"Document: {doc_id}")
        logger.info(f"Title: {content.get('title', 'N/A')}")
        logger.info(f"Extracted text preview (first 200 chars):")
        logger.info(f"  {extracted_text[:200]}...")

        if dry_run:
            logger.info(f"[DRY RUN] Would re-index {doc_id}")
            return True

        # Generate new embedding
        try:
            embedding = await self.generate_new_embedding(content)
            logger.info(f"Generated embedding: {len(embedding)} dimensions")

            # Update Qdrant
            success = await self.update_qdrant_embedding(doc_id, embedding)
            if success:
                logger.info(f"✅ Updated {doc_id}")
            else:
                logger.error(f"❌ Failed to update {doc_id}")
            return success

        except Exception as e:
            logger.error(f"❌ Error re-indexing {doc_id}: {e}")
            return False

    async def reindex_all(
        self,
        doc_id: Optional[str] = None,
        batch_size: int = 100,
        dry_run: bool = False
    ):
        """Re-index all documents or a specific one."""
        documents = self.fetch_documents(doc_id=doc_id, limit=10000)
        total = len(documents)
        logger.info(f"Found {total} documents to re-index")

        success_count = 0
        fail_count = 0

        for i, doc in enumerate(documents):
            logger.info(f"\nProcessing {i+1}/{total}")
            if await self.reindex_document(doc, dry_run=dry_run):
                success_count += 1
            else:
                fail_count += 1

            # Small delay to avoid overwhelming services
            if not dry_run and (i + 1) % batch_size == 0:
                logger.info(f"Processed {i+1}/{total}, pausing briefly...")
                await asyncio.sleep(1)

        logger.info(f"\n{'='*60}")
        logger.info(f"Re-indexing complete!")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failed: {fail_count}")
        logger.info(f"  Total: {total}")


async def main():
    parser = argparse.ArgumentParser(description="Re-index embeddings with contextual headers")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating")
    parser.add_argument("--id", type=str, help="Re-index specific document by ID")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    args = parser.parse_args()

    # Get configuration from environment
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_password:
        logger.error("NEO4J_PASSWORD environment variable is required")
        sys.exit(1)

    reindexer = EmbeddingReindexer(
        neo4j_host=os.environ.get("NEO4J_HOST", "neo4j"),
        neo4j_port=int(os.environ.get("NEO4J_PORT", "7687")),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=neo4j_password,
        qdrant_host=os.environ.get("QDRANT_HOST", "qdrant"),
        qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
    )

    try:
        await reindexer.initialize()
        await reindexer.reindex_all(
            doc_id=args.id,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    finally:
        reindexer.close()


if __name__ == "__main__":
    asyncio.run(main())
