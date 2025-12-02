#!/usr/bin/env python3
"""
hash_diff_embedder.py: Hash-based diff embedder for context storage

This component:
1. Provides hash-based change detection for documents
2. Generates embeddings for changed content
3. Integrates with vector storage systems
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import click
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


@dataclass
class DocumentHash:
    """Track document hashes and metadata"""

    document_id: str
    file_path: str
    content_hash: str
    embedding_hash: str
    last_embedded: str
    vector_id: str


@dataclass
class EmbeddingTask:
    """Task for embedding"""

    file_path: Path
    document_id: str
    content: str
    data: Dict[str, Any]


class HashDiffEmbedder:
    """Hash-based diff embedder for efficient document processing"""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        perf_config_path: str = "performance.yaml",
        verbose: bool = False,
        num_hashes: int = 128,
        embedding_dim: int = 384,
    ):
        self.config = self._load_config(config_path)
        self.perf_config = self._load_perf_config(perf_config_path)
        self.hash_cache_path = Path("context/.embeddings_cache/hash_cache.json")
        self.hash_cache: Dict[str, Any] = self._load_hash_cache()
        self.client: Optional[QdrantClient] = None
        self.openai_client: Optional[OpenAI] = None
        self.embedding_model = self.config.get("qdrant", {}).get(
            "embedding_model", "text-embedding-ada-002"
        )
        self.verbose = verbose
        self.num_hashes = num_hashes
        self.embedding_dim = embedding_dim

        # Performance settings
        embed_config = self.perf_config.get("vector_db", {}).get("embedding", {})
        self.batch_size = embed_config.get("batch_size", 100)
        self.max_retries = embed_config.get("max_retries", 3)
        self.initial_retry_delay = embed_config.get("initial_retry_delay", 1.0)
        self.retry_backoff_factor = embed_config.get("retry_backoff_factor", 2.0)
        self.request_timeout = embed_config.get("request_timeout", 30)

        # Rate limiting
        # Rate limiting handled differently in sync version

        # Initialize hash functions for MinHash/SimHash
        self.hash_funcs = self._generate_hash_functions(num_hashes)

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _compute_embedding_hash(self, embedding: List[float]) -> str:
        """Compute hash of embedding vector"""
        # Convert to bytes and hash
        embedding_bytes = json.dumps(embedding, sort_keys=True).encode()
        return hashlib.sha256(embedding_bytes).hexdigest()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                result = yaml.safe_load(f)
                return cast(Dict[str, Any], result) if result else {}
        except FileNotFoundError:
            return {}

    def _load_perf_config(self, perf_config_path: str) -> Dict[str, Any]:
        """Load performance configuration"""
        try:
            with open(perf_config_path, "r") as f:
                result = yaml.safe_load(f)
                return cast(Dict[str, Any], result) if result else {"vector_db": {"embedding": {}}}
        except FileNotFoundError:
            return {"vector_db": {"embedding": {}}}

    def _load_hash_cache(self) -> Dict[str, Any]:
        """Load hash cache from disk"""
        if self.hash_cache_path.exists():
            try:
                with open(self.hash_cache_path, "r") as f:
                    result = json.load(f)
                    return cast(Dict[str, Any], result) if result else {}
            except Exception:
                pass
        return {}

    def _generate_hash_functions(self, num_hashes: int) -> List:
        """Generate hash functions for MinHash/SimHash algorithms.

        Args:
            num_hashes: Number of hash functions to generate

        Returns:
            List of hash functions
        """
        import random

        hash_funcs = []
        # Use different seeds for each hash function
        for i in range(num_hashes):
            seed_a = random.randint(1, 2**32 - 1)
            seed_b = random.randint(0, 2**32 - 1)

            def make_hash_func(a, b):
                def hash_func(x):
                    if isinstance(x, str):
                        x = int(hashlib.md5(x.encode()).hexdigest()[:8], 16)
                    elif isinstance(x, bytes):
                        x = int(hashlib.md5(x).hexdigest()[:8], 16)
                    return (a * x + b) % (2**32)

                return hash_func

            hash_funcs.append(make_hash_func(seed_a, seed_b))

        return hash_funcs

    def compute_minhash(self, tokens: List[str]) -> List[int]:
        """Compute MinHash signature for a set of tokens.

        Args:
            tokens: List of tokens/shingles

        Returns:
            MinHash signature
        """
        if not tokens:
            return [0] * self.num_hashes

        signature = []
        for hash_func in self.hash_funcs:
            min_hash = float("inf")
            for token in tokens:
                hash_val = hash_func(token)
                if hash_val < min_hash:
                    min_hash = hash_val
            signature.append(min_hash)

        return signature

    def compute_simhash(self, tokens: List[str]) -> int:
        """Compute SimHash for a set of tokens.

        Args:
            tokens: List of tokens

        Returns:
            SimHash value
        """
        if not tokens:
            return 0

        # Initialize vector
        v = [0] * 64  # 64-bit SimHash

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)

            for i in range(64):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        # Create fingerprint
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= 1 << i

        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two hashes.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Hamming distance
        """
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity from MinHash signatures.

        Args:
            sig1: First MinHash signature
            sig2: Second MinHash signature

        Returns:
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def connect(self) -> bool:
        """Connect to services asynchronously"""
        # Connect to Qdrant
        try:
            from core.utils import get_secure_connection_config
        except ImportError:
            from ..core.utils import get_secure_connection_config

        qdrant_config = get_secure_connection_config(self.config, "qdrant")

        try:
            self.client = QdrantClient(
                host=qdrant_config["host"],
                port=qdrant_config.get("port", 6333),
                https=qdrant_config.get("ssl", False),
                timeout=qdrant_config.get("timeout", 5),
            )

            # Test connection
            if self.client is not None:
                self.client.get_collections()

            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                click.echo("Error: OPENAI_API_KEY not set", err=True)
                return False

            self.openai_client = OpenAI(api_key=api_key)

            return True

        except Exception as e:
            click.echo(f"Failed to connect: {e}", err=True)
            return False

    def _embed_with_retry(self, text: str) -> List[float]:
        """Embed text with retry logic"""
        retry_delay = self.initial_retry_delay

        for attempt in range(self.max_retries):
            try:
                # Rate limiting removed - sync version doesn't use semaphore
                if self.openai_client is None:
                    raise Exception("OpenAI client not initialized")
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model, input=text, timeout=self.request_timeout
                )
                return list(response.data[0].embedding)

            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    if self.verbose:
                        click.echo(f"Rate limit hit, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= self.retry_backoff_factor
                else:
                    raise e

        raise Exception("Max retries exceeded")

    def _process_embedding_task(self, task: EmbeddingTask) -> Optional[str]:
        """Process a single embedding task"""
        try:
            # Prepare content
            content_parts = []
            if "title" in task.data:
                content_parts.append(f"Title: {task.data['title']}")
            if "description" in task.data:
                content_parts.append(f"Description: {task.data['description']}")
            if "content" in task.data:
                content_parts.append(f"Content: {task.data['content']}")

            embedding_text = "\n\n".join(content_parts)

            # Get embedding
            embedding = self._embed_with_retry(embedding_text)

            # Generate vector ID
            embedding_hash = hashlib.sha256(json.dumps(embedding).encode()).hexdigest()
            vector_id = f"{task.document_id}-{embedding_hash[:8]}"

            # Prepare payload
            payload = {
                "document_id": task.document_id,
                "document_type": task.data.get("document_type", "unknown"),
                "file_path": str(task.file_path),
                "title": task.data.get("title", ""),
                "created_date": task.data.get("created_date", ""),
                "last_modified": task.data.get("last_modified", ""),
                "content_hash": hashlib.sha256(task.content.encode()).hexdigest(),
                "embedding_hash": embedding_hash,
                "embedded_at": datetime.now().isoformat(),
            }

            # Store in Qdrant
            collection_name = self.config.get("qdrant", {}).get(
                "collection_name", "context_embeddings"
            )
            if self.client is not None:
                self.client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=vector_id, vector=embedding, payload=payload)],
                    wait=True,  # Ensure immediate availability for retrieval
                )

            # Update cache
            self.hash_cache[str(task.file_path)] = {
                "document_id": task.document_id,
                "content_hash": payload["content_hash"],
                "embedding_hash": embedding_hash,
                "vector_id": vector_id,
                "last_embedded": payload["embedded_at"],
            }

            if self.verbose:
                click.echo(f"  ✓ Embedded {task.file_path}")

            return vector_id

        except Exception as e:
            click.echo(f"  ✗ Failed to embed {task.file_path}: {e}", err=True)
            return None

    def embed_directory(self, directory: Path) -> Tuple[int, int]:
        """Embed all documents in directory asynchronously"""
        tasks: List[EmbeddingTask] = []

        # Collect tasks
        for yaml_file in directory.rglob("*.yaml"):
            if any(skip in yaml_file.parts for skip in ["schemas", ".embeddings_cache", "archive"]):
                continue

            try:
                with open(yaml_file, "r") as f:
                    content = f.read()
                    data = yaml.safe_load(content)

                if not data:
                    continue

                # Check if needs embedding
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                cache_key = str(yaml_file)

                if cache_key in self.hash_cache:
                    cached = self.hash_cache[cache_key]
                    if cached.get("content_hash") == content_hash:
                        continue

                task = EmbeddingTask(
                    file_path=yaml_file,
                    document_id=data.get("id", yaml_file.stem),
                    content=content,
                    data=data,
                )
                tasks.append(task)

            except Exception as e:
                click.echo(f"Error reading {yaml_file}: {e}", err=True)

        total_count = len(list(directory.rglob("*.yaml")))

        if not tasks:
            return 0, total_count

        # Process in batches
        embedded_count = 0
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i : i + self.batch_size]
            if self.verbose:
                click.echo(
                    f"\nProcessing batch {i // self.batch_size + 1}/"
                    f"{(len(tasks) + self.batch_size - 1) // self.batch_size}"
                )

            # Process batch concurrently
            results = []
            for task in batch:
                try:
                    result = self._process_embedding_task(task)
                    results.append(result)
                except Exception as e:
                    results.append(e)

            embedded_count += sum(1 for r in results if r and not isinstance(r, Exception))

        # Save cache
        self.hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_cache_path, "w") as f:
            f.write(json.dumps(self.hash_cache, indent=2))

        return embedded_count, total_count


async def main():
    """Main async entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Async document embedder")
    parser.add_argument("path", type=Path, default=Path("context"), help="Directory to process")
    parser.add_argument("--config", default=".ctxrc.yaml", help="Configuration file")
    parser.add_argument(
        "--perf-config",
        default="performance.yaml",
        help="Performance configuration file",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    embedder = HashDiffEmbedder(
        config_path=args.config, perf_config_path=args.perf_config, verbose=args.verbose
    )

    if not embedder.connect():
        return

    click.echo("=== Async Hash-Diff Embedder ===\n")

    start_time = time.time()
    embedded, total = embedder.embed_directory(args.path)
    elapsed = time.time() - start_time

    click.echo("\nResults:")
    click.echo(f"  Embedded: {embedded}/{total}")
    click.echo(f"  Time: {elapsed:.2f}s")
    click.echo(f"  Rate: {embedded / elapsed:.2f} docs/sec" if elapsed > 0 else "")


if __name__ == "__main__":
    asyncio.run(main())
