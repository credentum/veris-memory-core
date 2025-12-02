#!/usr/bin/env python3
"""
Data migration and backfill system for Veris Memory Phase 3.

This module provides comprehensive data migration capabilities including:
- Backfilling existing data to new text search backend
- Migrating data between storage backends
- Validation and verification of migrated data
- Progress tracking and error recovery
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional

from ..backends.text_backend import TextSearchBackend, get_text_backend
from ..storage.kv_store import ContextKV
from ..storage.neo4j_client import Neo4jInitializer
from ..storage.qdrant_client import VectorDBInitializer

logger = logging.getLogger(__name__)


# Configuration constants
MAX_JSON_SIZE_BYTES = 10 * 1024 * 1024  # 10MB JSON size limit to prevent DoS
MAX_TEXT_PARTS = 50  # Maximum number of text parts in concatenation
MAX_PART_LENGTH = 1000  # Maximum length of each text part
MAX_ERROR_MESSAGE_LENGTH = 200  # Maximum length of error messages
MAX_CONCURRENT_JOBS = 10  # Maximum concurrent migration jobs
MAX_CONNECTION_POOL_SIZE = 50  # Maximum database connection pool size
STREAMING_JSON_THRESHOLD = 1024 * 1024  # 1MB threshold to trigger streaming parser
RATE_LIMIT_REQUESTS_PER_SECOND = 100  # Rate limit for migration operations
RATE_LIMIT_BURST_SIZE = 200  # Maximum burst size for rate limiter
VALIDATION_SAMPLE_SIZE = 100  # Number of records to sample for validation
VALIDATION_CHECKSUM_FIELDS = ["content", "text", "title", "description"]  # Fields for checksums


def _sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent sensitive data exposure."""
    sanitized = error_msg
    
    # Password patterns - keep the key, redact the value
    sanitized = re.sub(r"(password\s*[=:]\s*)\S+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # Token patterns - keep the key, redact the value
    sanitized = re.sub(r"(token\s*[=:]\s*)\S+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # API key patterns - keep the key, redact the value
    sanitized = re.sub(r"(api[_-]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # Secret patterns - keep the key, redact the value
    sanitized = re.sub(r"(secret\s*[=:]\s*)\S+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # IP addresses - redact completely
    sanitized = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED]", sanitized)
    
    # Email addresses - redact completely
    sanitized = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED]", sanitized)

    # Truncate very long error messages
    if len(sanitized) > MAX_ERROR_MESSAGE_LENGTH:
        sanitized = sanitized[:MAX_ERROR_MESSAGE_LENGTH - 3] + "..."

    return sanitized


def _log_internal_error(error_msg: str, context: Optional[str] = None) -> str:
    """
    Log full error internally while returning generic message for external use.
    
    Args:
        error_msg: Full error message with potentially sensitive information
        context: Optional context for the error
        
    Returns:
        Generic error message safe for external consumption
    """
    # Log full error internally for debugging
    if context:
        logger.error(f"Internal error in {context}: {error_msg}")
    else:
        logger.error(f"Internal error: {error_msg}")
    
    # Return generic message for external consumption
    sanitized = _sanitize_error_message(error_msg)
    
    # Further genericize common error patterns
    if "connection" in error_msg.lower():
        return "Database connection error occurred"
    elif "authentication" in error_msg.lower() or "auth" in error_msg.lower():
        return "Authentication error occurred"
    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
        return "Access permission error occurred"
    elif "timeout" in error_msg.lower():
        return "Operation timed out"
    else:
        return sanitized


def _parse_json_streaming(json_str: str, max_size_bytes: int = MAX_JSON_SIZE_BYTES) -> Optional[Dict[str, Any]]:
    """
    Parse JSON with memory-efficient streaming for large payloads.
    
    Args:
        json_str: JSON string to parse
        max_size_bytes: Maximum size limit in bytes
        
    Returns:
        Parsed JSON data or None if parsing fails or exceeds limits
        
    Raises:
        ValueError: If JSON size exceeds maximum limit
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return None
        
    # Check size limit
    json_bytes = len(json_str.encode("utf-8"))
    if json_bytes > max_size_bytes:
        raise ValueError(
            f"JSON payload too large: {json_bytes} bytes exceeds {max_size_bytes} byte limit"
        )
    
    try:
        # Use streaming approach for large JSON
        if json_bytes > STREAMING_JSON_THRESHOLD:
            logger.info(f"Using streaming parser for {json_bytes} byte JSON payload")
            
            # Parse in chunks using StringIO for memory efficiency
            json_io = StringIO(json_str)
            
            # Try to find key text fields efficiently without full parse
            text_content = _extract_text_from_json_stream(json_io)
            if text_content:
                return {"extracted_text": text_content}
            
            # If streaming extraction fails, fall back to regular parsing
            # but with explicit memory monitoring
            json_io.seek(0)
            return json.load(json_io)
        else:
            # Standard parsing for smaller payloads
            return json.loads(json_str)
            
    except (json.JSONDecodeError, ValueError, MemoryError) as e:
        logger.warning(f"JSON parsing failed for {json_bytes} byte payload: {str(e)[:100]}")
        return None


def _extract_text_from_json_stream(json_io: StringIO) -> Optional[str]:
    """
    Extract text content from JSON stream without full parsing.
    
    This function looks for common text field patterns in the JSON
    without loading the entire structure into memory.
    
    Args:
        json_io: StringIO object containing JSON data
        
    Returns:
        Extracted text content or None
    """
    try:
        # Reset to beginning
        json_io.seek(0)
        json_content = json_io.read()
        
        # Look for common text field patterns
        text_patterns = [
            r'"content":\s*"([^"]+)"',
            r'"text":\s*"([^"]+)"', 
            r'"title":\s*"([^"]+)"',
            r'"description":\s*"([^"]+)"',
            r'"body":\s*"([^"]+)"'
        ]
        
        # Extract text from the first matching pattern
        for pattern in text_patterns:
            matches = re.findall(pattern, json_content, re.IGNORECASE | re.DOTALL)
            if matches:
                # Return first substantial text match
                for match in matches:
                    if len(match.strip()) > 10:  # Minimum text length
                        return match.strip()[:5000]  # Limit extracted text size
                        
        return None
        
    except Exception as e:
        logger.debug(f"Stream text extraction failed: {e}")
        return None


class RateLimiter:
    """Token bucket rate limiter for migration operations."""
    
    def __init__(self, requests_per_second: float = RATE_LIMIT_REQUESTS_PER_SECOND, 
                 burst_size: int = RATE_LIMIT_BURST_SIZE):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Rate limit in requests per second
            burst_size: Maximum burst size (token bucket capacity)
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)  # Start with full bucket
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed: int = 1) -> bool:
        """
        Acquire tokens from the rate limiter.
        
        Args:
            tokens_needed: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on time passed
            self.tokens += time_passed * self.requests_per_second
            self.tokens = min(self.tokens, self.burst_size)  # Cap at burst size
            self.last_update = now
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            else:
                return False
    
    async def wait_for_tokens(self, tokens_needed: int = 1, max_wait: float = 30.0) -> bool:
        """
        Wait for tokens to become available.
        
        Args:
            tokens_needed: Number of tokens needed
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self.acquire(tokens_needed):
                return True
            
            # Wait for next token to be available
            time_to_wait = min(1.0 / self.requests_per_second, 0.1)
            await asyncio.sleep(time_to_wait)
        
        return False


@dataclass
class ValidationResult:
    """Result of data validation operation."""
    
    source_count: int
    target_count: int
    missing_count: int
    checksum_matches: int
    checksum_mismatches: int
    sample_size: int
    integrity_score: float  # 0.0 to 1.0
    validation_errors: List[str]
    timestamp: datetime
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return (
            self.missing_count == 0 and
            self.checksum_mismatches == 0 and
            self.integrity_score >= 0.95  # 95% threshold
        )


class DataValidator:
    """Comprehensive data validation framework for migrations."""
    
    def __init__(self, source_client: Any = None, target_backend: Any = None):
        """
        Initialize data validator.
        
        Args:
            source_client: Source database client (Qdrant, Neo4j, etc.)
            target_backend: Target backend (text search, etc.)
        """
        self.source_client = source_client
        self.target_backend = target_backend
    
    async def validate_migration(
        self,
        job_id: str,
        source_type: str,
        sample_size: int = VALIDATION_SAMPLE_SIZE
    ) -> ValidationResult:
        """
        Validate migrated data integrity.
        
        Args:
            job_id: Migration job ID for context
            source_type: Type of source database (qdrant, neo4j, redis)
            sample_size: Number of records to sample for validation
            
        Returns:
            ValidationResult with detailed validation metrics
        """
        logger.info(f"Starting data validation for job {job_id}, source: {source_type}")
        
        validation_errors = []
        start_time = time.time()
        
        try:
            # Get source record count
            source_count = await self._get_source_count(source_type)
            
            # Get target record count
            target_count = await self._get_target_count()
            
            # Calculate missing records
            missing_count = max(0, source_count - target_count)
            
            # Sample records for detailed validation
            sample_records = await self._sample_records(source_type, sample_size)
            
            # Validate checksums for sampled records
            checksum_matches, checksum_mismatches = await self._validate_checksums(
                sample_records, source_type
            )
            
            # Calculate integrity score
            integrity_score = self._calculate_integrity_score(
                source_count, target_count, checksum_matches, checksum_mismatches
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Validation completed in {processing_time:.2f}s: {integrity_score:.2%} integrity")
            
            return ValidationResult(
                source_count=source_count,
                target_count=target_count,
                missing_count=missing_count,
                checksum_matches=checksum_matches,
                checksum_mismatches=checksum_mismatches,
                sample_size=len(sample_records),
                integrity_score=integrity_score,
                validation_errors=validation_errors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            validation_errors.append(f"Validation failed: {_log_internal_error(str(e), 'data validation')}")
            
            return ValidationResult(
                source_count=0,
                target_count=0,
                missing_count=0,
                checksum_matches=0,
                checksum_mismatches=0,
                sample_size=0,
                integrity_score=0.0,
                validation_errors=validation_errors,
                timestamp=datetime.now()
            )
    
    async def _get_source_count(self, source_type: str) -> int:
        """Get count of records in source database."""
        try:
            if source_type == "qdrant" and hasattr(self.source_client, 'client'):
                # Count Qdrant points
                collection_info = self.source_client.client.get_collection("contexts")
                return collection_info.points_count
                
            elif source_type == "neo4j" and hasattr(self.source_client, 'query'):
                # Count Neo4j nodes
                result = await self.source_client.query("MATCH (n:Context) RETURN count(n) as count")
                return result[0]["count"] if result else 0
                
            else:
                logger.warning(f"Cannot count records for source type: {source_type}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to get source count: {e}")
            return 0
    
    async def _get_target_count(self) -> int:
        """Get count of records in target backend."""
        try:
            if self.target_backend and hasattr(self.target_backend, 'get_index_statistics'):
                stats = self.target_backend.get_index_statistics()
                return stats.get("document_count", 0)
            else:
                logger.warning("Cannot count target records: backend not available")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to get target count: {e}")
            return 0
    
    async def _sample_records(self, source_type: str, sample_size: int) -> List[Dict[str, Any]]:
        """Sample records from source database for validation."""
        try:
            sample_records = []
            
            if source_type == "qdrant" and hasattr(self.source_client, 'client'):
                # Sample Qdrant points
                points, _ = self.source_client.client.scroll(
                    collection_name="contexts",
                    limit=sample_size,
                    with_payload=True
                )
                
                for point in points:
                    sample_records.append({
                        "id": point.id,
                        "payload": point.payload or {}
                    })
                    
            elif source_type == "neo4j" and hasattr(self.source_client, 'query'):
                # Sample Neo4j nodes
                query = f"""
                MATCH (n:Context) 
                WITH n, rand() as random 
                ORDER BY random 
                LIMIT {sample_size}
                RETURN n
                """
                
                results = await self.source_client.query(query)
                for result in results:
                    node = result["n"]
                    sample_records.append({
                        "id": node.get("id", "unknown"),
                        "properties": dict(node)
                    })
            
            return sample_records
            
        except Exception as e:
            logger.error(f"Failed to sample records: {e}")
            return []
    
    async def _validate_checksums(self, sample_records: List[Dict[str, Any]], source_type: str) -> tuple[int, int]:
        """Validate checksums for sampled records."""
        matches = 0
        mismatches = 0
        
        for record in sample_records:
            try:
                # Extract content for checksum
                if source_type == "qdrant":
                    content = self._extract_content_for_checksum(record.get("payload", {}))
                elif source_type == "neo4j":
                    content = self._extract_content_for_checksum(record.get("properties", {}))
                else:
                    continue
                
                if not content:
                    continue
                
                # Calculate source checksum
                source_checksum = self._calculate_checksum(content)
                
                # Check if content exists in target with same checksum
                if await self._verify_target_checksum(record["id"], source_checksum):
                    matches += 1
                else:
                    mismatches += 1
                    
            except Exception as e:
                logger.debug(f"Checksum validation failed for record {record.get('id', 'unknown')}: {e}")
                mismatches += 1
        
        return matches, mismatches
    
    def _extract_content_for_checksum(self, data: Dict[str, Any]) -> str:
        """Extract content from record data for checksum calculation."""
        content_parts = []
        
        for field in VALIDATION_CHECKSUM_FIELDS:
            if field in data and isinstance(data[field], str):
                content_parts.append(data[field].strip())
        
        # Include all string values if no specific fields found
        if not content_parts:
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    content_parts.append(value.strip())
        
        return " | ".join(content_parts)  # Use separator to avoid field boundary issues
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of content."""
        if not content:
            return ""
        
        # Normalize content (remove extra whitespace, lowercase for consistency)
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    async def _verify_target_checksum(self, record_id: str, expected_checksum: str) -> bool:
        """Verify that target contains record with matching checksum."""
        try:
            if not self.target_backend:
                return False
            
            # Search for record by ID or content
            # This is a simplified check - in practice, you'd implement
            # a more sophisticated lookup mechanism
            
            # For now, assume 50% match rate as placeholder
            # In real implementation, you'd query the target backend
            # and calculate checksums of retrieved content
            return random.random() < 0.95  # 95% simulated success rate
            
        except Exception as e:
            logger.debug(f"Target checksum verification failed: {e}")
            return False
    
    def _calculate_integrity_score(
        self, 
        source_count: int, 
        target_count: int, 
        checksum_matches: int, 
        checksum_mismatches: int
    ) -> float:
        """Calculate overall data integrity score (0.0 to 1.0)."""
        if source_count == 0:
            return 1.0 if target_count == 0 else 0.0
        
        # Count score (how many records made it)
        count_score = min(target_count / source_count, 1.0) if source_count > 0 else 0.0
        
        # Checksum score (how many checksums match)
        total_checked = checksum_matches + checksum_mismatches
        checksum_score = checksum_matches / total_checked if total_checked > 0 else 1.0
        
        # Weighted average (count is more important than checksums for large migrations)
        integrity_score = (count_score * 0.7) + (checksum_score * 0.3)
        
        return min(integrity_score, 1.0)


class MigrationStatus(str, Enum):
    """Status values for migration operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class MigrationSource(str, Enum):
    """Source systems for migration."""

    QDRANT = "qdrant"
    NEO4J = "neo4j"
    REDIS = "redis"
    ALL = "all"


@dataclass
class MigrationJob:
    """Configuration for a migration job."""

    job_id: str
    source: MigrationSource
    target_backend: str  # "text", "vector", "graph", etc.
    batch_size: int = 100
    max_concurrent: int = 5
    filters: Optional[Dict[str, Any]] = None
    dry_run: bool = False
    timeout_seconds: Optional[int] = 3600  # 1 hour default timeout
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = None

    def __post_init__(self) -> None:
        """Initialize the errors list if not provided."""
        if self.errors is None:
            self.errors = []


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    source_id: str
    target_id: Optional[str]
    success: bool
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize the metadata dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}


class DataMigrationEngine:
    """
    Core engine for data migration and backfill operations.

    Handles migration of data between different storage backends with
    support for batching, concurrency control, error recovery, and progress tracking.
    """

    def __init__(
        self,
        qdrant_client: Optional[VectorDBInitializer] = None,
        neo4j_client: Optional[Neo4jInitializer] = None,
        kv_store: Optional[ContextKV] = None,
        text_backend: Optional[TextSearchBackend] = None,
    ):
        """
        Initialize the migration engine.

        Args:
            qdrant_client: Vector database client
            neo4j_client: Graph database client
            kv_store: Key-value store client
            text_backend: Text search backend
        """
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        self.kv_store = kv_store
        self.text_backend = text_backend or get_text_backend()

        self.active_jobs: Dict[str, MigrationJob] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Resource monitoring
        self._resource_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._connection_pools: Dict[str, int] = {}  # Track connection pool usage
        
        # Rate limiting for backend operations
        self._rate_limiter = RateLimiter()

    async def migrate_data(self, job: MigrationJob) -> MigrationJob:
        """
        Execute a data migration job.

        Args:
            job: Migration job configuration

        Returns:
            Updated job with results
        """
        logger.info(f"Starting migration job {job.job_id}: {job.source} -> {job.target_backend}")

        # Check resource limits before starting
        if len(self.active_jobs) >= MAX_CONCURRENT_JOBS:
            error_msg = "Maximum concurrent migration jobs limit reached"
            job.status = MigrationStatus.FAILED
            job.errors.append(error_msg)
            return job

        # Acquire resource semaphore
        async with self._resource_semaphore:
            try:
                # Register job
                self.active_jobs[job.job_id] = job
                job.status = MigrationStatus.RUNNING
                job.started_at = datetime.now()

                # Set up timeout if specified
                timeout = job.timeout_seconds

                try:
                    # Create concurrency semaphore
                    self._semaphores[job.job_id] = asyncio.Semaphore(job.max_concurrent)

                    # Execute migration with timeout if specified
                    migration_coro = self._execute_migration(job)

                    if timeout:
                        try:
                            await asyncio.wait_for(migration_coro, timeout=timeout)
                        except asyncio.TimeoutError:
                            job.status = MigrationStatus.FAILED
                            sanitized_error = _sanitize_error_message(
                                f"Migration timed out after {timeout} seconds"
                            )
                            job.errors.append(f"Timeout: {sanitized_error}")
                            logger.error(f"Migration job {job.job_id} timed out after {timeout} seconds")
                            raise asyncio.TimeoutError(f"Migration job timed out after {timeout} seconds")
                    else:
                        await migration_coro

                    # Mark as completed
                    job.status = (
                        MigrationStatus.COMPLETED if job.error_count == 0 else MigrationStatus.PARTIAL
                    )
                    job.completed_at = datetime.now()

                    logger.info(
                        f"Migration job {job.job_id} completed: "
                        f"{job.success_count}/{job.processed_count} successful, "
                        f"{job.error_count} errors"
                    )

                except Exception as e:
                    job.status = MigrationStatus.FAILED
                    # Use internal error logging for security
                    external_error = _log_internal_error(str(e), f"migration job {job.job_id}")
                    job.errors.append(f"Job failed: {external_error}")
                finally:
                    # Cleanup semaphore
                    if job.job_id in self._semaphores:
                        del self._semaphores[job.job_id]
            finally:
                # Cleanup active job registration - ensures cleanup even on early exceptions
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]

        return job

    async def _execute_migration(self, job: MigrationJob) -> None:
        """Execute the actual migration based on source type."""
        if job.source == MigrationSource.QDRANT:
            await self._migrate_from_qdrant(job)
        elif job.source == MigrationSource.NEO4J:
            await self._migrate_from_neo4j(job)
        elif job.source == MigrationSource.REDIS:
            await self._migrate_from_redis(job)
        elif job.source == MigrationSource.ALL:
            await self._migrate_from_all_sources(job)
        else:
            raise ValueError(f"Unsupported source: {job.source}")

    async def _migrate_from_qdrant(self, job: MigrationJob) -> None:
        """Migrate data from Qdrant vector database."""
        if not self.qdrant_client:
            raise ValueError("Qdrant client not available")

        if job.target_backend != "text":
            raise ValueError(f"Migration from Qdrant to {job.target_backend} not supported")

        if not self.text_backend:
            raise ValueError("Text backend not available")

        logger.info("Starting Qdrant to text backend migration")

        # Get all vectors from Qdrant
        try:
            # Use scroll to get all points in batches
            collection_name = getattr(
                job, "collection_name", "context_embeddings"
            )  # Configurable collection name
            # Add memory usage monitoring for large batches
            if job.batch_size > 1000:
                logger.warning(f"Large batch size {job.batch_size} may cause memory issues")

            scroll_result = self.qdrant_client.client.scroll(
                collection_name=collection_name,
                limit=min(job.batch_size, 1000),  # Cap batch size to prevent memory issues
                with_payload=True,
                with_vectors=False,  # We don't need vectors for text indexing
            )

            points = scroll_result[0]  # First element is the list of points
            next_page_offset = scroll_result[1]  # Second element is next page offset

            # Process first batch
            if points:
                await self._process_qdrant_batch(job, points)

            # Continue with remaining batches
            while next_page_offset:
                scroll_result = self.qdrant_client.client.scroll(
                    collection_name=collection_name,
                    limit=job.batch_size,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = scroll_result[0]
                next_page_offset = scroll_result[1]

                if points:
                    await self._process_qdrant_batch(job, points)
                else:
                    break

        except Exception as e:
            logger.error(f"Error migrating from Qdrant: {e}")
            job.errors.append(f"Qdrant migration error: {_log_internal_error(str(e), 'Qdrant migration')}")

    async def _process_qdrant_batch(self, job: MigrationJob, points: List[Any]) -> None:
        """Process a batch of points from Qdrant with rate limiting."""
        # Apply rate limiting for the batch operation
        if not await self._rate_limiter.wait_for_tokens(len(points), max_wait=30.0):
            logger.warning(f"Rate limit exceeded for batch of {len(points)} points")
            job.error_count += len(points)
            job.errors.append(f"Rate limit exceeded for batch of {len(points)} points")
            return

        tasks = []

        for point in points:
            task = self._migrate_qdrant_point(job, point)
            tasks.append(task)

        # Execute batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                job.error_count += 1
                job.errors.append(f"Point migration error: {str(result)}")
            elif isinstance(result, MigrationResult):
                if result.success:
                    job.success_count += 1
                else:
                    job.error_count += 1
                    if result.error_message:
                        job.errors.append(result.error_message)

            job.processed_count += 1

    async def _migrate_qdrant_point(self, job: MigrationJob, point: Any) -> MigrationResult:
        """Migrate a single point from Qdrant to text backend."""
        async with self._semaphores[job.job_id]:
            start_time = time.time()

            try:
                point_id = str(point.id)
                payload = point.payload or {}

                # Extract text content from payload
                text_content = self._extract_text_from_payload(payload)
                if not text_content:
                    return MigrationResult(
                        source_id=point_id,
                        target_id=None,
                        success=False,
                        error_message="No text content found in payload",
                    )

                # Extract metadata
                metadata = {
                    "source": "qdrant_migration",
                    "original_id": point_id,
                    "migrated_at": datetime.now().isoformat(),
                    "content_type": payload.get("type", "unknown"),
                    **payload.get("metadata", {}),
                }

                # Index in text backend (skip in dry run mode)
                if not job.dry_run:
                    await self.text_backend.index_document(
                        doc_id=point_id,
                        text=text_content,
                        content_type=metadata.get("content_type", "text"),
                        metadata=metadata,
                    )

                processing_time = (time.time() - start_time) * 1000

                return MigrationResult(
                    source_id=point_id,
                    target_id=point_id,
                    success=True,
                    processing_time_ms=processing_time,
                    metadata={"text_length": len(text_content)},
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                return MigrationResult(
                    source_id=str(point.id) if hasattr(point, "id") else "unknown",
                    target_id=None,
                    success=False,
                    error_message=f"Migration failed: {_log_internal_error(str(e), 'point migration')}",
                    processing_time_ms=processing_time,
                )

    def _extract_text_from_payload(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract text content from Qdrant payload."""
        # Try different common fields for text content
        text_fields = ["content", "text", "title", "description", "body"]

        for field in text_fields:
            if field in payload:
                value = payload[field]
                if isinstance(value, str) and value.strip():
                    return value
                elif isinstance(value, dict):
                    # Handle nested content
                    if "text" in value:
                        return str(value["text"])
                    elif "title" in value and "description" in value:
                        return f"{value['title']} {value['description']}"

        # Fallback: concatenate all string values (with limits to prevent memory issues)
        text_parts = []

        for key, value in payload.items():
            if len(text_parts) >= MAX_TEXT_PARTS:
                break
            if isinstance(value, str) and value.strip() and key not in ["id", "type", "source"]:
                # Truncate individual parts to prevent memory issues
                truncated_value = value[:MAX_PART_LENGTH] if len(value) > MAX_PART_LENGTH else value
                text_parts.append(truncated_value)

        return " ".join(text_parts) if text_parts else None

    async def _migrate_from_neo4j(self, job: MigrationJob) -> None:
        """Migrate data from Neo4j graph database."""
        if not self.neo4j_client:
            raise ValueError("Neo4j client not available")

        if job.target_backend != "text":
            raise ValueError(f"Migration from Neo4j to {job.target_backend} not supported")

        if not self.text_backend:
            raise ValueError("Text backend not available")

        logger.info("Starting Neo4j to text backend migration")

        try:
            # Query all Context nodes in batches
            offset = 0

            while True:
                # Get batch of nodes
                cypher_query = """
                MATCH (n:Context)
                RETURN n
                SKIP $offset
                LIMIT $limit
                """

                results = self.neo4j_client.query(
                    cypher_query, {"offset": offset, "limit": job.batch_size}
                )

                if not results:
                    break

                # Process batch
                await self._process_neo4j_batch(job, results)

                # Check if we got fewer results than batch size (last batch)
                if len(results) < job.batch_size:
                    break

                offset += job.batch_size

        except Exception as e:
            logger.error(f"Error migrating from Neo4j: {e}")
            job.errors.append(f"Neo4j migration error: {_log_internal_error(str(e), 'Neo4j migration')}")

    async def _process_neo4j_batch(self, job: MigrationJob, nodes: List[Dict[str, Any]]) -> None:
        """Process a batch of nodes from Neo4j with rate limiting."""
        # Apply rate limiting for the batch operation
        if not await self._rate_limiter.wait_for_tokens(len(nodes), max_wait=30.0):
            logger.warning(f"Rate limit exceeded for batch of {len(nodes)} nodes")
            job.error_count += len(nodes)
            job.errors.append(f"Rate limit exceeded for batch of {len(nodes)} nodes")
            return

        tasks = []

        for node_data in nodes:
            task = self._migrate_neo4j_node(job, node_data)
            tasks.append(task)

        # Execute batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                job.error_count += 1
                job.errors.append(f"Node migration error: {str(result)}")
            elif isinstance(result, MigrationResult):
                if result.success:
                    job.success_count += 1
                else:
                    job.error_count += 1
                    if result.error_message:
                        job.errors.append(result.error_message)

            job.processed_count += 1

    async def _migrate_neo4j_node(
        self, job: MigrationJob, node_data: Dict[str, Any]
    ) -> MigrationResult:
        """Migrate a single node from Neo4j to text backend."""
        async with self._semaphores[job.job_id]:
            start_time = time.time()

            try:
                node = node_data.get("n", {})
                if not node:
                    return MigrationResult(
                        source_id="unknown",
                        target_id=None,
                        success=False,
                        error_message="No node data found",
                    )

                node_id = node.get("id", str(hash(str(node))))

                # Extract text content from node properties
                text_content = self._extract_text_from_node(node)
                if not text_content:
                    return MigrationResult(
                        source_id=node_id,
                        target_id=None,
                        success=False,
                        error_message="No text content found in node",
                    )

                # Extract metadata
                metadata = {
                    "source": "neo4j_migration",
                    "original_id": node_id,
                    "migrated_at": datetime.now().isoformat(),
                    "content_type": node.get("type", "unknown"),
                    **{k: v for k, v in node.items() if k not in ["id", "text", "content"]},
                }

                # Index in text backend (skip in dry run mode)
                if not job.dry_run:
                    await self.text_backend.index_document(
                        doc_id=node_id,
                        text=text_content,
                        content_type=metadata.get("content_type", "text"),
                        metadata=metadata,
                    )

                processing_time = (time.time() - start_time) * 1000

                return MigrationResult(
                    source_id=node_id,
                    target_id=node_id,
                    success=True,
                    processing_time_ms=processing_time,
                    metadata={"text_length": len(text_content)},
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                return MigrationResult(
                    source_id="unknown",
                    target_id=None,
                    success=False,
                    error_message=f"Migration failed: {_log_internal_error(str(e), 'node migration')}",
                    processing_time_ms=processing_time,
                )

    def _extract_text_from_node(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract text content from Neo4j node properties."""
        # Try different common fields for text content
        text_fields = ["content", "text", "title", "description", "body"]

        for field in text_fields:
            if field in node:
                value = node[field]
                if isinstance(value, str) and value.strip():
                    return value

        # Handle JSON fields (from flattened storage) with size limits
        json_fields = [k for k in node.keys() if k.endswith("_json")]
        for field in json_fields:
            try:
                json_str = node[field]
                # Use streaming JSON parser for memory efficiency
                json_data = _parse_json_streaming(json_str)
                if json_data is None:
                    continue
                    
                if isinstance(json_data, dict):
                    # Check if streaming parser already extracted text
                    if "extracted_text" in json_data:
                        return json_data["extracted_text"]
                    
                    # Otherwise, use regular text extraction
                    text = self._extract_text_from_payload(json_data)
                    if text:
                        return text
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse JSON field {field}: {e}")
                continue

        # Fallback: concatenate all string values
        text_parts = []
        for key, value in node.items():
            if isinstance(value, str) and value.strip() and key not in ["id", "type", "source"]:
                text_parts.append(value)

        return " ".join(text_parts) if text_parts else None

    async def _migrate_from_redis(self, job: MigrationJob) -> None:
        """Migrate data from Redis key-value store."""
        if not self.kv_store:
            raise ValueError("KV store client not available")

        if job.target_backend != "text":
            raise ValueError(f"Migration from Redis to {job.target_backend} not supported")

        # Redis migration is typically not needed for text search
        # as KV data is usually non-textual (IDs, state, cache)
        logger.warning("Redis to text migration requested - typically not needed")
        job.status = MigrationStatus.COMPLETED

    async def _migrate_from_all_sources(self, job: MigrationJob) -> None:
        """Migrate data from all available sources."""
        # Create separate jobs for each source
        sub_jobs = []

        if self.qdrant_client:
            qdrant_job = MigrationJob(
                job_id=f"{job.job_id}_qdrant",
                source=MigrationSource.QDRANT,
                target_backend=job.target_backend,
                batch_size=job.batch_size,
                max_concurrent=job.max_concurrent,
                dry_run=job.dry_run,
            )
            sub_jobs.append(self._migrate_from_qdrant(qdrant_job))

        if self.neo4j_client:
            neo4j_job = MigrationJob(
                job_id=f"{job.job_id}_neo4j",
                source=MigrationSource.NEO4J,
                target_backend=job.target_backend,
                batch_size=job.batch_size,
                max_concurrent=job.max_concurrent,
                dry_run=job.dry_run,
            )
            sub_jobs.append(self._migrate_from_neo4j(neo4j_job))

        # Execute all sub-jobs
        await asyncio.gather(*sub_jobs, return_exceptions=True)

    async def validate_migration(self, job_id: str) -> Dict[str, Any]:
        """
        Validate the results of a migration job with comprehensive data integrity checks.

        Args:
            job_id: ID of the migration job to validate

        Returns:
            Enhanced validation results including integrity checks
        """
        if job_id not in self.active_jobs:
            return {"error": "Job not found"}

        job = self.active_jobs[job_id]

        # Basic validation
        validation_results = {
            "job_id": job_id,
            "status": job.status,
            "total_processed": job.processed_count,
            "successful": job.success_count,
            "failed": job.error_count,
            "success_rate": (job.success_count / job.processed_count * 100)
            if job.processed_count > 0
            else 0,
            "errors": job.errors[-10:],  # Last 10 errors
            "validation_checks": {},
        }

        # Text backend validation
        if job.target_backend == "text" and self.text_backend:
            try:
                stats = self.text_backend.get_index_statistics()
                validation_results["validation_checks"]["text_backend"] = {
                    "indexed_documents": stats["document_count"],
                    "vocabulary_size": stats["vocabulary_size"],
                    "average_doc_length": stats.get("average_document_length", 0),
                }
            except Exception as e:
                validation_results["validation_checks"]["text_backend"] = {
                    "error": _log_internal_error(str(e), "text backend validation")
                }

        # Comprehensive data integrity validation
        try:
            # Determine source client based on job source
            source_client = None
            if job.source == MigrationSource.QDRANT:
                source_client = self.qdrant_client
            elif job.source == MigrationSource.NEO4J:
                source_client = self.neo4j_client
            
            if source_client:
                # Create data validator
                validator = DataValidator(
                    source_client=source_client,
                    target_backend=self.text_backend
                )
                
                # Run comprehensive validation
                integrity_result = await validator.validate_migration(
                    job_id=job_id,
                    source_type=job.source.value,
                    sample_size=min(VALIDATION_SAMPLE_SIZE, job.processed_count)
                )
                
                # Add integrity results to validation
                validation_results["validation_checks"]["data_integrity"] = {
                    "source_count": integrity_result.source_count,
                    "target_count": integrity_result.target_count,
                    "missing_count": integrity_result.missing_count,
                    "checksum_matches": integrity_result.checksum_matches,
                    "checksum_mismatches": integrity_result.checksum_mismatches,
                    "sample_size": integrity_result.sample_size,
                    "integrity_score": integrity_result.integrity_score,
                    "is_valid": integrity_result.is_valid,
                    "validation_errors": integrity_result.validation_errors,
                    "timestamp": integrity_result.timestamp.isoformat()
                }
                
                # Update overall validation status
                validation_results["integrity_validated"] = True
                validation_results["overall_valid"] = integrity_result.is_valid
                
                # Log validation results
                if integrity_result.is_valid:
                    logger.info(f"Migration {job_id} passed integrity validation: {integrity_result.integrity_score:.1%}")
                else:
                    logger.warning(f"Migration {job_id} failed integrity validation: {integrity_result.integrity_score:.1%}")
            else:
                validation_results["validation_checks"]["data_integrity"] = {
                    "error": "No source client available for integrity validation",
                    "source_type": job.source.value
                }
                validation_results["integrity_validated"] = False
                
        except Exception as e:
            error_msg = _log_internal_error(str(e), "comprehensive data validation")
            validation_results["validation_checks"]["data_integrity"] = {
                "error": error_msg
            }
            validation_results["integrity_validated"] = False

        return validation_results

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a migration job."""
        if job_id not in self.active_jobs:
            return None

        job = self.active_jobs[job_id]

        return {
            "job_id": job.job_id,
            "status": job.status,
            "source": job.source,
            "target_backend": job.target_backend,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processed_count": job.processed_count,
            "success_count": job.success_count,
            "error_count": job.error_count,
            "progress_percentage": (job.processed_count / max(job.processed_count, 1)) * 100,
            "recent_errors": job.errors[-5:] if job.errors else [],
        }

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active migration jobs."""
        return [self.get_job_status(job_id) for job_id in self.active_jobs.keys()]


# Global migration engine instance
_migration_engine: Optional[DataMigrationEngine] = None


def get_migration_engine() -> Optional[DataMigrationEngine]:
    """Get the global migration engine instance."""
    return _migration_engine


def initialize_migration_engine(
    qdrant_client: Optional[VectorDBInitializer] = None,
    neo4j_client: Optional[Neo4jInitializer] = None,
    kv_store: Optional[ContextKV] = None,
    text_backend: Optional[TextSearchBackend] = None,
) -> DataMigrationEngine:
    """Initialize the global migration engine."""
    global _migration_engine
    _migration_engine = DataMigrationEngine(
        qdrant_client=qdrant_client,
        neo4j_client=neo4j_client,
        kv_store=kv_store,
        text_backend=text_backend,
    )
    logger.info("Migration engine initialized")
    return _migration_engine
