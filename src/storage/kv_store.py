#!/usr/bin/env python3
"""
kv_store.py: Key-Value store for context storage system

This component provides:
1. Redis connector for fast caching and session management
2. DuckDB integration for analytics and time-series data
3. Unified API for KV operations
4. Performance metrics collection
"""

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import duckdb
import redis
import yaml

# Handle imports flexibly
try:
    from ..core.base_component import DatabaseComponent
    from ..validators.kv_validators import (
        sanitize_metric_name,
        validate_metric_event,
        validate_redis_key,
        validate_time_range,
    )
except ImportError:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.base_component import DatabaseComponent
    from validators.kv_validators import (
        sanitize_metric_name,
        validate_metric_event,
        validate_redis_key,
        validate_time_range,
    )


@dataclass
class MetricEvent:
    """Represents a metric event"""

    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    document_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class CacheEntry:
    """Represents a cache entry"""

    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    last_accessed: Optional[datetime] = None


class RedisConnector(DatabaseComponent):
    """Redis connector for caching and session management"""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize Redis connector with optional config injection for testing.

        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
            config: Optional configuration dictionary (overrides file loading)
            test_mode: If True, use test defaults when config is missing
        """
        self.test_mode = test_mode
        if config is not None:
            # When config is provided, bypass parent class init
            self.config = config
            self.config_path = config_path
            self.verbose = verbose
            # Create a minimal logger for test mode
            import logging

            self.logger = logging.getLogger(self.__class__.__name__)
            self.environment = "test" if test_mode else "production"
        else:
            super().__init__(config_path, verbose)

        self.redis_client: Optional[redis.Redis[bytes]] = None
        self.pipeline: Optional[redis.client.Pipeline[bytes]] = None
        self.perf_config = self._load_performance_config()

    def _get_service_name(self) -> str:
        """Get service name for configuration"""
        return "redis"

    def _load_performance_config(self) -> Dict[str, Any]:
        """Load performance configuration"""
        try:
            with open("performance.yaml", "r") as f:
                perf = yaml.safe_load(f)
                kv_config = perf.get("kv_store", {})
                result = kv_config.get("redis", {}) if kv_config else {}
                return result if isinstance(result, dict) else {}
        except FileNotFoundError:
            return {}

    def connect(self, **kwargs: Any) -> bool:
        """Connect to Redis"""
        # Password comes from environment variables or config, not hardcoded
        password = kwargs.get("password", None)
        redis_config = self.config.get("redis", {})

        # Check REDIS_URL environment variable first for Docker deployments
        redis_url = os.getenv("REDIS_URL")
        password_from_url = None
        if redis_url:
            # Parse redis://:password@host:port/db format
            url_match = re.match(r"^redis://(?::([^@]+)@)?([^:/]+):?(\d+)?/?(\d+)?", redis_url)
            if url_match:
                password_from_url = url_match.group(1)  # Password (if present)
                host = url_match.group(2)
                port = int(url_match.group(3)) if url_match.group(3) else 6379
                db = int(url_match.group(4)) if url_match.group(4) else 0
            else:
                # Fallback to config
                self.log_warning(f"Failed to parse REDIS_URL: {redis_url}, falling back to config")
                host = redis_config.get("host", "localhost")
                port = redis_config.get("port", 6379)
                db = redis_config.get("database", 0)
        else:
            # Use config values
            host = redis_config.get("host", "localhost")
            port = redis_config.get("port", 6379)
            db = redis_config.get("database", 0)

        ssl = redis_config.get("ssl", False)

        pool_config = self.perf_config.get("connection_pool", {})
        max_connections = pool_config.get("max_size", 50)

        # Password priority: kwargs > URL > env var > config
        final_password = password or password_from_url or os.getenv("REDIS_PASSWORD") or redis_config.get("password")

        try:
            # Create connection pool
            pool_kwargs = {
                "host": host,
                "port": port,
                "db": db,
                "password": final_password,  # Password with proper priority chain
                "max_connections": max_connections,
                "decode_responses": True,
            }

            # Add SSL configuration if enabled
            if ssl:
                pool_kwargs["ssl"] = True
                pool_kwargs["ssl_cert_reqs"] = "required"
                # For local development with self-signed certs
                if self.environment != "production":
                    pool_kwargs["ssl_cert_reqs"] = "none"

            pool = redis.ConnectionPool(**pool_kwargs)

            self.redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            if self.redis_client:
                self.redis_client.ping()
                self.is_connected = True
                self.log_success(f"Connected to Redis at {host}:{port}")
                self.logger.debug(f"Redis connection details: host={host}, port={port}, db={db}, password={'***' if final_password else 'None'}")
                return True
            return False

        except Exception as e:
            # Sanitize error message to avoid exposing password
            sensitive_values = [password] if password else []
            self.log_error("Failed to connect to Redis", e, sensitive_values)
            return False

    def get_prefixed_key(self, key: str, prefix_type: str = "cache") -> str:
        """Get key with configured prefix"""
        prefixes = self.config.get("redis", {}).get("prefixes", {})
        prefix = prefixes.get(prefix_type, f"{prefix_type}:")
        return f"{prefix}{key}"

    def set_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set cache value with optional TTL"""
        if not self.ensure_connected():
            return False

        # Validate key
        if not validate_redis_key(key):
            self.log_error(f"Invalid cache key: {key}")
            return False

        try:
            prefixed_key = self.get_prefixed_key(key, "cache")

            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.perf_config.get("cache", {}).get("ttl_seconds", 3600)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl_seconds,
            )

            # Store as JSON
            if self.redis_client:
                self.redis_client.setex(
                    prefixed_key, ttl_seconds, json.dumps(asdict(entry), default=str)
                )

            return True

        except Exception as e:
            self.log_error(f"Failed to set cache for key {key}", e)
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.ensure_connected():
            return None

        try:
            prefixed_key = self.get_prefixed_key(key, "cache")
            if not self.redis_client:
                return None
            data = self.redis_client.get(prefixed_key)

            if data:
                entry_dict = json.loads(data)
                # Validate entry structure and update hit count
                if "hit_count" in entry_dict:
                    entry_dict["hit_count"] = entry_dict.get("hit_count", 0) + 1
                else:
                    entry_dict["hit_count"] = 1

                entry_dict["last_accessed"] = datetime.utcnow().isoformat()

                # Update in Redis
                if self.redis_client:
                    ttl = self.redis_client.ttl(prefixed_key)
                    if ttl > 0:
                        self.redis_client.setex(
                            prefixed_key, ttl, json.dumps(entry_dict, default=str)
                        )

                return entry_dict.get("value")

            return None

        except Exception as e:
            self.log_error(f"Failed to get cache for key {key}", e)
            return None

    def delete_cache(self, pattern: str) -> int:
        """Delete cache entries matching pattern"""
        if not self.ensure_connected():
            return 0

        try:
            prefix = self.get_prefixed_key("", "cache")
            full_pattern = f"{prefix}{pattern}"

            # Find matching keys
            if not self.redis_client:
                return 0
            keys = list(self.redis_client.scan_iter(match=full_pattern))

            if keys:
                return self.redis_client.delete(*keys)

            return 0

        except Exception as e:
            self.log_error(f"Failed to delete cache pattern {pattern}", e)
            return 0

    def set_session(self, session_id: str, data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """Store session data"""
        if not self.ensure_connected():
            return False

        try:
            prefixed_key = self.get_prefixed_key(session_id, "session")

            session_data = {
                "id": session_id,
                "data": data,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
            }

            if self.redis_client:
                self.redis_client.setex(prefixed_key, ttl_seconds, json.dumps(session_data))

            return True

        except Exception as e:
            self.log_error(f"Failed to set session {session_id}", e)
            return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.ensure_connected():
            return None

        try:
            prefixed_key = self.get_prefixed_key(session_id, "session")
            if not self.redis_client:
                return None
            data = self.redis_client.get(prefixed_key)

            if data:
                session_data = json.loads(data)
                # Update last activity
                session_data["last_activity"] = datetime.utcnow().isoformat()

                # Extend TTL
                if self.redis_client:
                    ttl = self.redis_client.ttl(prefixed_key)
                    if ttl > 0:
                        self.redis_client.setex(prefixed_key, ttl, json.dumps(session_data))

                data = session_data.get("data")
                return data if isinstance(data, dict) else None

            return None

        except Exception as e:
            self.log_error(f"Failed to get session {session_id}", e)
            return None

    def acquire_lock(self, resource: str, timeout: int = 30) -> Optional[str]:
        """Acquire distributed lock"""
        if not self.ensure_connected():
            return None

        try:
            lock_key = self.get_prefixed_key(resource, "lock")
            lock_id = hashlib.sha256(f"{resource}{time.time()}".encode()).hexdigest()[:16]

            # Try to acquire lock
            if not self.redis_client:
                return None
            acquired = self.redis_client.set(
                lock_key, lock_id, nx=True, ex=timeout  # Only set if not exists
            )

            return lock_id if acquired else None

        except Exception as e:
            self.log_error(f"Failed to acquire lock for {resource}", e)
            return None

    def release_lock(self, resource: str, lock_id: str) -> bool:
        """Release distributed lock"""
        if not self.ensure_connected():
            return False

        try:
            lock_key = self.get_prefixed_key(resource, "lock")

            # Lua script to ensure we only delete our own lock
            lua_script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """

            if not self.redis_client:
                return False
            result = self.redis_client.eval(lua_script, 1, lock_key, lock_id)
            return bool(result)

        except Exception as e:
            self.log_error(f"Failed to release lock for {resource}", e)
            return False

    def record_metric(self, metric: MetricEvent) -> bool:
        """Record metric event"""
        if not self.ensure_connected():
            return False

        # Validate metric
        metric_dict = asdict(metric)
        if not validate_metric_event(metric_dict):
            self.log_error(f"Invalid metric event: {metric.metric_name}")
            return False

        # Sanitize metric name
        metric.metric_name = sanitize_metric_name(metric.metric_name)

        try:
            # Create metric key
            metric_key = self.get_prefixed_key(
                f"{metric.metric_name}:{metric.timestamp.strftime('%Y%m%d%H%M')}",
                "metric",
            )

            # Store metric
            if self.redis_client:
                self.redis_client.zadd(
                    metric_key,
                    {json.dumps(asdict(metric), default=str): metric.timestamp.timestamp()},
                )

                # Set expiration (7 days)
                self.redis_client.expire(metric_key, 7 * 24 * 3600)

            return True

        except Exception as e:
            self.log_error(f"Failed to record metric {metric.metric_name}", e)
            return False

    def get_metrics(
        self, metric_name: str, start_time: datetime, end_time: datetime
    ) -> List[MetricEvent]:
        """Get metrics within time range"""
        if not self.ensure_connected():
            return []

        # Validate time range
        if not validate_time_range(start_time, end_time):
            self.log_error(f"Invalid time range: {start_time} to {end_time}")
            return []

        # Sanitize metric name
        metric_name = sanitize_metric_name(metric_name)

        metrics = []

        try:
            # Optimize by using hourly buckets instead of minute buckets
            # This reduces the number of keys to scan
            current_hour = start_time.replace(minute=0, second=0, microsecond=0)
            end_hour = end_time.replace(minute=59, second=59, microsecond=999999)

            while current_hour <= end_hour:
                # Check both minute and hour granularity keys
                for time_format in ["%Y%m%d%H", "%Y%m%d%H%M"]:
                    metric_key = self.get_prefixed_key(
                        f"{metric_name}:{current_hour.strftime(time_format)}", "metric"
                    )

                    # Get metrics from sorted set
                    if not self.redis_client:
                        continue
                    data = self.redis_client.zrangebyscore(
                        metric_key, start_time.timestamp(), end_time.timestamp()
                    )

                    for item in data:
                        metric_dict = json.loads(item)
                        metric_dict["timestamp"] = datetime.fromisoformat(metric_dict["timestamp"])
                        metrics.append(MetricEvent(**metric_dict))

                current_hour += timedelta(hours=1)

            return metrics

        except Exception as e:
            self.log_error(f"Failed to get metrics for {metric_name}", e)
            return []

    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                self.log_error("Error closing Redis connection", e)
            finally:
                self.redis_client = None
                self.is_connected = False

    def get(self, key: str) -> Optional[str]:
        """
        Compatibility method for simple Redis get operations.

        This method provides compatibility with simple Redis client usage
        by delegating to the appropriate cache method.
        """
        # For simple string keys, use cache operations
        try:
            value = self.get_cache(key)
            if value is not None:
                # If it's already a string, return it
                if isinstance(value, str):
                    return value
                # Otherwise, serialize it as JSON
                import json

                return json.dumps(value) if value is not None else None
            return None
        except Exception as e:
            self.log_error(f"Error in compatibility get method for key {key}", e)
            return None


class DuckDBAnalytics(DatabaseComponent):
    """DuckDB analytics layer for time-series and aggregations"""

    def __init__(self, config_path: str = ".ctxrc.yaml", verbose: bool = False):
        super().__init__(config_path, verbose)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.perf_config = self._load_performance_config()

    def _get_service_name(self) -> str:
        """Get service name for configuration"""
        return "duckdb"

    def _load_performance_config(self) -> Dict[str, Any]:
        """Load performance configuration"""
        try:
            with open("performance.yaml", "r") as f:
                perf = yaml.safe_load(f)
                kv_config = perf.get("kv_store", {})
                result = kv_config.get("duckdb", {}) if kv_config else {}
                return result if isinstance(result, dict) else {}
        except FileNotFoundError:
            return {}

    def connect(self, **kwargs) -> bool:
        """Connect to DuckDB"""
        duckdb_config = self.config.get("duckdb", {})
        db_path = duckdb_config.get("database_path", "context/.duckdb/analytics.db")
        memory_limit = duckdb_config.get("memory_limit", "2GB")
        threads = duckdb_config.get("threads", 4)

        try:
            # Create directory if needed
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Connect to DuckDB
            self.conn = duckdb.connect(db_path)

            # Set configuration
            if self.conn:
                self.conn.execute(f"SET memory_limit = '{memory_limit}'")
                self.conn.execute(f"SET threads = {threads}")

            # Initialize tables
            self._initialize_tables()

            self.is_connected = True
            self.log_success(f"Connected to DuckDB at {db_path}")
            return True

        except OSError as e:
            self.log_error(f"Failed to create DuckDB directory: {e}", e)
            return False
        except Exception as e:
            self.log_error(f"DuckDB connection error: {e}", e)
            return False

    def _initialize_tables(self):
        """Initialize analytics tables"""
        tables = self.config.get("duckdb", {}).get("tables", {})

        # Metrics table
        if not self.conn:
            return False
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tables.get('metrics', 'context_metrics')} (
                timestamp TIMESTAMP NOT NULL,
                metric_name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                document_id VARCHAR,
                agent_id VARCHAR,
                tags JSON,
                PRIMARY KEY (timestamp, metric_name)
            )
        """
        )

        # Events table
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tables.get('events', 'context_events')} (
                event_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                event_type VARCHAR NOT NULL,
                document_id VARCHAR,
                agent_id VARCHAR,
                event_data JSON
            )
        """
        )

        # Create indexes for events table
        events_table = tables.get("events", "context_events")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {events_table} (timestamp)")
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_event_type ON {events_table} (event_type)"
        )

        # Summaries table
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tables.get('summaries', 'context_summaries')} (
                summary_date DATE NOT NULL,
                summary_type VARCHAR NOT NULL,
                metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (summary_date, summary_type)
            )
        """
        )

        # Trends table
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tables.get('trends', 'context_trends')} (
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                trend_type VARCHAR NOT NULL,
                trend_data JSON,
                confidence DOUBLE,
                PRIMARY KEY (period_start, period_end, trend_type)
            )
        """
        )

    def insert_metrics(self, metrics: List[MetricEvent]) -> bool:
        """Batch insert metrics"""
        if not self.ensure_connected():
            return False

        try:
            tables = self.config.get("duckdb", {}).get("tables", {})
            metrics_table = tables.get("metrics", "context_metrics")

            # Prepare data
            values = []
            for metric in metrics:
                values.append(
                    (
                        metric.timestamp,
                        metric.metric_name,
                        metric.value,
                        metric.document_id,
                        metric.agent_id,
                        json.dumps(metric.tags),
                    )
                )

            # Batch insert
            if self.conn:
                self.conn.executemany(
                    f"""
                    INSERT INTO {metrics_table}
                    (timestamp, metric_name, value, document_id, agent_id, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    values,
                )

            return True

        except Exception as e:
            self.log_error("Failed to insert metrics", e)
            return False

    def query_metrics(
        self, query: str, params: Optional[Union[Dict[str, Any], List[Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Execute analytics query"""
        if not self.ensure_connected():
            return []

        try:
            # Set query timeout
            timeout = self.perf_config.get("query", {}).get("timeout_seconds", 60)
            if self.conn:
                self.conn.execute(f"SET statement_timeout = '{timeout}s'")

                # Execute query
                if params:
                    result = self.conn.execute(query, params).fetchall()
                else:
                    result = self.conn.execute(query).fetchall()
            else:
                return []

            # Get column names
            if self.conn and self.conn.description:
                columns = [desc[0] for desc in self.conn.description]
            else:
                return []

            # Convert to dictionaries
            return [dict(zip(columns, row)) for row in result]

        except Exception as e:
            self.log_error("Failed to execute query", e)
            return []

    def aggregate_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "avg",
    ) -> Dict[str, Any]:
        """Aggregate metrics over time period"""
        if not self.ensure_connected():
            return {}

        try:
            tables = self.config.get("duckdb", {}).get("tables", {})
            metrics_table = tables.get("metrics", "context_metrics")

            # Build aggregation query
            agg_funcs = {
                "avg": "AVG(value)",
                "sum": "SUM(value)",
                "min": "MIN(value)",
                "max": "MAX(value)",
                "count": "COUNT(*)",
                "stddev": "STDDEV(value)",
            }

            agg_expr = agg_funcs.get(aggregation, "AVG(value)")

            query = f"""
                SELECT
                    {agg_expr} as result,
                    COUNT(*) as count,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time
                FROM {metrics_table}
                WHERE metric_name = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
            """

            if not self.conn:
                return {}
            result = self.conn.execute(query, [metric_name, start_time, end_time]).fetchone()

            if result:
                return {
                    "aggregation": aggregation,
                    "value": result[0],
                    "count": result[1],
                    "start_time": result[2],
                    "end_time": result[3],
                }

            return {}

        except Exception as e:
            self.log_error(f"Failed to aggregate metrics for {metric_name}", e)
            return {}

    def generate_summary(self, summary_date: date, summary_type: str = "daily") -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.ensure_connected():
            return {}

        try:
            tables = self.config.get("duckdb", {}).get("tables", {})
            metrics_table = tables.get("metrics", "context_metrics")
            summaries_table = tables.get("summaries", "context_summaries")

            # Calculate date range
            if summary_type == "daily":
                start_time = datetime.combine(summary_date, datetime.min.time())
                end_time = datetime.combine(summary_date, datetime.max.time())
            elif summary_type == "weekly":
                start_time = datetime.combine(summary_date - timedelta(days=6), datetime.min.time())
                end_time = datetime.combine(summary_date, datetime.max.time())
            else:  # monthly
                start_time = datetime.combine(summary_date.replace(day=1), datetime.min.time())
                end_time = datetime.combine(summary_date, datetime.max.time())

            # Generate summary
            query = f"""
                SELECT
                    metric_name,
                    COUNT(*) as count,
                    AVG(value) as avg_value,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    STDDEV(value) as stddev_value
                FROM {metrics_table}
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY metric_name
            """

            if not self.conn:
                return {}
            results = self.conn.execute(query, [start_time, end_time]).fetchall()

            summary: Dict[str, Any] = {
                "summary_date": summary_date.isoformat(),
                "summary_type": summary_type,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "metrics": {},
            }

            for row in results:
                summary["metrics"][row[0]] = {
                    "count": row[1],
                    "avg": row[2],
                    "min": row[3],
                    "max": row[4],
                    "stddev": row[5],
                }

            # Store summary
            if self.conn:
                self.conn.execute(
                    f"""
                INSERT OR REPLACE INTO {summaries_table}
                (summary_date, summary_type, metrics)
                VALUES (?, ?, ?)
            """,
                    [summary_date, summary_type, json.dumps(summary)],
                )

            return summary

        except Exception as e:
            self.log_error(f"Failed to generate {summary_type} summary", e)
            return {}

    def detect_trends(self, metric_name: str, period_days: int = 7) -> Dict[str, Any]:
        """Detect trends in metrics"""
        if not self.ensure_connected():
            return {}

        try:
            tables = self.config.get("duckdb", {}).get("tables", {})
            metrics_table = tables.get("metrics", "context_metrics")

            # Calculate time periods
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=period_days)

            # Get time series data
            query = f"""
                SELECT
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(value) as avg_value,
                    COUNT(*) as count
                FROM {metrics_table}
                WHERE metric_name = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
                GROUP BY hour
                ORDER BY hour
            """

            if not self.conn:
                return {}
            results = self.conn.execute(query, [metric_name, start_time, end_time]).fetchall()

            if len(results) < 2:
                return {"trend": "insufficient_data"}

            # Simple trend detection
            values = [r[1] for r in results]
            first_half_avg = sum(values[: len(values) // 2]) / (len(values) // 2)
            second_half_avg = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)

            trend_direction = "increasing" if second_half_avg > first_half_avg else "decreasing"
            trend_strength = abs(second_half_avg - first_half_avg) / first_half_avg

            return {
                "metric_name": metric_name,
                "period_days": period_days,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "confidence": min(len(results) / (period_days * 24), 1.0),  # Based on data density
                "data_points": len(results),
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
            }

        except Exception as e:
            self.log_error(f"Failed to detect trends for {metric_name}", e)
            return {}

    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                self.log_error("Error closing DuckDB connection", e)
            finally:
                self.conn = None
                self.is_connected = False


class ContextKV:
    """Unified KV store interface"""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize ContextKV with optional config injection for testing.

        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
            config: Optional configuration dictionary (overrides file loading)
            test_mode: If True, use test defaults when config is missing
        """
        self.config = config
        self.test_mode = test_mode
        self.redis = RedisConnector(config_path, verbose, config, test_mode)
        # Only create DuckDB if not in test mode or explicitly needed
        if not test_mode:
            try:
                self.duckdb = DuckDBAnalytics(config_path, verbose)
            except Exception:
                self.duckdb = None
        else:
            self.duckdb = None
        self.verbose = verbose

    def connect(self, redis_password: Optional[str] = None) -> bool:
        """Connect to both stores"""
        redis_connected = self.redis.connect(password=redis_password)
        duckdb_connected = self.duckdb.connect()

        return redis_connected and duckdb_connected

    def record_event(
        self,
        event_type: str,
        document_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record an event to both stores"""
        timestamp = datetime.utcnow()

        # Create metric event for Redis
        metric = MetricEvent(
            timestamp=timestamp,
            metric_name=f"event.{event_type}",
            value=1.0,
            tags=data or {},
            document_id=document_id,
            agent_id=agent_id,
        )

        # Record in Redis for real-time
        redis_success = self.redis.record_metric(metric)

        # Record in DuckDB for analytics
        duckdb_success = self.duckdb.insert_metrics([metric])

        return redis_success and duckdb_success

    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent activity summary"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get metrics from DuckDB
        query = """
            SELECT
                metric_name,
                COUNT(*) as count,
                AVG(value) as avg_value
            FROM context_metrics
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY metric_name
            ORDER BY count DESC
        """

        metrics = self.duckdb.query_metrics(query, [start_time, end_time])

        return {
            "period_hours": hours,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": metrics,
        }

    def close(self):
        """Close all connections"""
        self.redis.close()
        self.duckdb.close()


@click.group()
def cli():
    """Context Key-Value Store Management"""
    pass


@cli.command()
@click.option("--redis-pass", help="Redis password")
@click.option("--verbose", is_flag=True, help="Verbose output")
def test_connection(redis_pass: str, verbose: bool) -> None:
    """Test KV store connections"""
    kv = ContextKV(verbose=verbose)

    click.echo("Testing KV store connections...")

    if kv.connect(redis_password=redis_pass):
        click.echo("✓ All connections successful!")

        # Test basic operations
        test_key = "test:connection"
        test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}

        # Test Redis cache
        if kv.redis.set_cache(test_key, test_value, ttl_seconds=60):
            retrieved = kv.redis.get_cache(test_key)
            if retrieved == test_value:
                click.echo("✓ Redis cache operations working")
            else:
                click.echo("✗ Redis cache retrieval failed")

        # Test DuckDB query
        try:
            result = kv.duckdb.query_metrics("SELECT 1 as test")
            if result and result[0]["test"] == 1:
                click.echo("✓ DuckDB queries working")
            else:
                click.echo("✗ DuckDB query failed")
        except Exception as e:
            click.echo(f"✗ DuckDB error: {e}")

        # Test event recording
        if kv.record_event("test", data={"source": "cli"}):
            click.echo("✓ Event recording working")
        else:
            click.echo("✗ Event recording failed")

        kv.close()
    else:
        click.echo("✗ Connection failed")


@cli.command()
@click.option("--metric", required=True, help="Metric name")
@click.option("--value", type=float, required=True, help="Metric value")
@click.option("--document-id", help="Associated document ID")
@click.option("--agent-id", help="Associated agent ID")
@click.option("--redis-pass", help="Redis password")
def record_metric(
    metric: str, value: float, document_id: str, agent_id: str, redis_pass: str
) -> None:
    """Record a metric"""
    kv = ContextKV()

    if not kv.connect(redis_password=redis_pass):
        click.echo("Failed to connect to KV stores", err=True)
        return

    try:
        metric_event = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name=metric,
            value=value,
            tags={},
            document_id=document_id,
            agent_id=agent_id,
        )

        if kv.redis.record_metric(metric_event) and kv.duckdb.insert_metrics([metric_event]):
            click.echo(f"✓ Recorded metric: {metric} = {value}")
        else:
            click.echo("✗ Failed to record metric", err=True)

    finally:
        kv.close()


@cli.command()
@click.option("--hours", default=24, help="Hours to look back")
@click.option("--redis-pass", help="Redis password")
def activity_summary(hours: int, redis_pass: str) -> None:
    """Show recent activity summary"""
    kv = ContextKV()

    if not kv.connect(redis_password=redis_pass):
        click.echo("Failed to connect to KV stores", err=True)
        return

    try:
        summary = kv.get_recent_activity(hours)

        click.echo(f"\n=== Activity Summary (last {hours} hours) ===")
        click.echo(f"Period: {summary['start_time']} to {summary['end_time']}")

        if summary["metrics"]:
            click.echo("\nMetrics:")
            for metric in summary["metrics"]:
                name = metric["metric_name"]
                count = metric["count"]
                avg = metric["avg_value"]
                click.echo(f"  {name}: {count} events, avg value: {avg:.2f}")
        else:
            click.echo("\nNo activity recorded in this period")

    finally:
        kv.close()


if __name__ == "__main__":
    cli()
