#!/usr/bin/env python3
"""
duckdb_analytics.py: DuckDB analytics module for time-series analysis

This module provides analytics capabilities using DuckDB for
efficient time-series queries and aggregations.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

try:
    from ..core.base_component import DatabaseComponent
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.base_component import DatabaseComponent


@dataclass
class AnalyticsResult:
    """Represents an analytics query result."""

    query_type: str
    start_time: datetime
    end_time: datetime
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class TimeSeriesData:
    """Represents time-series data point."""

    timestamp: datetime
    metric: str
    value: float
    dimensions: Dict[str, str]


class DuckDBAnalytics(DatabaseComponent):
    """DuckDB analytics for time-series and aggregations."""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize DuckDB analytics engine.

        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
            config: Optional configuration dictionary
            test_mode: If True, use in-memory database for testing
        """
        self.test_mode = test_mode
        self.config_path = config_path
        self.verbose = verbose

        if config is not None:
            self.config = config
        else:
            # Initialize with base class if no config provided
            super().__init__(config_path, verbose)

        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.is_connected = False

    def _get_service_name(self) -> str:
        """Get service name for configuration."""
        return "duckdb"

    def connect(self, **kwargs) -> bool:
        """Connect to DuckDB database.

        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            if self.test_mode:
                # Use in-memory database for testing
                self.conn = duckdb.connect(":memory:")
            else:
                # Use configured database path
                duckdb_config = self.config.get("duckdb", {})
                db_path = duckdb_config.get("database_path", "context/.duckdb/analytics.db")

                # Create directory if needed
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self.conn = duckdb.connect(db_path)

            # Initialize tables
            self._initialize_tables()
            self.is_connected = True
            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to connect to DuckDB: {e}")
            return False

    def _initialize_tables(self):
        """Initialize analytics tables."""
        if not self.conn:
            return

        # Create metrics table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP NOT NULL,
                metric_name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                dimensions JSON,
                PRIMARY KEY (timestamp, metric_name)
            )
        """
        )

        # Create events table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                event_type VARCHAR NOT NULL,
                event_data JSON
            )
        """
        )

        # Create aggregates table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aggregates (
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                metric_name VARCHAR NOT NULL,
                aggregation_type VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                sample_count INTEGER,
                PRIMARY KEY (period_start, period_end, metric_name, aggregation_type)
            )
        """
        )

    def insert_time_series(self, data: List[TimeSeriesData]) -> bool:
        """Insert time-series data points.

        Args:
            data: List of time-series data points

        Returns:
            bool: True if inserted successfully, False otherwise
        """
        if not self.ensure_connected():
            return False

        try:
            # Prepare data for insertion
            values = [(d.timestamp, d.metric, d.value, json.dumps(d.dimensions)) for d in data]

            # Batch insert
            if self.conn:
                self.conn.executemany(
                    """
                    INSERT INTO metrics (timestamp, metric_name, value, dimensions)
                    VALUES (?, ?, ?, ?)
                    """,
                    values,
                )
            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to insert time-series data: {e}")
            return False

    def query_time_series(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
        group_by: Optional[List[str]] = None,
    ) -> AnalyticsResult:
        """Query time-series data with optional aggregation.

        Args:
            metric_name: Name of the metric to query
            start_time: Start of the time range
            end_time: End of the time range
            aggregation: Optional aggregation function (avg, sum, min, max, count)
            group_by: Optional list of dimensions to group by

        Returns:
            AnalyticsResult with query results
        """
        if not self.ensure_connected():
            return AnalyticsResult(
                query_type="time_series",
                start_time=start_time,
                end_time=end_time,
                data=[],
                metadata={"error": "Not connected"},
            )

        try:
            if aggregation:
                # Build aggregation query
                agg_func = {
                    "avg": "AVG(value)",
                    "sum": "SUM(value)",
                    "min": "MIN(value)",
                    "max": "MAX(value)",
                    "count": "COUNT(*)",
                }.get(aggregation, "AVG(value)")

                query = f"""
                    SELECT
                        DATE_TRUNC('hour', timestamp) as hour,
                        {agg_func} as value,
                        COUNT(*) as sample_count
                    FROM metrics
                    WHERE metric_name = ?
                        AND timestamp >= ?
                        AND timestamp <= ?
                    GROUP BY hour
                    ORDER BY hour
                """
            else:
                # Raw data query
                query = """
                    SELECT timestamp, value, dimensions
                    FROM metrics
                    WHERE metric_name = ?
                        AND timestamp >= ?
                        AND timestamp <= ?
                    ORDER BY timestamp
                """

            if self.conn:
                result = self.conn.execute(query, [metric_name, start_time, end_time]).fetchall()

                # Convert to dictionaries
                if aggregation:
                    data = [
                        {"timestamp": row[0].isoformat(), "value": row[1], "sample_count": row[2]}
                        for row in result
                    ]
                else:
                    data = [
                        {
                            "timestamp": row[0].isoformat(),
                            "value": row[1],
                            "dimensions": json.loads(row[2]) if row[2] else {},
                        }
                        for row in result
                    ]
            else:
                data = []

            return AnalyticsResult(
                query_type="time_series",
                start_time=start_time,
                end_time=end_time,
                data=data,
                metadata={
                    "metric_name": metric_name,
                    "aggregation": aggregation,
                    "result_count": len(data),
                },
            )

        except Exception as e:
            if self.verbose:
                print(f"Failed to query time-series: {e}")
            return AnalyticsResult(
                query_type="time_series",
                start_time=start_time,
                end_time=end_time,
                data=[],
                metadata={"error": str(e)},
            )

    def calculate_statistics(
        self, metric_name: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate statistics for a metric over a time range.

        Args:
            metric_name: Name of the metric
            start_time: Start of the time range
            end_time: End of the time range

        Returns:
            Dictionary with statistical measures
        """
        if not self.ensure_connected():
            return {"error": "Not connected"}

        try:
            query = """
                SELECT
                    COUNT(*) as count,
                    AVG(value) as mean,
                    MIN(value) as min,
                    MAX(value) as max,
                    STDDEV(value) as stddev,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q3
                FROM metrics
                WHERE metric_name = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
            """

            if self.conn:
                result = self.conn.execute(query, [metric_name, start_time, end_time]).fetchone()

                if result and result[0] > 0:  # Check count > 0
                    return {
                        "metric_name": metric_name,
                        "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                        "statistics": {
                            "count": result[0],
                            "mean": result[1],
                            "min": result[2],
                            "max": result[3],
                            "stddev": result[4],
                            "median": result[5],
                            "q1": result[6],
                            "q3": result[7],
                            "iqr": result[7] - result[6] if result[6] and result[7] else None,
                        },
                    }

            return {
                "metric_name": metric_name,
                "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                "statistics": None,
                "message": "No data found for the specified period",
            }

        except Exception as e:
            if self.verbose:
                print(f"Failed to calculate statistics: {e}")
            return {"error": str(e)}

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                if self.verbose:
                    print(f"Error closing DuckDB connection: {e}")
            finally:
                self.conn = None
                self.is_connected = False


# Export classes and types
__all__ = ["DuckDBAnalytics", "AnalyticsResult", "TimeSeriesData"]
