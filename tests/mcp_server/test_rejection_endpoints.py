#!/usr/bin/env python3
"""
Test suite for rejection audit log MCP endpoints.

Tests /tools/list_rejections and /tools/rejection_stats endpoints.
Phase 4: Truth Pillar Compliance.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Set environment before importing anything else
os.environ["ENVIRONMENT"] = "test"

from fastapi.testclient import TestClient

# Test API key (matches the default test key in api_key_auth.py)
TEST_API_KEY = "vmk_test_a1b2c3d4e5f6789012345678901234567890"
TEST_HEADERS = {"X-API-Key": TEST_API_KEY, "Content-Type": "application/json"}


@pytest.fixture(scope="module")
def client():
    """Create test client with proper environment."""
    from src.mcp_server.main import app
    return TestClient(app)


class TestListRejectionsEndpoint:
    """Test suite for /tools/list_rejections endpoint."""

    @pytest.fixture
    def mock_rejection_store(self):
        """Mock the rejection store."""
        with patch("src.storage.rejection_store.get_rejection_store") as mock_getter:
            mock_store = MagicMock()
            mock_store.list_rejections = AsyncMock(return_value=[])
            mock_getter.return_value = mock_store
            yield mock_store

    def test_list_rejections_default_params(self, client, mock_rejection_store):
        """Test list_rejections with default parameters."""
        mock_rejection_store.list_rejections = AsyncMock(return_value=[
            {
                "rejection_id": "rej-123",
                "content_hash": "abc123",
                "content_title": "Test Decision",
                "context_type": "decision",
                "weight": 0.25,
                "threshold": 0.40,
                "rejected_at": "2025-12-29T12:00:00Z",
            }
        ])

        response = client.post("/tools/list_rejections", json={}, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["rejections"]) == 1
        assert data["query"]["days"] == 7  # Default

    def test_list_rejections_with_filters(self, client, mock_rejection_store):
        """Test list_rejections with type and weight filters."""
        mock_rejection_store.list_rejections = AsyncMock(return_value=[])

        response = client.post("/tools/list_rejections", json={
            "days": 14,
            "context_type": "decision",
            "min_weight": 0.25,
            "max_weight": 0.40,
            "limit": 100,
        }, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"]["days"] == 14
        assert data["query"]["context_type"] == "decision"
        assert data["query"]["min_weight"] == 0.25
        assert data["query"]["max_weight"] == 0.40

    def test_list_rejections_empty_results(self, client, mock_rejection_store):
        """Test list_rejections with no results."""
        mock_rejection_store.list_rejections = AsyncMock(return_value=[])

        response = client.post("/tools/list_rejections", json={"days": 1}, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 0
        assert data["rejections"] == []

    def test_list_rejections_validation_error(self, client):
        """Test list_rejections with invalid parameters."""
        response = client.post("/tools/list_rejections", json={
            "days": 0,  # Invalid: must be >= 1
        }, headers=TEST_HEADERS)

        assert response.status_code == 422  # Validation error

    def test_list_rejections_invalid_context_type(self, client):
        """Test list_rejections with invalid context_type."""
        response = client.post("/tools/list_rejections", json={
            "context_type": "invalid_type",
        }, headers=TEST_HEADERS)

        assert response.status_code == 422  # Validation error


class TestRejectionStatsEndpoint:
    """Test suite for /tools/rejection_stats endpoint."""

    @pytest.fixture
    def mock_rejection_store(self):
        """Mock the rejection store."""
        with patch("src.storage.rejection_store.get_rejection_store") as mock_getter:
            mock_store = MagicMock()
            mock_store.get_stats = AsyncMock(return_value={})
            mock_getter.return_value = mock_store
            yield mock_store

    def test_rejection_stats_default(self, client, mock_rejection_store):
        """Test rejection_stats with default parameters."""
        mock_rejection_store.get_stats = AsyncMock(return_value={
            "total_rejections": 45,
            "period_days": 7,
            "by_type": {"decision": 20, "design": 15, "log": 10},
            "by_author_type": {"agent": 40, "human": 5},
            "avg_weight": 0.22,
            "avg_threshold": 0.35,
            "close_calls": 8,
            "close_call_rate": 0.18,
        })

        response = client.post("/tools/rejection_stats", json={}, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_rejections"] == 45
        assert data["stats"]["by_type"]["decision"] == 20
        assert data["stats"]["close_calls"] == 8

    def test_rejection_stats_custom_days(self, client, mock_rejection_store):
        """Test rejection_stats with custom days parameter."""
        mock_rejection_store.get_stats = AsyncMock(return_value={
            "total_rejections": 100,
            "period_days": 30,
        })

        response = client.post("/tools/rejection_stats", json={"days": 30}, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_rejection_store.get_stats.assert_called_once_with(days=30)

    def test_rejection_stats_empty(self, client, mock_rejection_store):
        """Test rejection_stats with no data."""
        mock_rejection_store.get_stats = AsyncMock(return_value={
            "total_rejections": 0,
            "period_days": 7,
            "by_type": {},
            "by_author_type": {},
            "avg_weight": 0.0,
            "avg_threshold": 0.0,
            "close_calls": 0,
            "close_call_rate": 0.0,
        })

        response = client.post("/tools/rejection_stats", json={}, headers=TEST_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_rejections"] == 0


class TestRejectionEndpointModels:
    """Test suite for request model validation."""

    def test_list_rejections_days_bounds(self, client):
        """Test days parameter bounds validation."""
        # Too low
        response = client.post("/tools/list_rejections", json={"days": 0}, headers=TEST_HEADERS)
        assert response.status_code == 422

        # Too high
        response = client.post("/tools/list_rejections", json={"days": 100}, headers=TEST_HEADERS)
        assert response.status_code == 422

    def test_list_rejections_weight_bounds(self, client):
        """Test weight parameter bounds validation."""
        # Invalid min_weight
        response = client.post("/tools/list_rejections", json={"min_weight": -0.1}, headers=TEST_HEADERS)
        assert response.status_code == 422

        response = client.post("/tools/list_rejections", json={"min_weight": 1.5}, headers=TEST_HEADERS)
        assert response.status_code == 422

    def test_list_rejections_limit_bounds(self, client):
        """Test limit parameter bounds validation."""
        # Too low
        response = client.post("/tools/list_rejections", json={"limit": 0}, headers=TEST_HEADERS)
        assert response.status_code == 422

        # Too high
        response = client.post("/tools/list_rejections", json={"limit": 300}, headers=TEST_HEADERS)
        assert response.status_code == 422

    def test_rejection_stats_days_bounds(self, client):
        """Test rejection_stats days parameter bounds."""
        # Too low
        response = client.post("/tools/rejection_stats", json={"days": 0}, headers=TEST_HEADERS)
        assert response.status_code == 422

        # Too high
        response = client.post("/tools/rejection_stats", json={"days": 100}, headers=TEST_HEADERS)
        assert response.status_code == 422
