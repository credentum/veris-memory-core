#!/usr/bin/env python3
"""
Tests for Publish Operations API.

Tests cover:
- Successful publish (commit + push + PR creation)
- Error handling (repo-manager failures)
- Connection errors
- Timeout handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp_server import publish_operations


class TestPublishChangesEndpoint:
    """Tests for /tools/publish_changes endpoint."""

    @pytest.mark.asyncio
    async def test_publish_changes_success(self):
        """Test successful publish with PR creation."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-123",
            commit_message="feat: add new feature",
            pr_title="Add new feature",
            pr_body="This PR adds a new feature to the system.",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "pr_url": "https://github.com/credentum/agent-dev/pull/42",
            "pr_number": 42,
            "commit_sha": "abc123def456",
            "branch": "task/workspace-123",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

        assert result.success is True
        assert result.pr_url == "https://github.com/credentum/agent-dev/pull/42"
        assert result.pr_number == 42
        assert result.commit_sha == "abc123def456"
        assert result.branch == "task/workspace-123"
        assert result.message == "Changes published successfully"

    @pytest.mark.asyncio
    async def test_publish_changes_with_draft_pr(self):
        """Test publish with draft PR option."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-draft",
            commit_message="wip: work in progress",
            pr_title="Draft: Work in progress",
            pr_body="WIP - not ready for review",
            draft=True,
            base="develop",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "pr_url": "https://github.com/credentum/agent-dev/pull/43",
            "pr_number": 43,
            "commit_sha": "def789",
            "branch": "task/workspace-draft",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

            # Verify the request payload included draft and base
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["draft"] is True
            assert call_args[1]["json"]["base"] == "develop"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_publish_changes_with_specific_files(self):
        """Test publish with specific files to commit."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-files",
            commit_message="fix: update specific files",
            pr_title="Fix specific files",
            pr_body="Only commit specific files",
            files=["src/main.py", "tests/test_main.py"],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "pr_url": "https://github.com/credentum/agent-dev/pull/44",
            "pr_number": 44,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

            # Verify files were included in payload
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["files"] == ["src/main.py", "tests/test_main.py"]

        assert result.success is True

    @pytest.mark.asyncio
    async def test_publish_changes_repo_manager_error(self):
        """Test handling of repo-manager HTTP error."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-error",
            commit_message="feat: will fail",
            pr_title="Will fail",
            pr_body="This should fail",
        )

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error: git push failed"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

        assert result.success is False
        assert "500" in result.message
        assert "git push failed" in result.error

    @pytest.mark.asyncio
    async def test_publish_changes_repo_manager_not_found(self):
        """Test handling of repo-manager 404 (workspace not found)."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="nonexistent-workspace",
            commit_message="feat: will fail",
            pr_title="Will fail",
            pr_body="Workspace doesn't exist",
        )

        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.text = "Workspace not found: nonexistent-workspace"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

        assert result.success is False
        assert "404" in result.message
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_publish_changes_connection_error(self):
        """Test handling of connection error to repo-manager."""
        import httpx

        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-conn-error",
            commit_message="feat: will fail",
            pr_title="Will fail",
            pr_body="Connection will fail",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

        assert result.success is False
        assert "Cannot connect to repo-manager" in result.message
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_publish_changes_timeout(self):
        """Test handling of timeout to repo-manager."""
        import httpx

        request = publish_operations.PublishChangesRequest(
            workspace_id="workspace-timeout",
            commit_message="feat: will timeout",
            pr_title="Will timeout",
            pr_body="Request will timeout",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await publish_operations.publish_changes(request)

        assert result.success is False
        assert "Timeout" in result.message
        assert "timed out" in result.error.lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_repo_manager_headers_with_key(self):
        """Test header generation with API key."""
        with patch.object(publish_operations, "REPO_MANAGER_API_KEY", "vmk_repo_test123"):
            headers = publish_operations.get_repo_manager_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["X-API-Key"] == "vmk_repo_test123"

    def test_get_repo_manager_headers_with_server_format_key(self):
        """Test header generation with server-format API key (key:user:role:agent)."""
        with patch.object(
            publish_operations,
            "REPO_MANAGER_API_KEY",
            "vmk_repo_test123:repo_manager:writer:true",
        ):
            headers = publish_operations.get_repo_manager_headers()

        assert headers["Content-Type"] == "application/json"
        # Should extract just the key part (before first colon)
        assert headers["X-API-Key"] == "vmk_repo_test123"

    def test_get_repo_manager_headers_no_key(self):
        """Test header generation without API key."""
        with patch.object(publish_operations, "REPO_MANAGER_API_KEY", ""):
            headers = publish_operations.get_repo_manager_headers()

        assert headers["Content-Type"] == "application/json"
        assert "X-API-Key" not in headers


class TestModels:
    """Tests for Pydantic models."""

    def test_publish_changes_request_minimal(self):
        """Test request with minimal required fields."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="ws-123",
            commit_message="fix: bug",
            pr_title="Bug fix",
        )

        assert request.workspace_id == "ws-123"
        assert request.commit_message == "fix: bug"
        assert request.pr_title == "Bug fix"
        assert request.pr_body == ""
        assert request.draft is False
        assert request.base == "main"
        assert request.files == []

    def test_publish_changes_request_full(self):
        """Test request with all fields."""
        request = publish_operations.PublishChangesRequest(
            workspace_id="ws-456",
            commit_message="feat: new feature",
            pr_title="Add feature",
            pr_body="## Summary\nNew feature added",
            draft=True,
            base="develop",
            files=["src/feature.py"],
        )

        assert request.workspace_id == "ws-456"
        assert request.commit_message == "feat: new feature"
        assert request.pr_title == "Add feature"
        assert request.pr_body == "## Summary\nNew feature added"
        assert request.draft is True
        assert request.base == "develop"
        assert request.files == ["src/feature.py"]

    def test_publish_changes_response_success(self):
        """Test success response."""
        response = publish_operations.PublishChangesResponse(
            success=True,
            pr_url="https://github.com/org/repo/pull/1",
            pr_number=1,
            commit_sha="abc123",
            branch="task/ws-123",
            message="Published",
        )

        assert response.success is True
        assert response.pr_url == "https://github.com/org/repo/pull/1"
        assert response.pr_number == 1
        assert response.error is None

    def test_publish_changes_response_failure(self):
        """Test failure response."""
        response = publish_operations.PublishChangesResponse(
            success=False,
            message="Publish failed",
            error="Git push rejected",
        )

        assert response.success is False
        assert response.pr_url is None
        assert response.pr_number is None
        assert response.error == "Git push rejected"
