#!/usr/bin/env python3
"""
Tests for File Operations API.

Tests cover:
- File write (file_write)
- File read (file_read)
- File list (file_list)
- File delete (file_delete)
- File exists (file_exists)
- Path validation and security (traversal prevention)
- Workspace isolation by workspace_path
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.mcp_server import file_operations


# Default workspace base for tests
TEST_WORKSPACE_BASE = "/veris_storage/workspaces"


class TestPathValidation:
    """Tests for path validation and security."""

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            file_operations.FileWriteRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path="../etc/passwd",
                content="malicious",
            )

    def test_path_traversal_nested_blocked(self):
        """Test nested path traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            file_operations.FileReadRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path="data/../../secrets",
            )

    def test_absolute_path_blocked(self):
        """Test absolute paths are blocked."""
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            file_operations.FileWriteRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path="/etc/passwd",
                content="content",
            )

    def test_null_byte_blocked(self):
        """Test null bytes in path are blocked."""
        with pytest.raises(ValueError, match="Null bytes not allowed"):
            file_operations.FileReadRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path="file.txt\x00.exe",
            )

    def test_empty_path_blocked(self):
        """Test empty path is blocked for write/read/delete."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            file_operations.FileWriteRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path="",
                content="content",
            )

    def test_path_too_long_blocked(self):
        """Test path length limit."""
        long_path = "a" * 501
        with pytest.raises(ValueError, match="Path too long"):
            file_operations.FileReadRequest(
                workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                path=long_path,
            )

    def test_valid_path_accepted(self):
        """Test valid paths are accepted."""
        request = file_operations.FileWriteRequest(
            workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
            path="data/output.json",
            content='{"key": "value"}',
        )
        assert request.path == "data/output.json"

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed from path."""
        request = file_operations.FileReadRequest(
            workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
            path="  data/file.txt  ",
        )
        assert request.path == "data/file.txt"


class TestContentValidation:
    """Tests for content validation."""

    def test_content_size_limit(self):
        """Test content size limit is enforced."""
        # Default limit is 10MB, but we'll test with a mock
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        with patch.object(file_operations, "MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024):
            with pytest.raises(ValueError, match="Content too large"):
                file_operations.FileWriteRequest(
                    workspace_path=f"{TEST_WORKSPACE_BASE}/test_task",
                    path="large_file.txt",
                    content=large_content,
                )


class TestWorkspacePath:
    """Tests for workspace path resolution."""

    def test_get_workspace_path_valid(self):
        """Test valid workspace_path returns Path object."""
        with patch.object(file_operations, "WORKSPACE_BASE_DIR", "/veris_storage/workspaces"):
            path = file_operations.get_workspace_path("/veris_storage/workspaces/task-123")
            assert path == Path("/veris_storage/workspaces/task-123")

    def test_get_workspace_path_outside_base_blocked(self):
        """Test workspace_path outside base directory is blocked."""
        with patch.object(file_operations, "WORKSPACE_BASE_DIR", "/veris_storage/workspaces"):
            with pytest.raises(ValueError, match="workspace_path must be under"):
                file_operations.get_workspace_path("/tmp/malicious")

    def test_get_workspace_path_empty_blocked(self):
        """Test empty workspace_path raises error."""
        with pytest.raises(ValueError, match="workspace_path cannot be empty"):
            file_operations.get_workspace_path("")

    def test_workspace_path_validation_in_request(self):
        """Test workspace_path must be under base directory."""
        with pytest.raises(ValueError, match="workspace_path must be under"):
            file_operations.FileWriteRequest(
                workspace_path="/tmp/outside",
                path="file.txt",
                content="content",
            )


class TestResolveSafePath:
    """Tests for safe path resolution."""

    def test_resolve_normal_path(self):
        """Test normal path resolution works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            result = file_operations.resolve_safe_path(workspace, "data/file.txt")
            assert str(result).startswith(str(workspace.resolve()))
            assert result.name == "file.txt"

    def test_resolve_blocks_escape(self):
        """Test paths that try to escape workspace are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Even though we validate .. in request models, double-check here
            with pytest.raises(ValueError, match="escapes workspace"):
                file_operations.resolve_safe_path(workspace, "../outside")


class TestFileWriteEndpoint:
    """Tests for file_write endpoint."""

    @pytest.mark.asyncio
    async def test_file_write_success(self):
        """Test successful file write."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace
            task_workspace = Path(tmpdir) / "task-123"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        workspace_path=str(task_workspace),
                        path="output.txt",
                        content="Hello, World!",
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is True
                    assert result.bytes_written == 13
                    assert result.path == "output.txt"

                    # Verify file was created
                    written_path = task_workspace / "output.txt"
                    assert written_path.exists()
                    assert written_path.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_file_write_creates_directories(self):
        """Test file write creates parent directories."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-456"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        workspace_path=str(task_workspace),
                        path="deep/nested/dir/file.txt",
                        content="content",
                        create_dirs=True,
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is True
                    written_path = task_workspace / "deep/nested/dir/file.txt"
                    assert written_path.exists()

    @pytest.mark.asyncio
    async def test_file_write_no_overwrite(self):
        """Test file write respects overwrite=False."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-789"
            task_workspace.mkdir(parents=True)
            (task_workspace / "existing.txt").write_text("original")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        workspace_path=str(task_workspace),
                        path="existing.txt",
                        content="new content",
                        overwrite=False,
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is False
                    assert "already exists" in result.message

                    # Verify original content unchanged
                    assert (task_workspace / "existing.txt").read_text() == "original"

    @pytest.mark.asyncio
    async def test_file_write_logs_operation(self):
        """Test file write logs to Redis."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-log"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        workspace_path=str(task_workspace),
                        path="tracked.txt",
                        content="content",
                    )
                    await file_operations.file_write(request)

                    # Verify Redis logging
                    mock_redis.lpush.assert_called()
                    call_key = mock_redis.lpush.call_args[0][0]
                    assert "file_ops" in call_key


class TestFileReadEndpoint:
    """Tests for file_read endpoint."""

    @pytest.mark.asyncio
    async def test_file_read_success(self):
        """Test successful file read."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace and file
            task_workspace = Path(tmpdir) / "task-read-1"
            task_workspace.mkdir(parents=True)
            (task_workspace / "data.txt").write_text("Test content")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        workspace_path=str(task_workspace),
                        path="data.txt",
                    )
                    result = await file_operations.file_read(request)

                    assert result.success is True
                    assert result.content == "Test content"
                    assert result.size_bytes == 12

    @pytest.mark.asyncio
    async def test_file_read_not_found(self):
        """Test file read returns error for missing file."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty task workspace
            task_workspace = Path(tmpdir) / "task-read-2"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        workspace_path=str(task_workspace),
                        path="nonexistent.txt",
                    )
                    result = await file_operations.file_read(request)

                    assert result.success is False
                    assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_file_read_directory_error(self):
        """Test file read returns error for directory."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace and directory
            task_workspace = Path(tmpdir) / "task-read-3"
            (task_workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        workspace_path=str(task_workspace),
                        path="subdir",
                    )
                    result = await file_operations.file_read(request)

                    assert result.success is False
                    assert "not a file" in result.message.lower()


class TestFileListEndpoint:
    """Tests for file_list endpoint."""

    @pytest.mark.asyncio
    async def test_file_list_success(self):
        """Test successful directory listing."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with files
            task_workspace = Path(tmpdir) / "task-list-1"
            task_workspace.mkdir(parents=True)
            (task_workspace / "file1.txt").write_text("content1")
            (task_workspace / "file2.txt").write_text("content2")
            (task_workspace / "subdir").mkdir()

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        workspace_path=str(task_workspace),
                        path="",
                    )
                    result = await file_operations.file_list(request)

                    assert result.success is True
                    assert result.total_count == 3  # 2 files + 1 dir
                    names = [f.name for f in result.files]
                    assert "file1.txt" in names
                    assert "file2.txt" in names
                    assert "subdir" in names

    @pytest.mark.asyncio
    async def test_file_list_recursive(self):
        """Test recursive directory listing."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with nested structure
            task_workspace = Path(tmpdir) / "task-list-2"
            task_workspace.mkdir(parents=True)
            (task_workspace / "file1.txt").write_text("content")
            (task_workspace / "subdir").mkdir()
            (task_workspace / "subdir" / "nested.txt").write_text("nested")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        workspace_path=str(task_workspace),
                        path="",
                        recursive=True,
                    )
                    result = await file_operations.file_list(request)

                    assert result.success is True
                    paths = [f.path for f in result.files]
                    assert any("nested.txt" in p for p in paths)

    @pytest.mark.asyncio
    async def test_file_list_excludes_hidden(self):
        """Test hidden files are excluded by default."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with hidden file
            task_workspace = Path(tmpdir) / "task-list-3"
            task_workspace.mkdir(parents=True)
            (task_workspace / "visible.txt").write_text("visible")
            (task_workspace / ".hidden").write_text("hidden")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        workspace_path=str(task_workspace),
                        path="",
                        include_hidden=False,
                    )
                    result = await file_operations.file_list(request)

                    names = [f.name for f in result.files]
                    assert "visible.txt" in names
                    assert ".hidden" not in names

    @pytest.mark.asyncio
    async def test_file_list_includes_hidden(self):
        """Test hidden files included when requested."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with hidden file
            task_workspace = Path(tmpdir) / "task-list-4"
            task_workspace.mkdir(parents=True)
            (task_workspace / "visible.txt").write_text("visible")
            (task_workspace / ".hidden").write_text("hidden")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        workspace_path=str(task_workspace),
                        path="",
                        include_hidden=True,
                    )
                    result = await file_operations.file_list(request)

                    names = [f.name for f in result.files]
                    assert "visible.txt" in names
                    assert ".hidden" in names

    @pytest.mark.asyncio
    async def test_file_list_creates_empty_workspace(self):
        """Test listing creates workspace if it doesn't exist."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-list-new"

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        workspace_path=str(task_workspace),
                        path="",
                    )
                    result = await file_operations.file_list(request)

                    assert result.success is True
                    assert result.total_count == 0
                    assert task_workspace.exists()


class TestFileDeleteEndpoint:
    """Tests for file_delete endpoint."""

    @pytest.mark.asyncio
    async def test_file_delete_success(self):
        """Test successful file deletion."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with file
            task_workspace = Path(tmpdir) / "task-del-1"
            task_workspace.mkdir(parents=True)
            target = task_workspace / "to_delete.txt"
            target.write_text("delete me")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        workspace_path=str(task_workspace),
                        path="to_delete.txt",
                    )
                    result = await file_operations.file_delete(request)

                    assert result.success is True
                    assert not target.exists()

    @pytest.mark.asyncio
    async def test_file_delete_not_found(self):
        """Test delete returns error for missing file."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-del-2"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        workspace_path=str(task_workspace),
                        path="nonexistent.txt",
                    )
                    result = await file_operations.file_delete(request)

                    assert result.success is False
                    assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_file_delete_directory_blocked(self):
        """Test delete refuses to delete directories."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with directory
            task_workspace = Path(tmpdir) / "task-del-3"
            (task_workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        workspace_path=str(task_workspace),
                        path="subdir",
                    )
                    result = await file_operations.file_delete(request)

                    assert result.success is False
                    assert "only files" in result.message.lower() or "cannot delete directories" in result.message.lower()


class TestFileExistsEndpoint:
    """Tests for file_exists endpoint."""

    @pytest.mark.asyncio
    async def test_file_exists_true(self):
        """Test file_exists returns True for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with file
            task_workspace = Path(tmpdir) / "task-exists-1"
            task_workspace.mkdir(parents=True)
            (task_workspace / "exists.txt").write_text("content")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    workspace_path=str(task_workspace),
                    path="exists.txt",
                )
                result = await file_operations.file_exists(request)

                assert result.exists is True
                assert result.is_file is True
                assert result.is_dir is False
                assert result.size_bytes > 0

    @pytest.mark.asyncio
    async def test_file_exists_false(self):
        """Test file_exists returns False for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_workspace = Path(tmpdir) / "task-exists-2"
            task_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    workspace_path=str(task_workspace),
                    path="nonexistent.txt",
                )
                result = await file_operations.file_exists(request)

                assert result.exists is False
                assert result.is_file is False
                assert result.is_dir is False

    @pytest.mark.asyncio
    async def test_file_exists_directory(self):
        """Test file_exists correctly identifies directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task workspace with directory
            task_workspace = Path(tmpdir) / "task-exists-3"
            (task_workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    workspace_path=str(task_workspace),
                    path="subdir",
                )
                result = await file_operations.file_exists(request)

                assert result.exists is True
                assert result.is_file is False
                assert result.is_dir is True


class TestWorkspaceIsolation:
    """Tests for workspace isolation between tasks."""

    @pytest.mark.asyncio
    async def test_tasks_have_separate_workspaces(self):
        """Test different tasks have isolated workspaces."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workspaces for two tasks
            task1_workspace = Path(tmpdir) / "task-1"
            task1_workspace.mkdir(parents=True)
            (task1_workspace / "secret.txt").write_text("task1 secret")

            task2_workspace = Path(tmpdir) / "task-2"
            task2_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    # task2 workspace should not see task1's file
                    request = file_operations.FileReadRequest(
                        workspace_path=str(task2_workspace),
                        path="secret.txt",
                    )
                    result = await file_operations.file_read(request)

                    assert result.success is False
                    assert "not found" in result.message.lower()


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_routes(self):
        """Test routes are registered correctly."""
        mock_app = Mock()
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                file_operations.register_routes(mock_app, mock_redis)

                # Verify router was included
                mock_app.include_router.assert_called_once_with(file_operations.router)

                # Verify redis client was set
                assert file_operations._redis_client == mock_redis
