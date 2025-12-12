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
- Workspace isolation by user_id
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.mcp_server import file_operations


class TestPathValidation:
    """Tests for path validation and security."""

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            file_operations.FileWriteRequest(
                user_id="test_user",
                path="../etc/passwd",
                content="malicious",
            )

    def test_path_traversal_nested_blocked(self):
        """Test nested path traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            file_operations.FileReadRequest(
                user_id="test_user",
                path="data/../../secrets",
            )

    def test_absolute_path_blocked(self):
        """Test absolute paths are blocked."""
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            file_operations.FileWriteRequest(
                user_id="test_user",
                path="/etc/passwd",
                content="content",
            )

    def test_null_byte_blocked(self):
        """Test null bytes in path are blocked."""
        with pytest.raises(ValueError, match="Null bytes not allowed"):
            file_operations.FileReadRequest(
                user_id="test_user",
                path="file.txt\x00.exe",
            )

    def test_empty_path_blocked(self):
        """Test empty path is blocked for write/read/delete."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            file_operations.FileWriteRequest(
                user_id="test_user",
                path="",
                content="content",
            )

    def test_path_too_long_blocked(self):
        """Test path length limit."""
        long_path = "a" * 501
        with pytest.raises(ValueError, match="Path too long"):
            file_operations.FileReadRequest(
                user_id="test_user",
                path=long_path,
            )

    def test_valid_path_accepted(self):
        """Test valid paths are accepted."""
        request = file_operations.FileWriteRequest(
            user_id="test_user",
            path="data/output.json",
            content='{"key": "value"}',
        )
        assert request.path == "data/output.json"

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed from path."""
        request = file_operations.FileReadRequest(
            user_id="test_user",
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
                    user_id="test_user",
                    path="large_file.txt",
                    content=large_content,
                )


class TestWorkspacePath:
    """Tests for workspace path resolution."""

    def test_get_workspace_path_valid(self):
        """Test valid user_id generates correct workspace path."""
        with patch.object(file_operations, "WORKSPACE_BASE_DIR", "/veris_storage/workspaces"):
            path = file_operations.get_workspace_path("dev_team")
            assert path == Path("/veris_storage/workspaces/dev_team")

    def test_get_workspace_path_sanitizes_user_id(self):
        """Test user_id with special chars is sanitized."""
        with patch.object(file_operations, "WORKSPACE_BASE_DIR", "/veris_storage/workspaces"):
            path = file_operations.get_workspace_path("user@email.com")
            # @ and . should be stripped, leaving "useremailcom"
            assert "useremailcom" in str(path)

    def test_get_workspace_path_empty_user_id(self):
        """Test empty user_id raises error."""
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            file_operations.get_workspace_path("")

    def test_get_workspace_path_invalid_chars_only(self):
        """Test user_id with only invalid chars raises error."""
        with pytest.raises(ValueError, match="user_id contains no valid characters"):
            file_operations.get_workspace_path("@#$%")


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
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        user_id="test_team",
                        path="output.txt",
                        content="Hello, World!",
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is True
                    assert result.bytes_written == 13
                    assert result.path == "output.txt"

                    # Verify file was created
                    written_path = Path(tmpdir) / "test_team" / "output.txt"
                    assert written_path.exists()
                    assert written_path.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_file_write_creates_directories(self):
        """Test file write creates parent directories."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        user_id="test_team",
                        path="deep/nested/dir/file.txt",
                        content="content",
                        create_dirs=True,
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is True
                    written_path = Path(tmpdir) / "test_team" / "deep/nested/dir/file.txt"
                    assert written_path.exists()

    @pytest.mark.asyncio
    async def test_file_write_no_overwrite(self):
        """Test file write respects overwrite=False."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    # Create workspace and initial file
                    workspace = Path(tmpdir) / "test_team"
                    workspace.mkdir(parents=True)
                    (workspace / "existing.txt").write_text("original")

                    request = file_operations.FileWriteRequest(
                        user_id="test_team",
                        path="existing.txt",
                        content="new content",
                        overwrite=False,
                    )
                    result = await file_operations.file_write(request)

                    assert result.success is False
                    assert "already exists" in result.message

                    # Verify original content unchanged
                    assert (workspace / "existing.txt").read_text() == "original"

    @pytest.mark.asyncio
    async def test_file_write_logs_operation(self):
        """Test file write logs to Redis."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileWriteRequest(
                        user_id="test_team",
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
            # Create workspace and file
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "data.txt").write_text("Test content")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        user_id="test_team",
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
            # Create empty workspace
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        user_id="test_team",
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
            # Create workspace and directory
            workspace = Path(tmpdir) / "test_team"
            (workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileReadRequest(
                        user_id="test_team",
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
            # Create workspace with files
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "file1.txt").write_text("content1")
            (workspace / "file2.txt").write_text("content2")
            (workspace / "subdir").mkdir()

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        user_id="test_team",
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
            # Create workspace with nested structure
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "file1.txt").write_text("content")
            (workspace / "subdir").mkdir()
            (workspace / "subdir" / "nested.txt").write_text("nested")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        user_id="test_team",
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
            # Create workspace with hidden file
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "visible.txt").write_text("visible")
            (workspace / ".hidden").write_text("hidden")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        user_id="test_team",
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
            # Create workspace with hidden file
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "visible.txt").write_text("visible")
            (workspace / ".hidden").write_text("hidden")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        user_id="test_team",
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
            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileListRequest(
                        user_id="new_user",
                        path="",
                    )
                    result = await file_operations.file_list(request)

                    assert result.success is True
                    assert result.total_count == 0
                    assert (Path(tmpdir) / "new_user").exists()


class TestFileDeleteEndpoint:
    """Tests for file_delete endpoint."""

    @pytest.mark.asyncio
    async def test_file_delete_success(self):
        """Test successful file deletion."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workspace with file
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            target = workspace / "to_delete.txt"
            target.write_text("delete me")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        user_id="test_team",
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
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        user_id="test_team",
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
            # Create workspace with directory
            workspace = Path(tmpdir) / "test_team"
            (workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    request = file_operations.FileDeleteRequest(
                        user_id="test_team",
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
            # Create workspace with file
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)
            (workspace / "exists.txt").write_text("content")

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    user_id="test_team",
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
            workspace = Path(tmpdir) / "test_team"
            workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    user_id="test_team",
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
            # Create workspace with directory
            workspace = Path(tmpdir) / "test_team"
            (workspace / "subdir").mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                request = file_operations.FileExistsRequest(
                    user_id="test_team",
                    path="subdir",
                )
                result = await file_operations.file_exists(request)

                assert result.exists is True
                assert result.is_file is False
                assert result.is_dir is True


class TestWorkspaceIsolation:
    """Tests for workspace isolation between users."""

    @pytest.mark.asyncio
    async def test_users_have_separate_workspaces(self):
        """Test different users can't access each other's files."""
        mock_redis = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workspaces for two users
            user1_workspace = Path(tmpdir) / "user1"
            user1_workspace.mkdir(parents=True)
            (user1_workspace / "secret.txt").write_text("user1 secret")

            user2_workspace = Path(tmpdir) / "user2"
            user2_workspace.mkdir(parents=True)

            with patch.object(file_operations, "WORKSPACE_BASE_DIR", tmpdir):
                with patch.object(file_operations, "_redis_client", mock_redis):
                    # user2 should not see user1's file
                    request = file_operations.FileReadRequest(
                        user_id="user2",
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
