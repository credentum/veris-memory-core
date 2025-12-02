"""
Unit tests for delete/forget endpoints - Redis client parameter passing

This test suite specifically verifies that the redis_client parameter is
correctly passed to delete_operations functions, addressing the bug where
simple_redis.redis_client was incorrectly accessed instead of simple_redis.

Regression tests included to ensure the AttributeError doesn't recur.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from src.mcp_server.main import (
    delete_context_rest_endpoint,
    delete_context_endpoint,
    forget_context_endpoint,
    DeleteContextRequest,
    ForgetContextRequest,
    APIKeyInfo
)


class TestDeleteEndpointsRedisClient:
    """
    Test suite for delete endpoints focusing on redis_client parameter.

    These tests specifically verify the fix for:
    Bug: 'SimpleRedisClient' object has no attribute 'redis_client'
    """

    @pytest.fixture
    def mock_simple_redis(self):
        """Mock SimpleRedisClient with methods needed by DeleteAuditLogger"""
        redis_client = Mock()
        redis_client.setex = Mock(return_value=True)
        redis_client.get = Mock(return_value=None)
        redis_client.keys = Mock(return_value=[])
        redis_client.delete = Mock(return_value=1)
        return redis_client

    @pytest.fixture
    def mock_api_key_info(self):
        """Mock API key info for human user"""
        return APIKeyInfo(
            key_id="test_key",
            user_id="test_user",
            role="admin",
            capabilities=["delete", "forget"],
            is_agent=False,
            metadata={}
        )

    @pytest.fixture
    def mock_agent_api_key_info(self):
        """Mock API key info for agent"""
        return APIKeyInfo(
            key_id="agent_key",
            user_id="test_agent",
            role="writer",
            capabilities=["delete", "forget"],
            is_agent=True,
            metadata={}
        )

    @pytest.mark.asyncio
    async def test_delete_context_rest_endpoint_passes_simple_redis_directly(
        self, mock_simple_redis, mock_api_key_info
    ):
        """
        Verify delete_context_rest_endpoint passes simple_redis (not .redis_client)

        This is a REGRESSION TEST for the bug where the code incorrectly
        tried to access simple_redis.redis_client.
        """
        request = DeleteContextRequest(
            context_id="test-context-id",
            reason="Test deletion",
            hard_delete=True
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.qdrant_client") as mock_qdrant, \
             patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
             patch("src.tools.delete_operations.delete_context") as mock_delete_func:

            # Set up mock to return success
            mock_delete_func.return_value = {
                "success": True,
                "operation": "hard_delete",
                "context_id": "test-context-id"
            }

            # Call the endpoint
            result = await delete_context_rest_endpoint(
                context_id=request.context_id,
                reason=request.reason,
                hard_delete=request.hard_delete,
                api_key_info=mock_api_key_info
            )

            # Verify delete_context was called with simple_redis directly
            mock_delete_func.assert_called_once()
            call_args = mock_delete_func.call_args

            # The redis_client parameter should be simple_redis, NOT simple_redis.redis_client
            assert call_args.kwargs["redis_client"] is mock_simple_redis

    @pytest.mark.asyncio
    async def test_delete_context_endpoint_passes_simple_redis_directly(
        self, mock_simple_redis, mock_api_key_info
    ):
        """
        Verify delete_context_endpoint passes simple_redis (not .redis_client)

        Tests the MCP endpoint variant.
        """
        request = DeleteContextRequest(
            context_id="test-context-id-2",
            reason="Test deletion MCP",
            hard_delete=False
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.qdrant_client") as mock_qdrant, \
             patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
             patch("src.tools.delete_operations.delete_context") as mock_delete_func:

            mock_delete_func.return_value = {
                "success": True,
                "operation": "soft_delete",
                "context_id": "test-context-id-2"
            }

            result = await delete_context_endpoint(
                request=request,
                api_key_info=mock_api_key_info
            )

            # Verify correct parameter passing
            mock_delete_func.assert_called_once()
            call_args = mock_delete_func.call_args
            assert call_args.kwargs["redis_client"] is mock_simple_redis

    @pytest.mark.asyncio
    async def test_forget_context_endpoint_passes_simple_redis_directly(
        self, mock_simple_redis, mock_api_key_info
    ):
        """
        Verify forget_context_endpoint passes simple_redis (not .redis_client)

        Tests the forget endpoint which uses the same pattern.
        """
        request = ForgetContextRequest(
            context_id="test-forget-id",
            reason="Test forget",
            retention_days=7
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
             patch("src.tools.delete_operations.forget_context") as mock_forget_func:

            mock_forget_func.return_value = {
                "success": True,
                "operation": "forget",
                "context_id": "test-forget-id"
            }

            result = await forget_context_endpoint(
                request=request,
                api_key_info=mock_api_key_info
            )

            # Verify correct parameter passing
            mock_forget_func.assert_called_once()
            call_args = mock_forget_func.call_args
            assert call_args.kwargs["redis_client"] is mock_simple_redis

    @pytest.mark.asyncio
    async def test_regression_simple_redis_without_redis_client_attribute(
        self, mock_api_key_info
    ):
        """
        REGRESSION TEST: Verify endpoints work with SimpleRedisClient that
        has NO redis_client attribute (the original bug scenario)

        This test would have FAILED before the fix, catching the bug.
        """
        # Create a mock that explicitly does NOT have redis_client attribute
        mock_simple_redis = Mock(spec=["setex", "get", "keys", "delete", "connect", "set"])
        mock_simple_redis.setex = Mock(return_value=True)
        mock_simple_redis.get = Mock(return_value=None)

        # Verify our mock doesn't have the attribute
        assert not hasattr(mock_simple_redis, 'redis_client')

        request = DeleteContextRequest(
            context_id="regression-test-id",
            reason="Regression test",
            hard_delete=True
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.qdrant_client") as mock_qdrant, \
             patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
             patch("src.tools.delete_operations.delete_context") as mock_delete_func:

            mock_delete_func.return_value = {"success": True}

            # This should NOT raise AttributeError (the bug we fixed)
            try:
                result = await delete_context_rest_endpoint(
                    context_id=request.context_id,
                    reason=request.reason,
                    hard_delete=request.hard_delete,
                    api_key_info=mock_api_key_info
                )

                # Verify it succeeded
                mock_delete_func.assert_called_once()

            except AttributeError as e:
                if "redis_client" in str(e):
                    pytest.fail(
                        f"AttributeError raised - BUG IS BACK! "
                        f"Code is trying to access .redis_client: {e}"
                    )
                else:
                    raise  # Different AttributeError, re-raise

    @pytest.mark.asyncio
    async def test_delete_audit_logger_receives_simple_redis_instance(
        self, mock_simple_redis, mock_api_key_info
    ):
        """
        Verify DeleteAuditLogger receives SimpleRedisClient instance
        and can call its methods (.setex, .get, .keys)
        """
        request = DeleteContextRequest(
            context_id="audit-test-id",
            reason="Audit logging test",
            hard_delete=True
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.qdrant_client") as mock_qdrant, \
             patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
             patch("src.tools.delete_operations.DeleteAuditLogger") as MockAuditLogger:

            # Create a mock audit logger instance
            mock_audit_instance = Mock()
            mock_audit_instance.log_deletion = Mock(return_value="audit-id-123")
            MockAuditLogger.return_value = mock_audit_instance

            # Mock Neo4j to return successful deletion
            mock_neo4j.query = Mock(return_value=[{"deleted_count": 1}])

            # Import and call the actual delete_context function
            from src.tools.delete_operations import delete_context

            result = await delete_context(
                context_id="audit-test-id",
                reason="Audit logging test",
                hard_delete=True,
                api_key_info=mock_api_key_info,
                neo4j_client=mock_neo4j,
                qdrant_client=mock_qdrant,
                redis_client=mock_simple_redis  # Pass simple_redis directly
            )

            # Verify DeleteAuditLogger was initialized with simple_redis
            MockAuditLogger.assert_called_once_with(mock_simple_redis)

            # Verify audit logging was called
            assert mock_audit_instance.log_deletion.called

    @pytest.mark.asyncio
    async def test_audit_logger_calls_simple_redis_setex_successfully(
        self, mock_simple_redis
    ):
        """
        Verify audit logging can successfully call SimpleRedisClient.setex()

        This validates that SimpleRedisClient has the interface DeleteAuditLogger needs.
        """
        from src.tools.delete_operations import DeleteAuditLogger
        from datetime import timedelta

        # Create real DeleteAuditLogger with our mock SimpleRedisClient
        audit_logger = DeleteAuditLogger(mock_simple_redis)

        # Log a deletion
        audit_id = audit_logger.log_deletion(
            context_id="test-context",
            reason="Test reason",
            deleted_by="test_user",
            author_type="human",
            operation_type="delete",
            hard_delete=True
        )

        # Verify setex was called on simple_redis (not .redis_client)
        assert mock_simple_redis.setex.called

        # Verify the call structure
        call_args = mock_simple_redis.setex.call_args
        assert "audit:delete:" in call_args[0][0]  # Key starts with audit:delete:
        assert isinstance(call_args[0][1], timedelta)  # Second arg is timedelta
        assert audit_id in call_args[0][2]  # Audit entry contains the audit_id

    @pytest.mark.asyncio
    async def test_all_three_endpoints_use_consistent_pattern(
        self, mock_simple_redis, mock_api_key_info
    ):
        """
        Verify all three delete endpoints use the same correct pattern

        Ensures consistency across:
        - delete_context_rest_endpoint
        - delete_context_endpoint
        - forget_context_endpoint
        """
        test_cases = [
            {
                "endpoint": delete_context_rest_endpoint,
                "request": DeleteContextRequest(
                    context_id="rest-test",
                    reason="REST test",
                    hard_delete=True
                ),
                "patch_func": "delete_context"
            },
            {
                "endpoint": delete_context_endpoint,
                "request": DeleteContextRequest(
                    context_id="mcp-test",
                    reason="MCP test",
                    hard_delete=False
                ),
                "patch_func": "delete_context"
            },
            {
                "endpoint": forget_context_endpoint,
                "request": ForgetContextRequest(
                    context_id="forget-test",
                    reason="Forget test",
                    retention_days=7
                ),
                "patch_func": "forget_context"
            }
        ]

        for test_case in test_cases:
            with patch("src.mcp_server.main.neo4j_client"), \
                 patch("src.mcp_server.main.qdrant_client"), \
                 patch("src.mcp_server.main.simple_redis", mock_simple_redis), \
                 patch(f"src.tools.delete_operations.{test_case['patch_func']}") as mock_func:

                mock_func.return_value = {"success": True}

                # Call the endpoint (delete_context_rest_endpoint has different signature)
                request_obj = test_case["request"]
                if test_case["endpoint"] == delete_context_rest_endpoint:
                    await test_case["endpoint"](
                        context_id=request_obj.context_id,
                        reason=request_obj.reason,
                        hard_delete=request_obj.hard_delete,
                        api_key_info=mock_api_key_info
                    )
                else:
                    await test_case["endpoint"](
                        request=request_obj,
                        api_key_info=mock_api_key_info
                    )

                # Verify simple_redis was passed (not .redis_client)
                mock_func.assert_called_once()
                call_args = mock_func.call_args
                assert call_args.kwargs["redis_client"] is mock_simple_redis

                mock_func.reset_mock()


class TestDeleteEndpointsEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.mark.asyncio
    async def test_delete_with_none_simple_redis(self):
        """Verify endpoints handle None simple_redis gracefully"""
        request = DeleteContextRequest(
            context_id="test-id",
            reason="Test with None redis",
            hard_delete=True
        )

        mock_api_key_info = APIKeyInfo(
            key_id="test_key",
            user_id="test_user",
            role="admin",
            capabilities=["delete"],
            is_agent=False,
            metadata={}
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.qdrant_client") as mock_qdrant, \
             patch("src.mcp_server.main.simple_redis", None), \
             patch("src.tools.delete_operations.delete_context") as mock_delete_func:

            mock_delete_func.return_value = {"success": True}

            # Should pass None without error
            result = await delete_context_rest_endpoint(
                context_id=request.context_id,
                reason=request.reason,
                hard_delete=request.hard_delete,
                api_key_info=mock_api_key_info
            )

            # Verify None was passed (not None.redis_client which would error)
            call_args = mock_delete_func.call_args
            assert call_args.kwargs["redis_client"] is None

    @pytest.mark.asyncio
    async def test_forget_with_none_simple_redis(self):
        """Verify forget endpoint handles None simple_redis gracefully"""
        request = ForgetContextRequest(
            context_id="test-forget-id",
            reason="Test with None redis",
            retention_days=7
        )

        mock_api_key_info = APIKeyInfo(
            key_id="test_key",
            user_id="test_user",
            role="admin",
            capabilities=["forget"],
            is_agent=False,
            metadata={}
        )

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j, \
             patch("src.mcp_server.main.simple_redis", None), \
             patch("src.tools.delete_operations.forget_context") as mock_forget_func:

            mock_forget_func.return_value = {"success": True}

            result = await forget_context_endpoint(
                request=request,
                api_key_info=mock_api_key_info
            )

            # Verify None was passed correctly
            call_args = mock_forget_func.call_args
            assert call_args.kwargs["redis_client"] is None
