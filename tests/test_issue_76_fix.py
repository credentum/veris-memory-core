#!/usr/bin/env python3
"""
Test file specifically for verifying Issue #76 fix.

This test verifies that the store_context tool returns both 'id' and 'context_id'
fields correctly, addressing the MCP response formatting bug.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.mcp_server.server import store_context_tool
from src.core.response_validator import validate_mcp_response


class TestIssue76Fix:
    """Test suite specifically for Issue #76 fix verification."""

    @pytest.mark.asyncio
    async def test_store_context_response_fields_issue_76(self):
        """Test that store_context returns both id and context_id fields (Issue #76)."""
        arguments = {
            "content": {"title": "Issue 76 Test", "description": "Testing the fix"},
            "type": "trace",
            "metadata": {"issue": "76", "test": "response_fields"}
        }

        # Mock all dependencies to ensure we get a response
        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
            patch("src.mcp_server.server.input_validator") as mock_validator,
            patch("src.mcp_server.server.qa_generator") as mock_qa_gen,
            patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit,
        ):
            # Setup rate limiter
            mock_rate_limit.return_value = (True, None)
            
            # Setup input validator  
            mock_validation_result = Mock()
            mock_validation_result.valid = True
            mock_validator.validate_input.return_value = mock_validation_result
            mock_validator.validate_json_input.return_value = mock_validation_result
            
            # Setup Q&A generator
            mock_qa_gen.generate_qa_pairs_from_statement.return_value = []
            
            # Setup Qdrant mock - make it succeed
            mock_qdrant.client.upsert = AsyncMock()
            mock_qdrant.client.retrieve = Mock(return_value=[Mock()])  # Simulate successful retrieval
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}
            mock_qdrant.collection_name = "test_collection"
            mock_qdrant.store_vector = Mock()
            
            # Setup Neo4j mock - make it succeed
            mock_session = AsyncMock()
            mock_record = Mock()
            mock_record.__getitem__ = Mock(return_value="neo4j_node_id")
            mock_result = Mock()
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
            mock_neo4j.database = "neo4j"

            # Execute the function
            result = await store_context_tool(arguments)

            # Test the specific Issue #76 fix: both id and context_id should be present
            assert "id" in result, "Response missing 'id' field (Issue #76)"
            assert "context_id" in result, "Response missing 'context_id' field (Issue #76)"
            
            # Both fields should have the same non-null value when successful
            if result.get("success"):
                assert result["id"] is not None, "Field 'id' should not be null when success=True (Issue #76)"
                assert result["context_id"] is not None, "Field 'context_id' should not be null when success=True (Issue #76)"
                assert result["id"] == result["context_id"], "Fields 'id' and 'context_id' should match (Issue #76)"
                assert result["id"].startswith("ctx_"), "Generated ID should have proper prefix"

    @pytest.mark.asyncio  
    async def test_response_validation_issue_76(self):
        """Test that our response validation catches the Issue #76 problem."""
        # Test a response that would fail Issue #76 validation
        bad_response = {
            "success": True,
            "context_id": "ctx_12345",
            # Missing 'id' field - this is the Issue #76 problem
            "message": "Context stored successfully"
        }
        
        validation_result = validate_mcp_response("store_context", bad_response, log_results=False)
        assert not validation_result.valid, "Validation should fail for response missing 'id' field"
        
        # Check that the validation error mentions the missing id field
        error_messages = [error.message for error in validation_result.errors]
        assert any("id" in msg and "missing" in msg for msg in error_messages), "Should detect missing 'id' field"

        # Test a response that passes Issue #76 validation
        good_response = {
            "success": True,
            "id": "ctx_12345",
            "context_id": "ctx_12345",
            "message": "Context stored successfully"
        }
        
        validation_result = validate_mcp_response("store_context", good_response, log_results=False)
        assert validation_result.valid, f"Validation should pass for correct response: {[e.message for e in validation_result.errors]}"

    @pytest.mark.asyncio
    async def test_field_consistency_validation_issue_76(self):
        """Test validation of field consistency for Issue #76."""
        # Test response where id and context_id don't match
        inconsistent_response = {
            "success": True,
            "id": "ctx_12345",
            "context_id": "ctx_67890",  # Different value - should fail
            "message": "Context stored successfully"
        }
        
        validation_result = validate_mcp_response("store_context", inconsistent_response, log_results=False)
        assert not validation_result.valid, "Validation should fail when id and context_id don't match"
        
        # Check for the specific consistency error
        error_messages = [error.message for error in validation_result.errors]
        assert any("same value" in msg for msg in error_messages), "Should detect mismatched id/context_id"

    def test_contract_schema_compliance_issue_76(self):
        """Test that our response structure matches expected MCP contracts."""
        from src.core.response_validator import response_validator
        
        schema = response_validator.schemas.get("store_context")
        assert schema is not None, "store_context schema should be defined"
        
        # Verify that 'id' is a required field (fixing Issue #76)
        required_fields = schema.get("required_fields", set())
        assert "id" in required_fields, "Field 'id' should be required in store_context schema"
        
        # Verify that 'context_id' is optional (for backward compatibility)
        optional_fields = schema.get("optional_fields", set())
        assert "context_id" in optional_fields, "Field 'context_id' should be optional for backward compatibility"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])