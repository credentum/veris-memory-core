#!/usr/bin/env python3
"""
Comprehensive tests for src/core/mcp_validation.py

Tests cover:
- MCPContractValidator class with contract loading and validation
- Request and response validation against JSON schemas
- Global validator functions and middleware creation
- Error handling, edge cases, and contract management
- File system interactions and contract parsing
- FastAPI middleware functionality
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
import pytest

from src.core.mcp_validation import (
    MCPContractValidator,
    get_mcp_validator,
    validate_mcp_request,
    validate_mcp_response,
    create_validation_middleware,
    test_contract_loading
)


class TestMCPContractValidator:
    """Test MCPContractValidator class."""

    def test_init_with_default_contracts_dir(self):
        """Test initialization with default contracts directory."""
        with patch('src.core.mcp_validation.Path.exists', return_value=False):
            validator = MCPContractValidator()
            
        assert validator.contracts_dir == Path("context/mcp_contracts")
        assert validator.contracts == {}

    def test_init_with_custom_contracts_dir(self):
        """Test initialization with custom contracts directory."""
        custom_dir = "/custom/path"
        with patch('src.core.mcp_validation.Path.exists', return_value=False):
            validator = MCPContractValidator(custom_dir)
            
        assert validator.contracts_dir == Path(custom_dir)
        assert validator.contracts == {}

    def test_load_contracts_directory_not_exists(self):
        """Test load_contracts when directory doesn't exist."""
        with patch('src.core.mcp_validation.Path.exists', return_value=False):
            with patch('src.core.mcp_validation.logger') as mock_logger:
                validator = MCPContractValidator()
                
                mock_logger.warning.assert_called_once()
                assert "not found" in mock_logger.warning.call_args[0][0]

    def test_load_contracts_empty_directory(self):
        """Test load_contracts with empty directory."""
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[]):
                with patch('src.core.mcp_validation.logger') as mock_logger:
                    validator = MCPContractValidator()
                    
                    mock_logger.info.assert_called_with("Loading 0 MCP contract files")

    def test_load_contracts_valid_contract(self):
        """Test loading valid contract file."""
        contract_data = {
            "tool_name": "store_context",
            "version": "1.0",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"}
        }
        
        mock_file = mock_open(read_data=json.dumps(contract_data))
        mock_path = MagicMock()
        mock_path.__str__ = MagicMock(return_value="store_context_tool.json")
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[mock_path]):
                with patch('builtins.open', mock_file):
                    validator = MCPContractValidator()
                    
        assert "store_context" in validator.contracts
        assert validator.contracts["store_context"]["tool_name"] == "store_context"

    def test_load_contracts_invalid_json(self):
        """Test loading contract with invalid JSON."""
        mock_file = mock_open(read_data="invalid json content")
        mock_path = MagicMock()
        mock_path.__str__ = MagicMock(return_value="invalid_tool.json")
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[mock_path]):
                with patch('builtins.open', mock_file):
                    with patch('src.core.mcp_validation.logger') as mock_logger:
                        validator = MCPContractValidator()
                        
                        mock_logger.error.assert_called_once()
                        assert "Failed to load contract" in mock_logger.error.call_args[0][0]

    def test_load_contracts_missing_tool_name(self):
        """Test loading contract missing tool_name field."""
        contract_data = {"version": "1.0", "input_schema": {"type": "object"}}
        
        mock_file = mock_open(read_data=json.dumps(contract_data))
        mock_path = MagicMock()
        mock_path.__str__ = MagicMock(return_value="no_name_tool.json")
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[mock_path]):
                with patch('builtins.open', mock_file):
                    with patch('src.core.mcp_validation.logger') as mock_logger:
                        validator = MCPContractValidator()
                        
                        mock_logger.warning.assert_called()
                        assert "missing tool_name" in mock_logger.warning.call_args[0][0]

    def test_load_contracts_file_not_found(self):
        """Test loading contract when file doesn't exist."""
        mock_path = MagicMock()
        mock_path.__str__ = MagicMock(return_value="missing_tool.json")
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[mock_path]):
                with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
                    with patch('src.core.mcp_validation.logger') as mock_logger:
                        validator = MCPContractValidator()
                        
                        mock_logger.error.assert_called_once()
                        assert "Failed to load contract" in mock_logger.error.call_args[0][0]

    def test_get_contract_existing(self):
        """Test get_contract for existing tool."""
        validator = MCPContractValidator()
        validator.contracts = {"test_tool": {"tool_name": "test_tool", "version": "1.0"}}
        
        contract = validator.get_contract("test_tool")
        
        assert contract is not None
        assert contract["tool_name"] == "test_tool"

    def test_get_contract_nonexistent(self):
        """Test get_contract for non-existent tool."""
        validator = MCPContractValidator()
        validator.contracts = {}
        
        contract = validator.get_contract("nonexistent_tool")
        
        assert contract is None

    def test_validate_request_no_contract(self):
        """Test validate_request when no contract exists."""
        validator = MCPContractValidator()
        validator.contracts = {}
        
        errors = validator.validate_request("unknown_tool", {})
        
        assert len(errors) == 1
        assert "No MCP contract found" in errors[0]

    def test_validate_request_no_input_schema(self):
        """Test validate_request when contract has no input schema."""
        validator = MCPContractValidator()
        validator.contracts = {"test_tool": {"tool_name": "test_tool"}}
        
        errors = validator.validate_request("test_tool", {})
        
        assert len(errors) == 1
        assert "No input schema defined" in errors[0]

    def test_validate_request_valid_data(self):
        """Test validate_request with valid data."""
        input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "input_schema": input_schema
            }
        }
        
        errors = validator.validate_request("test_tool", {"name": "test"})
        
        assert len(errors) == 0

    def test_validate_request_invalid_data(self):
        """Test validate_request with invalid data."""
        input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "input_schema": input_schema
            }
        }
        
        errors = validator.validate_request("test_tool", {})
        
        assert len(errors) == 1
        assert "Request validation error" in errors[0]

    def test_validate_request_schema_exception(self):
        """Test validate_request when schema validation raises unexpected exception."""
        invalid_schema = {"type": "invalid_type"}
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "input_schema": invalid_schema
            }
        }
        
        errors = validator.validate_request("test_tool", {})
        
        assert len(errors) == 1
        assert "Unexpected validation error" in errors[0]

    def test_validate_response_no_contract(self):
        """Test validate_response when no contract exists."""
        validator = MCPContractValidator()
        validator.contracts = {}
        
        errors = validator.validate_response("unknown_tool", {})
        
        assert len(errors) == 1
        assert "No MCP contract found" in errors[0]

    def test_validate_response_no_output_schema(self):
        """Test validate_response when contract has no output schema."""
        validator = MCPContractValidator()
        validator.contracts = {"test_tool": {"tool_name": "test_tool"}}
        
        errors = validator.validate_response("test_tool", {})
        
        assert len(errors) == 1
        assert "No output schema defined" in errors[0]

    def test_validate_response_valid_data(self):
        """Test validate_response with valid data."""
        output_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "output_schema": output_schema
            }
        }
        
        errors = validator.validate_response("test_tool", {"result": "success"})
        
        assert len(errors) == 0

    def test_validate_response_invalid_data(self):
        """Test validate_response with invalid data."""
        output_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "output_schema": output_schema
            }
        }
        
        errors = validator.validate_response("test_tool", {})
        
        assert len(errors) == 1
        assert "Response validation error" in errors[0]

    def test_validate_response_schema_exception(self):
        """Test validate_response when schema validation raises unexpected exception."""
        invalid_schema = {"type": "invalid_type"}
        
        validator = MCPContractValidator()
        validator.contracts = {
            "test_tool": {
                "tool_name": "test_tool",
                "output_schema": invalid_schema
            }
        }
        
        errors = validator.validate_response("test_tool", {})
        
        assert len(errors) == 1
        assert "Unexpected validation error" in errors[0]

    def test_get_validation_summary_empty(self):
        """Test get_validation_summary with no contracts."""
        validator = MCPContractValidator()
        validator.contracts = {}
        validator.contracts_dir = Path("/test/path")
        
        summary = validator.get_validation_summary()
        
        assert summary["contracts_loaded"] == 0
        assert summary["available_tools"] == []
        assert summary["contracts_directory"] == "/test/path"
        assert summary["validation_enabled"] is False

    def test_get_validation_summary_with_contracts(self):
        """Test get_validation_summary with loaded contracts."""
        validator = MCPContractValidator()
        validator.contracts = {
            "tool1": {"tool_name": "tool1"},
            "tool2": {"tool_name": "tool2"}
        }
        validator.contracts_dir = Path("/test/path")
        
        summary = validator.get_validation_summary()
        
        assert summary["contracts_loaded"] == 2
        assert set(summary["available_tools"]) == {"tool1", "tool2"}
        assert summary["contracts_directory"] == "/test/path"
        assert summary["validation_enabled"] is True


class TestGlobalFunctions:
    """Test global functions and singleton behavior."""

    def test_get_mcp_validator_singleton(self):
        """Test that get_mcp_validator returns singleton instance."""
        # Reset global instance
        import src.core.mcp_validation
        src.core.mcp_validation._mcp_validator = None
        
        with patch('src.core.mcp_validation.MCPContractValidator') as mock_validator_class:
            mock_instance = MagicMock()
            mock_validator_class.return_value = mock_instance
            
            validator1 = get_mcp_validator()
            validator2 = get_mcp_validator()
            
            assert validator1 is validator2
            mock_validator_class.assert_called_once()

    def test_validate_mcp_request_function(self):
        """Test validate_mcp_request convenience function."""
        with patch('src.core.mcp_validation.get_mcp_validator') as mock_get:
            mock_validator = MagicMock()
            mock_validator.validate_request.return_value = ["error1", "error2"]
            mock_get.return_value = mock_validator
            
            errors = validate_mcp_request("test_tool", {"data": "test"})
            
            mock_validator.validate_request.assert_called_once_with("test_tool", {"data": "test"})
            assert errors == ["error1", "error2"]

    def test_validate_mcp_response_function(self):
        """Test validate_mcp_response convenience function."""
        with patch('src.core.mcp_validation.get_mcp_validator') as mock_get:
            mock_validator = MagicMock()
            mock_validator.validate_response.return_value = []
            mock_get.return_value = mock_validator
            
            errors = validate_mcp_response("test_tool", {"result": "success"})
            
            mock_validator.validate_response.assert_called_once_with("test_tool", {"result": "success"})
            assert errors == []


class TestMiddlewareCreation:
    """Test middleware creation and functionality."""

    def test_create_validation_middleware_default_params(self):
        """Test creating middleware with default parameters."""
        middleware = create_validation_middleware()
        
        assert callable(middleware)

    def test_create_validation_middleware_custom_params(self):
        """Test creating middleware with custom parameters."""
        middleware = create_validation_middleware(validate_requests=False, validate_responses=True)
        
        assert callable(middleware)

    @pytest.mark.asyncio
    async def test_middleware_execution_no_tool_match(self):
        """Test middleware execution when path doesn't match any tool."""
        middleware = create_validation_middleware()
        
        mock_request = MagicMock()
        mock_request.url.path = "/unknown/path"
        mock_request.method = "POST"
        
        mock_call_next = AsyncMock()
        mock_response = MagicMock()
        mock_call_next.return_value = mock_response
        
        result = await middleware(mock_request, mock_call_next)
        
        assert result is mock_response
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_middleware_execution_get_request(self):
        """Test middleware execution with GET request (no validation)."""
        middleware = create_validation_middleware()
        
        mock_request = MagicMock()
        mock_request.url.path = "/tools/store_context"
        mock_request.method = "GET"
        
        mock_call_next = AsyncMock()
        mock_response = MagicMock()
        mock_call_next.return_value = mock_response
        
        result = await middleware(mock_request, mock_call_next)
        
        assert result is mock_response
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_middleware_execution_post_no_body(self):
        """Test middleware execution with POST request but no body."""
        middleware = create_validation_middleware()
        
        mock_request = MagicMock()
        mock_request.url.path = "/tools/store_context"
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"")
        
        mock_call_next = AsyncMock()
        mock_response = MagicMock()
        mock_call_next.return_value = mock_response
        
        result = await middleware(mock_request, mock_call_next)
        
        assert result is mock_response
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_middleware_execution_post_with_body(self):
        """Test middleware execution with POST request and body."""
        request_data = {"tool": "store_context", "data": "test"}
        
        with patch('src.core.mcp_validation.validate_mcp_request') as mock_validate:
            mock_validate.return_value = []
            
            middleware = create_validation_middleware()
            
            mock_request = MagicMock()
            mock_request.url.path = "/tools/store_context"
            mock_request.method = "POST"
            mock_request.body = AsyncMock(return_value=json.dumps(request_data).encode())
            
            mock_call_next = AsyncMock()
            mock_response = MagicMock()
            mock_call_next.return_value = mock_response
            
            result = await middleware(mock_request, mock_call_next)
            
            mock_validate.assert_called_once_with("store_context", request_data)
            assert result is mock_response

    @pytest.mark.asyncio
    async def test_middleware_execution_validation_errors(self):
        """Test middleware execution with validation errors."""
        request_data = {"invalid": "data"}
        
        with patch('src.core.mcp_validation.validate_mcp_request') as mock_validate:
            with patch('src.core.mcp_validation.logger') as mock_logger:
                mock_validate.return_value = ["Validation error 1", "Validation error 2"]
                
                middleware = create_validation_middleware()
                
                mock_request = MagicMock()
                mock_request.url.path = "/tools/retrieve_context"
                mock_request.method = "POST"
                mock_request.body = AsyncMock(return_value=json.dumps(request_data).encode())
                
                mock_call_next = AsyncMock()
                mock_response = MagicMock()
                mock_call_next.return_value = mock_response
                
                result = await middleware(mock_request, mock_call_next)
                
                mock_logger.warning.assert_called_once()
                assert "validation failed" in mock_logger.warning.call_args[0][0]
                assert result is mock_response

    @pytest.mark.asyncio
    async def test_middleware_execution_json_decode_error(self):
        """Test middleware execution when JSON decode fails."""
        with patch('src.core.mcp_validation.logger') as mock_logger:
            middleware = create_validation_middleware()
            
            mock_request = MagicMock()
            mock_request.url.path = "/tools/query_graph"
            mock_request.method = "POST"
            mock_request.body = AsyncMock(return_value=b"invalid json")
            
            mock_call_next = AsyncMock()
            mock_response = MagicMock()
            mock_call_next.return_value = mock_response
            
            result = await middleware(mock_request, mock_call_next)
            
            mock_logger.error.assert_called_once()
            assert "Error during request validation" in mock_logger.error.call_args[0][0]
            assert result is mock_response

    @pytest.mark.asyncio
    async def test_middleware_execution_response_validation(self):
        """Test middleware with response validation enabled."""
        middleware = create_validation_middleware(validate_responses=True)
        
        mock_request = MagicMock()
        mock_request.url.path = "/tools/update_scratchpad"
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b'{"data": "test"}')
        
        mock_call_next = AsyncMock()
        mock_response = MagicMock()
        mock_call_next.return_value = mock_response
        
        result = await middleware(mock_request, mock_call_next)
        
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_middleware_endpoint_mapping(self):
        """Test all endpoint mappings in middleware."""
        endpoints = [
            "/tools/store_context",
            "/tools/retrieve_context", 
            "/tools/query_graph",
            "/tools/update_scratchpad",
            "/tools/get_agent_state"
        ]
        
        expected_tools = [
            "store_context",
            "retrieve_context",
            "query_graph", 
            "update_scratchpad",
            "get_agent_state"
        ]
        
        for endpoint, expected_tool in zip(endpoints, expected_tools):
            with patch('src.core.mcp_validation.validate_mcp_request') as mock_validate:
                mock_validate.return_value = []
                
                middleware = create_validation_middleware()
                
                mock_request = MagicMock()
                mock_request.url.path = endpoint
                mock_request.method = "POST"
                mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
                
                mock_call_next = AsyncMock()
                mock_response = MagicMock()
                mock_call_next.return_value = mock_response
                
                await middleware(mock_request, mock_call_next)
                
                mock_validate.assert_called_once_with(expected_tool, {"test": "data"})


class TestContractLoadingFunction:
    """Test test_contract_loading function."""

    def test_contract_loading_function(self):
        """Test test_contract_loading function execution."""
        mock_contracts = {
            "tool1": {
                "version": "1.0",
                "description": "Test tool 1"
            },
            "tool2": {
                "version": "2.0", 
                "description": "Test tool 2"
            }
        }
        
        with patch('src.core.mcp_validation.get_mcp_validator') as mock_get:
            with patch('builtins.print') as mock_print:
                mock_validator = MagicMock()
                mock_validator.get_validation_summary.return_value = {
                    'contracts_loaded': 2,
                    'available_tools': ['tool1', 'tool2'],
                    'validation_enabled': True
                }
                mock_validator.get_contract.side_effect = lambda name: mock_contracts.get(name)
                mock_validator.validate_request.return_value = ["Schema error"]
                mock_get.return_value = mock_validator
                
                test_contract_loading()
                
                # Verify print statements were called
                assert mock_print.call_count > 0
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("MCP Contract Validation Summary:" in call for call in print_calls)

    def test_contract_loading_function_empty_request_valid(self):
        """Test test_contract_loading with tools that accept empty requests."""
        mock_contracts = {"flexible_tool": {"version": "1.0", "description": "Flexible tool"}}
        
        with patch('src.core.mcp_validation.get_mcp_validator') as mock_get:
            with patch('builtins.print') as mock_print:
                mock_validator = MagicMock()
                mock_validator.get_validation_summary.return_value = {
                    'contracts_loaded': 1,
                    'available_tools': ['flexible_tool'],
                    'validation_enabled': True
                }
                mock_validator.get_contract.return_value = mock_contracts["flexible_tool"]
                mock_validator.validate_request.return_value = []  # No errors
                mock_get.return_value = mock_validator
                
                test_contract_loading()
                
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("allows empty requests" in call for call in print_calls)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_validator_with_multiple_contract_files(self):
        """Test validator loading multiple contract files."""
        contracts = [
            {"tool_name": "tool1", "version": "1.0"},
            {"tool_name": "tool2", "version": "2.0"},
            {"tool_name": "tool3", "version": "3.0"}
        ]
        
        mock_files = [
            mock_open(read_data=json.dumps(contract)).return_value
            for contract in contracts
        ]
        
        mock_paths = [MagicMock() for _ in range(3)]
        for i, path in enumerate(mock_paths):
            path.__str__ = MagicMock(return_value=f"tool{i+1}_tool.json")
        
        def mock_open_side_effect(file_path, mode='r'):
            for i, path in enumerate(mock_paths):
                if str(file_path) == str(path):
                    return mock_files[i]
            raise FileNotFoundError()
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=mock_paths):
                with patch('builtins.open', side_effect=mock_open_side_effect):
                    validator = MCPContractValidator()
                    
        assert len(validator.contracts) == 3
        assert all(f"tool{i}" in validator.contracts for i in range(1, 4))

    def test_validator_with_complex_schema(self):
        """Test validator with complex JSON schema."""
        complex_schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "pattern": "^agent_[0-9a-f]+$"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    },
                    "required": ["agent_id"]
                },
                "data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["metadata", "data"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "complex_tool": {
                "tool_name": "complex_tool",
                "input_schema": complex_schema
            }
        }
        
        # Test valid complex data
        valid_data = {
            "metadata": {
                "agent_id": "agent_12345abcdef",
                "timestamp": "2023-01-01T00:00:00Z"
            },
            "data": ["item1", "item2"]
        }
        
        errors = validator.validate_request("complex_tool", valid_data)
        assert len(errors) == 0
        
        # Test invalid complex data
        invalid_data = {
            "metadata": {"agent_id": "invalid_id"},
            "data": []
        }
        
        errors = validator.validate_request("complex_tool", invalid_data)
        assert len(errors) == 1

    def test_validator_schema_with_references(self):
        """Test validator with schema containing references."""
        schema_with_refs = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": {
                "agent": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["id"]
                }
            },
            "type": "object",
            "properties": {
                "agent": {"$ref": "#/definitions/agent"},
                "action": {"type": "string"}
            },
            "required": ["agent", "action"]
        }
        
        validator = MCPContractValidator()
        validator.contracts = {
            "ref_tool": {
                "tool_name": "ref_tool",
                "input_schema": schema_with_refs
            }
        }
        
        # Test valid data with references
        valid_data = {
            "agent": {"id": "agent123", "name": "Test Agent"},
            "action": "execute"
        }
        
        errors = validator.validate_request("ref_tool", valid_data)
        assert len(errors) == 0

    def test_global_state_isolation(self):
        """Test that global validator state is properly isolated."""
        # Reset global state
        import src.core.mcp_validation
        src.core.mcp_validation._mcp_validator = None
        
        # Create first validator instance
        with patch('src.core.mcp_validation.Path.exists', return_value=False):
            validator1 = get_mcp_validator()
        
        # Modify first instance
        validator1.contracts["test"] = {"tool_name": "test"}
        
        # Get second validator instance (should be same singleton)
        validator2 = get_mcp_validator()
        
        assert validator1 is validator2
        assert "test" in validator2.contracts

    def test_large_contract_file_handling(self):
        """Test handling of large contract files."""
        large_schema = {
            "type": "object",
            "properties": {}
        }
        
        # Create a large schema with many properties
        for i in range(1000):
            large_schema["properties"][f"field_{i}"] = {
                "type": "string",
                "description": f"Field {i} description with lots of text " * 10
            }
        
        large_contract = {
            "tool_name": "large_tool",
            "version": "1.0",
            "description": "Large tool with many properties",
            "input_schema": large_schema,
            "output_schema": large_schema
        }
        
        mock_file = mock_open(read_data=json.dumps(large_contract))
        mock_path = MagicMock()
        mock_path.__str__ = MagicMock(return_value="large_tool.json")
        
        with patch('src.core.mcp_validation.Path.exists', return_value=True):
            with patch('src.core.mcp_validation.Path.glob', return_value=[mock_path]):
                with patch('builtins.open', mock_file):
                    validator = MCPContractValidator()
                    
        assert "large_tool" in validator.contracts
        assert len(validator.contracts["large_tool"]["input_schema"]["properties"]) == 1000

    def test_concurrent_validation_requests(self):
        """Test handling concurrent validation requests."""
        validator = MCPContractValidator()
        validator.contracts = {
            "concurrent_tool": {
                "tool_name": "concurrent_tool",
                "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}
            }
        }
        
        # Simulate concurrent validation requests
        test_data = [{"id": f"request_{i}"} for i in range(100)]
        
        results = []
        for data in test_data:
            errors = validator.validate_request("concurrent_tool", data)
            results.append(len(errors))
        
        # All validations should succeed
        assert all(error_count == 0 for error_count in results)