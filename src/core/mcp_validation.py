#!/usr/bin/env python3
"""
MCP contract validation middleware.

Provides runtime validation of requests and responses against MCP tool contracts
to ensure compliance with the defined schemas.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import jsonschema
from jsonschema import ValidationError


logger = logging.getLogger(__name__)


class MCPContractValidator:
    """Validator for MCP tool contracts."""
    
    def __init__(self, contracts_dir: str = "context/mcp_contracts"):
        """Initialize validator with contract directory.
        
        Args:
            contracts_dir: Path to directory containing MCP contract JSON files
        """
        self.contracts_dir = Path(contracts_dir)
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.load_contracts()
    
    def load_contracts(self) -> None:
        """Load all MCP contract files from the contracts directory."""
        if not self.contracts_dir.exists():
            logger.warning(f"MCP contracts directory not found: {self.contracts_dir}")
            return
        
        contract_files = list(self.contracts_dir.glob("*_tool.json"))
        logger.info(f"Loading {len(contract_files)} MCP contract files")
        
        for contract_file in contract_files:
            try:
                with open(contract_file, 'r') as f:
                    contract = json.load(f)
                    tool_name = contract.get("tool_name")
                    if tool_name:
                        self.contracts[tool_name] = contract
                        logger.debug(f"Loaded contract for tool: {tool_name}")
                    else:
                        logger.warning(f"Contract file missing tool_name: {contract_file}")
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                logger.error(f"Failed to load contract from {contract_file}: {e}")
    
    def get_contract(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get contract for a specific tool.
        
        Args:
            tool_name: Name of the MCP tool
            
        Returns:
            Contract dictionary or None if not found
        """
        return self.contracts.get(tool_name)
    
    def validate_request(self, tool_name: str, request_data: Dict[str, Any]) -> List[str]:
        """Validate request data against MCP contract input schema.
        
        Args:
            tool_name: Name of the MCP tool
            request_data: Request data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        contract = self.get_contract(tool_name)
        if not contract:
            return [f"No MCP contract found for tool: {tool_name}"]
        
        input_schema = contract.get("input_schema")
        if not input_schema:
            return [f"No input schema defined in contract for tool: {tool_name}"]
        
        try:
            jsonschema.validate(request_data, input_schema)
            return []  # No errors
        except ValidationError as e:
            return [f"Request validation error: {e.message}"]
        except Exception as e:
            return [f"Unexpected validation error: {str(e)}"]
    
    def validate_response(self, tool_name: str, response_data: Dict[str, Any]) -> List[str]:
        """Validate response data against MCP contract output schema.
        
        Args:
            tool_name: Name of the MCP tool
            response_data: Response data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        contract = self.get_contract(tool_name)
        if not contract:
            return [f"No MCP contract found for tool: {tool_name}"]
        
        output_schema = contract.get("output_schema")
        if not output_schema:
            return [f"No output schema defined in contract for tool: {tool_name}"]
        
        try:
            jsonschema.validate(response_data, output_schema)
            return []  # No errors
        except ValidationError as e:
            return [f"Response validation error: {e.message}"]
        except Exception as e:
            return [f"Unexpected validation error: {str(e)}"]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of loaded contracts and validation capabilities.
        
        Returns:
            Dictionary with validation summary information
        """
        return {
            "contracts_loaded": len(self.contracts),
            "available_tools": list(self.contracts.keys()),
            "contracts_directory": str(self.contracts_dir),
            "validation_enabled": len(self.contracts) > 0
        }


# Global validator instance
_mcp_validator: Optional[MCPContractValidator] = None


def get_mcp_validator() -> MCPContractValidator:
    """Get or create global MCP validator instance."""
    global _mcp_validator
    if _mcp_validator is None:
        _mcp_validator = MCPContractValidator()
    return _mcp_validator


def validate_mcp_request(tool_name: str, request_data: Dict[str, Any]) -> List[str]:
    """Convenience function to validate MCP request.
    
    Args:
        tool_name: Name of the MCP tool
        request_data: Request data to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    validator = get_mcp_validator()
    return validator.validate_request(tool_name, request_data)


def validate_mcp_response(tool_name: str, response_data: Dict[str, Any]) -> List[str]:
    """Convenience function to validate MCP response.
    
    Args:
        tool_name: Name of the MCP tool
        response_data: Response data to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    validator = get_mcp_validator()
    return validator.validate_response(tool_name, response_data)


def create_validation_middleware(validate_requests: bool = True, validate_responses: bool = False):
    """Create FastAPI middleware for MCP contract validation.
    
    Args:
        validate_requests: Whether to validate incoming requests
        validate_responses: Whether to validate outgoing responses
        
    Returns:
        FastAPI middleware function
    """
    async def mcp_validation_middleware(request, call_next):
        """FastAPI middleware for MCP contract validation."""
        # Extract tool name from request path
        path = str(request.url.path)
        tool_name = None
        
        # Map endpoints to tool names
        endpoint_mapping = {
            "/tools/store_context": "store_context",
            "/tools/retrieve_context": "retrieve_context", 
            "/tools/query_graph": "query_graph",
            "/tools/update_scratchpad": "update_scratchpad",
            "/tools/get_agent_state": "get_agent_state"
        }
        
        tool_name = endpoint_mapping.get(path)
        
        # Validate request if enabled and tool contract exists
        if validate_requests and tool_name and request.method == "POST":
            try:
                # Get request body
                body = await request.body()
                if body:
                    request_data = json.loads(body)
                    validation_errors = validate_mcp_request(tool_name, request_data)
                    
                    if validation_errors:
                        logger.warning(f"MCP request validation failed for {tool_name}: {validation_errors}")
                        # Could return validation error response here if desired
            except Exception as e:
                logger.error(f"Error during request validation for {tool_name}: {e}")
        
        # Process the request
        response = await call_next(request)
        
        # Validate response if enabled
        if validate_responses and tool_name:
            try:
                # Note: Response validation would require capturing response body
                # This is a simplified implementation
                pass
            except Exception as e:
                logger.error(f"Error during response validation for {tool_name}: {e}")
        
        return response
    
    return mcp_validation_middleware


# Test function to verify contract loading
def test_contract_loading():
    """Test function to verify MCP contract loading."""
    validator = get_mcp_validator()
    summary = validator.get_validation_summary()
    
    print("MCP Contract Validation Summary:")
    print(f"  Contracts loaded: {summary['contracts_loaded']}")
    print(f"  Available tools: {summary['available_tools']}")
    print(f"  Validation enabled: {summary['validation_enabled']}")
    
    # Test validation for each loaded contract
    for tool_name in summary['available_tools']:
        contract = validator.get_contract(tool_name)
        print(f"\nContract for {tool_name}:")
        print(f"  Version: {contract.get('version', 'unknown')}")
        print(f"  Description: {contract.get('description', 'no description')}")
        
        # Test with empty request to see schema requirements
        errors = validator.validate_request(tool_name, {})
        if errors:
            print(f"  Schema validation requirements: {len(errors)} errors for empty request")
        else:
            print(f"  Schema allows empty requests")


if __name__ == "__main__":
    test_contract_loading()