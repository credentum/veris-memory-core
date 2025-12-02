"""
Test suite for MCP contract validation.

This test validates all MCP tool contracts in context/mcp_contracts/ to ensure
they conform to the MCP protocol specification and maintain consistency.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest


# Add project root to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMCPContractValidation:
    """Test MCP contract JSON schemas for validity and completeness."""

    @pytest.fixture
    def contracts_dir(self) -> Path:
        """Return path to MCP contracts directory."""
        return Path(__file__).parent.parent / "context" / "mcp_contracts"

    @pytest.fixture
    def all_contracts(self, contracts_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load all MCP contract JSON files."""
        contracts = {}
        for json_file in contracts_dir.glob("*.json"):
            with open(json_file, "r") as f:
                contracts[json_file.name] = json.load(f)
        return contracts

    def test_contracts_directory_exists(self, contracts_dir: Path):
        """Test that contracts directory exists and contains JSON files."""
        assert contracts_dir.exists(), f"Contracts directory not found: {contracts_dir}"
        json_files = list(contracts_dir.glob("*.json"))
        assert len(json_files) > 0, "No JSON contract files found"

    def test_all_contracts_valid_json(self, contracts_dir: Path):
        """Test that all contract files contain valid JSON."""
        for json_file in contracts_dir.glob("*.json"):
            with open(json_file, "r") as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {json_file.name}: {e}")

    def test_contracts_have_required_fields(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that all contracts have required MCP fields."""
        for contract_name, contract in all_contracts.items():
            # Check for description
            assert "description" in contract, (
                f"{contract_name} missing required field: description"
            )

            # Check for name (either 'name' or 'tool_name')
            assert "name" in contract or "tool_name" in contract, (
                f"{contract_name} missing name field (should have 'name' or 'tool_name')"
            )

            # Check for inputSchema (either 'inputSchema' or 'input_schema')
            assert "inputSchema" in contract or "input_schema" in contract, (
                f"{contract_name} missing input schema (should have 'inputSchema' or 'input_schema')"
            )

    def test_store_context_tool_contract(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test store_context_tool.json contract specifically."""
        contract = all_contracts.get("store_context_tool.json")
        assert contract is not None, "store_context_tool.json not found"

        # Validate name
        assert contract["name"] == "store_context", (
            f"Expected name 'store_context', got '{contract['name']}'"
        )

        # Validate inputSchema has required properties
        input_schema = contract.get("inputSchema", {})
        required_input = ["type", "content"]  # author and author_type may be in metadata

        properties = input_schema.get("properties", {})
        for field in required_input:
            assert field in properties, (
                f"Missing required input field: {field}"
            )

        # Validate outputSchema (added in PR #170)
        assert "outputSchema" in contract, "Missing outputSchema"
        output_schema = contract["outputSchema"]

        # Check for fields added in backend restoration
        output_props = output_schema.get("properties", {})
        assert "embedding_status" in output_props, (
            "Missing embedding_status in outputSchema (Phase 1 fix)"
        )
        assert "embedding_message" in output_props, (
            "Missing embedding_message in outputSchema (Phase 1 fix)"
        )
        assert "relationships_created" in output_props, (
            "Missing relationships_created in outputSchema (Phase 3 fix)"
        )

        # Validate embedding_status enum values
        embedding_status = output_props["embedding_status"]
        assert "enum" in embedding_status, "embedding_status missing enum"
        expected_statuses = ["completed", "failed", "unavailable"]
        assert set(embedding_status["enum"]) == set(expected_statuses), (
            f"Expected embedding_status enum {expected_statuses}, "
            f"got {embedding_status['enum']}"
        )

        # Validate relationships_created is integer with minimum 0
        relationships_created = output_props["relationships_created"]
        assert relationships_created["type"] == "integer", (
            "relationships_created should be integer"
        )
        assert relationships_created.get("minimum") == 0, (
            "relationships_created should have minimum: 0"
        )

    def test_all_contracts_have_descriptions(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that all contracts have non-empty descriptions."""
        for contract_name, contract in all_contracts.items():
            description = contract.get("description", "")
            assert len(description) > 0, (
                f"{contract_name} has empty description"
            )
            assert len(description) > 10, (
                f"{contract_name} description too short: '{description}'"
            )

    def test_input_schemas_have_type_and_properties(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that all inputSchemas have type and properties."""
        for contract_name, contract in all_contracts.items():
            # Handle both naming conventions
            input_schema = contract.get("inputSchema") or contract.get("input_schema", {})
            assert "type" in input_schema, (
                f"{contract_name} inputSchema missing 'type'"
            )
            assert input_schema["type"] == "object", (
                f"{contract_name} inputSchema type should be 'object'"
            )
            assert "properties" in input_schema, (
                f"{contract_name} inputSchema missing 'properties'"
            )

    def test_required_fields_exist_in_properties(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that all required fields are defined in properties."""
        for contract_name, contract in all_contracts.items():
            # Handle both naming conventions
            input_schema = contract.get("inputSchema") or contract.get("input_schema", {})
            required = input_schema.get("required", [])
            properties = input_schema.get("properties", {})

            for field in required:
                assert field in properties, (
                    f"{contract_name}: required field '{field}' not in properties"
                )

    def test_property_types_are_valid(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that all property types are valid JSON Schema types."""
        valid_types = ["string", "number", "integer", "boolean", "object", "array", "null"]

        for contract_name, contract in all_contracts.items():
            # Handle both naming conventions
            input_schema = contract.get("inputSchema") or contract.get("input_schema", {})
            properties = input_schema.get("properties", {})

            for prop_name, prop_schema in properties.items():
                if "type" in prop_schema:
                    prop_type = prop_schema["type"]
                    # Handle both single type and array of types
                    types = [prop_type] if isinstance(prop_type, str) else prop_type
                    for t in types:
                        assert t in valid_types, (
                            f"{contract_name}.{prop_name} has invalid type: {t}"
                        )

    def test_enum_fields_have_valid_values(self, all_contracts: Dict[str, Dict[str, Any]]):
        """Test that enum fields have non-empty value lists."""
        for contract_name, contract in all_contracts.items():
            # Handle both naming conventions
            input_schema = contract.get("inputSchema") or contract.get("input_schema", {})
            properties = input_schema.get("properties", {})

            for prop_name, prop_schema in properties.items():
                if "enum" in prop_schema:
                    enum_values = prop_schema["enum"]
                    assert isinstance(enum_values, list), (
                        f"{contract_name}.{prop_name} enum must be a list"
                    )
                    assert len(enum_values) > 0, (
                        f"{contract_name}.{prop_name} enum cannot be empty"
                    )

    def test_contracts_are_consistently_formatted(self, contracts_dir: Path):
        """Test that all contracts use consistent JSON formatting."""
        for json_file in contracts_dir.glob("*.json"):
            with open(json_file, "r") as f:
                content = f.read()
                # Check for consistent indentation (2 or 4 spaces)
                lines = content.split("\n")
                indented_lines = [line for line in lines if line.startswith(" ")]
                if indented_lines:
                    # Check first indented line
                    first_indent = len(indented_lines[0]) - len(indented_lines[0].lstrip())
                    assert first_indent in [2, 4], (
                        f"{json_file.name} uses inconsistent indentation: {first_indent} spaces"
                    )


class TestMCPContractIntegration:
    """Test that MCP contracts match actual implementation."""

    def test_store_context_response_matches_contract(self):
        """Test that store_context implementation matches outputSchema."""
        # This would be an integration test that calls the actual endpoint
        # and validates the response against the contract
        # Placeholder for now - requires actual MCP server running
        pass

    def test_retrieve_context_response_matches_contract(self):
        """Test that retrieve_context implementation matches outputSchema."""
        # Placeholder for integration test
        pass


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
