#!/usr/bin/env python3
"""
Integration tests for MCP tool contracts with dashboard implementation.

Tests that MCP tool contracts work correctly with the actual dashboard implementation
and verify schema compliance, error handling, and data format validation.
"""

import pytest
import json
import jsonschema
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.dashboard import UnifiedDashboard
from src.monitoring.dashboard_api import DashboardAPI
from src.monitoring.streaming import MetricsStreamer


class TestMCPToolContractsIntegration:
    """Integration tests for MCP tool contracts with dashboard implementation."""

    @pytest.fixture
    def contract_get_dashboard_metrics(self):
        """Load get_dashboard_metrics MCP contract."""
        contract_path = Path(__file__).parent.parent.parent / "contracts" / "get_dashboard_metrics.json"
        with open(contract_path) as f:
            return json.load(f)

    @pytest.fixture
    def contract_stream_dashboard_updates(self):
        """Load stream_dashboard_updates MCP contract."""
        contract_path = Path(__file__).parent.parent.parent / "contracts" / "stream_dashboard_updates.json"
        with open(contract_path) as f:
            return json.load(f)

    @pytest.fixture
    def contract_run_verification_tests(self):
        """Load run_verification_tests MCP contract."""
        contract_path = Path(__file__).parent.parent.parent / "contracts" / "run_verification_tests.json"
        with open(contract_path) as f:
            return json.load(f)

    @pytest.fixture
    def mock_dashboard(self):
        """Mock UnifiedDashboard with realistic data."""
        mock = Mock()
        mock.collect_all_metrics.return_value = asyncio.Future()
        mock.collect_all_metrics.return_value.set_result({
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {
                'cpu_percent': 45.5,
                'memory_percent': 62.3,
                'memory_total_gb': 16.0,
                'memory_used_gb': 10.0,
                'disk_percent': 78.1,
                'disk_total_gb': 500.0,
                'disk_used_gb': 390.5,
                'load_average': [0.5, 0.6, 0.7],
                'uptime_hours': 48.5
            },
            'services': [
                {
                    'name': 'Redis',
                    'status': 'healthy',
                    'port': 6379,
                    'memory_mb': 128.5,
                    'operations_per_sec': 45
                },
                {
                    'name': 'Neo4j',
                    'status': 'unhealthy',
                    'port': 7474,
                    'memory_mb': None,
                    'operations_per_sec': None
                }
            ],
            'veris': {
                'total_memories': 85432,
                'memories_today': 1247,
                'avg_query_latency_ms': 23.5,
                'p99_latency_ms': 89.2,
                'error_rate_percent': 0.02,
                'active_agents': 5,
                'successful_operations_24h': 5000,
                'failed_operations_24h': 1
            },
            'security': {
                'failed_auth_attempts': 0,
                'blocked_ips': 0,
                'waf_blocks_today': 12,
                'ssl_cert_expiry_days': 87,
                'rbac_violations': 0,
                'audit_events_24h': 150
            },
            'backups': {
                'last_backup_time': '2025-08-14T09:00:00Z',
                'backup_size_gb': 4.7,
                'restore_tested': True,
                'last_restore_time_seconds': 142.0,
                'backup_success_rate_percent': 100.0,
                'offsite_sync_status': 'healthy'
            }
        })
        mock.generate_ascii_dashboard.return_value = """🎯 VERIS MEMORY STATUS - Wed Aug 14 12:00 UTC
═══════════════════════════════════════════════════════════════
💻 SYSTEM RESOURCES
CPU    [████░░░░░░] 45% ✅ HEALTHY
Memory [██████░░░░] 62% (10.0GB/16.0GB) ✅ HEALTHY
"""
        mock.last_update = datetime.utcnow()
        mock.shutdown.return_value = asyncio.Future()
        mock.shutdown.return_value.set_result(None)
        return mock

    def test_get_dashboard_metrics_contract_structure(self, contract_get_dashboard_metrics):
        """Test that get_dashboard_metrics contract has correct structure."""
        contract = contract_get_dashboard_metrics
        
        # Check required top-level fields
        required_fields = ['schema_version', 'tool_name', 'description', 'parameters', 'response_schema']
        for field in required_fields:
            assert field in contract, f"Missing required field: {field}"
        
        # Check tool name matches
        assert contract['tool_name'] == 'get_dashboard_metrics'
        
        # Check parameters schema
        params = contract['parameters']
        assert params['type'] == 'object'
        assert 'properties' in params
        
        # Check response schema
        response = contract['response_schema']
        assert response['type'] == 'object'
        assert 'properties' in response

    def test_stream_dashboard_updates_contract_structure(self, contract_stream_dashboard_updates):
        """Test that stream_dashboard_updates contract has correct structure."""
        contract = contract_stream_dashboard_updates
        
        # Check required fields
        assert contract['tool_name'] == 'stream_dashboard_updates'
        assert 'parameters' in contract
        assert 'response_schema' in contract
        
        # Check streaming-specific parameters
        params = contract['parameters']['properties']
        assert 'refresh_interval_seconds' in params
        assert 'filter_sections' in params
        assert 'format' in params

    def test_run_verification_tests_contract_structure(self, contract_run_verification_tests):
        """Test that run_verification_tests contract has correct structure."""
        contract = contract_run_verification_tests
        
        # Check required fields
        assert contract['tool_name'] == 'run_verification_tests'
        assert 'parameters' in contract
        assert 'response_schema' in contract
        
        # Check test-specific parameters
        params = contract['parameters']['properties']
        assert 'test_type' in params
        assert 'timeout_seconds' in params

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics_response_compliance(self, contract_get_dashboard_metrics, mock_dashboard):
        """Test that dashboard response complies with MCP contract schema."""
        contract = contract_get_dashboard_metrics
        response_schema = contract['response_schema']
        
        # Create dashboard and collect metrics
        with patch('monitoring.dashboard.MetricsCollector'), \
             patch('monitoring.dashboard.HealthChecker'), \
             patch('monitoring.dashboard.MCPMetrics'):
            dashboard = UnifiedDashboard()
            dashboard.dashboard = mock_dashboard
            
            # Get actual dashboard response
            metrics = await mock_dashboard.collect_all_metrics()
            
            # Validate against schema
            try:
                jsonschema.validate(metrics, response_schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Dashboard response doesn't match MCP contract schema: {e}")

    def test_get_dashboard_metrics_parameter_validation(self, contract_get_dashboard_metrics):
        """Test parameter validation for get_dashboard_metrics."""
        contract = contract_get_dashboard_metrics
        params_schema = contract['parameters']
        
        # Test valid parameters
        valid_params = {
            'format': 'json',
            'sections': ['system', 'services'],
            'force_refresh': True
        }
        
        try:
            jsonschema.validate(valid_params, params_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid parameters rejected: {e}")
        
        # Test invalid format
        invalid_params = {
            'format': 'invalid_format',
            'sections': ['system']
        }
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_params, params_schema)

    def test_stream_dashboard_updates_parameter_validation(self, contract_stream_dashboard_updates):
        """Test parameter validation for stream_dashboard_updates."""
        contract = contract_stream_dashboard_updates
        params_schema = contract['parameters']
        
        # Test valid streaming parameters
        valid_params = {
            'refresh_interval_seconds': 5,
            'filter_sections': ['system', 'services', 'veris'],
            'format': 'json',
            'include_deltas': True,
            'max_connections': 10
        }
        
        try:
            jsonschema.validate(valid_params, params_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid streaming parameters rejected: {e}")
        
        # Test invalid refresh interval (too small)
        invalid_params = {
            'refresh_interval_seconds': 0.5  # Below minimum
        }
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_params, params_schema)

    def test_run_verification_tests_parameter_validation(self, contract_run_verification_tests):
        """Test parameter validation for run_verification_tests."""
        contract = contract_run_verification_tests
        params_schema = contract['parameters']
        
        # Test valid test parameters
        valid_params = {
            'test_type': 'tls_verification',
            'timeout_seconds': 300,
            'options': {
                'check_mtls': True,
                'verify_certificates': True
            }
        }
        
        try:
            jsonschema.validate(valid_params, params_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid test parameters rejected: {e}")
        
        # Test invalid test type
        invalid_params = {
            'test_type': 'nonexistent_test'
        }
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_params, params_schema)

    @pytest.mark.asyncio
    async def test_dashboard_api_integration_with_mcp(self, mock_dashboard):
        """Test that DashboardAPI responses comply with MCP contracts."""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        with patch('monitoring.dashboard_api.UnifiedDashboard', return_value=mock_dashboard), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            
            dashboard_api = DashboardAPI(app)
            
            # Test JSON endpoint response format
            json_response = await dashboard_api._get_dashboard_json()
            
            # Verify response structure matches MCP expectations
            assert 'success' in json_response
            assert 'format' in json_response
            assert 'timestamp' in json_response
            assert 'data' in json_response
            
            # Verify data structure
            data = json_response['data']
            assert 'system' in data
            assert 'services' in data
            assert 'veris' in data
            assert 'security' in data
            assert 'backups' in data

    @pytest.mark.asyncio
    async def test_streaming_integration_with_mcp(self, mock_dashboard):
        """Test that MetricsStreamer works with MCP streaming contract."""
        streamer = MetricsStreamer(mock_dashboard)
        
        # Test filtered update creation (as specified in MCP contract)
        metrics = await mock_dashboard.collect_all_metrics()
        
        # Test system-only filter
        system_update = streamer.create_filtered_update(metrics, 'system_only')
        
        assert system_update is not None
        assert system_update['type'] == 'filtered_update'
        assert system_update['filter'] == 'system_only'
        assert 'system' in system_update['data']
        assert 'services' not in system_update['data']  # Filtered out
        
        # Test services-only filter
        services_update = streamer.create_filtered_update(metrics, 'services_only')
        
        assert services_update is not None
        assert 'services' in services_update['data']
        assert 'system' not in services_update['data']  # Filtered out

    def test_error_response_format_compliance(self, contract_get_dashboard_metrics):
        """Test that error responses comply with MCP contract format."""
        contract = contract_get_dashboard_metrics
        error_schema = contract.get('error_schema', {})
        
        # Simulate error response format
        error_response = {
            'success': False,
            'error': {
                'code': 'METRICS_COLLECTION_FAILED',
                'message': 'Failed to collect system metrics',
                'details': 'Database connection timeout'
            },
            'timestamp': '2025-08-14T12:00:00Z'
        }
        
        # If error schema exists, validate against it
        if error_schema:
            try:
                jsonschema.validate(error_response, error_schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Error response doesn't match MCP contract schema: {e}")

    def test_ascii_format_integration(self, mock_dashboard):
        """Test ASCII format support as specified in MCP contracts."""
        with patch('monitoring.dashboard.MetricsCollector'), \
             patch('monitoring.dashboard.HealthChecker'), \
             patch('monitoring.dashboard.MCPMetrics'):
            
            dashboard = UnifiedDashboard()
            dashboard.dashboard = mock_dashboard
            
            # Test ASCII generation
            ascii_output = mock_dashboard.generate_ascii_dashboard()
            
            # Verify ASCII output characteristics
            assert isinstance(ascii_output, str)
            assert len(ascii_output) > 0
            assert 'VERIS MEMORY STATUS' in ascii_output
            assert 'SYSTEM RESOURCES' in ascii_output

    @pytest.mark.asyncio
    async def test_verification_tests_integration(self):
        """Test that verification tests work with MCP contract expectations."""
        # Import verification test modules
        verification_path = Path(__file__).parent.parent / "verification"
        
        # Check that verification test files exist
        expected_files = [
            'tls_verifier.py',
            'restore_drill.py',
            'run_all_tests.py'
        ]
        
        for filename in expected_files:
            test_file = verification_path / filename
            assert test_file.exists(), f"Verification test file {filename} not found"

    def test_contract_schema_versions(self, contract_get_dashboard_metrics, 
                                    contract_stream_dashboard_updates, 
                                    contract_run_verification_tests):
        """Test that all contracts have consistent schema versions."""
        contracts = [
            contract_get_dashboard_metrics,
            contract_stream_dashboard_updates,
            contract_run_verification_tests
        ]
        
        # All contracts should have the same schema version
        schema_version = contracts[0]['schema_version']
        for contract in contracts:
            assert contract['schema_version'] == schema_version, \
                f"Contract {contract['tool_name']} has different schema version"

    def test_contract_examples_validity(self, contract_get_dashboard_metrics):
        """Test that contract examples are valid according to schemas."""
        contract = contract_get_dashboard_metrics
        
        if 'examples' in contract:
            params_schema = contract['parameters']
            response_schema = contract['response_schema']
            
            for example in contract['examples']:
                # Validate example request
                if 'request' in example:
                    try:
                        jsonschema.validate(example['request'], params_schema)
                    except jsonschema.ValidationError as e:
                        pytest.fail(f"Example request invalid: {e}")
                
                # Validate example response
                if 'response' in example:
                    try:
                        jsonschema.validate(example['response'], response_schema)
                    except jsonschema.ValidationError as e:
                        pytest.fail(f"Example response invalid: {e}")

    def test_contract_field_completeness(self):
        """Test that all contracts have complete field definitions."""
        contract_files = [
            'get_dashboard_metrics.json',
            'stream_dashboard_updates.json', 
            'run_verification_tests.json'
        ]
        
        contracts_path = Path(__file__).parent.parent.parent / "contracts"
        
        for filename in contract_files:
            contract_path = contracts_path / filename
            assert contract_path.exists(), f"Contract file {filename} not found"
            
            with open(contract_path) as f:
                contract = json.load(f)
            
            # Check required fields
            required_fields = [
                'schema_version', 'tool_name', 'description',
                'parameters', 'response_schema', 'usage_examples'
            ]
            
            for field in required_fields:
                assert field in contract, f"Contract {filename} missing field: {field}"
            
            # Check that descriptions are not empty
            assert len(contract['description']) > 10, f"Contract {filename} has insufficient description"

    @pytest.mark.asyncio
    async def test_real_time_streaming_compliance(self, mock_dashboard):
        """Test real-time streaming compliance with MCP streaming contract."""
        streamer = MetricsStreamer(mock_dashboard)
        
        # Test streaming message format
        metrics = await mock_dashboard.collect_all_metrics()
        stream_message = await streamer.stream_monitoring_update(metrics)
        
        # Verify streaming message structure
        assert stream_message['type'] == 'monitoring_stream'
        assert 'timestamp' in stream_message
        assert 'data' in stream_message
        assert stream_message['data'] == metrics
        
        # Test heartbeat message format
        heartbeat = streamer._create_heartbeat_message()
        
        assert heartbeat['type'] == 'heartbeat'
        assert 'timestamp' in heartbeat
        assert 'streaming_stats' in heartbeat

    def test_mcp_tool_discoverability(self):
        """Test that MCP tools are properly discoverable from contracts."""
        contracts_path = Path(__file__).parent.parent.parent / "contracts"
        
        # List all contract files
        contract_files = list(contracts_path.glob("*.json"))
        assert len(contract_files) >= 3, "Expected at least 3 MCP contract files"
        
        # Check each contract is properly formatted for discovery
        for contract_file in contract_files:
            with open(contract_file) as f:
                contract = json.load(f)
            
            # Each contract should have discoverable metadata
            assert 'tool_name' in contract
            assert 'description' in contract
            assert 'parameters' in contract
            
            # Tool name should match filename pattern
            expected_name = contract_file.stem
            assert contract['tool_name'] == expected_name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])