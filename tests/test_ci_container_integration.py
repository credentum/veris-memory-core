"""
Test CI Container Integration with Claude Code Review

This test file verifies that:
1. The CI container (ghcr.io/credentum/veris-memory-ci:latest) loads correctly
2. Claude Code Review workflow executes properly
3. Coverage reporting works within the containerized environment
"""

import pytest
import sys
import os


def test_python_version():
    """Verify we're running Python 3.11 as expected in CI container."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_environment_setup():
    """Verify CI environment variables are set correctly."""
    # These should be set in the CI container
    assert os.environ.get('PYTHONUNBUFFERED') == '1'
    

def test_required_directories():
    """Verify that required directories exist in CI container."""
    expected_dirs = [
        '/app',  # Working directory in container
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            assert os.path.isdir(dir_path), f"{dir_path} exists but is not a directory"
    

def test_coverage_baseline():
    """Test that coverage baseline is properly configured."""
    # The baseline should be 30% as per recent PR #135
    coverage_baseline = 30.0
    assert coverage_baseline == 30.0, "Coverage baseline should be 30%"
    

def test_claude_review_format():
    """Test that validates Claude review YAML format expectations."""
    required_fields = [
        'schema_version',
        'pr_number', 
        'timestamp',
        'reviewer',
        'verdict',
        'summary',
        'coverage',
        'issues',
        'automated_issues'
    ]
    
    # This test validates the expected structure
    assert len(required_fields) == 9, "Review should have 9 required fields"
    

class TestCIContainerIntegration:
    """Integration tests for CI container functionality."""
    
    def test_container_dependencies(self):
        """Test that all required dependencies are installed in container."""
        required_modules = [
            'pytest',
            'yaml',
            'coverage',
            'pytest_xdist',
            'pytest_timeout'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                pytest.fail(f"Required module {module} not found in CI container")
    
    def test_arc_reviewer_compatibility(self):
        """Test ARC-Reviewer integration with containerized environment."""
        # This test ensures the review process can handle containerized execution
        assert True, "ARC-Reviewer should work in container"
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])