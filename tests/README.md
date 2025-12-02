# Veris Memory Test Suite

Comprehensive testing infrastructure for the Veris Memory context storage and retrieval platform with **parallel execution support** for 6-8x speed improvements.

## 🚀 Quick Start - Parallel Testing

### Fastest Development Workflow
```bash
# Fast unit tests (30 seconds with high parallelization)
./scripts/run-parallel-tests.sh fast

# Security module tests (1 minute with moderate parallelization)
./scripts/run-parallel-tests.sh security

# Complete test suite (2-3 minutes with smart strategy)
./scripts/run-parallel-tests.sh all
```

### Performance Comparison
- **Sequential**: 10+ minutes for full test suite
- **Parallel**: 1-2 minutes for full test suite (**6-8x faster**)
- **Fast Tests**: 30 seconds for unit tests only

## 📁 Test Directory Structure

```
tests/
├── core/                    # Core utilities and configuration (14 files)
│   ├── test_utils.py        # 100% coverage - utility functions
│   ├── test_config.py       # Configuration validation
│   ├── test_monitoring.py   # System monitoring
│   └── ...
├── security/                # Security infrastructure (13 files)
│   ├── test_compliance_reporter_comprehensive.py  # 27.0% coverage (155/456)
│   ├── test_security_scanner_comprehensive.py     # 22.6% coverage (118/400)
│   └── ...
├── storage/                 # Storage backends (18 files)
│   ├── test_neo4j_client_comprehensive.py     # Neo4j graph database
│   ├── test_qdrant_client_comprehensive.py    # Vector database
│   ├── test_kv_store_comprehensive.py         # Key-value storage
│   └── ...
├── mcp_server/              # MCP protocol server (14 files)
│   ├── test_main_comprehensive.py             # FastAPI application
│   ├── test_server_working.py                 # MCP SDK implementation
│   └── ...
├── validators/              # Input validation (7 files)
├── monitoring/              # Observability (6 files)
├── integration/             # End-to-end tests (2 files)
└── README.md               # This file
```

## ⚡ Parallel Execution Framework

### Installation & Setup
```bash
# Install parallel testing dependencies
pip install -r requirements-dev-parallel.txt

# Validate parallel execution works
./scripts/run-parallel-tests.sh validate
```

### Execution Modes

#### 1. Fast Development (High Parallelization)
```bash
./scripts/run-parallel-tests.sh fast
# - Runs unit tests only
# - Uses all available CPU cores
# - ~30 seconds execution time
# - Perfect for rapid development cycles
```

#### 2. Security Testing (Moderate Parallelization)  
```bash
./scripts/run-parallel-tests.sh security
# - Runs security infrastructure tests
# - Uses 6 workers
# - ~1 minute execution time
# - Comprehensive security validation
```

#### 3. Complete Suite (Smart Strategy)
```bash
./scripts/run-parallel-tests.sh all
# - Phase 1: Fast unit tests (high parallelization)
# - Phase 2: Integration tests (moderate parallelization)  
# - Phase 3: Slow tests (limited parallelization)
# - ~2-3 minutes total execution time
```

#### 4. Custom Execution
```bash
# Custom test pattern with specific worker count
./scripts/run-parallel-tests.sh custom tests/core/ 4

# Direct pytest with parallel execution
python3 -m pytest tests/security/ -n 4 -v

# Specific test file with coverage
python3 -m pytest tests/core/test_utils.py -n 2 --cov=src.core.utils
```

#### 5. Performance Benchmarking
```bash
./scripts/run-parallel-tests.sh benchmark
# Compares sequential vs parallel execution times
```

### Parallel Execution Benefits

✅ **Perfect Test Isolation**
- 352 instances of temporary file usage
- No global state or singleton patterns
- Zero conflicts between parallel workers

✅ **Optimal Resource Utilization**
- Auto-detection of available CPU cores (12 cores available)
- Smart worker distribution based on test type
- Leaves 2-4 cores for system processes

✅ **Smart Test Distribution**
- **Unit tests**: High parallelization (auto workers)
- **Integration tests**: Moderate parallelization (4 workers)
- **Slow tests**: Limited parallelization (2 workers)
- **Performance tests**: Serial execution

## 📊 Coverage Achievements

### Overall Coverage: 24.9% (Major Improvement from 0.4%)

**Security Infrastructure**: 273/856 statements covered
- **Compliance Reporter**: 27.0% coverage (155/456 statements)
- **Security Scanner**: 22.6% coverage (118/400 statements)

**Core Components**: High coverage on critical modules
- **Core Utils**: 100% coverage - utility functions
- **Rate Limiter**: 100% coverage - performance controls
- **Config Validator**: 93.1% coverage - configuration validation

**Enterprise Features Tested**:
- Multi-framework compliance (SOC2, GDPR, OWASP, PCI DSS, ISO 27001)
- Vulnerability scanning and secret detection
- Audit trails and risk assessment
- Performance monitoring and observability

### Generate Coverage Reports

```bash
# JSON report for automation
python3 -m pytest tests/ --cov=src --cov-report=json:coverage.json

# Terminal report with missing lines
python3 -m pytest tests/ --cov=src --cov-report=term-missing

# HTML report for detailed analysis
python3 -m pytest tests/ --cov=src --cov-report=html:coverage_html

# Coverage gate validation (0.8% minimum threshold)
./scripts/simple-coverage-gate.sh
```

## 🧪 Test Development Guidelines

### Writing Parallel-Safe Tests

**1. Use Temporary Files/Directories**
```python
def test_file_operations(tmp_path):
    # ✅ Good - uses pytest tmp_path fixture
    test_file = tmp_path / "test.json"
    test_file.write_text('{"test": "data"}')
    
def test_temp_directory(tmpdir):
    # ✅ Good - uses pytest tmpdir fixture  
    config_file = tmpdir.join("config.yaml")
    config_file.write("setting: value")
```

**2. Mock External Dependencies**
```python
@patch('src.storage.neo4j_client.GraphDatabase.driver')
def test_neo4j_connection(mock_driver):
    # ✅ Good - mocks external Neo4j service
    mock_driver.return_value = Mock()
    client = Neo4jClient()
    # Test logic here
```

**3. Follow Existing Patterns**
```python
class TestSecurityScanner:
    """Test security scanner functionality."""
    
    def test_scan_type_enum(self):
        """Test ScanType enum values."""
        assert ScanType.DEPENDENCY_SCAN.value == "dependency_scan"
        
    @pytest.mark.asyncio
    async def test_async_scanning(self):
        """Test async scanning operations."""
        # Async test implementation
```

### Test Categories & Markers

**Use markers for proper test categorization:**
```python
@pytest.mark.unit
def test_fast_function():
    """Fast unit test - high parallelization."""
    pass

@pytest.mark.integration  
def test_component_integration():
    """Integration test - moderate parallelization."""
    pass

@pytest.mark.slow
def test_complex_operation():
    """Slow test - limited parallelization."""
    pass

@pytest.mark.security
def test_security_validation():
    """Security test - dedicated security suite."""
    pass

@pytest.mark.serial
def test_must_run_alone():
    """Rare - test that must run serially."""
    pass
```

### Best Practices

**✅ DO:**
- Use `pytest.fixture` for reusable setup
- Use `tmpdir` or `tmp_path` for file operations
- Mock external services completely (Neo4j, Qdrant, Redis)
- Test both success and error paths
- Include edge cases and boundary conditions
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

**❌ DON'T:**
- Use global variables or singleton patterns
- Hardcode file paths or ports
- Make real network calls
- Depend on external services running
- Share state between tests
- Use time.sleep() (use proper async patterns)

### Example Test Structure

```python
#!/usr/bin/env python3
"""
Comprehensive tests for src/module/component.py

This test suite provides comprehensive coverage of the component,
testing all major functionality including error handling and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the module under test
from src.module.component import ComponentClass, ComponentEnum


class TestComponentEnum:
    """Test component enums and constants."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert ComponentEnum.OPTION_A.value == "option_a"
        assert ComponentEnum.OPTION_B.value == "option_b"


class TestComponentClass:
    """Test main component functionality."""
    
    @pytest.fixture
    def component(self):
        """Create component instance for testing."""
        return ComponentClass()
    
    def test_init(self, component):
        """Test component initialization."""
        assert component is not None
        assert hasattr(component, 'config')
    
    @patch('src.module.component.external_service')
    def test_component_method(self, mock_service, component):
        """Test component method with mocked dependencies."""
        # Arrange
        mock_service.return_value = {"result": "success"}
        
        # Act
        result = component.process_data("test_input")
        
        # Assert
        assert result == "success"
        mock_service.assert_called_once_with("test_input")
    
    def test_error_handling(self, component):
        """Test error handling scenarios."""
        with pytest.raises(ValueError, match="Invalid input"):
            component.process_data(None)


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete component workflow."""
        # Use tmp_path for file operations
        config_file = tmp_path / "config.yaml"
        config_file.write_text("setting: test_value")
        
        component = ComponentClass(str(config_file))
        result = component.run_workflow()
        
        assert result["status"] == "completed"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        component = ComponentClass()
        
        # Test various invalid inputs
        invalid_inputs = [None, "", [], {}]
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                component.process_data(invalid_input)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
```

## 🔧 Configuration Files

### pytest.ini (Main Configuration)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
pythonpath = . src

addopts =
    --cov=src
    --cov-report=term-missing:skip-covered
    --cov-report=json:coverage.json
    --cov-branch
    --cov-fail-under=15
    --strict-markers
    --strict-config
    --tb=short
    --durations=10
    -v
```

### pytest-parallel.ini (Parallel Configuration)
```ini
[tool:pytest]
# Same as above plus:
addopts =
    -n auto                    # Auto-detect optimal worker count
    --maxfail=10              # Stop after 10 failures
    --timeout=300             # 5-minute timeout per test
    
dist = loadscope              # Distribute by scope for better isolation
```

## 🚀 CI/CD Integration

### GitHub Actions Example
```yaml
name: Parallel Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev-parallel.txt
      
      - name: Run parallel tests
        run: |
          ./scripts/run-parallel-tests.sh all
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.json
```

### Local Development Workflow

```bash
# 1. Start development
git checkout -b feature/my-feature

# 2. Make changes to code
# ... edit files ...

# 3. Run fast tests for immediate feedback
./scripts/run-parallel-tests.sh fast

# 4. Run relevant module tests
./scripts/run-parallel-tests.sh custom tests/security/ 4

# 5. Before committing, run full suite
./scripts/run-parallel-tests.sh all

# 6. Commit and push
git add .
git commit -m "feat: add new feature with tests"
git push -u origin feature/my-feature
```

## 📈 Performance Monitoring

### Test Execution Times
```bash
# Monitor slowest tests
python3 -m pytest tests/ --durations=20

# Benchmark parallel vs sequential
./scripts/run-parallel-tests.sh benchmark

# Profile specific test files
python3 -m pytest tests/security/ --durations=10 -v
```

### Resource Usage
```bash
# Monitor CPU usage during parallel execution
htop  # Run in separate terminal during tests

# Check memory usage
python3 -m pytest tests/ --profile

# Optimize worker count based on system
./scripts/run-parallel-tests.sh custom tests/ 8  # Try different worker counts
```

## 🔍 Troubleshooting

### Common Issues

**Tests failing in parallel but passing sequentially:**
- Check for global state or shared resources
- Ensure proper use of temporary files/directories
- Verify external service mocking

**Slow parallel execution:**
- Reduce worker count for CPU-bound tests
- Check for I/O bottlenecks
- Profile test execution with `--durations`

**Coverage issues:**
- Use proper module path: `--cov=src.module.component`
- Ensure all test files are discovered
- Check for import issues in parallel workers

### Debug Commands
```bash
# Run single test with full output
python3 -m pytest tests/path/to/test.py::test_function -v -s

# Run without parallelization for debugging
python3 -m pytest tests/security/ -v --tb=long

# Check test discovery
python3 -m pytest --collect-only tests/

# Validate parallel execution
./scripts/run-parallel-tests.sh validate
```

## 📚 Additional Resources

- **Test Coverage Report**: Run `python3 -m pytest tests/ --cov-report=html` and open `coverage_html/index.html`
- **Parallel Testing Docs**: [pytest-xdist documentation](https://pytest-xdist.readthedocs.io/)
- **Async Testing**: [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- **Mocking Guide**: [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)

## 🎯 Next Steps

1. **Use parallel testing** for all development workflows
2. **Add test markers** to new tests for proper categorization
3. **Monitor coverage trends** and aim for continuous improvement
4. **Optimize test execution** based on performance monitoring
5. **Contribute test improvements** following established patterns

The Veris Memory test suite provides a robust foundation for ensuring code quality while maximizing developer productivity through parallel execution and comprehensive coverage reporting.