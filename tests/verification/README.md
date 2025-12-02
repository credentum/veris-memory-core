# Verification Test Scripts

This directory contains standalone verification test scripts that can be run manually when needed, without affecting the main monitoring dashboard.

## Available Tests

### 1. TLS/mTLS Verification (`tls_verifier.py`)
Tests TLS/mTLS configurations for Veris Memory services:
- Certificate validation and expiry tracking
- Cipher suite security assessment  
- Protocol version compliance
- Client certificate authentication (mTLS)

**Usage:**
```bash
python3 tests/verification/tls_verifier.py
```

### 2. Backup Restore Drill (`restore_drill.py`)
Tests backup restore procedures with timing and integrity verification:
- Redis AOF restore testing
- Qdrant snapshot restore testing
- Neo4j dump restore testing
- Sub-300 second compliance checking
- Data integrity verification

**Usage:**
```bash
python3 tests/verification/restore_drill.py
```

### 3. Comprehensive Test Suite (`run_all_tests.py`)
Runs all verification tests and provides comprehensive reporting:
- Full test suite with detailed reporting
- Quick test mode for rapid checks
- Overall compliance assessment
- Recommendations for improvements

**Usage:**
```bash
# Full test suite
python3 tests/verification/run_all_tests.py

# Quick test mode
python3 tests/verification/run_all_tests.py --quick
```

## Test Results

All tests provide:
- ✅ **PASS** - Requirements met
- ⚠️ **WARNING** - Minor issues detected
- ❌ **FAIL** - Requirements not met
- 🚨 **ERROR** - Test execution failed

## Configuration

Each test script can be configured by modifying the `_get_default_config()` method or passing a config dictionary to the constructor.

### Common Configuration Options:
- `target_restore_time_seconds`: Maximum allowed restore time (default: 300)
- `connection_timeout_seconds`: Network connection timeout (default: 10)
- `certificate_warning_days`: Certificate expiry warning threshold (default: 30)
- `cleanup_after_drill`: Clean up temporary files (default: True)

## Integration

These test scripts are designed to be:
- **Standalone** - Can run independently without the main application
- **Manual execution** - Run on-demand rather than continuously
- **CI/CD friendly** - Return proper exit codes for automation
- **Logging enabled** - Provide detailed execution information

## Mock Testing

For development and testing purposes, the scripts include mock implementations:
- **TLS tests** will attempt real connections but gracefully handle connection failures
- **Restore drills** create mock backup files and simulate restore procedures
- **All tests** provide realistic timing and verification scenarios

This allows the tests to run in any environment, even when actual services aren't available.