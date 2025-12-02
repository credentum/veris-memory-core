#!/bin/bash
# ================================================================
# VERIS MEMORY UNIFIED TEST RUNNER - SINGLE SOURCE OF TRUTH
# ================================================================
# This is THE authoritative script for running tests in veris-memory
# All other test scripts should delegate to this one
# 
# Usage:
#   ./scripts/run-tests.sh [mode] [options]
#
# Modes:
#   quick     - Fast feedback loop (unit tests only, ~30 seconds)
#   standard  - Default test run (all tests, ~2-3 minutes)  
#   parallel  - Parallel execution with optimal workers
#   coverage  - Full coverage analysis with detailed report
#   ci        - CI/CD mode (matches GitHub Actions)
#   debug     - Sequential run with verbose output
#
# Options:
#   --path PATH      - Test specific path/file
#   --workers N      - Override worker count for parallel mode
#   --no-cov         - Skip coverage reporting
#   --markers MARKS  - Run tests matching markers (e.g., "unit and not slow")
#
# Examples:
#   ./scripts/run-tests.sh                    # Standard test run
#   ./scripts/run-tests.sh quick              # Fast unit tests only
#   ./scripts/run-tests.sh coverage           # Full coverage report
#   ./scripts/run-tests.sh parallel --workers 4
#   ./scripts/run-tests.sh ci                 # Match CI environment
# ================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=15  # Current minimum threshold
COVERAGE_TARGET=35     # Target we're working towards
DEFAULT_WORKERS="auto"
MAX_FAILURES=10

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Function to print colored status
print_header() {
    echo -e "\n${CYAN}================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================${NC}\n"
}

print_status() {
    echo -e "${BLUE}[TEST-RUNNER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to detect optimal worker count
detect_workers() {
    local cpu_count=$(nproc 2>/dev/null || echo 4)
    local optimal_workers=$((cpu_count - 2))
    
    # Minimum 2 workers, maximum 12 workers
    if [ $optimal_workers -lt 2 ]; then
        optimal_workers=2
    elif [ $optimal_workers -gt 12 ]; then
        optimal_workers=12
    fi
    
    echo $optimal_workers
}

# Function to ensure dependencies are installed
ensure_dependencies() {
    print_status "Checking test dependencies..."
    
    # Check for required packages
    local missing_packages=""
    
    if ! python3 -c "import pytest" 2>/dev/null; then
        missing_packages="$missing_packages pytest"
    fi
    
    if ! python3 -c "import pytest_cov" 2>/dev/null; then
        missing_packages="$missing_packages pytest-cov"
    fi
    
    if ! python3 -c "import xdist" 2>/dev/null; then
        missing_packages="$missing_packages pytest-xdist"
    fi
    
    if ! python3 -c "import pytest_timeout" 2>/dev/null; then
        missing_packages="$missing_packages pytest-timeout"
    fi
    
    if ! python3 -c "import pytest_asyncio" 2>/dev/null; then
        missing_packages="$missing_packages pytest-asyncio"
    fi
    
    if [ ! -z "$missing_packages" ]; then
        print_warning "Installing missing packages: $missing_packages"
        pip install $missing_packages
    else
        print_success "All test dependencies installed"
    fi
}

# Function to run tests in QUICK mode (unit tests only with parallel)
run_quick_tests() {
    local workers="$(detect_workers)"
    print_header "QUICK TEST MODE - Unit Tests Only (Parallel: $workers workers)"
    print_status "Running fast unit tests for quick feedback..."
    
    python3 -m pytest tests/ \
        -n $workers \
        -m "unit or (not slow and not integration and not e2e)" \
        --tb=short \
        --maxfail=$MAX_FAILURES \
        -q \
        --no-header \
        --no-summary \
        -r fE
}

# Function to run tests in STANDARD mode (all tests with parallel execution)
run_standard_tests() {
    local workers="$(detect_workers)"
    print_header "STANDARD TEST MODE - All Tests (Parallel: $workers workers)"
    print_status "Running complete test suite with parallel execution..."
    
    python3 -m pytest tests/ \
        -n $workers \
        --dist loadscope \
        --tb=short \
        --maxfail=$MAX_FAILURES \
        --cov=src \
        --cov-report=term:skip-covered \
        --cov-report=json:coverage.json \
        -v
}

# Function to run tests in PARALLEL mode
run_parallel_tests() {
    local workers="${1:-$(detect_workers)}"
    
    print_header "PARALLEL TEST MODE - $workers Workers"
    print_status "Running tests in parallel for maximum speed..."
    
    # Use pytest-parallel.ini configuration
    python3 -m pytest tests/ \
        -c pytest-parallel.ini \
        -n $workers \
        --dist loadscope \
        --maxfail=$MAX_FAILURES \
        --tb=short \
        -v
}

# Function to run tests in COVERAGE mode (with parallel execution)
run_coverage_tests() {
    local workers="$(detect_workers)"
    print_header "COVERAGE TEST MODE - Detailed Analysis (Parallel: $workers workers)"
    print_status "Running full test suite with comprehensive coverage reporting..."
    
    # Clean previous coverage data
    rm -f .coverage coverage.xml coverage.json
    
    # Run tests with full coverage IN PARALLEL
    python3 -m pytest tests/ \
        -n $workers \
        --dist loadscope \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=json:coverage.json \
        --cov-report=html:coverage_html \
        --cov-branch \
        --tb=short \
        -v
    
    # Extract and display coverage metrics
    if [ -f coverage.json ]; then
        print_status "Analyzing coverage results..."
        
        coverage_pct=$(python3 -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)
print(f\"{data['totals']['percent_covered']:.1f}\")
" 2>/dev/null || echo "0.0")
        
        echo ""
        print_status "Coverage Summary:"
        echo -e "   ${CYAN}Current Coverage: ${coverage_pct}%${NC}"
        echo -e "   ${YELLOW}Minimum Required: ${COVERAGE_THRESHOLD}%${NC}"
        echo -e "   ${GREEN}Target Goal: ${COVERAGE_TARGET}%${NC}"
        
        # Check if we meet minimum threshold
        if python3 -c "exit(0 if $coverage_pct >= $COVERAGE_THRESHOLD else 1)"; then
            print_success "Coverage threshold met!"
        else
            print_warning "Coverage below minimum threshold"
        fi
        
        # Module breakdown
        echo ""
        print_status "Coverage by Module:"
        python3 -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)

modules = {}
for filename, info in data['files'].items():
    if filename.startswith('src/'):
        parts = filename.split('/')
        module = parts[1] if len(parts) > 1 else 'root'
        if module not in modules:
            modules[module] = {'statements': 0, 'covered': 0, 'files': 0}
        modules[module]['statements'] += info['summary']['num_statements']
        modules[module]['covered'] += info['summary']['covered_lines']
        modules[module]['files'] += 1

for module, stats in sorted(modules.items()):
    if stats['statements'] > 0:
        pct = (stats['covered'] / stats['statements']) * 100
        print(f'   {module:15s}: {pct:5.1f}% ({stats[\"covered\"]}/{stats[\"statements\"]} lines, {stats[\"files\"]} files)')
"
        
        echo ""
        print_status "HTML coverage report generated: coverage_html/index.html"
    fi
}

# Function to run tests in CI mode (matches GitHub Actions)
run_ci_tests() {
    local workers="$(detect_workers)"
    print_header "CI TEST MODE - GitHub Actions Compatible"
    print_status "Running tests as configured in CI/CD pipeline with $workers parallel workers..."
    
    # Run tests with parallel execution for speed in CI
    python3 -m pytest tests/ \
        -n $workers \
        --dist loadscope \
        --cov=src \
        --cov-report=term \
        --cov-report=json:coverage.json \
        -m "not integration and not e2e" \
        --tb=short \
        --maxfail=$MAX_FAILURES
}

# Function to run tests in DEBUG mode
run_debug_tests() {
    print_header "DEBUG TEST MODE - Verbose Sequential"
    print_status "Running tests sequentially with full output..."
    
    python3 -m pytest tests/ \
        --tb=long \
        --capture=no \
        -vv \
        --log-cli-level=DEBUG
}

# Function to display usage
show_usage() {
    print_header "VERIS MEMORY TEST RUNNER"
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  quick     - Fast unit tests only (~30 seconds)"
    echo "  standard  - All tests with basic coverage (default)"
    echo "  parallel  - Parallel execution for speed"
    echo "  coverage  - Full coverage analysis with reports"
    echo "  ci        - Match CI/CD environment configuration"
    echo "  debug     - Sequential with verbose output"
    echo ""
    echo "Options:"
    echo "  --path PATH     - Test specific path/file"
    echo "  --workers N     - Worker count for parallel mode"
    echo "  --no-cov        - Skip coverage reporting"
    echo "  --markers MARKS - Run tests matching markers"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run standard tests"
    echo "  $0 quick                     # Quick unit tests"
    echo "  $0 coverage                  # Full coverage report"
    echo "  $0 parallel --workers 4      # Parallel with 4 workers"
    echo "  $0 --path tests/security/    # Test specific directory"
}

# Function to display current test statistics
show_statistics() {
    print_header "TEST STATISTICS"
    
    # Count test files
    total_files=$(find tests/ -name "test_*.py" | wc -l)
    echo "Total test files: $total_files"
    
    # Count by directory
    echo ""
    echo "Tests by module:"
    for dir in core health integration mcp_server monitoring security storage validators unit; do
        if [ -d "tests/$dir" ]; then
            count=$(find tests/$dir -name "*.py" 2>/dev/null | wc -l)
            echo "  $dir: $count files"
        fi
    done
    
    # Count test functions
    echo ""
    total_tests=$(python3 -m pytest tests/ --collect-only -q 2>/dev/null | tail -1 | awk '{print $1}')
    echo "Total test functions: $total_tests"
}

# Main execution
main() {
    local mode="${1:-standard}"
    shift || true
    
    # Parse additional options
    local test_path=""
    local workers=""
    local no_cov=false
    local markers=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --path)
                test_path="$2"
                shift 2
                ;;
            --workers)
                workers="$2"
                shift 2
                ;;
            --no-cov)
                no_cov=true
                shift
                ;;
            --markers)
                markers="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Ensure dependencies
    ensure_dependencies
    
    # Show statistics if requested
    if [ "$mode" = "stats" ]; then
        show_statistics
        exit 0
    fi
    
    # Execute based on mode
    case $mode in
        quick)
            run_quick_tests
            ;;
        standard)
            run_standard_tests
            ;;
        parallel)
            run_parallel_tests "$workers"
            ;;
        coverage)
            run_coverage_tests
            ;;
        ci)
            run_ci_tests
            ;;
        debug)
            run_debug_tests
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_usage
            exit 1
            ;;
    esac
    
    # Final status
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Test run completed successfully!"
    else
        echo ""
        print_error "Test run failed. Check output above for details."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"