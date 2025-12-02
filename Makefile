# Veris Memory - Local CI/CD and Coverage Targets

.PHONY: help test coverage coverage-gate coverage-report lint format type-check pre-commit install-dev clean

# Default target
help:
	@# Check for common issues and warn
	@if [ -f coverage.py ]; then \
		echo "⚠️  WARNING: coverage.py found - this will break pytest-cov!"; \
		echo "   Run 'make fix-coverage' to resolve"; \
		echo ""; \
	fi
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  VERIS MEMORY TEST COMMANDS - START HERE!"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "🚀 QUICK START (Try these first!):"
	@echo "  make coverage     Get code coverage (~25% expected)"
	@echo "  make test         Run all tests with parallel execution"
	@echo "  make quick        Fast test run for quick feedback"
	@echo ""
	@echo "📊 Testing & Coverage:"
	@echo "  coverage          Run tests and get coverage % [USE THIS!]"
	@echo "  test              Run all tests (parallel)"
	@echo "  quick             Fast test subset"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  coverage-gate     Run coverage gate (fails if below threshold)"
	@echo "  coverage-report   Generate and open HTML coverage report"
	@echo ""
	@echo "🔧 Troubleshooting:"
	@echo "  fix-coverage      Fix common test issues (run if tests fail)"
	@echo "  test-subset       Quick coverage on working tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run all linting (flake8, etc.)"
	@echo "  format            Format code (black, isort)"
	@echo "  type-check        Run mypy type checking"
	@echo "  pre-commit        Run all pre-commit checks"
	@echo ""
	@echo "Development:"
	@echo "  install-dev       Install development dependencies"
	@echo "  clean             Clean generated files"
	@echo ""

# Test targets
test:
	@echo "🧪 Running all tests..."
	@# Use parallel execution to prevent hanging
	@if [ -f ./scripts/run-tests.sh ]; then \
		./scripts/run-tests.sh standard; \
	else \
		python3 -m pytest tests/ -n auto -v --tb=short --maxfail=10; \
	fi

test-unit:
	@echo "🧪 Running unit tests..."
	python3 -m pytest tests/unit/ -v -m unit

# Quick test - agents often try this first
quick:
	@echo "⚡ Running quick tests (parallel execution)..."
	@if [ -f ./scripts/run-tests.sh ]; then \
		./scripts/run-tests.sh quick; \
	else \
		python3 -m pytest tests/ -n auto --tb=short --maxfail=5 -q; \
	fi

# Fix common issues that break testing
fix-coverage:
	@echo "🔧 Fixing common test issues..."
	@if [ -f coverage.py ]; then \
		echo "  ✅ Renaming coverage.py to run_coverage.py"; \
		mv coverage.py run_coverage.py; \
	fi
	@echo "  ✅ Installing test dependencies..."
	@pip install pytest pytest-cov pytest-xdist pytest-timeout -q
	@echo "🎉 Test environment fixed!"

# Subset coverage - always works  
test-subset:
	@echo "📊 Running subset for quick coverage..."
	@rm -f .coverage coverage.json 2>/dev/null || true
	@python3 -m pytest tests/core tests/storage tests/security \
		--cov=src \
		--cov-report=json:coverage.json \
		--cov-report=term:skip-covered \
		--tb=no \
		--maxfail=50 \
		-q || true
	@echo ""
	@if [ -f coverage.json ]; then \
		python3 -c "import json; data=json.load(open('coverage.json')); print(f'📊 SUBSET COVERAGE: {data[\"totals\"][\"percent_covered\"]:.1f}%')"; \
		echo "   (Partial - full coverage would be higher)"; \
	fi

test-integration:
	@echo "🧪 Running integration tests..."
	python3 -m pytest tests/integration/ -v -m integration

test-fast:
	@echo "🧪 Running fast tests (excluding slow)..."
	python3 -m pytest tests/ -v -m "not slow"

# Coverage targets
coverage:
	@echo "📊 Running tests with coverage (2-3 minutes)..."
	@# Fix common issues first
	@if [ -f coverage.py ]; then \
		echo "⚠️  Removing coverage.py (conflicts with Python package)..."; \
		rm -f coverage.py; \
	fi
	@# Clean old coverage data
	@rm -f .coverage coverage.json 2>/dev/null || true
	@# Run tests with coverage - sequential for reliability
	@echo "Running all tests (some failures expected)..."
	@python3 -m pytest tests/ \
		--cov=src \
		--cov-report=json:coverage.json \
		--cov-report=term:skip-covered \
		--tb=short \
		--maxfail=100 \
		--continue-on-collection-errors \
		-q || true
	@# Always show the result
	@echo ""
	@echo "════════════════════════════════════════════════════════"
	@if [ -f coverage.json ]; then \
		python3 -c "import json; data=json.load(open('coverage.json')); print(f'📊 TOTAL COVERAGE: {data[\"totals\"][\"percent_covered\"]:.1f}%')"; \
		echo "   Expected: ~25% (this is correct, not 0.4%)"; \
		echo "   Coverage saved to: coverage.json"; \
	else \
		echo "❌ No coverage data (too many test failures)"; \
		echo "   Try: make test-subset for partial coverage"; \
	fi
	@echo "════════════════════════════════════════════════════════"

coverage-gate: scripts/coverage-gate.sh
	@echo "🚪 Running coverage gate check..."
	@./scripts/coverage-gate.sh

coverage-report: coverage
	@echo "📄 Coverage report generated..."
	@echo "JSON report: coverage.json"
	@python3 -c "import json; data=json.load(open('coverage.json')); print(f'Total coverage: {data[\"totals\"][\"percent_covered\"]:.1f}%')"

coverage-ci:
	@echo "🤖 Running CI coverage check..."
	python3 -m pytest \
		--cov=src \
		--cov-report=term \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=15 \
		--tb=short \
		-q

# Code quality targets
lint:
	@echo "🔍 Running linting..."
	@if command -v flake8 > /dev/null 2>&1; then \
		flake8 src/ tests/; \
	else \
		echo "⚠️  flake8 not installed - skipping"; \
	fi

format:
	@echo "🎨 Formatting code..."
	@if command -v black > /dev/null 2>&1; then \
		black src/ tests/; \
	else \
		echo "⚠️  black not installed - skipping"; \
	fi
	@if command -v isort > /dev/null 2>&1; then \
		isort src/ tests/; \
	else \
		echo "⚠️  isort not installed - skipping"; \
	fi

type-check:
	@echo "🔍 Running type checking..."
	@if command -v mypy > /dev/null 2>&1; then \
		mypy src/; \
	else \
		echo "⚠️  mypy not installed - skipping"; \
	fi

# Pre-commit workflow
pre-commit: format lint type-check coverage-gate
	@echo "✅ All pre-commit checks passed!"

# Quick pre-commit (without coverage)
pre-commit-fast: format lint type-check
	@echo "✅ Fast pre-commit checks passed!"

# Development setup
install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -e .
	@if [ -f requirements-dev.txt ]; then \
		pip install -r requirements-dev.txt; \
	fi
	@if [ -f requirements.txt ]; then \
		pip install -r requirements.txt; \
	fi

# Security checks
security:
	@echo "🔒 Running security checks..."
	@if command -v safety > /dev/null 2>&1; then \
		safety check; \
	else \
		echo "⚠️  safety not installed - skipping"; \
	fi
	@if command -v bandit > /dev/null 2>&1; then \
		bandit -r src/; \
	else \
		echo "⚠️  bandit not installed - skipping"; \
	fi

# Performance tests
test-performance:
	@echo "⚡ Running performance tests..."
	python3 -m pytest tests/performance/ -v -m performance

# Cleanup
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf .coverage coverage.xml coverage.json
	rm -rf .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf *.egg-info/ build/ dist/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Docker targets
docker-test:
	@echo "🐳 Running tests in Docker..."
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit

docker-coverage:
	@echo "🐳 Running coverage in Docker..."
	docker-compose -f docker-compose.test.yml run --rm test make coverage

# Continuous Integration simulation
ci: clean install-dev lint type-check security coverage-ci
	@echo "🤖 CI pipeline completed successfully!"

# Watch mode for development
watch:
	@echo "👀 Starting test watcher..."
	@if command -v pytest-watch > /dev/null 2>&1; then \
		ptw -- --testmon; \
	else \
		echo "⚠️  pytest-watch not installed"; \
		echo "Install with: pip install pytest-watch"; \
	fi

# Coverage tracking over time
coverage-track:
	@echo "📈 Tracking coverage over time..."
	@mkdir -p reports/coverage-history/
	@timestamp=$$(date +%Y%m%d-%H%M%S); \
	python3 -c "import json; data=json.load(open('coverage.json')); print(f'{data[\"totals\"][\"percent_covered\"]:.1f}')" > reports/coverage-history/$$timestamp.txt
	@echo "Coverage snapshot saved to reports/coverage-history/"