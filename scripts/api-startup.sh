#!/bin/sh
# API Startup Script with Enhanced Error Handling
set -e  # Exit on error

echo "=========================================="
echo "🚀 Veris Memory API Container Starting..."
echo "=========================================="
echo "Time: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

# Function to handle errors
handle_error() {
    echo ""
    echo "❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌"
    echo "❌ CRITICAL ERROR: API FAILED TO START!"
    echo "❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌"
    echo "Error details: $1"
    echo ""
    echo "Diagnostic Information:"
    echo "-----------------------"
    echo "Python version:"
    python3 --version
    echo ""
    echo "Installed packages:"
    pip list | grep -E "(uvicorn|fastapi|pydantic)" || true
    echo ""
    echo "Directory contents:"
    ls -la
    echo ""
    echo "Environment variables:"
    env | sort
    echo "❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌"
    exit 1
}

# Check environment variables
echo "📋 Environment Configuration:"
echo "-----------------------------"
echo "NEO4J_URI=${NEO4J_URI:-NOT SET}"
echo "NEO4J_USER=${NEO4J_USER:-NOT SET}"
echo "NEO4J_PASSWORD=${NEO4J_PASSWORD:+[SET]}"
echo "QDRANT_URL=${QDRANT_URL:-NOT SET}"
echo "REDIS_URL=${REDIS_URL:-NOT SET}"
echo "API_SERVER_PORT=${API_SERVER_PORT:-8001}"
echo "LOG_LEVEL=${LOG_LEVEL:-info}"
echo ""

# Check config file
echo "📁 Configuration File Check:"
echo "----------------------------"
if [ -f ".ctxrc.yaml" ]; then
    echo "✅ Config file .ctxrc.yaml exists"
    echo "   Size: $(stat -c%s .ctxrc.yaml 2>/dev/null || echo 'unknown') bytes"
else
    echo "⚠️  WARNING: Config file .ctxrc.yaml not found"
    echo "   This may affect some features but API should still start"
fi
echo ""

# Check source files
echo "📦 Source Files Check:"
echo "----------------------"
if [ -d "src/api" ]; then
    echo "✅ API source directory exists"
    if [ -f "src/api/main.py" ]; then
        echo "✅ main.py exists"
    else
        handle_error "src/api/main.py not found"
    fi
else
    handle_error "src/api directory not found"
fi
echo ""

# Test Python import
echo "🐍 Testing Python Import:"
echo "-------------------------"
python3 -c "
import sys
print(f'Python {sys.version}')
try:
    from src.api import main
    print('✅ API module import successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
except SyntaxError as e:
    print(f'❌ Syntax error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || handle_error "Python import test failed"
echo ""

# Start the API server
echo "🚀 Starting Uvicorn Server:"
echo "============================"
echo "Command: python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port ${API_SERVER_PORT:-8001} --log-level ${LOG_LEVEL:-info}"
echo ""

# Use exec to replace shell with uvicorn process
exec python3 -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port "${API_SERVER_PORT:-8001}" \
    --log-level "${LOG_LEVEL:-info}" \
    || handle_error "Uvicorn failed to start"