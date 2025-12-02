#!/bin/bash
# 🚪 CI/CD Deployment Gate with Manifest Verification
# Prevents silent mis-indexing by validating vector dimensions before deployment

set -euo pipefail

echo "🚪 CI/CD DEPLOYMENT GATE"
echo "========================"

# Configuration
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
VECTOR_COLLECTION="${VECTOR_COLLECTION:-context_store}"
DATE_STAMP=$(date +"%Y%m%d-%H%M%S")
MANIFEST_FILE="${MANIFEST_FILE:-deployments/manifest-${DATE_STAMP}.json}"

echo "🌐 Qdrant URL: $QDRANT_URL"
echo "📁 Collection: $VECTOR_COLLECTION" 
echo "📄 Manifest: $MANIFEST_FILE"
echo ""

# Step 1: Pre-flight Config Validation
echo "1️⃣ PRE-FLIGHT CONFIG VALIDATION"
echo "==============================="

if [[ ! -f "production_locked_config.yaml" ]]; then
    echo "❌ production_locked_config.yaml not found"
    exit 1
fi

if [[ ! -f "$MANIFEST_FILE" ]]; then
    echo "❌ Manifest not found: $MANIFEST_FILE"
    exit 1
fi

echo "✅ Required files present"

# Step 2: Manifest Cross-Validation
echo ""
echo "2️⃣ MANIFEST CROSS-VALIDATION"
echo "============================="

echo "🔍 Running manifest verifier..."
python ops/verify/manifest_verifier.py \
  --config production_locked_config.yaml \
  --manifest "$MANIFEST_FILE" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$VECTOR_COLLECTION" \
  --require-text-index

VERIFIER_EXIT_CODE=$?

if [[ $VERIFIER_EXIT_CODE -eq 0 ]]; then
    echo "✅ Manifest verification: PASSED"
else
    echo "❌ Manifest verification: FAILED (exit code: $VERIFIER_EXIT_CODE)"
    echo ""
    echo "🚨 DEPLOYMENT BLOCKED"
    echo "====================="
    echo "Vector dimension or distance mismatch detected!"
    echo "This prevents silent mis-indexing in production."
    echo ""
    echo "🔧 To fix:"
    echo "1. Check production_locked_config.yaml embedding configuration"
    echo "2. Verify Qdrant collection vector settings match"
    echo "3. Ensure manifest vector_dim matches frozen config"
    echo "4. Re-run deployment after fixing configuration drift"
    exit 1
fi

# Step 3: Smoke Test Gate
echo ""
echo "3️⃣ SMOKE TEST GATE"
echo "=================="

if [[ -f "/tmp/veris_smoke_report.json" ]]; then
    echo "🧪 Running deploy gate validation..."
    python ops/smoke/deploy_guard.py /tmp/veris_smoke_report.json
    GATE_EXIT_CODE=$?
    
    if [[ $GATE_EXIT_CODE -eq 0 ]]; then
        echo "✅ Smoke test gate: GREEN (PASSED)"
    else
        echo "❌ Smoke test gate: RED (FAILED)"
        echo ""
        echo "🚨 DEPLOYMENT BLOCKED" 
        echo "====================="
        echo "Smoke tests failed - deployment not safe!"
        exit 1
    fi
else
    echo "⚠️  No smoke report found - running basic smoke test..."
    python ops/smoke/smoke_runner.py
    python ops/smoke/deploy_guard.py /tmp/veris_smoke_report.json
    echo "✅ Emergency smoke test: PASSED"
fi

# Step 4: Final Go/No-Go Decision
echo ""
echo "4️⃣ DEPLOYMENT DECISION"
echo "======================"

echo "✅ All gates passed:"
echo "   🔍 Manifest verification: PASSED"
echo "   🧪 Smoke test gate: GREEN"
echo "   📊 Vector dimensions: Aligned"
echo "   🔒 Configuration: Validated"

echo ""
echo "🚀 DEPLOYMENT: APPROVED ✅"
echo "=========================="
echo "Safe to proceed with production deployment"
echo ""
echo "Next steps:"
echo "  docker-compose up -d --build"
echo "  python ops/verify/manifest_verifier.py --require-text-index  # Post-deploy verification"
echo ""

exit 0