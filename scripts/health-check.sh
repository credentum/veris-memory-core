#!/bin/bash
# Comprehensive health check for Context Store deployment
# Verifies all services are running with correct configuration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
NEO4J_URL="${NEO4J_URL:-http://localhost:7474}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
EXPECTED_DIM="${EXPECTED_DIM:-384}"
EXPECTED_DISTANCE="${EXPECTED_DISTANCE:-Cosine}"

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Logging functions
log_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((CHECKS_PASSED++))
}

log_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((CHECKS_FAILED++))
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Check if service is reachable
check_service() {
    local name=$1
    local url=$2
    local expected_content=$3
    
    if curl -s -f -o /dev/null -w "%{http_code}" "$url" | grep -q "200"; then
        if [ -n "$expected_content" ]; then
            if curl -s "$url" | grep -q "$expected_content"; then
                log_pass "$name is running and healthy"
                return 0
            else
                log_warn "$name is running but response unexpected"
                return 1
            fi
        else
            log_pass "$name is running"
            return 0
        fi
    else
        log_fail "$name is not reachable at $url"
        return 1
    fi
}

# Check Qdrant collection configuration
check_qdrant_collection() {
    local collection_url="$QDRANT_URL/collections/context_embeddings"
    
    if ! curl -s -f "$collection_url" > /tmp/qdrant_collection.json 2>/dev/null; then
        log_fail "Qdrant collection 'context_embeddings' not found"
        return 1
    fi
    
    # Check dimensions
    local dimensions=$(python3 -c "
import json
with open('/tmp/qdrant_collection.json') as f:
    data = json.load(f)
    size = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
    print(size)
" 2>/dev/null || echo "0")
    
    if [ "$dimensions" = "$EXPECTED_DIM" ]; then
        log_pass "Qdrant collection has correct dimensions: $EXPECTED_DIM"
    else
        log_fail "Qdrant collection has wrong dimensions: $dimensions (expected $EXPECTED_DIM)"
        return 1
    fi
    
    # Check distance metric
    local distance=$(python3 -c "
import json
with open('/tmp/qdrant_collection.json') as f:
    data = json.load(f)
    dist = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('distance', '')
    print(dist)
" 2>/dev/null || echo "Unknown")
    
    if [ "$distance" = "$EXPECTED_DISTANCE" ]; then
        log_pass "Qdrant collection uses correct distance metric: $EXPECTED_DISTANCE"
    else
        log_fail "Qdrant collection uses wrong distance metric: $distance (expected $EXPECTED_DISTANCE)"
        return 1
    fi
    
    # Check collection status
    local status=$(python3 -c "
import json
with open('/tmp/qdrant_collection.json') as f:
    data = json.load(f)
    status = data.get('result', {}).get('status', 'unknown')
    print(status)
" 2>/dev/null || echo "unknown")
    
    if [ "$status" = "green" ]; then
        log_pass "Qdrant collection status is healthy (green)"
    elif [ "$status" = "yellow" ]; then
        log_warn "Qdrant collection status is yellow (degraded)"
    else
        log_fail "Qdrant collection status is unhealthy: $status"
        return 1
    fi
    
    return 0
}

# Check Docker containers
check_docker_containers() {
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found, skipping container checks"
        return 0
    fi
    
    local containers=("qdrant" "neo4j" "redis")
    
    for container in "${containers[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "$container"; then
            local status=$(docker ps --filter "name=$container" --format "{{.Status}}" | head -1)
            if echo "$status" | grep -q "healthy"; then
                log_pass "Container $container is healthy"
            elif echo "$status" | grep -q "Up"; then
                log_warn "Container $container is up but health status unknown"
            else
                log_fail "Container $container status: $status"
            fi
        else
            log_fail "Container $container not found"
        fi
    done
}

# Main health check
main() {
    header "Context Store Health Check"
    log_info "Checking deployment health..."
    log_info "Expected configuration: $EXPECTED_DIM dimensions, $EXPECTED_DISTANCE distance"
    
    header "Service Connectivity"
    check_service "Qdrant" "$QDRANT_URL" "title.*Qdrant"
    check_service "Neo4j" "$NEO4J_URL" ""
    
    header "Qdrant Configuration"
    check_qdrant_collection
    
    header "Docker Container Status"
    check_docker_containers
    
    header "Summary"
    echo -e "${GREEN}Passed: $CHECKS_PASSED${NC}"
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    fi
    
    if [ $CHECKS_FAILED -gt 0 ]; then
        echo -e "${RED}Failed: $CHECKS_FAILED${NC}"
        echo ""
        log_fail "Health check failed! Please review the errors above."
        exit 1
    else
        echo ""
        log_pass "All health checks passed! Deployment is healthy."
        
        if [ $WARNINGS -gt 0 ]; then
            log_warn "There are $WARNINGS warnings that should be reviewed."
        fi
    fi
    
    # Clean up
    rm -f /tmp/qdrant_collection.json
}

# Run main function
main "$@"