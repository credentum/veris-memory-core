#!/bin/bash
#
# NGINX CONFIGURATION TESTS
# Purpose: Test nginx configuration syntax and rate limiting setup
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            Nginx Configuration Tests                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test 1: Nginx config files exist
echo -e "${BLUE}Test 1: Nginx config files exist${NC}"
if [[ -f "${REPO_ROOT}/docker/nginx/nginx.conf" ]]; then
    test_pass "nginx.conf exists"
else
    test_fail "nginx.conf missing"
fi

if [[ -f "${REPO_ROOT}/docker/nginx/conf.d/voice-bot.conf" ]]; then
    test_pass "voice-bot.conf exists"
else
    test_fail "voice-bot.conf missing"
fi

if [[ -f "${REPO_ROOT}/docker/nginx/conf.d/livekit.conf" ]]; then
    test_pass "livekit.conf exists"
else
    test_fail "livekit.conf missing"
fi

echo ""

# Test 2: Rate limiting configuration present
echo -e "${BLUE}Test 2: Rate limiting configuration${NC}"
if grep -q "limit_req_zone" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "Rate limiting zones configured"
else
    test_fail "Rate limiting zones not found"
fi

if grep -q "voicebot_limit" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "Voice-bot rate limit zone defined"
else
    test_fail "Voice-bot rate limit zone not found"
fi

if grep -q "livekit_limit" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "LiveKit rate limit zone defined"
else
    test_fail "LiveKit rate limit zone not found"
fi

echo ""

# Test 3: Environment variable substitution configured
echo -e "${BLUE}Test 3: Environment variable configuration${NC}"
if grep -q '\${VOICE_BOT_RATE_LIMIT}' "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "Voice-bot rate limit is configurable"
else
    test_fail "Voice-bot rate limit not configurable"
fi

if grep -q '\${LIVEKIT_RATE_LIMIT}' "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "LiveKit rate limit is configurable"
else
    test_fail "LiveKit rate limit not configurable"
fi

if grep -q '\${RATE_LIMIT_ZONE_SIZE}' "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "Rate limit zone size is configurable"
else
    test_fail "Rate limit zone size not configurable"
fi

echo ""

# Test 4: Security headers configured
echo -e "${BLUE}Test 4: Security headers${NC}"
if grep -q "X-Frame-Options" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "X-Frame-Options header configured"
else
    test_fail "X-Frame-Options header not found"
fi

if grep -q "X-Content-Type-Options" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "X-Content-Type-Options header configured"
else
    test_fail "X-Content-Type-Options header not found"
fi

if grep -q "X-XSS-Protection" "${REPO_ROOT}/docker/nginx/nginx.conf"; then
    test_pass "X-XSS-Protection header configured"
else
    test_fail "X-XSS-Protection header not found"
fi

echo ""

# Test 5: SSL configuration
echo -e "${BLUE}Test 5: SSL configuration${NC}"
if grep -q "ssl_certificate" "${REPO_ROOT}/docker/nginx/conf.d/voice-bot.conf"; then
    test_pass "SSL certificate path configured"
else
    test_fail "SSL certificate path not found"
fi

if grep -q "ssl_protocols" "${REPO_ROOT}/docker/nginx/conf.d/voice-bot.conf"; then
    test_pass "SSL protocols configured"
else
    test_fail "SSL protocols not found"
fi

if grep -q "TLSv1.2\\|TLSv1.3" "${REPO_ROOT}/docker/nginx/conf.d/voice-bot.conf"; then
    test_pass "Modern TLS versions configured"
else
    test_fail "Modern TLS versions not found"
fi

echo ""

# Test 6: Dockerfile checks
echo -e "${BLUE}Test 6: Dockerfile configuration${NC}"
if [[ -f "${REPO_ROOT}/dockerfiles/Dockerfile.nginx" ]]; then
    test_pass "Nginx Dockerfile exists"
else
    test_fail "Nginx Dockerfile missing"
fi

if grep -q "sha256:" "${REPO_ROOT}/dockerfiles/Dockerfile.nginx"; then
    test_pass "Nginx base image is pinned with SHA256"
else
    test_fail "Nginx base image not pinned"
fi

if grep -q "envsubst" "${REPO_ROOT}/dockerfiles/Dockerfile.nginx"; then
    test_pass "Dockerfile includes envsubst for config templating"
else
    test_fail "envsubst not found in Dockerfile"
fi

echo ""

# Test 7: SSL certificate directory
echo -e "${BLUE}Test 7: SSL certificate setup${NC}"
if [[ -d "${REPO_ROOT}/voice-bot/certs" ]]; then
    test_pass "SSL certs directory exists"
else
    test_fail "SSL certs directory missing"
fi

if [[ -f "${REPO_ROOT}/voice-bot/certs/README.md" ]]; then
    test_pass "SSL certs README exists"
else
    test_fail "SSL certs README missing"
fi

if [[ -f "${REPO_ROOT}/scripts/generate-ssl-certs.sh" ]]; then
    test_pass "SSL cert generation script exists"
else
    test_fail "SSL cert generation script missing"
fi

echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   Test Summary                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "  ${RED}Failed: ${TESTS_FAILED}${NC}"
echo ""

if [[ ${TESTS_FAILED} -eq 0 ]]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
