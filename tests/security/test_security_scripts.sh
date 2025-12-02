#!/bin/bash
#
# SECURITY SCRIPTS TESTS
# Purpose: Test security enhancement scripts and configurations
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
echo -e "${BLUE}║           Security Scripts Tests                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test 1: Security scripts exist
echo -e "${BLUE}Test 1: Security scripts exist${NC}"
if [[ -f "${REPO_ROOT}/scripts/apply-security-enhancements.sh" ]]; then
    test_pass "apply-security-enhancements.sh exists"
else
    test_fail "apply-security-enhancements.sh missing"
fi

if [[ -f "${REPO_ROOT}/scripts/generate-ssl-certs.sh" ]]; then
    test_pass "generate-ssl-certs.sh exists"
else
    test_fail "generate-ssl-certs.sh missing"
fi

if [[ -f "${REPO_ROOT}/scripts/security/docker-firewall-rules.sh" ]]; then
    test_pass "docker-firewall-rules.sh exists"
else
    test_fail "docker-firewall-rules.sh missing"
fi

if [[ -f "${REPO_ROOT}/scripts/security/security-audit.sh" ]]; then
    test_pass "security-audit.sh exists"
else
    test_fail "security-audit.sh missing"
fi

echo ""

# Test 2: Scripts are executable
echo -e "${BLUE}Test 2: Scripts are executable${NC}"
if [[ -x "${REPO_ROOT}/scripts/apply-security-enhancements.sh" ]]; then
    test_pass "apply-security-enhancements.sh is executable"
else
    test_fail "apply-security-enhancements.sh is not executable"
fi

if [[ -x "${REPO_ROOT}/scripts/generate-ssl-certs.sh" ]]; then
    test_pass "generate-ssl-certs.sh is executable"
else
    test_fail "generate-ssl-certs.sh is not executable"
fi

if [[ -x "${REPO_ROOT}/scripts/security/docker-firewall-rules.sh" ]]; then
    test_pass "docker-firewall-rules.sh is executable"
else
    test_fail "docker-firewall-rules.sh is not executable"
fi

echo ""

# Test 3: Crontab backup logic
echo -e "${BLUE}Test 3: Crontab backup logic${NC}"
if grep -q "CRONTAB_BACKUP" "${REPO_ROOT}/scripts/apply-security-enhancements.sh"; then
    test_pass "Crontab backup logic implemented"
else
    test_fail "Crontab backup logic missing"
fi

if grep -q "mkdir -p.*crontab" "${REPO_ROOT}/scripts/apply-security-enhancements.sh"; then
    test_pass "Crontab backup directory creation"
else
    test_fail "Crontab backup directory creation missing"
fi

echo ""

# Test 4: Pre-commit configuration
echo -e "${BLUE}Test 4: Pre-commit configuration${NC}"
if [[ -f "${REPO_ROOT}/.pre-commit-config-security.yaml" ]]; then
    test_pass ".pre-commit-config-security.yaml exists"
else
    test_fail ".pre-commit-config-security.yaml missing"
fi

if grep -q "pre-commit-config-security.yaml" "${REPO_ROOT}/scripts/apply-security-enhancements.sh"; then
    test_pass "Script references pre-commit config"
else
    test_fail "Script doesn't reference pre-commit config"
fi

echo ""

# Test 5: Documentation exists
echo -e "${BLUE}Test 5: Documentation${NC}"
if [[ -f "${REPO_ROOT}/docs/SECURITY_ENHANCEMENTS.md" ]]; then
    test_pass "SECURITY_ENHANCEMENTS.md exists"
else
    test_fail "SECURITY_ENHANCEMENTS.md missing"
fi

if [[ -f "${REPO_ROOT}/docs/MCP_DOCKER_BRIDGE_FIX.md" ]]; then
    test_pass "MCP_DOCKER_BRIDGE_FIX.md exists"
else
    test_fail "MCP_DOCKER_BRIDGE_FIX.md missing"
fi

echo ""

# Test 6: SSL certificate generation
echo -e "${BLUE}Test 6: SSL certificate generation${NC}"
if grep -q "self-signed" "${REPO_ROOT}/scripts/generate-ssl-certs.sh"; then
    test_pass "Self-signed certificate generation supported"
else
    test_fail "Self-signed certificate generation not found"
fi

if grep -q "letsencrypt" "${REPO_ROOT}/scripts/generate-ssl-certs.sh"; then
    test_pass "Let's Encrypt support included"
else
    test_fail "Let's Encrypt support missing"
fi

if grep -q "openssl" "${REPO_ROOT}/scripts/generate-ssl-certs.sh"; then
    test_pass "OpenSSL commands present"
else
    test_fail "OpenSSL commands missing"
fi

echo ""

# Test 7: Docker firewall rules
echo -e "${BLUE}Test 7: Docker firewall rules${NC}"
if grep -q "DOCKER-USER" "${REPO_ROOT}/scripts/security/docker-firewall-rules.sh"; then
    test_pass "DOCKER-USER chain configured"
else
    test_fail "DOCKER-USER chain not found"
fi

if grep -q "iptables" "${REPO_ROOT}/scripts/security/docker-firewall-rules.sh"; then
    test_pass "Iptables rules configured"
else
    test_fail "Iptables rules missing"
fi

if grep -q "backup" "${REPO_ROOT}/scripts/security/docker-firewall-rules.sh"; then
    test_pass "Iptables backup logic present"
else
    test_fail "Iptables backup logic missing"
fi

echo ""

# Test 8: Error handling
echo -e "${BLUE}Test 8: Error handling${NC}"
if grep -q "set -euo pipefail" "${REPO_ROOT}/scripts/apply-security-enhancements.sh"; then
    test_pass "Strict error handling enabled (apply-security-enhancements.sh)"
else
    test_fail "Strict error handling not enabled (apply-security-enhancements.sh)"
fi

if grep -q "set -euo pipefail" "${REPO_ROOT}/scripts/generate-ssl-certs.sh"; then
    test_pass "Strict error handling enabled (generate-ssl-certs.sh)"
else
    test_fail "Strict error handling not enabled (generate-ssl-certs.sh)"
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
