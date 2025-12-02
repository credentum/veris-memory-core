#!/bin/bash
#
# RUN ALL TESTS
# Purpose: Run all security enhancement tests
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_PASSED=0
TOTAL_FAILED=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Security Enhancements - Test Suite                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run nginx tests
echo -e "${YELLOW}Running nginx configuration tests...${NC}"
echo ""
if bash "${SCRIPT_DIR}/nginx/test_nginx_config.sh"; then
    echo ""
    echo -e "${GREEN}✅ Nginx tests passed${NC}"
else
    echo ""
    echo -e "${RED}❌ Nginx tests failed${NC}"
    ((TOTAL_FAILED++))
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Run security script tests
echo -e "${YELLOW}Running security scripts tests...${NC}"
echo ""
if bash "${SCRIPT_DIR}/security/test_security_scripts.sh"; then
    echo ""
    echo -e "${GREEN}✅ Security scripts tests passed${NC}"
else
    echo ""
    echo -e "${RED}❌ Security scripts tests failed${NC}"
    ((TOTAL_FAILED++))
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Overall summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║               Overall Test Summary                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ ${TOTAL_FAILED} -eq 0 ]]; then
    echo -e "${GREEN}✅ All test suites passed!${NC}"
    echo ""
    echo "Tests verified:"
    echo "  • Nginx configuration and rate limiting"
    echo "  • Security scripts and automation"
    echo "  • SSL certificate setup"
    echo "  • Docker security configuration"
    echo ""
    exit 0
else
    echo -e "${RED}❌ ${TOTAL_FAILED} test suite(s) failed${NC}"
    echo ""
    echo "Please review the failed tests above."
    echo ""
    exit 1
fi
