#!/bin/bash
# Initialize Neo4j Schema
# This script ensures Neo4j has the required schema (constraints and indexes)
# Can be run manually or as part of automated deployment

set -e

# Configuration
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-veris-memory-dev-neo4j-1}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "🔧 Neo4j Schema Initialization"
echo "================================"

# Validate NEO4J_PASSWORD
if [ -z "$NEO4J_PASSWORD" ]; then
    echo -e "${RED}❌ ERROR: NEO4J_PASSWORD environment variable is not set${NC}"
    echo "Please set NEO4J_PASSWORD before running this script."
    exit 1
fi

# Validate password format (basic security checks)
PASSWORD_LENGTH=${#NEO4J_PASSWORD}
if [ "$PASSWORD_LENGTH" -lt 8 ]; then
    echo -e "${YELLOW}⚠️  WARNING: Password is less than 8 characters${NC}"
    echo "For production, use a password with at least 8 characters."
fi

# Check for special characters that might need escaping in Cypher
if echo "$NEO4J_PASSWORD" | grep -q '["\$`\\]'; then
    echo -e "${YELLOW}⚠️  WARNING: Password contains special characters that may need escaping${NC}"
fi

echo -e "${GREEN}✅ Password validation passed${NC}"
echo ""

# Check if running in Docker environment or local
if docker ps --format '{{.Names}}' | grep -q "$NEO4J_CONTAINER"; then
    echo "✅ Detected Docker environment"
    DOCKER_MODE=true
else
    echo "ℹ️  Running in local mode (no Docker container detected)"
    DOCKER_MODE=false
fi

# Function to execute Cypher via docker
# Uses environment variable passing to avoid password exposure in process listings
execute_cypher_docker() {
    local query="$1"
    docker exec -e NEO4J_PASSWORD="$NEO4J_PASSWORD" "$NEO4J_CONTAINER" \
        sh -c "cypher-shell -u \"$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"$query\""
}

# Function to execute Cypher via Python (fallback)
# Uses separate Python script for better testability
execute_cypher_python() {
    local query="$1"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 "$SCRIPT_DIR/init_neo4j_schema.py" --query "$query"
}

# Determine execution method
if [ "$DOCKER_MODE" = true ]; then
    EXEC_FN=execute_cypher_docker
else
    EXEC_FN=execute_cypher_python
fi

echo ""
echo "📊 Checking Neo4j connection..."
if [ "$DOCKER_MODE" = true ]; then
    # Use environment variable passing to avoid password exposure
    if docker exec -e NEO4J_PASSWORD="$NEO4J_PASSWORD" "$NEO4J_CONTAINER" \
        sh -c "cypher-shell -u \"$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"RETURN 1\"" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected to Neo4j${NC}"
    else
        echo -e "${RED}❌ Failed to connect to Neo4j${NC}"
        exit 1
    fi
fi

echo ""
echo "🏗️  Creating schema constraints and indexes..."
echo ""

# Read and execute the Cypher init script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CYPHER_FILE="$PROJECT_ROOT/deployments/neo4j-init/001-init-schema.cypher"

if [ -f "$CYPHER_FILE" ]; then
    echo "📄 Loading schema from: $CYPHER_FILE"

    if [ "$DOCKER_MODE" = true ]; then
        # Copy file to container and execute (using env var for password security)
        docker cp "$CYPHER_FILE" "$NEO4J_CONTAINER":/tmp/init-schema.cypher
        docker exec -e NEO4J_PASSWORD="$NEO4J_PASSWORD" "$NEO4J_CONTAINER" \
            sh -c "cypher-shell -u \"$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -f /tmp/init-schema.cypher"
        docker exec "$NEO4J_CONTAINER" rm /tmp/init-schema.cypher
    else
        # Execute via Python using separate script
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        python3 "$SCRIPT_DIR/init_neo4j_schema.py" --file "$CYPHER_FILE"
    fi
else
    echo -e "${YELLOW}⚠️  Cypher file not found, creating schema programmatically${NC}"

    # Fallback: Create schema using individual commands
    CONSTRAINTS=(
        "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (c:Context) REQUIRE c.id IS UNIQUE"
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        "CREATE CONSTRAINT sprint_id_unique IF NOT EXISTS FOR (s:Sprint) REQUIRE s.id IS UNIQUE"
        "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE"
        "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
    )

    for constraint in "${CONSTRAINTS[@]}"; do
        echo "  Creating constraint..."
        $EXEC_FN "$constraint" || echo "    (may already exist)"
    done

    INDEXES=(
        "CREATE INDEX context_type_idx IF NOT EXISTS FOR (c:Context) ON (c.type)"
        "CREATE INDEX context_created_at_idx IF NOT EXISTS FOR (c:Context) ON (c.created_at)"
        "CREATE INDEX context_author_idx IF NOT EXISTS FOR (c:Context) ON (c.author)"
        "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.document_type)"
    )

    for index in "${INDEXES[@]}"; do
        echo "  Creating index..."
        $EXEC_FN "$index" || echo "    (may already exist)"
    done
fi

echo ""
echo "🔍 Verifying schema..."
if [ "$DOCKER_MODE" = true ]; then
    # Use environment variable passing for password security
    CONSTRAINT_COUNT=$(docker exec -e NEO4J_PASSWORD="$NEO4J_PASSWORD" "$NEO4J_CONTAINER" \
        sh -c "cypher-shell -u \"$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"CALL db.constraints() YIELD name RETURN count(name) as count\"" | grep -oE '[0-9]+' | head -1)
    INDEX_COUNT=$(docker exec -e NEO4J_PASSWORD="$NEO4J_PASSWORD" "$NEO4J_CONTAINER" \
        sh -c "cypher-shell -u \"$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"CALL db.indexes() YIELD name RETURN count(name) as count\"" | grep -oE '[0-9]+' | head -1)

    echo -e "${GREEN}✅ Constraints: $CONSTRAINT_COUNT${NC}"
    echo -e "${GREEN}✅ Indexes: $INDEX_COUNT${NC}"
fi

echo ""
echo -e "${GREEN}✅ Neo4j schema initialization completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart context-store service: docker compose restart context-store"
echo "  2. Verify Context label exists: docker exec $NEO4J_CONTAINER cypher-shell -u $NEO4J_USER -p \$NEO4J_PASSWORD 'CALL db.labels()'"
echo ""
