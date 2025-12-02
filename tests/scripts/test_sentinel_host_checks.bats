#!/usr/bin/env bats
# Tests for sentinel-host-checks.sh

setup() {
    export TEST_OUTPUT="/tmp/test-sentinel-output.json"
    export PATH="$BATS_TEST_DIRNAME/mocks:$PATH"
}

teardown() {
    rm -f "$TEST_OUTPUT"
    rm -rf "$BATS_TEST_DIRNAME/mocks"
}

@test "Script has executable permissions" {
    [ -x "scripts/sentinel-host-checks.sh" ]
}

@test "Script has proper shebang" {
    run head -n 1 scripts/sentinel-host-checks.sh
    [[ "$output" =~ "#!/bin/bash" ]]
}

@test "Script passes shellcheck" {
    if command -v shellcheck &>/dev/null; then
        run shellcheck -x scripts/sentinel-host-checks.sh
        [ "$status" -eq 0 ]
    else
        skip "shellcheck not installed"
    fi
}

@test "Script has no hardcoded secrets" {
    run grep -i "password.*=" scripts/sentinel-host-checks.sh
    # Should not find hardcoded password values
    [[ ! "$output" =~ "PASSWORD='[a-zA-Z0-9]" ]]
    [[ ! "$output" =~ 'PASSWORD="[a-zA-Z0-9]' ]]
}

@test "Script has no hardcoded API keys" {
    run grep -iE "(api_key|token|secret).*=" scripts/sentinel-host-checks.sh
    # Should not find hardcoded credentials (exclude variable declarations)
    [[ ! "$output" =~ "API_KEY='[a-zA-Z0-9]" ]]
    [[ ! "$output" =~ 'TOKEN="[a-zA-Z0-9]' ]]
}

@test "Script requires sudo for UFW commands" {
    run grep "sudo ufw" scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
    [[ "$output" =~ "sudo ufw" ]]
}

@test "Script validates sudo privileges early" {
    run grep -E "sudo -n true|check.*sudo|require.*sudo" scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
}

@test "Script checks for required dependencies (curl, jq)" {
    run grep -E "command -v (curl|jq)" scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
}

@test "Dry-run mode does not submit to API" {
    # Create mock UFW command
    mkdir -p "$BATS_TEST_DIRNAME/mocks"
    cat > "$BATS_TEST_DIRNAME/mocks/ufw" << 'EOF'
#!/bin/bash
echo "Status: active"
echo "To                         Action      From"
echo "--                         ------      ----"
echo "22/tcp                     ALLOW       Anywhere"
echo "80/tcp                     ALLOW       Anywhere"
echo "443/tcp                    ALLOW       Anywhere"
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/ufw"

    # Create mock sudo that calls our mock ufw
    cat > "$BATS_TEST_DIRNAME/mocks/sudo" << 'EOF'
#!/bin/bash
# Mock sudo - just pass through to the command
if [[ "$1" == "-n" && "$2" == "true" ]]; then
    exit 0
fi
if [[ "$1" == "ufw" ]]; then
    ufw "${@:2}"
else
    "$@"
fi
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/sudo"

    # Mock curl to detect if it gets called
    cat > "$BATS_TEST_DIRNAME/mocks/curl" << 'EOF'
#!/bin/bash
# Dry-run mode should NOT reach here
echo "ERROR: curl was called in dry-run mode" >&2
exit 1
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/curl"

    # Run in dry-run mode
    run scripts/sentinel-host-checks.sh --dry-run

    # Should succeed (status 0) without calling curl
    [ "$status" -eq 0 ]
    [[ ! "$output" =~ "ERROR: curl was called" ]]
    [[ "$output" =~ "DRY RUN" ]]
}

@test "Script generates valid JSON payload" {
    # Create mock commands
    mkdir -p "$BATS_TEST_DIRNAME/mocks"

    cat > "$BATS_TEST_DIRNAME/mocks/ufw" << 'EOF'
#!/bin/bash
echo "Status: active"
echo "22/tcp                     ALLOW       Anywhere"
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/ufw"

    cat > "$BATS_TEST_DIRNAME/mocks/sudo" << 'EOF'
#!/bin/bash
if [[ "$1" == "-n" && "$2" == "true" ]]; then
    exit 0
fi
if [[ "$1" == "ufw" ]]; then
    ufw "${@:2}"
else
    "$@"
fi
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/sudo"

    # Run in dry-run to capture JSON
    run scripts/sentinel-host-checks.sh --dry-run

    # Extract JSON from output (between "Would submit" and end)
    json_output=$(echo "$output" | sed -n '/^{/,/^}/p')

    # Validate JSON structure with jq
    if command -v jq &>/dev/null; then
        echo "$json_output" | jq -e '.check_id' >/dev/null
        echo "$json_output" | jq -e '.timestamp' >/dev/null
        echo "$json_output" | jq -e '.status' >/dev/null
        echo "$json_output" | jq -e '.message' >/dev/null
        echo "$json_output" | jq -e '.details' >/dev/null
    fi
}

@test "Script handles UFW not installed gracefully" {
    # Create mock that simulates UFW not installed
    mkdir -p "$BATS_TEST_DIRNAME/mocks"

    cat > "$BATS_TEST_DIRNAME/mocks/sudo" << 'EOF'
#!/bin/bash
exit 0
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/sudo"

    # Create mock command that returns "not found" for ufw
    cat > "$BATS_TEST_DIRNAME/mocks/command" << 'EOF'
#!/bin/bash
if [[ "$2" == "ufw" ]]; then
    exit 1
fi
exit 0
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/command"

    # Mock curl to capture the JSON payload
    cat > "$BATS_TEST_DIRNAME/mocks/curl" << 'EOF'
#!/bin/bash
# Extract the JSON data from -d argument
for arg in "$@"; do
    if [[ "$arg" =~ ^\{.*\}$ ]]; then
        echo "$arg" > /tmp/test-sentinel-output.json
    fi
done
echo '{"success": true}'
EOF
    chmod +x "$BATS_TEST_DIRNAME/mocks/curl"

    # Override command builtin behavior
    export -f command 2>/dev/null || true

    # Run script - it should handle missing UFW
    run scripts/sentinel-host-checks.sh

    # Should complete (may fail or warn, but not crash)
    # Status can be 0 or 1 depending on whether UFW is required
    [[ "$status" -eq 0 ]] || [[ "$status" -eq 1 ]]
}

@test "Script validates JSON payload before submission" {
    run grep -E "jq.*\." scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
}

@test "Script uses localhost for Sentinel API URL" {
    run grep "localhost:9090" scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
    [[ "$output" =~ "localhost:9090" ]]
}

@test "Script has error handling for API submission failure" {
    run grep -A 5 "curl.*POST" scripts/sentinel-host-checks.sh
    [[ "$output" =~ "success" ]] || [[ "$output" =~ "error" ]]
}

@test "Script sets appropriate exit codes" {
    # Check that script uses exit commands
    run grep "exit" scripts/sentinel-host-checks.sh
    [ "$status" -eq 0 ]
    [[ "$output" =~ "exit 0" ]] || [[ "$output" =~ "exit 1" ]]
}

@test "JSON payload includes all required fields" {
    mkdir -p "$BATS_TEST_DIRNAME/mocks"

    # Mock all dependencies
    for cmd in sudo ufw curl jq date; do
        cat > "$BATS_TEST_DIRNAME/mocks/$cmd" << 'EOF'
#!/bin/bash
case "$cmd" in
    ufw) echo "Status: active" ;;
    date) echo "2025-11-15T18:00:00Z" ;;
    jq) cat ;;
    sudo) if [[ "$1" == "ufw" ]]; then echo "Status: active"; fi ;;
    curl) echo '{"success": true}' ;;
esac
EOF
        chmod +x "$BATS_TEST_DIRNAME/mocks/$cmd"
    done

    # Run in dry-run mode
    run scripts/sentinel-host-checks.sh --dry-run

    # Check output contains required fields
    [[ "$output" =~ "check_id" ]]
    [[ "$output" =~ "timestamp" ]]
    [[ "$output" =~ "status" ]]
    [[ "$output" =~ "message" ]]
    [[ "$output" =~ "details" ]]
}
