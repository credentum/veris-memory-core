#!/usr/bin/env bats
# Tests for backup-large-volumes.sh

# Setup and teardown
setup() {
    # Source the script functions (without executing main)
    export RESTIC_PASSWORD="test-password"
    export RESTIC_REPOSITORY="/tmp/test-restic-repo"
    TEST_LOG="/tmp/test-backup.log"

    # Create mock functions for testing
    docker() {
        echo "mock-docker"
        return 0
    }
    export -f docker

    restic() {
        echo "mock-restic $@"
        return 0
    }
    export -f restic
}

teardown() {
    rm -f "$TEST_LOG"
    rm -rf "$RESTIC_REPOSITORY"
}

@test "Script requires RESTIC_PASSWORD environment variable" {
    unset RESTIC_PASSWORD
    run bash -c 'source scripts/backup-large-volumes.sh 2>&1 | head -5'
    [[ "$output" =~ "RESTIC_PASSWORD not set" ]] || [[ "$status" -eq 1 ]]
}

@test "validate_command accepts valid commands" {
    # Source the validate_command function
    source <(grep -A 10 "^validate_command()" scripts/backup-large-volumes.sh)

    run validate_command "backup"
    [ "$status" -eq 0 ]

    run validate_command "list"
    [ "$status" -eq 0 ]

    run validate_command "stats"
    [ "$status" -eq 0 ]
}

@test "validate_command rejects invalid commands" {
    source <(grep -A 10 "^validate_command()" scripts/backup-large-volumes.sh)

    run validate_command "rm -rf /"
    [ "$status" -eq 1 ]

    run validate_command "../../etc/passwd"
    [ "$status" -eq 1 ]
}

@test "Input sanitization removes dangerous characters" {
    # Test that dangerous input is sanitized
    run bash scripts/backup-large-volumes.sh "backup; rm -rf /"
    [[ ! "$output" =~ "rm -rf" ]]
}

@test "check_dependencies detects missing docker" {
    skip "Requires docker to not be installed"
}

@test "check_dependencies detects missing restic" {
    skip "Requires restic to not be installed"
}

@test "Script has executable permissions" {
    [ -x "scripts/backup-large-volumes.sh" ]
}

@test "Script has proper shebang" {
    run head -n 1 scripts/backup-large-volumes.sh
    [[ "$output" =~ "#!/bin/bash" ]]
}

@test "Script passes shellcheck" {
    if command -v shellcheck &>/dev/null; then
        run shellcheck -x scripts/backup-large-volumes.sh
        [ "$status" -eq 0 ]
    else
        skip "shellcheck not installed"
    fi
}

@test "Script has no hardcoded passwords" {
    run grep -i "password.*=" scripts/backup-large-volumes.sh
    # Should only find RESTIC_PASSWORD variable assignments, not hardcoded values
    [[ ! "$output" =~ "RESTIC_PASSWORD='[^$]" ]]
    [[ ! "$output" =~ 'RESTIC_PASSWORD="[^$]' ]]
}

@test "Log function creates log entries" {
    source <(grep -A 3 "^log()" scripts/backup-large-volumes.sh)

    run log "INFO" "Test message"
    [[ "$output" =~ "INFO" ]]
    [[ "$output" =~ "Test message" ]]
}

@test "Script handles missing volumes gracefully" {
    # This would require mocking docker volume ls
    skip "Requires docker mocking setup"
}

@test "Error messages are not suppressed" {
    # Ensure no 2>/dev/null without proper error handling
    run grep -n "2>/dev/null" scripts/backup-large-volumes.sh
    # Should have very few or none
    [ "${#lines[@]}" -lt 3 ]
}
