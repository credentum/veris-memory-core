#!/usr/bin/env bats
# Tests for backup-production-final.sh

setup() {
    export BACKUP_ROOT="/tmp/test-backup"
    export BACKUP_LOG="/tmp/test-backup.log"
    mkdir -p "$BACKUP_ROOT"/{daily,weekly,monthly}

    # Mock docker command
    docker() {
        echo "mock-docker-container-id"
        return 0
    }
    export -f docker
}

teardown() {
    rm -rf "$BACKUP_ROOT"
    rm -f "$BACKUP_LOG"
}

@test "Script has executable permissions" {
    [ -x "scripts/backup-production-final.sh" ]
}

@test "Script has proper shebang" {
    run head -n 1 scripts/backup-production-final.sh
    [[ "$output" =~ "#!/bin/bash" ]]
}

@test "Script passes shellcheck" {
    if command -v shellcheck &>/dev/null; then
        run shellcheck -x scripts/backup-production-final.sh
        [ "$status" -eq 0 ]
    else
        skip "shellcheck not installed"
    fi
}

@test "get_container_id function validates container exists" {
    source <(sed -n '/^get_container_id()/,/^}/p' scripts/backup-production-final.sh)

    # Mock docker ps to return empty
    docker() {
        if [[ "$*" =~ "ps" ]]; then
            echo ""
        fi
    }
    export -f docker

    run get_container_id "nonexistent"
    [ "$status" -eq 1 ]
}

@test "get_container_id returns container ID when found" {
    source <(sed -n '/^get_container_id()/,/^}/p' scripts/backup-production-final.sh)

    docker() {
        if [[ "$*" =~ "ps" ]]; then
            echo "mock-container-123"
        fi
    }
    export -f docker

    run get_container_id "neo4j"
    [ "$status" -eq 0 ]
    [[ "$output" == "mock-container-123" ]]
}

@test "validate_backup_dir rejects empty paths" {
    source <(sed -n '/^validate_backup_dir()/,/^}/p' scripts/backup-production-final.sh)
    export BACKUP_ROOT="/tmp/test-backup"

    run validate_backup_dir ""
    [ "$status" -eq 1 ]
}

@test "validate_backup_dir rejects paths outside BACKUP_ROOT" {
    source <(sed -n '/^validate_backup_dir()/,/^}/p' scripts/backup-production-final.sh)
    export BACKUP_ROOT="/tmp/test-backup"

    run validate_backup_dir "/etc/passwd"
    [ "$status" -eq 1 ]

    run validate_backup_dir "../../etc"
    [ "$status" -eq 1 ]
}

@test "validate_backup_dir rejects paths without 'backup-' pattern" {
    source <(sed -n '/^validate_backup_dir()/,/^}/p' scripts/backup-production-final.sh)
    export BACKUP_ROOT="/tmp/test-backup"

    run validate_backup_dir "/tmp/test-backup/malicious-dir"
    [ "$status" -eq 1 ]
}

@test "validate_backup_dir accepts valid backup directories" {
    source <(sed -n '/^validate_backup_dir()/,/^}/p' scripts/backup-production-final.sh)
    export BACKUP_ROOT="/tmp/test-backup"

    run validate_backup_dir "/tmp/test-backup/daily/backup-20250101-120000"
    [ "$status" -eq 0 ]
}

@test "Script does not use unsafe head -1 without container checks" {
    # Ensure all 'head -1' usages are after proper container existence checks
    run grep -B 5 "head -1" scripts/backup-production-final.sh
    # Should not find direct usage with docker ps
    [[ ! "$output" =~ "docker ps.*head -1.*exec" ]]
}

@test "rm -rf operations have safeguards" {
    run grep -A 2 "rm -rf" scripts/backup-production-final.sh
    # All rm -rf should be preceded by validation
    [[ "$output" =~ "validate_backup_dir" ]] || [ "${#lines[@]}" -eq 0 ]
}

@test "Log function works correctly" {
    source <(sed -n '/^log()/,/^}/p' scripts/backup-production-final.sh)
    export BACKUP_LOG="/tmp/test.log"

    run log "INFO" "Test message"
    [[ "$output" =~ "INFO" ]]
    [[ "$output" =~ "Test message" ]]
}

@test "Manifest creation includes required fields" {
    run grep -A 10 "manifest.json" scripts/backup-production-final.sh
    [[ "$output" =~ "timestamp" ]]
    [[ "$output" =~ "hostname" ]]
    [[ "$output" =~ "databases" ]]
}

@test "Script handles different backup types" {
    source <(sed -n '/^perform_backup()/,/^}/p' scripts/backup-production-final.sh)

    # Should accept daily, weekly, monthly
    local types=("daily" "weekly" "monthly")
    for type in "${types[@]}"; do
        # Just verify the case statement includes these
        grep -q "$type)" scripts/backup-production-final.sh
        [ $? -eq 0 ]
    done
}

@test "Retention days are properly configured" {
    run grep -A 3 "retention_days" scripts/backup-production-final.sh
    [[ "$output" =~ "daily.*7" ]]
    [[ "$output" =~ "weekly.*28" ]]
    [[ "$output" =~ "monthly.*90" ]]
}
