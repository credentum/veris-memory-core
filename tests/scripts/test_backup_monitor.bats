#!/usr/bin/env bats
# Tests for backup-monitor.sh

setup() {
    export BACKUP_LOG="/tmp/test-backup-cron.log"
    export ALERT_LOG="/tmp/test-backup-monitor.log"
    mkdir -p /tmp/backup

    # Create mock log file
    cat > "$BACKUP_LOG" << EOF
[2025-11-06 01:00:00] [INFO] Starting backup
[2025-11-06 01:05:00] [INFO] Successfully backed up volume1
[2025-11-06 01:10:00] [INFO] backup completed
EOF
}

teardown() {
    rm -f "$BACKUP_LOG" "$ALERT_LOG"
    rm -rf /tmp/backup
}

@test "Script has executable permissions" {
    [ -x "scripts/backup-monitor.sh" ]
}

@test "Script has proper shebang" {
    run head -n 1 scripts/backup-monitor.sh
    [[ "$output" =~ "#!/bin/bash" ]]
}

@test "Script passes shellcheck" {
    if command -v shellcheck &>/dev/null; then
        run shellcheck -x scripts/backup-monitor.sh
        [ "$status" -eq 0 ]
    else
        skip "shellcheck not installed"
    fi
}

@test "Script has no hardcoded passwords" {
    run grep -i "password.*=" scripts/backup-monitor.sh
    # Should not find hardcoded password values
    [[ ! "$output" =~ "PASSWORD='[a-zA-Z0-9]" ]]
    [[ ! "$output" =~ 'PASSWORD="[a-zA-Z0-9]' ]]
}

@test "Script has secure credential documentation" {
    run head -n 30 scripts/backup-monitor.sh
    [[ "$output" =~ "SECURE CREDENTIAL" ]] || [[ "$output" =~ "TELEGRAM_BOT_TOKEN" ]]
    [[ "$output" =~ "/etc/backup" ]] || [[ "$output" =~ "environment" ]]
}

@test "Telegram credentials loaded from config file if available" {
    run grep -A 5 "telegram.conf" scripts/backup-monitor.sh
    [[ "$output" =~ "source" ]]
    [ "$status" -eq 0 ]
}

@test "Script handles missing Telegram credentials gracefully" {
    unset TELEGRAM_BOT_TOKEN
    unset TELEGRAM_CHAT_ID

    # Source the send_telegram_alert function
    source <(sed -n '/^send_telegram_alert()/,/^}/p' scripts/backup-monitor.sh)

    run send_telegram_alert "Test message"
    # Should not fail, just skip sending
    [ "$status" -eq 0 ]
}

@test "Disk usage threshold is configurable" {
    run grep "DISK_USAGE_THRESHOLD" scripts/backup-monitor.sh
    [[ "$output" =~ "DISK_USAGE_THRESHOLD=" ]]
    [ "$status" -eq 0 ]
}

@test "Backup failure threshold is configurable" {
    run grep "BACKUP_FAILURE_THRESHOLD" scripts/backup-monitor.sh
    [[ "$output" =~ "BACKUP_FAILURE_THRESHOLD=" ]]
    [ "$status" -eq 0 ]
}

@test "check_disk_usage function exists" {
    run grep "check_disk_usage()" scripts/backup-monitor.sh
    [ "$status" -eq 0 ]
}

@test "check_backup_status function exists" {
    run grep "check_backup_status()" scripts/backup-monitor.sh
    [ "$status" -eq 0 ]
}

@test "check_retention function exists" {
    run grep "check_retention()" scripts/backup-monitor.sh
    [ "$status" -eq 0 ]
}

@test "generate_daily_summary function exists" {
    run grep "generate_daily_summary()" scripts/backup-monitor.sh
    [ "$status" -eq 0 ]
}

@test "Log functions include timestamps" {
    source <(sed -n '/^log()/,/^}/p' scripts/backup-monitor.sh)

    run log "Test message"
    [[ "$output" =~ "[INFO]" ]]
    [[ "$output" =~ "Test message" ]]
    # Should contain date format YYYY-MM-DD
    [[ "$output" =~ [0-9]{4}-[0-9]{2}-[0-9]{2} ]]
}

@test "Error logging distinguishes error levels" {
    source <(sed -n '/^log()/,/^}/p' scripts/backup-monitor.sh)
    source <(sed -n '/^warning()/,/^}/p' scripts/backup-monitor.sh)
    source <(sed -n '/^error()/,/^}/p' scripts/backup-monitor.sh)

    run log "Info message"
    [[ "$output" =~ "INFO" ]]

    run warning "Warning message"
    [[ "$output" =~ "WARN" ]]

    run error "Error message"
    [[ "$output" =~ "ERROR" ]]
}

@test "Telegram alert priorities are supported" {
    run grep -A 10 "send_telegram_alert" scripts/backup-monitor.sh
    [[ "$output" =~ "critical\|warning\|info" ]] || [[ "$output" =~ "priority" ]]
}

@test "Script monitors multiple partitions" {
    run grep -A 5 "partitions=" scripts/backup-monitor.sh
    [[ "$output" =~ "/backup" ]]
    [[ "$output" =~ "/" ]]
}

@test "Script checks for missing backup log" {
    source <(sed -n '/^check_backup_status()/,/^}/p' scripts/backup-monitor.sh)
    export BACKUP_LOG="/nonexistent/path/backup.log"

    run check_backup_status
    # Should handle missing log gracefully
    [[ "$output" =~ "not found" ]] || [ "$status" -ne 0 ]
}

@test "Retention policy checks old backups" {
    run grep -A 10 "check_retention" scripts/backup-monitor.sh
    [[ "$output" =~ "find.*mtime.*30" ]] || [[ "$output" =~ "retention" ]]
}

@test "Daily summary includes key metrics" {
    run grep -A 20 "generate_daily_summary" scripts/backup-monitor.sh
    [[ "$output" =~ "Disk Usage\|disk usage" ]]
    [[ "$output" =~ "Backup Status\|backup status" ]]
}
