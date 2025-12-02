# Sentinel Telegram Alerting

## Overview

The Veris Memory Sentinel monitoring system now supports Telegram alerting for real-time notifications about system health, failures, and performance issues.

## Quick Start

### 1. Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Choose a name for your bot (e.g., "Veris Sentinel")
4. Choose a username (e.g., `veris_sentinel_bot`)
5. Copy the bot token provided by BotFather

### 2. Get Your Chat ID

1. Send any message to your new bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for `"chat":{"id": <YOUR_CHAT_ID>}` in the response
4. Copy this chat ID

### 3. Configure Environment

```bash
# Copy the template
cp .env.sentinel.template .env.sentinel

# Edit with your credentials
vim .env.sentinel

# Set these values:
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true
```

### 4. Test the Connection

```bash
# Run the test script
python scripts/test_telegram_alerts.py
```

## Alert Types

### Critical Alerts (🚨)
- Immediate Telegram notification
- GitHub issue auto-created
- Examples: Service down, security breaches, data corruption

### High Alerts (⚠️)
- Telegram notification
- Examples: Performance degradation, core functionality errors

### Warning Alerts (⚡)
- Included in daily summary
- Examples: Config drift, minor issues

### Info Alerts (ℹ️)
- Logged only
- Examples: Successful checks, status updates

## Message Formats

### Individual Alert
```
🚨 CRITICAL: Veris Memory Alert
━━━━━━━━━━━━━━━━━━━━━
Check: S5-security-negatives
Status: FAILED ❌
Time: 2025-08-19 23:45:00 UTC
Latency: 250.5ms

Message:
Unauthorized access detected

Details:
• attempts: 15
• source: 192.168.1.100

Action Required: Immediate investigation
━━━━━━━━━━━━━━━━━━━━━
```

### Daily Summary
```
📊 Veris Sentinel Report
━━━━━━━━━━━━━━━━━━━━━
Period: Last 24 hours
Total Checks: 1,440
✅ Passed: 1,420 (98.6%)
❌ Failed: 20 (1.4%)

Top Issues:
1. S8-capacity-smoke: 10 failures
2. S3-paraphrase-robustness: 5 failures

Avg Response Time: 45.2ms
Uptime: 98.6%
━━━━━━━━━━━━━━━━━━━━━
```

## Configuration Options

### Core Settings
- `TELEGRAM_BOT_TOKEN`: Your bot's API token
- `TELEGRAM_CHAT_ID`: Target chat/channel ID
- `TELEGRAM_ENABLED`: Enable/disable alerting
- `TELEGRAM_RATE_LIMIT`: Max messages per minute (default: 30)

### Alert Management
- `ALERT_DEDUP_WINDOW_MIN`: Deduplication window (default: 30)
- `ALERT_THRESHOLD_FAILURES`: Failures before alerting (default: 3)
- `SUMMARY_INTERVAL_HOURS`: Summary frequency (default: 24)

### GitHub Integration (Optional)
- `GITHUB_TOKEN`: Personal access token
- `GITHUB_REPO`: Repository for issues
- `GITHUB_ISSUES_ENABLED`: Enable issue creation

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Sentinel       │────▶│  Alert Manager   │────▶│ Telegram Bot │
│  Checks         │     │  (Deduplication) │     │  API         │
└─────────────────┘     └──────────────────┘     └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  GitHub API  │
                        └──────────────┘
```

## Key Features

### Alert Deduplication
- Prevents duplicate alerts within 30-minute window
- Uses message hashing for comparison
- Configurable window size

### Rate Limiting
- Default: 30 messages per minute
- Automatic queueing when limit reached
- Background queue processing

### Severity-Based Routing
- Critical: Telegram + GitHub issue
- High: Telegram only
- Warning: Daily summary
- Info: Logs only

### Failure Threshold
- Configurable number of failures before alerting
- Default: 3 failures trigger alert
- Per-check tracking

## Troubleshooting

### Bot Not Responding
1. Verify bot token is correct
2. Check bot is not blocked
3. Ensure chat ID is valid
4. Test with: `curl https://api.telegram.org/bot<TOKEN>/getMe`

### Messages Not Received
1. Check rate limiting (30/min default)
2. Verify deduplication window
3. Check failure threshold setting
4. Review logs for errors

### GitHub Issues Not Created
1. Verify GitHub token has repo/issues permissions
2. Check repository exists and is accessible
3. Ensure GITHUB_ISSUES_ENABLED=true
4. Only critical alerts create issues

## API Integration

### Python Usage
```python
from src.monitoring.sentinel.telegram_alerter import TelegramAlerter, AlertSeverity

alerter = TelegramAlerter(bot_token, chat_id)

# Send alert
await alerter.send_alert(
    check_id="S1-health",
    status="fail",
    message="Service unavailable",
    severity=AlertSeverity.CRITICAL,
    details={"error": "Connection timeout"}
)

# Send summary
await alerter.send_summary(
    period_hours=24,
    total_checks=1440,
    passed_checks=1400,
    failed_checks=40,
    top_failures=[],
    avg_latency_ms=50.0,
    uptime_percent=97.2
)
```

### Alert Manager Usage
```python
from src.monitoring.sentinel.alert_manager import AlertManager
from src.monitoring.sentinel.models import CheckResult

manager = AlertManager(
    telegram_token=token,
    telegram_chat_id=chat_id,
    github_token=github_token,
    github_repo="credentum/veris-memory"
)

# Process check result
result = CheckResult(
    check_id="S5-security",
    status="fail",
    message="Security breach detected",
    ...
)

await manager.process_check_result(result)
```

## Security Considerations

1. **Never commit tokens**: Use environment variables
2. **Rotate tokens regularly**: Update bot token periodically
3. **Limit chat access**: Use private chats or groups
4. **Monitor rate limits**: Prevent API abuse
5. **Secure configuration**: Protect .env.sentinel file

## Next Steps

- Phase 2: Implement critical checks (S5, S6, S8)
- Phase 3: Add monitoring checks (S4, S7)
- Phase 4: Advanced semantic checks (S3, S9, S10)

## Support

For issues or questions:
- Check logs: `/var/log/sentinel/`
- Review configuration: `.env.sentinel`
- Test connection: `scripts/test_telegram_alerts.py`
- GitHub issues: https://github.com/credentum/veris-memory/issues
