# Veris Smoke Test - Quick Setup

## Files
- `veris_smoke_report.schema.json` ← JSON schema validation
- `veris-smoke-60s.yaml` ← Suite specification  
- `smoke_runner.py` ← Minimal 60-second runner
- `deploy_guard.py` ← Gate logic (GREEN/RED decisions)

## Environment Setup
```bash
export VERIS_BASE_URL="http://127.0.0.1:8000"   # or your prod URL
export VERIS_NAMESPACE="smoke"
export VERIS_TIMEOUT_MS=60000
export VERIS_TOKEN="(if you use auth)"          # optional
```

## Usage

### Local Testing
```bash
# Run smoke test
python ops/smoke/smoke_runner.py

# Check gate logic (exits non-zero on RED)
python ops/smoke/deploy_guard.py /tmp/veris_smoke_report.json
```

### Production Deployment
```bash
# Single command validation
python ops/smoke/smoke_runner.py && \
python ops/smoke/deploy_guard.py /tmp/veris_smoke_report.json
```

### Cron/Systemd (Daily 6am)
```bash
# /etc/cron.d/veris-smoke
0 6 * * * root /usr/bin/python /srv/veris/ops/smoke/smoke_runner.py && \
               /usr/bin/python /srv/veris/ops/smoke/deploy_guard.py /tmp/veris_smoke_report.json
```

## Test Coverage (SM-1 to SM-6)
- **SM-1**: Health probe (API + deps)
- **SM-2**: Store→index→count (ingestion)
- **SM-3**: Needle retrieval (semantic)
- **SM-4**: Paraphrase robustness 
- **SM-5**: Index freshness (<1s)
- **SM-6**: SLO validation

## SLO Thresholds
- P95 Latency ≤ 300ms
- Error Rate ≤ 0.5%
- Recovery ≥ 95%
- Index Freshness ≤ 1s

## Store Reports Back to Veris (Optional)
```bash
curl -X POST "$VERIS_BASE_URL/tools/store_context" \
  -H "Content-Type: application/json" \
  -d @/tmp/veris_smoke_report.json
```

## Output
- **GREEN**: All tests pass, SLOs met → Deploy safe
- **RED**: Test failures or SLO violations → Block deploy
- Report saved to `/tmp/veris_smoke_report.json`