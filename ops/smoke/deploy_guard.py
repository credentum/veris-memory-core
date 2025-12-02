import json, sys

def gate(report_path: str) -> int:
  r = json.load(open(report_path))
  s, t = r["summary"], r.get("thresholds", {})

  # thresholds with sane defaults
  thr_lat = t.get("p95_latency_ms", 300)
  thr_err = t.get("error_rate_pct", 0.5)
  thr_rec = t.get("recovery_top1_pct", 95)
  thr_idx = t.get("index_freshness_s", 1)

  checks = [
    s["overall_status"] == "pass",
    s["p95_latency_ms"] <= thr_lat,
    s["error_rate_pct"] <= thr_err,
    s["recovery_top1_pct"] >= thr_rec,
    s["index_freshness_s"] <= thr_idx,
    len(s["failed_tests"]) == 0
  ]

  ok = all(checks)
  print("DEPLOY:", "GREEN" if ok else "RED")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(gate(sys.argv[1]))