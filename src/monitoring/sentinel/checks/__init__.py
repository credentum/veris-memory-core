"""
Sentinel Check Modules

This package contains all the individual check implementations,
split from the original monolithic file for better maintainability.

Each check is responsible for a specific aspect of Veris Memory monitoring:
- S1: Health Probes (system liveness/readiness)
- S2: Golden Fact Recall (data integrity)
- S3: Paraphrase Robustness (semantic consistency)
- S4: Metrics Wiring (monitoring infrastructure)
- S5: Security RBAC (access control)
- S6: Backup/Restore (data protection)
- S7: Configuration Parity (deployment consistency)
- S8: Capacity Smoke (performance limits)
- S9: Graph Intent Validation (query correctness)
- S10: Content Pipeline (data processing)
- S11: Firewall Status (security infrastructure)
"""

from .s1_health_probes import VerisHealthProbe
from .s2_golden_fact_recall import GoldenFactRecall
from .s3_paraphrase_robustness import ParaphraseRobustness
from .s4_metrics_wiring import MetricsWiring
from .s5_security_negatives import SecurityNegatives
from .s6_backup_restore import BackupRestore
from .s7_config_parity import ConfigParity
from .s8_capacity_smoke import CapacitySmoke
from .s9_graph_intent import GraphIntentValidation
from .s10_content_pipeline import ContentPipelineMonitoring
from .s11_firewall_status import S11FirewallStatus

# Registry of all available checks
CHECK_REGISTRY = {
    "S1-probes": VerisHealthProbe,
    "S2-golden-fact-recall": GoldenFactRecall,
    "S3-paraphrase-robustness": ParaphraseRobustness,
    "S4-metrics-wiring": MetricsWiring,
    "S5-security-negatives": SecurityNegatives,
    "S6-backup-restore": BackupRestore,
    "S7-config-parity": ConfigParity,
    "S8-capacity-smoke": CapacitySmoke,
    "S9-graph-intent": GraphIntentValidation,
    "S10-content-pipeline": ContentPipelineMonitoring,
    "S11-firewall-status": S11FirewallStatus,
}

__all__ = [
    'VerisHealthProbe',
    'GoldenFactRecall', 
    'ParaphraseRobustness',
    'MetricsWiring',
    'SecurityNegatives',
    'BackupRestore',
    'ConfigParity',
    'CapacitySmoke',
    'GraphIntentValidation',
    'ContentPipelineMonitoring',
    'S11FirewallStatus',
    'CHECK_REGISTRY',
]