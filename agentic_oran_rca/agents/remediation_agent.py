"""
Remediation Agent: maps predicted root cause to remediation actions and simulates execution.
Used by the self-healing extension. Does not modify the core RCA flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


ROOT_CAUSE_TO_REMEDIATION = {
    "DU failure": "restart_du",
    "backhaul failure": "failover_to_backup_path",
    "congestion": "prb_reallocation",
    "antenna fault": "antenna_tilt_adjustment",
    "power supply failure": "power_cycle_du",
    "misconfiguration": "rollback_config",
}


REMEDIATION_DESCRIPTIONS = {
    "restart_du": "Restart Distributed Unit to recover from software/hardware hang",
    "failover_to_backup_path": "Switch traffic to backup transport path",
    "prb_reallocation": "Reallocate PRB resources to relieve congestion",
    "antenna_tilt_adjustment": "Adjust antenna tilt to restore coverage",
    "power_cycle_du": "Power cycle DU to recover from power anomaly",
    "rollback_config": "Rollback to last known good configuration",
}


@dataclass(frozen=True)
class RemediationResult:
    action: str
    action_description: str
    success: bool
    message: str
    timestamp_utc: str


class RemediationAgent:
    """
    Maps root cause to remediation action and simulates execution.
    In a production O-RAN rApp, this would invoke E2/Near-RT RIC or OAM APIs.
    """

    def __init__(self, success_rate: float = 0.85) -> None:
        self._success_rate = max(0.0, min(1.0, success_rate))

    def run(
        self,
        cell: str,
        du: str,
        cu: str,
        alarm: str,
        predicted_root_cause: str,
        confidence: float,
    ) -> RemediationResult:
        action = ROOT_CAUSE_TO_REMEDIATION.get(
            predicted_root_cause,
            "restart_du",
        )
        action_desc = REMEDIATION_DESCRIPTIONS.get(
            action,
            "Default remediation action",
        )

        # Deterministic simulation: hash-based for reproducibility within a run
        seed = abs(hash((cell, du, alarm, predicted_root_cause))) % 1000
        success = (seed / 1000.0) < self._success_rate

        if success:
            message = f"Remediation '{action}' completed successfully for {cell}."
        else:
            message = f"Remediation '{action}' failed for {cell}. Manual intervention required."

        ts = datetime.now(timezone.utc).isoformat()
        return RemediationResult(
            action=action,
            action_description=action_desc,
            success=success,
            message=message,
            timestamp_utc=ts,
        )
