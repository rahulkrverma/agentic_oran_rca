"""
Notification Service: persists auto-correction reports and notifications.
Used by the self-healing extension. Writes to files for research traceability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealingReport:
    timestamp_utc: str
    cell: str
    du: str
    cu: str
    alarm: str
    kpi: str
    predicted_root_cause: str
    confidence: float
    remediation_action: str
    remediation_success: bool
    notification_type: str
    message: str


@dataclass(frozen=True)
class NotificationRecord:
    timestamp_utc: str
    notification_type: str
    cell: str
    alarm: str
    root_cause: str
    remediation_success: bool
    message: str
    report_path: str | None


class NotificationService:
    """
    Persists healing reports and notifications to disk.
    In production, this would integrate with NOC ticketing, PagerDuty, or OSS.
    """

    def __init__(
        self,
        reports_dir: Path,
        notifications_path: Path,
        api_responses_dir: Path | None = None,
    ) -> None:
        self._reports_dir = reports_dir
        self._notifications_path = notifications_path
        self._api_responses_dir = api_responses_dir

    def write_healing_report(self, report: HealingReport) -> Path:
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = report.timestamp_utc.replace(":", "-").replace(".", "-")[:19]
        filename = f"healing_{report.cell}_{ts_safe}.json"
        path = self._reports_dir / filename
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        logger.info("Healing report written: %s", path)
        return path

    def send_notification(
        self,
        notification_type: str,
        cell: str,
        alarm: str,
        root_cause: str,
        remediation_success: bool,
        message: str,
        report_path: Path | None = None,
    ) -> None:
        record = NotificationRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            notification_type=notification_type,
            cell=cell,
            alarm=alarm,
            root_cause=root_cause,
            remediation_success=remediation_success,
            message=message,
            report_path=str(report_path) if report_path else None,
        )
        self._notifications_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._notifications_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        logger.info("Notification recorded: %s for %s", notification_type, cell)

    def write_api_response(self, endpoint: str, cell: str, payload: dict[str, Any]) -> Path | None:
        if self._api_responses_dir is None:
            return None
        self._api_responses_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")[:19]
        cell_safe = cell.replace(" ", "_")
        filename = f"{endpoint}_{cell_safe}_{ts_safe}.json"
        path = self._api_responses_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("API response written: %s", path)
        return path
