from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_oran_rca.graph.graph_builder import (
    ALARM_TYPES,
    ALARM_TO_FAULT,
    ROOT_CAUSES,
    TopologySpec,
    generate_oran_topology,
)

logger = logging.getLogger(__name__)


KPI_TYPES = [
    "RSRP Drop",
    "RSRQ Drop",
    "SINR Drop",
    "Throughput Drop",
    "Latency Spike",
    "PRB Utilization High",
    "Packet Retransmissions High",
]


FAULT_TO_KPI_BIAS = {
    "DU failure": {"Throughput Drop": 0.30, "Latency Spike": 0.25, "RSRP Drop": 0.10},
    "backhaul failure": {"Packet Retransmissions High": 0.35, "Latency Spike": 0.30, "Throughput Drop": 0.15},
    "congestion": {"PRB Utilization High": 0.45, "Latency Spike": 0.20, "Throughput Drop": 0.20},
    "antenna fault": {"RSRP Drop": 0.40, "RSRQ Drop": 0.25, "SINR Drop": 0.20},
    "power supply failure": {"RSRP Drop": 0.30, "Throughput Drop": 0.20, "SINR Drop": 0.15},
    "misconfiguration": {"RSRQ Drop": 0.20, "Throughput Drop": 0.20, "Latency Spike": 0.15},
}


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    seed: int
    start_time_utc: str
    span_days: int


def _weighted_choice(rnd: random.Random, weights: dict[str, float], universe: list[str]) -> str:
    w = np.array([weights.get(k, 0.0) for k in universe], dtype=float)
    if float(w.sum()) <= 0:
        return rnd.choice(universe)
    w = w / w.sum()
    return str(rnd.choices(universe, weights=w.tolist(), k=1)[0])


def generate_synthetic_dataset(
    topo: dict,
    spec: DatasetSpec,
) -> pd.DataFrame:
    rnd = random.Random(spec.seed)
    np.random.seed(spec.seed)

    start_dt = datetime.fromisoformat(spec.start_time_utc).replace(tzinfo=timezone.utc)
    cells: list[str] = topo["cells"]

    records: list[dict] = []

    for i in range(spec.rows):
        cell = rnd.choice(cells)
        du = topo["cell_to_du"][cell]
        cu = topo["du_to_cu"][du]

        alarm = rnd.choice(ALARM_TYPES)

        candidates = ALARM_TO_FAULT[alarm]
        # Make alarm-only ambiguous on purpose: random among plausible candidates
        expected_root_cause = rnd.choice(candidates)

        kpi = _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(expected_root_cause, {}), KPI_TYPES)

        # Timestamp
        offset_minutes = rnd.randint(0, spec.span_days * 24 * 60 - 1)
        ts = start_dt + timedelta(minutes=offset_minutes)

        records.append(
            {
                "cell": cell,
                "du": du,
                "cu": cu,
                "alarm": alarm,
                "kpi": kpi,
                "expected_root_cause": expected_root_cause,
                "timestamp": ts.isoformat(),
            }
        )

    df = pd.DataFrame.from_records(records)

    # Add structured signal that context-aware model can exploit:
    # - Certain DUs are more prone to power supply failure
    # - Certain CUs correlate with backhaul issues
    du_list = sorted(set(df["du"].tolist()))
    cu_list = sorted(set(df["cu"].tolist()))

    rnd.shuffle(du_list)
    power_prone_dus = set(du_list[: max(1, len(du_list) // 4)])

    rnd.shuffle(cu_list)
    backhaul_prone_cus = set(cu_list[: max(1, len(cu_list) // 3)])

    mask_power_alarm = df["alarm"].isin(["Cell Down", "Power Failure", "Signal Degradation"])
    df.loc[mask_power_alarm & df["du"].isin(power_prone_dus), "expected_root_cause"] = "power supply failure"

    mask_transport_alarm = df["alarm"].isin(["Transport Link Failure", "Packet Loss", "High Latency"])
    df.loc[mask_transport_alarm & df["cu"].isin(backhaul_prone_cus), "expected_root_cause"] = "backhaul failure"

    # Re-sample KPI after the above adjustments
    df["kpi"] = [
        _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(f, {}), KPI_TYPES) for f in df["expected_root_cause"].tolist()
    ]

    # Ensure all root causes appear
    present = set(df["expected_root_cause"].unique())
    missing = [c for c in ROOT_CAUSES if c not in present]
    if missing:
        for c in missing:
            idx = rnd.randrange(0, len(df))
            df.at[idx, "expected_root_cause"] = c
            df.at[idx, "kpi"] = _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(c, {}), KPI_TYPES)

    return df


def write_artifacts(
    project_root: Path,
    topo_spec: TopologySpec,
    dataset_spec: DatasetSpec,
) -> tuple[Path, Path]:
    data_dir = project_root / "agentic_oran_rca" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    topo = generate_oran_topology(topo_spec)
    topo_path = data_dir / "topology.json"
    topo_path.write_text(json.dumps(topo, indent=2), encoding="utf-8")

    df = generate_synthetic_dataset(topo, dataset_spec)
    csv_path = data_dir / "telecom_dataset.csv"
    df.to_csv(csv_path, index=False)

    logger.info("Wrote topology: %s", topo_path)
    logger.info("Wrote dataset: %s", csv_path)
    return topo_path, csv_path

