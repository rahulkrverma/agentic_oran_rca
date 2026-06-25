from __future__ import annotations

import json
import logging
import random
import string
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

VENDORS = ["Ericsson", "Nokia", "Samsung", "ZTE", "Huawei"]
REGIONS = ["North", "South", "East", "West", "Central"]
RAT_TYPES = ["5G_SA", "5G_NSA", "LTE"]
DATA_SOURCES = ["FM", "PM", "NearRT_RIC", "OSS_Ticket"]
SEVERITIES = ["Critical", "Major", "Minor", "Warning"]
ALARM_STATUS = ["Active", "Cleared"]

ALARM_METADATA = {
    "Cell Down": {"code": "7701", "severity": "Critical", "prob_critical": 0.85},
    "High Latency": {"code": "198092553", "severity": "Major", "prob_critical": 0.35},
    "Packet Loss": {"code": "198097605", "severity": "Major", "prob_critical": 0.40},
    "Signal Degradation": {"code": "198092322", "severity": "Major", "prob_critical": 0.45},
    "Power Failure": {"code": "7706", "severity": "Critical", "prob_critical": 0.80},
    "Transport Link Failure": {"code": "198098001", "severity": "Critical", "prob_critical": 0.75},
}

KPI_UNITS = {
    "RSRP Drop": "dBm",
    "RSRQ Drop": "dB",
    "SINR Drop": "dB",
    "Throughput Drop": "Mbps",
    "Latency Spike": "ms",
    "PRB Utilization High": "%",
    "Packet Retransmissions High": "%",
}

FAULT_KPI_VALUE_RANGES = {
    "DU failure": {"Throughput Drop": (2.0, 45.0), "Latency Spike": (35.0, 180.0), "RSRP Drop": (-125.0, -95.0)},
    "backhaul failure": {
        "Packet Retransmissions High": (3.0, 22.0),
        "Latency Spike": (40.0, 220.0),
        "Throughput Drop": (5.0, 60.0),
    },
    "congestion": {"PRB Utilization High": (72.0, 98.0), "Latency Spike": (25.0, 120.0), "Throughput Drop": (8.0, 55.0)},
    "antenna fault": {"RSRP Drop": (-118.0, -88.0), "RSRQ Drop": (-18.0, -8.0), "SINR Drop": (-5.0, 8.0)},
    "power supply failure": {"RSRP Drop": (-120.0, -90.0), "Throughput Drop": (1.0, 35.0), "SINR Drop": (-8.0, 5.0)},
    "misconfiguration": {"RSRQ Drop": (-16.0, -7.0), "Throughput Drop": (10.0, 70.0), "Latency Spike": (20.0, 90.0)},
}


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    seed: int
    start_time_utc: str
    span_days: int


@dataclass(frozen=True)
class EnterpriseDatasetSpec(DatasetSpec):
    business_hours_weight: float = 0.72
    cleared_alarm_ratio: float = 0.38


def _weighted_choice(rnd: random.Random, weights: dict[str, float], universe: list[str]) -> str:
    w = np.array([weights.get(k, 0.0) for k in universe], dtype=float)
    if float(w.sum()) <= 0:
        return rnd.choice(universe)
    w = w / w.sum()
    return str(rnd.choices(universe, weights=w.tolist(), k=1)[0])


def _incident_id(rnd: random.Random, ts: datetime) -> str:
    suffix = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=6))
    return f"INC-{ts.strftime('%Y%m%d%H%M%S')}-{suffix}"


def _ticket_id(rnd: random.Random, ts: datetime) -> str:
    return f"TT-{ts.strftime('%Y%m%d')}-{rnd.randint(100000, 999999)}"


def _sample_timestamp(
    rnd: random.Random,
    np_rng: np.random.Generator,
    start_dt: datetime,
    span_days: int,
    business_hours_weight: float,
) -> datetime:
    offset_minutes = int(np_rng.integers(0, max(span_days * 24 * 60 - 1, 1)))
    ts = start_dt + timedelta(minutes=offset_minutes)
    if rnd.random() < business_hours_weight:
        hour = int(np_rng.integers(8, 23))
        ts = ts.replace(hour=hour, minute=int(np_rng.integers(0, 60)), second=int(np_rng.integers(0, 60)))
    return ts


def _sample_kpi_value(np_rng: np.random.Generator, fault: str, kpi: str) -> float:
    ranges = FAULT_KPI_VALUE_RANGES.get(fault, {})
    low, high = ranges.get(kpi, (1.0, 100.0))
    return float(np_rng.uniform(low, high))


def _derive_radio_metrics(np_rng: np.random.Generator, fault: str, kpi: str, kpi_value: float) -> dict[str, float]:
    if kpi == "RSRP Drop":
        rsrp = kpi_value
    else:
        rsrp = float(np_rng.uniform(-118.0, -82.0))
    if kpi == "SINR Drop":
        sinr = kpi_value
    else:
        sinr = float(np_rng.uniform(-4.0, 18.0))
    if kpi == "Throughput Drop":
        throughput = kpi_value
    else:
        throughput = float(np_rng.uniform(12.0, 420.0))
    if kpi == "Latency Spike":
        latency = kpi_value
    else:
        latency = float(np_rng.uniform(8.0, 65.0))
    if kpi == "Packet Retransmissions High":
        packet_loss = kpi_value
    else:
        packet_loss = float(np_rng.uniform(0.1, 8.0))
    if kpi == "PRB Utilization High":
        prb = kpi_value
    else:
        prb = float(np_rng.uniform(35.0, 92.0))

    if fault == "congestion":
        prb = max(prb, 78.0)
    if fault == "backhaul failure":
        latency = max(latency, 45.0)
        packet_loss = max(packet_loss, 2.5)
    if fault == "antenna fault":
        rsrp = min(rsrp, -90.0)

    return {
        "rsrp_dbm": round(rsrp, 2),
        "sinr_db": round(sinr, 2),
        "throughput_mbps": round(throughput, 2),
        "latency_ms": round(latency, 2),
        "packet_loss_pct": round(packet_loss, 2),
        "prb_utilization_pct": round(prb, 2),
    }


def _assign_structural_bias(df: pd.DataFrame, rnd: random.Random) -> pd.DataFrame:
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
    return df


def generate_enterprise_dataset(topo: dict, spec: EnterpriseDatasetSpec) -> pd.DataFrame:
    rnd = random.Random(spec.seed)
    np_rng = np.random.default_rng(spec.seed)

    start_dt = datetime.fromisoformat(spec.start_time_utc).replace(tzinfo=timezone.utc)
    cells: list[str] = topo["cells"]

    cell_region = {cell: REGIONS[hash(cell) % len(REGIONS)] for cell in cells}
    cell_site = {cell: f"SITE-{cell_region[cell][:3].upper()}-{int(cell.replace('Cell', '')):04d}" for cell in cells}
    cell_gnb = {cell: f"gNB-{topo['cell_to_du'][cell].replace('DU', '')}-{cell.replace('Cell', '')}" for cell in cells}

    records: list[dict] = []

    for _ in range(spec.rows):
        cell = rnd.choice(cells)
        du = topo["cell_to_du"][cell]
        cu = topo["du_to_cu"][du]
        region = cell_region[cell]
        site_id = cell_site[cell]
        gnb_id = cell_gnb[cell]
        sector_id = rnd.randint(1, 3)

        alarm = rnd.choice(ALARM_TYPES)
        candidates = ALARM_TO_FAULT[alarm]
        expected_root_cause = rnd.choice(candidates)
        kpi = _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(expected_root_cause, {}), KPI_TYPES)

        ts = _sample_timestamp(rnd, np_rng, start_dt, spec.span_days, spec.business_hours_weight)
        alarm_meta = ALARM_METADATA[alarm]
        severity = alarm_meta["severity"] if rnd.random() < alarm_meta["prob_critical"] else rnd.choice(["Major", "Minor"])

        kpi_value = _sample_kpi_value(np_rng, expected_root_cause, kpi)
        radio = _derive_radio_metrics(np_rng, expected_root_cause, kpi, kpi_value)

        alarm_status = "Cleared" if rnd.random() < spec.cleared_alarm_ratio else "Active"
        clear_ts = ""
        duration_minutes = 0
        if alarm_status == "Cleared":
            duration_minutes = int(np_rng.integers(5, 720))
            clear_dt = ts + timedelta(minutes=duration_minutes)
            clear_ts = clear_dt.isoformat()

        customer_impact = 5 if severity == "Critical" else 4 if severity == "Major" else rnd.randint(1, 3)

        records.append(
            {
                "incident_id": _incident_id(rnd, ts),
                "ticket_id": _ticket_id(rnd, ts),
                "timestamp": ts.isoformat(),
                "alarm_clear_time": clear_ts,
                "duration_minutes": duration_minutes,
                "cell": cell,
                "du": du,
                "cu": cu,
                "site_id": site_id,
                "gnb_id": gnb_id,
                "sector_id": sector_id,
                "region": region,
                "cluster_id": f"CLU-{region[:2].upper()}-{int(du.replace('DU', '')):02d}",
                "vendor": rnd.choice(VENDORS),
                "rat": rnd.choices(RAT_TYPES, weights=[0.62, 0.23, 0.15], k=1)[0],
                "alarm": alarm,
                "alarm_code": alarm_meta["code"],
                "alarm_severity": severity,
                "alarm_status": alarm_status,
                "kpi": kpi,
                "kpi_value": round(kpi_value, 3),
                "kpi_unit": KPI_UNITS[kpi],
                "kpi_direction": "degrade" if "Drop" in kpi or "High" in kpi else "spike",
                "rsrp_dbm": radio["rsrp_dbm"],
                "sinr_db": radio["sinr_db"],
                "throughput_mbps": radio["throughput_mbps"],
                "latency_ms": radio["latency_ms"],
                "packet_loss_pct": radio["packet_loss_pct"],
                "prb_utilization_pct": radio["prb_utilization_pct"],
                "customer_impact_score": customer_impact,
                "data_source": rnd.choices(DATA_SOURCES, weights=[0.45, 0.25, 0.15, 0.15], k=1)[0],
                "autott_ticket": rnd.random() < 0.18,
                "healing_applied": rnd.random() < 0.07,
                "expected_root_cause": expected_root_cause,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = _assign_structural_bias(df, rnd)

    df["kpi"] = [
        _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(f, {}), KPI_TYPES) for f in df["expected_root_cause"].tolist()
    ]
    df["kpi_value"] = [
        round(_sample_kpi_value(np_rng, str(fault), str(kpi)), 3)
        for fault, kpi in zip(df["expected_root_cause"], df["kpi"], strict=True)
    ]
    df["kpi_unit"] = df["kpi"].map(KPI_UNITS)

    radio_cols = []
    for fault, kpi, kpi_value in zip(df["expected_root_cause"], df["kpi"], df["kpi_value"], strict=True):
        radio_cols.append(_derive_radio_metrics(np_rng, str(fault), str(kpi), float(kpi_value)))
    radio_df = pd.DataFrame(radio_cols)
    for col in radio_df.columns:
        df[col] = radio_df[col]

    present = set(df["expected_root_cause"].unique())
    missing = [c for c in ROOT_CAUSES if c not in present]
    if missing:
        for cause in missing:
            idx = rnd.randrange(0, len(df))
            df.at[idx, "expected_root_cause"] = cause
            df.at[idx, "kpi"] = _weighted_choice(rnd, FAULT_TO_KPI_BIAS.get(cause, {}), KPI_TYPES)
            df.at[idx, "kpi_value"] = round(_sample_kpi_value(np_rng, cause, str(df.at[idx, "kpi"])), 3)

    column_order = [
        "incident_id",
        "ticket_id",
        "timestamp",
        "alarm_clear_time",
        "duration_minutes",
        "cell",
        "du",
        "cu",
        "site_id",
        "gnb_id",
        "sector_id",
        "region",
        "cluster_id",
        "vendor",
        "rat",
        "alarm",
        "alarm_code",
        "alarm_severity",
        "alarm_status",
        "kpi",
        "kpi_value",
        "kpi_unit",
        "kpi_direction",
        "rsrp_dbm",
        "sinr_db",
        "throughput_mbps",
        "latency_ms",
        "packet_loss_pct",
        "prb_utilization_pct",
        "customer_impact_score",
        "data_source",
        "autott_ticket",
        "healing_applied",
        "expected_root_cause",
    ]
    return df[column_order]


def generate_synthetic_dataset(topo: dict, spec: DatasetSpec) -> pd.DataFrame:
    """Backward-compatible entry point; uses enterprise generator."""
    enterprise_spec = EnterpriseDatasetSpec(
        rows=spec.rows,
        seed=spec.seed,
        start_time_utc=spec.start_time_utc,
        span_days=spec.span_days,
    )
    return generate_enterprise_dataset(topo, enterprise_spec)


def default_topology_for_rows(rows: int, seed: int) -> TopologySpec:
    if rows >= 10000:
        return TopologySpec(num_cus=8, num_dus=40, num_cells=180, neighbor_k=4, seed=seed)
    if rows >= 5000:
        return TopologySpec(num_cus=6, num_dus=28, num_cells=120, neighbor_k=4, seed=seed)
    if rows >= 2000:
        return TopologySpec(num_cus=4, num_dus=16, num_cells=80, neighbor_k=3, seed=seed)
    return TopologySpec(num_cus=2, num_dus=6, num_cells=40, neighbor_k=3, seed=seed)


def write_artifacts(
    project_root: Path,
    topo_spec: TopologySpec,
    dataset_spec: DatasetSpec,
) -> tuple[Path, Path]:
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    topo = generate_oran_topology(topo_spec)
    topo_path = data_dir / "topology.json"
    topo_path.write_text(json.dumps(topo, indent=2), encoding="utf-8")

    df = generate_synthetic_dataset(topo, dataset_spec)
    csv_path = data_dir / "telecom_dataset.csv"
    df.to_csv(csv_path, index=False)

    logger.info("Wrote topology: %s", topo_path)
    logger.info("Wrote dataset: %s (rows=%d, columns=%d)", csv_path, len(df), len(df.columns))
    return topo_path, csv_path
