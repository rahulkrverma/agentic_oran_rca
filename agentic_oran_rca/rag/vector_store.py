from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import chromadb
import pandas as pd
from chromadb.api import ClientAPI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorStoreConfig:
    chroma_host: str
    chroma_port: int
    collection_name: str
    ollama_base_url: str
    ollama_embed_model: str


def build_chroma_http_client(host: str, port: int) -> ClientAPI:
    return chromadb.HttpClient(host=host, port=port)


def build_vector_store(cfg: VectorStoreConfig) -> Chroma:
    client = build_chroma_http_client(cfg.chroma_host, cfg.chroma_port)
    embeddings = OllamaEmbeddings(base_url=cfg.ollama_base_url, model=cfg.ollama_embed_model)
    vs = Chroma(
        client=client,
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
    )
    return vs


def format_incident_document(
    *,
    cell: str,
    du: str,
    cu: str,
    alarm: str,
    kpi: str,
    root_cause: str,
    timestamp: str,
    incident_id: str | None = None,
    site_id: str | None = None,
    gnb_id: str | None = None,
    sector_id: int | str | None = None,
    region: str | None = None,
    vendor: str | None = None,
    rat: str | None = None,
    alarm_code: str | None = None,
    alarm_severity: str | None = None,
    kpi_value: float | str | None = None,
    kpi_unit: str | None = None,
) -> str:
    parts = [
        f"cell={cell}",
        f"du={du}",
        f"cu={cu}",
        f"alarm={alarm}",
        f"kpi={kpi}",
        f"root_cause={root_cause}",
        f"timestamp={timestamp}",
    ]
    optional = [
        ("incident_id", incident_id),
        ("site_id", site_id),
        ("gnb_id", gnb_id),
        ("sector_id", sector_id),
        ("region", region),
        ("vendor", vendor),
        ("rat", rat),
        ("alarm_code", alarm_code),
        ("alarm_severity", alarm_severity),
        ("kpi_value", kpi_value),
        ("kpi_unit", kpi_unit),
    ]
    for key, value in optional:
        if value is not None and str(value) != "":
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_incident_metadata(
    *,
    cell: str,
    du: str,
    cu: str,
    alarm: str,
    kpi: str,
    root_cause: str,
    timestamp: str,
    source: str,
    remediation_action: str | None = None,
    remediation_success: bool | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "cell": cell,
        "du": du,
        "cu": cu,
        "alarm": alarm,
        "kpi": kpi,
        "expected_root_cause": root_cause,
        "timestamp": timestamp,
        "source": source,
    }
    if remediation_action is not None:
        metadata["remediation_action"] = remediation_action
    if remediation_success is not None:
        metadata["remediation_success"] = remediation_success
    return metadata


def healing_document_id(cell: str, timestamp_utc: str) -> str:
    cell_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", cell.strip())
    ts_safe = timestamp_utc.replace(":", "-").replace(".", "-")
    return f"healing_{cell_safe}_{ts_safe}"


def upsert_incident_documents(
    vs: Chroma,
    rows: list[dict[str, Any]],
    id_prefix: str,
) -> int:
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for i, r in enumerate(rows):
        root_cause = str(r["expected_root_cause"])
        timestamp = str(r["timestamp"])
        doc_kwargs = {
            "cell": str(r["cell"]),
            "du": str(r["du"]),
            "cu": str(r["cu"]),
            "alarm": str(r["alarm"]),
            "kpi": str(r["kpi"]),
            "root_cause": root_cause,
            "timestamp": timestamp,
        }
        for key in (
            "incident_id",
            "site_id",
            "gnb_id",
            "sector_id",
            "region",
            "vendor",
            "rat",
            "alarm_code",
            "alarm_severity",
            "kpi_value",
            "kpi_unit",
        ):
            if key in r and pd.notna(r[key]):
                doc_kwargs[key] = r[key]
        texts.append(format_incident_document(**doc_kwargs))
        metadata = format_incident_metadata(
            cell=str(r["cell"]),
            du=str(r["du"]),
            cu=str(r["cu"]),
            alarm=str(r["alarm"]),
            kpi=str(r["kpi"]),
            root_cause=root_cause,
            timestamp=timestamp,
            source="dataset",
        )
        if "incident_id" in r and pd.notna(r["incident_id"]):
            metadata["incident_id"] = str(r["incident_id"])
        metadatas.append(metadata)
        doc_id = str(r["incident_id"]) if "incident_id" in r and pd.notna(r["incident_id"]) else f"{id_prefix}{i}"
        ids.append(doc_id)

    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(ids)

