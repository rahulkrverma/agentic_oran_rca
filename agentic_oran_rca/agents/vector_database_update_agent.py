"""
Vector Database Update Agent: persists validated healing outcomes into Chroma
so future RCA runs can retrieve the latest corrected incident context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma

from agentic_oran_rca.rag.vector_store import (
    format_incident_document,
    format_incident_metadata,
    healing_document_id,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealingCorrectionRecord:
    cell: str
    du: str
    cu: str
    alarm: str
    kpi: str
    root_cause: str
    timestamp_utc: str
    remediation_action: str
    remediation_success: bool


@dataclass(frozen=True)
class VectorUpdateResult:
    vector_indexed: bool
    chroma_document_id: str | None
    skipped_reason: str | None


class VectorDatabaseUpdateAgent:
    """
    Upserts incident knowledge into the vector store after a successful healing correction.
    Skips indexing when remediation did not succeed.
    """

    def __init__(self, vector_store: Chroma) -> None:
        self._vector_store = vector_store

    def run(self, record: HealingCorrectionRecord) -> VectorUpdateResult:
        if not record.remediation_success:
            logger.info(
                "Skipping vector update for %s: remediation was not successful",
                record.cell,
            )
            return VectorUpdateResult(
                vector_indexed=False,
                chroma_document_id=None,
                skipped_reason="remediation_not_successful",
            )

        doc_id = healing_document_id(record.cell, record.timestamp_utc)
        doc_text = format_incident_document(
            cell=record.cell,
            du=record.du,
            cu=record.cu,
            alarm=record.alarm,
            kpi=record.kpi,
            root_cause=record.root_cause,
            timestamp=record.timestamp_utc,
        )
        metadata = format_incident_metadata(
            cell=record.cell,
            du=record.du,
            cu=record.cu,
            alarm=record.alarm,
            kpi=record.kpi,
            root_cause=record.root_cause,
            timestamp=record.timestamp_utc,
            source="healing_correction",
            remediation_action=record.remediation_action,
            remediation_success=True,
        )
        embedding = self._vector_store._embedding_function.embed_documents([doc_text])
        self._vector_store._collection.upsert(
            ids=[doc_id],
            embeddings=embedding,
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.info("VectorDatabaseUpdateAgent indexed healing correction: %s", doc_id)
        return VectorUpdateResult(
            vector_indexed=True,
            chroma_document_id=doc_id,
            skipped_reason=None,
        )
