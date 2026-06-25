from __future__ import annotations

import logging
from typing import Any

from langchain_community.vectorstores import Chroma

from agentic_oran_rca.data.telco_llm_dataset import TelcoFaultRecord, format_telco_document, telco_metadata
from agentic_oran_rca.rag.vector_store import VectorStoreConfig, build_vector_store

logger = logging.getLogger(__name__)

TELCO_COLLECTION_NAME = "telco_fault_incidents"


def build_telco_vector_store(cfg: VectorStoreConfig) -> Chroma:
    return build_vector_store(cfg)


def upsert_telco_records(vs: Chroma, records: list[TelcoFaultRecord]) -> int:
    texts = [format_telco_document(r) for r in records]
    metadatas: list[dict[str, Any]] = [telco_metadata(r) for r in records]
    ids = [r.record_id for r in records]
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(ids)
