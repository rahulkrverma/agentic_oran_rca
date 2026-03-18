from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from langchain_community.embeddings.ollama import OllamaEmbeddings
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


def upsert_incident_documents(
    vs: Chroma,
    rows: list[dict[str, Any]],
    id_prefix: str,
) -> int:
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for i, r in enumerate(rows):
        doc = (
            f"cell={r['cell']} du={r['du']} cu={r['cu']} "
            f"alarm={r['alarm']} kpi={r['kpi']} "
            f"root_cause={r['expected_root_cause']} timestamp={r['timestamp']}"
        )
        texts.append(doc)
        metadatas.append(
            {
                "cell": r["cell"],
                "du": r["du"],
                "cu": r["cu"],
                "alarm": r["alarm"],
                "kpi": r["kpi"],
                "expected_root_cause": r["expected_root_cause"],
                "timestamp": r["timestamp"],
            }
        )
        ids.append(f"{id_prefix}{i}")

    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(ids)

