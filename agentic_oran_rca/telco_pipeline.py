from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_oran_rca.agents.telco_fault_agent import (
    TelcoAgentConfig,
    TelcoContextAgent,
    TelcoFaultAnalysisAgent,
)
from agentic_oran_rca.config import Settings
from agentic_oran_rca.rag.retriever import IncidentRetriever
from agentic_oran_rca.rag.telco_vector_store import TELCO_COLLECTION_NAME, build_telco_vector_store
from agentic_oran_rca.rag.vector_store import VectorStoreConfig


@dataclass(frozen=True)
class TelcoFaultPipeline:
    ctx_agent: TelcoContextAgent
    analysis_agent: TelcoFaultAnalysisAgent
    retrieve_k: int

    def run(self, symptoms: str, use_rag: bool) -> dict[str, Any]:
        ctx = self.ctx_agent.run(symptoms=symptoms, k=self.retrieve_k, use_rag=use_rag)
        out = self.analysis_agent.run(ctx)
        return {
            "symptoms": symptoms,
            "use_rag": use_rag,
            "predicted_cause": out.predicted_cause,
            "confidence": float(out.confidence),
            "explanation": out.explanation,
            "recommended_actions": out.recommended_actions,
            "similar_incidents": ctx.similar_incidents,
        }


def build_telco_pipeline(settings: Settings, *, with_retriever: bool) -> TelcoFaultPipeline:
    retriever: IncidentRetriever | None = None
    if with_retriever:
        vs = build_telco_vector_store(
            VectorStoreConfig(
                chroma_host=settings.chroma_host,
                chroma_port=settings.chroma_port,
                collection_name=TELCO_COLLECTION_NAME,
                ollama_base_url=settings.ollama_base_url,
                ollama_embed_model=settings.ollama_embed_model,
            )
        )
        retriever = IncidentRetriever(vs)
    return TelcoFaultPipeline(
        ctx_agent=TelcoContextAgent(retriever=retriever),
        analysis_agent=TelcoFaultAnalysisAgent(
            TelcoAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model)
        ),
        retrieve_k=7,
    )
