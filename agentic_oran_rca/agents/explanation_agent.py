from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from agentic_oran_rca.agents.context_agent import RetrievedContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationAgentConfig:
    ollama_base_url: str
    ollama_model: str


class ExplanationAgent:
    def __init__(self, cfg: ExplanationAgentConfig) -> None:
        self._llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
        )

    def run(
        self,
        alarm: str,
        kpi: str,
        predicted_root_cause: str,
        confidence_score: float,
        ctx: RetrievedContext,
    ) -> str:
        prompt = (
            "You are a telecom RCA assistant.\n"
            "Write a concise human-readable explanation (3-7 sentences).\n"
            "Use the provided topology, alarm, KPI anomaly, and similar incidents.\n\n"
            f"Cell: {ctx.cell}\n"
            f"Serving DU: {ctx.du}\n"
            f"Connected CU: {ctx.cu}\n"
            f"Neighbors: {', '.join(ctx.neighbor_cells) if ctx.neighbor_cells else 'None'}\n\n"
            f"Alarm: {alarm}\n"
            f"KPI: {kpi}\n\n"
            f"Predicted root cause: {predicted_root_cause}\n"
            f"Confidence: {confidence_score:.2f}\n\n"
            "Top similar incidents:\n"
            + "\n".join(
                [
                    f"- root_cause={s['metadata'].get('expected_root_cause')} score={s['score']:.4f}"
                    for s in ctx.similar_incidents[:3]
                ]
            )
        )

        msg = self._llm.invoke(prompt)
        return str(getattr(msg, "content", msg)).strip()

