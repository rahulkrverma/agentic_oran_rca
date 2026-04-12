"""
Auto-correction agent: orchestrates context retrieval, RCA, and explanation agents,
then runs a review step to confirm or correct the predicted root cause with rationale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

from agentic_oran_rca.agents.context_agent import ContextRetrievalAgent, RetrievedContext
from agentic_oran_rca.agents.explanation_agent import ExplanationAgent
from agentic_oran_rca.agents.rca_agent import RCAAnalysisAgent, RCAOutput, _extract_json
from agentic_oran_rca.graph.graph_builder import ROOT_CAUSES

logger = logging.getLogger(__name__)


def _context_to_dict(ctx: RetrievedContext) -> dict[str, Any]:
    return {
        "cell": ctx.cell,
        "du": ctx.du,
        "cu": ctx.cu,
        "neighbor_cells": list(ctx.neighbor_cells),
        "related_faults": list(ctx.related_faults),
        "similar_incidents": list(ctx.similar_incidents),
    }


@dataclass(frozen=True)
class AutoCorrectionAgentConfig:
    ollama_base_url: str
    ollama_model: str


class CorrectionReviewOutput(BaseModel):
    corrected_root_cause: str = Field(..., min_length=1)
    correction_rationale: str = Field(..., min_length=1)
    final_explanation: str = Field(..., min_length=1)


class AutoCorrectionAgent:
    """
    Fetches context via ContextRetrievalAgent, predicts root cause via RCAAnalysisAgent,
    drafts an explanation via ExplanationAgent, then invokes a reviewer LLM to produce
    a corrected root cause (when justified) and a consolidated explanation.
    """

    def __init__(
        self,
        ctx_agent: ContextRetrievalAgent,
        rca_agent: RCAAnalysisAgent,
        expl_agent: ExplanationAgent,
        cfg: AutoCorrectionAgentConfig,
    ) -> None:
        self._ctx_agent = ctx_agent
        self._rca_agent = rca_agent
        self._expl_agent = expl_agent
        self._review_llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
        )

    def run(self, cell: str, alarm: str, kpi: str, *, context_k: int = 7) -> dict[str, Any]:
        ctx = self._ctx_agent.run(cell_id=cell, alarm=alarm, k=context_k)
        rca: RCAOutput = self._rca_agent.run(alarm=alarm, kpi=kpi, ctx=ctx)
        initial_explanation = self._expl_agent.run(
            alarm=alarm,
            kpi=kpi,
            predicted_root_cause=rca.predicted_root_cause,
            confidence_score=rca.confidence_score,
            ctx=ctx,
        )

        allowed = ", ".join(sorted(ROOT_CAUSES))
        review_prompt = (
            "You are a senior telecom RCA reviewer.\n"
            "Given the retrieved context, the initial RCA prediction, and its draft explanation, "
            "either CONFIRM the predicted root cause or CORRECT it if evidence clearly favors another label.\n"
            "Return ONLY valid JSON matching this schema:\n"
            '{"corrected_root_cause": "<string>", "correction_rationale": "<string>", '
            '"final_explanation": "<string>"}\n\n'
            f"Allowed root-cause labels (use exactly one spelling): {allowed}\n\n"
            f"Cell: {ctx.cell}\nDU: {ctx.du}  CU: {ctx.cu}\n"
            f"Neighbors: {', '.join(ctx.neighbor_cells) if ctx.neighbor_cells else 'None'}\n\n"
            f"Alarm: {alarm}\nKPI anomaly: {kpi}\n\n"
            "Graph candidate faults (fault, historical_cases):\n"
            + "\n".join([f"- {f['fault']}: {f['historical_cases']}" for f in ctx.related_faults])
            + "\n\n"
            "Top similar incidents:\n"
            + "\n".join(
                [
                    f"- score={s['score']:.4f} root_cause={s['metadata'].get('expected_root_cause')} text={s['text'][:200]}"
                    for s in ctx.similar_incidents[:5]
                ]
            )
            + "\n\n"
            f"Initial prediction: {rca.predicted_root_cause}\n"
            f"Initial confidence: {rca.confidence_score:.4f}\n\n"
            f"Initial explanation:\n{initial_explanation}\n\n"
            "Rules:\n"
            "- corrected_root_cause MUST be one of the allowed labels.\n"
            "- If the initial prediction is sound, set corrected_root_cause to the same value and explain why in correction_rationale.\n"
            "- correction_rationale must state why you kept or changed the label (cite topology, alarm/KPI, or similar incidents).\n"
            "- final_explanation must be a self-contained 4–8 sentence narrative for operators (may reuse and refine the initial explanation).\n"
        )

        msg = self._review_llm.invoke(review_prompt)
        raw = str(getattr(msg, "content", msg))
        try:
            obj = _extract_json(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Reviewer LLM returned non-JSON output: {raw}") from e

        review = CorrectionReviewOutput.model_validate(obj)
        if review.corrected_root_cause not in ROOT_CAUSES:
            raise RuntimeError(
                f"Reviewer returned invalid corrected_root_cause={review.corrected_root_cause!r}; "
                f"allowed: {ROOT_CAUSES}"
            )

        correction_applied = review.corrected_root_cause != rca.predicted_root_cause

        return {
            "context_retrieval": _context_to_dict(ctx),
            "initial_predicted_root_cause": rca.predicted_root_cause,
            "initial_confidence": float(rca.confidence_score),
            "initial_explanation": initial_explanation,
            "corrected_root_cause": review.corrected_root_cause,
            "correction_applied": correction_applied,
            "correction_rationale": review.correction_rationale,
            "final_explanation": review.final_explanation,
        }
