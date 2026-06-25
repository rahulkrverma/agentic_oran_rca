from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agentic_oran_rca.agents.context_agent import RetrievedContext

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if code_block:
        block = code_block.group(1).strip()
        start = block.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(block[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(block[start : i + 1])
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(raw[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start : i + 1])
    return json.loads(raw)


class RCAOutput(BaseModel):
    predicted_root_cause: str = Field(..., min_length=1)
    confidence_score: float = Field(..., ge=0.0, le=1.0)


@dataclass(frozen=True)
class RCAAgentConfig:
    ollama_base_url: str
    ollama_model: str


class RCAAnalysisAgent:
    def __init__(self, cfg: RCAAgentConfig) -> None:
        self._llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
        )

    def run(self, alarm: str, kpi: str, ctx: RetrievedContext) -> RCAOutput:
        # Strict JSON output required for robust parsing.
        prompt = (
            "You are a telecom network troubleshooting expert.\n"
            "Return ONLY valid JSON matching this schema:\n"
            '{"predicted_root_cause": "<string>", "confidence_score": <number 0..1>}\n\n'
            f"Cell: {ctx.cell}\n"
            f"Serving DU: {ctx.du}\n"
            f"Connected CU: {ctx.cu}\n"
            f"Neighbor cells: {', '.join(ctx.neighbor_cells) if ctx.neighbor_cells else 'None'}\n\n"
            f"Alarm: {alarm}\n"
            f"KPI anomaly: {kpi}\n\n"
            "Candidate faults from knowledge graph (fault, historical_cases):\n"
            + "\n".join([f"- {f['fault']}: {f['historical_cases']}" for f in ctx.related_faults])
            + "\n\n"
            "Similar incidents (top matches):\n"
            + "\n".join(
                [
                    f"- score={s['score']:.4f} root_cause={s['metadata'].get('expected_root_cause')} text={s['text']}"
                    for s in ctx.similar_incidents[:5]
                ]
            )
            + "\n\n"
            "Decision rules:\n"
            "- If alarm is 'Transport Link Failure', prefer 'backhaul failure'.\n"
            "- If alarm is 'Power Failure', prefer 'power supply failure' or 'DU failure'.\n"
            "- If KPI includes 'PRB Utilization High', prefer 'congestion'.\n"
            "- If KPI includes 'RSRP Drop'/'RSRQ Drop'/'SINR Drop' and alarm is 'Signal Degradation', prefer 'antenna fault'.\n"
            "- Use CU/DU context and similar incidents as tie-breakers.\n"
            "- Confidence should be higher when multiple signals agree.\n"
        )

        msg = self._llm.invoke(prompt)
        raw = str(getattr(msg, "content", msg))
        try:
            obj = _extract_json(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM returned non-JSON output: {raw}") from e

        return RCAOutput.model_validate(obj)

