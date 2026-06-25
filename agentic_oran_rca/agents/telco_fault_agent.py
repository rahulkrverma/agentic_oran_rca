from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agentic_oran_rca.rag.retriever import IncidentRetriever, SimilarIncident

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelcoRetrievedContext:
    symptoms: str
    similar_incidents: list[dict[str, Any]]
    rag_enabled: bool


@dataclass(frozen=True)
class TelcoAgentConfig:
    ollama_base_url: str
    ollama_model: str


class TelcoFaultOutput(BaseModel):
    predicted_cause: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    recommended_actions: str = Field(..., min_length=1)


def _fix_invalid_recommended_actions_array(text: str) -> str:
    """LLMs often emit [1. "step", 2. "step"] which is not valid JSON."""
    pattern = r'"recommended_actions"\s*:\s*\[([\s\S]*?)\]'
    match = re.search(pattern, text)
    if not match:
        return text
    inner = match.group(1)
    steps = re.findall(r'\d+\.\s*"([^"]*)"', inner)
    if not steps:
        steps = re.findall(r'"([^"]*)"', inner)
    if not steps:
        return text
    actions_str = "\\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    replacement = '"recommended_actions": ' + json.dumps(actions_str)
    return text[: match.start()] + replacement + text[match.end() :]


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = _fix_invalid_recommended_actions_array(text)
    return json.loads(candidate)


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
                        return _parse_json_object(block[start : i + 1])
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(raw[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return _parse_json_object(raw[start : i + 1])
    return _parse_json_object(raw)


def _normalize_telco_payload(obj: dict[str, Any]) -> dict[str, Any]:
    actions = obj.get("recommended_actions")
    if isinstance(actions, list):
        lines: list[str] = []
        for i, item in enumerate(actions, 1):
            lines.append(f"{i}. {item}" if isinstance(item, str) else str(item))
        obj["recommended_actions"] = "\n".join(lines)
    elif actions is not None and not isinstance(actions, str):
        obj["recommended_actions"] = str(actions)
    return obj


class TelcoContextAgent:
    def __init__(self, retriever: IncidentRetriever | None) -> None:
        self._retriever = retriever

    def run(self, symptoms: str, k: int, use_rag: bool) -> TelcoRetrievedContext:
        similar: list[SimilarIncident] = []
        if use_rag:
            if self._retriever is None:
                raise RuntimeError("RAG requested but no telco retriever configured")
            similar = self._retriever.retrieve(query=f"symptoms={symptoms}", k=k)
        return TelcoRetrievedContext(
            symptoms=symptoms,
            similar_incidents=[
                {"text": s.text, "metadata": s.metadata, "score": float(s.score)} for s in similar
            ],
            rag_enabled=use_rag,
        )


class TelcoFaultAnalysisAgent:
    def __init__(self, cfg: TelcoAgentConfig) -> None:
        self._llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
            format="json",
        )

    def run(self, ctx: TelcoRetrievedContext) -> TelcoFaultOutput:
        similar_block = "None"
        if ctx.similar_incidents:
            similar_block = "\n".join(
                [
                    f"- score={s['score']:.4f} text={s['text'][:400]}"
                    for s in ctx.similar_incidents[:5]
                ]
            )
        rag_note = "Retrieval-augmented context is enabled." if ctx.rag_enabled else "No retrieval context (baseline mode)."
        prompt = (
            "You are a 4G/5G mobile network fault analysis expert for core and RAN domains.\n"
            f"{rag_note}\n"
            "Return ONLY valid JSON (no markdown, no preamble) with exactly these keys:\n"
            '{"predicted_cause": "<string>", "confidence": <number 0..1>, '
            '"explanation": "<3-7 sentences as one string>", '
            '"recommended_actions": "<numbered operator steps as one string, not an array>"}\n\n'
            f"Symptoms:\n{ctx.symptoms}\n\n"
            f"Similar historical faults:\n{similar_block}\n"
        )
        msg = self._llm.invoke(prompt)
        raw = str(getattr(msg, "content", msg))
        try:
            obj = _normalize_telco_payload(_extract_json(raw))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM returned non-JSON output: {raw}") from e
        return TelcoFaultOutput.model_validate(obj)
