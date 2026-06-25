from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_oran_rca.config import Settings
from agentic_oran_rca.data.telco_llm_dataset import TelcoFaultRecord, export_llm_backend_incidents
from agentic_oran_rca.telco_pipeline import build_telco_pipeline

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def token_overlap_score(reference: str, generated: str) -> float:
    ref = _tokenize(reference)
    gen = _tokenize(generated)
    if not ref or not gen:
        return 0.0
    return len(ref & gen) / len(ref)


@dataclass(frozen=True)
class TelcoEvalSummary:
    split: str
    n_evaluated: int
    use_rag: bool
    mean_action_token_overlap: float
    mean_cause_token_overlap: float
    mean_combined_text_overlap: float


def _run_one(
    pipeline,
    record: TelcoFaultRecord,
    use_rag: bool,
) -> dict[str, Any]:
    out = pipeline.run(symptoms=record.symptoms, use_rag=use_rag)
    action_overlap = token_overlap_score(record.reference_actions, out["recommended_actions"])
    cause_overlap = token_overlap_score(record.reference_cause, out["predicted_cause"])
    combined = token_overlap_score(
        record.reference_cause + " " + record.reference_actions,
        out["predicted_cause"] + " " + out["explanation"] + " " + out["recommended_actions"],
    )
    return {
        "record_id": record.record_id,
        "symptoms": record.symptoms,
        "reference_cause": record.reference_cause,
        "reference_actions": record.reference_actions,
        "use_rag": use_rag,
        "predicted_cause": out["predicted_cause"],
        "confidence": out["confidence"],
        "explanation": out["explanation"],
        "recommended_actions": out["recommended_actions"],
        "similar_incidents": out["similar_incidents"],
        "action_token_overlap": action_overlap,
        "cause_token_overlap": cause_overlap,
        "combined_text_overlap": combined,
    }


def run_telco_evaluation(
    records: list[TelcoFaultRecord],
    settings: Settings,
    *,
    split: str,
    use_rag: bool,
    output_dir: Path,
    limit: int,
) -> TelcoEvalSummary:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(records):
        raise ValueError(f"limit {limit} exceeds split size {len(records)}")

    subset = records[:limit]
    pipeline = build_telco_pipeline(settings, with_retriever=use_rag)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query: list[dict[str, Any]] = []
    for record in subset:
        logger.info("Evaluating %s (rag=%s)", record.record_id, use_rag)
        per_query.append(_run_one(pipeline, record, use_rag))

    summary = TelcoEvalSummary(
        split=split,
        n_evaluated=len(per_query),
        use_rag=use_rag,
        mean_action_token_overlap=sum(r["action_token_overlap"] for r in per_query) / len(per_query),
        mean_cause_token_overlap=sum(r["cause_token_overlap"] for r in per_query) / len(per_query),
        mean_combined_text_overlap=sum(r["combined_text_overlap"] for r in per_query) / len(per_query),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rag_tag = "rag" if use_rag else "no_rag"
    detail_path = output_dir / f"telco_eval_{split}_{rag_tag}_{ts}.json"
    summary_path = output_dir / f"telco_eval_summary_{split}_{rag_tag}_{ts}.json"
    csv_path = output_dir / f"telco_eval_{split}_{rag_tag}_{ts}.csv"

    detail_path.write_text(json.dumps(per_query, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "action_token_overlap",
                "cause_token_overlap",
                "combined_text_overlap",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in per_query:
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "action_token_overlap": row["action_token_overlap"],
                    "cause_token_overlap": row["cause_token_overlap"],
                    "combined_text_overlap": row["combined_text_overlap"],
                    "confidence": row["confidence"],
                }
            )

    report_path = output_dir / f"telco_eval_report_{split}_{rag_tag}_{ts}.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Telco fault evaluation ({split}, rag={use_rag})",
                "",
                f"- Records evaluated: {summary.n_evaluated}",
                f"- Mean action token overlap: {summary.mean_action_token_overlap:.4f}",
                f"- Mean cause token overlap: {summary.mean_cause_token_overlap:.4f}",
                f"- Mean combined text overlap: {summary.mean_combined_text_overlap:.4f}",
                "",
                f"Detail JSON: `{detail_path.name}`",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Wrote telco evaluation to %s", output_dir)
    return summary


def export_telco_for_llm_validation(
    records: list[TelcoFaultRecord],
    settings: Settings,
    output_path: Path,
    *,
    use_rag: bool,
) -> None:
    pipeline = build_telco_pipeline(settings, with_retriever=use_rag)
    context_by_id: dict[str, str] = {}
    for r in records:
        if use_rag:
            ctx = pipeline.ctx_agent.run(symptoms=r.symptoms, k=7, use_rag=True)
            context_by_id[r.record_id] = "\n".join(s["text"][:300] for s in ctx.similar_incidents[:3])
        else:
            context_by_id[r.record_id] = ""
    payload = export_llm_backend_incidents(records, context_by_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported %d telco incidents to %s", len(payload), output_path)
