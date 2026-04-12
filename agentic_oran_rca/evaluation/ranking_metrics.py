from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from agentic_oran_rca.graph.graph_builder import ALARM_TO_FAULT
from agentic_oran_rca.rag.retriever import IncidentRetriever, SimilarIncident

logger = logging.getLogger(__name__)

# Indexed incident text shape (see vector_store.upsert_incident_documents).
_INCIDENT_DOC_RE = re.compile(
    r"^cell=(.*?) du=(.*?) cu=(.*?) alarm=(.*?) kpi=(.*?) root_cause=(.*?) timestamp=(.*)$",
    re.DOTALL,
)

# Display order matches common paper tables: Simple, Difficult, Mixed.
_DIFFICULTY_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("simple", "Simple"),
    ("difficult", "Difficult"),
    ("mixed", "Mixed"),
)


def difficulty_tier_for_alarm(alarm: str) -> str:
    """
    Maps an alarm to a coarse difficulty tier from ALARM_TO_FAULT cardinality:
    - simple: a single plausible root-cause class for the alarm
    - mixed: two or three candidates (moderate ambiguity)
    - difficult: four or more candidates (high alarm-only ambiguity)
    """
    n = len(ALARM_TO_FAULT.get(str(alarm), []))
    if n <= 1:
        return "simple"
    if n <= 3:
        return "mixed"
    return "difficult"


def _normalize_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError, OSError, OverflowError):
        return str(value).strip()
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()


def _row_match_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["cell"]).strip(),
        _normalize_timestamp(row.get("timestamp")),
        str(row["alarm"]).strip(),
        str(row["kpi"]).strip(),
    )


def _meta_match_key(meta: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(meta.get("cell", "")).strip(),
        _normalize_timestamp(meta.get("timestamp")),
        str(meta.get("alarm", "")).strip(),
        str(meta.get("kpi", "")).strip(),
    )


def _parse_incident_doc(text: str) -> dict[str, str] | None:
    m = _INCIDENT_DOC_RE.match(text.strip())
    if not m:
        return None
    cell, _du, _cu, alarm, kpi, root_cause, ts = m.groups()
    return {
        "cell": cell.strip(),
        "alarm": alarm.strip(),
        "kpi": kpi.strip(),
        "root_cause": root_cause.strip(),
        "timestamp": _normalize_timestamp(ts.strip()),
    }


def _hit_identity_key(hit: SimilarIncident) -> tuple[str, str, str, str] | None:
    parsed = _parse_incident_doc(hit.text)
    if parsed is None:
        return None
    return (
        parsed["cell"],
        parsed["timestamp"],
        parsed["alarm"],
        parsed["kpi"],
    )


def _root_cause_from_hit(hit: SimilarIncident) -> str:
    for key in ("expected_root_cause", "root_cause"):
        raw = hit.metadata.get(key)
        if raw is not None:
            s = str(raw).strip()
            if s:
                return s
    parsed = _parse_incident_doc(hit.text)
    if parsed is not None:
        return parsed["root_cause"].strip()
    return ""


def _is_self_hit(row: dict[str, Any], hit: SimilarIncident) -> bool:
    rk = _row_match_key(row)
    if _meta_match_key(hit.metadata) == rk:
        return True
    hk = _hit_identity_key(hit)
    return hk is not None and hk == rk


def _ranked_root_causes(hits: list[SimilarIncident]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for h in hits:
        rc = _root_cause_from_hit(h)
        if not rc or rc in seen:
            continue
        seen.add(rc)
        order.append(rc)
    return order


def precision_recall_f1_at_k(gold: str, ranked_labels: list[str], k: int) -> tuple[float, float, float]:
    """
    Single relevant label. Top-k is the first k entries of ranked_labels.
    P@K = |rel ∩ top_k| / K, R@K = |rel ∩ top_k| / |rel|, F1@K harmonic mean.
    """
    if k <= 0:
        return 0.0, 0.0, 0.0
    top = ranked_labels[:k]
    rel_hit = 1 if gold in top else 0
    p = rel_hit / float(k)
    r = float(rel_hit)
    if p + r <= 0.0:
        return p, r, 0.0
    f1 = 2.0 * p * r / (p + r)
    return p, r, f1


@dataclass(frozen=True)
class RankingEvalSummary:
    generated_at_utc: str
    n_test: int
    test_size: float
    seed: int
    retrieve_pool: int
    k_values: list[int]
    mean_precision_at_k: dict[str, float]
    mean_recall_at_k: dict[str, float]
    mean_f1_at_k: dict[str, float]


def _mean_prf_for_tier(
    per_query: list[dict[str, Any]],
    k_key: str,
    tier: str,
) -> tuple[float, float, float, int]:
    rows = [pq for pq in per_query if pq["difficulty_tier"] == tier]
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    acc: list[tuple[float, float, float]] = []
    for pq in rows:
        m = pq["metrics_at_k"][k_key]
        acc.append((m["precision"], m["recall"], m["f1"]))
    return (
        sum(t[0] for t in acc) / n,
        sum(t[1] for t in acc) / n,
        sum(t[2] for t in acc) / n,
        n,
    )


def _write_difficulty_flat_csv(
    path: Path,
    model_name: str,
    sorted_k: list[int],
    per_query: list[dict[str, Any]],
) -> None:
    header: list[str] = ["Model"]
    for kk in sorted_k:
        for tier_key, tier_label in _DIFFICULTY_DISPLAY_ORDER:
            header.append(f"Precision@{kk}_{tier_label}")
        for tier_key, tier_label in _DIFFICULTY_DISPLAY_ORDER:
            header.append(f"Recall@{kk}_{tier_label}")
        for tier_key, tier_label in _DIFFICULTY_DISPLAY_ORDER:
            header.append(f"F1@{kk}_{tier_label}")
    row: list[str | float] = [model_name]
    for kk in sorted_k:
        k_key = str(kk)
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            row.append(round(p, 4) if n else 0.0)
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            row.append(round(r, 4) if n else 0.0)
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            row.append(round(f1, 4) if n else 0.0)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def _write_difficulty_table_txt(
    path: Path,
    model_name: str,
    sorted_k: list[int],
    per_query: list[dict[str, Any]],
) -> None:
    """
    Paper-style table: for each K, Precision@K / Recall@K / F1@K × (Simple, Difficult, Mixed).
    """
    w_model = 34
    w_cell = 12
    gap = "  "
    sub_hdr = gap.join(f"{lbl:>{w_cell}}" for _, lbl in _DIFFICULTY_DISPLAY_ORDER)
    block_w = len(sub_hdr)
    lines: list[str] = [
        "Mean Precision@K, Recall@K, and F1@K by alarm difficulty (held-out test queries).",
        "Tiers from ALARM_TO_FAULT cardinality: Simple = 1 candidate; Mixed = 2–3; Difficult = 4+.",
        "",
    ]

    for kk in sorted_k:
        k_key = str(kk)
        lines.append(f"K = {kk}")
        title_p = f"Precision@{kk}"
        title_r = f"Recall@{kk}"
        title_f = f"F1@{kk}"
        lines.append(
            f"{'Model':<{w_model}}"
            f"{title_p:^{block_w}}{gap}"
            f"{title_r:^{block_w}}{gap}"
            f"{title_f:^{block_w}}"
        )
        lines.append(f"{'':<{w_model}}{sub_hdr}{gap}{sub_hdr}{gap}{sub_hdr}")
        vals_p: list[str] = []
        vals_r: list[str] = []
        vals_f: list[str] = []
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            vals_p.append(f"{p:>{w_cell}.4f}" if n else f"{'—':>{w_cell}}")
            vals_r.append(f"{r:>{w_cell}.4f}" if n else f"{'—':>{w_cell}}")
            vals_f.append(f"{f1:>{w_cell}.4f}" if n else f"{'—':>{w_cell}}")
        row = (
            f"{model_name[:w_model]:<{w_model}}"
            f"{gap.join(vals_p)}{gap}"
            f"{gap.join(vals_r)}{gap}"
            f"{gap.join(vals_f)}"
        )
        lines.append(row)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_difficulty_markdown(
    path: Path,
    model_name: str,
    sorted_k: list[int],
    per_query: list[dict[str, Any]],
) -> None:
    lines: list[str] = ["# Ranking metrics by difficulty", ""]
    for kk in sorted_k:
        k_key = str(kk)
        lines.append(f"## K = {kk}")
        hdr = (
            "| Model | "
            + " | ".join(
                [f"P@{kk} {lbl}" for _, lbl in _DIFFICULTY_DISPLAY_ORDER]
                + [f"R@{kk} {lbl}" for _, lbl in _DIFFICULTY_DISPLAY_ORDER]
                + [f"F1@{kk} {lbl}" for _, lbl in _DIFFICULTY_DISPLAY_ORDER]
            )
            + " |"
        )
        sep = "| " + " | ".join(["---"] * (1 + 3 * 3)) + " |"
        cells: list[str] = []
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            p, _, _, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            cells.append(f"{p:.4f}" if n else "—")
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            _, r, _, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            cells.append(f"{r:.4f}" if n else "—")
        for tier_key, _ in _DIFFICULTY_DISPLAY_ORDER:
            _, _, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            cells.append(f"{f1:.4f}" if n else "—")
        row = "| " + model_name.replace("|", "\\|") + " | " + " | ".join(cells) + " |"
        lines.extend([hdr, sep, row, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ranking_evaluation(
    df: pd.DataFrame,
    retriever: IncidentRetriever,
    *,
    test_size: float,
    seed: int,
    k_values: list[int],
    retrieve_pool: int,
    output_dir: Path,
    report_model_name: str,
) -> tuple[Path, Path]:
    """
    Evaluates retrieval ranking: ranked unique root causes from Chroma (excluding self-hit).
    Writes JSON summary and a text report under output_dir.
    """
    if not k_values:
        raise ValueError("k_values must be non-empty")
    k_max = max(k_values)
    if retrieve_pool < k_max:
        raise ValueError("retrieve_pool must be >= max(k_values)")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["expected_root_cause"],
    )

    rows = test_df.to_dict(orient="records")
    per_query: list[dict[str, Any]] = []

    for row in rows:
        query = f"cell={row['cell']} du={row['du']} cu={row['cu']} alarm={row['alarm']}"
        raw = retriever.retrieve(query=query, k=retrieve_pool)
        filtered = [h for h in raw if not _is_self_hit(row, h)]
        ranked = _ranked_root_causes(filtered)
        if not ranked and raw:
            logger.warning(
                "Retrieval returned %d hits but ranked root causes empty (cell=%s); check metadata vs document text",
                len(raw),
                row.get("cell"),
            )
        elif not raw:
            logger.warning("Retrieval returned no hits for cell=%s", row.get("cell"))
        gold = str(row["expected_root_cause"])

        metrics_k: dict[str, dict[str, float]] = {}
        for kk in sorted(set(k_values)):
            p, r, f1 = precision_recall_f1_at_k(gold, ranked, kk)
            metrics_k[str(kk)] = {"precision": p, "recall": r, "f1": f1}

        per_query.append(
            {
                "cell": row["cell"],
                "alarm": row["alarm"],
                "kpi": row["kpi"],
                "expected_root_cause": gold,
                "difficulty_tier": difficulty_tier_for_alarm(str(row["alarm"])),
                "ranked_root_causes": ranked[: k_max + 5],
                "metrics_at_k": metrics_k,
            }
        )

    mean_p: dict[str, float] = {}
    mean_r: dict[str, float] = {}
    mean_f: dict[str, float] = {}
    sorted_k = sorted(set(k_values))
    for kk in sorted_k:
        acc: list[tuple[float, float, float]] = []
        for pq in per_query:
            m = pq["metrics_at_k"][str(kk)]
            acc.append((m["precision"], m["recall"], m["f1"]))
        mp, mr, mf = (
            sum(t[0] for t in acc) / len(acc),
            sum(t[1] for t in acc) / len(acc),
            sum(t[2] for t in acc) / len(acc),
        )
        key = str(kk)
        mean_p[key] = mp
        mean_r[key] = mr
        mean_f[key] = mf

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = RankingEvalSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        n_test=len(rows),
        test_size=float(test_size),
        seed=int(seed),
        retrieve_pool=int(retrieve_pool),
        k_values=sorted_k,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_f1_at_k=mean_f,
    )

    mean_by_difficulty_at_k: dict[str, Any] = {}
    for kk in sorted_k:
        k_key = str(kk)
        tier_block: dict[str, Any] = {}
        for tier_key, tier_label in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            tier_block[tier_key] = {
                "label": tier_label,
                "n": n,
                "precision": round(p, 6) if n else 0.0,
                "recall": round(r, 6) if n else 0.0,
                "f1": round(f1, 6) if n else 0.0,
            }
        mean_by_difficulty_at_k[k_key] = tier_block

    summary_path = output_dir / f"ranking_metrics_summary_{ts}.json"
    latest_path = output_dir / "ranking_metrics_summary_latest.json"
    payload = {
        **asdict(summary),
        "report_model_name": report_model_name,
        "difficulty_tier_definition": {
            "simple": "Alarms with exactly one candidate root cause in ALARM_TO_FAULT",
            "mixed": "Two or three candidates",
            "difficult": "Four or more candidates",
        },
        "mean_by_difficulty_at_k": mean_by_difficulty_at_k,
        "per_query_sample": per_query[: min(20, len(per_query))],
        "per_query_total": len(per_query),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Ranking evaluation (Precision@K, Recall@K, F1@K)",
        f"Generated (UTC): {summary.generated_at_utc}",
        f"n_test={summary.n_test} test_size={summary.test_size} seed={summary.seed} retrieve_pool={summary.retrieve_pool}",
        "",
        "Mean metrics (macro over test queries):",
    ]
    for kk in sorted_k:
        key = str(kk)
        lines.append(
            f"  K={kk}: P@{kk}={mean_p[key]:.4f}  R@{kk}={mean_r[key]:.4f}  F1@{kk}={mean_f[key]:.4f}"
        )
    lines.append("")
    lines.append(f"Full JSON: {summary_path.name}")
    lines.append(f"Latest JSON: {latest_path.name}")
    lines.append("")
    lines.append("By difficulty (Simple / Difficult / Mixed), mean P/R/F1:")
    for kk in sorted_k:
        k_key = str(kk)
        lines.append(f"  K={kk}:")
        for tier_key, tier_label in _DIFFICULTY_DISPLAY_ORDER:
            p, r, f1, n = _mean_prf_for_tier(per_query, k_key, tier_key)
            if n:
                lines.append(
                    f"    {tier_label}: n={n}  P@{kk}={p:.4f}  R@{kk}={r:.4f}  F1@{kk}={f1:.4f}"
                )
            else:
                lines.append(f"    {tier_label}: n=0  (no test rows)")
    lines.append("")
    lines.append(f"Paper-style table (TXT): ranking_metrics_difficulty_table_{ts}.txt")
    lines.append(f"Paper-style table (CSV wide): ranking_metrics_difficulty_flat_{ts}.csv")
    lines.append(f"Paper-style table (Markdown): ranking_metrics_difficulty_table_{ts}.md")

    report_path = output_dir / f"ranking_metrics_report_{ts}.txt"
    latest_report = output_dir / "ranking_metrics_report_latest.txt"
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    latest_report.write_text(text, encoding="utf-8")

    csv_path = output_dir / f"ranking_metrics_by_k_{ts}.csv"
    csv_latest = output_dir / "ranking_metrics_by_k_latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "mean_precision_at_k", "mean_recall_at_k", "mean_f1_at_k"])
        for kk in sorted_k:
            key = str(kk)
            w.writerow([kk, mean_p[key], mean_r[key], mean_f[key]])
    shutil.copyfile(csv_path, csv_latest)

    diff_flat = output_dir / f"ranking_metrics_difficulty_flat_{ts}.csv"
    diff_flat_latest = output_dir / "ranking_metrics_difficulty_flat_latest.csv"
    _write_difficulty_flat_csv(diff_flat, report_model_name, sorted_k, per_query)
    shutil.copyfile(diff_flat, diff_flat_latest)

    diff_txt = output_dir / f"ranking_metrics_difficulty_table_{ts}.txt"
    diff_txt_latest = output_dir / "ranking_metrics_difficulty_table_latest.txt"
    _write_difficulty_table_txt(diff_txt, report_model_name, sorted_k, per_query)
    shutil.copyfile(diff_txt, diff_txt_latest)

    diff_md = output_dir / f"ranking_metrics_difficulty_table_{ts}.md"
    diff_md_latest = output_dir / "ranking_metrics_difficulty_table_latest.md"
    _write_difficulty_markdown(diff_md, report_model_name, sorted_k, per_query)
    shutil.copyfile(diff_md, diff_md_latest)

    logger.info("Wrote ranking evaluation to %s and %s", summary_path, report_path)
    return summary_path, report_path
