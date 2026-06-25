from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

from agentic_oran_rca.agents.auto_correction_agent import AutoCorrectionAgent, AutoCorrectionAgentConfig
from agentic_oran_rca.agents.master_orchestrator_agent import MasterOrchestratorAgent
from agentic_oran_rca.agents.context_agent import ContextRetrievalAgent
from agentic_oran_rca.agents.explanation_agent import ExplanationAgent, ExplanationAgentConfig
from agentic_oran_rca.agents.notification_service import HealingReport, NotificationService
from agentic_oran_rca.agents.remediation_agent import RemediationAgent
from agentic_oran_rca.agents.rca_agent import RCAAnalysisAgent, RCAAgentConfig
from agentic_oran_rca.agents.vector_database_update_agent import (
    HealingCorrectionRecord,
    VectorDatabaseUpdateAgent,
)
from agentic_oran_rca.config import Settings, load_settings
from agentic_oran_rca.evaluation.graph_generator import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_fault_distribution,
    plot_kpi_distribution,
    plot_network_topology_graph,
)
from agentic_oran_rca.evaluation.metrics import compute_metrics
from agentic_oran_rca.evaluation.ranking_metrics import run_ranking_evaluation
from agentic_oran_rca.evaluation.telco_explanation_eval import (
    export_telco_for_llm_validation,
    run_telco_evaluation,
)
from agentic_oran_rca.data.telco_llm_dataset import DEFAULT_TELCO_DIR, load_telco_split
from agentic_oran_rca.graph.graph_builder import (
    TopologySpec,
    generate_oran_topology,
    load_fault_affects_from_dataset,
    reset_and_load_neo4j,
)
from agentic_oran_rca.graph.neo4j_client import Neo4jClient, Neo4jConfig
from agentic_oran_rca.logging_utils import setup_logging
from agentic_oran_rca.rag.retriever import IncidentRetriever
from agentic_oran_rca.rag.telco_vector_store import TELCO_COLLECTION_NAME, build_telco_vector_store, upsert_telco_records
from agentic_oran_rca.rag.vector_store import (
    VectorStoreConfig,
    build_chroma_http_client,
    build_vector_store,
    upsert_incident_documents,
)
from agentic_oran_rca.telco_pipeline import build_telco_pipeline

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
RANKING_EVALUATION_DIR = RESULTS_DIR / "ranking_evaluation"
MTP1_DIR = PROJECT_ROOT / "MTP1"
MTP1_RESPONSES_DIR = MTP1_DIR / "responses"
MTP1_HEALING_REPORTS_DIR = MTP1_RESPONSES_DIR / "healing_reports"
MTP1_API_RESPONSES_DIR = MTP1_RESPONSES_DIR / "api"
MTP1_TELCO_DIR = MTP1_RESPONSES_DIR / "telco"
MTP1_NOTIFICATIONS_PATH = MTP1_RESPONSES_DIR / "notifications.jsonl"
TELCO_DATASET_DIR = DATA_DIR / "dataSet"
TELCO_EVAL_DIR = RESULTS_DIR / "telco_evaluation"


class RCAPipeline:
    def __init__(
        self,
        ctx_agent: ContextRetrievalAgent,
        rca_agent: RCAAnalysisAgent,
        expl_agent: ExplanationAgent,
    ) -> None:
        self._ctx_agent = ctx_agent
        self._rca_agent = rca_agent
        self._expl_agent = expl_agent

    def run(self, cell: str, alarm: str, kpi: str) -> dict[str, Any]:
        ctx = self._ctx_agent.run(cell_id=cell, alarm=alarm, k=7)
        rca = self._rca_agent.run(alarm=alarm, kpi=kpi, ctx=ctx)
        explanation = self._expl_agent.run(
            alarm=alarm,
            kpi=kpi,
            predicted_root_cause=rca.predicted_root_cause,
            confidence_score=rca.confidence_score,
            ctx=ctx,
        )
        return {
            "predicted_root_cause": rca.predicted_root_cause,
            "confidence": float(rca.confidence_score),
            "explanation": explanation,
        }


def build_pipeline(settings: Settings) -> RCAPipeline:
    neo4j = Neo4jClient(
        Neo4jConfig(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    )
    vs = build_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name="oran_rca_incidents",
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )
    retriever = IncidentRetriever(vs)

    ctx_agent = ContextRetrievalAgent(neo4j=neo4j, retriever=retriever)
    rca_agent = RCAAnalysisAgent(RCAAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model))
    expl_agent = ExplanationAgent(
        ExplanationAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model)
    )
    return RCAPipeline(ctx_agent=ctx_agent, rca_agent=rca_agent, expl_agent=expl_agent)


class HealingPipeline:
    """
    Self-healing extension: runs RCA (unchanged), then remediation, notification,
    and VectorDatabaseUpdateAgent when remediation succeeds.
    Does not modify the core RCA flow.
    """

    def __init__(
        self,
        rca_pipeline: RCAPipeline,
        ctx_agent: ContextRetrievalAgent,
        remediation_agent: RemediationAgent,
        notification_service: NotificationService,
        vector_update_agent: VectorDatabaseUpdateAgent,
    ) -> None:
        self._rca = rca_pipeline
        self._ctx_agent = ctx_agent
        self._remediation = remediation_agent
        self._notifier = notification_service
        self._vector_update_agent = vector_update_agent

    def run_with_healing(self, cell: str, alarm: str, kpi: str) -> dict[str, Any]:
        rca_out = self._rca.run(cell=cell, alarm=alarm, kpi=kpi)
        ctx = self._ctx_agent.run(cell_id=cell, alarm=alarm, k=7)
        rem = self._remediation.run(
            cell=cell,
            du=ctx.du,
            cu=ctx.cu,
            alarm=alarm,
            predicted_root_cause=rca_out["predicted_root_cause"],
            confidence=rca_out["confidence"],
        )
        report = HealingReport(
            timestamp_utc=rem.timestamp_utc,
            cell=cell,
            du=ctx.du,
            cu=ctx.cu,
            alarm=alarm,
            kpi=kpi,
            predicted_root_cause=rca_out["predicted_root_cause"],
            confidence=rca_out["confidence"],
            remediation_action=rem.action,
            remediation_success=rem.success,
            notification_type="auto_correction_success" if rem.success else "auto_correction_failure",
            message=rem.message,
        )
        report_path = self._notifier.write_healing_report(report)
        self._notifier.send_notification(
            notification_type=report.notification_type,
            cell=cell,
            alarm=alarm,
            root_cause=rca_out["predicted_root_cause"],
            remediation_success=rem.success,
            message=rem.message,
            report_path=report_path,
        )
        vector_update = self._vector_update_agent.run(
            HealingCorrectionRecord(
                cell=cell,
                du=ctx.du,
                cu=ctx.cu,
                alarm=alarm,
                kpi=kpi,
                root_cause=rca_out["predicted_root_cause"],
                timestamp_utc=rem.timestamp_utc,
                remediation_action=rem.action,
                remediation_success=rem.success,
            )
        )
        payload = {
            "predicted_root_cause": rca_out["predicted_root_cause"],
            "confidence": rca_out["confidence"],
            "explanation": rca_out["explanation"],
            "healing": {
                "remediation_action": rem.action,
                "remediation_description": rem.action_description,
                "success": rem.success,
                "message": rem.message,
                "report_path": str(report_path),
                "notification_sent": True,
                "vector_indexed": vector_update.vector_indexed,
                "chroma_document_id": vector_update.chroma_document_id,
                "vector_update_skipped_reason": vector_update.skipped_reason,
            },
        }
        self._notifier.write_api_response(endpoint="run_rca_with_healing", cell=cell, payload=payload)
        return payload


def build_healing_pipeline(settings: Settings) -> HealingPipeline:
    pipeline = build_pipeline(settings)
    neo4j = Neo4jClient(
        Neo4jConfig(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    )
    vs = build_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name="oran_rca_incidents",
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )
    ctx_agent = ContextRetrievalAgent(neo4j=neo4j, retriever=IncidentRetriever(vs))
    remediation = RemediationAgent(success_rate=0.85)
    notifier = NotificationService(
        reports_dir=MTP1_HEALING_REPORTS_DIR,
        notifications_path=MTP1_NOTIFICATIONS_PATH,
        api_responses_dir=MTP1_API_RESPONSES_DIR,
    )
    vector_update_agent = VectorDatabaseUpdateAgent(vs)
    return HealingPipeline(
        rca_pipeline=pipeline,
        ctx_agent=ctx_agent,
        remediation_agent=remediation,
        notification_service=notifier,
        vector_update_agent=vector_update_agent,
    )


def build_auto_correction_agent(settings: Settings) -> AutoCorrectionAgent:
    neo4j = Neo4jClient(
        Neo4jConfig(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    )
    vs = build_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name="oran_rca_incidents",
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )
    retriever = IncidentRetriever(vs)
    ctx_agent = ContextRetrievalAgent(neo4j=neo4j, retriever=retriever)
    rca_agent = RCAAnalysisAgent(RCAAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model))
    expl_agent = ExplanationAgent(
        ExplanationAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model)
    )
    return AutoCorrectionAgent(
        ctx_agent=ctx_agent,
        rca_agent=rca_agent,
        expl_agent=expl_agent,
        cfg=AutoCorrectionAgentConfig(ollama_base_url=settings.ollama_base_url, ollama_model=settings.ollama_model),
    )


def build_master_orchestrator(settings: Settings) -> MasterOrchestratorAgent:
    """
    Constructs a master orchestrator that wraps existing pipelines and agents.
    Does not replace or alter RCAPipeline, HealingPipeline, or AutoCorrectionAgent behavior.
    """
    rca_pipeline = build_pipeline(settings)
    healing_pipeline = build_healing_pipeline(settings)
    auto_correction_agent = build_auto_correction_agent(settings)
    return MasterOrchestratorAgent(
        rca_pipeline=rca_pipeline,
        healing_pipeline=healing_pipeline,
        auto_correction_agent=auto_correction_agent,
    )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")


def _load_dataset_rows(limit: int) -> list[dict[str, Any]]:
    _require_file(DATA_DIR / "telecom_dataset.csv")
    df = pd.read_csv(DATA_DIR / "telecom_dataset.csv")
    total = len(df)
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")
    if limit > total:
        raise ValueError(f"--limit {limit} exceeds dataset size {total}")
    if limit < total:
        logger.info("Using first %d of %d rows from telecom_dataset.csv (file unchanged)", limit, total)
    return df.head(limit).to_dict(orient="records")


def cmd_generate_data(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from agentic_oran_rca.data.dataset_generator import (
        DatasetSpec as DS,
        default_topology_for_rows,
        generate_enterprise_dataset,
        EnterpriseDatasetSpec,
    )

    topo_spec = default_topology_for_rows(rows=int(args.rows), seed=int(args.seed))
    topo = generate_oran_topology(topo_spec)
    (DATA_DIR / "topology.json").write_text(json.dumps(topo, indent=2), encoding="utf-8")

    df = generate_enterprise_dataset(
        topo=topo,
        spec=EnterpriseDatasetSpec(
            rows=int(args.rows),
            seed=int(args.seed),
            start_time_utc=args.start_time_utc,
            span_days=int(args.span_days),
        ),
    )
    df.to_csv(DATA_DIR / "telecom_dataset.csv", index=False)
    logger.info(
        "Generated enterprise dataset rows=%d columns=%d topology(cells=%d dus=%d cus=%d)",
        len(df),
        len(df.columns),
        topo_spec.num_cells,
        topo_spec.num_dus,
        topo_spec.num_cus,
    )


def cmd_build_graph(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    _require_file(DATA_DIR / "topology.json")

    topo = json.loads((DATA_DIR / "topology.json").read_text(encoding="utf-8"))
    rows = _load_dataset_rows(int(args.limit))

    neo4j = Neo4jClient(
        Neo4jConfig(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    )
    reset_and_load_neo4j(neo4j, topo)
    load_fault_affects_from_dataset(neo4j, rows)
    neo4j.close()
    logger.info("Neo4j graph loaded")


def cmd_index_vectors(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    rows = _load_dataset_rows(int(args.limit))

    # Clear and re-index deterministically by dropping and recreating the collection via raw Chroma client
    client = build_chroma_http_client(settings.chroma_host, settings.chroma_port)
    try:
        client.delete_collection(name="oran_rca_incidents")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to delete existing Chroma collection (possibly first run): %s", exc)

    vs = build_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name="oran_rca_incidents",
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )

    n = upsert_incident_documents(vs=vs, rows=rows, id_prefix="case_")
    logger.info("Indexed %d incidents into Chroma", n)


def _build_alarm_only_model(df: pd.DataFrame) -> SkPipeline:
    x = (df["alarm"].astype(str) + " | " + df["kpi"].astype(str)).tolist()
    y = df["expected_root_cause"].astype(str).tolist()

    model = SkPipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )
    model.fit(x, y)
    return model


def _build_context_aware_model(df: pd.DataFrame) -> SkPipeline:
    # Adds topology context columns (du, cu) to reduce ambiguity introduced in dataset generation.
    x = (df["alarm"].astype(str) + " | " + df["kpi"].astype(str) + " | " + df["du"].astype(str) + " | " + df["cu"].astype(str)).tolist()
    y = df["expected_root_cause"].astype(str).tolist()

    model = SkPipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )
    model.fit(x, y)
    return model


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    _require_file(DATA_DIR / "telecom_dataset.csv")
    _require_file(DATA_DIR / "topology.json")

    df = pd.read_csv(DATA_DIR / "telecom_dataset.csv")

    train_df, test_df = train_test_split(df, test_size=float(args.test_size), random_state=int(args.seed), stratify=df["expected_root_cause"])

    alarm_only = _build_alarm_only_model(train_df)
    context_aware = _build_context_aware_model(train_df)

    labels = sorted(df["expected_root_cause"].astype(str).unique().tolist())

    y_true = test_df["expected_root_cause"].astype(str).tolist()

    x_alarm = (test_df["alarm"].astype(str) + " | " + test_df["kpi"].astype(str)).tolist()
    pred_alarm = alarm_only.predict(x_alarm).tolist()
    m_alarm = compute_metrics(y_true=y_true, y_pred=pred_alarm, labels=labels)

    x_ctx = (
        test_df["alarm"].astype(str)
        + " | "
        + test_df["kpi"].astype(str)
        + " | "
        + test_df["du"].astype(str)
        + " | "
        + test_df["cu"].astype(str)
    ).tolist()
    pred_ctx = context_aware.predict(x_ctx).tolist()
    m_ctx = compute_metrics(y_true=y_true, y_pred=pred_ctx, labels=labels)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {"method": "Alarm-only RCA", "accuracy": m_alarm.accuracy, "precision_macro": m_alarm.precision_macro, "recall_macro": m_alarm.recall_macro, "f1_macro": m_alarm.f1_macro},
            {"method": "Context-aware RCA", "accuracy": m_ctx.accuracy, "precision_macro": m_ctx.precision_macro, "recall_macro": m_ctx.recall_macro, "f1_macro": m_ctx.f1_macro},
        ]
    )
    results.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    (RESULTS_DIR / "alarm_only_report.txt").write_text(m_alarm.report_text, encoding="utf-8")
    (RESULTS_DIR / "context_aware_report.txt").write_text(m_ctx.report_text, encoding="utf-8")

    plot_accuracy_comparison(GRAPHS_DIR, results)
    plot_confusion_matrix(GRAPHS_DIR, labels, m_alarm.confusion, "Confusion Matrix - Alarm-only", "confusion_matrix_alarm_only")
    plot_confusion_matrix(GRAPHS_DIR, labels, m_ctx.confusion, "Confusion Matrix - Context-aware", "confusion_matrix_context_aware")
    plot_fault_distribution(GRAPHS_DIR, df)
    plot_kpi_distribution(GRAPHS_DIR, df)

    topo = json.loads((DATA_DIR / "topology.json").read_text(encoding="utf-8"))
    edges: list[tuple[str, str]] = []
    for cell, nbrs in topo["neighbors"].items():
        for nbr in nbrs:
            edges.append((cell, nbr))
    plot_network_topology_graph(GRAPHS_DIR, edges, "network_topology")

    logger.info("Saved evaluation results to %s", RESULTS_DIR)


def cmd_evaluate_ranking(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    _require_file(DATA_DIR / "telecom_dataset.csv")
    df = pd.read_csv(DATA_DIR / "telecom_dataset.csv")

    k_values = [int(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
    if not k_values:
        raise RuntimeError("k_values must list at least one integer K")

    vs = build_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name="oran_rca_incidents",
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )
    retriever = IncidentRetriever(vs)

    report_name = str(getattr(args, "report_name", "")).strip()
    if not report_name:
        report_name = f"Chroma+{settings.ollama_embed_model}"

    run_ranking_evaluation(
        df,
        retriever,
        test_size=float(args.test_size),
        seed=int(args.seed),
        k_values=k_values,
        retrieve_pool=int(args.retrieve_pool),
        output_dir=RANKING_EVALUATION_DIR,
        report_model_name=report_name,
    )


def _parse_bool_flag(value: str, flag_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{flag_name} must be 'true' or 'false', got {value!r}")


def _resolve_telco_dir(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    return TELCO_DATASET_DIR


def cmd_index_telco_vectors(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    data_dir = _resolve_telco_dir(args.data_dir)
    train_records = load_telco_split(data_dir, "train")
    if not train_records:
        raise RuntimeError(f"No train records found under {data_dir}")

    client = build_chroma_http_client(settings.chroma_host, settings.chroma_port)
    try:
        client.delete_collection(name=TELCO_COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to delete existing telco Chroma collection (possibly first run): %s", exc)

    vs = build_telco_vector_store(
        VectorStoreConfig(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name=TELCO_COLLECTION_NAME,
            ollama_base_url=settings.ollama_base_url,
            ollama_embed_model=settings.ollama_embed_model,
        )
    )
    n = upsert_telco_records(vs=vs, records=train_records)
    logger.info("Indexed %d telco train records into Chroma collection %s", n, TELCO_COLLECTION_NAME)


def cmd_evaluate_telco(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    data_dir = _resolve_telco_dir(args.data_dir)
    records = load_telco_split(data_dir, args.split)
    use_rag = _parse_bool_flag(args.use_rag, "--use-rag")
    summary = run_telco_evaluation(
        records,
        settings,
        split=args.split,
        use_rag=use_rag,
        output_dir=TELCO_EVAL_DIR,
        limit=int(args.limit),
    )
    logger.info(
        "Telco eval (%s, rag=%s): n=%d action_overlap=%.4f cause_overlap=%.4f combined=%.4f",
        summary.split,
        summary.use_rag,
        summary.n_evaluated,
        summary.mean_action_token_overlap,
        summary.mean_cause_token_overlap,
        summary.mean_combined_text_overlap,
    )


def cmd_run_telco_sample(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    data_dir = _resolve_telco_dir(args.data_dir)
    records = load_telco_split(data_dir, args.split)
    index = int(args.index)
    if index < 0 or index >= len(records):
        raise ValueError(f"--index {index} out of range for split {args.split!r} (size {len(records)})")

    record = records[index]
    use_rag = _parse_bool_flag(args.use_rag, "--use-rag")
    pipeline = build_telco_pipeline(settings, with_retriever=use_rag)
    out = pipeline.run(symptoms=record.symptoms, use_rag=use_rag)

    MTP1_TELCO_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    rag_tag = "rag" if use_rag else "no_rag"
    filename = f"telco_{record.record_id}_{rag_tag}_{ts}.json"
    payload = {
        "record_id": record.record_id,
        "split": record.split,
        "index": index,
        "symptoms": record.symptoms,
        "reference_cause": record.reference_cause,
        "reference_actions": record.reference_actions,
        "use_rag": use_rag,
        "predicted_cause": out["predicted_cause"],
        "confidence": out["confidence"],
        "explanation": out["explanation"],
        "recommended_actions": out["recommended_actions"],
        "similar_incidents": out["similar_incidents"],
    }
    out_path = MTP1_TELCO_DIR / filename
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote telco sample response to %s", out_path)


def cmd_export_telco_json(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    data_dir = _resolve_telco_dir(args.data_dir)
    records = load_telco_split(data_dir, args.split)
    use_rag = _parse_bool_flag(args.use_rag, "--use-rag")
    output_path = Path(args.output)
    export_telco_for_llm_validation(records, settings, output_path, use_rag=use_rag)
    logger.info("Exported telco JSON for LLM backend validation to %s", output_path)


def cmd_serve(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    import uvicorn

    uvicorn.run("agentic_oran_rca.api.server:app", host=args.host, port=int(args.port), reload=False, log_level=settings.log_level.lower())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic_oran_rca")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data")
    g.add_argument("--rows", type=int, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--start-time-utc", type=str, required=True)
    g.add_argument("--span-days", type=int, required=True)
    g.add_argument("--log-level", type=str, required=True)
    g.set_defaults(func=cmd_generate_data)

    b = sub.add_parser("build-graph")
    b.add_argument("--limit", type=int, required=True, help="Use first N rows from telecom_dataset.csv")
    b.set_defaults(func=cmd_build_graph)

    i = sub.add_parser("index-vectors")
    i.add_argument("--limit", type=int, required=True, help="Use first N rows from telecom_dataset.csv")
    i.set_defaults(func=cmd_index_vectors)

    e = sub.add_parser("evaluate")
    e.add_argument("--test-size", type=float, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.set_defaults(func=cmd_evaluate)

    rk = sub.add_parser("evaluate-ranking")
    rk.add_argument("--test-size", type=float, required=True)
    rk.add_argument("--seed", type=int, required=True)
    rk.add_argument("--k-values", type=str, required=True, help="Comma-separated K values, e.g. 1,3,5,7,10")
    rk.add_argument("--retrieve-pool", type=int, required=True, help="Chroma hits per query before dedupe (>= max K)")
    rk.add_argument(
        "--report-name",
        type=str,
        default="",
        help="Label for the model/system row in difficulty tables (default: Chroma+<OLLAMA_EMBED_MODEL>)",
    )
    rk.set_defaults(func=cmd_evaluate_ranking)

    s = sub.add_parser("serve")
    s.add_argument("--host", type=str, required=True)
    s.add_argument("--port", type=int, required=True)
    s.set_defaults(func=cmd_serve)

    tv = sub.add_parser("index-telco-vectors")
    tv.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to telco JSONL directory (default: data/dataSet)",
    )
    tv.set_defaults(func=cmd_index_telco_vectors)

    te = sub.add_parser("evaluate-telco")
    te.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    te.add_argument("--limit", type=int, required=True, help="Evaluate first N records from the split")
    te.add_argument("--use-rag", type=str, required=True, help="true or false")
    te.add_argument("--data-dir", type=str, default=None)
    te.set_defaults(func=cmd_evaluate_telco)

    ts = sub.add_parser("run-telco-sample")
    ts.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    ts.add_argument("--index", type=int, required=True, help="Zero-based record index within the split")
    ts.add_argument("--use-rag", type=str, required=True, help="true or false")
    ts.add_argument("--data-dir", type=str, default=None)
    ts.set_defaults(func=cmd_run_telco_sample)

    ex = sub.add_parser("export-telco-json")
    ex.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    ex.add_argument("--use-rag", type=str, required=True, help="true or false")
    ex.add_argument("--output", type=str, required=True, help="Output JSON path")
    ex.add_argument("--data-dir", type=str, default=None)
    ex.set_defaults(func=cmd_export_telco_json)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

