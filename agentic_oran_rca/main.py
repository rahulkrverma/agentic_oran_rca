from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

from agentic_oran_rca.agents.context_agent import ContextRetrievalAgent
from agentic_oran_rca.agents.explanation_agent import ExplanationAgent, ExplanationAgentConfig
from agentic_oran_rca.agents.rca_agent import RCAAnalysisAgent, RCAAgentConfig
from agentic_oran_rca.config import Settings, load_settings
from agentic_oran_rca.evaluation.graph_generator import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_fault_distribution,
    plot_kpi_distribution,
    plot_network_topology_graph,
)
from agentic_oran_rca.evaluation.metrics import compute_metrics
from agentic_oran_rca.graph.graph_builder import (
    TopologySpec,
    generate_oran_topology,
    load_fault_affects_from_dataset,
    reset_and_load_neo4j,
)
from agentic_oran_rca.graph.neo4j_client import Neo4jClient, Neo4jConfig
from agentic_oran_rca.logging_utils import setup_logging
from agentic_oran_rca.rag.retriever import IncidentRetriever
from agentic_oran_rca.rag.vector_store import (
    VectorStoreConfig,
    build_chroma_http_client,
    build_vector_store,
    upsert_incident_documents,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


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


def _require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")


def cmd_generate_data(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    topo = generate_oran_topology(
        TopologySpec(num_cus=2, num_dus=6, num_cells=40, neighbor_k=3, seed=int(args.seed))
    )
    (DATA_DIR / "topology.json").write_text(json.dumps(topo, indent=2), encoding="utf-8")

    from agentic_oran_rca.data.dataset_generator import DatasetSpec as DS, generate_synthetic_dataset

    df = generate_synthetic_dataset(
        topo=topo,
        spec=DS(rows=int(args.rows), seed=int(args.seed), start_time_utc=args.start_time_utc, span_days=int(args.span_days)),
    )
    df.to_csv(DATA_DIR / "telecom_dataset.csv", index=False)
    logger.info("Generated dataset rows=%d", len(df))


def cmd_build_graph(_: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    _require_file(DATA_DIR / "topology.json")
    _require_file(DATA_DIR / "telecom_dataset.csv")

    topo = json.loads((DATA_DIR / "topology.json").read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_DIR / "telecom_dataset.csv")
    rows = df.to_dict(orient="records")

    neo4j = Neo4jClient(
        Neo4jConfig(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    )
    reset_and_load_neo4j(neo4j, topo)
    load_fault_affects_from_dataset(neo4j, rows)
    neo4j.close()
    logger.info("Neo4j graph loaded")


def cmd_index_vectors(_: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    _require_file(DATA_DIR / "telecom_dataset.csv")
    df = pd.read_csv(DATA_DIR / "telecom_dataset.csv")

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

    n = upsert_incident_documents(vs=vs, rows=df.to_dict(orient="records"), id_prefix="case_")
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
    b.set_defaults(func=cmd_build_graph)

    i = sub.add_parser("index-vectors")
    i.set_defaults(func=cmd_index_vectors)

    e = sub.add_parser("evaluate")
    e.add_argument("--test-size", type=float, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("serve")
    s.add_argument("--host", type=str, required=True)
    s.add_argument("--port", type=int, required=True)
    s.set_defaults(func=cmd_serve)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

