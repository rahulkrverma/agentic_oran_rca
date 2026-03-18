from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _save_fig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(out_dir: Path, results: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=results, x="method", y="accuracy")
    plt.ylim(0, 1.0)
    plt.title("RCA Accuracy Comparison")
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    _save_fig(out_dir, "accuracy_comparison")


def plot_confusion_matrix(out_dir: Path, labels: list[str], cm: np.ndarray, title: str, name: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _save_fig(out_dir, name)


def plot_fault_distribution(out_dir: Path, df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="expected_root_cause", order=df["expected_root_cause"].value_counts().index)
    plt.xticks(rotation=30, ha="right")
    plt.title("Fault Distribution")
    plt.xlabel("Root cause")
    plt.ylabel("Count")
    _save_fig(out_dir, "fault_distribution_hist")


def plot_kpi_distribution(out_dir: Path, df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="kpi", order=df["kpi"].value_counts().index)
    plt.xticks(rotation=30, ha="right")
    plt.title("KPI Anomaly Distribution")
    plt.xlabel("KPI anomaly")
    plt.ylabel("Count")
    _save_fig(out_dir, "kpi_anomaly_distribution")


def plot_network_topology_graph(out_dir: Path, edges: list[tuple[str, str]], name: str) -> None:
    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(edges)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(g, seed=42, k=0.25)
    nx.draw_networkx_nodes(g, pos, node_size=120, node_color="#4C78A8", alpha=0.9)
    nx.draw_networkx_edges(g, pos, width=0.8, alpha=0.4)
    nx.draw_networkx_labels(g, pos, font_size=7, font_color="black")
    plt.title("Cell Neighbor Topology (Synthetic)")
    plt.axis("off")
    _save_fig(out_dir, name)

