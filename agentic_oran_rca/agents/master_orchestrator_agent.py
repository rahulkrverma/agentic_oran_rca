"""
Master orchestrator agent: single entry point that delegates to existing pipelines
and agents without replacing or modifying their implementations.
"""

from __future__ import annotations

from typing import Any, Literal

from agentic_oran_rca.agents.auto_correction_agent import AutoCorrectionAgent

TaskName = Literal["rca", "rca_with_healing", "auto_correction"]


class MasterOrchestratorAgent:
    """
    Coordinates RCA workflows by calling the same objects used elsewhere in the app:
    - RCAPipeline: context retrieval → RCA → explanation
    - HealingPipeline: RCAPipeline output → remediation → notification persistence
    - AutoCorrectionAgent: context → RCA → explanation → reviewer correction pass

    This class does not implement duplicate logic; it only routes `run()` to the
    appropriate existing pipeline or agent.
    """

    def __init__(
        self,
        rca_pipeline: Any,
        healing_pipeline: Any,
        auto_correction_agent: AutoCorrectionAgent,
    ) -> None:
        self._rca_pipeline = rca_pipeline
        self._healing_pipeline = healing_pipeline
        self._auto_correction_agent = auto_correction_agent

    def tasks(self) -> list[dict[str, str]]:
        """Human-readable list of orchestrated task identifiers and descriptions."""
        return [
            {
                "task": "rca",
                "description": "Standard RCA: Neo4j/Chroma context, LLM root cause, explanation.",
            },
            {
                "task": "rca_with_healing",
                "description": "RCA plus remediation simulation, healing report, notification log.",
            },
            {
                "task": "auto_correction",
                "description": "RCA stack plus reviewer LLM for corrected root cause and final explanation.",
            },
        ]

    def run(
        self,
        task: TaskName,
        cell: str,
        alarm: str,
        kpi: str,
    ) -> dict[str, Any]:
        if task == "rca":
            return dict(self._rca_pipeline.run(cell=cell, alarm=alarm, kpi=kpi))
        if task == "rca_with_healing":
            return dict(self._healing_pipeline.run_with_healing(cell=cell, alarm=alarm, kpi=kpi))
        if task == "auto_correction":
            return dict(self._auto_correction_agent.run(cell=cell, alarm=alarm, kpi=kpi))
        raise ValueError(f"Unknown task: {task!r}")
