from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentic_oran_rca.graph.neo4j_client import Neo4jClient
from agentic_oran_rca.rag.retriever import IncidentRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedContext:
    cell: str
    du: str
    cu: str
    neighbor_cells: list[str]
    related_faults: list[dict[str, Any]]
    similar_incidents: list[dict[str, Any]]


class ContextRetrievalAgent:
    def __init__(self, neo4j: Neo4jClient, retriever: IncidentRetriever) -> None:
        self._neo4j = neo4j
        self._retriever = retriever

    def run(self, cell_id: str, alarm: str, k: int) -> RetrievedContext:
        topo = self._neo4j.run(
            """
            MATCH (c:Cell {id: $cell})-[:SERVED_BY]->(d:DU)-[:CONNECTED_TO]->(u:CU)
            OPTIONAL MATCH (c)-[:NEIGHBOR]->(n:Cell)
            RETURN c.id AS cell, d.id AS du, u.id AS cu, collect(DISTINCT n.id) AS neighbors
            """.strip(),
            {"cell": cell_id},
        )
        if not topo:
            raise RuntimeError(f"Cell not found in Neo4j: {cell_id}")

        cell = str(topo[0]["cell"])
        du = str(topo[0]["du"])
        cu = str(topo[0]["cu"])
        neighbors = sorted([x for x in topo[0]["neighbors"] if x])

        faults = self._neo4j.run(
            """
            MATCH (a:Alarm {name: $alarm})-[:INDICATES]->(f:Fault)
            OPTIONAL MATCH (f)-[r:AFFECTS]->(c:Cell)
            RETURN f.name AS fault, coalesce(sum(r.case_count), 0) AS historical_cases
            ORDER BY historical_cases DESC
            """.strip(),
            {"alarm": alarm},
        )

        query = f"cell={cell} du={du} cu={cu} alarm={alarm}"
        similar = self._retriever.retrieve(query=query, k=k)

        return RetrievedContext(
            cell=cell,
            du=du,
            cu=cu,
            neighbor_cells=neighbors,
            related_faults=[{"fault": f["fault"], "historical_cases": int(f["historical_cases"])} for f in faults],
            similar_incidents=[
                {"text": s.text, "metadata": s.metadata, "score": float(s.score)} for s in similar
            ],
        )

