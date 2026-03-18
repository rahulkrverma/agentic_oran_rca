from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx

from agentic_oran_rca.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


ALARM_TYPES = [
    "Cell Down",
    "High Latency",
    "Packet Loss",
    "Signal Degradation",
    "Power Failure",
    "Transport Link Failure",
]

ROOT_CAUSES = [
    "DU failure",
    "backhaul failure",
    "congestion",
    "antenna fault",
    "power supply failure",
    "misconfiguration",
]


ALARM_TO_FAULT = {
    "Cell Down": ["DU failure", "power supply failure", "backhaul failure", "misconfiguration"],
    "High Latency": ["congestion", "backhaul failure", "misconfiguration"],
    "Packet Loss": ["backhaul failure", "congestion", "misconfiguration"],
    "Signal Degradation": ["antenna fault", "misconfiguration", "power supply failure"],
    "Power Failure": ["power supply failure", "DU failure"],
    "Transport Link Failure": ["backhaul failure"],
}


@dataclass(frozen=True)
class TopologySpec:
    num_cus: int
    num_dus: int
    num_cells: int
    neighbor_k: int
    seed: int


def generate_oran_topology(spec: TopologySpec) -> dict:
    rnd = random.Random(spec.seed)

    cus = [f"CU{i+1}" for i in range(spec.num_cus)]
    dus = [f"DU{i+1}" for i in range(spec.num_dus)]
    cells = [f"Cell{i+1}" for i in range(spec.num_cells)]

    du_to_cu = {du: rnd.choice(cus) for du in dus}
    cell_to_du = {cell: rnd.choice(dus) for cell in cells}

    g = nx.Graph()
    g.add_nodes_from(cells)

    for cell in cells:
        candidates = [c for c in cells if c != cell]
        rnd.shuffle(candidates)
        for nbr in candidates[: spec.neighbor_k]:
            g.add_edge(cell, nbr)

    return {
        "cus": cus,
        "dus": dus,
        "cells": cells,
        "du_to_cu": du_to_cu,
        "cell_to_du": cell_to_du,
        "neighbors": {n: sorted(list(g.neighbors(n))) for n in cells},
    }


def reset_and_load_neo4j(client: Neo4jClient, topo: dict) -> None:
    logger.info("Resetting Neo4j graph")
    client.run_write("MATCH (n) DETACH DELETE n")

    logger.info("Creating constraints")
    client.run_write("CREATE CONSTRAINT cell_id IF NOT EXISTS FOR (c:Cell) REQUIRE c.id IS UNIQUE")
    client.run_write("CREATE CONSTRAINT du_id IF NOT EXISTS FOR (d:DU) REQUIRE d.id IS UNIQUE")
    client.run_write("CREATE CONSTRAINT cu_id IF NOT EXISTS FOR (c:CU) REQUIRE c.id IS UNIQUE")
    client.run_write("CREATE CONSTRAINT alarm_id IF NOT EXISTS FOR (a:Alarm) REQUIRE a.name IS UNIQUE")
    client.run_write("CREATE CONSTRAINT fault_id IF NOT EXISTS FOR (f:Fault) REQUIRE f.name IS UNIQUE")
    client.run_write("CREATE CONSTRAINT kpi_id IF NOT EXISTS FOR (k:KPI) REQUIRE k.name IS UNIQUE")

    logger.info("Loading topology nodes")
    for cu in topo["cus"]:
        client.run_write("MERGE (:CU {id: $id})", {"id": cu})

    for du in topo["dus"]:
        client.run_write("MERGE (:DU {id: $id})", {"id": du})
        client.run_write(
            """
            MATCH (d:DU {id: $du}), (c:CU {id: $cu})
            MERGE (d)-[:CONNECTED_TO]->(c)
            """.strip(),
            {"du": du, "cu": topo["du_to_cu"][du]},
        )

    for cell in topo["cells"]:
        client.run_write("MERGE (:Cell {id: $id})", {"id": cell})
        client.run_write(
            """
            MATCH (c:Cell {id: $cell}), (d:DU {id: $du})
            MERGE (c)-[:SERVED_BY]->(d)
            """.strip(),
            {"cell": cell, "du": topo["cell_to_du"][cell]},
        )

    logger.info("Loading neighbor relations")
    for cell, nbrs in topo["neighbors"].items():
        for nbr in nbrs:
            client.run_write(
                """
                MATCH (a:Cell {id: $a}), (b:Cell {id: $b})
                MERGE (a)-[:NEIGHBOR]->(b)
                """.strip(),
                {"a": cell, "b": nbr},
            )

    logger.info("Loading alarm/fault ontology")
    for alarm in ALARM_TYPES:
        client.run_write("MERGE (:Alarm {name: $name})", {"name": alarm})

    for fault in ROOT_CAUSES:
        client.run_write("MERGE (:Fault {name: $name})", {"name": fault})

    for alarm, faults in ALARM_TO_FAULT.items():
        for fault in faults:
            client.run_write(
                """
                MATCH (a:Alarm {name: $alarm}), (f:Fault {name: $fault})
                MERGE (a)-[:INDICATES]->(f)
                """.strip(),
                {"alarm": alarm, "fault": fault},
            )


def load_fault_affects_from_dataset(client: Neo4jClient, rows: list[dict]) -> None:
    """
    Adds Fault -> AFFECTS -> Cell edges with aggregated case counts.
    Also materializes KPI nodes for retrieval context.
    """
    agg: dict[tuple[str, str], int] = {}
    kpis: set[str] = set()

    for r in rows:
        fault = str(r["expected_root_cause"])
        cell = str(r["cell"])
        agg[(fault, cell)] = agg.get((fault, cell), 0) + 1
        kpis.add(str(r["kpi"]))

    for k in sorted(kpis):
        client.run_write("MERGE (:KPI {name: $name})", {"name": k})

    for (fault, cell), n in agg.items():
        client.run_write(
            """
            MATCH (f:Fault {name: $fault}), (c:Cell {id: $cell})
            MERGE (f)-[r:AFFECTS]->(c)
            SET r.case_count = $n
            """.strip(),
            {"fault": fault, "cell": cell, "n": int(n)},
        )

