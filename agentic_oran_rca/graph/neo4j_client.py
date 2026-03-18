from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


class Neo4jClient:
    def __init__(self, cfg: Neo4jConfig) -> None:
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self) -> None:
        self._driver.close()

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]

    def run_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}).consume())

    def run_many(self, statements: Iterable[tuple[str, dict[str, Any]]]) -> None:
        with self._driver.session() as session:
            def _work(tx):
                for cypher, params in statements:
                    tx.run(cypher, params).consume()

            session.execute_write(_work)

