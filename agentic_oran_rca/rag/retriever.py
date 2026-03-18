from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarIncident:
    text: str
    metadata: dict[str, Any]
    score: float


class IncidentRetriever:
    def __init__(self, vs: Chroma) -> None:
        self._vs = vs

    def retrieve(self, query: str, k: int) -> list[SimilarIncident]:
        docs_with_scores = self._vs.similarity_search_with_score(query, k=k)
        out: list[SimilarIncident] = []
        for doc, score in docs_with_scores:
            out.append(SimilarIncident(text=doc.page_content, metadata=dict(doc.metadata), score=float(score)))
        return out

