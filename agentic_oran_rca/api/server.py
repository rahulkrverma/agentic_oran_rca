from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentic_oran_rca.config import load_settings
from agentic_oran_rca.logging_utils import setup_logging
from agentic_oran_rca.main import build_pipeline

logger = logging.getLogger(__name__)


class RCARequest(BaseModel):
    cell: str = Field(..., min_length=1)
    alarm: str = Field(..., min_length=1)
    kpi: str = Field(..., min_length=1)


class RCAResponse(BaseModel):
    predicted_root_cause: str
    confidence: float
    explanation: str


def create_app() -> FastAPI:
    settings = load_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title="Agentic O-RAN RCA rApp Prototype", version="1.0.0")
    pipeline = build_pipeline(settings)

    @app.post("/run_rca", response_model=RCAResponse)
    def run_rca(req: RCARequest) -> RCAResponse:
        out = pipeline.run(cell=req.cell, alarm=req.alarm, kpi=req.kpi)
        return RCAResponse(
            predicted_root_cause=out["predicted_root_cause"],
            confidence=float(out["confidence"]),
            explanation=out["explanation"],
        )

    return app


app = create_app()

