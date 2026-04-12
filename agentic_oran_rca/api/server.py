from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_oran_rca.config import load_settings
from agentic_oran_rca.logging_utils import setup_logging
from agentic_oran_rca.main import RANKING_EVALUATION_DIR, build_auto_correction_agent, build_healing_pipeline, build_pipeline

logger = logging.getLogger(__name__)

_SAFE_REPORT_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve_ranking_report_file(filename: str) -> Path:
    if not _SAFE_REPORT_FILENAME.fullmatch(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    base = RANKING_EVALUATION_DIR.resolve()
    candidate = (RANKING_EVALUATION_DIR / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return candidate


class RCARequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell: str | None = Field(None, min_length=1)
    alarm: str | None = Field(None, min_length=1)
    kpi: str | None = Field(None, min_length=1)
    cell_id: str | None = Field(None, min_length=1)
    alarm_type: str | None = Field(None, min_length=1)
    kpi_metric: str | None = Field(None, min_length=1)

    def get_cell(self) -> str:
        return (self.cell or self.cell_id) or ""

    def get_alarm(self) -> str:
        return (self.alarm or self.alarm_type) or ""

    def get_kpi(self) -> str:
        return (self.kpi or self.kpi_metric) or ""

    @model_validator(mode="after")
    def require_fields(self) -> "RCARequest":
        if not (self.cell or self.cell_id):
            raise ValueError("cell or cell_id required")
        if not (self.alarm or self.alarm_type):
            raise ValueError("alarm or alarm_type required")
        if not (self.kpi or self.kpi_metric):
            raise ValueError("kpi or kpi_metric required")
        return self


async def _parse_body(request: Request) -> dict:
    body = await request.body()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if isinstance(body, str):
        body = body.strip()
        if not body:
            raise ValueError("Empty request body")
        data = json.loads(body)
        while isinstance(data, str) and data.strip().startswith("{"):
            data = json.loads(data)
        return data if isinstance(data, dict) else {}
    return body if isinstance(body, dict) else {}


class RCAResponse(BaseModel):
    predicted_root_cause: str
    confidence: float
    explanation: str


class HealingReportSchema(BaseModel):
    remediation_action: str
    remediation_description: str
    success: bool
    message: str
    report_path: str
    notification_sent: bool


class RCAResponseWithHealing(BaseModel):
    predicted_root_cause: str
    confidence: float
    explanation: str
    healing: HealingReportSchema


class AutoCorrectionResponse(BaseModel):
    context_retrieval: dict[str, Any]
    initial_predicted_root_cause: str
    initial_confidence: float
    initial_explanation: str
    corrected_root_cause: str
    correction_applied: bool
    correction_rationale: str
    final_explanation: str


def create_app() -> FastAPI:
    settings = load_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title="Agentic O-RAN RCA rApp Prototype", version="1.0.0")
    pipeline = build_pipeline(settings)
    healing_pipeline = build_healing_pipeline(settings)
    auto_correction_agent = build_auto_correction_agent(settings)

    @app.post("/run_rca", response_model=RCAResponse)
    async def run_rca(request: Request) -> RCAResponse | JSONResponse:
        try:
            data = await _parse_body(request)
            req = RCARequest(**data)
            out = pipeline.run(cell=req.get_cell(), alarm=req.get_alarm(), kpi=req.get_kpi())
            return RCAResponse(
                predicted_root_cause=out["predicted_root_cause"],
                confidence=float(out["confidence"]),
                explanation=out["explanation"],
            )
        except Exception as e:
            logger.exception("run_rca failed")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e), "traceback": traceback.format_exc()},
            )

    @app.post("/run_rca_with_healing", response_model=RCAResponseWithHealing)
    async def run_rca_with_healing(request: Request) -> RCAResponseWithHealing | JSONResponse:
        try:
            data = await _parse_body(request)
            req = RCARequest(**data)
            out = healing_pipeline.run_with_healing(cell=req.get_cell(), alarm=req.get_alarm(), kpi=req.get_kpi())
            return RCAResponseWithHealing(
                predicted_root_cause=out["predicted_root_cause"],
                confidence=float(out["confidence"]),
                explanation=out["explanation"],
                healing=HealingReportSchema(**out["healing"]),
            )
        except Exception as e:
            logger.exception("run_rca_with_healing failed")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e), "traceback": traceback.format_exc()},
            )

    @app.post("/run_auto_correction", response_model=AutoCorrectionResponse)
    async def run_auto_correction(request: Request) -> AutoCorrectionResponse | JSONResponse:
        try:
            data = await _parse_body(request)
            req = RCARequest(**data)
            out = auto_correction_agent.run(
                cell=req.get_cell(),
                alarm=req.get_alarm(),
                kpi=req.get_kpi(),
            )
            return AutoCorrectionResponse(**out)
        except Exception as e:
            logger.exception("run_auto_correction failed")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e), "traceback": traceback.format_exc()},
            )

    @app.get("/evaluation/ranking/reports")
    async def list_ranking_reports() -> dict[str, object]:
        d = RANKING_EVALUATION_DIR
        if not d.is_dir():
            return {
                "directory": str(d.resolve()),
                "reports": [],
            }
        items: list[dict[str, object]] = []
        for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime_ns, reverse=True):
            if not p.is_file():
                continue
            st = p.stat()
            items.append(
                {
                    "name": p.name,
                    "size_bytes": st.st_size,
                    "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        return {"directory": str(d.resolve()), "reports": items}

    @app.get("/evaluation/ranking/reports/{filename}", response_model=None)
    async def get_ranking_report(filename: str):
        path = _resolve_ranking_report_file(filename)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")

    return app


app = create_app()

