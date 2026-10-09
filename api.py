"""
Optional FastAPI wrapper for the investigation agent.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /investigate
    GET  /runs/{run_id}
    GET  /health
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from agent import config
from agent.core.agent import InvestigationAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Code Investigation Agent",
    description=(
        "Autonomous source-code investigation agent powered by Google Gemini. "
        "Provide a repository path and a production issue description; "
        "the agent will investigate the codebase and return a structured Root Cause Analysis."
    ),
    version="1.0.0",
)

# In-memory run store (replace with persistence layer later)
_runs: dict[str, dict[str, Any]] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    repository_path: str = Field(
        ...,
        description="Absolute path to the repository on the server filesystem.",
        example="C:/projects/order-management",
    )
    issue: str = Field(
        ...,
        description="Plain-language description of the production issue.",
        example="Active customers are unable to create orders.",
    )
    max_iterations: int | None = Field(
        default=None,
        description=f"Maximum investigation iterations (default: {config.MAX_ITERATIONS}).",
    )


class InvestigateResponse(BaseModel):
    run_id: str
    status: str
    rca: dict[str, Any] | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": config.GEMINI_MODEL}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest) -> InvestigateResponse:
    """
    Run a full investigation and return the RCA.

    This is a synchronous endpoint — it blocks until the investigation completes.
    For long investigations consider wrapping with a background task queue.
    """
    # Validate repository path
    repo_path = Path(request.repository_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Repository path does not exist: {repo_path}",
        )

    try:
        config.validate()
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        agent = InvestigationAgent(
            repo_path=str(repo_path),
            max_iterations=request.max_iterations,
        )
        result = agent.investigate(issue=request.issue)
    except Exception as exc:
        logger.exception("Investigation failed")
        raise HTTPException(status_code=500, detail=f"Investigation failed: {exc}")

    _runs[result["run_id"]] = result

    return InvestigateResponse(
        run_id=result["run_id"],
        status=result["status"],
        rca=result["rca"],
    )


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Retrieve a previous investigation run by ID."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _runs[run_id]
