"""
InvestigationAgent — top-level facade.

Wires together: workspace, tool registry, LLM provider, agent loop.
This is the public entrypoint for both the CLI and the FastAPI server.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import config
from agent.core.loop import AgentLoop
from agent.core.scope import validate_issue_scope
from agent.llm.base import LLMProvider
from agent.llm.gemini import GeminiProvider
from agent.models.investigation import InvestigationState, InvestigationStatus
from agent.repository.workspace import Workspace
from agent.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class InvestigationAgent:
    """
    High-level agent interface.

    Usage
    -----
    ::

        agent = InvestigationAgent(repo_path="…")
        result = agent.investigate(issue="Active customers cannot create orders")
        print(result["rca"])
    """

    def __init__(
        self,
        repo_path: str,
        llm_provider: LLMProvider | None = None,
        max_iterations: int | None = None,
        request_budget: int | None = None,
        progress_callback: Any | None = None,
        runs_dir: str | Path | None = None,
    ) -> None:
        self._workspace = Workspace(repo_path)
        self._registry = ToolRegistry(self._workspace)
        self._llm = llm_provider or GeminiProvider()
        self._max_iter = max_iterations if max_iterations is not None else config.MAX_ITERATIONS
        self._request_budget = request_budget if request_budget is not None else config.LLM_REQUEST_BUDGET
        if hasattr(self._llm, "request_budget"):
            self._llm.request_budget = self._request_budget
        self._progress_callback = progress_callback
        self._runs_dir = Path(runs_dir) if runs_dir else config.RUNS_DIR

    @property
    def llm_config(self) -> dict[str, Any]:
        """Effective LLM configuration (model, thinking level, SDK version)."""
        return self._llm.describe()

    # ── public ───────────────────────────────────────────────────────────────

    def investigate(self, issue: str) -> dict[str, Any]:
        """
        Run a full investigation and return:

        ::

            {
                "run_id":     "…",
                "status":     "COMPLETED" | "INCOMPLETE" | "INVESTIGATION_LIMIT_REACHED" | "OUT_OF_SCOPE" | "GENERAL_QUESTION" | "ERROR",
                "intent":     "CODE_INVESTIGATION" | "GENERAL_QUESTION" | "OUT_OF_SCOPE",
                "intent_reason": "…",
                "direct_answer": "…" | None,
                "request_count": int,
                "request_budget": int,
                "termination_reason": "…",
                "rca":        { … canonical RCAReport.to_dict() … } | None,
                "llm_config": { … },
                "run_file":   "path/to/saved.json" | None,
                "trace":      [ … ],
            }
        """
        if not issue or not issue.strip():
            raise ValueError("Issue description must not be empty.")

        if hasattr(self._llm, "request_budget"):
            self._llm.request_budget = self._request_budget

        # Intent routing layer
        from agent.core.router import route_intent
        from agent.llm.base import LLMQuotaExhaustedError, LLMUnavailableError, LLMError

        try:
            intent_res, used_llm = route_intent(issue, llm=self._llm)
        except LLMQuotaExhaustedError as exc:
            state = InvestigationState(
                issue=issue.strip(),
                repository_path=str(self._workspace.root),
                status=InvestigationStatus.ERROR,
                termination_reason=f"LLM quota exhausted: {exc}",
                request_count=getattr(self._llm, "request_count", 1),
                request_budget=self._request_budget,
            )
            state.llm_config = self.llm_config
            state.add_trace("ERROR", f"LLM quota exhausted: {exc}")
            run_file = self._save_run(state)
            return {
                "run_id": state.run_id,
                "status": state.status.value,
                "intent": "ERROR",
                "intent_reason": str(exc),
                "direct_answer": None,
                "request_count": state.request_count,
                "request_budget": state.request_budget,
                "termination_reason": state.termination_reason,
                "rca": None,
                "llm_config": state.llm_config,
                "run_file": str(run_file) if run_file else None,
                "trace": [
                    {
                        "step": t.step,
                        "kind": t.kind,
                        "description": t.description,
                        "timestamp": t.timestamp,
                    }
                    for t in state.trace
                ],
            }

        request_count = getattr(self._llm, "request_count", 1 if used_llm else 0)

        # Handle GENERAL_QUESTION
        if intent_res.intent == "GENERAL_QUESTION":
            state = InvestigationState(
                issue=issue.strip(),
                repository_path=str(self._workspace.root),
                status=InvestigationStatus.GENERAL_QUESTION,
                intent=intent_res.intent,
                intent_reason=intent_res.reason,
                general_answer=intent_res.direct_answer,
                request_count=request_count,
                request_budget=self._request_budget,
                termination_reason="Answered general question directly.",
            )
            state.llm_config = self.llm_config
            flabel = f"[>>] Intent: GENERAL_QUESTION ({intent_res.reason})"
            if self._progress_callback:
                self._progress_callback(flabel, intent_res.direct_answer)
            state.add_trace("INTENT_ROUTER", flabel, intent_res.direct_answer)
            run_file = self._save_run(state)
            return {
                "run_id": state.run_id,
                "status": state.status.value,
                "intent": state.intent,
                "intent_reason": state.intent_reason,
                "direct_answer": state.general_answer,
                "request_count": state.request_count,
                "request_budget": state.request_budget,
                "termination_reason": state.termination_reason,
                "rca": None,
                "llm_config": state.llm_config,
                "run_file": str(run_file) if run_file else None,
                "trace": [
                    {
                        "step": t.step,
                        "kind": t.kind,
                        "description": t.description,
                        "timestamp": t.timestamp,
                    }
                    for t in state.trace
                ],
            }

        # Handle OUT_OF_SCOPE
        if intent_res.intent == "OUT_OF_SCOPE":
            state = InvestigationState(
                issue=issue.strip(),
                repository_path=str(self._workspace.root),
                status=InvestigationStatus.OUT_OF_SCOPE,
                intent=intent_res.intent,
                intent_reason=intent_res.reason,
                request_count=request_count,
                request_budget=self._request_budget,
                termination_reason=f"Input out of scope: {intent_res.reason}",
            )
            state.llm_config = self.llm_config
            flabel = f"[!!] Issue OUT_OF_SCOPE: {intent_res.reason}"
            if self._progress_callback:
                self._progress_callback(flabel, intent_res.reason)
            state.add_trace("INTENT_ROUTER", flabel, intent_res.reason)
            run_file = self._save_run(state)
            return {
                "run_id": state.run_id,
                "status": state.status.value,
                "intent": state.intent,
                "intent_reason": state.intent_reason,
                "scope_reason": intent_res.reason,
                "direct_answer": None,
                "request_count": state.request_count,
                "request_budget": state.request_budget,
                "termination_reason": state.termination_reason,
                "rca": None,
                "llm_config": state.llm_config,
                "run_file": str(run_file) if run_file else None,
                "trace": [
                    {
                        "step": t.step,
                        "kind": t.kind,
                        "description": t.description,
                        "timestamp": t.timestamp,
                    }
                    for t in state.trace
                ],
            }

        # Handle CODE_INVESTIGATION: pass control to existing investigation loop
        state = InvestigationState(
            issue=issue.strip(),
            repository_path=str(self._workspace.root),
            intent="CODE_INVESTIGATION",
            intent_reason=intent_res.reason,
            request_count=request_count,
            request_budget=self._request_budget,
        )
        state.llm_config = self.llm_config
        flabel = f"[>>] Intent: CODE_INVESTIGATION ({intent_res.reason})"
        if self._progress_callback:
            self._progress_callback(flabel, intent_res.reason)
        state.add_trace("INTENT_ROUTER", flabel, intent_res.reason)

        loop = AgentLoop(
            llm=self._llm,
            registry=self._registry,
            max_iterations=self._max_iter,
            request_budget=self._request_budget,
            progress_callback=self._progress_callback,
        )

        state = loop.run(state)

        run_file = self._save_run(state)

        return {
            "run_id": state.run_id,
            "status": state.status.value,
            "outcome_type": state.outcome_type or (state.final_rca.get("outcome_type") if state.final_rca else None),
            "intent": state.intent,
            "intent_reason": state.intent_reason,
            "rca": state.final_rca,
            "request_count": state.request_count,
            "request_budget": state.request_budget,
            "termination_reason": state.termination_reason,
            "llm_config": state.llm_config,
            "run_file": str(run_file) if run_file else None,
            "trace": [
                {
                    "step": t.step,
                    "kind": t.kind,
                    "description": t.description,
                    "timestamp": t.timestamp,
                }
                for t in state.trace
            ],
        }

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """JSON-safe copy of the conversation (provider objects are dropped)."""
        out = []
        for m in messages:
            out.append({k: v for k, v in m.items() if k not in ("provider_content", "raw_candidate")})
        return out

    def _save_run(self, state: InvestigationState) -> Path | None:
        """Persist the full investigation trace as JSON."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        filename = self._runs_dir / f"{ts}_{state.run_id[:8]}.json"

        payload = {
            "run_id": state.run_id,
            "started_at": state.started_at,
            "status": state.status.value,
            "outcome_type": state.outcome_type or (state.final_rca.get("outcome_type") if state.final_rca else None),
            "intent": state.intent,
            "intent_reason": state.intent_reason,
            "general_answer": state.general_answer,
            "request_count": state.request_count,
            "request_budget": state.request_budget,
            "termination_reason": state.termination_reason,
            "issue": state.issue,
            "repository_path": state.repository_path,
            "llm_config": state.llm_config,
            "max_iterations": self._max_iter,
            "iterations": state.iteration,
            "files_inspected": state.files_inspected,
            "symbols_inspected": state.symbols_inspected,
            "candidate_files": state.candidate_files,
            "finish_accepted_count": state.finish_accepted_count,
            "finish_attempts": state.finish_attempts,
            "trace": [
                {
                    "step": t.step,
                    "kind": t.kind,
                    "description": t.description,
                    "detail": t.detail,
                    "timestamp": t.timestamp,
                }
                for t in state.trace
            ],
            "tool_calls": [
                {
                    "iteration": tc.iteration,
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                    "tool_result": tc.tool_result,
                    "timestamp": tc.timestamp,
                }
                for tc in state.tool_calls
            ],
            "messages": self._serialise_messages(state.messages),
            # The one canonical RCAReport — identical to what the CLI renders.
            "final_rca": state.final_rca,
        }

        try:
            filename.write_text(
                json.dumps(payload, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Run saved to %s", filename)
            return filename
        except OSError as exc:
            logger.warning("Could not save run file: %s", exc)
            return None
