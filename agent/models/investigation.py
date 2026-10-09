"""
Investigation data models.
All plain dataclasses — no framework dependency.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class InvestigationStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    INVESTIGATION_LIMIT_REACHED = "INVESTIGATION_LIMIT_REACHED"
    INCOMPLETE = "INCOMPLETE"  # agent concluded evidence is insufficient
    ERROR = "ERROR"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"  # conversational, non-technical, or out of scope
    GENERAL_QUESTION = "GENERAL_QUESTION"  # general technical/conceptual question answered directly


@dataclass
class ToolCall:
    """A single tool invocation recorded in the investigation."""
    iteration: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any  # raw result from the tool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TraceEntry:
    """One line in the readable investigation trace."""
    step: int
    kind: str          # e.g. "USER_ISSUE", "ROUTER", "LLM_ACTION", "TOOL_RESULT", "RCA"
    description: str
    detail: Any = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class InvestigationState:
    """Full mutable state of one investigation run."""

    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Inputs
    issue: str = ""
    repository_path: str = ""

    # Routing
    intent: str = ""
    intent_reason: str = ""
    general_answer: str | None = None

    # Request Budgeting
    request_count: int = 0
    request_budget: int = 15
    termination_reason: str = ""

    # Progress
    iteration: int = 0
    status: InvestigationStatus = InvestigationStatus.IN_PROGRESS
    outcome_type: str = ""

    # LLM conversation history (provider-agnostic list of message dicts)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Tool history
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Discovery
    files_inspected: list[str] = field(default_factory=list)
    symbols_inspected: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)

    # Reasoning
    hypotheses: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    # Readable trace
    trace: list[TraceEntry] = field(default_factory=list)

    # Completion bookkeeping
    # Every finish_investigation evaluation (accepted or rejected) is recorded.
    finish_attempts: list[dict[str, Any]] = field(default_factory=list)
    finish_accepted_count: int = 0

    # Effective LLM settings (model, thinking level, ...)
    llm_config: dict[str, Any] = field(default_factory=dict)

    # Final output: the canonical RCAReport serialised with RCAReport.to_dict()
    final_rca: dict[str, Any] | None = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def add_trace(self, kind: str, description: str, detail: Any = None) -> None:
        step = len(self.trace) + 1
        self.trace.append(TraceEntry(step=step, kind=kind, description=description, detail=detail))

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: Any,
    ) -> None:
        tc = ToolCall(
            iteration=self.iteration,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
        )
        self.tool_calls.append(tc)

        # Track files inspected
        for key in ("file_path", "path"):
            if key in tool_input and isinstance(tool_input[key], str):
                p = tool_input[key]
                if p not in self.files_inspected:
                    self.files_inspected.append(p)

        # Track symbols
        if tool_name == "search_symbol" and "symbol" in tool_input:
            sym = tool_input["symbol"]
            if sym not in self.symbols_inspected:
                self.symbols_inspected.append(sym)
