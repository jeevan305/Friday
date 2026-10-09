"""
Integration test for the agent loop using a mocked LLM.

Verifies:
  1. LLM chooses a tool → tool executes → result goes back to LLM
  2. Context (messages) accumulates across iterations
  3. LLM eventually calls finish_investigation
  4. Final RCA contains structured evidence

No real API key required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.core.loop import AgentLoop
from agent.llm.base import LLMProvider, LLMResponse
from agent.models.investigation import InvestigationState, InvestigationStatus
from agent.repository.workspace import Workspace
from agent.tools.tool_registry import ToolRegistry


# ── Minimal test repository ───────────────────────────────────────────────────

@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    """Minimal Java repo with a deliberate bug."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "OrderService.java").write_text(
        "public class OrderService {\n"
        "    public Order createOrder(Long customerId) {\n"
        "        customerService.validateCustomerEligibility(customerId);\n"
        "        return new Order();\n"
        "    }\n"
        "}\n"
    )
    (src / "CustomerService.java").write_text(
        "public class CustomerService {\n"
        "    public void validateCustomerEligibility(Long id) {\n"
        "        if (customer.getStatus() == CustomerStatus.ACTIVE) {\n"
        "            throw new CustomerNotEligibleException();\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    return tmp_path


# ── Scripted mock LLM ─────────────────────────────────────────────────────────

class ScriptedLLM(LLMProvider):
    """
    A mock LLM that returns pre-scripted responses in sequence.

    Each response is an LLMResponse specifying a tool call.
    The final response calls finish_investigation.
    """

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._call_index = 0
        self.received_messages: list[list[dict[str, Any]]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.received_messages.append(list(messages))
        if self._call_index < len(self._script):
            response = self._script[self._call_index]
        else:
            # Safety: if script runs out, call finish_investigation
            response = _finish_response()
        self._call_index += 1
        return response


def _finish_response() -> LLMResponse:
    return LLMResponse(
        text="Investigation complete.",
        tool_calls=[
            {
                "name": "finish_investigation",
                "arguments": {
                    "status": "COMPLETED",
                    "root_cause": "The eligibility check in CustomerService.validateCustomerEligibility "
                                  "is inverted — it rejects ACTIVE customers instead of accepting them.",
                    "evidence": [
                        {
                            "file": "src/CustomerService.java",
                            "symbol": "validateCustomerEligibility",
                            "start_line": 2,
                            "end_line": 6,
                            "explanation": "The condition `== CustomerStatus.ACTIVE` throws for ACTIVE customers.",
                        }
                    ],
                    "recommended_fix": "In CustomerService.validateCustomerEligibility change `==` to `!=` "
                                       "in the CustomerStatus.ACTIVE check.",
                    "developer_next_steps": [
                        "Change the condition from `== ACTIVE` to `!= ACTIVE`.",
                        "Add a unit test for CustomerService.validateCustomerEligibility.",
                    ],
                    "confidence": 0.88,
                    "verification_performed": [
                        "Read OrderService.createOrder and CustomerService.validateCustomerEligibility",
                    ],
                    "remaining_uncertainty": ["Intended business rule not confirmed against a spec."],
                    "investigation_summary": [
                        "Listed repository root",
                        "Searched for 'createOrder' → found OrderService.java",
                        "Read OrderService.createOrder()",
                        "Found call to validateCustomerEligibility",
                        "Read CustomerService — found inverted condition",
                    ],
                },
            }
        ],
    )


def _scripted_investigation() -> list[LLMResponse]:
    """A scripted 4-step investigation before finish."""
    return [
        # Step 1: list_directory
        LLMResponse(
            text="Let me start by exploring the repository.",
            tool_calls=[{"name": "list_directory", "arguments": {"path": "."}}],
        ),
        # Step 2: search for order creation
        LLMResponse(
            text="Searching for order creation code.",
            tool_calls=[{"name": "search_code", "arguments": {"query": "createOrder"}}],
        ),
        # Step 3: read OrderService
        LLMResponse(
            text="Reading OrderService.",
            tool_calls=[{
                "name": "read_file",
                "arguments": {"file_path": "src/OrderService.java", "start_line": "1", "end_line": "10"},
            }],
        ),
        # Step 4: read CustomerService
        LLMResponse(
            text="Reading CustomerService.",
            tool_calls=[{
                "name": "read_file",
                "arguments": {"file_path": "src/CustomerService.java", "start_line": "1", "end_line": "10"},
            }],
        ),
        # Step 5: finish
        _finish_response(),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_agent_loop_runs_multiple_iterations(demo_repo: Path) -> None:
    """Verify the loop calls the LLM multiple times before finishing."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    result_state = loop.run(state)

    # LLM should have been called multiple times
    assert mock_llm._call_index >= 4, (
        f"Expected ≥4 LLM calls, got {mock_llm._call_index}"
    )


def test_context_accumulates_across_iterations(demo_repo: Path) -> None:
    """Verify each LLM call receives more messages than the previous one."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    loop.run(state)

    # Each call should receive at least as many messages as the previous one
    call_lengths = [len(msgs) for msgs in mock_llm.received_messages]
    assert len(call_lengths) >= 2

    for i in range(1, len(call_lengths)):
        assert call_lengths[i] >= call_lengths[i - 1], (
            f"Call {i} received fewer messages than call {i-1}: "
            f"{call_lengths[i]} < {call_lengths[i-1]}"
        )


def test_finish_investigation_terminates_loop(demo_repo: Path) -> None:
    """Verify the loop stops when finish_investigation is called."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    result_state = loop.run(state)

    assert result_state.status == InvestigationStatus.COMPLETED


def test_final_rca_is_structured(demo_repo: Path) -> None:
    """Verify the RCA output contains required fields."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    result_state = loop.run(state)

    rca = result_state.final_rca
    assert rca is not None
    assert rca["status"] == "COMPLETED"
    assert rca["root_cause"]
    assert isinstance(rca["evidence"], list)
    assert len(rca["evidence"]) >= 1
    assert rca["confidence"] > 0


def test_rca_evidence_contains_file_reference(demo_repo: Path) -> None:
    """Evidence items must reference actual file paths."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    result_state = loop.run(state)

    rca = result_state.final_rca
    ev = rca["evidence"][0]
    assert "file" in ev
    assert "CustomerService" in ev["file"]
    assert ev["start_line"] > 0


def test_tool_results_returned_to_llm(demo_repo: Path) -> None:
    """Tool results must appear in subsequent LLM message context."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)
    mock_llm = ScriptedLLM(_scripted_investigation())

    state = InvestigationState(
        issue="Active customers cannot create orders",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=20)
    loop.run(state)

    # After first tool call, subsequent calls should include tool results in messages
    if len(mock_llm.received_messages) >= 2:
        second_call_messages = mock_llm.received_messages[1]
        # There should be a tool result message
        roles = [m.get("role") for m in second_call_messages]
        assert "tool" in roles, (
            "Tool results were not passed back to LLM in subsequent messages"
        )


def test_max_iterations_respected(demo_repo: Path) -> None:
    """Verify INVESTIGATION_LIMIT_REACHED when max iterations exceeded."""
    ws = Workspace(demo_repo)
    registry = ToolRegistry(ws)

    # Script that never calls finish_investigation
    infinite_script = [
        LLMResponse(
            text="Searching...",
            tool_calls=[{"name": "list_directory", "arguments": {"path": "."}}],
        )
    ] * 100  # way more than our limit

    mock_llm = ScriptedLLM(infinite_script)

    state = InvestigationState(
        issue="Test issue",
        repository_path=str(demo_repo),
    )

    loop = AgentLoop(llm=mock_llm, registry=registry, max_iterations=3)
    result_state = loop.run(state)

    assert result_state.status == InvestigationStatus.INVESTIGATION_LIMIT_REACHED


# ── Optional: live Gemini test ────────────────────────────────────────────────

def test_live_gemini_investigation(tmp_path: Path) -> None:
    """
    Real end-to-end test against the Gemini API.

    Only runs when RUN_LIVE_LLM_TEST=true is set in the environment.
    Requires GEMINI_API_KEY to be configured.
    """
    import os
    if os.getenv("RUN_LIVE_LLM_TEST", "").lower() != "true":
        pytest.skip("Set RUN_LIVE_LLM_TEST=true to run live LLM tests")

    # Use the demo repo from examples/
    demo_path = Path(__file__).parent.parent / "examples" / "demo-order-service"
    if not demo_path.exists():
        pytest.skip("demo-order-service not found")

    from agent import config
    config.validate()

    from agent.core.agent import InvestigationAgent

    agent = InvestigationAgent(
        repo_path=str(demo_path),
        max_iterations=15,
    )

    result = agent.investigate(
        issue="Active customers are unable to create orders."
    )

    assert result["status"] in ("COMPLETED", "INVESTIGATION_LIMIT_REACHED")
    assert result["rca"] is not None
    rca = result["rca"]
    assert rca.get("root_cause")

    # If completed, verify evidence points to CustomerService
    if result["status"] == "COMPLETED":
        evidence_files = [e.get("file", "") for e in rca.get("evidence", [])]
        assert any("CustomerService" in f for f in evidence_files), (
            f"Expected evidence pointing to CustomerService, got: {evidence_files}"
        )
