"""
Deterministic regression tests for the investigation agent (mocked LLM, no network).

Covers: tool execution and result feedback, context retention, multiple tool calls,
hypothesis revision, the evidence-based completion gate, incomplete results,
single processing of finish_investigation, report rendering and JSON/CLI agreement,
honesty about test execution, reminders, and the Gemini message conversion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from agent.core.agent import InvestigationAgent
from agent.core.completion import (
    TEST_EXECUTION_NOTE,
    EvidenceLedger,
    coerce_list,
    validate_finish,
)
from agent.core.loop import AgentLoop
from agent.core.prompts import build_iteration_prompt
from agent.core.report import SECTION_TITLES, format_confidence, render_report
from agent.llm.base import LLMProvider, LLMResponse
from agent.models.investigation import InvestigationState, InvestigationStatus
from agent.models.rca import InvestigationOutcome, RCAReport, RCAStatus
from agent.repository.workspace import Workspace
from agent.tools.tool_registry import FINISH_TOOL_DEFINITION, ToolRegistry


# ── fixtures / helpers ───────────────────────────────────────────────────────

CUSTOMER_SERVICE = (
    "public class CustomerService {\n"                                          # 1
    "    public void validateCustomerEligibility(Long id) {\n"                  # 2
    "        if (customer.getStatus() == CustomerStatus.ACTIVE) {\n"            # 3
    "            throw new CustomerNotEligibleException();\n"                   # 4
    "        }\n"                                                               # 5
    "    }\n"                                                                   # 6
    "}\n"                                                                       # 7
)
ORDER_SERVICE = (
    "public class OrderService {\n"
    "    public Order createOrder(Long customerId) {\n"
    "        customerService.validateCustomerEligibility(customerId);\n"
    "        return new Order();\n"
    "    }\n"
    "}\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "CustomerService.java").write_text(CUSTOMER_SERVICE)
    (src / "OrderService.java").write_text(ORDER_SERVICE)
    return tmp_path


class FnLLM(LLMProvider):
    """Mock LLM whose reply is computed from the conversation it receives."""

    def __init__(self, fn: Callable[[list[dict[str, Any]], int], LLMResponse]) -> None:
        self._fn = fn
        self.calls = 0
        self.snapshots: list[list[dict[str, Any]]] = []

    @property
    def model_name(self) -> str:
        return "mock"

    def chat(self, messages, tools=None, system_prompt=None) -> LLMResponse:
        self.snapshots.append(list(messages))
        resp = self._fn(messages, self.calls)
        self.calls += 1
        self.request_count += 1
        return resp


def scripted(*responses: LLMResponse) -> FnLLM:
    seq = list(responses)
    return FnLLM(lambda msgs, i: seq[i] if i < len(seq) else seq[-1])


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "arguments": args}


def resp(*calls: dict[str, Any], text: str = "") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls))


def read(path: str, start: int = 1, end: int = 50) -> dict[str, Any]:
    return call("read_file", file_path=path, start_line=start, end_line=end)


def good_finish(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "status": "COMPLETED",
        "root_cause": "CustomerService.validateCustomerEligibility rejects ACTIVE customers: "
                      "the condition `== CustomerStatus.ACTIVE` is inverted.",
        "evidence": [{
            "file": "src/CustomerService.java", "symbol": "validateCustomerEligibility",
            "start_line": 2, "end_line": 6,
            "explanation": "`customer.getStatus() == CustomerStatus.ACTIVE` throws "
                           "CustomerNotEligibleException for ACTIVE customers.",
        }],
        "recommended_fix": "In CustomerService.validateCustomerEligibility use `!=` so only "
                           "non-CustomerStatus.ACTIVE customers are rejected.",
        "developer_next_steps": ["Add a regression test for validateCustomerEligibility."],
        "confidence": 0.85,
        "verification_performed": ["Read the full validateCustomerEligibility method."],
        "remaining_uncertainty": ["Business rule not confirmed against a specification."],
        "investigation_summary": ["Read CustomerService.java"],
    }
    args.update(overrides)
    return call("finish_investigation", **args)


def run_loop(repo: Path, llm: LLMProvider, max_iterations: int = 10) -> InvestigationState:
    registry = ToolRegistry(Workspace(repo))
    state = InvestigationState(issue="Active customers cannot create orders", repository_path=str(repo))
    return AgentLoop(llm=llm, registry=registry, max_iterations=max_iterations).run(state)


def make_ledger(repo: Path, *reads: tuple[str, int, int]) -> EvidenceLedger:
    ws = Workspace(repo)
    reg = ToolRegistry(ws)
    ledger = EvidenceLedger(ws)
    for path, s, e in reads:
        ledger.record("read_file", reg.execute("read_file", {"file_path": path, "start_line": s, "end_line": e}))
    return ledger


def tool_messages(state: InvestigationState) -> list[dict[str, Any]]:
    return [m for m in state.messages if m["role"] == "tool"]


def section_counts(text: str) -> dict[str, int]:
    return {t: len(re.findall(rf"^  {re.escape(t)}$", text, flags=re.MULTILINE)) for t in SECTION_TITLES}


# ── tool execution & context ─────────────────────────────────────────────────

def test_tool_call_executed_and_result_returned_to_llm(repo: Path) -> None:
    llm = scripted(resp(read("src/CustomerService.java")), resp(good_finish()))
    state = run_loop(repo, llm)

    second = llm.snapshots[1]
    tool_msg = [m for m in second if m["role"] == "tool"][0]
    entry = tool_msg["tool_results"][0]
    assert entry["name"] == "read_file"
    assert "CustomerStatus.ACTIVE" in entry["response"]["output"]["content"]
    assert state.status == InvestigationStatus.COMPLETED


def test_multiple_iterations_retain_previous_context(repo: Path) -> None:
    llm = scripted(
        resp(call("list_directory", path=".")),
        resp(read("src/OrderService.java")),
        resp(read("src/CustomerService.java")),
        resp(good_finish()),
    )
    run_loop(repo, llm)
    assert llm.calls == 4
    for prev, cur in zip(llm.snapshots, llm.snapshots[1:]):
        assert len(cur) > len(prev)
        assert all(a is b for a, b in zip(prev, cur)), "earlier turns must be preserved unchanged"
    final = llm.snapshots[-1]
    names = [r["name"] for m in final if m["role"] == "tool" for r in m["tool_results"]]
    assert names == ["list_directory", "read_file", "read_file"]


def test_multiple_tool_calls_in_one_response(repo: Path) -> None:
    llm = scripted(
        resp(read("src/OrderService.java"), read("src/CustomerService.java"), text="Reading both."),
        resp(good_finish()),
    )
    state = run_loop(repo, llm)

    assistants = [m for m in state.messages if m["role"] == "assistant"]
    assert len(assistants) == 2  # one per response, no duplicates
    assert len(assistants[0]["tool_calls"]) == 2
    tm = tool_messages(state)
    assert len(tm) == 1 and len(tm[0]["tool_results"]) == 2
    files = [r["response"]["output"]["file"].replace("\\", "/") for r in tm[0]["tool_results"]]
    assert files == ["src/OrderService.java", "src/CustomerService.java"]
    assert state.status == InvestigationStatus.COMPLETED


def test_tool_errors_are_structured_and_investigation_continues(repo: Path) -> None:
    llm = scripted(resp(read("src/Missing.java")), resp(call("no_such_tool")), resp(read("src/CustomerService.java")),
                   resp(good_finish()))
    state = run_loop(repo, llm)
    results = [r["response"] for m in tool_messages(state) for r in m["tool_results"]]
    assert "error" in results[0] and "error" in results[1]
    assert state.status == InvestigationStatus.COMPLETED


def test_hypothesis_revised_after_contradictory_evidence(repo: Path) -> None:
    """The mock LLM first suspects OrderService, sees contradicting code, then revises."""

    def brain(messages: list[dict[str, Any]], i: int) -> LLMResponse:
        if i == 0:
            return resp(read("src/OrderService.java"), text="Hypothesis: createOrder is at fault.")
        if i == 1:
            seen = json.dumps(messages[-1]["tool_results"])
            assert "validateCustomerEligibility" in seen and "getStatus" not in seen  # no status logic here
            return resp(read("src/CustomerService.java"), text="Revised: the check lives in CustomerService.")
        return resp(good_finish())

    state = run_loop(repo, FnLLM(brain))
    assert state.status == InvestigationStatus.COMPLETED
    assert state.final_rca["evidence"][0]["file"] == "src/CustomerService.java"


# ── completion gate ──────────────────────────────────────────────────────────

def test_valid_evidence_allows_completion(repo: Path) -> None:
    state = run_loop(repo, scripted(resp(read("src/CustomerService.java")), resp(good_finish())))
    assert state.status == InvestigationStatus.COMPLETED
    report = RCAReport.model_validate(state.final_rca)
    assert report.status == RCAStatus.COMPLETED and report.evidence


@pytest.mark.parametrize(
    "evidence, needle",
    [
        ([{"file": "../outside.java", "start_line": 1, "end_line": 2, "explanation": "`CustomerService`"}],
         "outside the repository"),
        ([{"file": "src/Nope.java", "start_line": 1, "end_line": 2, "explanation": "`CustomerService`"}],
         "does not exist"),
        ([{"file": "src/CustomerService.java", "start_line": 2, "end_line": 99,
           "explanation": "`validateCustomerEligibility`"}], "exceed the file length"),
        ([{"file": "src/CustomerService.java", "start_line": 0, "end_line": 3,
           "explanation": "`validateCustomerEligibility`"}], "invalid line range"),
        ([{"file": "src/CustomerService.java", "start_line": 3, "end_line": 2,
           "explanation": "`validateCustomerEligibility`"}], "invalid line range"),
        ([{"file": "src/CustomerService.java", "start_line": 2, "end_line": 6,
           "explanation": "The method `if (user.isBanned()) rejectUserNow()` is wrong."}], "does not appear"),
        ([{"file": "src/CustomerService.java", "start_line": 2, "end_line": 6,
           "explanation": "This looks wrong."}], "does not reference any identifier"),
        ([], "No valid evidence"),
    ],
)
def test_invalid_evidence_prevents_completion(repo: Path, evidence: list, needle: str) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 50))
    outcome = validate_finish(json.loads(json.dumps(good_finish(evidence=evidence)["arguments"])),
                              ledger, "issue")
    assert not outcome.ok
    assert any(needle in e for e in outcome.errors), outcome.errors


def test_evidence_for_lines_never_read_is_rejected(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 2))  # lines 3-6 never returned
    outcome = validate_finish(good_finish()["arguments"], ledger, "issue")
    assert not outcome.ok
    assert any("never returned by a tool" in e for e in outcome.errors)


def test_evidence_for_file_never_read_is_rejected(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/OrderService.java", 1, 6))
    outcome = validate_finish(good_finish()["arguments"], ledger, "issue")
    assert not outcome.ok


def test_root_cause_and_fix_must_match_evidence(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    unrelated_cause = validate_finish(
        good_finish(root_cause="The database connection pool is exhausted.")["arguments"], ledger, "i")
    assert not unrelated_cause.ok and any("root_cause is not supported" in e for e in unrelated_cause.errors)
    unrelated_fix = validate_finish(
        good_finish(recommended_fix="Restart the server and clear the caches.")["arguments"], ledger, "i")
    assert not unrelated_fix.ok and any("recommended_fix does not address" in e for e in unrelated_fix.errors)


def test_rejected_finish_returns_structured_error_and_loop_continues(repo: Path) -> None:
    bad = good_finish()  # evidence cites lines never read -> rejected
    llm = scripted(resp(bad), resp(read("src/CustomerService.java")), resp(good_finish()))
    state = run_loop(repo, llm)

    first_result = tool_messages(state)[0]["tool_results"][0]
    assert first_result["name"] == "finish_investigation"
    assert "validation_errors" in first_result["response"]
    assert first_result["response"]["inspected_ranges"] == {}
    assert state.status == InvestigationStatus.COMPLETED
    assert [a["accepted"] for a in state.finish_attempts] == [False, True]
    assert state.finish_accepted_count == 1


def test_insufficient_investigation_stays_incomplete(repo: Path) -> None:
    llm = scripted(resp(good_finish()))  # never reads the file; keeps resubmitting
    state = run_loop(repo, llm, max_iterations=4)

    assert state.status in (InvestigationStatus.INCOMPLETE, InvestigationStatus.INVESTIGATION_LIMIT_REACHED)
    assert state.status != InvestigationStatus.COMPLETED
    report = RCAReport.model_validate(state.final_rca)
    assert report.status == RCAStatus.INCOMPLETE
    assert report.confidence == 0.0 and report.evidence == []
    assert any("UNVERIFIED hypothesis" in u for u in report.remaining_uncertainty)
    assert state.finish_accepted_count == 0 and len(state.finish_attempts) <= 4


def test_explicit_incomplete_finish_is_accepted_as_incomplete(repo: Path) -> None:
    llm = scripted(resp(read("src/OrderService.java")), resp(call(
        "finish_investigation", status="INCOMPLETE",
        root_cause="Established only that OrderService.createOrder delegates to validateCustomerEligibility.",
        evidence=[], recommended_fix="Inspect CustomerService.validateCustomerEligibility.",
        developer_next_steps=["Read CustomerService."], confidence=0.2,
        verification_performed=["Read OrderService.createOrder"],
        remaining_uncertainty=["CustomerService was not inspected."],
        investigation_summary=["Read OrderService"])))
    state = run_loop(repo, llm)
    assert state.status == InvestigationStatus.INCOMPLETE
    report = RCAReport.model_validate(state.final_rca)
    assert report.status == RCAStatus.INCOMPLETE and report.confidence == 0.2


def test_llm_failure_yields_error_status_with_valid_report(repo: Path) -> None:
    def boom(messages, i):
        raise RuntimeError("quota exceeded")
    state = run_loop(repo, FnLLM(boom))
    assert state.status == InvestigationStatus.ERROR
    report = RCAReport.model_validate(state.final_rca)
    assert report.status == RCAStatus.ERROR
    assert any("quota exceeded" in u for u in report.remaining_uncertainty)


# ── finish handled exactly once ──────────────────────────────────────────────

def test_finish_is_processed_once_and_loop_stops(repo: Path) -> None:
    llm = scripted(resp(read("src/CustomerService.java")), resp(good_finish(), good_finish()), resp(good_finish()))
    state = run_loop(repo, llm)
    assert llm.calls == 2, "no LLM call may follow an accepted finish"
    assert state.finish_accepted_count == 1
    assert len(state.finish_attempts) == 1, "the duplicate finish in the same response is not evaluated"
    assert sum(1 for t in state.trace if t.kind == "RCA") == 1
    assert state.status == InvestigationStatus.COMPLETED


def test_finish_alongside_other_tool_uses_evidence_from_same_response(repo: Path) -> None:
    llm = scripted(resp(good_finish(), read("src/CustomerService.java")))
    state = run_loop(repo, llm)
    assert state.status == InvestigationStatus.COMPLETED and llm.calls == 1


def test_duplicate_finish_in_rejected_response_gets_error_for_each_call(repo: Path) -> None:
    llm = scripted(resp(good_finish(), good_finish()), resp(read("src/CustomerService.java")), resp(good_finish()))
    state = run_loop(repo, llm)
    first = tool_messages(state)[0]
    assert len(first["tool_results"]) == 2 and all("error" in r["response"] for r in first["tool_results"])
    assert state.finish_accepted_count == 1


# ── report model, renderer, JSON/CLI agreement ───────────────────────────────

def completed_report(repo: Path) -> RCAReport:
    state = run_loop(repo, scripted(resp(read("src/CustomerService.java")), resp(good_finish())))
    return RCAReport.model_validate(state.final_rca)


def test_rca_sections_appear_exactly_once_in_order(repo: Path) -> None:
    text = render_report(completed_report(repo))
    counts = section_counts(text)
    assert counts == {t: 1 for t in SECTION_TITLES}, counts
    positions = [text.index(f"  {t}\n") for t in SECTION_TITLES]
    assert positions == sorted(positions)


def test_sections_once_even_for_incomplete_and_empty_lists(repo: Path) -> None:
    state = run_loop(repo, scripted(resp(good_finish())), max_iterations=2)
    text = render_report(RCAReport.model_validate(state.final_rca))
    assert section_counts(text) == {t: 1 for t in SECTION_TITLES}
    minimal = RCAReport(status="INCOMPLETE", problem="p", root_cause="r", confidence=0.0, recommended_fix="f")
    text = render_report(minimal)
    assert section_counts(text) == {t: 1 for t in SECTION_TITLES}
    assert "Unavailable" in text


def test_confidence_displayed_as_percentage(repo: Path) -> None:
    assert format_confidence(0.9) == "90%" and format_confidence(0.0) == "0%" and format_confidence(1.0) == "100%"
    text = render_report(completed_report(repo))
    assert re.search(r"CONFIDENCE\n=+\n  85%\n", text)


@pytest.mark.parametrize("bad", [1.5, -0.1, "high", True, None])
def test_confidence_must_be_numeric_between_0_and_1(bad: Any) -> None:
    with pytest.raises(ValidationError):
        RCAReport(status="INCOMPLETE", problem="p", root_cause="r", confidence=bad, recommended_fix="f")


def test_missing_required_field_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        RCAReport.model_validate({"status": "INCOMPLETE", "problem": "p"})


def test_list_fields_stay_lists(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    args = good_finish(
        developer_next_steps="['Fix the condition', 'Add a regression test']",   # python-repr string
        investigation_summary='["a", "b"]',                                      # JSON string
        verification_performed="Read the method",                                 # plain string
    )["arguments"]
    out = validate_finish(args, ledger, "issue")
    assert out.ok, out.errors
    r = out.report
    assert r.developer_next_steps == ["Fix the condition", "Add a regression test"]
    assert r.investigation_summary == ["a", "b"]
    assert "Read the method" in r.verification_performed
    assert all(isinstance(x, str) and not x.startswith("[") for x in r.developer_next_steps)
    assert coerce_list("['x']") == ["x"] and coerce_list(None) == []


def test_json_artifact_and_cli_report_agree(repo: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Customer eligibility defect"}))
    llm = scripted(router_resp, resp(read("src/CustomerService.java")), resp(good_finish()))
    agent = InvestigationAgent(str(repo), llm_provider=llm, max_iterations=5, runs_dir=runs)
    result = agent.investigate("Active customers cannot create orders")

    saved = json.loads(Path(result["run_file"]).read_text(encoding="utf-8"))
    assert saved["final_rca"] == result["rca"], "artifact and returned RCA must be the same object"
    assert saved["finish_accepted_count"] == 1
    cli_text = render_report(RCAReport.model_validate(result["rca"]))
    json_text = render_report(RCAReport.model_validate(saved["final_rca"]))
    assert cli_text == json_text
    assert saved["llm_config"]["model"] == "mock"
    assert all("provider_content" not in m for m in saved["messages"])


# ── honesty about verification ───────────────────────────────────────────────

@pytest.mark.parametrize("claim", [
    "All unit tests passed after the change.",
    "Ran the test suite and it passes.",
    "Build succeeded and tests are green.",
])
def test_fabricated_test_results_are_rejected(repo: Path, claim: str) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    out = validate_finish(good_finish(verification_performed=[claim])["arguments"], ledger, "issue")
    assert not out.ok and any("cannot execute tests" in e for e in out.errors)


def test_report_states_tests_were_not_run(repo: Path) -> None:
    report = completed_report(repo)
    assert TEST_EXECUTION_NOTE in report.verification_performed
    assert any("src/CustomerService.java:1-7" in v for v in report.verification_performed)
    assert not any(re.search(r"tests? (passed|pass)\b", v, re.I) for v in report.verification_performed)
    assert any("not applied" in u for u in report.remaining_uncertainty)


def test_honest_negative_test_statement_is_allowed(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    out = validate_finish(
        good_finish(verification_performed=["Tests were not run; the agent is read-only."])["arguments"],
        ledger, "issue")
    assert out.ok, out.errors


# ── reminders & conversation shape ───────────────────────────────────────────

def test_iteration_reminder_does_not_add_user_messages(repo: Path) -> None:
    llm = scripted(resp(call("list_directory", path=".")))
    state = run_loop(repo, llm, max_iterations=12)
    assert [m["role"] for m in state.messages].count("user") == 1  # only the initial issue message
    noted = [m for m in tool_messages(state) if m.get("note")]
    assert 1 <= len(noted) <= 3  # iterations 5, 10 and the final one
    roles = [m["role"] for m in state.messages]
    assert all(a != b for a, b in zip(roles[1:], roles[2:])), "assistant/tool turns must alternate"


def test_reminder_text_is_evidence_oriented_not_blind_continue() -> None:
    text = build_iteration_prompt(5, 30)
    assert "establishes" in text and "gap" in text and "repeat" in text and "finish_investigation" in text
    assert "Few iterations" in build_iteration_prompt(29, 30)
    final = build_iteration_prompt(30, 30)
    assert "FINAL" in final and "INCOMPLETE" in final


def test_text_only_response_gets_single_nudge(repo: Path) -> None:
    llm = scripted(resp(text="Let me think."), resp(read("src/CustomerService.java")), resp(good_finish()))
    state = run_loop(repo, llm)
    assert [m["role"] for m in state.messages].count("user") == 2  # issue + one nudge
    assert state.status == InvestigationStatus.COMPLETED


# ── Gemini conversion & configuration ────────────────────────────────────────

def test_gemini_conversion_pairs_calls_with_responses_and_keeps_provider_content(repo: Path) -> None:
    import google.genai.types as t
    from agent.llm.gemini import _to_gemini_messages

    original = t.Content(role="model", parts=[
        t.Part(function_call=t.FunctionCall(name="read_file", args={"file_path": "a"})),
        t.Part(function_call=t.FunctionCall(name="list_directory", args={})),
    ])
    llm = scripted(resp(read("src/OrderService.java"), call("list_directory", path=".")), resp(good_finish()))
    state = run_loop(repo, llm)
    state.messages[1]["provider_content"] = original  # as the real provider would store it

    contents = _to_gemini_messages(state.messages[:3])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is original
    responses = [p.function_response for p in contents[2].parts if p.function_response]
    calls = [p.function_call for p in contents[1].parts if p.function_call]
    assert [r.name for r in responses] == [c.name for c in calls] == ["read_file", "list_directory"]
    assert "output" in responses[0].response


def test_gemini_conversion_rebuilds_function_calls_without_provider_content(repo: Path) -> None:
    from agent.llm.gemini import _to_gemini_messages

    state = run_loop(repo, scripted(resp(read("src/OrderService.java"), text="go"), resp(good_finish())))
    contents = _to_gemini_messages(state.messages[:3])
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "go" and contents[1].parts[1].function_call.name == "read_file"


def test_reminder_note_becomes_text_part_in_same_content() -> None:
    from agent.llm.gemini import _to_gemini_messages
    from agent.core.prompts import build_tool_response_message

    msg = build_tool_response_message([call("read_file")], [{"error": "nope"}], note="REMINDER")
    (content,) = _to_gemini_messages([msg])
    assert content.role == "user" and len(content.parts) == 2
    assert content.parts[0].function_response.response == {"error": "nope"}
    assert content.parts[1].text == "REMINDER"


def test_finish_schema_uses_typed_arrays_and_converts_for_gemini() -> None:
    import google.genai.types as t
    from agent.llm.gemini import _build_tool

    props = FINISH_TOOL_DEFINITION["parameters"]["properties"]
    assert props["evidence"]["type"] == "array" and props["confidence"]["type"] == "number"
    tool = _build_tool([FINISH_TOOL_DEFINITION])
    schema = tool.function_declarations[0].parameters
    assert schema.properties["evidence"].type == t.Type.ARRAY
    assert schema.properties["evidence"].items.properties["start_line"].type == t.Type.INTEGER
    assert schema.properties["confidence"].type == t.Type.NUMBER
    assert schema.properties["developer_next_steps"].items.type == t.Type.STRING


def test_thinking_level_is_configurable_validated_and_reported() -> None:
    from agent.llm.gemini import GeminiProvider

    p = GeminiProvider(api_key="test-key", model="gemini-3.5-flash-lite", thinking_level="low")
    d = p.describe()
    assert d["model"] == "gemini-3.5-flash-lite" and d["thinking_level"] == "low"
    default = GeminiProvider(api_key="test-key", thinking_level="")
    assert "not set" in default.describe()["thinking_level"]
    with pytest.raises(ValueError, match="Unsupported GEMINI_THINKING_LEVEL"):
        GeminiProvider(api_key="test-key", thinking_level="extreme")


def test_parse_response_keeps_function_call_ids_and_skips_thoughts() -> None:
    import google.genai.types as t
    from agent.llm.gemini import GeminiProvider

    content = t.Content(role="model", parts=[
        t.Part(text="internal reasoning", thought=True),
        t.Part(text="visible"),
        t.Part(function_call=t.FunctionCall(id="abc", name="read_file", args={"file_path": "x"})),
    ])
    fake = type("R", (), {"candidates": [type("C", (), {"content": content, "finish_reason": None})()]})()
    out = GeminiProvider(api_key="k", thinking_level="")._parse_response(fake)
    assert out.text == "visible"
    assert out.tool_calls == [{"name": "read_file", "arguments": {"file_path": "x"}, "id": "abc"}]
    assert out.raw_candidate is content


def test_enumeration_prefixes_are_stripped_from_list_items(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    out = validate_finish(
        good_finish(investigation_summary=["1. Read the file", "2) Found the condition", "- done"])["arguments"],
        ledger, "issue")
    assert out.ok and out.report.investigation_summary == ["Read the file", "Found the condition", "done"]


# ── confidence calibration & scenario regressions ────────────────────────────

def test_confidence_100_percent_rejected(repo: Path) -> None:
    ledger = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    out = validate_finish(good_finish(confidence=1.0)["arguments"], ledger, "issue")
    assert not out.ok
    assert any("Confidence cannot be 1.0 (100%)" in err for err in out.errors)


def test_high_confidence_static_evidence_multi_file_vs_isolated(repo: Path) -> None:
    ledger_single = make_ledger(repo, ("src/CustomerService.java", 1, 7))
    out_single = validate_finish(good_finish(confidence=0.88)["arguments"], ledger_single, "issue")
    assert not out_single.ok
    assert any("investigation is localized to a single file" in err for err in out_single.errors)

    ledger_multi = make_ledger(
        repo,
        ("src/CustomerService.java", 1, 7),
        ("src/OrderService.java", 1, 10),
    )
    out_multi = validate_finish(good_finish(confidence=0.88)["arguments"], ledger_multi, "issue")
    assert out_multi.ok, out_multi.errors
    assert out_multi.report.confidence == 0.88


def test_contradictory_evidence_limits_confidence(repo: Path) -> None:
    ledger = make_ledger(
        repo,
        ("src/CustomerService.java", 1, 7),
        ("src/OrderService.java", 1, 10),
    )
    out_high = validate_finish(
        good_finish(
            confidence=0.80,
            remaining_uncertainty=["Contradictory evidence in caller suggests competing hypothesis."],
        )["arguments"],
        ledger,
        "issue",
    )
    assert not out_high.ok
    assert any("contradictory evidence or competing hypotheses" in err for err in out_high.errors)

    out_calibrated = validate_finish(
        good_finish(
            confidence=0.65,
            remaining_uncertainty=["Contradictory evidence in caller suggests competing hypothesis."],
        )["arguments"],
        ledger,
        "issue",
    )
    assert out_calibrated.ok, out_calibrated.errors


def test_unverified_runtime_behavior_limits_confidence(repo: Path) -> None:
    ledger = make_ledger(
        repo,
        ("src/CustomerService.java", 1, 7),
        ("src/OrderService.java", 1, 10),
    )
    out_high = validate_finish(
        good_finish(
            confidence=0.85,
            remaining_uncertainty=["Runtime behavior dependent on dynamic configuration and unverified runtime."],
        )["arguments"],
        ledger,
        "issue",
    )
    assert not out_high.ok
    assert any("unverified runtime behavior is noted" in err for err in out_high.errors)

    out_calibrated = validate_finish(
        good_finish(
            confidence=0.75,
            remaining_uncertainty=["Runtime behavior dependent on dynamic configuration and unverified runtime."],
        )["arguments"],
        ledger,
        "issue",
    )
    assert out_calibrated.ok, out_calibrated.errors


def test_regression_scenario_active_customer_mocked(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "CustomerService.java").write_text(CUSTOMER_SERVICE)
    (src / "OrderService.java").write_text(ORDER_SERVICE)

    llm = scripted(
        resp(read("src/OrderService.java", 1, 15)),
        resp(read("src/CustomerService.java", 1, 7)),
        resp(good_finish(
            confidence=0.88,
            verification_performed=["Traced createOrder to validateCustomerEligibility.", "Verified customer status check in CustomerService."],
        )),
    )
    before_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    state = run_loop(tmp_path, llm)
    after_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

    assert before_stat == after_stat, "Repository files must remain unmodified (read-only)"
    assert state.status == InvestigationStatus.COMPLETED
    assert state.final_rca is not None
    assert state.final_rca["confidence"] == 0.88
    assert "CustomerStatus.ACTIVE" in state.final_rca["root_cause"]


def test_regression_scenario_caller_dependency_mismatch_mocked(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    order_svc_content = (
        "public class OrderService {\n"
        "    public Order confirmOrder(Long orderId) {\n"
        "        Order order = orderRepository.findById(orderId).orElseThrow();\n"
        "        paymentService.processPayment(order.getId(), order.getTotal());\n"
        "        return order;\n"
        "    }\n"
        "}\n"
    )
    payment_svc_content = (
        "public class PaymentService {\n"
        "    public void processPayment(Long customerId, BigDecimal amount) {\n"
        "        Customer customer = customerRepository.findById(customerId)\n"
        "            .orElseThrow(() -> new CustomerNotFoundException(customerId));\n"
        "    }\n"
        "}\n"
    )
    (src / "OrderService.java").write_text(order_svc_content)
    (src / "PaymentService.java").write_text(payment_svc_content)

    finish_call = call(
        "finish_investigation",
        status="COMPLETED",
        root_cause="OrderService.confirmOrder passes `order.getId()` instead of `order.getCustomerId()` to PaymentService.processPayment, leading to CustomerNotFoundException.",
        evidence=[
            {
                "file": "src/OrderService.java",
                "symbol": "confirmOrder",
                "start_line": 2,
                "end_line": 6,
                "explanation": "Line 4 calls `paymentService.processPayment(order.getId(), ...)` passing order ID instead of customer ID.",
            },
            {
                "file": "src/PaymentService.java",
                "symbol": "processPayment",
                "start_line": 1,
                "end_line": 5,
                "explanation": "`processPayment` expects `customerId` and queries `customerRepository` with that ID.",
            },
        ],
        recommended_fix="In OrderService.confirmOrder, update the call to `paymentService.processPayment(order.getCustomerId(), order.getTotal());`.",
        developer_next_steps=["Update OrderService.confirmOrder and add integration test with different order and customer IDs."],
        confidence=0.88,
        verification_performed=["Inspected OrderService.confirmOrder and PaymentService.processPayment."],
        remaining_uncertainty=["Runtime payment gateway communication was not executed."],
        investigation_summary=["Traced confirmOrder to processPayment"],
    )

    llm = scripted(
        resp(read("src/OrderService.java", 1, 7)),
        resp(read("src/PaymentService.java", 1, 6)),
        resp(finish_call),
    )
    before_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    state = run_loop(tmp_path, llm)
    after_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

    assert before_stat == after_stat, "Repository files must remain unmodified (read-only)"
    assert state.status == InvestigationStatus.COMPLETED
    assert state.final_rca is not None
    assert state.final_rca["confidence"] == 0.88
    assert "processPayment" in state.final_rca["root_cause"]


def test_regression_scenario_incorrect_validation_logic_mocked(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pricing_svc_content = (
        "public class PricingService {\n"
        "    public void validateDiscountEligibility(BigDecimal total, BigDecimal min) {\n"
        "        if (total.compareTo(min) <= 0) {\n"
        "            throw new InvalidOrderException();\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    order_svc_content = (
        "public class OrderService {\n"
        "    public Order applyDiscount(Long orderId, BigDecimal min) {\n"
        "        pricingService.validateDiscountEligibility(order.getTotal(), min);\n"
        "        return order;\n"
        "    }\n"
        "}\n"
    )
    (src / "PricingService.java").write_text(pricing_svc_content)
    (src / "OrderService.java").write_text(order_svc_content)

    finish_call = call(
        "finish_investigation",
        status="COMPLETED",
        root_cause="PricingService.validateDiscountEligibility rejects orders meeting exact minimum threshold because the boundary condition uses `<= 0` instead of `< 0` in `total.compareTo(min)`.",
        evidence=[
            {
                "file": "src/PricingService.java",
                "symbol": "validateDiscountEligibility",
                "start_line": 2,
                "end_line": 6,
                "explanation": "`total.compareTo(min) <= 0` incorrectly rejects qualifying orders where order total equals the required minimum.",
            },
            {
                "file": "src/OrderService.java",
                "symbol": "applyDiscount",
                "start_line": 1,
                "end_line": 5,
                "explanation": "OrderService delegates discount validation to `pricingService.validateDiscountEligibility`.",
            },
        ],
        recommended_fix="Change `total.compareTo(min) <= 0` to `total.compareTo(min) < 0` so that orders meeting the exact threshold qualify.",
        developer_next_steps=["Fix boundary condition in PricingService and add boundary test cases for exact minimum amount."],
        confidence=0.88,
        verification_performed=["Inspected PricingService.validateDiscountEligibility and OrderService.applyDiscount."],
        remaining_uncertainty=["Promotion campaign configuration was not verified in database."],
        investigation_summary=["Examined discount validation boundary check."],
    )

    llm = scripted(
        resp(read("src/OrderService.java", 1, 6)),
        resp(read("src/PricingService.java", 1, 7)),
        resp(finish_call),
    )
    before_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    state = run_loop(tmp_path, llm)
    after_stat = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

    assert before_stat == after_stat, "Repository files must remain unmodified (read-only)"
    assert state.status == InvestigationStatus.COMPLETED
    assert state.final_rca is not None
    assert state.final_rca["confidence"] == 0.88
    assert "validateDiscountEligibility" in state.final_rca["root_cause"]


# ── Scope Guard, Circuit Breaker, and Rate-Limit Tests ───────────────────────

def test_what_is_your_name_returns_out_of_scope_without_llm_call(repo: Path) -> None:
    def fail_if_called(msgs: Any, i: int) -> LLMResponse:
        raise AssertionError("LLM should not be called for fast-path input!")

    mock_llm = FnLLM(fail_if_called)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("what is your name")

    assert result["status"] in ("OUT_OF_SCOPE", "GENERAL_QUESTION")
    assert result["rca"] is None
    assert mock_llm.calls == 0


def test_greeting_or_general_knowledge_does_not_enter_loop(repo: Path) -> None:
    def fail_if_called(msgs: Any, i: int) -> LLMResponse:
        raise AssertionError("LLM should not be called for fast-path input!")

    mock_llm = FnLLM(fail_if_called)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)

    for prompt in ["hello", "good morning!", "what is the capital of France?", "tell me a joke"]:
        result = agent.investigate(prompt)
        assert result["status"] in ("OUT_OF_SCOPE", "GENERAL_QUESTION")
        assert result["rca"] is None
        assert mock_llm.calls == 0


def test_legitimate_bug_report_enters_investigation_loop(repo: Path) -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Customer eligibility defect"}))
    mock_llm = scripted(router_resp, resp(read("src/CustomerService.java", 1, 7)), resp(good_finish()))
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Active customers are unable to create orders")

    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None
    assert mock_llm.calls == 3


def test_ambiguous_plausible_bug_reports_not_rejected() -> None:
    from agent.core.scope import validate_issue_scope

    plausible = [
        "it doesn't work",
        "orders broken",
        "crashes on startup",
        "payment fails",
        "discount calculation wrong",
        "NullPointerException in controller",
        "500 Internal Server Error",
    ]
    for issue in plausible:
        res = validate_issue_scope(issue)
        assert res.is_in_scope, f"Expected '{issue}' to be in-scope"


def test_repeated_identical_completion_gate_failures_trigger_circuit_breaker(repo: Path) -> None:
    bad_finish = call("finish_investigation", status="COMPLETED", root_cause="bogus", evidence=[])
    mock_llm = scripted(resp(bad_finish), resp(bad_finish), resp(bad_finish))

    state = run_loop(repo, mock_llm, max_iterations=10)

    # Must stop on attempt 2 without executing all 10 iterations
    assert state.status == InvestigationStatus.INCOMPLETE
    assert state.final_rca is not None
    assert state.final_rca["status"] == "INCOMPLETE"
    assert len(state.finish_attempts) == 2
    assert state.finish_attempts[1]["repeated"] is True
    assert any("repeated identical finish_investigation payload" in u for u in state.final_rca["remaining_uncertainty"])


def test_completion_rejection_gives_llm_actual_validation_errors(repo: Path) -> None:
    bad_finish = call("finish_investigation", status="COMPLETED", root_cause="CustomerService.validateCustomerEligibility", evidence=[])
    mock_llm = scripted(resp(bad_finish), resp(read("src/CustomerService.java", 1, 7)), resp(good_finish()))

    # First attempt has no read_file evidence; second turn reads file; third finishes
    state = run_loop(repo, mock_llm, max_iterations=5)

    assert len(state.finish_attempts) == 2
    assert state.finish_attempts[0]["accepted"] is False
    assert len(state.finish_attempts[0]["errors"]) > 0
    assert state.finish_attempts[1]["accepted"] is True
    assert state.status == InvestigationStatus.COMPLETED

    # Inspect the tool response returned to the LLM in message turn 2
    tool_msg = state.messages[2]
    assert tool_msg["role"] == "tool"
    first_res = tool_msg["tool_results"][0]["response"]
    assert "attempt 1 of 3" in first_res["error"]
    assert "validation_errors" in first_res
    assert len(first_res["validation_errors"]) > 0


def test_reaching_rejection_limit_returns_incomplete_instead_of_looping(repo: Path) -> None:
    # 3 distinct invalid finish calls
    bad_1 = call("finish_investigation", status="COMPLETED", root_cause="hypothesis 1", evidence=[])
    bad_2 = call("finish_investigation", status="COMPLETED", root_cause="hypothesis 2", evidence=[])
    bad_3 = call("finish_investigation", status="COMPLETED", root_cause="hypothesis 3", evidence=[])
    bad_4 = call("finish_investigation", status="COMPLETED", root_cause="hypothesis 4", evidence=[])

    mock_llm = scripted(resp(bad_1), resp(bad_2), resp(bad_3), resp(bad_4))

    state = run_loop(repo, mock_llm, max_iterations=10)

    # Rejection limit is 3: must stop after attempt 3
    assert state.status == InvestigationStatus.INCOMPLETE
    assert len(state.finish_attempts) == 3
    assert state.final_rca is not None
    assert state.final_rca["status"] == "INCOMPLETE"
    assert any("maximum finish_investigation rejection limit (3 attempts) reached" in u for u in state.final_rca["remaining_uncertainty"])


def test_quota_exhaustion_handled_without_unbounded_retry_or_traceback(repo: Path) -> None:
    from agent.llm.base import LLMQuotaExhaustedError

    class QuotaFailingLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def model_name(self) -> str:
            return "quota-mock"

        def chat(self, messages, tools=None, system_prompt=None) -> LLMResponse:
            self.calls += 1
            raise LLMQuotaExhaustedError("RESOURCE_EXHAUSTED: quota exceeded for model")

    llm = QuotaFailingLLM()
    state = run_loop(repo, llm)

    assert state.status == InvestigationStatus.ERROR
    assert state.final_rca is not None
    assert state.final_rca["status"] == "ERROR"
    assert any("quota exhausted" in u.lower() for u in state.final_rca["remaining_uncertainty"])
    assert any(t.kind == "ERROR" and "quota exhausted" in t.description.lower() for t in state.trace)


# ── SECTION 5: Intent Router & Request Budget Regression Tests ────────────────

def test_what_is_java_classified_as_general_question_without_repo_tools(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "GENERAL_QUESTION",
        "reason": "General programming conceptual question.",
        "direct_answer": "Java is a class-based, object-oriented programming language."
    }))
    mock_llm = scripted(router_resp)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("What is Java?")

    assert result["status"] == "GENERAL_QUESTION"
    assert result["intent"] == "GENERAL_QUESTION"
    assert "Java is a class-based" in result["direct_answer"]
    assert result["rca"] is None
    assert mock_llm.calls == 1
    assert all(t["kind"] != "TOOL_RESULT" for t in result["trace"])


def test_what_is_your_name_answered_or_routed_without_repo_investigation(repo: Path) -> None:
    def fail_if_tools(msgs: Any, i: int) -> LLMResponse:
        raise AssertionError("LLM should not be called with tools!")
    mock_llm = FnLLM(fail_if_tools)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("What is your name?")

    assert result["status"] in ("GENERAL_QUESTION", "OUT_OF_SCOPE")
    assert result["rca"] is None
    assert all(t["kind"] != "TOOL_RESULT" for t in result["trace"])


def test_hello_does_not_trigger_repo_searches(repo: Path) -> None:
    def fail_if_called(msgs: Any, i: int) -> LLMResponse:
        raise AssertionError("Should use deterministic fast path!")
    mock_llm = FnLLM(fail_if_called)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Hello")

    assert result["status"] in ("GENERAL_QUESTION", "OUT_OF_SCOPE")
    assert result["rca"] is None
    assert all(t["kind"] != "TOOL_RESULT" for t in result["trace"])
    assert mock_llm.calls == 0


def test_general_question_requires_only_one_router_response(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "GENERAL_QUESTION",
        "reason": "Conceptual computer science question.",
        "direct_answer": "Recursion is a programming technique where a function calls itself."
    }))
    mock_llm = scripted(router_resp)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Explain recursion")

    assert result["status"] == "GENERAL_QUESTION"
    assert result["direct_answer"] is not None
    assert "calls itself" in result["direct_answer"]
    assert result["request_count"] == 1
    assert mock_llm.calls == 1


def test_customer_story_endpoint_returns_500_enters_code_investigation(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "CODE_INVESTIGATION",
        "reason": "HTTP 500 server error."
    }))
    mock_llm = scripted(
        router_resp,
        resp(read("src/CustomerService.java", 1, 7)),
        resp(good_finish()),
    )
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("the customer story endpoint returns 500")

    assert result["intent"] == "CODE_INVESTIGATION"
    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None
    assert mock_llm.calls == 3


def test_payment_fails_enters_code_investigation(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "CODE_INVESTIGATION",
        "reason": "Payment failure defect."
    }))
    mock_llm = scripted(
        router_resp,
        resp(read("src/CustomerService.java", 1, 7)),
        resp(good_finish()),
    )
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Payment fails")

    assert result["intent"] == "CODE_INVESTIGATION"
    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None
    assert mock_llm.calls == 3


def test_ambiguous_inputs_handled_conservatively() -> None:
    from agent.core.router import route_intent
    for prompt in ["orders broken", "it crashes", "payment fails"]:
        res, _ = route_intent(prompt, llm=None)
        assert res.intent == "CODE_INVESTIGATION"


def test_invalid_router_output_handled_safely(repo: Path) -> None:
    bad_resp = resp(text="Plain text that is not valid JSON at all.")
    mock_llm = scripted(
        bad_resp,
        resp(read("src/CustomerService.java", 1, 7)),
        resp(good_finish()),
    )
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("orders broken")

    assert result["intent"] == "CODE_INVESTIGATION"
    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None


def test_request_budget_stops_additional_llm_requests(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "CODE_INVESTIGATION",
        "reason": "Defect report."
    }))
    mock_llm = scripted(
        router_resp,
        resp(read("src/CustomerService.java", 1, 5)),
        resp(read("src/CustomerService.java", 6, 10)),
        resp(read("src/CustomerService.java", 11, 15)),
        resp(read("src/CustomerService.java", 16, 20)),
    )
    # Budget of 3: 1 for router, 2 for loop turns, then stops before turn 4
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm, request_budget=3)
    result = agent.investigate("orders broken")

    assert result["status"] == "INCOMPLETE"
    assert result["request_count"] == 3
    assert result["request_budget"] == 3
    assert "budget" in result["termination_reason"].lower()
    assert mock_llm.calls == 3


def test_repeated_unproductive_searches_are_bounded(repo: Path) -> None:
    search_call = call("search_code", query="order")
    mock_llm = scripted(
        resp(search_call),
        resp(search_call),
        resp(search_call),
    )
    state = run_loop(repo, mock_llm, max_iterations=5)

    assert state.status == InvestigationStatus.INCOMPLETE
    assert "repeated unproductive searches" in state.termination_reason.lower()


def test_http_429_with_retry_metadata_respects_specified_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.llm.gemini import GeminiProvider

    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    class MockClientError(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"
        message = "Resource exhausted. Please retry after 7.5s."
        details = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "7.5s",
                    }
                ]
            }
        }

    provider = GeminiProvider(api_key="mock-key", max_retries=2)
    attempt_count = 0

    def mock_generate(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise MockClientError()
        class MockCand:
            content = None
            finish_reason = "STOP"
        class MockResp:
            candidates = []
        return MockResp()

    provider._client.models.generate_content = mock_generate
    res = provider.chat(messages=[{"role": "user", "content": "test"}])

    assert attempt_count == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 7.5


def test_quota_errors_produce_controlled_status_without_raw_traceback(repo: Path) -> None:
    from agent.llm.base import LLMQuotaExhaustedError

    class QuotaErrorLLM(LLMProvider):
        @property
        def model_name(self) -> str:
            return "mock-quota"
        def chat(self, messages, tools=None, system_prompt=None):
            raise LLMQuotaExhaustedError("RESOURCE_EXHAUSTED: daily quota limit reached")

    agent = InvestigationAgent(repo_path=str(repo), llm_provider=QuotaErrorLLM())
    result = agent.investigate("the checkout crashes")

    assert result["status"] == "ERROR"
    assert "quota exhausted" in result["termination_reason"].lower()
    assert any(t["kind"] == "ERROR" for t in result["trace"])


def test_existing_completion_gate_and_rca_behavior_preserved(repo: Path) -> None:
    # Verify completion gate continues to reject fabricated test claims
    fabricated = good_finish(verification_performed=["Ran the test suite and it passes."])
    llm = scripted(resp(read("src/CustomerService.java")), resp(fabricated))
    state = run_loop(repo, llm)
    assert state.finish_attempts[0]["accepted"] is False
    assert any("cannot execute tests" in e for e in state.finish_attempts[0]["errors"])


def test_readonly_repository_constraint_preserved(repo: Path) -> None:
    initial_files = {p: p.read_bytes() for p in repo.rglob("*.java")}

    router_resp = resp(text=json.dumps({
        "intent": "CODE_INVESTIGATION",
        "reason": "Bug report."
    }))
    mock_llm = scripted(
        router_resp,
        resp(read("src/CustomerService.java", 1, 7)),
        resp(good_finish()),
    )
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Active customers cannot create orders")

    assert result["status"] == "COMPLETED"
    current_files = {p: p.read_bytes() for p in repo.rglob("*.java")}
    assert initial_files == current_files


def test_numeric_query_500_does_not_crash_search_code(repo: Path) -> None:
    from agent.tools.search_code import search_code
    ws = Workspace(repo)
    reg = ToolRegistry(ws)

    # Test direct int query
    res1 = search_code(ws, 500)
    assert "error" not in res1
    assert res1["query"] == "500"
    assert isinstance(res1["matches"], list)

    # Test via tool registry
    res2 = reg.execute("search_code", {"query": 500})
    assert "error" not in res2
    assert res2["query"] == "500"

    # Test via string "500"
    res3 = reg.execute("search_code", {"query": "500"})
    assert "error" not in res3
    assert res3["query"] == "500"

    # Test empty query
    res4 = reg.execute("search_code", {"query": ""})
    assert "error" in res4

    # Test whitespace query
    res5 = reg.execute("search_code", {"query": "   "})
    assert "error" in res5


def test_plausible_endpoint_failure_follows_focused_exploration_strategy(repo: Path) -> None:
    router_resp = resp(text=json.dumps({
        "intent": "CODE_INVESTIGATION",
        "reason": "Endpoint 500 error report."
    }))
    step1_search = resp(call("search_code", query="CustomerService"))
    step2_read = resp(read("src/CustomerService.java", 1, 7))
    step3_finish = resp(good_finish())

    mock_llm = scripted(router_resp, step1_search, step2_read, step3_finish)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["intent"] == "CODE_INVESTIGATION"
    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None
    assert mock_llm.calls == 4


def test_zero_result_searches_cause_strategy_change_rather_than_endless_broad_searching(repo: Path) -> None:
    mock_llm = scripted(
        resp(call("search_code", query="nonexistent1")),
        resp(call("search_code", query="nonexistent2")),
        resp(call("search_code", query="nonexistent3")),
        resp(call("search_code", query="nonexistent4")),
    )
    state = run_loop(repo, mock_llm, max_iterations=10)

    assert state.status == InvestigationStatus.INCOMPLETE
    assert "zero-result" in state.termination_reason.lower()
    assert state.iteration <= 4
    assert state.final_rca is not None
    assert any("Searches that yielded no useful matches" in u for u in state.final_rca["remaining_uncertainty"])


def test_useful_file_evidence_retained_when_budget_exhausted(repo: Path) -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Bug"}))
    read_call = resp(read("src/CustomerService.java", 1, 7))
    mock_llm = scripted(router_resp, read_call, resp(call("search_code", query="something_else")))
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm, request_budget=2)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["status"] == "INCOMPLETE"
    assert result["request_count"] == 2
    assert result["rca"] is not None
    assert len(result["rca"]["evidence"]) > 0
    assert result["rca"]["evidence"][0]["file"] == "src/CustomerService.java"
    assert result["rca"]["evidence"][0]["start_line"] == 1
    assert result["rca"]["evidence"][0]["end_line"] == 7
    assert any("src/CustomerService.java:1-7" in v for v in result["rca"]["verification_performed"])


def test_actual_outbound_requests_including_retries_never_exceed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.llm.base import LLMQuotaExhaustedError
    from agent.llm.gemini import GeminiProvider

    monkeypatch.setattr("time.sleep", lambda s: None)

    class RateLimitError(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"
        message = "Rate limit hit. Retry in 1s."

    provider = GeminiProvider(api_key="mock", max_retries=5)
    provider.request_budget = 3

    attempt_calls = 0
    def mock_generate(*args, **kwargs):
        nonlocal attempt_calls
        attempt_calls += 1
        raise RateLimitError()

    provider._client.models.generate_content = mock_generate

    with pytest.raises(LLMQuotaExhaustedError):
        provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert attempt_calls == 3
    assert provider.request_count == 3


def test_repository_inventory_identifies_spring_controllers_and_services() -> None:
    from agent.repository.inventory import build_repository_inventory
    ws = Workspace("examples/demo-order-service")
    inv = build_repository_inventory(ws, "The customer story endpoint returns 500")

    assert inv.framework == "Spring Boot (Java)"
    assert any("src/main/java" in r for r in inv.source_roots)

    # Controller inspection
    ctrl_names = [c.name for c in inv.controllers]
    assert "OrderController.java" in ctrl_names
    order_ctrl = next(c for c in inv.controllers if c.name == "OrderController.java")
    assert any("POST /api/v1/orders" in r for r in order_ctrl.routes)
    assert any("GET /api/v1/orders/{id}" in r for r in order_ctrl.routes)

    # Services inspection
    svc_names = [s.name for s in inv.services]
    assert "CustomerService.java" in svc_names
    assert "OrderService.java" in svc_names

    # Repositories & Exceptions
    repo_names = [r.name for r in inv.repositories]
    assert "OrderRepository.java" in repo_names
    exc_names = [e.name for e in inv.exceptions]
    assert "CustomerNotEligibleException.java" in exc_names

    # Candidate ranking
    ranked_paths = [r.file_path for r in inv.ranked_candidates]
    assert inv.ranked_candidates[0].file_path.endswith("OrderController.java")
    assert inv.ranked_candidates[0].metadata.role == "controller"
    assert any("CustomerService.java" in p for p in ranked_paths)

    # Formatted prompt summary
    summary = inv.to_prompt_summary()
    assert "Spring Boot (Java)" in summary
    assert "OrderController.java" in summary
    assert "POST /api/v1/orders" in summary


def test_repository_inventory_identifies_angular_components_and_routing(tmp_path: Path) -> None:
    from agent.repository.inventory import build_repository_inventory
    src = tmp_path / "src" / "app"
    src.mkdir(parents=True)
    (tmp_path / "angular.json").write_text("{}", encoding="utf-8")
    (src / "app.routes.ts").write_text("export const routes = [{ path: 'customers', component: CustComp }];", encoding="utf-8")
    (src / "customer.component.ts").write_text("@Component({ selector: 'cust' }) export class CustComp {}", encoding="utf-8")
    (src / "customer.service.ts").write_text("@Injectable({ providedIn: 'root' }) export class CustService {}", encoding="utf-8")

    ws = Workspace(tmp_path)
    inv = build_repository_inventory(ws, "customer routing fails")

    assert inv.framework == "Angular (TypeScript)"
    assert any("app.routes.ts" in c for c in inv.all_source_files)
    assert any("customer.component.ts" in c for c in inv.all_source_files)
    assert any("customer.service.ts" in s.name for s in inv.services)


def test_endpoint_investigation_inspects_candidate_controller_before_broad_searches(repo: Path) -> None:
    received_messages: list[Any] = []
    def capture_prompt(msgs: Any, i: int) -> LLMResponse:
        received_messages.extend(msgs)
        if i == 0:
            return resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Endpoint 500 error"}))
        return resp(read("src/CustomerService.java", 1, 7))

    mock_llm = FnLLM(capture_prompt)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm, request_budget=2)
    res = agent.investigate("The customer story endpoint returns 500")

    # Verify prompt sent to LLM during loop included inventory and prioritized candidates
    assert any(
        "Deterministic Repository Inventory" in m.get("content", "") or "Prioritized Candidate" in m.get("content", "")
        for m in received_messages
    )


def test_controller_to_service_tracing_is_prioritized(repo: Path) -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Defect"}))
    step1_read_ctrl = resp(read("src/OrderService.java", 1, 10))
    step2_read_svc = resp(read("src/CustomerService.java", 1, 7))
    step3_finish = resp(good_finish(
        evidence=[
            {
                "file": "src/OrderService.java", "symbol": "createOrder",
                "start_line": 1, "end_line": 5,
                "explanation": "OrderService calls customerService.validateCustomerEligibility.",
            },
            {
                "file": "src/CustomerService.java", "symbol": "validateCustomerEligibility",
                "start_line": 2, "end_line": 6,
                "explanation": "`customer.getStatus() == CustomerStatus.ACTIVE` is inverted.",
            },
        ],
        verification_performed=["Traced OrderService.createOrder to CustomerService.validateCustomerEligibility."],
    ))

    mock_llm = scripted(router_resp, step1_read_ctrl, step2_read_svc, step3_finish)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Orders cannot be created by customers")

    assert result["status"] == "COMPLETED"
    assert result["rca"] is not None
    # Tracing verified both files
    ev_files = {e["file"] for e in result["rca"]["evidence"]}
    assert "src/OrderService.java" in ev_files
    assert "src/CustomerService.java" in ev_files


def test_missing_endpoints_reported_honestly(repo: Path) -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Missing endpoint"}))
    read_ctrl = resp(read("src/CustomerService.java", 1, 7))
    finish_incomplete = resp(call(
        "finish_investigation",
        status="INCOMPLETE",
        root_cause="Unavailable: the reported customer story endpoint does not exist in any registered controller.",
        confidence=0.0,
        recommended_fix="Unavailable: verify endpoint URL and service deployment routing.",
        remaining_uncertainty=["The endpoint route could not be found in the repository."],
        investigation_summary=["Inspected controllers; customer story endpoint is not defined in this service."],
    ))

    mock_llm = scripted(router_resp, read_ctrl, finish_incomplete)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["status"] == "INCOMPLETE"
    assert result["rca"] is not None
    assert result["rca"]["confidence"] == 0.0
    assert "customer story endpoint does not exist" in result["rca"]["root_cause"]


def test_budget_exhaustion_distinguishes_inspected_and_uninspected_candidates_and_selects_relevant_next_step() -> None:
    ws = Workspace("examples/demo-order-service")
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Endpoint 500 error"}))
    read_order_ctrl = resp(read("src/main/java/com/example/orders/controller/OrderController.java", 1, 30))
    
    # Budget of 2: 1 for router, 1 for read_file, then exhausted before turn 3
    mock_llm = scripted(router_resp, read_order_ctrl, resp(call("search_code", query="anything")))
    agent = InvestigationAgent(repo_path="examples/demo-order-service", llm_provider=mock_llm, request_budget=2)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["status"] == "INCOMPLETE"
    assert result["request_count"] == 2
    rca = result["rca"]
    assert rca is not None
    # Preserves evidence from inspected file
    assert len(rca["evidence"]) > 0
    assert rca["evidence"][0]["file"] == "src/main/java/com/example/orders/controller/OrderController.java"
    
    # Distinguishes inspected and uninspected candidates
    uncertainty_text = "\n".join(rca["remaining_uncertainty"])
    assert "Files inspected" in uncertainty_text
    assert "OrderController.java" in uncertainty_text
    assert "Candidate files not yet inspected" in uncertainty_text
    
    # Prefers next unread controller/service candidate over generic README
    next_step = rca["developer_next_steps"][0]
    assert "Inspect next promising candidate file:" in next_step
    assert "README" not in next_step
    assert any(c in next_step for c in ("CustomerService", "OrderService", "CustomerNotEligibleException", "CustomerRepository"))


def test_outcome_validated_code_defect(repo: Path) -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Active customers rejected"}))
    read_file_resp = resp(read("src/CustomerService.java", 1, 7))
    finish_resp = resp(good_finish(outcome_type="VALIDATED_DEFECT"))

    mock_llm = scripted(router_resp, read_file_resp, finish_resp)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("Active customers cannot create orders")

    assert result["status"] == "COMPLETED"
    assert result["outcome_type"] == InvestigationOutcome.VALIDATED_DEFECT.value
    rca_data = result["rca"]
    assert rca_data is not None
    rca = RCAReport.model_validate(rca_data)
    assert rca.outcome_type == InvestigationOutcome.VALIDATED_DEFECT
    assert rca.has_validated_root_cause is True
    assert rca.confidence == 0.85
    assert rca.status == RCAStatus.COMPLETED


def test_outcome_absent_endpoint_with_adequate_inspection(repo: Path) -> None:
    ctrl_path = repo / "src" / "OrderController.java"
    ctrl_content = (
        "@RestController\n"
        "@RequestMapping(\"/api/v1/orders\")\n"
        "public class OrderController {\n"
        "    @PostMapping\n"
        "    public Order createOrder() { return new Order(); }\n"
        "}\n"
    )
    ctrl_path.write_text(ctrl_content)

    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Endpoint 500"}))
    read_ctrl = resp(read("src/OrderController.java", 1, 6))

    # 1. Overinflated confidence for ENDPOINT_NOT_FOUND is rejected by the completion gate
    ws = Workspace(repo)
    ledger = EvidenceLedger(ws)
    reg = ToolRegistry(ws)
    ledger.record("read_file", reg.execute("read_file", {"file_path": "src/OrderController.java", "start_line": 1, "end_line": 6}))

    overconfident_payload = {
        "status": "INCOMPLETE",
        "outcome_type": "ENDPOINT_NOT_FOUND",
        "confidence": 0.45,
        "root_cause": "No matching route for customer story was found in OrderController.java. The actual source of HTTP 500 remains unverified.",
        "recommended_fix": "Confirm intended route with API specification before implementing any new endpoint.",
    }
    gate_res = validate_finish(overconfident_payload, ledger, "The customer story endpoint returns 500")
    assert not gate_res.ok
    assert any("Confidence (0.45) cannot exceed 0.40 when outcome_type is ENDPOINT_NOT_FOUND" in e for e in gate_res.errors)

    # 2. Valid finish payload with calibrated confidence and honest observed facts
    finish_valid = resp(call(
        "finish_investigation",
        status="INCOMPLETE",
        outcome_type="ENDPOINT_NOT_FOUND",
        confidence=0.30,
        root_cause="No matching route for customer story was found in the inspected controller inventory and source files (OrderController.java maps /api/v1/orders). The actual source of the reported HTTP 500 remains unverified.",
        recommended_fix="Confirm the intended route with the API specification and inspect server logs or gateway routing before implementing any new endpoint.",
        remaining_uncertainty=["The actual source of the reported HTTP 500 remains unverified in runtime logs or traces."],
        developer_next_steps=["Confirm the intended endpoint URL and check gateway/server access logs for the 500 error."],
        investigation_summary=["Inspected OrderController.java mappings under /api/v1/orders; customer-story endpoint is not defined in this service."],
    ))

    mock_llm = scripted(router_resp, read_ctrl, finish_valid)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["status"] == "INCOMPLETE"
    assert result["outcome_type"] == InvestigationOutcome.ENDPOINT_NOT_FOUND.value
    rca = RCAReport.model_validate(result["rca"])
    assert rca.outcome_type == InvestigationOutcome.ENDPOINT_NOT_FOUND
    assert rca.has_validated_root_cause is False
    assert rca.confidence <= 0.40
    assert "No matching route" in rca.root_cause


def test_outcome_absent_endpoint_budget_exhaustion() -> None:
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Endpoint 500 error"}))
    read_order_ctrl = resp(read("src/main/java/com/example/orders/controller/OrderController.java", 1, 30))

    mock_llm = scripted(router_resp, read_order_ctrl, resp(call("search_code", query="customer-story")))
    agent = InvestigationAgent(repo_path="examples/demo-order-service", llm_provider=mock_llm, request_budget=2)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["status"] == "INCOMPLETE"
    assert result["outcome_type"] == InvestigationOutcome.BUDGET_EXHAUSTED.value
    rca = RCAReport.model_validate(result["rca"])
    assert rca.outcome_type == InvestigationOutcome.BUDGET_EXHAUSTED
    assert rca.has_validated_root_cause is False
    assert rca.confidence <= 0.50
    uncertainty_text = "\n".join(rca.remaining_uncertainty)
    assert "budget" in uncertainty_text.lower() and "exhausted" in uncertainty_text.lower()
    assert "Candidate files not yet inspected" in uncertainty_text


def test_outcome_reported_http_500_remains_unverified(repo: Path) -> None:
    ws = Workspace(repo)
    ledger = EvidenceLedger(ws)
    reg = ToolRegistry(ws)
    ledger.record("read_file", reg.execute("read_file", {"file_path": "src/CustomerService.java", "start_line": 1, "end_line": 7}))

    # 1. Unsupported causal claim: asserting missing endpoint caused HTTP 500 without evidence
    unsupported_causal_payload = {
        "status": "INCOMPLETE",
        "outcome_type": "ENDPOINT_NOT_FOUND",
        "confidence": 0.30,
        "root_cause": "The missing customer story endpoint caused HTTP 500 errors when accessed by clients.",
        "recommended_fix": "Confirm the intended route with the API specification.",
    }
    gate_res = validate_finish(unsupported_causal_payload, ledger, "The customer story endpoint returns 500")
    assert not gate_res.ok
    assert any("Do not report 'the missing endpoint caused HTTP 500'" in e for e in gate_res.errors)

    # 2. Completed investigation where runtime failure is unverified
    router_resp = resp(text=json.dumps({"intent": "CODE_INVESTIGATION", "reason": "Endpoint 500"}))
    read_ctrl = resp(read("src/CustomerService.java", 1, 7))
    finish_runtime_unverified = resp(call(
        "finish_investigation",
        status="INCOMPLETE",
        outcome_type="RUNTIME_UNVERIFIED",
        confidence=0.40,
        root_cause="CustomerService.validateCustomerEligibility was inspected, but the reported HTTP 500 failure was not reproduced or traced through a runtime failure path.",
        recommended_fix="Obtain server logs and stack traces to identify the exact runtime exception causing HTTP 500.",
        remaining_uncertainty=["The runtime failure mechanism and error stack trace remain unverified."],
        developer_next_steps=["Examine server application logs for HTTP 500 stack traces."],
        investigation_summary=["Inspected CustomerService.java; runtime failure path unverified."],
    ))

    mock_llm = scripted(router_resp, read_ctrl, finish_runtime_unverified)
    agent = InvestigationAgent(repo_path=str(repo), llm_provider=mock_llm)
    result = agent.investigate("The customer story endpoint returns 500")

    assert result["outcome_type"] == InvestigationOutcome.RUNTIME_UNVERIFIED.value
    rca = RCAReport.model_validate(result["rca"])
    assert rca.outcome_type == InvestigationOutcome.RUNTIME_UNVERIFIED
    assert rca.has_validated_root_cause is False
    assert rca.confidence <= 0.75


def test_backward_compatibility_saved_run_json_and_cli() -> None:
    # 1. Load existing saved run JSON without outcome_type
    saved_file = Path("runs/scenario1_result.json")
    assert saved_file.exists(), "runs/scenario1_result.json must exist"
    data = json.loads(saved_file.read_text(encoding="utf-8"))
    rca_data = data["rca"]

    # Verify that the saved JSON does not contain outcome_type
    assert "outcome_type" not in rca_data

    # Parsing into RCAReport should seamlessly backfill outcome_type
    rca = RCAReport.model_validate(rca_data)
    assert rca.outcome_type == InvestigationOutcome.VALIDATED_DEFECT
    assert rca.has_validated_root_cause is True

    # CLI report renderer should render all 9 section titles
    rendered = render_report(rca)
    for title in SECTION_TITLES:
        assert f"  {title}\n" in rendered

    # 2. Legacy INCOMPLETE payload without outcome_type backfills INSUFFICIENT_EVIDENCE
    legacy_incomplete = {
        "status": "INCOMPLETE",
        "problem": "Issue without enough evidence",
        "root_cause": "Unavailable: root cause not found",
        "confidence": 0.20,
        "evidence": [],
        "recommended_fix": "Unavailable: no fix",
        "developer_next_steps": ["Inspect more files"],
        "verification_performed": ["Searched code"],
        "remaining_uncertainty": ["Evidence is insufficient"],
        "investigation_summary": ["Searched code but found nothing"],
    }
    rca_inc = RCAReport.model_validate(legacy_incomplete)
    assert rca_inc.outcome_type == InvestigationOutcome.INSUFFICIENT_EVIDENCE
    assert rca_inc.has_validated_root_cause is False
    rendered_inc = render_report(rca_inc)
    for title in SECTION_TITLES:
        assert f"  {title}\n" in rendered_inc
    assert "INCOMPLETE (INSUFFICIENT_EVIDENCE)" in rendered_inc






