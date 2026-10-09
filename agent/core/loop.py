"""
Investigation agent loop.

Drives the iterative LLM <-> tool <-> LLM cycle until one of:
  * ``finish_investigation`` is accepted by the completion gate  -> COMPLETED / INCOMPLETE
  * the iteration ceiling is reached                              -> INVESTIGATION_LIMIT_REACHED
  * the LLM call fails                                            -> ERROR

Conversation invariants (see ``agent.core.prompts`` for message shapes):
  * exactly one assistant message per LLM response (text and tool calls together);
  * exactly one tool message per assistant message that had tool calls, holding one
    result per call, in order (the ``finish_investigation`` call included);
  * the iteration reminder is attached to the latest tool message, never added as an
    extra user turn;
  * a text-only response is followed by a single nudge user message.
"""

from __future__ import annotations

import logging
from typing import Any

from agent import config
from agent.core.completion import (
    TEST_EXECUTION_NOTE,
    UNVALIDATED_FIX_NOTE,
    EvidenceLedger,
    GateOutcome,
    validate_finish,
)
from agent.core.prompts import (
    SYSTEM_PROMPT,
    build_initial_message,
    build_iteration_prompt,
    build_tool_response_message,
    cap_tool_result,
    reminder_due,
)
from agent.core.scope import validate_issue_scope
from agent.llm.base import (
    LLMProvider,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
)
from agent.models.investigation import InvestigationState, InvestigationStatus
from agent.models.rca import EvidenceItem, RCAReport, RCAStatus, InvestigationOutcome
from agent.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

FINISH = "finish_investigation"
MAX_REJECTED_FINISH_ATTEMPTS = 3

_NUDGE = (
    "Your last reply contained no tool call. Continue by calling one of the available "
    "tools, or call finish_investigation if you are ready to conclude (use "
    "status=\"INCOMPLETE\" if the evidence is insufficient)."
)


def _normalize_evidence(ev: Any) -> list[tuple[str, int, int]]:
    out = []
    if isinstance(ev, list):
        for item in ev:
            if isinstance(item, dict):
                f = str(item.get("file", "")).strip().replace("\\", "/").lower()
                try:
                    s, e = int(item.get("start_line", 0)), int(item.get("end_line", 0))
                except (ValueError, TypeError):
                    s, e = 0, 0
                out.append((f, s, e))
    return sorted(out)


def _is_repeated_payload(prev: dict[str, Any] | None, curr: dict[str, Any]) -> bool:
    """Return True if curr finish payload is identical or unchanged compared to prev."""
    if not prev:
        return False
    prev_rc = str(prev.get("root_cause", "")).strip()
    curr_rc = str(curr.get("root_cause", "")).strip()
    prev_status = str(prev.get("status", "")).strip().upper()
    curr_status = str(curr.get("status", "")).strip().upper()
    prev_fix = str(prev.get("recommended_fix", "")).strip()
    curr_fix = str(curr.get("recommended_fix", "")).strip()
    prev_ev = _normalize_evidence(prev.get("evidence"))
    curr_ev = _normalize_evidence(curr.get("evidence"))

    if prev_rc == curr_rc and prev_ev == curr_ev:
        return True
    if prev_rc == curr_rc and prev_status == curr_status and prev_fix == curr_fix:
        return True
    return False


class AgentLoop:
    """
    Orchestrates the investigation.

    Kept separate from the agent entrypoint so it can be tested in isolation.
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        max_iterations: int | None = None,
        request_budget: int | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._max_iter = max_iterations if max_iterations is not None else config.MAX_ITERATIONS
        self._request_budget = request_budget if request_budget is not None else config.LLM_REQUEST_BUDGET
        self._progress_callback = progress_callback  # callable(step_label, detail)

    # ── public ───────────────────────────────────────────────────────────────

    def run(self, state: InvestigationState) -> InvestigationState:
        """Execute the investigation loop and populate *state* with results."""
        # 1. Pre-investigation scope guard
        if state.status in (InvestigationStatus.OUT_OF_SCOPE, InvestigationStatus.GENERAL_QUESTION):
            return state

        scope_res = validate_issue_scope(state.issue)
        if not scope_res.is_in_scope:
            state.status = InvestigationStatus.OUT_OF_SCOPE
            state.termination_reason = f"Input out of scope: {scope_res.reason}"
            flabel = f"[!!] Issue OUT_OF_SCOPE: {scope_res.reason}"
            self._emit(flabel, scope_res.reason)
            state.add_trace("SCOPE_GUARD", flabel, scope_res.reason)
            return state

        ledger = EvidenceLedger(self._registry.workspace)
        state.llm_config = self._llm.describe()
        state.request_budget = self._request_budget

        # 2. Build deterministic repository inventory and prioritized candidate rankings
        from agent.repository.inventory import build_repository_inventory
        inventory = build_repository_inventory(self._registry.workspace, state.issue)
        state.candidate_files = [c.file_path for c in inventory.ranked_candidates]
        candidate_files_found: list[str] = list(state.candidate_files)

        self._emit("[01] USER ISSUE", state.issue)
        state.add_trace("USER_ISSUE", state.issue)
        state.messages.append(build_initial_message(state.issue, state.repository_path, inventory=inventory))

        tool_definitions = self._registry.tool_definitions()
        step = 2
        last_rejected: GateOutcome | None = None
        executed_searches: dict[tuple[str, str], Any] = {}
        repeated_search_count = 0
        zero_match_searches: list[str] = []
        consecutive_unproductive_searches = 0

        for iteration in range(1, self._max_iter + 1):
            state.iteration = iteration

            # Check request budget before issuing another LLM request
            current_requests = getattr(self._llm, "request_count", state.request_count)
            state.request_count = current_requests
            if current_requests >= self._request_budget:
                flabel = f"[!!] LLM request budget exhausted ({current_requests}/{self._request_budget}). Stopping investigation."
                logger.warning(flabel)
                self._emit(flabel, "")
                state.status = InvestigationStatus.INCOMPLETE
                state.termination_reason = f"LLM request budget exhausted ({current_requests}/{self._request_budget})."
                state.add_trace("BUDGET_EXHAUSTED", flabel, state.termination_reason)
                self._set_report(state, self._fallback_report(
                    state,
                    RCAStatus.INCOMPLETE,
                    f"Investigation stopped: LLM request budget of {self._request_budget} calls exhausted.",
                    last_rejected,
                    ledger,
                    zero_match_searches=zero_match_searches,
                    candidate_files=candidate_files_found,
                ))
                return state

            if reminder_due(iteration, self._max_iter):
                _attach_note(state.messages, build_iteration_prompt(iteration, self._max_iter))

            logger.debug("Iteration %d — calling LLM …", iteration)

            try:
                response = self._llm.chat(
                    messages=state.messages,
                    tools=tool_definitions,
                    system_prompt=SYSTEM_PROMPT,
                )
                state.request_count = getattr(self._llm, "request_count", state.request_count + 1)
            except LLMQuotaExhaustedError as exc:
                state.request_count = getattr(self._llm, "request_count", state.request_count)
                logger.error("LLM quota exhausted: %s", exc)
                state.status = InvestigationStatus.ERROR
                state.termination_reason = f"LLM quota exhausted: {exc}"
                state.add_trace("ERROR", f"Quota exhausted: {exc}")
                self._set_report(state, self._fallback_report(
                    state, RCAStatus.ERROR, f"Gemini API quota exhausted (HTTP 429 RESOURCE_EXHAUSTED): {exc}",
                    last_rejected, ledger, zero_match_searches, candidate_files_found))
                return state
            except LLMUnavailableError as exc:
                state.request_count = getattr(self._llm, "request_count", state.request_count)
                logger.error("LLM model unavailable: %s", exc)
                state.status = InvestigationStatus.ERROR
                state.termination_reason = f"LLM model unavailable: {exc}"
                state.add_trace("ERROR", f"Model unavailable: {exc}")
                self._set_report(state, self._fallback_report(
                    state, RCAStatus.ERROR, f"LLM model unavailable: {exc}",
                    last_rejected, ledger, zero_match_searches, candidate_files_found))
                return state
            except Exception as exc:
                state.request_count = getattr(self._llm, "request_count", state.request_count)
                logger.error("LLM call failed: %s", exc)
                state.status = InvestigationStatus.ERROR
                state.termination_reason = f"LLM call failed: {exc}"
                state.add_trace("ERROR", f"LLM call failed: {exc}")
                self._set_report(state, self._fallback_report(
                    state, RCAStatus.ERROR, f"LLM call failed: {exc}",
                    last_rejected, ledger, zero_match_searches, candidate_files_found))
                return state

            calls = [dict(c) for c in response.tool_calls]

            # ── text-only turn ───────────────────────────────────────────
            if not calls:
                label = f"[{step:02d}] LLM → (no tool call, text response)"
                self._emit(label, response.text[:200] if response.text else "")
                state.add_trace("LLM_TEXT", label, response.text)
                step += 1
                state.messages.append({
                    "role": "assistant",
                    "content": response.text or "(no output)",
                    "provider_content": getattr(response, "raw_candidate", None),
                })
                state.messages.append({"role": "user", "content": _NUDGE})
                continue

            # ── one assistant message for the whole response ─────────────
            state.messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": calls,
                "provider_content": getattr(response, "raw_candidate", None),
            })

            results: list[dict[str, Any] | None] = [None] * len(calls)
            finish_indexes: list[int] = []

            # Pass 1: execute every ordinary tool call.
            for idx, call in enumerate(calls):
                name = call["name"]
                args = call.get("arguments") or {}
                label = f"[{step:02d}] LLM → {name}({_fmt_args(args)})"
                self._emit(label, args)
                state.add_trace("LLM_ACTION", label, args)
                step += 1

                if name == FINISH:
                    finish_indexes.append(idx)
                    continue

                if name == "read_file":
                    consecutive_unproductive_searches = 0

                if name in ("search_code", "search_symbol"):
                    import json
                    search_key = (name, json.dumps(args, sort_keys=True))
                    if search_key in executed_searches:
                        repeated_search_count += 1
                        prev_res = executed_searches[search_key]
                        result = {
                            "warning": (
                                f"Repeated search: '{name}' with identical arguments was already executed earlier. "
                                "No new evidence will be discovered by repeating it. "
                                "Inspect discovered files using read_file or conclude your investigation."
                            ),
                            "previous_matches": prev_res.get("matches", prev_res.get("symbols", [])),
                        }
                        results[idx] = result
                        rlabel = f"[{step:02d}] TOOL → {name} [BOUNDED REPEATED SEARCH]"
                        self._emit(rlabel, result)
                        state.add_trace("TOOL_RESULT", rlabel, result)
                        step += 1

                        if repeated_search_count >= 3:
                            flabel = "[!!] Repeated unproductive searches ceiling reached (3). Circuit breaker triggered."
                            self._emit(flabel, "")
                            state.status = InvestigationStatus.INCOMPLETE
                            state.termination_reason = "Stopped: repeated unproductive searches bounded."
                            state.add_trace("SEARCH_CIRCUIT_BREAKER", flabel, state.termination_reason)
                            self._set_report(state, self._fallback_report(
                                state,
                                RCAStatus.INCOMPLETE,
                                "Investigation stopped: repeated unproductive searches without new evidence.",
                                last_rejected,
                                ledger,
                                zero_match_searches=zero_match_searches,
                                candidate_files=candidate_files_found,
                            ))
                            return state
                        continue

                result = cap_tool_result(self._registry.execute(name, args))
                if name in ("search_code", "search_symbol"):
                    import json
                    search_key = (name, json.dumps(args, sort_keys=True))
                    executed_searches[search_key] = result

                    # Progress & match accounting
                    matches_list = result.get("matches") or result.get("symbols") or []
                    match_count = result.get("total_matches", len(matches_list)) if isinstance(matches_list, list) else 0

                    if match_count == 0:
                        consecutive_unproductive_searches += 1
                        q_term = str(args.get("query", args.get("name", ""))).strip()
                        if q_term and q_term not in zero_match_searches:
                            zero_match_searches.append(q_term)

                        if consecutive_unproductive_searches >= 2:
                            if candidate_files_found:
                                result["strategy_guidance"] = (
                                    f"Consecutive searches returned 0 matches. Strategy recommendation: "
                                    f"pivot to reading discovered candidate files using read_file: {candidate_files_found[:3]}."
                                )
                            else:
                                result["strategy_guidance"] = (
                                    "Consecutive searches returned 0 matches. Strategy recommendation: "
                                    "use list_directory to inspect package structure or locate controllers."
                                )
                    else:
                        consecutive_unproductive_searches = 0
                        for m in matches_list:
                            if isinstance(m, dict) and m.get("file"):
                                fpath_found = m["file"]
                                if fpath_found not in candidate_files_found:
                                    candidate_files_found.append(fpath_found)

                    if consecutive_unproductive_searches >= 4 and len(ledger.inspected_ranges()) == 0:
                        flabel = "[!!] Unproductive search sequence ceiling reached (4 zero-match searches). Stopping investigation."
                        self._emit(flabel, "")
                        state.status = InvestigationStatus.INCOMPLETE
                        state.termination_reason = "Investigation stopped: zero-result searches caused strategy termination rather than endless searching."
                        state.add_trace("SEARCH_STRATEGY_BOUND", flabel, state.termination_reason)
                        self._set_report(state, self._fallback_report(
                            state,
                            RCAStatus.INCOMPLETE,
                            "Investigation stopped: consecutive zero-result searches yielded no candidate files.",
                            last_rejected,
                            ledger,
                            zero_match_searches=zero_match_searches,
                            candidate_files=candidate_files_found,
                        ))
                        return state

                ledger.record(name, result)
                state.record_tool_call(name, args, result)
                results[idx] = result

                rlabel = f"[{step:02d}] TOOL → {_summarise_result(name, result)}"
                self._emit(rlabel, result)
                state.add_trace("TOOL_RESULT", rlabel, result)
                step += 1

            # Pass 2: evaluate finish_investigation (only the first one per response).
            if finish_indexes:
                first = finish_indexes[0]
                for extra in finish_indexes[1:]:
                    results[extra] = {
                        "error": "Only one finish_investigation call is evaluated per response; "
                                 "this duplicate call was ignored."
                    }
                payload = calls[first].get("arguments") or {}

                # Check if identical to the previous rejected payload
                rejected_attempts = [a for a in state.finish_attempts if not a.get("accepted")]
                is_repeated = False
                if rejected_attempts:
                    is_repeated = _is_repeated_payload(rejected_attempts[-1].get("payload"), payload)

                outcome = validate_finish(
                    payload, ledger, state.issue, tool_summary=_tool_summary(state)
                )
                state.finish_attempts.append({
                    "iteration": iteration,
                    "accepted": outcome.ok,
                    "requested_status": outcome.requested_status,
                    "errors": outcome.errors,
                    "payload": payload,
                    "repeated": is_repeated,
                })

                if outcome.ok and outcome.report is not None:
                    state.finish_accepted_count += 1
                    flabel = f"[{step:02d}] LLM → finish_investigation (accepted)"
                    self._emit(flabel, "")
                    state.add_trace("FINISH", flabel, {
                        "status": outcome.report.status.value,
                        "outcome_type": outcome.report.outcome_type.value,
                    })
                    step += 1
                    state.status = (
                        InvestigationStatus.COMPLETED
                        if outcome.report.status == RCAStatus.COMPLETED
                        else InvestigationStatus.INCOMPLETE
                    )
                    state.outcome_type = outcome.report.outcome_type.value
                    state.termination_reason = "Completed investigation accepted by completion gate."
                    self._set_report(state, outcome.report)
                    rlabel = f"[{step:02d}] RCA GENERATED ({outcome.report.status.value} - {outcome.report.outcome_type.value})"
                    self._emit(rlabel, outcome.report.root_cause)
                    state.add_trace("RCA", rlabel, state.final_rca)
                    return state  # never iterate again after acceptance

                last_rejected = outcome
                rejected_count = len([a for a in state.finish_attempts if not a.get("accepted")])

                # Circuit breaker 1: Repeated identical or unchanged finish payload
                if is_repeated:
                    flabel = (
                        f"[{step:02d}] finish_investigation REJECTED (repeated unchanged payload). "
                        "Circuit breaker triggered."
                    )
                    self._emit(flabel, outcome.errors)
                    state.add_trace("FINISH_REJECTED", flabel, outcome.errors)
                    state.status = InvestigationStatus.INCOMPLETE
                    state.termination_reason = "Circuit breaker triggered: repeated unchanged finish_investigation payload."
                    self._set_report(state, self._fallback_report(
                        state,
                        RCAStatus.INCOMPLETE,
                        "Investigation stopped: repeated identical finish_investigation payload without fixing validation errors.",
                        outcome,
                        ledger,
                        zero_match_searches=zero_match_searches,
                        candidate_files=candidate_files_found,
                    ))
                    return state

                # Circuit breaker 2: Maximum rejected attempts reached (3)
                if rejected_count >= MAX_REJECTED_FINISH_ATTEMPTS:
                    flabel = (
                        f"[{step:02d}] finish_investigation REJECTED ({len(outcome.errors)} problem(s)). "
                        f"Rejection ceiling reached ({rejected_count}/{MAX_REJECTED_FINISH_ATTEMPTS})."
                    )
                    self._emit(flabel, outcome.errors)
                    state.add_trace("FINISH_REJECTED", flabel, outcome.errors)
                    state.status = InvestigationStatus.INCOMPLETE
                    state.termination_reason = f"Circuit breaker triggered: maximum finish_investigation rejection limit ({MAX_REJECTED_FINISH_ATTEMPTS} attempts) reached."
                    self._set_report(state, self._fallback_report(
                        state,
                        RCAStatus.INCOMPLETE,
                        f"Investigation stopped: maximum finish_investigation rejection limit ({MAX_REJECTED_FINISH_ATTEMPTS} attempts) reached.",
                        outcome,
                        ledger,
                        zero_match_searches=zero_match_searches,
                        candidate_files=candidate_files_found,
                    ))
                    return state

                flabel = f"[{step:02d}] finish_investigation REJECTED ({len(outcome.errors)} problem(s), attempt {rejected_count}/{MAX_REJECTED_FINISH_ATTEMPTS})"
                self._emit(flabel, outcome.errors)
                state.add_trace("FINISH_REJECTED", flabel, outcome.errors)
                step += 1
                results[first] = {
                    "error": (
                        f"finish_investigation was rejected by the completion gate "
                        f"(attempt {rejected_count} of {MAX_REJECTED_FINISH_ATTEMPTS}). "
                        f"Fix the following errors or keep investigating with other tools: "
                        + "; ".join(outcome.errors)
                    ),
                    "validation_errors": outcome.errors,
                    "inspected_ranges": ledger.inspected_ranges(),
                }

            state.messages.append(build_tool_response_message(calls, results))  # type: ignore[arg-type]

        # ── iteration ceiling reached ────────────────────────────────────────
        logger.warning("MAX_ITERATIONS (%d) reached without an accepted finish_investigation", self._max_iter)
        state.status = InvestigationStatus.INVESTIGATION_LIMIT_REACHED
        state.termination_reason = f"Iteration ceiling ({self._max_iter}) reached."
        self._set_report(state, self._fallback_report(
            state,
            RCAStatus.INCOMPLETE,
            f"The iteration limit ({self._max_iter}) was reached before the evidence supported a "
            f"validated conclusion.",
            last_rejected,
            ledger,
            zero_match_searches=zero_match_searches,
            candidate_files=candidate_files_found,
        ))
        self._emit(f"[!!] INVESTIGATION_LIMIT_REACHED after {self._max_iter} iterations", "")
        return state

    # ── internal ─────────────────────────────────────────────────────────────

    def _emit(self, label: str, detail: Any) -> None:
        if self._progress_callback:
            self._progress_callback(label, detail)

    @staticmethod
    def _set_report(state: InvestigationState, report: RCAReport) -> None:
        state.final_rca = report.to_dict()
        state.outcome_type = report.outcome_type.value if hasattr(report.outcome_type, "value") else str(report.outcome_type)

    @staticmethod
    def _fallback_report(
        state: InvestigationState,
        status: RCAStatus,
        reason: str,
        last_rejected: GateOutcome | None,
        ledger: EvidenceLedger,
        zero_match_searches: list[str] | None = None,
        candidate_files: list[str] | None = None,
        outcome_type: InvestigationOutcome | None = None,
    ) -> RCAReport:
        """Canonical INCOMPLETE/ERROR report. Never claims a validated conclusion."""
        uncertainty = [reason]
        evidence: list[EvidenceItem] = []
        if last_rejected is not None:
            evidence = list(last_rejected.valid_evidence)
            if last_rejected.candidate_root_cause:
                uncertainty.append(
                    "UNVERIFIED hypothesis from the agent (rejected by the evidence check): "
                    + last_rejected.candidate_root_cause
                )
            uncertainty.extend(f"Evidence check: {e}" for e in last_rejected.errors)
        
        inspected_ranges = ledger.inspected_ranges()
        inspected_files = sorted(list(inspected_ranges.keys()))
        cand_list = candidate_files or []
        uninspected_candidates = [f for f in cand_list if f not in set(inspected_files)]

        # Preserve useful evidence from read_file ranges if last_rejected has none
        if not evidence and inspected_ranges:
            for fpath, ranges in inspected_ranges.items():
                for start_l, end_l in ranges:
                    evidence.append(EvidenceItem(
                        file=fpath,
                        start_line=start_l,
                        end_line=end_l,
                        explanation=f"Inspected lines {start_l}-{end_l} in {fpath} during investigation.",
                    ))

        # Distinguish inspected files, candidate files, and files not yet inspected
        uncertainty.append(
            f"Files inspected ({len(inspected_files)}): {', '.join(inspected_files) if inspected_files else 'None'}"
        )
        uncertainty.append(
            f"Candidate files identified ({len(cand_list)}): {', '.join(cand_list[:6]) if cand_list else 'None'}"
        )
        uncertainty.append(
            f"Candidate files not yet inspected ({len(uninspected_candidates)}): {', '.join(uninspected_candidates[:6]) if uninspected_candidates else 'None'}"
        )

        if zero_match_searches:
            uncertainty.append(
                f"Searches that yielded no useful matches: {', '.join(repr(s) for s in zero_match_searches[:5])}"
            )
        uncertainty.append(
            f"Why root cause could not be validated: {reason}. Insufficient static evidence to establish the complete failure path."
        )
        uncertainty.append(UNVALIDATED_FIX_NOTE)

        inspected = [
            f"Inspected {p}:{a}-{b} (read_file)"
            for p, rs in inspected_ranges.items() for a, b in rs
        ]

        verification = list(inspected)
        if inspected_files:
            verification.insert(0, f"Inspected files: {', '.join(inspected_files)}")
        if zero_match_searches:
            verification.append(f"Evaluated search queries: {len(zero_match_searches)} query/queries returned no matches.")
        verification.append(TEST_EXECUTION_NOTE)

        summary = _tool_summary(state) or ["No tool calls were completed."]

        # Next most useful file or investigation step
        next_steps = []
        if uninspected_candidates:
            next_steps.append(f"Inspect next promising candidate file: {uninspected_candidates[0]}")
            if len(uninspected_candidates) > 1:
                next_steps.append(f"Inspect related candidate component: {uninspected_candidates[1]}")
        else:
            next_steps.append("Verify whether the reported route is handled by an external service, API gateway, or configuration.")
        next_steps.append("Review the investigation trace in the saved run JSON.")
        next_steps.append("Re-run with a higher request budget or more specific issue details.")

        # Determine effective outcome type
        if outcome_type is not None:
            eff_outcome = outcome_type
        elif "budget" in reason.lower() or state.request_count >= state.request_budget:
            eff_outcome = InvestigationOutcome.BUDGET_EXHAUSTED
        elif zero_match_searches or "endpoint" in state.issue.lower():
            eff_outcome = InvestigationOutcome.ENDPOINT_NOT_FOUND
        elif status == RCAStatus.ERROR:
            eff_outcome = InvestigationOutcome.INSUFFICIENT_EVIDENCE
        else:
            eff_outcome = InvestigationOutcome.INSUFFICIENT_EVIDENCE

        return RCAReport(
            status=status,
            outcome_type=eff_outcome,
            problem=state.issue,
            root_cause=(
                "Unavailable: no validated root cause was established. See Remaining Uncertainty."
            ),
            confidence=0.0,
            evidence=evidence,
            recommended_fix="Unavailable: manual review required; no fix is supported by validated evidence.",
            developer_next_steps=next_steps,
            verification_performed=verification,
            remaining_uncertainty=uncertainty,
            investigation_summary=summary[:50],
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _attach_note(messages: list[dict[str, Any]], note: str) -> None:
    """Attach the reminder to the latest turn instead of creating a new user message."""
    if not messages:
        return
    last = messages[-1]
    if last.get("role") == "tool":
        last["note"] = (last.get("note") + "\n\n" if last.get("note") else "") + note
    elif last.get("role") == "user":
        last["content"] = f"{last.get('content', '')}\n\n{note}"
    # an assistant message is never last when the loop calls the LLM


def _tool_summary(state: InvestigationState) -> list[str]:
    return [
        f"Iteration {tc.iteration}: {tc.tool_name}({_fmt_args(tc.tool_input)})"
        for tc in state.tool_calls
    ]


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool arguments for the trace display."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _summarise_result(tool_name: str, result: dict[str, Any]) -> str:
    """One-line summary of a tool result for the trace."""
    if "error" in result:
        return f"ERROR: {result['error']}"
    if tool_name == "list_directory":
        dirs = len(result.get("directories", []))
        files = len(result.get("files", []))
        return f"{dirs} dirs, {files} files in '{result.get('path', '.')}'"
    if tool_name == "search_code":
        n = result.get("total_matches", 0)
        return f"{n} match(es) for '{result.get('query', '')}'"
    if tool_name == "read_file":
        return (
            f"lines {result.get('start_line')}–{result.get('end_line')} "
            f"of '{result.get('file')}' ({result.get('total_lines')} total)"
        )
    if tool_name == "search_symbol":
        n = result.get("total_results", 0)
        return f"{n} symbol(s) matching '{result.get('symbol', '')}'"
    if tool_name == "git_history":
        op = result.get("operation", "")
        if op == "log":
            n = len(result.get("commits", []))
            return f"{n} commit(s) for '{result.get('file', 'repo')}'"
        return f"git {op} result"
    return "result returned"
