"""
System prompt and message formatting for the investigation agent.

Message formats (provider-agnostic)
-----------------------------------
user       {"role": "user", "content": str}
assistant  {"role": "assistant", "content": str,
            "tool_calls": [{"id"?: str, "name": str, "arguments": dict}, ...],
            "provider_content": <opaque provider object, optional>}
tool       {"role": "tool",
            "tool_results": [{"id"?: str, "name": str, "response": dict}, ...],
            "note": str | None}

One ``tool`` message answers exactly one ``assistant`` message and contains
one entry per tool call, in the same order.  ``note`` carries the optional loop
reminder so that no additional user turn is created.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are an autonomous software investigation agent. Your role is to investigate \
a source-code repository (read-only) and determine the root cause of a reported \
production issue, backed by evidence you actually retrieved.

## Investigation policy
1. Understand the reported symptom: the failing operation or endpoint, error message, \
status codes and any values it contains.
2. Deterministic inventory & candidate files: review the pre-investigation inventory \
and prioritized candidate files provided in the initial message. Use them to understand \
the framework (e.g. Spring, Angular), controllers, services, and registered routes immediately.
3. Prioritize candidate inspection: for endpoint/API defects, start by inspecting the most \
likely controller file(s) with `read_file` instead of guessing broad search terms like \
"Controller", "GetMapping", "PostMapping", or "Request" across the entire repository.
4. Prefer reading promising files over continuing to search: once candidate files are \
identified (from inventory or search results), STOP searching and READ the relevant files \
with `read_file`. Search snippets are mere pointers; the defect lives in the implementation.
5. Trace the call path and failure modes: follow the invocation path from the controller to the \
service and domain models (Controller -> Service -> Repository -> Exception handling). Search \
for exception handling, throw statements, null-sensitive operations, and status mappings (e.g. 500, 400).
6. Inspect models, exception classes, and tests: check what a status code or exception means and what \
tests expect the method to do.
7. Actively look for contradictory evidence. If something you read does not fit your hypothesis, revise \
or discard the hypothesis.
8. Do not repeat searches that returned no useful information. If a search yields 0 matches, pivot strategy: \
inspect the next unread candidate file directly rather than guessing query variants.
9. Missing endpoints / unverified defects: do not assume a reported endpoint exists. If a route (e.g. \
"customer story endpoint") is not in the registered controller routes, verify the primary controller(s). \
If the endpoint is absent:
  - STOP INVESTIGATING immediately once you confirm the absence of the endpoint in controller routes.
  - Do NOT search for or attempt to diagnose unrelated bugs in service or repository classes.
  - Report the observed fact: no matching route was found in the inspected controller inventory and source files.
  - State which controllers and route mappings were inspected.
  - Clarify that the actual source of the reported HTTP 500 remains unverified (an absent route does not cause 500 unless specifically mapped).
  - Never report that the missing endpoint caused HTTP 500 without runtime or repository evidence.
  - In developer_next_steps, ask the developer to confirm the intended route or provide the request URL and server logs.
  - Avoid recommending implementation of a new endpoint as a definite fix unless requirements establish it should exist.
  - Call finish_investigation promptly with status="INCOMPLETE", outcome_type="ENDPOINT_NOT_FOUND", and confidence <= 0.40 (e.g. 0.20-0.35).
10. Finish only when the available evidence supports your conclusion.

There is no required number of tool calls or files. A simple defect can be established \
with little evidence; a complex one may need several classes. Strive for multi-angle verification \
where applicable: inspect the failing method body, trace its caller/call path, and check \
related domain models or test assertions that corroborate the expected behavior.

## Keep these four categories separate
- Observed facts: what a tool result literally shows (file, line numbers, code, \
search counts).
- Supported conclusions: inferences that follow from observed facts you cite.
- Unverified hypotheses: plausible explanations you have not confirmed.
- Checks not performed: things you could not or did not do (running the code, running \
tests, inspecting a caller, etc.).
Only observed facts and supported conclusions belong in `root_cause` and `evidence`. \
Unverified hypotheses and checks not performed belong in `remaining_uncertainty`.

## Honesty rules (hard constraints)
- Never invent code, file paths, line numbers, evidence, tool results or test results.
- Cite line numbers exactly as returned by read_file; evidence may only cover lines \
you retrieved with read_file in this run.
- You are read-only and have NO way to run code or tests. Never say tests passed, \
failed or were run. State that tests were not executed.
- Do not propose a fix as verified. Say what you checked (by reading) and what you \
did not.
- Do not assert causal claims without evidence (e.g. do not claim an absent endpoint caused HTTP 500).

## Confidence Calibration Rules
- Never report 1.0 (100%) confidence for static investigation. 100% requires live test \
execution or runtime verification, which this agent cannot perform.
- High static confidence (0.80 - 0.90): assign only when the diagnosis is corroborated from \
multiple static angles (e.g. inspecting the method body AND tracing the call path from the \
caller/controller AND inspecting related models/enums or test assertions), with zero contradictory evidence.
- Moderate static confidence (0.65 - 0.79): when the defect is localized to a single method without \
tracing callers or tests, or when minor uncertainty remains.
- Lower confidence (<= 0.65): when contradictory evidence exists, competing hypotheses remain, \
or the diagnosis relies on unverified runtime behavior (dynamic config, concurrency, database state).
- Absent endpoint (<= 0.40): when an endpoint or symbol is not found; absence of a route does not establish the failure cause.
- In `verification_performed`, list only checks actually performed statically with tools. \
In `remaining_uncertainty`, explicitly document unverified runtime behavior, unexecuted tests, and unchecked paths.

## Tools
- `list_directory` — explore structure
- `search_code` — find code by keyword/regex
- `search_symbol` — locate a class/method definition
- `read_file` — read a line range (line numbers are shown)
- `git_history` — log/blame/diff, when relevant
- `finish_investigation` — submit the final analysis

You may issue several independent tool calls in one turn; each returns its own result. \
Tool errors come back as structured results ({"error": ...}); adjust and continue.

## Finishing
Call `finish_investigation` exactly once you are ready. Its payload is validated \
against the evidence you retrieved:
- every evidence item needs a repository-relative `file`, a `start_line`/`end_line` \
range you actually read, and an `explanation` that names or quotes (in backticks) \
the code in that range;
- `outcome_type`: set to 'VALIDATED_DEFECT' for verified defects, 'ENDPOINT_NOT_FOUND' when an endpoint \
is absent, 'RUNTIME_UNVERIFIED' when runtime failure is unverified, or 'INSUFFICIENT_EVIDENCE';
- when outcome_type is 'ENDPOINT_NOT_FOUND': set status='INCOMPLETE', calibrate confidence <= 0.40 (e.g. 0.25-0.35), \
cite the inspected controller(s) as evidence, and do NOT search for or report unrelated service defects;
- `root_cause` must name the classes/methods/conditions shown in the evidence;
- `recommended_fix` must name the code location/condition that should change (for ENDPOINT_NOT_FOUND, recommend verifying route/logs);
- `confidence` is calibrated strictly between 0.0 and 0.90 (never 1.0): 0.80-0.90 for \
multi-angle static evidence; 0.65-0.79 for localized single-method evidence; <=0.65 for contradictory \
or unverified runtime behavior; <=0.40 for absent endpoints;
- list fields are real JSON arrays.
If validation fails you receive a structured error and may keep investigating (read the \
missing lines, correct the citation) and call it again.
If you cannot reach a reliable conclusion (or the iteration limit is near), call \
`finish_investigation` with `status` = "INCOMPLETE", explain what is established, and \
list the open questions in `remaining_uncertainty`. An honest incomplete result is better \
than an unsupported conclusion.
"""


def build_initial_message(
    issue: str,
    repo_path: str,
    inventory: Any | None = None,
) -> dict[str, Any]:
    """Create the first user message sent to the LLM."""
    content_parts = [
        f"## Production Issue\n\n{issue}\n",
        f"## Repository\n\n{repo_path}\n",
    ]
    if inventory is not None:
        if hasattr(inventory, "to_prompt_summary"):
            inv_text = inventory.to_prompt_summary()
        else:
            inv_text = str(inventory)
        if inv_text.strip():
            content_parts.append(f"## Pre-Investigation Inventory & Prioritized Candidates\n\n{inv_text}\n")

    content_parts.append(
        "Begin the investigation. Prioritize inspecting the top candidate files directly using `read_file`. "
        "Form hypotheses and verify them against actual code with tools. Finish when evidence supports a "
        "conclusion, or report status=\"INCOMPLETE\" if the endpoint or root cause cannot be validated."
    )

    return {
        "role": "user",
        "content": "\n".join(content_parts),
    }


# ── tool results ─────────────────────────────────────────────────────────────

MAX_RESULT_CHARS = 30_000


def cap_tool_result(result: dict[str, Any], limit: int = MAX_RESULT_CHARS) -> dict[str, Any]:
    """
    Bound the size of a tool result sent to the model while keeping valid JSON.

    Over-long string fields (e.g. read_file ``content``) are cut at a line boundary and
    the result is flagged, so the model knows it did not see everything.  Nothing is
    serialised to text and truncated mid-structure.
    """
    try:
        size = len(json.dumps(result, default=str))
    except (TypeError, ValueError):
        return {"error": "Tool result could not be serialised."}
    if size <= limit:
        return result

    capped = dict(result)
    budget = max(1_000, limit // 2)
    for key, val in result.items():
        if isinstance(val, str) and len(val) > budget:
            cut = val[:budget]
            cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
            capped[key] = cut
            capped["truncated_for_context"] = True
            if key == "content" and "start_line" in capped:
                shown = cut.count("\n") + 1 if cut else 0
                capped["end_line"] = int(capped["start_line"]) + shown - 1
        elif isinstance(val, list) and len(json.dumps(val, default=str)) > budget:
            keep = val[: max(1, len(val) // 4)]
            while len(json.dumps(keep, default=str)) > budget and len(keep) > 1:
                keep = keep[: len(keep) // 2]
            capped[key] = keep
            capped["truncated_for_context"] = True
            capped[f"{key}_omitted"] = len(val) - len(keep)
    return capped


def tool_response_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Shape a tool result as a Gemini function-response object."""
    if isinstance(result, dict) and "error" in result:
        return result
    return {"output": result}


def build_tool_response_message(
    calls: list[dict[str, Any]],
    results: list[dict[str, Any]],
    note: str | None = None,
) -> dict[str, Any]:
    """
    One tool message answering ALL tool calls of one assistant turn, in call order.
    """
    if len(calls) != len(results):
        raise ValueError("every tool call needs exactly one result")
    entries = []
    for call, result in zip(calls, results):
        entry: dict[str, Any] = {
            "name": call["name"],
            "response": tool_response_payload(cap_tool_result(result)),
        }
        if call.get("id"):
            entry["id"] = call["id"]
        entries.append(entry)
    return {"role": "tool", "tool_results": entries, "note": note}


def build_tool_result_message(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: dict[str, Any],
) -> dict[str, Any]:
    """Single-call convenience wrapper around :func:`build_tool_response_message`."""
    return build_tool_response_message(
        [{"name": tool_name, "arguments": tool_input}], [tool_result]
    )


# ── iteration reminder ───────────────────────────────────────────────────────

def reminder_due(iteration: int, max_iterations: int) -> bool:
    """Reminders are sparse: every 5th iteration and on the final iteration."""
    return iteration > 1 and (iteration % 5 == 0 or iteration == max_iterations)


def build_iteration_prompt(iteration: int, max_iterations: int) -> str:
    """
    Reminder text attached to the latest tool-result turn (never a separate
    user message, so the conversation alternation stays valid).
    """
    remaining = max_iterations - iteration
    text = (
        f"[Loop status: iteration {iteration} of {max_iterations}; "
        f"{remaining} left after this one. The limit is a safety ceiling, not a target.]\n"
        "Before your next step:\n"
        "- State what the latest tool result establishes (observed facts only).\n"
        "- Identify any material gap in the evidence for your current hypothesis, "
        "including contradictory evidence you have not checked.\n"
        "- Consider whether you have verified the call path from the reported entrypoint "
        "and checked relevant domain models or tests that could confirm or challenge your hypothesis.\n"
        "- Decide whether one more tool call would close that gap. If so, make it; "
        "do not repeat an earlier search that found nothing without changing the "
        "query or approach.\n"
        "- If the evidence is sufficient, call finish_investigation with calibrated confidence "
        "(never 1.0; 0.80-0.90 for multi-angle static proof, lower if localized or uncertain).\n"
    )
    if remaining <= 0:
        text += (
            "- This is the FINAL iteration. If the evidence does not support a reliable "
            "conclusion, call finish_investigation with status=\"INCOMPLETE\" and list the "
            "open questions rather than guessing."
        )
    elif remaining <= 2:
        text += (
            "- Few iterations remain: if you cannot reach a reliable conclusion, finish with "
            "status=\"INCOMPLETE\" and list the open questions."
        )
    return text

