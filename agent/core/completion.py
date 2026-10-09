"""
Evidence ledger and completion gate.

The ledger records the source lines that tools actually returned during the
current run.  The gate validates a ``finish_investigation`` payload against
that ledger before the run may be marked COMPLETED.

What the gate checks (deterministically, no extra LLM calls):

* every evidence file is a repository-relative path inside the repository
  and exists;
* every evidence line range is valid for the real file;
* every cited line was actually returned by ``read_file`` in this run;
* the explanation refers to code that is present in the cited lines and does
  not quote code that is absent;
* the root-cause statement refers to identifiers present in the evidence;
* the recommended fix refers to code locations/identifiers from the evidence
  or the root cause;
* no claim of passing tests (the agent cannot execute tests).

The "supports / addresses" checks are lexical heuristics: they catch claims
that are unrelated to the inspected code, but they cannot prove semantic
correctness.  That limitation is documented in the README / final report.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.models.rca import EvidenceItem, RCAReport, RCAStatus, InvestigationOutcome
from agent.repository.workspace import PathTraversalError, Workspace

_LINE_RE = re.compile(r"^\s*(\d+) \| ?(.*)$")

TEST_EXECUTION_NOTE = (
    "Automated test execution: NOT performed. The agent is read-only and has no "
    "test runner, so no test results are claimed."
)
UNVALIDATED_FIX_NOTE = (
    "The recommended fix was not applied and has not been validated by running code or tests."
)


# ── ledger ───────────────────────────────────────────────────────────────────

@dataclass
class _FileSource:
    total_lines: int = 0
    lines: dict[int, str] = field(default_factory=dict)


class EvidenceLedger:
    """Source lines actually returned to the model by ``read_file``."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._files: dict[str, _FileSource] = {}

    def record(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record source from a tool result (only successful read_file results count)."""
        if tool_name != "read_file" or not isinstance(result, dict) or "error" in result:
            return
        raw_file = result.get("file")
        content = result.get("content")
        if not isinstance(raw_file, str) or not isinstance(content, str):
            return
        path = raw_file.replace("\\", "/")
        src = self._files.setdefault(path, _FileSource())
        try:
            src.total_lines = max(src.total_lines, int(result.get("total_lines", 0)))
        except (TypeError, ValueError):
            pass
        for raw in content.splitlines():
            m = _LINE_RE.match(raw)
            if m:
                src.lines[int(m.group(1))] = m.group(2)

    def normalize_path(self, raw: str) -> tuple[str | None, str | None]:
        """Return (repo-relative posix path, None) or (None, error message)."""
        if not isinstance(raw, str) or not raw.strip():
            return None, "file path is empty"
        try:
            resolved = self._workspace.resolve(raw.strip())
        except PathTraversalError:
            return None, f"'{raw}' is outside the repository"
        if not resolved.is_file():
            return None, f"'{raw}' does not exist in the repository"
        return self._workspace.relative(resolved).replace("\\", "/"), None

    def total_lines(self, path: str) -> int | None:
        src = self._files.get(path)
        if src and src.total_lines:
            return src.total_lines
        try:
            text = self._workspace.resolve(path).read_text(encoding="utf-8", errors="replace")
        except (OSError, PathTraversalError):
            return None
        return len(text.splitlines())

    def missing_lines(self, path: str, start: int, end: int) -> list[int]:
        src = self._files.get(path)
        have = src.lines if src else {}
        return [n for n in range(start, end + 1) if n not in have]

    def text(self, path: str, start: int, end: int) -> str:
        src = self._files.get(path)
        have = src.lines if src else {}
        return "\n".join(have[n] for n in range(start, end + 1) if n in have)

    def file_text(self, path: str) -> str:
        src = self._files.get(path)
        return "\n".join(src.lines[n] for n in sorted(src.lines)) if src else ""

    def inspected_ranges(self) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {}
        for path, src in self._files.items():
            nums = sorted(src.lines)
            ranges: list[list[int]] = []
            for n in nums:
                if ranges and n == ranges[-1][1] + 1:
                    ranges[-1][1] = n
                else:
                    ranges.append([n, n])
            out[path] = ranges
        return out


# ── payload coercion ─────────────────────────────────────────────────────────

def coerce_list(value: Any) -> Any:
    """
    Return a real list.  Models occasionally send arrays as JSON text or as a
    Python-repr string ("['a', 'b']"); decode those instead of keeping a string.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            break
        return [value]
    return [value]


_ENUM_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def _str_list(value: Any) -> list[str]:
    items = (_ENUM_PREFIX.sub("", str(v)).strip() for v in coerce_list(value))
    return [s for s in items if s]


# ── token helpers ────────────────────────────────────────────────────────────

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _code_tokens(text: str) -> set[str]:
    """Identifier-like tokens: camelCase/PascalCase, snake_case, or ALLCAPS (len>=4)."""
    tokens = set()
    for t in _IDENT_RE.findall(text):
        if (
            re.search(r"[a-z][A-Z]", t)
            or "_" in t
            or (t.isupper() and len(t) >= 4)
        ):
            tokens.add(t)
    return tokens


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ── outcome ──────────────────────────────────────────────────────────────────

@dataclass
class GateOutcome:
    ok: bool
    errors: list[str] = field(default_factory=list)
    report: RCAReport | None = None
    valid_evidence: list[EvidenceItem] = field(default_factory=list)
    requested_status: str = "COMPLETED"
    candidate_root_cause: str = ""
    outcome_type: InvestigationOutcome = InvestigationOutcome.VALIDATED_DEFECT


_CLAIMS_TESTS_PASS = re.compile(
    r"\b(tests?|test suite|unit tests?|integration tests?|build|ci)\b[^.;\n]*"
    r"\b(pass(?:ed|es|ing)?|succeed(?:ed|s)?|green|ran successfully|run successfully)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(not|no|never|n't|unable|cannot|without|could not|couldn't)\b", re.IGNORECASE)


def _claims_passing_tests(text: str) -> bool:
    for sentence in re.split(r"(?<=[.;])\s+|\n", text):
        if _CLAIMS_TESTS_PASS.search(sentence) and not _NEGATION.search(sentence):
            return True
    return False


# ── confidence calibration ceilings ──────────────────────────────────────────
STATIC_MAX_CONFIDENCE = 0.90
LOCALIZED_STATIC_CEILING = 0.85
UNVERIFIED_RUNTIME_CEILING = 0.75
CONTRADICTORY_EVIDENCE_CEILING = 0.65

_CONTRADICTORY_RE = re.compile(
    r"\b(contradict(?:ion|ory|s)?|conflict(?:ing|s)?|inconsistent|competing hypothes(?:is|es)|alternative explanation)\b",
    re.IGNORECASE,
)
_RUNTIME_UNVERIFIED_RE = re.compile(
    r"\b(runtime (?:behavior|state|environment|dispatch)|unverified runtime|concurrency|race condition|database state|dynamic configuration|environment variable)\b",
    re.IGNORECASE,
)


# ── gate ─────────────────────────────────────────────────────────────────────

def validate_finish(
    payload: dict[str, Any],
    ledger: EvidenceLedger,
    issue: str,
    tool_summary: list[str] | None = None,
) -> GateOutcome:
    """Validate a finish_investigation payload against the evidence ledger."""
    errors: list[str] = []
    requested = str(payload.get("status", "COMPLETED") or "COMPLETED").strip().upper()
    if requested not in ("COMPLETED", "INCOMPLETE"):
        errors.append("status must be 'COMPLETED' or 'INCOMPLETE'.")
        requested = "COMPLETED"
    incomplete = requested == "INCOMPLETE"

    root_cause = str(payload.get("root_cause", "") or "").strip()
    fix = str(payload.get("recommended_fix", "") or "").strip()

    # ── evidence items ───────────────────────────────────────────────────
    valid_items: list[EvidenceItem] = []
    evidence_code: list[str] = []
    evidence_idents: set[str] = set()
    raw_items = coerce_list(payload.get("evidence"))

    for idx, raw in enumerate(raw_items, 1):
        tag = f"evidence[{idx}]"
        if not isinstance(raw, dict):
            errors.append(f"{tag}: must be an object with file, symbol, start_line, end_line, explanation.")
            continue
        item_errors: list[str] = []
        path, perr = ledger.normalize_path(str(raw.get("file", "")))
        if perr:
            item_errors.append(f"{tag}: {perr}.")
        try:
            start, end = int(raw.get("start_line")), int(raw.get("end_line"))
        except (TypeError, ValueError):
            item_errors.append(f"{tag}: start_line and end_line must be integers.")
            start = end = None  # type: ignore[assignment]
        explanation = str(raw.get("explanation", "") or "").strip()
        if not explanation:
            item_errors.append(f"{tag}: explanation is empty.")

        code = ""
        if path and start is not None and end is not None and not item_errors:
            if start < 1 or end < start:
                item_errors.append(f"{tag}: invalid line range {start}-{end}.")
            else:
                total = ledger.total_lines(path)
                if total is not None and end > total:
                    item_errors.append(
                        f"{tag}: lines {start}-{end} exceed the file length ({total} lines) of {path}."
                    )
                else:
                    missing = ledger.missing_lines(path, start, end)
                    if missing:
                        item_errors.append(
                            f"{tag}: lines {_fmt_missing(missing)} of {path} were never returned by a tool "
                            f"in this run. Call read_file on that range first. "
                            f"Lines inspected so far for this file: {ledger.inspected_ranges().get(path, [])}."
                        )
                    else:
                        code = ledger.text(path, start, end)

        if not item_errors and path:
            code_ids = _code_tokens(code)
            symbol = str(raw.get("symbol", "") or "").strip()
            if symbol:
                sym = re.split(r"[.#:]", symbol.split("(")[0])[-1]
                if sym and sym not in ledger.file_text(path):
                    item_errors.append(
                        f"{tag}: symbol '{symbol}' does not appear in the inspected source of {path}."
                    )
            expl_ids = _code_tokens(explanation)
            if not (expl_ids & code_ids):
                item_errors.append(
                    f"{tag}: the explanation does not reference any identifier that appears in the cited "
                    f"lines. Quote or name the code (e.g. in backticks) it is explaining."
                )
            squashed_code = _squash(code)
            for snippet in _BACKTICK_RE.findall(explanation):
                snip_ids = _code_tokens(snippet)
                if snip_ids and not (snip_ids & code_ids) and _squash(snippet) not in squashed_code:
                    item_errors.append(
                        f"{tag}: the explanation quotes `{snippet}`, which does not appear in "
                        f"{path}:{start}-{end}."
                    )

        if item_errors:
            errors.extend(item_errors)
            continue
        item = EvidenceItem(
            file=path, symbol=str(raw.get("symbol", "") or "").strip(),
            start_line=start, end_line=end, explanation=explanation,
        )
        valid_items.append(item)
        evidence_code.append(code)
        evidence_idents |= _code_tokens(code) | _code_tokens(item.symbol)
        evidence_idents |= _code_tokens(item.file.rsplit("/", 1)[-1].split(".")[0])

    # ── determine outcome_type early ─────────────────────────────────────
    raw_outcome = payload.get("outcome_type")
    outcome_type: InvestigationOutcome | None = None
    if raw_outcome:
        try:
            outcome_type = InvestigationOutcome(str(raw_outcome).strip().upper())
        except ValueError:
            errors.append(
                f"outcome_type '{raw_outcome}' is invalid. Allowed values: "
                f"{', '.join(o.value for o in InvestigationOutcome)}."
            )

    is_absent_endpoint = bool(re.search(
        r"\b(missing|absent|not found|not defined|no matching|does not exist)\b[^.;\n]*\b(endpoint|route|mapping|controller|symbol)\b",
        root_cause + " " + issue,
        re.IGNORECASE,
    ))
    is_runtime_unverified = bool(_RUNTIME_UNVERIFIED_RE.search(
        root_cause + " " + " ".join(_str_list(payload.get("remaining_uncertainty")))
    ))

    if outcome_type is None:
        if is_absent_endpoint:
            outcome_type = InvestigationOutcome.ENDPOINT_NOT_FOUND
        elif is_runtime_unverified:
            outcome_type = InvestigationOutcome.RUNTIME_UNVERIFIED
        elif incomplete:
            outcome_type = InvestigationOutcome.INSUFFICIENT_EVIDENCE
        else:
            outcome_type = InvestigationOutcome.VALIDATED_DEFECT

    # ── claim <-> evidence consistency (skipped for explicit INCOMPLETE) ─
    if not incomplete:
        if not valid_items:
            errors.append(
                "No valid evidence. A completed investigation needs at least one evidence item "
                "citing lines you actually retrieved with read_file."
            )
        else:
            rc_ids = _code_tokens(root_cause)
            if not (rc_ids & evidence_idents):
                errors.append(
                    "root_cause is not supported by the cited evidence: it does not mention any "
                    "class, method, field or constant that appears in the cited code."
                )
            if outcome_type == InvestigationOutcome.VALIDATED_DEFECT:
                fix_ids = _code_tokens(fix)
                if not (fix_ids & (evidence_idents | rc_ids)):
                    errors.append(
                        "recommended_fix does not address the stated cause: it must name the "
                        "file/method/condition from the evidence that should change."
                    )

    # ── honesty about verification ───────────────────────────────────────
    verification = _str_list(payload.get("verification_performed"))
    summary = _str_list(payload.get("investigation_summary"))
    for label, entries in (("verification_performed", verification), ("investigation_summary", summary)):
        for e in entries:
            if _claims_passing_tests(e):
                errors.append(
                    f"{label} claims a test/build result ('{e[:80]}'). The agent cannot execute tests; "
                    f"remove the claim and list only checks performed with the available tools."
                )

    # ── confidence calibration ───────────────────────────────────────────
    raw_conf = payload.get("confidence")
    parsed_conf: float | None = None
    if raw_conf is None:
        errors.append("confidence is required.")
    else:
        try:
            parsed_conf = float(raw_conf)
        except (ValueError, TypeError):
            errors.append("confidence must be a number between 0.0 and 1.0.")

        if parsed_conf is not None:
            if parsed_conf < 0.0 or parsed_conf > 1.0:
                errors.append("confidence must be between 0.0 and 1.0.")
            elif incomplete:
                if parsed_conf > 0.50:
                    errors.append(
                        f"Confidence ({parsed_conf:.2f}) is too high for an INCOMPLETE investigation. "
                        "Incomplete investigations must have confidence <= 0.50 reflecting unresolved gaps."
                    )
            else:
                # COMPLETED investigation
                if parsed_conf >= 1.0:
                    errors.append(
                        "Confidence cannot be 1.0 (100%): this investigation is based on static code analysis "
                        "without runtime execution or test verification. Distinguish well-supported static diagnosis "
                        "(maximum 0.85-0.90) from execution verification, and calibrate confidence accordingly."
                    )
                else:
                    uncertainty_items = _str_list(payload.get("remaining_uncertainty"))
                    has_contradiction = any(_CONTRADICTORY_RE.search(u) for u in uncertainty_items) or bool(_CONTRADICTORY_RE.search(root_cause))
                    has_runtime_unverified = any(_RUNTIME_UNVERIFIED_RE.search(u) for u in uncertainty_items)

                    inspected_paths = [p.lower() for p in ledger.inspected_ranges().keys()]
                    files_read = len(inspected_paths)
                    has_multi_angle = files_read >= 2 and any(
                        any(k in p for k in ("controller", "order", "model", "enum", "status", "test", "repository"))
                        for p in inspected_paths
                    )

                    ceiling = STATIC_MAX_CONFIDENCE
                    reasons = []

                    if has_contradiction:
                        ceiling = min(ceiling, CONTRADICTORY_EVIDENCE_CEILING)
                        reasons.append(f"contradictory evidence or competing hypotheses are present (ceiling {CONTRADICTORY_EVIDENCE_CEILING:.2f})")
                    if has_runtime_unverified:
                        ceiling = min(ceiling, UNVERIFIED_RUNTIME_CEILING)
                        reasons.append(f"unverified runtime behavior is noted (ceiling {UNVERIFIED_RUNTIME_CEILING:.2f})")
                    if not has_multi_angle and files_read <= 1:
                        ceiling = min(ceiling, LOCALIZED_STATIC_CEILING)
                        reasons.append(f"investigation is localized to a single file without cross-component inspection (ceiling {LOCALIZED_STATIC_CEILING:.2f})")

                    if parsed_conf > ceiling + 0.001:
                        errors.append(
                            f"Confidence ({parsed_conf:.2f}) is overconfident for this investigation. "
                            f"{'; '.join(reasons)}. Maximum allowable confidence is {ceiling:.2f}. "
                            "Please calibrate confidence to reflect the actual evidence quality and coverage."
                        )

    # ── outcome-specific checks & causal claim validation ─────────────────

    # Check causal claim: missing endpoint caused HTTP 500
    _CAUSAL_MISSING_500 = re.compile(
        r"\b(missing|absent|non-?existent|not found)\b[^.;\n]*\b(caused|causes|leads? to|results? in|returns?|triggers?)\b[^.;\n]*\b500\b",
        re.IGNORECASE,
    )
    _500_BECAUSE_MISSING = re.compile(
        r"\b500\b[^.;\n]*\b(because|due to|caused by|result of)\b[^.;\n]*\b(missing|absent|non-?existent|not found)\b",
        re.IGNORECASE,
    )
    if _CAUSAL_MISSING_500.search(root_cause) or _500_BECAUSE_MISSING.search(root_cause):
        has_500_in_evidence = any("500" in c or "INTERNAL_SERVER_ERROR" in c for c in evidence_code)
        if not has_500_in_evidence:
            errors.append(
                "Do not report 'the missing endpoint caused HTTP 500' without repository or runtime evidence "
                "supporting that causal claim. Report the absent route as an observed fact, and clarify that "
                "the actual source of the reported HTTP 500 remains unverified."
            )

    if outcome_type == InvestigationOutcome.ENDPOINT_NOT_FOUND:
        if parsed_conf is not None and parsed_conf > 0.40:
            errors.append(
                f"Confidence ({parsed_conf:.2f}) cannot exceed 0.40 when outcome_type is ENDPOINT_NOT_FOUND: "
                "absence of a search match does not verify the runtime failure mechanism or establish the root cause of the reported error."
            )
        _DEFINITE_NEW_ENDPOINT = re.compile(
            r"\b(implement|create|add)\b[^.;\n]*\b(the|new)?\s*(missing|absent)?\s*(endpoint|route|controller)\b",
            re.IGNORECASE,
        )
        if _DEFINITE_NEW_ENDPOINT.search(fix) and not re.search(r"\b(confirm|verify|check|specification|spec|log|gateway|intended)\b", fix, re.IGNORECASE):
            errors.append(
                "recommended_fix should avoid declaring implementation of a new endpoint as a definite fix without requirements. "
                "Recommend confirming the intended route with the API specification and checking server logs or gateway routing."
            )

    # ── build the canonical report ───────────────────────────────────────
    candidate = root_cause
    if errors:
        return GateOutcome(False, errors, None, valid_items, requested, candidate, outcome_type or InvestigationOutcome.INSUFFICIENT_EVIDENCE)

    uncertainty = _str_list(payload.get("remaining_uncertainty"))
    if UNVALIDATED_FIX_NOTE not in uncertainty:
        uncertainty.append(UNVALIDATED_FIX_NOTE)
    verification = _dedupe(
        verification
        + [f"Inspected {p}:{a}-{b} (read_file)" for p, rs in ledger.inspected_ranges().items() for a, b in rs]
        + [TEST_EXECUTION_NOTE]
    )
    if not summary:
        summary = list(tool_summary or []) or ["Summary unavailable: no investigation summary was provided."]

    steps = _str_list(payload.get("developer_next_steps")) or [
        "Unavailable: no developer next steps were provided."
    ]

    data = {
        "status": RCAStatus.INCOMPLETE if incomplete else RCAStatus.COMPLETED,
        "outcome_type": outcome_type,
        "problem": issue,
        "root_cause": root_cause or "Unavailable: no root cause was established.",
        "confidence": parsed_conf if parsed_conf is not None else 0.0,
        "evidence": [e.to_dict() for e in valid_items],
        "recommended_fix": fix or "Unavailable: no fix could be recommended from the evidence.",
        "developer_next_steps": steps,
        "verification_performed": verification,
        "remaining_uncertainty": uncertainty,
        "investigation_summary": summary,
    }
    try:
        report = RCAReport.model_validate(data)
    except ValidationError as exc:
        msgs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return GateOutcome(False, [f"Invalid report: {m}" for m in msgs], None, valid_items, requested, candidate, outcome_type)
    return GateOutcome(True, [], report, valid_items, requested, candidate, outcome_type)


def _fmt_missing(nums: list[int]) -> str:
    return f"{nums[0]}-{nums[-1]}" if len(nums) > 3 else ",".join(map(str, nums))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
