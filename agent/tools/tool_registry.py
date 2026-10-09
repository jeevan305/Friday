"""
Tool registry — central registry of all agent tools.

Each tool exposes:
  - A TOOL_DEFINITION dict (provider-agnostic schema)
  - A callable that takes (workspace, **kwargs) → dict

The finish_investigation pseudo-tool is handled separately by the agent loop.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.repository.workspace import Workspace
from agent.tools import (
    list_directory,
    read_file,
    search_code,
    search_symbol,
    git_history,
)

# ── finish_investigation ──────────────────────────────────────────────────────
# This is not a real tool — it is a signal for the agent loop to stop.
# We still declare it so the LLM knows it can call it.

_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

FINISH_TOOL_DEFINITION: dict[str, Any] = {
    "name": "finish_investigation",
    "description": (
        "Submit the final analysis. The payload is validated against the source code you "
        "retrieved with read_file in this run; invalid evidence is rejected with a structured "
        "error and you may continue investigating. Use status='INCOMPLETE' when the evidence "
        "does not support a reliable conclusion. Never claim tests were run: you cannot run them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["COMPLETED", "INCOMPLETE"],
                "description": (
                    "'COMPLETED' when a code defect is validated by cited evidence; "
                    "'INCOMPLETE' when evidence is incomplete or when outcome_type is ENDPOINT_NOT_FOUND / RUNTIME_UNVERIFIED."
                ),
            },
            "outcome_type": {
                "type": "string",
                "enum": [
                    "VALIDATED_DEFECT",
                    "ENDPOINT_NOT_FOUND",
                    "BUDGET_EXHAUSTED",
                    "INSUFFICIENT_EVIDENCE",
                    "RUNTIME_UNVERIFIED",
                ],
                "description": (
                    "Semantic outcome category: "
                    "'VALIDATED_DEFECT' for verified code defect with evidence; "
                    "'ENDPOINT_NOT_FOUND' when the reported endpoint or symbol does not exist in the codebase (confidence must be <= 0.40); "
                    "'RUNTIME_UNVERIFIED' when code was inspected but original runtime failure mechanism is unverified; "
                    "'INSUFFICIENT_EVIDENCE' when evidence is incomplete."
                ),
            },
            "root_cause": {
                "type": "string",
                "description": (
                    "Supported conclusion only, naming the classes/methods/conditions shown in the "
                    "evidence. For INCOMPLETE: what is established so far."
                ),
            },
            "evidence": {
                "type": "array",
                "description": (
                    "Evidence items. Each must cite a repository-relative file and a line range "
                    "you actually read with read_file, and explain the code in that range."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Repository-relative path."},
                        "symbol": {"type": "string", "description": "Class/method the lines belong to."},
                        "start_line": {"type": "integer", "description": "First line (1-indexed)."},
                        "end_line": {"type": "integer", "description": "Last line (inclusive)."},
                        "explanation": {
                            "type": "string",
                            "description": "What these lines show; name or quote the code in backticks.",
                        },
                    },
                    "required": ["file", "start_line", "end_line", "explanation"],
                },
            },
            "recommended_fix": {
                "type": "string",
                "description": "The recommended fix or action. For VALIDATED_DEFECT: name the file/method/condition that should change. For ENDPOINT_NOT_FOUND: recommend verifying the route/logs rather than creating new code.",
            },
            "developer_next_steps": {
                **_STRING_ARRAY,
                "description": "Follow-up actions for a developer (e.g. add a regression test).",
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Confidence score from 0.0 to 0.90 for static code analysis (never 1.0, which requires runtime execution). "
                    "Use 0.80-0.90 for multi-angle static verification (call path + method + models/tests); "
                    "0.65-0.79 for localized single-method evidence; <=0.65 if contradictory evidence or unverified runtime behavior exists. "
                    "CRITICAL: when outcome_type is ENDPOINT_NOT_FOUND, confidence MUST be <= 0.40 (e.g. 0.20-0.35)."
                ),
            },
            "verification_performed": {
                **_STRING_ARRAY,
                "description": (
                    "Checks you actually performed with the tools (e.g. 'read the full method', "
                    "'inspected CustomerStatus enum'). Do not list tests as run."
                ),
            },
            "remaining_uncertainty": {
                **_STRING_ARRAY,
                "description": "Unverified hypotheses and checks not performed.",
            },
            "investigation_summary": {
                **_STRING_ARRAY,
                "description": "Step-by-step summary of the investigation path.",
            },
        },
        "required": [
            "status",
            "root_cause",
            "evidence",
            "recommended_fix",
            "developer_next_steps",
            "confidence",
            "verification_performed",
            "remaining_uncertainty",
            "investigation_summary",
        ],
    },
}



# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Holds all registered tools and dispatches calls."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}
        self._register_defaults()

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def _register_defaults(self) -> None:
        self.register(
            list_directory.TOOL_DEFINITION,
            lambda **kw: list_directory.list_directory(self._workspace, **kw),
        )
        self.register(
            read_file.TOOL_DEFINITION,
            lambda **kw: read_file.read_file(self._workspace, **kw),
        )
        self.register(
            search_code.TOOL_DEFINITION,
            lambda **kw: search_code.search_code(self._workspace, **kw),
        )
        self.register(
            search_symbol.TOOL_DEFINITION,
            lambda **kw: search_symbol.search_symbol(self._workspace, **kw),
        )
        self.register(
            git_history.TOOL_DEFINITION,
            lambda **kw: git_history.git_history(self._workspace, **kw),
        )

    def register(
        self,
        definition: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self._tools[definition["name"]] = (definition, handler)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions including finish_investigation."""
        defs = [defn for defn, _ in self._tools.values()]
        defs.append(FINISH_TOOL_DEFINITION)
        return defs

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a named tool with the given arguments.

        Returns a dict result. Never raises — errors are returned in the dict.
        """
        if name == "finish_investigation":
            # Signal; handled by the loop — return as-is
            return {"__finish__": True, **arguments}

        if name not in self._tools:
            return {"error": f"Unknown tool: '{name}'"}

        _, handler = self._tools[name]

        # Coerce string arguments where the tool expects int/bool
        clean_args = _coerce_args(arguments)

        try:
            result = handler(**clean_args)
        except TypeError as exc:
            return {"error": f"Invalid arguments for tool '{name}': {exc}"}
        except Exception as exc:
            return {"error": f"Tool '{name}' raised an unexpected error: {exc}"}

        return result if isinstance(result, dict) else {"result": result}

    def __contains__(self, name: str) -> bool:
        return name in self._tools or name == "finish_investigation"


_INT_FIELDS = {"start_line", "end_line", "max_commits"}
_BOOL_FIELDS = {"is_regex"}
_STR_FIELDS = {"query", "path", "file_pattern", "name", "file_path", "operation", "target"}


def _coerce_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Coerce arguments based on expected field types.
    The LLM may pass strings for integer/boolean fields,
    or integers/booleans for string fields (e.g. query=500).
    """
    coerced: dict[str, Any] = {}
    for k, v in arguments.items():
        if k in _INT_FIELDS:
            if isinstance(v, str) and (v.isdigit() or (v.startswith("-") and v[1:].isdigit())):
                try:
                    coerced[k] = int(v)
                    continue
                except ValueError:
                    pass
        elif k in _BOOL_FIELDS:
            if isinstance(v, str):
                if v.lower() in ("true", "1", "yes"):
                    coerced[k] = True
                    continue
                elif v.lower() in ("false", "0", "no"):
                    coerced[k] = False
                    continue
        elif k in _STR_FIELDS or isinstance(v, (int, float, bool)):
            if isinstance(v, (int, float, bool)):
                coerced[k] = str(v)
                continue
        coerced[k] = v
    return coerced
