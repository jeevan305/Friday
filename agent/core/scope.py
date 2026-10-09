"""
Pre-investigation scope guard.

Adapter module that delegates to agent.core.router.route_intent
to avoid duplicated classification logic while preserving backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.core.router import IntentResult, route_intent
from agent.llm.base import LLMProvider


@dataclass
class ScopeResult:
    is_in_scope: bool
    reason: str | None = None
    category: str | None = None


def validate_issue_scope(issue: str, llm: LLMProvider | None = None) -> ScopeResult:
    """
    Validate whether the issue is a software defect requiring repository investigation.
    Delegates to the central router.
    """
    intent_res, _ = route_intent(issue, llm=llm)
    return ScopeResult(
        is_in_scope=(intent_res.intent == "CODE_INVESTIGATION"),
        reason=intent_res.reason,
        category=intent_res.intent,
    )
