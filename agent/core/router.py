"""
Intent routing layer for the AI Code Investigation Agent.

Routes incoming queries into:
1. CODE_INVESTIGATION: enters the multi-turn repository investigation loop.
2. GENERAL_QUESTION: answered directly by the LLM without repository tools.
3. OUT_OF_SCOPE: rejected with an explanation without starting repository investigation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agent.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class IntentResult(BaseModel):
    """Structured classification result from the intent router."""
    intent: str = Field(description="CODE_INVESTIGATION | GENERAL_QUESTION | OUT_OF_SCOPE")
    reason: str = Field(description="Short rationale for the routing decision")
    direct_answer: str | None = Field(
        default=None,
        description="Direct answer when intent is GENERAL_QUESTION, else None",
    )


# ── Fast-path deterministic patterns ──────────────────────────────────────────

_GREETING_PATTERN = re.compile(
    r"^\s*("
    r"hi|hello|hey|howdy|sup|greetings|hola|yo|"
    r"good\s+(morning|afternoon|evening|day|night)|"
    r"how\s+are\s+you(\s+doing)?|"
    r"what'?s\s+up|"
    r"nice\s+to\s+meet\s+you"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

_IDENTITY_PATTERN = re.compile(
    r"^\s*("
    r"what('?s|\s+is)\s+your\s+name|"
    r"who\s+are\s+you|"
    r"what\s+are\s+you|"
    r"who\s+(made|created|built|developed)\s+you|"
    r"what\s+can\s+you\s+do|"
    r"tell\s+me\s+about\s+yourself|"
    r"introduce\s+yourself|"
    r"are\s+you\s+(an?\s+)?(ai|bot|robot|llm|chatgpt|gemini|claude|human)|"
    r"what\s+is\s+your\s+purpose"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

_NON_TECHNICAL_OUT_OF_SCOPE = re.compile(
    r"^\s*("
    r"what('?s|\s+is)\s+the\s+capital\s+of\b.*|"
    r"who\s+is\s+the\s+president\b.*|"
    r"how\s+far\s+is\s+the\s+(moon|sun|earth)\b.*|"
    r"tell\s+me\s+a\s+(joke|story|fact)\b.*|"
    r"write\s+(me\s+)?a\s+(poem|song|essay|story)\b.*|"
    r"what('?s|\s+is)\s+the\s+weather\b.*|"
    r"how\s+(do|can)\s+i\s+(cook|bake|make)\b.*|"
    r"who\s+won\s+the\b.*|"
    r"sing\s+a\s+song\b.*|"
    r"recommend\s+(me\s+)?a\s+(movie|book|song|restaurant)\b.*|"
    r"what\s+is\s+the\s+meaning\s+of\s+life\b.*|"
    r"what\s+time\s+is\s+it\b.*"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

_DEFECT_KEYWORDS = re.compile(
    r"\b("
    r"error|exception|fault|bug|defect|crash|crashes|crashed|"
    r"fail|fails|failed|failure|failures|"
    r"broken|invalid|incorrect|unexpected|wrong|mismatch|"
    r"unable|cannot|can't|doesnt|doesn't|does not|wont|won't|"
    r"timeout|leak|deadlock|5\d\d|4\d\d"
    r")\b",
    re.IGNORECASE,
)


ROUTER_SYSTEM_PROMPT = """You are an intent router for an AI Code Investigation Agent.
Classify the user's issue/query into exactly one of three categories:

1. "CODE_INVESTIGATION": The user is reporting a software bug, defect, error, failure, or unexpected behavior in the target repository to be diagnosed.
   CRITICAL: If the input is ambiguous or could be a bug report (e.g. "orders broken", "payment fails", "it crashes", "the customer story endpoint returns 500"), you MUST classify it as CODE_INVESTIGATION.

2. "GENERAL_QUESTION": The user is asking a general programming, computer science, conceptual, or educational question (e.g. "What is Java?", "Explain recursion", "How does garbage collection work?").
   Do NOT classify technical conceptual questions as CODE_INVESTIGATION just because they mention "class", "method", "service", or "Java".
   For GENERAL_QUESTION, provide a concise, high-quality explanation in "direct_answer".

3. "OUT_OF_SCOPE": The query is non-technical, conversational chit-chat, or completely unrelated to software development.

You MUST respond ONLY with a single valid JSON object in this exact schema:
{
  "intent": "CODE_INVESTIGATION" | "GENERAL_QUESTION" | "OUT_OF_SCOPE",
  "reason": "Brief one-sentence rationale",
  "direct_answer": "Direct answer if intent is GENERAL_QUESTION, else null"
}
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract and parse JSON object from model output text."""
    if not text:
        return None
    # Strip markdown code blocks
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    # Find outermost { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def route_intent(
    issue: str,
    llm: LLMProvider | None = None,
) -> tuple[IntentResult, bool]:
    """
    Route the user's input to the appropriate intent.

    Returns:
        (IntentResult, used_llm)
        where used_llm is True if an LLM model call was issued.
    """
    cleaned = issue.strip()
    if not cleaned:
        return (
            IntentResult(
                intent="OUT_OF_SCOPE",
                reason="The issue description is empty.",
                direct_answer=None,
            ),
            False,
        )

    # ── Fast path 1: Greetings ────────────────────────────────────────────────
    if _GREETING_PATTERN.match(cleaned):
        return (
            IntentResult(
                intent="GENERAL_QUESTION",
                reason="Conversational greeting.",
                direct_answer=(
                    "Hello! I am the AI Code Investigation Agent. "
                    "Provide a software bug or defect report, and I will investigate the repository."
                ),
            ),
            False,
        )

    # ── Fast path 2: Identity questions ───────────────────────────────────────
    if _IDENTITY_PATTERN.match(cleaned):
        return (
            IntentResult(
                intent="GENERAL_QUESTION",
                reason="Assistant identity inquiry.",
                direct_answer=(
                    "I am the AI Code Investigation Agent (FRIDAY). "
                    "I analyze repositories to identify root causes of software defects and gather evidence."
                ),
            ),
            False,
        )

    # ── Fast path 3: Clear non-technical out-of-scope trivia ──────────────────
    if _NON_TECHNICAL_OUT_OF_SCOPE.match(cleaned) and not _DEFECT_KEYWORDS.search(cleaned):
        return (
            IntentResult(
                intent="OUT_OF_SCOPE",
                reason=f"The input '{cleaned}' is general non-technical trivia or chit-chat, not a software issue.",
                direct_answer=None,
            ),
            False,
        )

    # ── LLM Intent Classification ─────────────────────────────────────────────
    if llm is not None:
        try:
            response = llm.chat(
                messages=[{"role": "user", "content": f"User query: {cleaned}"}],
                system_prompt=ROUTER_SYSTEM_PROMPT,
            )
            data = _extract_json(response.text)
            if data and isinstance(data, dict):
                intent_raw = str(data.get("intent", "")).upper()
                if intent_raw in ("CODE_INVESTIGATION", "GENERAL_QUESTION", "OUT_OF_SCOPE"):
                    direct_answer = data.get("direct_answer")
                    if intent_raw == "GENERAL_QUESTION" and not direct_answer:
                        direct_answer = response.text or "Here is the information requested."
                    return (
                        IntentResult(
                            intent=intent_raw,
                            reason=str(data.get("reason", "Classified by intent router.")),
                            direct_answer=direct_answer,
                        ),
                        True,
                    )
        except Exception as exc:
            logger.warning("LLM intent routing encountered an error: %s. Using conservative fallback.", exc)

    # ── Conservative Fallback (Invalid LLM output or no LLM) ──────────────────
    # If it contains defect indicators or could plausibly be a bug, default to CODE_INVESTIGATION.
    if _DEFECT_KEYWORDS.search(cleaned) or len(cleaned.split()) <= 6:
        return (
            IntentResult(
                intent="CODE_INVESTIGATION",
                reason="Conservative routing to code investigation for plausible defect report.",
                direct_answer=None,
            ),
            llm is not None,
        )

    # If it starts with question words (What is, How does, Explain), treat as general question
    if re.match(r"^\s*(what\s+is|what\s+are|how\s+does|how\s+to|explain)\b", cleaned, re.IGNORECASE):
        return (
            IntentResult(
                intent="GENERAL_QUESTION",
                reason="General conceptual inquiry.",
                direct_answer=f"'{cleaned}' is a general programming concept.",
            ),
            llm is not None,
        )

    return (
        IntentResult(
            intent="CODE_INVESTIGATION",
            reason="Ambiguous input conservatively routed to code investigation.",
            direct_answer=None,
        ),
        llm is not None,
    )
