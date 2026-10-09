"""
Google Gemini LLM provider.

Uses the official google-genai Python SDK with automatic retry on rate-limit errors.

Conversation mapping (provider-agnostic message -> Gemini ``Content``)
---------------------------------------------------------------------
* user       -> ``Content(role="user",  parts=[text])``
* assistant  -> the model's original ``Content`` (``provider_content``) when present, so
                function-call parts and thought signatures survive into the next turn;
                otherwise rebuilt from ``content`` / ``tool_calls``.
* tool       -> ONE ``Content(role="user")`` with one ``function_response`` Part per tool
                call of the preceding assistant turn (same order), plus an optional text
                Part for the loop reminder.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import google.genai as genai
import google.genai.types as genai_types

from agent import config
from agent.llm.base import (
    LLMError,
    LLMProvider,
    LLMQuotaExhaustedError,
    LLMResponse,
    LLMTransientError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "string": genai_types.Type.STRING,
    "integer": genai_types.Type.INTEGER,
    "number": genai_types.Type.NUMBER,
    "boolean": genai_types.Type.BOOLEAN,
    "array": genai_types.Type.ARRAY,
    "object": genai_types.Type.OBJECT,
}


def _convert_schema(schema: dict[str, Any]) -> genai_types.Schema:
    """Recursively convert a JSON-schema-like dict into a Gemini Schema."""
    kwargs: dict[str, Any] = {
        "type": _TYPE_MAP.get(str(schema.get("type", "string")).lower(), genai_types.Type.STRING),
    }
    if schema.get("description"):
        kwargs["description"] = schema["description"]
    if schema.get("enum"):
        kwargs["enum"] = [str(e) for e in schema["enum"]]
    if "items" in schema:
        kwargs["items"] = _convert_schema(schema["items"])
    if "properties" in schema:
        kwargs["properties"] = {k: _convert_schema(v) for k, v in schema["properties"].items()}
    if schema.get("required"):
        kwargs["required"] = list(schema["required"])
    return genai_types.Schema(**kwargs)


def _build_tool(tool_defs: list[dict[str, Any]]) -> genai_types.Tool:
    """Convert agent-agnostic tool definitions -> one Gemini Tool."""
    decls = []
    for tool_def in tool_defs:
        params = tool_def.get("parameters") or {"type": "object", "properties": {}}
        decls.append(
            genai_types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                parameters=_convert_schema({"type": "object", **params}),
            )
        )
    return genai_types.Tool(function_declarations=decls)


def _text_part(text: str) -> genai_types.Part:
    return genai_types.Part.from_text(text=text)


def _function_response_part(entry: dict[str, Any]) -> genai_types.Part:
    kwargs: dict[str, Any] = {"name": entry["name"], "response": entry["response"]}
    if entry.get("id"):
        kwargs["id"] = entry["id"]
    return genai_types.Part(function_response=genai_types.FunctionResponse(**kwargs))


def _function_call_part(call: dict[str, Any]) -> genai_types.Part:
    kwargs: dict[str, Any] = {"name": call["name"], "args": call.get("arguments") or {}}
    if call.get("id"):
        kwargs["id"] = call["id"]
    return genai_types.Part(function_call=genai_types.FunctionCall(**kwargs))


def _to_gemini_messages(messages: list[dict[str, Any]]) -> list[genai_types.Content]:
    """Convert the provider-agnostic message list -> Gemini Content list."""
    contents: list[genai_types.Content] = []
    for msg in messages:
        role = msg["role"]

        if role == "tool":
            entries = msg.get("tool_results")
            if entries is None:  # legacy single-result shape
                result = msg.get("tool_result")
                entries = [{
                    "name": msg.get("name", "unknown"),
                    "response": result if isinstance(result, dict) else {"output": msg.get("content", "")},
                }]
            parts = [_function_response_part(e) for e in entries]
            if msg.get("note"):
                parts.append(_text_part(msg["note"]))
            contents.append(genai_types.Content(role="user", parts=parts))
            continue

        if role == "assistant":
            provider_content = msg.get("provider_content", msg.get("raw_candidate"))
            if isinstance(provider_content, genai_types.Content):
                contents.append(provider_content)
                continue
            parts = []
            if msg.get("content"):
                parts.append(_text_part(str(msg["content"])))
            parts.extend(_function_call_part(c) for c in msg.get("tool_calls") or [])
            if not parts:
                parts.append(_text_part("(no output)"))
            contents.append(genai_types.Content(role="model", parts=parts))
            continue

        contents.append(
            genai_types.Content(role="user", parts=[_text_part(str(msg.get("content", "")))])
        )
    return contents


def _thinking_config(level: str) -> genai_types.ThinkingConfig | None:
    if not level:
        return None
    # ThinkingLevel is a lenient enum in the SDK (unknown values only warn), so check
    # against its declared members ourselves instead of silently sending a bad value.
    declared = {
        m.name.lower(): m
        for m in genai_types.ThinkingLevel.__members__.values()
        if m.name != "THINKING_LEVEL_UNSPECIFIED"
    }
    if level.lower() not in declared:
        raise ValueError(
            f"Unsupported GEMINI_THINKING_LEVEL '{level}'. Valid values: "
            f"{', '.join(declared)} (or leave empty to use the model default)."
        )
    return genai_types.ThinkingConfig(thinking_level=declared[level.lower()])



_SECRET_RE = re.compile(
    r"(AIza[0-9A-Za-z-_]{35}|key=[a-zA-Z0-9_\-]+|api_key=[a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)

_HARD_QUOTA_RE = re.compile(
    r"\b(quota\s+exceeded|daily\s+limit|resource_exhausted.*quota|check\s+quota|billing|free\s+tier|individual\s+quota\s+reached)\b",
    re.IGNORECASE,
)

_MODEL_UNAVAILABLE_RE = re.compile(
    r"\b(model\s+not\s+found|is\s+not\s+supported|not\s+available\s+for\s+this\s+model|unknown\s+model)\b",
    re.IGNORECASE,
)

_RETRY_AFTER_RE = re.compile(
    r"\b(?:retry\s+(?:after|in)|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?\b",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    """Mask potential secrets and API keys from error messages."""
    return _SECRET_RE.sub("[REDACTED]", text)


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        retry_initial_wait: float | None = None,
        thinking_level: str | None = None,
    ) -> None:
        self._api_key = api_key or config.GEMINI_API_KEY
        self._model = model or config.GEMINI_MODEL
        self._max_retries = max_retries if max_retries is not None else config.LLM_MAX_RETRIES
        self._retry_initial_wait = (
            retry_initial_wait
            if retry_initial_wait is not None
            else config.LLM_RETRY_INITIAL_WAIT
        )
        level = config.GEMINI_THINKING_LEVEL if thinking_level is None else thinking_level
        self._thinking_level = (level or "").strip().lower()
        self._thinking_config = _thinking_config(self._thinking_level)  # validates early
        self._client = genai.Client(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self._model,
            "thinking_level": self._thinking_level or "not set (model default)",
            "sdk_version": getattr(genai, "__version__", "unknown"),
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        gemini_contents = _to_gemini_messages(messages)

        config_kwargs: dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if tools:
            config_kwargs["tools"] = [_build_tool(tools)]
            # The loop executes tools itself; do not let the SDK auto-call anything.
            config_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(
                disable=True
            )
        if self._thinking_config is not None:
            config_kwargs["thinking_config"] = self._thinking_config

        generate_config = genai_types.GenerateContentConfig(**config_kwargs)

        for attempt in range(self._max_retries + 1):
            if self.request_budget is not None and self.request_count >= self.request_budget:
                logger.warning("LLM request budget (%d) reached before outbound call.", self.request_budget)
                raise LLMQuotaExhaustedError(
                    f"LLM request budget of {self.request_budget} outbound calls reached."
                )

            self.request_count += 1

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=gemini_contents,
                    config=generate_config,
                )
                return self._parse_response(response)

            except Exception as exc:
                code = getattr(exc, "code", None)
                status_str = getattr(exc, "status", "")
                details = getattr(exc, "details", None)
                raw_msg = str(getattr(exc, "message", exc) or exc)
                clean_msg = _sanitize(f"{status_str}: {raw_msg}" if status_str and status_str not in raw_msg else raw_msg)

                # Check if model is unsupported or unavailable
                if code == 404 or _MODEL_UNAVAILABLE_RE.search(clean_msg):
                    logger.error("Gemini model %s is unavailable: %s", self._model, clean_msg)
                    raise LLMUnavailableError(
                        f"Model '{self._model}' is unavailable: {clean_msg}"
                    ) from exc

                is_rate_limit = (code == 429) or ("429" in clean_msg) or ("RESOURCE_EXHAUSTED" in clean_msg) or (status_str == "RESOURCE_EXHAUSTED")
                is_server_error = (code == 503) or ("503" in clean_msg) or ("UNAVAILABLE" in clean_msg) or (status_str == "UNAVAILABLE")
                is_hard_quota = bool(_HARD_QUOTA_RE.search(clean_msg))

                # Check for structured RetryInfo in details
                suggested_delay = None
                if isinstance(details, dict):
                    err_dict = details.get("error", details)
                    sub_details = err_dict.get("details", []) if isinstance(err_dict, dict) else []
                    for item in sub_details:
                        if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
                            delay_val = item.get("retryDelay")
                            if delay_val:
                                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(delay_val))
                                if m:
                                    suggested_delay = float(m.group(1))
                                    break

                if suggested_delay is None:
                    delay_match = _RETRY_AFTER_RE.search(clean_msg)
                    if delay_match:
                        try:
                            suggested_delay = float(delay_match.group(1))
                        except (ValueError, TypeError):
                            pass

                # If hard quota exhaustion without a retry window, fail immediately without burning retries
                if is_hard_quota and suggested_delay is None:
                    logger.error("Gemini API quota exhausted: %s", clean_msg)
                    raise LLMQuotaExhaustedError(
                        f"Gemini API quota exhausted (RESOURCE_EXHAUSTED): {clean_msg}"
                    ) from exc

                if (is_rate_limit or is_server_error) and attempt < self._max_retries:
                    # Check if budget would allow another outbound request
                    if self.request_budget is not None and self.request_count >= self.request_budget:
                        logger.warning("Aborting retry: LLM request budget (%d) reached.", self.request_budget)
                        raise LLMQuotaExhaustedError(
                            f"LLM request budget of {self.request_budget} calls reached; aborting rate-limit retry."
                        ) from exc

                    if suggested_delay is not None:
                        if suggested_delay > 30.0:
                            logger.error("Rate-limit retry delay (%.1fs) exceeds practical wait window.", suggested_delay)
                            raise LLMQuotaExhaustedError(
                                f"Rate-limit retry delay ({suggested_delay:.1f}s) exceeds practical wait window."
                            ) from exc
                        wait = max(suggested_delay, 1.0)
                    else:
                        wait = min(self._retry_initial_wait * (2 ** attempt), 30.0)

                    logger.warning(
                        "Transient error / rate limit hit (attempt %d/%d). Retrying in %.1fs …",
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                if is_rate_limit or is_hard_quota:
                    raise LLMQuotaExhaustedError(
                        f"Gemini API quota or rate-limit exhausted: {clean_msg}"
                    ) from exc

                if is_server_error:
                    raise LLMTransientError(
                        f"Gemini API service unavailable after {attempt} retries: {clean_msg}"
                    ) from exc

                raise LLMError(f"Gemini API request failed: {clean_msg}") from exc

        raise LLMError("LLM request failed after all retries")

    # ── internal ─────────────────────────────────────────────────────────────

    def _parse_response(self, response: Any) -> LLMResponse:
        """Extract text and tool calls from a Gemini response."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        provider_content = None

        candidates = getattr(response, "candidates", None)
        if not candidates:
            logger.warning(
                "Gemini returned no candidates (prompt_feedback=%s)",
                getattr(response, "prompt_feedback", None),
            )
        else:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            if content is not None and parts:
                provider_content = content
            elif getattr(candidate, "finish_reason", None) is not None:
                logger.warning("Gemini candidate had no content (finish_reason=%s)", candidate.finish_reason)
            for part in parts:
                if getattr(part, "thought", False):
                    continue  # reasoning summaries are not answer text
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    call: dict[str, Any] = {
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    }
                    if getattr(fc, "id", None):
                        call["id"] = fc.id
                    tool_calls.append(call)

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
            raw_candidate=provider_content,
        )
