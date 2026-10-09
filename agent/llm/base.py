"""
Abstract LLM provider interface.

Any new provider (OpenAI, Anthropic, Ollama, …) must implement this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """Base class for LLM provider errors."""
    pass


class LLMQuotaExhaustedError(LLMError):
    """Raised when an API quota or rate-limit has been completely exhausted."""
    pass


class LLMTransientError(LLMError):
    """Raised when a transient network or server error occurs."""
    pass


class LLMUnavailableError(LLMError):
    """Raised when the requested model is unavailable or misconfigured."""
    pass


class LLMResponse:
    """Normalised response from any LLM provider."""

    def __init__(
        self,
        text: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        raw: Any = None,
        raw_candidate: Any = None,
    ) -> None:
        self.text = text or ""
        self.tool_calls: list[dict[str, Any]] = tool_calls or []
        self.raw = raw  # provider-specific object for debugging
        self.raw_candidate = raw_candidate

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(text={self.text!r:.80}, "
            f"tool_calls={[tc.get('name') for tc in self.tool_calls]})"
        )


class LLMProvider(ABC):
    """Provider-agnostic interface for LLM access."""

    request_count: int = 0
    request_budget: int | None = None

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to the LLM and return a normalised response.

        Parameters
        ----------
        messages:
            Conversation history in provider-agnostic format.
            Each message is a dict with a ``role`` of user / assistant / tool.
            See ``agent.core.prompts`` for the exact shapes (an assistant message
            may carry ``tool_calls``; a tool message carries ``tool_results``
            answering every call of the preceding assistant message).
        tools:
            Optional list of tool declarations in provider-agnostic format.
        system_prompt:
            Optional system instruction.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    def describe(self) -> dict[str, Any]:
        """Effective provider configuration (reported in the run artifact)."""
        return {"model": self.model_name}
