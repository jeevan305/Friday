"""
Root Cause Analysis output model.

``RCAReport`` is the single canonical, Pydantic-validated representation of an
investigation result.  The CLI renderer, the run-JSON artifact, the API
response and the tests all consume this same object (via ``to_dict()``), so
they cannot drift apart.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RCAStatus(str, Enum):
    COMPLETED = "COMPLETED"    # evidence gate passed
    INCOMPLETE = "INCOMPLETE"  # evidence insufficient / iteration ceiling hit
    ERROR = "ERROR"            # run aborted (e.g. LLM failure)


class InvestigationOutcome(str, Enum):
    VALIDATED_DEFECT = "VALIDATED_DEFECT"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    RUNTIME_UNVERIFIED = "RUNTIME_UNVERIFIED"


class EvidenceItem(BaseModel):
    """One piece of code evidence: a repo-relative file and an inclusive line range."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    symbol: str = ""
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "EvidenceItem":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RCAReport(BaseModel):
    """Canonical root-cause-analysis report."""

    model_config = ConfigDict(extra="forbid")

    status: RCAStatus
    outcome_type: InvestigationOutcome = Field(
        default=InvestigationOutcome.VALIDATED_DEFECT,
        description="Semantic category of the investigation outcome."
    )
    problem: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_fix: str = Field(min_length=1)
    developer_next_steps: list[str] = Field(default_factory=list)
    verification_performed: list[str] = Field(default_factory=list)
    remaining_uncertainty: list[str] = Field(default_factory=list)
    investigation_summary: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_outcome_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "outcome_type" not in data or data["outcome_type"] is None:
                status = data.get("status")
                if status in ("INCOMPLETE", RCAStatus.INCOMPLETE, "ERROR", RCAStatus.ERROR):
                    data["outcome_type"] = InvestigationOutcome.INSUFFICIENT_EVIDENCE
                else:
                    data["outcome_type"] = InvestigationOutcome.VALIDATED_DEFECT
        return data

    @property
    def has_validated_root_cause(self) -> bool:
        """True if the report represents a genuinely validated code defect."""
        return self.status == RCAStatus.COMPLETED and self.outcome_type == InvestigationOutcome.VALIDATED_DEFECT

    @field_validator("confidence", mode="before")
    @classmethod
    def _numeric_confidence(cls, v: Any) -> Any:
        # Accept numeric strings ("0.8") but never bool / percentages / junk.
        if isinstance(v, bool):
            raise ValueError("confidence must be a number between 0.0 and 1.0")
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                raise ValueError("confidence must be a number between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _completed_requires_evidence(self) -> "RCAReport":
        if self.status == RCAStatus.COMPLETED:
            if not self.evidence:
                raise ValueError("a COMPLETED report requires at least one evidence item")
            if self.confidence >= 1.0:
                raise ValueError(
                    "confidence cannot be 1.0 (100%) for static code investigation without runtime execution or test verification"
                )
            if self.outcome_type == InvestigationOutcome.ENDPOINT_NOT_FOUND and self.confidence > 0.40:
                raise ValueError(
                    "confidence cannot exceed 0.40 when outcome_type is ENDPOINT_NOT_FOUND: absence of a route does not establish the root cause of the reported error"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RCAReport":
        return cls.model_validate(data)


# Backwards-compatible alias for older imports.
RCA = RCAReport
