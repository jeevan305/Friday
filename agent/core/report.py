"""
CLI renderer for the canonical RCAReport.

Every section is produced exactly once, in a fixed order, from a single
validated ``RCAReport``.  The same object is what gets saved to JSON.
"""

from __future__ import annotations

from agent.models.rca import RCAReport

SECTION_TITLES: tuple[str, ...] = (
    "STATUS",
    "ROOT CAUSE",
    "CONFIDENCE",
    "EVIDENCE",
    "RECOMMENDED FIX",
    "DEVELOPER NEXT STEPS",
    "VERIFICATION PERFORMED",
    "REMAINING UNCERTAINTY",
    "INVESTIGATION SUMMARY",
)

_UNAVAILABLE = "  Unavailable (none reported)"


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.0f}%"


def _banner(title: str) -> str:
    bar = "=" * 56
    return f"\n{bar}\n  {title}\n{bar}"


def _numbered(items: list[str]) -> str:
    if not items:
        return _UNAVAILABLE
    return "\n".join(f"  {i}. {s}" for i, s in enumerate(items, 1))


def render_report(report: RCAReport) -> str:
    """Return the full human-readable report."""
    parts: list[str] = []

    parts.append(_banner("STATUS"))
    status_str = f"  {report.status.value}"
    if hasattr(report, "outcome_type") and report.outcome_type:
        val = report.outcome_type.value if hasattr(report.outcome_type, "value") else str(report.outcome_type)
        if val != "VALIDATED_DEFECT":
            status_str += f" ({val})"
    parts.append(status_str)

    parts.append(_banner("ROOT CAUSE"))
    parts.append(report.root_cause)

    parts.append(_banner("CONFIDENCE"))
    parts.append(f"  {format_confidence(report.confidence)}")

    parts.append(_banner("EVIDENCE"))
    if report.evidence:
        blocks = []
        for i, ev in enumerate(report.evidence, 1):
            lines = [f"  [{i}] {ev.file}  lines {ev.start_line}-{ev.end_line}"]
            if ev.symbol:
                lines.append(f"      Symbol  : {ev.symbol}")
            lines.append(f"      Explains: {ev.explanation}")
            blocks.append("\n".join(lines))
        parts.append("\n\n".join(blocks))
    else:
        parts.append("  Unavailable (no validated evidence)")

    parts.append(_banner("RECOMMENDED FIX"))
    parts.append(report.recommended_fix)

    parts.append(_banner("DEVELOPER NEXT STEPS"))
    parts.append(_numbered(report.developer_next_steps))

    parts.append(_banner("VERIFICATION PERFORMED"))
    parts.append(_numbered(report.verification_performed))

    parts.append(_banner("REMAINING UNCERTAINTY"))
    parts.append(_numbered(report.remaining_uncertainty))

    parts.append(_banner("INVESTIGATION SUMMARY"))
    parts.append(_numbered(report.investigation_summary))

    return "\n".join(parts) + "\n"
