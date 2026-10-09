#!/usr/bin/env python3
"""
AI Code Investigation Agent — CLI entrypoint.

Usage
-----
::

    python main.py --repo "C:/projects/order-management" \\
                   --issue "Active customers are unable to create orders"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── make sure stdout and stderr use utf-8 on Windows ──────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── make sure the project root is on sys.path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agent import config  # noqa: E402
from agent.core.agent import InvestigationAgent  # noqa: E402
from agent.core.report import render_report  # noqa: E402
from agent.models.rca import RCAReport  # noqa: E402
from pydantic import ValidationError  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ── output helpers ────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {title}")
    print(f"{'=' * 56}")


def _section(title: str, content: str) -> None:
    _banner(title)
    print(content)


def _print_trace(label: str, detail: object) -> None:
    """Live trace callback — called by the agent loop on every step."""
    print(label)


def _display_rca(rca: dict) -> bool:
    """Validate and print the canonical RCA exactly once. Returns False if invalid."""
    try:
        report = RCAReport.model_validate(rca)
    except ValidationError as exc:
        print("\n❌  The RCA result failed validation and cannot be displayed:", file=sys.stderr)
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            print(f"    - {loc}: {err['msg']}", file=sys.stderr)
        return False
    print(render_report(report))
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Code Investigation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Absolute path to the repository to investigate.",
    )
    parser.add_argument(
        "--issue",
        required=True,
        help="Description of the production issue to investigate.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=f"Maximum investigation iterations (default: {config.MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--request-budget",
        type=int,
        default=None,
        help=f"Maximum LLM request budget per run (default: {config.LLM_REQUEST_BUDGET}).",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        default=None,
        help="Write the full result JSON to this file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate API key
    try:
        config.validate()
    except EnvironmentError as exc:
        print(f"\n❌  {exc}\n", file=sys.stderr)
        return 1

    # Validate repo path
    repo_path = Path(args.repo)
    if not repo_path.exists():
        print(f"\n❌  Repository path does not exist: {repo_path}\n", file=sys.stderr)
        return 1

    # Build the agent first so an invalid configuration (e.g. unsupported thinking
    # level) fails loudly before the run starts.
    try:
        agent = InvestigationAgent(
            repo_path=str(repo_path),
            max_iterations=args.max_iterations,
            request_budget=args.request_budget,
            progress_callback=_print_trace,
        )
    except Exception as exc:
        print(f"\n❌  Could not initialise the agent: {exc}\n", file=sys.stderr)
        return 1

    llm_cfg = agent.llm_config
    _banner("AI CODE INVESTIGATION AGENT")
    print(f"  Repository : {repo_path}")
    print(f"  Issue      : {args.issue}")
    print(f"  Model      : {llm_cfg.get('model')}")
    print(f"  Thinking   : {llm_cfg.get('thinking_level', 'n/a')}")
    print(f"  Max iters  : {args.max_iterations or config.MAX_ITERATIONS} (safety ceiling)")
    print(f"  Req budget : {args.request_budget or config.LLM_REQUEST_BUDGET} requests")
    print()
    print("── INVESTIGATION TRACE ─────────────────────────────────")

    try:
        result = agent.investigate(issue=args.issue)
    except Exception as exc:
        print(f"\n❌  Investigation failed: {exc}\n", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    status = result.get("status", "UNKNOWN")

    print("\n── INVESTIGATION FINISHED ──────────────────────────────")
    print(f"  Run ID     : {result.get('run_id', 'N/A')}")
    print(f"  Status     : {status}")
    if result.get("intent"):
        print(f"  Intent     : {result['intent']}")
    if result.get("request_count") is not None:
        print(f"  Requests   : {result['request_count']} / {result.get('request_budget', 'N/A')}")
    if result.get("termination_reason"):
        print(f"  Termination: {result['termination_reason']}")
    if result.get("run_file"):
        print(f"  Run file   : {result['run_file']}")

    if status == "GENERAL_QUESTION":
        _banner("STATUS")
        print("  GENERAL_QUESTION")
        _banner("ANSWER")
        print(f"  {result.get('direct_answer') or 'No answer generated.'}\n")
        valid = True
    elif status == "OUT_OF_SCOPE":
        _banner("STATUS")
        print("  OUT_OF_SCOPE")
        _banner("SCOPE GUARD")
        print(f"  {result.get('scope_reason') or result.get('intent_reason') or 'The provided input is not a software defect report.'}")
        print("\n  The Code Investigation Agent investigates software defects and unexpected repository behavior.")
        print("  Please provide an error message, reproduction steps, or unexpected system behavior to investigate.\n")
        valid = True
    elif status == "ERROR":
        valid = _display_rca(result.get("rca") or {}) if result.get("rca") else False
        print("\n❌  Investigation terminated with ERROR. See trace and run file for details.\n", file=sys.stderr)
    else:
        # The report below is the only place the RCA sections are printed.
        valid = _display_rca(result.get("rca") or {})

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(
            json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Full result written to: {out_path}")

    if status == "GENERAL_QUESTION":
        return 0
    if status == "OUT_OF_SCOPE":
        return 2
    if not valid or status == "ERROR":
        return 1
    return 0 if status == "COMPLETED" else 2



if __name__ == "__main__":
    sys.exit(main())
