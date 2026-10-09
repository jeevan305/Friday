"""
Deterministic repository inventory and candidate prioritization.

Scans the repository using existing read-only Workspace and RepositoryScanner
to build a compact structural inventory (controllers, services, repositories,
exceptions, routing) and ranks candidate files for an issue before the LLM
multi-turn loop begins.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.repository.repository_scanner import RepositoryScanner, SKIP_DIRS
from agent.repository.workspace import Workspace

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "with", "from", "by", "that", "this", "it", "not", "be",
    "as", "returns", "return", "returned", "returning", "fails", "failed", "failing",
    "crashes", "crashed", "broken", "issue", "problem", "error", "bug", "defect",
    "service", "app", "application", "cannot", "unable", "can", "does", "doesn't",
})

ENDPOINT_KEYWORDS = frozenset({
    "endpoint", "route", "url", "api", "controller", "http", "rest", "get", "post",
    "put", "delete", "patch", "500", "404", "400", "401", "403", "request", "response"
})

FAILURE_KEYWORDS = frozenset({
    "500", "400", "404", "502", "503", "exception", "error", "fail", "failure", "crash",
    "nullpointer", "nullpointerexception", "npe", "unhandled"
})


@dataclass
class FileMetadata:
    path: str               # repo-relative path using forward slashes
    name: str               # file name
    role: str               # "controller", "service", "repository", "exception", "routing", "component", "config", "other"
    annotations: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    snippet: str = ""       # first ~200 lines excerpt for fast analysis


@dataclass
class CandidateRanking:
    file_path: str
    metadata: FileMetadata
    score: float
    reasons: list[str]


@dataclass
class RepositoryInventory:
    framework: str
    source_roots: list[str] = field(default_factory=list)
    controllers: list[FileMetadata] = field(default_factory=list)
    services: list[FileMetadata] = field(default_factory=list)
    repositories: list[FileMetadata] = field(default_factory=list)
    exceptions: list[FileMetadata] = field(default_factory=list)
    readme_files: list[str] = field(default_factory=list)
    all_source_files: list[str] = field(default_factory=list)
    ranked_candidates: list[CandidateRanking] = field(default_factory=list)

    def to_prompt_summary(self, max_candidates: int = 5) -> str:
        """Format a compact, bounded Markdown summary for the LLM initial prompt."""
        lines = [
            "### Deterministic Repository Inventory",
            f"- **Framework / Stack:** {self.framework}",
        ]
        if self.source_roots:
            lines.append(f"- **Source Roots:** {', '.join(self.source_roots[:3])}")

        # Controllers and mapped endpoints
        if self.controllers:
            lines.append("- **Controllers & Registered Endpoints:**")
            for c in self.controllers[:8]:
                routes_str = f" [Routes: {', '.join(c.routes[:6])}]" if c.routes else ""
                lines.append(f"  * `{c.name}` ({c.path}){routes_str}")
        else:
            lines.append("- **Controllers:** None identified via standard annotations.")

        # Services
        if self.services:
            svc_names = [f"`{s.name}` ({s.path})" for s in self.services[:6]]
            lines.append(f"- **Services:** {', '.join(svc_names)}")

        # Repositories
        if self.repositories:
            repo_names = [f"`{r.name}`" for r in self.repositories[:6]]
            lines.append(f"- **Repositories:** {', '.join(repo_names)}")

        # Exceptions
        if self.exceptions:
            exc_names = [f"`{e.name}`" for e in self.exceptions[:6]]
            lines.append(f"- **Exception Classes:** {', '.join(exc_names)}")

        # Candidate rankings
        if self.ranked_candidates:
            lines.append("")
            lines.append("### Prioritized Candidate Files for this Issue")
            for i, cand in enumerate(self.ranked_candidates[:max_candidates], start=1):
                reason_str = "; ".join(cand.reasons[:2])
                lines.append(f"{i}. `{cand.file_path}` — {reason_str}")

        lines.append("")
        lines.append("### Recommended Action Plan")
        lines.append("1. Prefer reading prioritized controller/service files directly using `read_file` instead of broad searching.")
        lines.append("2. For endpoint issues, check if the reported route exists in the controllers listed above.")
        lines.append("   - If the endpoint route is NOT defined in any controller in the repository:")
        lines.append("     * Read the primary controller to confirm the registered routes.")
        lines.append("     * If confirmed absent, STOP SEARCHING IMMEDIATELY. Do not investigate unrelated service or exception classes.")
        lines.append("     * Call finish_investigation promptly with status=\"INCOMPLETE\", outcome_type=\"ENDPOINT_NOT_FOUND\", confidence <= 0.40 (e.g. 0.25-0.35),")
        lines.append("       citing the controller routes as evidence, stating that the route is absent, and clarifying that HTTP 500 is unverified at runtime.")
        lines.append("3. For confirmed endpoints, follow the call chain: Controller -> Service -> Repository -> Exception handling.")

        return "\n".join(lines)


def extract_issue_tokens(issue: str) -> list[str]:
    """Extract lowercased significant tokens from the issue string."""
    words = re.findall(r"[a-zA-Z0-9_-]+", issue.lower())
    tokens: list[str] = []
    for w in words:
        clean = w.strip("_-")
        if len(clean) >= 3 and clean not in STOP_WORDS:
            tokens.append(clean)
    return tokens


def _parse_spring_routes(content: str) -> tuple[list[str], list[str]]:
    """Extract Spring annotations and normalized HTTP routes from Java source."""
    annotations: list[str] = []
    routes: list[str] = []

    # Detect class-level annotations
    if "@RestController" in content:
        annotations.append("@RestController")
    if "@Controller" in content:
        annotations.append("@Controller")

    # Base path mapping on class
    base_path = ""
    base_match = re.search(
        r'@RequestMapping\s*\(\s*(?:(?:value|path)\s*=\s*)?["\']([^"\']+)["\']',
        content,
    )
    if base_match:
        base_path = base_match.group(1).rstrip("/")
        annotations.append(f'@RequestMapping("{base_path}")')

    # Method mappings
    method_pattern = re.compile(
        r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\s*(?:\(\s*(?:(?:value|path)\s*=\s*)?["\']([^"\']+)["\']\s*\))?'
    )
    for m in method_pattern.finditer(content):
        mapping_type = m.group(1)
        subpath = m.group(2) or ""
        if subpath and not subpath.startswith("/"):
            subpath = "/" + subpath

        http_method = mapping_type.replace("Mapping", "").upper()
        if http_method == "REQUEST":
            http_method = "ANY"

        full_route = f"{http_method} {base_path}{subpath}".strip()
        if full_route not in routes:
            routes.append(full_route)
        if mapping_type not in annotations:
            annotations.append(f"@{mapping_type}")

    return annotations, routes


def _parse_angular_routes(content: str) -> tuple[list[str], list[str]]:
    """Extract Angular routing path definitions from TypeScript source."""
    annotations: list[str] = []
    routes: list[str] = []

    if "@Component" in content:
        annotations.append("@Component")
    if "@Injectable" in content:
        annotations.append("@Injectable")

    # Match path: '...' or path: "..."
    for m in re.finditer(r"path:\s*['\"]([^'\"]*)['\"]", content):
        p = m.group(1).strip()
        if p:
            routes.append(f"ROUTE /{p}")

    return annotations, routes


def build_repository_inventory(workspace: Workspace, issue: str = "") -> RepositoryInventory:
    """
    Construct a deterministic repository inventory using existing read-only filesystem tools.
    """
    scanner = RepositoryScanner(workspace)
    source_files = list(scanner.iter_source_files())

    source_roots: list[str] = []
    for candidate_root in ("src/main/java", "src/test/java", "src", "app", "lib"):
        if (workspace.root / candidate_root).is_dir():
            source_roots.append(candidate_root)

    controllers: list[FileMetadata] = []
    services: list[FileMetadata] = []
    repositories: list[FileMetadata] = []
    exceptions: list[FileMetadata] = []
    readme_files: list[str] = []
    all_rel_paths: list[str] = []

    is_spring = False
    is_angular = False
    is_python = False

    # Check config/build indicators in root
    for f in workspace.root.glob("*"):
        if f.is_file():
            fname = f.name.lower()
            if fname in ("pom.xml", "build.gradle", "build.gradle.kts"):
                is_spring = True
            elif fname in ("angular.json", "package.json"):
                try:
                    pkg_text = f.read_text(encoding="utf-8", errors="replace")
                    if "@angular" in pkg_text or fname == "angular.json":
                        is_angular = True
                except Exception:
                    pass
            elif fname in ("requirements.txt", "pyproject.toml", "setup.py"):
                is_python = True
            if fname.startswith("readme"):
                readme_files.append(workspace.relative(f))

    for fpath in source_files:
        rel_path = workspace.relative(fpath).replace("\\", "/")
        all_rel_paths.append(rel_path)
        fname = fpath.name

        # Read first ~200 lines (or 8 KB) for fast static analysis
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                head_lines = [fh.readline() for _ in range(200)]
            content = "".join(head_lines)
        except Exception:
            content = ""

        # Check framework markers
        if "@RestController" in content or "@Controller" in content or "org.springframework" in content:
            is_spring = True
        if "@Component" in content and "@angular/core" in content:
            is_angular = True

        # Classify by role
        role = "other"
        annotations: list[str] = []
        routes: list[str] = []

        # Spring / Java classification
        if fpath.suffix == ".java":
            if "@RestController" in content or "@Controller" in content or fname.endswith("Controller.java"):
                role = "controller"
                annotations, routes = _parse_spring_routes(content)
            elif "@Service" in content or fname.endswith("Service.java"):
                role = "service"
                if "@Service" in content:
                    annotations.append("@Service")
            elif "@Repository" in content or fname.endswith("Repository.java") or "extends JpaRepository" in content:
                role = "repository"
                if "@Repository" in content:
                    annotations.append("@Repository")
            elif fname.endswith("Exception.java") or "@ControllerAdvice" in content or "@ExceptionHandler" in content:
                role = "exception"
                if "@ControllerAdvice" in content:
                    annotations.append("@ControllerAdvice")

        # Angular / TypeScript classification
        elif fpath.suffix in (".ts", ".js"):
            if "routes.ts" in fname or "routing" in fname or "Routes = " in content:
                role = "routing"
                annotations, routes = _parse_angular_routes(content)
            elif fname.endswith(".component.ts") or "@Component" in content:
                role = "component"
                annotations, routes = _parse_angular_routes(content)
            elif fname.endswith(".service.ts") or "@Injectable" in content:
                role = "service"
                if "@Injectable" in content:
                    annotations.append("@Injectable")

        # Python classification
        elif fpath.suffix == ".py":
            if "router" in fname or "controller" in fname or "@app." in content or "@router." in content:
                role = "controller"
            elif "service" in fname:
                role = "service"
            elif "repository" in fname or "repo" in fname:
                role = "repository"
            elif "exception" in fname or "error" in fname:
                role = "exception"

        meta = FileMetadata(
            path=rel_path,
            name=fname,
            role=role,
            annotations=annotations,
            routes=routes,
            snippet=content[:500],
        )

        if role == "controller":
            controllers.append(meta)
        elif role == "service":
            services.append(meta)
        elif role == "repository":
            repositories.append(meta)
        elif role == "exception":
            exceptions.append(meta)

    # Determine framework label
    if is_spring:
        framework = "Spring Boot (Java)"
    elif is_angular:
        framework = "Angular (TypeScript)"
    elif is_python:
        framework = "Python"
    else:
        framework = "Generic Source Repository"

    inventory = RepositoryInventory(
        framework=framework,
        source_roots=source_roots,
        controllers=controllers,
        services=services,
        repositories=repositories,
        exceptions=exceptions,
        readme_files=readme_files,
        all_source_files=all_rel_paths,
    )

    if issue:
        inventory.ranked_candidates = rank_candidates(inventory, issue)

    return inventory


def rank_candidates(
    inventory: RepositoryInventory,
    issue: str,
    inspected_files: set[str] | None = None,
) -> list[CandidateRanking]:
    """
    Rank candidate files using evidence from issue tokens, file paths,
    framework annotations, endpoint mappings, and role heuristics.
    """
    inspected = inspected_files or set()
    issue_tokens = extract_issue_tokens(issue)
    is_endpoint_issue = any(t in ENDPOINT_KEYWORDS for t in issue_tokens)
    is_failure_issue = any(t in FAILURE_KEYWORDS for t in issue_tokens)

    all_files: list[FileMetadata] = (
        inventory.controllers + inventory.services + inventory.repositories + inventory.exceptions
    )

    rankings: list[CandidateRanking] = []

    for meta in all_files:
        score = 0.0
        reasons: list[str] = []
        fname_lower = meta.name.lower()
        path_lower = meta.path.lower()

        # 1. Exact or partial token matches in filename
        for token in issue_tokens:
            if token in fname_lower:
                score += 25.0
                reasons.append(f"filename matches '{token}'")

        # 2. Endpoint route matching
        for route in meta.routes:
            route_lower = route.lower()
            for token in issue_tokens:
                if token in route_lower:
                    score += 35.0
                    reasons.append(f"endpoint route matches '{token}' ({route})")

        # 3. Role-based prioritization for endpoint defect
        if is_endpoint_issue:
            if meta.role == "controller":
                score += 50.0
                reasons.append("primary controller for endpoint issue")
            elif meta.role == "service":
                score += 10.0
                reasons.append("service component")
            elif meta.role in ("routing", "component"):
                score += 30.0
                reasons.append("routing/component definition")

        # 4. Role-based prioritization for failure codes (e.g. 500)
        if is_failure_issue:
            if meta.role == "exception":
                score += 15.0
                reasons.append("exception handling for failure symptom")

        # 5. Penalize test files relative to application code
        if "test" in path_lower:
            score -= 15.0

        if score > 0.0:
            rankings.append(CandidateRanking(
                file_path=meta.path,
                metadata=meta,
                score=score,
                reasons=reasons,
            ))

    # Sort descending by score
    rankings.sort(key=lambda r: r.score, reverse=True)

    # Deduplicate while preserving order
    seen_paths = set()
    deduped: list[CandidateRanking] = []
    for r in rankings:
        if r.file_path not in seen_paths:
            seen_paths.add(r.file_path)
            deduped.append(r)

    # If no candidates matched tokens, default to all controllers then services
    if not deduped:
        for c in inventory.controllers:
            deduped.append(CandidateRanking(
                file_path=c.path,
                metadata=c,
                score=10.0,
                reasons=["controller file in repository"],
            ))
        for s in inventory.services:
            deduped.append(CandidateRanking(
                file_path=s.path,
                metadata=s,
                score=5.0,
                reasons=["service file in repository"],
            ))

    return deduped
