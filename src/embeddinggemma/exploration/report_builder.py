"""
Report builder for goal-driven exploration.

Aggregates discoveries from multiple phases and generates structured reports.
"""
from __future__ import annotations
from typing import Any
from datetime import datetime

from .goals import get_goal


class ExplorationReport:
    """Builder for exploration reports."""

    def __init__(self, goal_type: str, run_id: str):
        self.goal_type = goal_type
        self.run_id = run_id
        self.goal = get_goal(goal_type)
        self.start_time = datetime.now()
        self.end_time: datetime | None = None

        # Track discoveries per phase
        self.phases_completed: list[int] = []
        self.phase_discoveries: dict[int, dict[str, Any]] = {}
        self.files_accessed: set[str] = set()
        self.unique_queries: set[str] = set()

        # Aggregate discoveries across phases
        self.entry_points: list[dict[str, str]] = []
        self.modules: list[dict[str, str]] = []
        self.classes: list[dict[str, str]] = []
        self.functions: list[dict[str, str]] = []
        self.imports: list[str] = []
        self.patterns: list[dict[str, str]] = []
        self.data_flows: list[dict[str, str]] = []
        self.security_findings: list[dict[str, str]] = []
        self.error_handlers: list[dict[str, str]] = []

    def add_phase_discovery(
        self,
        phase_index: int,
        category: str,
        item: dict[str, Any] | str,
    ) -> None:
        """Add a discovery for a specific phase."""
        if phase_index not in self.phase_discoveries:
            self.phase_discoveries[phase_index] = {}

        if category not in self.phase_discoveries[phase_index]:
            self.phase_discoveries[phase_index][category] = []

        self.phase_discoveries[phase_index][category].append(item)

        # Also add to aggregate lists
        if isinstance(item, dict):
            if category == "entry_points":
                self.entry_points.append(item)
            elif category == "modules":
                self.modules.append(item)
            elif category == "classes":
                self.classes.append(item)
            elif category == "functions":
                self.functions.append(item)
            elif category == "patterns":
                self.patterns.append(item)
            elif category == "data_flows":
                self.data_flows.append(item)
            elif category == "security":
                self.security_findings.append(item)
            elif category == "error_handling":
                self.error_handlers.append(item)
        elif isinstance(item, str):
            if category == "imports":
                self.imports.append(item)

    def mark_phase_complete(self, phase_index: int) -> None:
        """Mark a phase as completed."""
        if phase_index not in self.phases_completed:
            self.phases_completed.append(phase_index)

    def finalize(self) -> None:
        """Mark report as complete."""
        self.end_time = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        runtime = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else (datetime.now() - self.start_time).total_seconds()
        )

        return {
            "goal_type": self.goal_type,
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "runtime_seconds": runtime,
            "phases_completed": self.phases_completed,
            "total_phases": len(self.goal["phases"]) if self.goal else 0,
            "summary": {
                "files_accessed": len(self.files_accessed),
                "unique_queries": len(self.unique_queries),
                "entry_points_found": len(self.entry_points),
                "modules_found": len(self.modules),
                "classes_found": len(self.classes),
                "functions_found": len(self.functions),
                "imports_found": len(self.imports),
                "patterns_found": len(self.patterns),
                "data_flows_found": len(self.data_flows),
                "security_findings": len(self.security_findings),
                "error_handlers_found": len(self.error_handlers),
            },
            "discoveries": {
                "entry_points": self.entry_points[:20],  # Top 20
                "modules": self.modules[:20],
                "classes": self.classes[:30],
                "functions": self.functions[:50],
                "imports": list(set(self.imports))[:30],
                "patterns": self.patterns[:15],
                "data_flows": self.data_flows[:20],
                "security_findings": self.security_findings[:20],
                "error_handlers": self.error_handlers[:30],
            },
            "phase_discoveries": self.phase_discoveries,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        if not self.goal:
            return "# Error: Invalid goal type\n"

        template = self.goal.get("report_template", "")
        report = f"# {self.goal['description']}\n\n"
        report += f"**Run ID:** {self.run_id}\n"
        report += f"**Goal Type:** {self.goal_type}\n"
        report += f"**Completed Phases:** {len(self.phases_completed)}/{len(self.goal['phases'])}\n"
        report += f"**Runtime:** {self.to_dict()['runtime_seconds']:.2f} seconds\n\n"

        report += "---\n\n"

        # Architecture report
        if self.goal_type == "architecture":
            report += self._build_architecture_report()
        elif self.goal_type == "bugs":
            report += self._build_bugs_report()
        elif self.goal_type == "security":
            report += self._build_security_report()
        else:
            report += self._build_generic_report()

        return report

    def _build_architecture_report(self) -> str:
        """Build architecture-specific report."""
        report = "## System Overview\n\n"

        if self.entry_points:
            report += "### Entry Points\n\n"
            for ep in self.entry_points[:10]:
                file_path = ep.get("file_path", "unknown")
                name = ep.get("name", "unknown")
                description = ep.get("description", "")
                report += f"- **{name}** ([{file_path}]({file_path}))\n"
                if description:
                    report += f"  - {description}\n"
            report += "\n"

        if self.modules:
            report += "### Core Modules\n\n"
            for mod in self.modules[:15]:
                name = mod.get("name", "unknown")
                responsibility = mod.get("responsibility", "")
                files = mod.get("files", [])
                report += f"#### {name}\n"
                if responsibility:
                    report += f"- **Purpose:** {responsibility}\n"
                if files:
                    report += f"- **Files:** {', '.join(files[:5])}\n"
                report += "\n"

        if self.data_flows:
            report += "### Data Flow\n\n"
            for flow in self.data_flows[:10]:
                name = flow.get("name", "unknown")
                description = flow.get("description", "")
                report += f"- **{name}:** {description}\n"
            report += "\n"

        if self.imports:
            unique_imports = list(set(self.imports))
            external = [imp for imp in unique_imports if not imp.startswith(".") and "embeddinggemma" not in imp]
            internal = [imp for imp in unique_imports if imp.startswith(".") or "embeddinggemma" in imp]

            report += "### Dependencies\n\n"
            if external:
                report += "**External Dependencies:**\n"
                for imp in external[:20]:
                    report += f"- {imp}\n"
                report += "\n"

            if internal:
                report += "**Internal Modules:**\n"
                for imp in internal[:20]:
                    report += f"- {imp}\n"
                report += "\n"

        if self.patterns:
            report += "### Design Patterns\n\n"
            for pattern in self.patterns[:10]:
                name = pattern.get("name", "unknown")
                description = pattern.get("description", "")
                location = pattern.get("location", "")
                report += f"- **{name}**"
                if location:
                    report += f" ([{location}]({location}))"
                report += "\n"
                if description:
                    report += f"  - {description}\n"
            report += "\n"

        return report

    def _build_bugs_report(self) -> str:
        """Build bugs-specific report."""
        report = "## Error Handling Analysis\n\n"

        if self.error_handlers:
            report += f"**Total Error Handlers Found:** {len(self.error_handlers)}\n\n"
            report += "### Error Handling Patterns\n\n"
            for handler in self.error_handlers[:20]:
                file_path = handler.get("file_path", "unknown")
                pattern = handler.get("pattern", "unknown")
                description = handler.get("description", "")
                report += f"- **{pattern}** in [{file_path}]({file_path})\n"
                if description:
                    report += f"  - {description}\n"
            report += "\n"

        report += "## Recommendations\n\n"
        report += "- Review error handling coverage\n"
        report += "- Add validation for edge cases\n"
        report += "- Implement consistent error logging\n\n"

        return report

    def _build_security_report(self) -> str:
        """Build security-specific report."""
        report = "## Security Analysis\n\n"

        if self.security_findings:
            report += f"**Total Findings:** {len(self.security_findings)}\n\n"
            report += "### Security Findings\n\n"
            for finding in self.security_findings[:20]:
                severity = finding.get("severity", "INFO")
                title = finding.get("title", "unknown")
                location = finding.get("location", "")
                description = finding.get("description", "")

                report += f"**[{severity}]** {title}\n"
                if location:
                    report += f"- Location: [{location}]({location})\n"
                if description:
                    report += f"- {description}\n"
                report += "\n"

        report += "## Security Recommendations\n\n"
        report += "- Implement input validation\n"
        report += "- Review authentication mechanisms\n"
        report += "- Audit authorization checks\n\n"

        return report

    def _build_generic_report(self) -> str:
        """Build generic report for unknown goal types."""
        report = "## Discoveries\n\n"
        summary = self.to_dict()["summary"]

        for key, value in summary.items():
            if value > 0:
                report += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        report += "\n"
        return report
