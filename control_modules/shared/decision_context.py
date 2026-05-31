"""
Decision Context Manager for cross-module coordination.

This module provides a system for tracking and sharing optimization decisions
across control modules, enabling coordinated decision-making in joint optimization.

Example usage:
    manager = DecisionContextManager()

    # Record a decision after signal timing optimization
    manager.record_decision(
        module="signal_timing",
        summary="Increased northbound green time at intersections 123, 456",
        changes=[{"entity": "inter_123", "action": "increased", "phase": "NT", "delta": 15}],
        optimization_focus="northbound_congestion"
    )

    # Get context for bus scheduling optimization
    context = manager.get_coordination_context(
        for_module="bus_scheduling",
        dependencies={"signal_timing": ["bus_scheduling"]}
    )
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import time


class DecisionContextManager:
    """
    Manages decision context for cross-module coordination.

    Captures optimization decisions with semantic meaning and provides
    coordination context for LLM prompts.
    """

    def __init__(self):
        """Initialize the decision context manager."""
        # {module: [decisions]} where each decision is a dict
        self.decisions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._checkpoint_counter = 0

    def record_decision(
        self,
        module: str,
        summary: str,
        changes: List[Dict[str, Any]],
        optimization_focus: Optional[str] = None,
        checkpoint: Optional[int] = None
    ) -> None:
        """
        Record an optimization decision.

        Args:
            module: Name of the control module (e.g., "signal_timing")
            summary: Human-readable summary of what changed
            changes: List of detailed change dicts with entity, action, details
            optimization_focus: Optional focus of optimization (e.g., "northbound_congestion")
            checkpoint: Optional checkpoint number
        """
        if checkpoint is None:
            checkpoint = self._checkpoint_counter

        # Extract affected entities from changes
        affected_entities = []
        for change in changes:
            entity = change.get("entity")
            if entity and entity not in affected_entities:
                affected_entities.append(entity)

        decision = {
            "module": module,
            "checkpoint": checkpoint,
            "timestamp": time.time(),
            "summary": summary,
            "optimization_focus": optimization_focus,
            "affected_entities": affected_entities,
            "changes": changes
        }

        self.decisions[module].append(decision)

    def set_checkpoint(self, checkpoint: int) -> None:
        """Set the current checkpoint number."""
        self._checkpoint_counter = checkpoint

    def get_recent_decisions(
        self,
        module: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get recent decisions, optionally filtered by module.

        Args:
            module: Optional module name to filter by
            limit: Maximum number of decisions to return

        Returns:
            List of recent decision dictionaries
        """
        if module:
            decisions = self.decisions.get(module, [])
            return decisions[-limit:] if decisions else []

        # Get all decisions across modules, sorted by timestamp
        all_decisions = []
        for mod_decisions in self.decisions.values():
            all_decisions.extend(mod_decisions)

        all_decisions.sort(key=lambda d: d.get("timestamp", 0), reverse=True)
        return all_decisions[:limit]

    def get_coordination_context(
        self,
        for_module: str,
        dependencies: Dict[str, List[str]]
    ) -> str:
        """
        Build coordination context string for prompts.

        Args:
            for_module: The module that needs coordination context
            dependencies: Dependency graph {module: [affected_modules]}

        Returns:
            Formatted string with relevant decision context
        """
        lines = []

        # Find modules that affect the target module
        affecting_modules = []
        for upstream_module, affected_list in dependencies.items():
            if for_module in affected_list:
                affecting_modules.append(upstream_module)

        if not affecting_modules:
            return ""

        for upstream in affecting_modules:
            decisions = self.decisions.get(upstream, [])
            if not decisions:
                continue

            recent = decisions[-1]  # Most recent decision
            lines.append(f"\n## {upstream.replace('_', ' ').title()} (affects {for_module})")
            lines.append(f"- **Summary**: {recent['summary']}")

            if recent.get('optimization_focus'):
                lines.append(f"- **Focus**: {recent['optimization_focus']}")

            if recent.get('affected_entities'):
                entities = recent['affected_entities'][:5]
                lines.append(f"- **Affected**: {', '.join(entities)}")
                if len(recent['affected_entities']) > 5:
                    lines.append(f"  (and {len(recent['affected_entities']) - 5} more...)")

        return "\n".join(lines)

    def get_all_coordination_context(
        self,
        for_modules: List[str],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """
        Build coordination context for multiple modules.

        Args:
            for_modules: List of modules that need coordination context
            dependencies: Dependency graph {module: [affected_modules]}

        Returns:
            Formatted string with all relevant decision contexts
        """
        all_context = []
        seen_upstreams = set()

        for module in for_modules:
            # Find modules that affect this module
            for upstream_module, affected_list in dependencies.items():
                if module in affected_list and upstream_module not in seen_upstreams:
                    decisions = self.decisions.get(upstream_module, [])
                    if decisions:
                        seen_upstreams.add(upstream_module)
                        recent = decisions[-1]

                        lines = []
                        lines.append(f"\n## {upstream_module.replace('_', ' ').title()} (affects {module})")
                        lines.append(f"- **Summary**: {recent['summary']}")

                        if recent.get('optimization_focus'):
                            lines.append(f"- **Focus**: {recent['optimization_focus']}")

                        if recent.get('affected_entities'):
                            entities = recent['affected_entities'][:5]
                            lines.append(f"- **Affected**: {', '.join(entities)}")
                            if len(recent['affected_entities']) > 5:
                                lines.append(f"  (and {len(recent['affected_entities']) - 5} more...)")

                        all_context.append("\n".join(lines))

        return "\n".join(all_context)

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Serialize for passing to code sandbox or prompt builder.

        Returns:
            Dictionary copy of all decisions by module
        """
        return dict(self.decisions)

    def clear(self, module: Optional[str] = None) -> None:
        """
        Clear decisions.

        Args:
            module: Optional module name to clear. If None, clears all decisions.
        """
        if module:
            self.decisions[module] = []
        else:
            self.decisions.clear()

    def merge_from(self, other: "DecisionContextManager") -> None:
        """
        Merge decisions from another manager.

        Args:
            other: Another DecisionContextManager to merge from
        """
        for module, decisions in other.decisions.items():
            self.decisions[module].extend(decisions)

    def __repr__(self) -> str:
        total = sum(len(d) for d in self.decisions.values())
        modules = list(self.decisions.keys())
        return f"DecisionContextManager(modules={modules}, total_decisions={total})"
