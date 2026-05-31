"""
Shared infrastructure for control modules.

This package provides:
- DomainKnowledgeRegistry: Unified domain knowledge for all modules
- DecisionContextManager: Cross-module decision context for coordination
- EventScheduler: Event-based scheduling for simulation acceleration
- Shared tools and utilities for cross-module analysis
- Common data structures and dependencies
"""

from .domain_knowledge import DomainKnowledgeRegistry
from .decision_context import DecisionContextManager
from .event_scheduler import EventScheduler, ControlEvent
from .shared_tools import (
    analyze_zone_traffic,
    identify_congestion_hotspots,
    calculate_network_metrics,
)

__all__ = [
    "DomainKnowledgeRegistry",
    "DecisionContextManager",
    "EventScheduler",
    "ControlEvent",
    "analyze_zone_traffic",
    "identify_congestion_hotspots",
    "calculate_network_metrics",
]
