"""
Event Scheduler for Control Modules.

This module provides event-based scheduling for time-sensitive control modules
(signal_timing, ramp_metering) to enable sub-step simulation acceleration.

The scheduler allows pre-computing control events within a time window,
enabling larger macro-steps with precise event execution at exact times.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq


@dataclass(order=True)
class ControlEvent:
    """
    A control event scheduled for a specific time.

    Attributes:
        time: Simulation time when the event should be executed
        priority: Priority for ordering events at the same time (lower = higher priority)
        module: Name of the control module that owns this event
        entity_id: ID of the controlled entity (e.g., intersection_id, ramp_id)
        action: Action to execute (e.g., phase index for signals, True/False for ramps)
        metadata: Additional event-specific data
    """
    time: float
    module: str = field(compare=False)
    entity_id: str = field(compare=False)
    action: Any = field(compare=False)
    priority: int = field(default=0, compare=True)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


class EventScheduler:
    """
    Event scheduler for managing control events within a time window.

    This scheduler enables simulation acceleration by:
    1. Pre-computing all control events in a time window
    2. Identifying unique event times for sub-step execution
    3. Managing event execution in time order

    Usage:
        scheduler = EventScheduler()
        scheduler.add_events(signal_events)
        scheduler.add_events(ramp_events)

        unique_times = scheduler.get_unique_times(start, end)
        for t in unique_times:
            env.simulationStep(t)
            events = scheduler.pop_events_until(t)
            for event in events:
                apply_action(event)
    """

    def __init__(self):
        """Initialize an empty event scheduler."""
        self._events: List[ControlEvent] = []
        self._event_count = 0  # Counter for stable sorting

    def clear(self):
        """Clear all scheduled events."""
        self._events = []
        self._event_count = 0

    def add_event(self, event: ControlEvent):
        """
        Add a single event to the scheduler.

        Args:
            event: ControlEvent to schedule
        """
        heapq.heappush(self._events, event)
        self._event_count += 1

    def add_events(self, events: List[ControlEvent]):
        """
        Add multiple events to the scheduler.

        Args:
            events: List of ControlEvent objects to schedule
        """
        for event in events:
            heapq.heappush(self._events, event)
            self._event_count += 1

    def pop_events_until(self, end_time: float, tolerance: float = 1e-6) -> List[ControlEvent]:
        """
        Pop and return all events up to and including end_time.

        Args:
            end_time: Maximum time for events to include
            tolerance: Floating point comparison tolerance

        Returns:
            List of ControlEvent objects in time order
        """
        result = []
        while self._events and self._events[0].time <= end_time + tolerance:
            result.append(heapq.heappop(self._events))
        return result

    def peek_next_time(self) -> Optional[float]:
        """
        Get the time of the next event without removing it.

        Returns:
            Time of next event, or None if no events scheduled
        """
        return self._events[0].time if self._events else None

    def get_unique_times(self, start_time: float, end_time: float, tolerance: float = 1e-6) -> List[float]:
        """
        Get sorted list of unique event times within a time range.

        Args:
            start_time: Start of time range (exclusive)
            end_time: End of time range (inclusive)
            tolerance: Floating point comparison tolerance

        Returns:
            Sorted list of unique event times
        """
        times = set()
        for event in self._events:
            if start_time + tolerance < event.time <= end_time + tolerance:
                times.add(event.time)
        return sorted(times)

    def get_events_in_range(self, start_time: float, end_time: float, tolerance: float = 1e-6) -> List[ControlEvent]:
        """
        Get all events within a time range without removing them.

        Args:
            start_time: Start of time range (exclusive)
            end_time: End of time range (inclusive)
            tolerance: Floating point comparison tolerance

        Returns:
            List of ControlEvent objects in time order
        """
        events = []
        for event in self._events:
            if start_time + tolerance < event.time <= end_time + tolerance:
                events.append(event)
        return sorted(events, key=lambda e: (e.time, e.priority))

    def is_empty(self) -> bool:
        """Check if the scheduler has no pending events."""
        return len(self._events) == 0

    def __len__(self) -> int:
        """Return the number of pending events."""
        return len(self._events)

    def __repr__(self) -> str:
        return f"EventScheduler({len(self._events)} events)"
