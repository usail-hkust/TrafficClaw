"""
Domain knowledge registry for control modules.

Provides unified access to domain knowledge from all control modules,
plus shared knowledge about traffic patterns and cross-module interactions.
"""

from typing import Dict, List, Optional, Any


class DomainKnowledgeRegistry:
    """
    Registry for domain knowledge across all control modules.

    Provides:
    - Access to individual module domain knowledge
    - Shared time-based traffic pattern knowledge
    - Cross-module interaction principles
    - Combined knowledge for joint optimization
    """

    # Shared knowledge applicable across all modules
    SHARED_KNOWLEDGE = """## Time-Based Traffic Patterns

### Morning Rush (6:00-11:00)
- **Traffic Flow**: Primarily inbound to city center and business districts
- **Signal Timing**: Prioritize arterial roads heading into downtown
- **Highway**: Expect congestion on inbound routes; VSL may help smooth flow
- **Transit**: Higher demand on routes serving business districts
- **Taxi**: High demand for trips to offices and commercial areas

### Midday (11:00-14:00)
- **Traffic Flow**: Mixed directions, generally lower volume
- **Signal Timing**: Balanced phase allocation
- **Transit**: Moderate, steady demand
- **Taxi**: Moderate demand, good time for repositioning

### Afternoon (14:00-16:00)
- **Traffic Flow**: Building toward evening rush
- **Signal Timing**: Begin transitioning to outbound priority
- **Transit**: Increasing demand toward residential areas
- **Taxi**: Building demand for return trips

### Evening Rush (16:00-21:00)
- **Traffic Flow**: Primarily outbound from city center
- **Signal Timing**: Prioritize arterial roads heading out of downtown
- **Highway**: Expect congestion on outbound routes; aggressive ramp metering
- **Transit**: Highest demand period for many routes
- **Taxi**: High demand for trips home and entertainment venues

### Night (21:00-6:00)
- **Traffic Flow**: Low volume, mostly local trips
- **Signal Timing**: Can use shorter cycles, possibly actuated control
- **Highway**: Free flow conditions usually
- **Transit**: Reduced service, focus on efficiency
- **Taxi**: Lower but steady demand, entertainment venues early night

## Cross-Module Interaction Principles

### Signal Timing ↔ Bus Scheduling
- Signal priority for buses at intersections can improve bus reliability
- Coordinated signals can create green waves for bus routes
- Bus bunching can cause queue spillback affecting signal efficiency

### Signal Timing ↔ Taxi Scheduling
- Signal coordination affects taxi travel times
- Congested intersections increase taxi pickup/dropoff delays
- Consider taxi high-demand zones when allocating signal phases

### Highway Speed Limit ↔ Ramp Metering
- VSL and ramp metering should be coordinated for smooth traffic flow
- When highway is congested, restrict ramp inflow
- When highway is free-flowing, relax ramp restrictions
- Harmonize VSL changes with ramp metering cycles

### Transit ↔ Taxi
- Transit delays may increase taxi demand
- Taxi can serve as last-mile complement to transit
- Coordinate service levels during peak periods

## Performance Optimization Hierarchy

1. **Safety First**: Never compromise safety for throughput
2. **Network Throughput**: Maximize vehicles served
3. **Travel Time**: Minimize average travel time
4. **Reliability**: Reduce travel time variance
5. **Equity**: Balance service across zones
6. **Energy/Environment**: Consider fuel efficiency and emissions
"""

    # Module-specific knowledge cache
    _module_knowledge_cache: Dict[str, str] = {}

    def __init__(self):
        """Initialize the registry and cache module knowledge."""
        self._load_module_knowledge()

    def _load_module_knowledge(self):
        """Load domain knowledge from all control module classes."""
        try:
            from control_modules.registry import CONTROL_MODULES

            for module_name, module_class in CONTROL_MODULES.items():
                if hasattr(module_class, 'DOMAIN_KNOWLEDGE'):
                    self._module_knowledge_cache[module_name] = module_class.DOMAIN_KNOWLEDGE
        except ImportError:
            # Handle case where registry is not available
            pass

    def get_knowledge(self, module_name: str) -> str:
        """
        Get domain knowledge for a specific module.

        Args:
            module_name: Name of the control module

        Returns:
            Domain knowledge string for the module, or empty string if not found
        """
        return self._module_knowledge_cache.get(module_name, "")

    @classmethod
    def get_shared_knowledge(cls) -> str:
        """
        Get shared domain knowledge applicable to all modules.

        Returns:
            Shared domain knowledge string
        """
        return cls.SHARED_KNOWLEDGE

    def get_combined_knowledge(
        self,
        modules: List[str],
        include_shared: bool = True,
        include_interactions: bool = True
    ) -> str:
        """
        Get combined domain knowledge for multiple modules.

        Args:
            modules: List of module names to include
            include_shared: Whether to include shared knowledge
            include_interactions: Whether to include cross-module interaction info

        Returns:
            Combined domain knowledge string
        """
        parts = []

        # Add shared knowledge if requested
        if include_shared:
            parts.append("# Shared Domain Knowledge")
            parts.append(self.SHARED_KNOWLEDGE)
            parts.append("")

        # Add module-specific knowledge
        for module_name in modules:
            knowledge = self.get_knowledge(module_name)
            if knowledge:
                parts.append(f"# {module_name.replace('_', ' ').title()} Domain Knowledge")
                parts.append(knowledge)
                parts.append("")

        # Add interaction notes if multiple modules and requested
        if include_interactions and len(modules) > 1:
            interaction_notes = self._get_interaction_notes(modules)
            if interaction_notes:
                parts.append("# Cross-Module Interaction Notes")
                parts.append(interaction_notes)
                parts.append("")

        return "\n".join(parts)

    def _get_interaction_notes(self, modules: List[str]) -> str:
        """Get specific interaction notes for the given modules."""
        interactions = []

        # Check for signal + bus interaction
        if "signal_timing" in modules and "bus_scheduling" in modules:
            interactions.append("""### Signal Timing + Bus Scheduling
- Coordinate signal phases with bus arrival times
- Consider transit signal priority (TSP) at key intersections
- Bus schedule adherence affects intersection queue lengths""")

        # Check for signal + taxi interaction
        if "signal_timing" in modules and "taxi_scheduling" in modules:
            interactions.append("""### Signal Timing + Taxi Scheduling
- High taxi demand zones may need adjusted signal timing
- Consider taxi pickup/dropoff locations when setting phases
- Congested intersections increase taxi wait times""")

        # Check for highway + ramp interaction
        if "highway_speed_limit" in modules and "ramp_metering" in modules:
            interactions.append("""### Highway Speed Limit + Ramp Metering
- Coordinate VSL changes with ramp metering cycles
- When VSL is reduced, consider restricting ramp inflow
- Smooth traffic flow requires synchronized control""")

        # Check for highway + signal interaction
        if "highway_speed_limit" in modules and "signal_timing" in modules:
            interactions.append("""### Highway Speed Limit + Signal Timing
- Highway congestion affects arterial signal loads
- Consider feeder roads to highway ramps
- Coordinate for smooth transition from arterials to highway""")

        # Check for transit + taxi interaction
        if ("bus_scheduling" in modules or "subway_scheduling" in modules) and "taxi_scheduling" in modules:
            interactions.append("""### Transit + Taxi Scheduling
- Transit delays increase taxi demand
- Coordinate service levels at transit hubs
- Taxi repositioning should consider transit schedules""")

        return "\n\n".join(interactions)

    def get_time_aware_guidance(
        self,
        modules: List[str],
        current_time: float
    ) -> str:
        """
        Get time-aware optimization guidance for modules.

        Args:
            modules: List of active modules
            current_time: Current simulation time in seconds

        Returns:
            Time-specific guidance string
        """
        hour = (current_time / 3600) % 24

        if 6 <= hour < 11:
            period = "Morning Rush"
            general_guidance = "Focus on inbound traffic flow. Prioritize arterials to downtown."
            specific_guidance = {
                "signal_timing": "Favor green time for inbound movements",
                "highway_speed_limit": "Monitor inbound highway segments closely",
                "ramp_metering": "Apply aggressive metering on inbound ramps",
                "bus_scheduling": "Ensure capacity for peak inbound demand",
                "taxi_scheduling": "Position taxis near residential areas for pickups",
            }
        elif 11 <= hour < 14:
            period = "Midday"
            general_guidance = "Balanced optimization. Good time for testing changes."
            specific_guidance = {
                "signal_timing": "Balanced phase allocation",
                "highway_speed_limit": "Standard speeds, monitor for incidents",
                "ramp_metering": "Relaxed metering unless congestion detected",
                "bus_scheduling": "Steady operations, focus on schedule adherence",
                "taxi_scheduling": "Reposition for afternoon demand",
            }
        elif 14 <= hour < 16:
            period = "Afternoon"
            general_guidance = "Transition period. Begin preparing for evening rush."
            specific_guidance = {
                "signal_timing": "Start transitioning to outbound priority",
                "highway_speed_limit": "Prepare for outbound congestion",
                "ramp_metering": "Moderate metering, increasing toward peak",
                "bus_scheduling": "Build capacity for outbound routes",
                "taxi_scheduling": "Start positioning near offices",
            }
        elif 16 <= hour < 21:
            period = "Evening Rush"
            general_guidance = "Focus on outbound traffic flow. Maximum system stress."
            specific_guidance = {
                "signal_timing": "Favor green time for outbound movements",
                "highway_speed_limit": "Active management of outbound segments",
                "ramp_metering": "Aggressive metering to protect mainline",
                "bus_scheduling": "Maximum frequency on major outbound routes",
                "taxi_scheduling": "Position taxis near offices and transit hubs",
            }
        else:
            period = "Night"
            general_guidance = "Low traffic. Focus on efficiency and safety."
            specific_guidance = {
                "signal_timing": "Shorter cycles, consider actuated control",
                "highway_speed_limit": "Standard or slightly higher speeds",
                "ramp_metering": "Minimal or no metering needed",
                "bus_scheduling": "Reduced service, focus on reliability",
                "taxi_scheduling": "Focus on entertainment districts early night",
            }

        parts = [
            f"## Time Period: {period} ({hour:.0f}:00)",
            f"**General Guidance**: {general_guidance}",
            "",
            "**Module-Specific Guidance**:"
        ]

        for module in modules:
            if module in specific_guidance:
                parts.append(f"- **{module.replace('_', ' ').title()}**: {specific_guidance[module]}")

        return "\n".join(parts)

    def list_available_modules(self) -> List[str]:
        """List all modules with available domain knowledge."""
        return list(self._module_knowledge_cache.keys())
