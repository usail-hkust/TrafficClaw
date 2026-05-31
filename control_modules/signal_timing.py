"""
Traffic signal timing control module.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import ControlModule

DEFAULT_PHASE_DURATION = 15
MIN_PHASE_DURATION = 10  # Minimum phase duration in seconds

class TrafficSignalModule(ControlModule):
    """Control module for traffic signal timing."""

    DOMAIN_KNOWLEDGE = f"""Traffic signal control is critical for managing intersection capacity and reducing delays.

## 1. Core Principles and Constraints

- **Phase Configuration Rules:**
  - Each intersection has predefined phases - only modify timing for phases that exist in `current_signal_config`, do not create new phases
  - The minimal green time for each phase is {MIN_PHASE_DURATION}s - you MUST allocate more than {MIN_PHASE_DURATION}s for each phase
  - To plan a new signal configuration for a specific intersection, you MUST include all phases in the new configuration. DO NOT miss any existing phases or create new phases
  - DO NOT create signal configuration from scratch. You MUST start from `current_signal_config`, and gradually adjust the timing of each phase according to simulation results
  - Configuration format: {{"intersection_id": {{"phase_name": duration, ...}}, ...}}
  - Example: signal_config["intersection_id"] = {{"NT": 13, "ETWT": 10, "ELWL": 10, "NL": 12}} (Note: phases vary by intersection. Always check phases using `current_signal_config` first)

## 2. Data Analysis and Historical Traffic Data Strategy

- **IMPORTANT: Historical Traffic Analysis Strategy**
  - **Prioritize recent historical traffic data**: When analyzing traffic patterns, prioritize the most recent time steps as they reflect current traffic conditions most accurately
  - **Available time range for analysis**: You can ONLY use traffic snapshot data from time 0 to current simulation time. The optimization time window is in the FUTURE and has NO SNAPSHOT DATA AVAILABLE
  - **Task**: Analyze historical traffic data to identify traffic demand of intersections and phase directions (both busy and free-flow sections)

## 3. Intersection Prioritization and Demand Analysis

- **IMPORTANT: Intersection Prioritization Strategy**
  - **Free-flow intersections (lane queue length < 5)**: For intersections experiencing low traffic volumes, minimize green time for low demand phases to {MIN_PHASE_DURATION} seconds. Then, allocate green time to directions with long waiting times
  - **Busy intersections (lane queue length >= 5)**: Use traffic indicators (e.g., high queue length, high occupancy, or high arrival rates) to identify busy directions. Then, minimize green time for free-flow directions and increase phase durations of busy directions
  - **Direction-based demand analysis**: For each intersection, assess traffic demand for all 8 standard directions (ET, EL, WT, WL, NT, NL, ST, SL). Use the `loc_dir` attribute from `lane_states` to determine each lane's direction. Each phase allows vehicles at most two directions to pass (e.g., ETWT phase allows ET and WT directions to pass, ET phase only allows ET direction to pass)
  - Always use traffic metrics obtained from GET_CONTROL_APIs to quantitatively evaluate and compare traffic demands for each intersection and direction before making adjustments

## 4. Signal Timing Optimization Strategies

- **Duration Fine-tuning (with POLICY_PLANNING)**
  - **Identify demands**: Identify demand at different phases and intersections using traffic data from GET_CONTROL_APIs
  - **Use cached analysis results**: If you cached intersection analysis or phase-level demand in DATA_ANALYSIS, use `load_cache(key)` to retrieve them
  - **Allocation principles:**
    - Allocate green time of each phase proportionally to traffic volume (i.e., higher volume = longer green time, lower volume = less green time)
    - At intersections with higher traffic volume: gradually increase green time of busy phases, or gradually decrease green time of low-demand phases (DO NOT reduce below {MIN_PHASE_DURATION}s)
    - At intersections with free-flow traffic: decrease green time of low-demand phases as much as possible (DO NOT reduce below {MIN_PHASE_DURATION}s)
    - Increase phase durations if current cycle time (sum of phases) is too short to accommodate traffic volume
  - **Optimization Process**: Use POLICY_PLANNING action to optimize signal timing. Note: Simulation is automatically executed after POLICY_PLANNING - you will receive results and comparison automatically. Complete with FINISH action when satisfied

- **Optimization Time Window Considerations**
  - **Important**: The optimization time window is in the FUTURE and has NO SNAPSHOT DATA AVAILABLE. Your optimized signal timing will be applied during this future period
  - **Consider the optimization time window** when optimizing:
    - Your optimized signal timing will be active during the specified time window
    - During rush hours (Morning: 6:00-11:00, Evening: 16:00-21:00), traffic patterns are different from off-peak periods
    - If the optimization window spans multiple time periods, consider the dominant period or design adaptive strategies
    - Adjust signal timing strategies based on expected traffic demand for the upcoming period
    - Use historical data from similar time periods to inform your optimization decisions

## 5. Performance Metrics

- **Primary optimization goal**: Average travel time (lower is better)
- **Secondary metrics**:
  - Queue length: Shorter queues indicate better signal efficiency
  - Waiting time: Lower waiting time indicates better signal efficiency

## 6. Common Traffic Patterns

- Morning rush: 6:00-11:00
- Evening rush: 16:00-21:00"""

    def __init__(self, config_dir_name: Optional[str] = None):
        super().__init__("signal_timing", "signal_timing.json", config_dir_name=config_dir_name)
    
    def get_default_config(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initialize signal timing configuration from SUMO environment.
        Creates default timing for all intersections based on filtered four phases: ['ETWT', 'NTST', 'ELWL', 'NLSL'].
        
        Args:
            env: SUMOEnv instance with initialized intersections
            
        Returns:
            Dictionary with default timing configurations
            Format: {"intersection_id": {"phase_name": duration, ...}}
            Note: Cycle is automatically calculated from sum of phase durations
        """
        config = {}
        
        if env is None:
            print("Warning: Environment is None, cannot generate default configuration")
            return config
        
        if not env.intersection_dict:
            print("Warning: No intersections found in environment")
            return config
        
        for inter in env.intersection_dict.values():
            inter_id = inter.inter_id
            
            # Get filtered four phases using _get_four_phase() method
            filtered_2_eight, eight_2_filtered, filtered_phases = inter._get_four_phase()
            
            if not filtered_phases:
                print(f"Warning: No filtered phases found for intersection {inter_id}")
                continue
            
            timing = {}
            for phase_name in filtered_phases:
                timing[phase_name] = DEFAULT_PHASE_DURATION
            
            # New format: directly use phase dictionary, cycle is auto-calculated
            config[inter_id] = timing
        
        return config
    
    def validate_config(self, config: Dict[str, Any], reference_config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate traffic signal configuration.
        Checks all intersections and collects all errors, but reports each error type with only one example.
        
        Args:
            config: Configuration dictionary to validate
                    Format: {"intersection_id": {"phase_name": duration, ...}, ...}
            reference_config: Optional reference configuration to check phase completeness.
                            If provided, validates that config contains all phases from reference_config.
                            Format: {"intersection_id": {"phase_name": duration, ...}, ...}
            
        Returns:
            Tuple of (is_valid, error_message):
            - is_valid: True if configuration is valid, False otherwise
            - error_message: Error message string if invalid, None if valid
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        if len(config) == 0:
            return False, "Configuration is empty"
        
        # Collect all errors by type
        errors_by_type = {}
        
        for inter_id, inter_config in config.items():
            # Check if inter_config is a dictionary
            if not isinstance(inter_config, dict):
                error_type = "not_dict"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(inter_id)
                continue
            
            # inter_config is now directly the phase dictionary: {"phase_name": duration, ...}
            timing = inter_config
            
            # Check if timing dictionary is empty
            if len(timing) == 0:
                error_type = "timing_empty"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(inter_id)
                continue
            
            # Validate each phase duration
            for phase_name, phase_duration in timing.items():
                if not isinstance(phase_duration, (int, float)):
                    error_type = "phase_duration_not_number"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((inter_id, phase_name))
                    continue
                
                if phase_duration < MIN_PHASE_DURATION:
                    error_type = "phase_duration_too_short"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((inter_id, phase_name, phase_duration))
                    continue
            
            # Validate phase completeness if reference_config is provided
            if reference_config is not None and inter_id in reference_config:
                ref_timing = reference_config[inter_id]
                # Handle backward compatibility: old format with "timing" field
                if isinstance(ref_timing, dict) and "timing" in ref_timing:
                    ref_timing = ref_timing["timing"]
                
                    if isinstance(ref_timing, dict):
                        required_phases = set(ref_timing.keys())
                        provided_phases = set(timing.keys())
                        missing_phases = required_phases - provided_phases
                        redundant_phases = provided_phases - required_phases
                        if missing_phases:
                            error_type = "missing_phases"
                            if error_type not in errors_by_type:
                                errors_by_type[error_type] = []
                            errors_by_type[error_type].append((inter_id, sorted(missing_phases)))
                        if redundant_phases:
                            error_type = "redundant_phases"
                            if error_type not in errors_by_type:
                                errors_by_type[error_type] = []
                            errors_by_type[error_type].append((inter_id, sorted(redundant_phases)))
        
        # If no errors, return valid
        if not errors_by_type:
            return True, None
        
        # Build error message with examples only
        error_messages = []
        for error_type, error_list in errors_by_type.items():
            count = len(error_list)
            example = error_list[0]  # Use first occurrence as example
            
            if error_type == "not_dict":
                error_messages.append(
                    f"Configuration must be a dictionary (e.g., intersection '{example}' is not a dictionary). "
                    f"Found {count} intersection(s) with this issue."
                )
            elif error_type == "timing_empty":
                error_messages.append(
                    f"Phase dictionary is empty (e.g., intersection '{example}'). "
                    f"Found {count} intersection(s) with this issue."
                )
            elif error_type == "phase_duration_not_number":
                inter_id, phase_name = example
                error_messages.append(
                    f"Phase duration must be a number (e.g., intersection '{inter_id}', phase '{phase_name}'). "
                    f"Found {count} phase(s) with this issue."
                )
            elif error_type == "phase_duration_too_short":
                inter_id, phase_name, phase_duration = example
                error_messages.append(
                    f"Phase duration must be at least {MIN_PHASE_DURATION}s "
                    f"(e.g., intersection '{inter_id}', phase '{phase_name}' has {phase_duration}s). "
                    f"Found {count} phase(s) with this issue."
                )
            elif error_type == "missing_phases":
                inter_id, missing_phases = example
                error_messages.append(
                    f"Missing required phases (e.g., intersection '{inter_id}': missing {sorted(missing_phases)}). "
                    f"Found {count} intersection(s) with this issue."
                )
            elif error_type == "redundant_phases":
                inter_id, redundant_phases = example
                error_messages.append(
                    f"Contains redundant phases (e.g., intersection '{inter_id}': redundant {sorted(redundant_phases)}). "
                    f"Found {count} intersection(s) with this issue."
                )
        
        return False, "; ".join(error_messages)
    
    def apply_control(
        self,
        env: Any,
        config: Dict[str, Any],
        current_time: float,
        control_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply traffic signal timing control logic.
        
        This function calculates min_remaining based on current simulation time.
        It determines which intersections need phase switches and calculates the minimum
        time until the next switch is needed.
        
        IMPORTANT: When multiple control modules are used together, the actual step_duration
        may be smaller than this module's remaining duration (if other modules have shorter
        remaining durations). The update_control_state method handles this by recalculating
        cycle_time based on the actual current_time after step execution.
        
        Args:
            env: SUMOEnv instance
            config: Signal timing configuration
                    Format: {"intersection_id": {"phase_name": duration, ...}, ...}
                    Note: Cycle is automatically calculated from sum of phase durations
            current_time: Current simulation time
            control_state: Current control state (maintained across steps)
                          If None, will initialize state
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dictionary containing:
                - switch_actions: Dict mapping intersection_id to action_idx for phase switches
                - control_state: Control state (not updated yet, will be updated after step)
                - min_remaining: Minimum time until next phase switch (based on current_time)
                - selected_intersections: List of intersections that need to switch now
        """
        # Extract timing_dict and cycle_dict from config
        # Config format: {"intersection_id": {"phase_name": duration, ...}, ...}
        # Cycle is automatically calculated from sum of phase durations (direct phase switching, no yellow)
        timing_dict = {}
        cycle_dict = {}
        for inter_id, inter_config in config.items():
            timing_dict[inter_id] = inter_config if isinstance(inter_config, dict) else {}

            # Auto-calculate cycle from sum of phase durations (direct switching)
            cycle_dict[inter_id] = sum(timing_dict[inter_id].values()) if timing_dict[inter_id] else 60

        # Initialize control state if not provided
        if control_state is None:
            control_state = self._initialize_control_state(env, timing_dict, cycle_dict, current_time)

        # Get initial time from control state (or use current_time if not set)
        initial_time = control_state.get("initial_time", current_time)

        # Calculate cycle time for each intersection based on current simulation time
        # cycle_time = (current_time - initial_time) % cycle
        # Direct phase switching, no yellow phases
        intersection_cycle_time = {}
        for inter_id in control_state["timing_dict_code"]:
            # Cycle is just the sum of phase durations (no yellow)
            cycle = cycle_dict.get(inter_id, 60)
            elapsed_time = current_time - initial_time
            intersection_cycle_time[inter_id] = elapsed_time % cycle

        # Determine which intersections need phase switches based on relative cycle time
        switch_actions = {}
        selected_intersections = []

        intersection_next_switch_time = control_state["intersection_next_switch_time"]
        intersection_phase_index = control_state["intersection_phase_index"]
        intersection_phase_schedule = control_state["intersection_phase_schedule"]

        for inter_id, cycle_time in intersection_cycle_time.items():
            schedule = intersection_phase_schedule[inter_id]
            idx = intersection_phase_index[inter_id]
            next_switch = intersection_next_switch_time[inter_id]
            cycle = cycle_dict.get(inter_id, 60)

            # Check if we've reached or passed the switch time using relative time
            # Use a small tolerance for floating point comparison
            tolerance = 1e-6
            # Special handling for cycle boundary: if next_switch == cycle, 
            # it means next switch is at the start of next cycle (cycle_time = 0)
            if next_switch >= cycle - tolerance:
                # Next switch is at cycle boundary (start of next cycle)
                # Only trigger when cycle_time is close to 0 (start of cycle)
                if cycle_time < tolerance:
                    selected_intersections.append(inter_id)
                    if idx < len(schedule):
                        phase_idx = schedule[idx][0]  # Action index
                        switch_actions[inter_id] = phase_idx
            elif cycle_time >= next_switch - tolerance:
                # We've reached or passed the switch time within current cycle
                selected_intersections.append(inter_id)
                # idx is the index of the NEXT phase to switch to (updated in update_control_state)
                if idx < len(schedule):
                    phase_idx = schedule[idx][0]  # Action index
                    switch_actions[inter_id] = phase_idx

        # Calculate minimum time until next switch based on relative cycle time
        min_remaining = float('inf')
        timing_dict_code = control_state["timing_dict_code"]
        for inter_id in timing_dict_code:
            # Cycle is just the sum of phase durations (no yellow)
            cycle = cycle_dict.get(inter_id, 60)
            cycle_time = intersection_cycle_time[inter_id]
            next_switch = intersection_next_switch_time[inter_id]
            schedule = intersection_phase_schedule[inter_id]
            idx = intersection_phase_index[inter_id]

            # Calculate remaining time until next switch
            tolerance = 1e-6
            # Special handling for cycle boundary: if next_switch == cycle, 
            # it means next switch is at the start of next cycle (cycle_time = 0)
            if next_switch >= cycle - tolerance:
                # Next switch is at cycle boundary
                if cycle_time < tolerance:
                    # We're at the switch time (start of cycle)
                    remaining = 0.0
                else:
                    # Time until cycle wraps to 0
                    remaining = cycle - cycle_time
            elif cycle_time < next_switch - tolerance:
                # Next switch is within the current cycle
                remaining = next_switch - cycle_time
            else:
                # We're at or past the switch time
                # Remaining should be 0 to indicate we need to switch now (maintain current signal process)
                remaining = 0.0

            min_remaining = min(min_remaining, remaining)

        # Return switch actions and state (state will be updated after step)
        return {
            "actions": switch_actions,
            "control_state": control_state,
            "min_remaining": min_remaining,
            "selected_intersections": selected_intersections
        }

    def schedule_events(
        self,
        env: Any,
        config: Dict[str, Any],
        start_time: float,
        end_time: float,
        control_state: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Any], Dict[str, Any]]:
        """
        Pre-compute all phase switch events within a time range.

        This method enables simulation acceleration by pre-calculating all
        signal phase transitions that will occur in the specified time window.
        Instead of checking each second, the simulation can step directly to
        event times.

        Args:
            env: SUMOEnv instance
            config: Signal timing configuration
                    Format: {"intersection_id": {"phase_name": duration, ...}, ...}
            start_time: Start of time window (current simulation time)
            end_time: End of time window
            control_state: Current control state (will be initialized if None)

        Returns:
            Tuple of (events, control_state):
                - events: List of ControlEvent objects for phase switches
                - control_state: Updated control state
        """
        from control_modules.shared.event_scheduler import ControlEvent

        events = []

        # Extract timing_dict and cycle_dict from config
        timing_dict = {}
        cycle_dict = {}
        for inter_id, inter_config in config.items():
            timing_dict[inter_id] = inter_config if isinstance(inter_config, dict) else {}
            cycle_dict[inter_id] = sum(timing_dict[inter_id].values()) if timing_dict[inter_id] else 60

        # Initialize control state if not provided
        if control_state is None:
            control_state = self._initialize_control_state(env, timing_dict, cycle_dict, start_time)

        initial_time = control_state.get("initial_time", start_time)
        timing_dict_code = control_state["timing_dict_code"]
        intersection_phase_schedule = control_state["intersection_phase_schedule"]

        # Generate events for each intersection
        for inter_id in timing_dict_code:
            schedule = intersection_phase_schedule.get(inter_id, [])
            if not schedule:
                continue

            cycle = cycle_dict.get(inter_id, 60)
            if cycle <= 0:
                continue

            # Calculate which cycle we're starting in
            elapsed_from_init = start_time - initial_time
            cycles_completed = int(elapsed_from_init / cycle)
            cycle_start = initial_time + cycles_completed * cycle

            # Generate events for current and future cycles within the window
            current_cycle_start = cycle_start
            while current_cycle_start < end_time:
                cumulative_time = 0.0
                for phase_idx, (action_idx, phase_start, duration) in enumerate(schedule):
                    switch_time = current_cycle_start + cumulative_time
                    # Only include events strictly after start_time and up to end_time
                    if start_time < switch_time <= end_time:
                        events.append(ControlEvent(
                            time=switch_time,
                            module="signal_timing",
                            entity_id=inter_id,
                            action=action_idx,
                            priority=0,  # Signal timing has highest priority
                            metadata={"phase_idx": phase_idx}
                        ))
                    cumulative_time += duration
                current_cycle_start += cycle

        return events, control_state

    def update_control_state(
        self,
        control_state: Dict[str, Any],
        cycle_dict: Dict[str, int],
        step_duration: float,
        selected_intersections: List[str],
        current_time: float
    ) -> Dict[str, Any]:
        """
        Update signal control state after a simulation step.

        Args:
            control_state: Current control state
            cycle_dict: Dictionary mapping intersection_id to cycle time (direct switching, no yellow)
            step_duration: Duration of the simulation step
            selected_intersections: List of intersections that switched phases
            current_time: Current simulation time after the step

        Returns:
            Updated control state
        """
        intersection_phase_index = control_state["intersection_phase_index"]
        intersection_phase_schedule = control_state["intersection_phase_schedule"]
        intersection_next_switch_time = control_state["intersection_next_switch_time"]
        timing_dict_code = control_state["timing_dict_code"]
        initial_time = control_state.get("initial_time", 0.0)

        # IMPORTANT: Recalculate cycle times based on current_time (after step execution)
        # This accounts for the actual step_duration that was executed (which may be smaller
        # than this module's remaining duration if other modules had shorter remaining durations)
        # Direct phase switching, no yellow phases
        intersection_cycle_time = {}
        for inter_id in timing_dict_code:
            # Cycle is just the sum of phase durations (no yellow)
            cycle = cycle_dict.get(inter_id, 60)
            elapsed_time = current_time - initial_time
            intersection_cycle_time[inter_id] = elapsed_time % cycle

        # Update phase indices for switched intersections and calculate next switch time (relative time)
        # Note: selected_intersections contains intersections that were switched in this step
        # Their next_switch_time needs to be updated to the next phase's start time
        for inter_id in selected_intersections:
            current_idx = intersection_phase_index[inter_id]
            schedule = intersection_phase_schedule[inter_id]
            next_idx = (current_idx + 1) % len(schedule)
            intersection_phase_index[inter_id] = next_idx

            # Calculate next switch time using relative time
            if schedule:
                phase_start_time_relative = schedule[next_idx][1]
                cycle = cycle_dict.get(inter_id, 60)
                if next_idx == 0:
                    # Next switch is at the start of next cycle (cycle_time = 0)
                    # Set next_switch to cycle so we wait until cycle_time wraps around
                    intersection_next_switch_time[inter_id] = cycle
                else:
                    # Update relative switch time (within cycle)
                    intersection_next_switch_time[inter_id] = phase_start_time_relative

        # Note: For intersections NOT in selected_intersections, their next_switch_time
        # remains unchanged because they haven't reached their switch time yet.
        # The cycle_time has been updated based on current_time, so next apply_control
        # will correctly calculate remaining duration based on the updated cycle_time
        # and the unchanged next_switch_time.

        # Update control state with new cycle times
        control_state["intersection_cycle_time"] = intersection_cycle_time
        control_state["intersection_next_switch_time"] = intersection_next_switch_time

        return control_state

    def _initialize_control_state(
        self,
        env: Any,
        timing_dict: Dict[str, Dict[str, int]],
        cycle_dict: Dict[str, int],
        initial_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Initialize signal control state for all intersections.

        Args:
            env: SUMOEnv instance
            timing_dict: Dictionary mapping intersection_id to phase timings
            cycle_dict: Dictionary mapping intersection_id to cycle time
            initial_time: Initial simulation time when state is initialized

        Returns:
            Initial control state dictionary
        """
        # Convert phase names to phase indices
        timing_dict_code = {}
        for inter_id, phases in timing_dict.items():
            if inter_id not in env.inter_info_dict:
                print(f"Warning: Intersection {inter_id} not found in environment")
                continue

            control_phases = env.inter_info_dict[inter_id].get('control_phases', [])
            if not control_phases:
                print(f"Warning: No control phases for intersection {inter_id}")
                continue

            timing_dict_code[inter_id] = {}
            for phase_name, phase_duration in phases.items():
                if phase_name in control_phases:
                    phase_idx = control_phases.index(phase_name)
                    timing_dict_code[inter_id][phase_idx] = phase_duration
                else:
                    print(f"Warning: Phase {phase_name} not found in control_phases for {inter_id}")

        # Create phase schedule for direct phase switching (no yellow phases)
        # Schedule format: list of (phase_idx, relative_time, duration) tuples
        intersection_phase_schedule = {}

        for inter_id, phases in timing_dict_code.items():
            schedule = []
            cumulative_time = 0

            # Direct switching between green phases, no yellow transition
            for phase_idx, duration in phases.items():
                schedule.append((phase_idx, cumulative_time, duration))
                cumulative_time += duration

            intersection_phase_schedule[inter_id] = schedule

        # Initialize intersection state
        # intersection_phase_index tracks the NEXT phase to switch to (not the current phase)
        intersection_phase_index = {inter_id: 0 for inter_id in timing_dict_code}
        intersection_cycle_time = {inter_id: 0 for inter_id in timing_dict_code}
        # next_switch_time stores the relative time (within cycle) for next switch
        intersection_next_switch_time = {
            inter_id: (schedule[0][1] if schedule else 0)
            for inter_id, schedule in intersection_phase_schedule.items()
        }

        # Initialize current phases
        current_phases = {}
        for inter_id, schedule in intersection_phase_schedule.items():
            if schedule:
                phase_idx = schedule[0][0]
                control_phases = env.inter_info_dict[inter_id].get('control_phases', [])
                if phase_idx >= 0 and phase_idx < len(control_phases):
                    current_phases[inter_id] = control_phases[phase_idx]

        # Set initial phases using phase indices from schedule
        # Start with the first phase directly
        for inter_id, schedule in intersection_phase_schedule.items():
            if schedule:
                # Start with the first phase
                phase_idx = schedule[0][0]
                inter_obj = env.intersection_dict.get(inter_id)
                inter_obj.set_signal(phase_idx, action_pattern="set")
        
        return {
            "timing_dict_code": timing_dict_code,
            "intersection_phase_schedule": intersection_phase_schedule,
            "intersection_phase_index": intersection_phase_index,
            "intersection_cycle_time": intersection_cycle_time,
            "intersection_next_switch_time": intersection_next_switch_time,
            "current_phases": current_phases,
            "initial_time": initial_time  # Store initial time for relative time calculation
        }
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for traffic signal control.

        Returns:
            Dictionary with initialized metric structures
        """
        return {
            'total_reward': 0.0,
            'queue_length_episode': [],
            'waiting_time_episode': [],
            'global_waiting_times': [],
            'initial_arrived_count': None,  # Will be set when metrics are first updated
            'intersection_vehicle_ids': set()  # Track vehicles that have passed through intersections
        }
    
    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update training metrics with current step data for traffic signal control.
        Calculates metrics directly from system_states for consistency and efficiency.

        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            **kwargs: Additional arguments (e.g., step_duration) - not used by this module
        """
        if reward:
            metrics['total_reward'] += sum(reward)
        
        # Initialize initial_arrived_vehicle_ids on first update (for throughput calculation)
        if metrics.get('initial_arrived_vehicle_ids') is None:
            # Get initial arrived vehicle IDs at the start of this simulation
            initial_arrived_vehicle_ids = set()
            if hasattr(env, 'get_arrived_vehicle_travel_times'):
                initial_arrived_vehicle_ids = set(env.get_arrived_vehicle_travel_times().keys())
            elif hasattr(env, '_arrived_vehicle_tt'):
                initial_arrived_vehicle_ids = set(env._arrived_vehicle_tt.keys())
            metrics['initial_arrived_vehicle_ids'] = initial_arrived_vehicle_ids
        
        # Get data from system_states
        system_states = env.system_states
        vehicle_speeds = system_states.get("get_vehicle_speed", {})
        lane_vehicles = system_states.get("get_lane_vehicles", {})
        
        # Calculate queue lengths directly from system_states
        # Sum of queue lengths across all intersections (not average)
        total_queue_length = 0
        for intersection in env.intersection_dict.values():
            queue_length = 0
            # Count waiting vehicles (speed < 0.1 m/s) on incoming lanes
            for lane_id in intersection.list_entering_lanes:
                if lane_id is not None:
                    vehicles_on_lane = lane_vehicles.get(lane_id, [])
                    for vehicle_id in vehicles_on_lane:
                        speed = vehicle_speeds.get(vehicle_id, 0.0)
                        if speed < 0.1:
                            queue_length += 1
            total_queue_length += queue_length
        
        metrics['queue_length_episode'].append(float(total_queue_length))
        
        # Collect all intersection lanes (entering + exiting) for filtering
        intersection_lanes = set()
        for intersection in env.intersection_dict.values():
            for lane_id in intersection.list_entering_lanes:
                if lane_id is not None:
                    intersection_lanes.add(lane_id)
            for lane_id in intersection.list_exiting_lanes:
                if lane_id is not None:
                    intersection_lanes.add(lane_id)
        
        # Get waiting times from waiting_vehicle_list, but only for vehicles on intersection lanes
        # Handle both old format ({v_id: waiting_time}) and new format ({v_id: {"time": waiting_time, "lane": lane_id}})
        waiting_times = []
        for v_id, time_info in env.waiting_vehicle_list.items():
            # Check if vehicle is on an intersection lane
            is_on_intersection_lane = False
            if isinstance(time_info, dict):
                # New format: check lane field
                lane_id = time_info.get("lane")
                if lane_id in intersection_lanes:
                    is_on_intersection_lane = True
            else:
                # Old format: need to check current lane from system_states
                # Build vehicle -> lane mapping
                vehicle_to_lane = {}
                for lane_id, vehicle_list in lane_vehicles.items():
                    for vehicle_id in vehicle_list:
                        vehicle_to_lane[vehicle_id] = lane_id
                current_lane = vehicle_to_lane.get(v_id)
                if current_lane in intersection_lanes:
                    is_on_intersection_lane = True
            
            if is_on_intersection_lane:
                if isinstance(time_info, dict):
                    # New format: extract "time" field
                    waiting_times.append(time_info.get("time", 0.0))
                else:
                    # Old format: time_info is directly the waiting time
                    waiting_times.append(float(time_info))
                
                # Track this vehicle as having passed through an intersection
                metrics['intersection_vehicle_ids'].add(v_id)
        
        # Also track vehicles currently on intersection lanes (for travel time calculation)
        for lane_id in intersection_lanes:
            vehicles_on_lane = lane_vehicles.get(lane_id, [])
            for vehicle_id in vehicles_on_lane:
                metrics['intersection_vehicle_ids'].add(vehicle_id)
        
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        metrics['waiting_time_episode'].append(avg_waiting_time)
        metrics['global_waiting_times'].extend(waiting_times)
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any
    ) -> Dict[str, float]:
        """
        Calculate final training results and metrics for traffic signal control.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            
        Returns:
            Dictionary with final metric values
        """
        # Calculate average travel time only for vehicles that passed through intersections
        intersection_vehicle_ids = metrics.get('intersection_vehicle_ids', set())
        
        # Get travel times for all arrived vehicles
        arrived_vehicle_tt = {}
        if hasattr(env, 'get_arrived_vehicle_travel_times'):
            arrived_vehicle_tt = env.get_arrived_vehicle_travel_times()
        elif hasattr(env, '_arrived_vehicle_tt'):
            arrived_vehicle_tt = env._arrived_vehicle_tt
        
        # Filter to only include vehicles that passed through intersections
        intersection_travel_times = []
        for v_id, tt in arrived_vehicle_tt.items():
            if v_id in intersection_vehicle_ids:
                intersection_travel_times.append(tt)
        
        # Calculate average travel time for intersection vehicles only
        if intersection_travel_times:
            avg_travel_time = float(np.mean(intersection_travel_times))
        else:
            # Fallback: if no intersection vehicles arrived, return 0
            avg_travel_time = 0.0
        
        # Calculate throughput: number of intersection vehicles that arrived during this simulation
        initial_arrived_vehicle_ids = metrics.get('initial_arrived_vehicle_ids', set())
        final_arrived_vehicle_ids = set(arrived_vehicle_tt.keys())
        
        # Count vehicles that:
        # 1. Passed through intersections (in intersection_vehicle_ids)
        # 2. Arrived during this simulation (in final_arrived_vehicle_ids but not in initial_arrived_vehicle_ids)
        intersection_arrived_vehicles = intersection_vehicle_ids & final_arrived_vehicle_ids
        intersection_new_arrived_vehicles = intersection_arrived_vehicles - initial_arrived_vehicle_ids
        throughput = len(intersection_new_arrived_vehicles)
        
        return {
            "reward": float(metrics['total_reward']),
            "avg_queue_len": float(np.mean(metrics['queue_length_episode']) if metrics['queue_length_episode'] else 0),
            "queuing_vehicle": float(np.sum(metrics['queue_length_episode']) if metrics['queue_length_episode'] else 0),
            "avg_waiting_time": float(np.mean(metrics['waiting_time_episode']) if metrics['waiting_time_episode'] else 0),
            "avg_travel_time": float(avg_travel_time),
            "throughput": float(throughput)  # Number of vehicles arrived during this simulation
        }
