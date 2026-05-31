"""
Ramp metering control module.
Implements time-based ramp metering control (0/1 control: open/closed).
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base import ControlModule

# Default durations
DEFAULT_OPEN_DURATION = 100  # Default open duration in seconds
DEFAULT_CLOSE_DURATION = 0  # Default close duration (0 means always open)


class RampMeteringModule(ControlModule):
    """Control module for ramp metering (0/1 control: open/closed)."""
    
    DOMAIN_KNOWLEDGE = """Ramp metering control manages highway on-ramps using time-based 0/1 control (open/closed).

- **Configuration Format:**
  - Format: {"ramp_id": {"OPEN": <duration_seconds>, "CLOSE": <duration_seconds>}, ...}
  - OPEN: Duration in seconds that the ramp remains open (vehicles can enter)
  - CLOSE: Duration in seconds that the ramp remains closed (no vehicles can enter)
  - Default configuration: All ramps always open (OPEN: 600, CLOSE: 0)
  - Example: {"ramp_1": {"OPEN": 600, "CLOSE": 300}} means ramp_1 is open for 600s, then closed for 300s, repeating

- **Control Logic:**
  - Time-based periodic control: Each ramp alternates between OPEN and CLOSE states
  - Cycle time = OPEN duration + CLOSE duration
  - If CLOSE duration is 0, the ramp remains always open
  - State transitions occur automatically based on elapsed time within each cycle

- **Occupancy-Based Control Strategy:**
  Use ramp_lane_graph to analyze upstream and downstream (up to 2 hops) lane occupancy for intelligent control decisions.
  
  **Rule 1: Low Downstream Occupancy (< 0.8 vehicles/meter)**
  - Keep ramp open as much as possible (minimize CLOSE duration, maximize OPEN duration)
  - If upstream occupancy is high, consider extending OPEN duration but include some CLOSE time
  - Use predict_arima tool to forecast downstream occupancy trends - if prediction shows future congestion, proactively add CLOSE time even when current occupancy is low
  
  **Rule 2: High Downstream Occupancy (>= 0.8 vehicles/meter)**
  - Extend CLOSE duration to prevent further congestion
  - If upstream occupancy > downstream occupancy, CLOSE duration can be less than OPEN duration (to balance upstream queue while protecting downstream)
  - If upstream occupancy <= downstream occupancy, CLOSE duration should be >= OPEN duration (prioritize downstream protection)
  
  **How to Use ramp_lane_graph:**
  - For each ramp, find its controlled lanes: ramp_lane_graph.successors(ramp_id) where node_type='controlled_lane'
  - Find downstream lanes: For each controlled_lane, get ramp_lane_graph.successors(controlled_lane) where node_type='downstream_lane'
  - Find upstream lanes: For each controlled_lane, get ramp_lane_graph.predecessors(controlled_lane) where node_type='upstream_lane'
  - Calculate average occupancy across downstream lanes and upstream lanes from read_ramp_lane_traffic_states() data
  - Use these occupancy values to determine OPEN/CLOSE durations following Rules 1 and 2

- **Performance Metrics:**
  - Queue length at ramps: Shorter queues indicate better ramp control
  - Throughput: Higher throughput indicates better overall flow
  - Average travel time: Lower travel time indicates better traffic flow
  - Waiting time: Lower waiting time indicates less congestion

- **Optimization Time Window Considerations:**
  - When optimizing ramp metering, consider the optimization time window during which your configuration will be active
  - The optimized ramp metering will be applied for a specific duration (checkpoint interval)
  - Consider the time period characteristics when making optimization decisions:
    - **Rush Hours**: Morning (6:00-11:00) and Evening (16:00-21:00) periods have different traffic patterns than off-peak periods
    - During rush hours, traffic demand is higher and congestion is more likely
    - If the optimization window spans multiple time periods, consider the dominant period or design adaptive strategies
  - Adjust ramp OPEN/CLOSE durations based on expected traffic demand for the upcoming period:
    - Longer CLOSE durations can help reduce highway congestion during peak hours
    - Shorter CLOSE durations or always OPEN can improve ramp throughput during off-peak hours
  - Use historical data from similar time periods to inform your optimization decisions

- **Optimization Task Guidelines:**
  - Analyze historical ramp traffic data to identify congested ramps, queue lengths, waiting times, and throughput patterns
  - Consider the optimization time window when making optimization decisions
  - Optimize ramp metering OPEN/CLOSE durations to improve traffic flow for the upcoming period
  - Use available traffic snapshot data within the current simulation time range for analysis
  - Consider both current conditions and expected future conditions when optimizing"""
    
    def __init__(self, config_dir_name: Optional[str] = None):
        super().__init__("ramp_metering", "ramp_metering.json", config_dir_name=config_dir_name)
    
    def get_default_config(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initialize ramp metering configuration from SUMO environment.
        Creates default configuration: all ramps always open (OPEN: 600, CLOSE: 0).
        
        Args:
            env: SUMOEnv instance with initialized ramps
            
        Returns:
            Dictionary with default ramp metering configurations
            Format: {"ramp_id": {"OPEN": duration, "CLOSE": duration}, ...}
        """
        config = {}
        
        if env is None:
            print("Warning: Environment is None, cannot generate default configuration")
            return config
        
        if not hasattr(env, 'ramp_dict') or not env.ramp_dict:
            print("Warning: No ramps found in environment")
            return config
        
        # Default: all ramps always open
        for ramp_id in env.ramp_dict.keys():
            config[ramp_id] = {
                "OPEN": DEFAULT_OPEN_DURATION,
                "CLOSE": DEFAULT_CLOSE_DURATION
            }
        
        return config
    
    def validate_config(self, config: Dict[str, Any], reference_config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate ramp metering configuration.
        
        Args:
            config: Configuration dictionary to validate
                    Format: {"ramp_id": {"OPEN": duration, "CLOSE": duration}, ...}
            reference_config: Optional reference configuration to check completeness.
                            If provided, validates that config contains all ramps from reference_config.
            
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
        
        for ramp_id, ramp_config in config.items():
            # Check if ramp_config is a dictionary
            if not isinstance(ramp_config, dict):
                error_type = "not_dict"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(ramp_id)
                continue
            
            # Check for required keys
            if "OPEN" not in ramp_config:
                error_type = "missing_open"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(ramp_id)
            
            if "CLOSE" not in ramp_config:
                error_type = "missing_close"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(ramp_id)
            
            # Check if values are valid numbers
            if "OPEN" in ramp_config:
                open_duration = ramp_config["OPEN"]
                if not isinstance(open_duration, (int, float)) or open_duration < 0:
                    error_type = "invalid_open"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append(ramp_id)
            
            if "CLOSE" in ramp_config:
                close_duration = ramp_config["CLOSE"]
                if not isinstance(close_duration, (int, float)) or close_duration < 0:
                    error_type = "invalid_close"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append(ramp_id)
        
        # Check completeness against reference_config if provided
        if reference_config:
            missing_ramps = set(reference_config.keys()) - set(config.keys())
            if missing_ramps:
                error_type = "missing_ramps"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].extend(list(missing_ramps))
        
        # Format error messages
        if errors_by_type:
            error_parts = []
            for error_type, ramp_ids in errors_by_type.items():
                if error_type == "not_dict":
                    error_parts.append(f"Ramp configs must be dictionaries (e.g., {ramp_ids[0]})")
                elif error_type == "missing_open":
                    error_parts.append(f"Missing 'OPEN' key (e.g., {ramp_ids[0]})")
                elif error_type == "missing_close":
                    error_parts.append(f"Missing 'CLOSE' key (e.g., {ramp_ids[0]})")
                elif error_type == "invalid_open":
                    error_parts.append(f"Invalid 'OPEN' value (must be non-negative number, e.g., {ramp_ids[0]})")
                elif error_type == "invalid_close":
                    error_parts.append(f"Invalid 'CLOSE' value (must be non-negative number, e.g., {ramp_ids[0]})")
                elif error_type == "missing_ramps":
                    error_parts.append(f"Missing ramps: {', '.join(list(missing_ramps)[:5])}{'...' if len(missing_ramps) > 5 else ''}")
            
            return False, "; ".join(error_parts)
        
        return True, None
    
    def apply_control(
        self,
        env: Any,
        config: Dict[str, Any],
        current_time: float,
        control_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply ramp metering control based on time-based periodic schedule.
        
        This function calculates min_remaining based on current simulation time.
        It determines which ramps need state changes and calculates the minimum
        time until the next change is needed.
        
        IMPORTANT: When multiple control modules are used together, the actual step_duration
        may be smaller than this module's remaining duration (if other modules have shorter
        remaining durations). The update_control_state method handles this by recalculating
        state based on the actual current_time after step execution.
        
        Args:
            env: SUMOEnv instance
            config: Ramp metering configuration
                    Format: {"ramp_id": {"OPEN": duration, "CLOSE": duration}, ...}
            current_time: Current simulation time in seconds
            control_state: Optional control state dictionary to track ramp states
                          Format: {"ramp_id": {"state": "OPEN"/"CLOSE", "state_start_time": float, "cycle_start_time": float}, ...}
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dictionary containing:
                - actions: Dict mapping ramp_id to boolean (True=OPEN, False=CLOSE) for all ramps
                          Actions are always returned based on target state calculated from time,
                          regardless of current_state, to ensure signal light synchronization
                - control_state: Control state (not updated yet, will be updated after step)
                - min_remaining: Minimum time until next state change (based on current_time)
                - selected_ramps: List of all ramp_ids that have actions applied
        """
        if control_state is None:
            # Initialize control state for all ramps
            control_state = {}
            for ramp_id in config.keys():
                control_state[ramp_id] = {
                    "state": "NONE",
                    "state_start_time": current_time,
                    "cycle_start_time": current_time
                }
        
        # Determine which ramps need state changes based on current time
        actions = {}
        selected_ramps = []
        min_remaining = float('inf')
        
        # Process each ramp to determine if state change is needed
        for ramp_id, ramp_config in config.items():
            if ramp_id not in env.ramp_dict:
                continue
            
            open_duration = ramp_config.get("OPEN", DEFAULT_OPEN_DURATION)
            close_duration = ramp_config.get("CLOSE", DEFAULT_CLOSE_DURATION)
            
            # Get current state for this ramp
            if ramp_id not in control_state:
                control_state[ramp_id] = {
                    "state": "OPEN",
                    "state_start_time": current_time,
                    "cycle_start_time": current_time
                }
            
            # If CLOSE duration is 0, ramp is always open
            if close_duration == 0:
                # Always open - ensure action is added to keep signal light synchronized
                selected_ramps.append(ramp_id)
                actions[ramp_id] = True  # Always open
                # Set remaining to inf for this ramp (never needs to switch)
                remaining = float('inf')
                min_remaining = min(min_remaining, remaining)
                continue  # Skip the rest of the logic for this ramp
            
            ramp_state = control_state[ramp_id]
            current_state = ramp_state["state"]
            state_start_time = ramp_state["state_start_time"]
            cycle_start_time = ramp_state.get("cycle_start_time", state_start_time)
            
            # Calculate cycle time
            cycle_time = open_duration + close_duration
            
            # Calculate elapsed time since cycle start
            elapsed_since_cycle_start = current_time - cycle_start_time
            
            # Calculate which cycle we're in and time within that cycle
            cycles_completed = int(elapsed_since_cycle_start / cycle_time)
            time_in_current_cycle = elapsed_since_cycle_start % cycle_time
            
            # Determine target state based on time in current cycle
            if time_in_current_cycle < open_duration:
                target_state = "OPEN"
                target_state_start_time = cycle_start_time + cycles_completed * cycle_time
                next_change_time = cycle_start_time + cycles_completed * cycle_time + open_duration
            else:
                target_state = "CLOSE"
                target_state_start_time = cycle_start_time + cycles_completed * cycle_time + open_duration
                next_change_time = cycle_start_time + (cycles_completed + 1) * cycle_time
            
            # IMPORTANT: Always add action based on target state (calculated from time)
            # This ensures SUMO signal light state is always synchronized with target state,
            # regardless of what current_state says. current_state is only for tracking,
            # not for determining whether to apply action.
            # This fixes the issue where update_control_state updates current_state,
            # but SUMO signal light might not have changed yet.
            selected_ramps.append(ramp_id)
            is_open = (target_state == "OPEN")
            actions[ramp_id] = is_open
            
            # Calculate remaining time until next state change
            remaining = next_change_time - current_time
            if remaining < 0:
                # Should not happen, but handle wrap-around
                remaining = cycle_time - time_in_current_cycle + (open_duration if target_state == "OPEN" else close_duration)
            
            min_remaining = min(min_remaining, remaining)
        
        # Return actions and state (state will be updated after step)
        return {
            "actions": actions,
            "control_state": control_state,
            "min_remaining": min_remaining,
            "selected_ramps": selected_ramps
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
        Pre-compute all ramp state switch events within a time range.

        This method enables simulation acceleration by pre-calculating all
        ramp state transitions (OPEN/CLOSE) that will occur in the specified
        time window. Instead of checking each second, the simulation can step
        directly to event times.

        Args:
            env: SUMOEnv instance
            config: Ramp metering configuration
                    Format: {"ramp_id": {"OPEN": duration, "CLOSE": duration}, ...}
            start_time: Start of time window (current simulation time)
            end_time: End of time window
            control_state: Current control state (will be initialized if None)

        Returns:
            Tuple of (events, control_state):
                - events: List of ControlEvent objects for ramp state switches
                - control_state: Updated control state
        """
        from control_modules.shared.event_scheduler import ControlEvent

        events = []

        # Initialize control state if not provided
        if control_state is None:
            control_state = {}
            for ramp_id in config.keys():
                control_state[ramp_id] = {
                    "state": "OPEN",
                    "state_start_time": start_time,
                    "cycle_start_time": start_time
                }

        # Generate events for each ramp
        for ramp_id, ramp_config in config.items():
            if not hasattr(env, 'ramp_dict') or ramp_id not in env.ramp_dict:
                continue

            open_duration = ramp_config.get("OPEN", DEFAULT_OPEN_DURATION)
            close_duration = ramp_config.get("CLOSE", DEFAULT_CLOSE_DURATION)

            # If CLOSE duration is 0, ramp is always open - no events needed
            if close_duration == 0:
                continue

            cycle_time = open_duration + close_duration
            if cycle_time <= 0:
                continue

            # Get or initialize cycle start time for this ramp
            if ramp_id not in control_state:
                control_state[ramp_id] = {
                    "state": "OPEN",
                    "state_start_time": start_time,
                    "cycle_start_time": start_time
                }

            cycle_start_time = control_state[ramp_id].get("cycle_start_time", start_time)

            # Calculate which cycle we're starting in
            elapsed_since_cycle_start = start_time - cycle_start_time
            cycles_completed = int(elapsed_since_cycle_start / cycle_time)
            current_cycle_start = cycle_start_time + cycles_completed * cycle_time

            # Generate events for current and future cycles within the window
            while current_cycle_start < end_time + cycle_time:
                # OPEN -> CLOSE transition
                close_switch_time = current_cycle_start + open_duration
                if start_time < close_switch_time <= end_time:
                    events.append(ControlEvent(
                        time=close_switch_time,
                        module="ramp_metering",
                        entity_id=ramp_id,
                        action=False,  # False = CLOSE
                        priority=1,  # Ramp metering has lower priority than signals
                        metadata={"state": "CLOSE"}
                    ))

                # CLOSE -> OPEN transition
                open_switch_time = current_cycle_start + cycle_time
                if start_time < open_switch_time <= end_time:
                    events.append(ControlEvent(
                        time=open_switch_time,
                        module="ramp_metering",
                        entity_id=ramp_id,
                        action=True,  # True = OPEN
                        priority=1,
                        metadata={"state": "OPEN"}
                    ))

                current_cycle_start += cycle_time

        return events, control_state

    def update_control_state(
        self,
        control_state: Dict[str, Any],
        step_duration: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update ramp metering control state after a simulation step.
        For ramps that had state changes applied, updates state, state_start_time, and cycle_start_time.
        
        IMPORTANT: This method recalculates state based on current_time (after step execution),
        which accounts for the actual step_duration that was executed (which may be smaller
        than this module's remaining duration if other modules had shorter remaining durations).
        
        Args:
            control_state: Current control state
                          Format: {"ramp_id": {"state": "OPEN"/"CLOSE", "state_start_time": float, "cycle_start_time": float}, ...}
            step_duration: Duration of the simulation step (may be smaller than this module's remaining duration)
            **kwargs: Additional arguments:
                - env: SUMOEnv instance (optional, for validation)
                - current_time: Current simulation time after step (required)
                - applied_actions: Dictionary of actions that were applied (ramp_id -> is_open (bool))
                - config: Ramp metering configuration (required for cycle_start_time updates)
            
        Returns:
            Updated control state
        """
        current_time = kwargs.get("current_time")
        applied_actions = kwargs.get("applied_actions", {})
        config = kwargs.get("config", {})
        
        if current_time is None:
            # If current_time not provided, just return state as-is
            return control_state
        
        # IMPORTANT: Recalculate state for all ramps based on current_time (after step execution)
        # This accounts for the actual step_duration that was executed
        for ramp_id, ramp_config in config.items():
            if ramp_id not in control_state:
                continue
            
            open_duration = ramp_config.get("OPEN", DEFAULT_OPEN_DURATION)
            close_duration = ramp_config.get("CLOSE", DEFAULT_CLOSE_DURATION)

            # Skip if always open (close_duration == 0)
            if close_duration == 0:
                continue
            
            cycle_time = open_duration + close_duration
            
            # Get or initialize cycle_start_time
            cycle_start_time = control_state[ramp_id].get("cycle_start_time", current_time)
            
            # Recalculate state based on current_time
            elapsed_since_cycle_start = current_time - cycle_start_time
            cycles_completed = int(elapsed_since_cycle_start / cycle_time)
            time_in_current_cycle = elapsed_since_cycle_start % cycle_time
            
            # Determine target state based on time in current cycle
            if time_in_current_cycle < open_duration:
                target_state = "OPEN"
                target_state_start_time = cycle_start_time + cycles_completed * cycle_time
            else:
                target_state = "CLOSE"
                target_state_start_time = cycle_start_time + cycles_completed * cycle_time + open_duration
            
            # Update state and times based on recalculated values
            control_state[ramp_id]["state"] = target_state
            control_state[ramp_id]["state_start_time"] = target_state_start_time
            
            # Update cycle_start_time to the start of the current cycle
            current_cycle_start = cycle_start_time + cycles_completed * cycle_time
            control_state[ramp_id]["cycle_start_time"] = current_cycle_start
        
        return control_state
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for tracking performance.

        Returns:
            Dictionary with initialized metric structures
        """
        return {
            'total_reward': 0.0,
            'queue_length_episode': [],
            'waiting_time_episode': [],
            'travel_time_episode': [],
            'initial_arrived_count': None,  # Will be set when metrics are first updated (for throughput calculation)
            'ramp_vehicle_ids': set()  # Track vehicles that have passed through ramp segments
        }
    
    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update training metrics with current step data.
        
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
        
        # Collect ramp-level metrics
        queue_lengths = []
        waiting_times = []
        
        if env and hasattr(env, 'ramp_dict'):
            for ramp_id, ramp_obj in env.ramp_dict.items():
                features = ramp_obj.get_feature()
                
                # Collect queue length
                ramp_queue_length = features.get('ramp_queue_length', 0)
                if ramp_queue_length > 0:
                    queue_lengths.append(ramp_queue_length)
                
                # Collect waiting time
                ramp_waiting_time = features.get('ramp_waiting_time', 0.0)
                if ramp_waiting_time > 0:
                    waiting_times.append(ramp_waiting_time)
        
        # Store aggregated metrics
        if queue_lengths:
            metrics['queue_length_episode'].append(np.mean(queue_lengths))
        
        if waiting_times:
            metrics['waiting_time_episode'].append(np.mean(waiting_times))
        
        # Track vehicles currently on ramp lanes (for travel time calculation)
        if env and hasattr(env, 'ramp_dict'):
            # Collect all ramp lanes from all ramps
            ramp_lanes = set()
            for ramp_obj in env.ramp_dict.values():
                if hasattr(ramp_obj, 'ramp_lanes') and ramp_obj.ramp_lanes:
                    ramp_lanes.update(ramp_obj.ramp_lanes)
            
            # Track vehicles on ramp lanes
            lane_vehicles_dict = env.system_states.get("get_lane_vehicles", {})
            for lane_id in ramp_lanes:
                vehicle_list = lane_vehicles_dict.get(lane_id, [])
                for vehicle_id in vehicle_list:
                    metrics['ramp_vehicle_ids'].add(vehicle_id)
        
        # Travel times are collected from completed vehicles via env.get_average_travel_time()
        # This is updated in _update_system_states() in sumo_env.py
        # Throughput is now calculated as number of vehicles arrived during simulation (in calculate_final_results)
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any
    ) -> Dict[str, float]:
        """
        Calculate final training results and metrics for ramp metering control.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            
        Returns:
            Dictionary with final metric values
        """
        # Calculate average travel time only for vehicles that passed through ramp segments
        ramp_vehicle_ids = metrics.get('ramp_vehicle_ids', set())
        
        # Get travel times for all arrived vehicles
        arrived_vehicle_tt = {}
        if hasattr(env, 'get_arrived_vehicle_travel_times'):
            arrived_vehicle_tt = env.get_arrived_vehicle_travel_times()
        elif hasattr(env, '_arrived_vehicle_tt'):
            arrived_vehicle_tt = env._arrived_vehicle_tt
        
        # Filter to only include vehicles that passed through ramp segments in this episode
        ramp_travel_times = []
        for v_id, tt in arrived_vehicle_tt.items():
            if v_id in ramp_vehicle_ids:
                ramp_travel_times.append(tt)
        
        # Calculate average travel time for ramp vehicles only
        if ramp_travel_times:
            avg_travel_time = float(np.mean(ramp_travel_times))
        else:
            # Fallback: if no ramp vehicles arrived in this episode, return 0
            avg_travel_time = 0.0
        
        # Calculate throughput: number of ramp vehicles that arrived during this simulation
        initial_arrived_vehicle_ids = metrics.get('initial_arrived_vehicle_ids', set())
        final_arrived_vehicle_ids = set(arrived_vehicle_tt.keys())
        
        # Count vehicles that:
        # 1. Passed through ramps (in ramp_vehicle_ids)
        # 2. Arrived during this simulation (in final_arrived_vehicle_ids but not in initial_arrived_vehicle_ids)
        ramp_arrived_vehicles = ramp_vehicle_ids & final_arrived_vehicle_ids
        ramp_new_arrived_vehicles = ramp_arrived_vehicles - initial_arrived_vehicle_ids
        throughput = len(ramp_new_arrived_vehicles)
        
        return {
            "reward": float(metrics['total_reward']),
            "avg_queue_len": float(np.mean(metrics['queue_length_episode']) if metrics['queue_length_episode'] else 0),
            "queuing_vehicle": float(np.sum(metrics['queue_length_episode']) if metrics['queue_length_episode'] else 0),
            "avg_waiting_time": float(np.mean(metrics['waiting_time_episode']) if metrics['waiting_time_episode'] else 0),
            "avg_travel_time": float(avg_travel_time),
            "throughput": float(throughput)  # Number of vehicles arrived during this simulation
        }
