"""
Highway speed limit control module.
Implements variable speed limit control for highway segments to alleviate congestion.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base import ControlModule

# Speed limit constants (mph to m/s conversion: 1 mph = 0.44704 m/s)
MPH_TO_MPS = 0.44704
MIN_SPEED_LIMIT_MPH = 5
MAX_SPEED_LIMIT_MPH = 65
SPEED_LIMIT_INCREMENT_MPH = 5

# Available speed limits in mph
AVAILABLE_SPEED_LIMITS_MPH = list(range(MIN_SPEED_LIMIT_MPH, MAX_SPEED_LIMIT_MPH + 1, SPEED_LIMIT_INCREMENT_MPH))
# Convert to m/s
AVAILABLE_SPEED_LIMITS_MPS = [mph * MPH_TO_MPS for mph in AVAILABLE_SPEED_LIMITS_MPH]


class HighwaySpeedLimitModule(ControlModule):
    """Control module for highway speed limit control."""

    DOMAIN_KNOWLEDGE = """Highway speed limit control uses Variable Speed Limits (VSL) to dynamically adjust speed limits on highway segments. The core principle of VSL is **Capacity Drop Prevention**: when traffic breaks down at a bottleneck, the discharge flow rate drops 5-15% below the pre-breakdown capacity. By moderately reducing upstream speed limits (10-15 mph reduction), VSL prevents this breakdown, maintaining BOTH higher throughput AND lower travel time simultaneously.

    - **CRITICAL: Dual-Objective Optimization**
      - **Both travel time AND throughput matter equally.** A strategy that improves travel time but destroys throughput is NOT acceptable, and vice versa.
      - A strategy that improves one but significantly worsens the other is NOT acceptable.
      - The ideal VSL strategy simultaneously: (1) prevents capacity drop at bottlenecks, (2) maintains high discharge flow rate, (3) reduces travel time by smoothing traffic flow.

    - **Optimization Time Window:**
      - The system will inform you of the optimization time window (start time, end time, duration)
      - Your optimized speed limits will be applied from the start time to the end time of the optimization window
      - Consider the optimization time window when making optimization decisions
      - If the optimization window spans multiple time periods, consider the dominant period or design adaptive strategies
      - Adjust speed limits based on expected traffic demand for the upcoming period within the optimization window

    - **CRITICAL: Decision Gate - Read This First!**
      - BEFORE making ANY speed limit changes, you MUST first determine if intervention is needed:
        1. **Check for Congestion Indicators:**
           - Any segment with speed_ratio < 0.6? (vehicles moving much slower than limit)
           - Any segment with segment_occupancy > 0.15? (high vehicle density)
           - Any segment with segment_congestion_ratio > 0.3? (many congested roads)
        2. **If NO congestion detected (all indicators are good):**
           → DO NOT reduce any speed limits
           → Set ALL segments to maximum speed limit (65 mph)
           → This maximizes throughput
        3. **If congestion IS detected:**
           → Identify the bottleneck segment(s)
           → Apply speed reduction ONLY to upstream segments (1-2 segments before bottleneck)
           → NEVER reduce speed below 30 mph - this destroys throughput
           → DO NOT reduce speed at the bottleneck itself
           → Set downstream segments to 65 mph to create "pull" effect and increase discharge rate
      - Unnecessary speed reduction in free-flow conditions DESTROYS throughput with zero benefit.

    - **CRITICAL: Minimum Speed Floor - NEVER Below 30 mph:**
      - Speed limits below 30 mph on a highway destroy throughput (flow = density × speed).
      - Even in severe congestion, the lowest acceptable speed limit is 30 mph.
      - Speed limits of 5, 10, 15, 20, 25 mph are NEVER appropriate for highway VSL control.
      - Typical effective VSL range: 40-65 mph. Aggressive but safe minimum: 30 mph.

    - **Default Action When Traffic is Flowing Well:**
      - If average speed_ratio > 0.8 and average occupancy < 0.10 across all segments:
        → Maintain or increase speed limits to 65 mph
        → Speed reduction in good conditions REDUCES throughput without benefit

    - **Speed Limit Constraints:**
      - Speed limits are discrete values with 5 mph increments
      - Available speed limits: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65 mph
      - Maximum speed limit is 65 mph for the study freeway
      - **Practical range for VSL: 30-65 mph** (never use values below 30)
      - You MUST use only these discrete values - do not use intermediate values like 52 or 58 mph

    - **Configuration Format:**
      - Format: {"highway_segment_id": [{"time": <TIME_IN_OPTIMIZATION_TIME_WINDOW>, "speed_limit": <SPEED_IN_MPH>}, ...], ...}
      - Each segment can have multiple scheduled speed limit changes within the optimization time window
      - TIME_IN_OPTIMIZATION_TIME_WINDOW: Time in seconds from start of optimization time window (e.g., if time window is 1800s, valid range is 0-1800)
      - SPEED_IN_MPH: Speed limit in mph (must be one of the discrete values, minimum 30 for VSL)
      - Example: {"highway_segment_0": [{"time": 0, "speed_limit": 55}, {"time": 600, "speed_limit": 50}], "highway_segment_1": [{"time": 0, "speed_limit": 60}]}
      - This allows proactive speed limit adjustments based on predicted traffic patterns

    - **Bottleneck Identification and Upstream Control Strategy:**
      - **Step 1: Identify Traffic Bottlenecks**
        - Use DATA_ANALYSIS to examine highway traffic conditions (speed, density, throughput, occupancy, congestion indicators)
        - **Key bottleneck indicators:**
          - Low segment_speed compared to speed_limit (speed_ratio < 0.6 indicates severe congestion)
          - High segment_occupancy (> 0.15 vehicles/meter indicates congestion)
          - Low throughput despite high density (indicates capacity breakdown)
          - segment_congestion_ratio > 0.5 (more than half of roads in segment are congested)
        - **Use cache to store analysis results**: In DATA_ANALYSIS, use `save_cache(dict)` to store bottleneck locations and characteristics
          - Format: `save_cache({"bottleneck_segments": {"value": [...], "description": "List of congested segment IDs"}})`
          - Example: `save_cache({"bottleneck_analysis": {"value": {"highway_segment_2": {"speed_ratio": 0.45, "occupancy": 0.18}}, "description": "Bottleneck characteristics"}})`
        - Use `list_cache()` to see available cached analysis results

      - **Step 2: Use ARIMA Prediction for Proactive Control**
        - Use `read_highway_traffic_states()` to extract historical traffic data
        - Extract time series values for the metric you want to predict (e.g., segment_occupancy, segment_speed)
        - Use the generic `predict_arima()` function to forecast future values:
        - Convert forecast horizon (seconds) to time steps based on data sampling rate
        - Pass time series values, history_window (time steps), prediction_window (time steps), and forecast_interval
      - Based on predictions, schedule multiple speed limit changes within the optimization time window:
        - If congestion predicted to worsen: Apply early speed reduction (time: 0-300s)
        - If congestion predicted to ease: Schedule gradual speed limit increase (time: 900-1800s)
        - If congestion predicted to remain stable: Maintain current strategy

      - **Step 3: Identify Upstream Non-Congested Segments and Apply Speed Reduction**
        - Use highway_segment_graph to find upstream segments (predecessors) of bottleneck segments
        - Look for upstream segments with:
          - High speed_ratio (> 0.8) - vehicles moving close to speed limit
          - Low to moderate occupancy (< 0.10 vehicles/meter)
          - segment_congestion_ratio < 0.3
        - **Apply moderate speed reduction on upstream segments**: 10-15 mph below current limit (e.g., 65 → 50 or 55 mph, never below 30 mph)
        - **Gradual spatial reduction**: Create a "speed funnel" approaching the bottleneck
          - Example: Segment 3 upstream (60 mph) → Segment 2 upstream (50 mph) → Segment 1 bottleneck (keep current/65 mph)
        - **Cache upstream analysis**: `save_cache({"upstream_control_segments": {"value": [...], "description": "Segments to apply speed reduction"}})`
        - **In POLICY_PLANNING, use `load_cache(key)` to retrieve cached analysis** instead of re-analyzing

      - **Step 4: Bottleneck and Downstream Speed Management**
        - **At the bottleneck segment**: DO NOT reduce speed limits - vehicles are already slow, reducing the limit further only hurts throughput
        - **Downstream of the bottleneck**: Set speed limits to MAXIMUM (65 mph) to create a "pull" effect
          - Higher downstream speed limits help vehicles clear the bottleneck faster, increasing discharge flow rate
          - This is critical for maintaining throughput through the bottleneck
        - **Summary**: Reduce upstream → Keep/maximize bottleneck → Maximize downstream

      - **Step 5: Speed Harmonization**
        - Adjacent segments should NOT have speed limit differences greater than 10-15 mph
        - Large speed differentials between adjacent segments create secondary shockwaves that worsen congestion
        - Create smooth speed transitions: e.g., 65 → 55 → 45 (good) vs 65 → 30 (bad)

      - **Step 6: Speed Limit Increase for High Speed Ratio Conditions**
        - **When to consider speed limit increase**: If all or most segments have high speed_ratio (> 0.85) and low congestion indicators
        - **Conditions for speed increase**:
          - Average speed_ratio across all segments > 0.85
          - Low segment_occupancy (< 0.10 vehicles/meter) across most segments
          - segment_congestion_ratio < 0.2
          - Current speed limits are below maximum (65 mph)
        - **Speed increase strategy**:
          - Gradually increase speed limits by 5-10 mph increments
          - Use ARIMA predictions to ensure conditions will remain favorable

    - **Common Pitfalls That Destroy Throughput (AVOID THESE):**
      1. Setting speed limits below 30 mph - catastrophic for throughput
      2. Reducing speed at the bottleneck itself - vehicles are already slow, this only reduces discharge rate
      3. Reducing speed when there is NO congestion - pure throughput loss with no benefit
      4. Reducing speed on ALL segments instead of only upstream of bottleneck
      5. Not setting downstream segments to maximum (65 mph) - misses the "pull" effect
      6. Speed differences > 15 mph between adjacent segments - creates secondary shockwaves

    - **Performance Metrics:**
      - Average travel time: Lower is better (EQUALLY IMPORTANT)
      - Average throughput: Higher is better - vehicles completed (EQUALLY IMPORTANT)
      - Average speed: Higher average speed indicates better flow
      - Road occupancy: Lower occupancy indicates less congestion
      - Congestion ratio: Lower congestion ratio indicates better conditions
      - Speed ratio (current_speed / speed_limit): Should be close to 1.0 for optimal flow

    - **Common Patterns:**
      - Rush hours: morning: 6:00-11:00, evening: 16:00-21:00
      - Bottlenecks typically form at:
        - Lane drops (highway narrows from 4 lanes to 3 lanes)
        - Uphill grades (vehicles slow down)
      - Upstream speed control is most effective 1-2 segments before the bottleneck
      - Recovery from congestion: Gradually increase speed limits as bottleneck clears (use ARIMA predictions)

    - **Time-Aware Optimization:**
      - The system will inform you of the current city time and optimization time window duration
      - Use time information and ARIMA predictions to schedule proactive speed limit changes
      - **Important**: Consider the optimization time window when optimizing:
        - Your optimized speed limits will be active during the specified optimization time window
        - During rush hours (Morning: 6:00-11:00, Evening: 16:00-21:00), traffic patterns are different from off-peak periods
        - If the optimization window spans multiple time periods, consider the dominant period or design adaptive strategies
        - Adjust speed limits based on expected traffic demand for the upcoming period
        - Use historical data from similar time periods to inform your optimization decisions
        - Lower speed limits can help prevent capacity drop ONLY during peak hours with actual congestion
        - Higher speed limits maximize throughput - use 65 mph when traffic is flowing well

    - **Task Description:**
      - Analyze historical highway traffic data to check for congestion indicators (speed_ratio, occupancy, congestion_ratio)
      - **FIRST DECISION**: Determine if ANY congestion exists - if not, set all limits to 65 mph
      - If congestion exists: reduce upstream segments to 50-55 mph (never below 30 mph), keep bottleneck and downstream at 65 mph
      - Consider the optimization time window when making optimization decisions
      - Optimize highway speed limits using POLICY_PLANNING action
      - Note: Simulation is automatically executed after POLICY_PLANNING - you will receive results and comparison automatically
      - If throughput decreased compared to best, your speed limits may be too aggressive - raise them closer to 65 mph
      - Complete with FINISH action when satisfied with BOTH travel time and throughput"""
    
    def __init__(self, config_dir_name: Optional[str] = None):
        super().__init__("highway_speed_limit", "highway_speed_limit.json", config_dir_name=config_dir_name)
    
    def get_default_config(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate default configuration for highway speed limit control.
        Creates default speed limits for all highway segments.
        
        Args:
            env: SUMOEnv instance with initialized highways
            
        Returns:
            Dictionary with default speed limit configurations
            Format: {"highway_segment_id": [{"time": 0, "speed_limit": 65}], ...}
        """
        config = {}
        
        if env is None:
            print("Warning: Environment is None, cannot generate default configuration")
            return config
        
        if not env.highway_dict:
            print("Warning: No highway segments found in environment")
            return config
        
        # Default speed limit: 65 mph (maximum)
        default_speed_limit_mph = MAX_SPEED_LIMIT_MPH
        default_speed_limit_mps = default_speed_limit_mph * MPH_TO_MPS
        
        for highway_id, highway_obj in env.highway_dict.items():
            # Get default speed from segment's default speed limit
            # Use segment-level default speed limit if available
            features = highway_obj.get_feature()
            segment_default_speed_limit = features.get('segment_default_speed_limit', 0.0)
            
            if segment_default_speed_limit > 0:
                # Convert to mph and round to nearest available speed limit
                default_speed_mph = segment_default_speed_limit / MPH_TO_MPS
                closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - default_speed_mph))
            else:
                # Fallback: use default from first road in segment
                if highway_obj.highway_road_ids:
                    first_road_id = highway_obj.highway_road_ids[0]
                    road_info = highway_obj.road_dict.get(first_road_id, {})
                    default_speed = road_info.get("max_speed", default_speed_limit_mps)
                    default_speed_mph = default_speed / MPH_TO_MPS
                    closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - default_speed_mph))
                else:
                    closest_mph = default_speed_limit_mph
            
            # New format: list of scheduled speed limit changes
            # Default: apply speed limit at time 0
            config[highway_id] = [{"time": 0, "speed_limit": closest_mph}]
            
            print(f"Initialized default speed limit for {highway_id}: {closest_mph} mph")
        
        return config
    
    def validate_config(self, config: Dict[str, Any], reference_config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate highway speed limit configuration.
        
        Args:
            config: Configuration dictionary to validate
                    Format: {"highway_segment_id": [{"time": <TIME_IN_OPTIMIZATION_TIME_WINDOW>, "speed_limit": <SPEED_IN_MPH>}, ...], ...}
            reference_config: Optional reference configuration to check completeness.
                            If provided, validates that config contains all highway segments from reference_config.
            
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
        
        for highway_id, schedule in config.items():
            # Check if schedule is a list
            if not isinstance(schedule, list):
                error_type = "schedule_not_list"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(highway_id)
                continue
            
            if len(schedule) == 0:
                error_type = "schedule_empty"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(highway_id)
                continue
            
            # Validate each scheduled speed limit change
            for idx, entry in enumerate(schedule):
                if not isinstance(entry, dict):
                    error_type = "schedule_entry_not_dict"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx))
                    continue
                
                # Check required fields
                if "time" not in entry:
                    error_type = "missing_time_field"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx))
                    continue
                
                if "speed_limit" not in entry:
                    error_type = "missing_speed_limit_field"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx))
                    continue
                
                # Validate time
                time_value = entry["time"]
                if not isinstance(time_value, (int, float)):
                    error_type = "time_not_number"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx, time_value))
                    continue
                
                if time_value < 0:
                    error_type = "time_negative"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx, time_value))
                    continue
                
                # Validate speed_limit
                speed_limit_mph = entry["speed_limit"]
                if not isinstance(speed_limit_mph, (int, float)):
                    error_type = "speed_limit_not_number"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx))
                    continue
                
                if speed_limit_mph < MIN_SPEED_LIMIT_MPH or speed_limit_mph > MAX_SPEED_LIMIT_MPH:
                    error_type = "speed_limit_out_of_range"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx, speed_limit_mph))
                    continue
                
                if speed_limit_mph not in AVAILABLE_SPEED_LIMITS_MPH:
                    error_type = "speed_limit_not_discrete"
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append((highway_id, idx, speed_limit_mph))
                    continue
        
        # Check for missing/unknown highway segments if reference_config is provided
        if reference_config is not None:
            reference_keys = set(reference_config.keys())
            config_keys = set(config.keys())
            missing_segments = reference_keys - config_keys
            if missing_segments:
                error_type = "missing_highway_segments"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].extend(sorted(missing_segments))
            extra_segments = config_keys - reference_keys
            if extra_segments:
                error_type = "unknown_highway_segments"
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].extend(sorted(extra_segments))
        
        # If no errors, return valid
        if not errors_by_type:
            return True, None
        
        # Build error message with examples only
        error_messages = []
        for error_type, error_list in errors_by_type.items():
            count = len(error_list)
            example = error_list[0]
            
            if error_type == "schedule_not_list":
                error_messages.append(
                    f"Schedule must be a list (e.g., highway segment '{example}'). "
                    f"Found {count} segment(s) with this issue."
                )
            elif error_type == "schedule_empty":
                error_messages.append(
                    f"Schedule cannot be empty (e.g., highway segment '{example}'). "
                    f"Found {count} segment(s) with this issue."
                )
            elif error_type == "schedule_entry_not_dict":
                highway_id, idx = example
                error_messages.append(
                    f"Schedule entry must be a dictionary (e.g., segment '{highway_id}', entry {idx}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "missing_time_field":
                highway_id, idx = example
                error_messages.append(
                    f"Schedule entry missing 'time' field (e.g., segment '{highway_id}', entry {idx}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "missing_speed_limit_field":
                highway_id, idx = example
                error_messages.append(
                    f"Schedule entry missing 'speed_limit' field (e.g., segment '{highway_id}', entry {idx}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "time_not_number":
                highway_id, idx, time_value = example
                error_messages.append(
                    f"Time must be a number (e.g., segment '{highway_id}', entry {idx}, time={time_value}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "time_negative":
                highway_id, idx, time_value = example
                error_messages.append(
                    f"Time must be non-negative (e.g., segment '{highway_id}', entry {idx}, time={time_value}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "speed_limit_not_number":
                highway_id, idx = example
                error_messages.append(
                    f"Speed limit must be a number (e.g., segment '{highway_id}', entry {idx}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "speed_limit_out_of_range":
                highway_id, idx, speed_limit_mph = example
                error_messages.append(
                    f"Speed limit must be between {MIN_SPEED_LIMIT_MPH} and {MAX_SPEED_LIMIT_MPH} mph "
                    f"(e.g., segment '{highway_id}', entry {idx}, speed_limit={speed_limit_mph}). "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "speed_limit_not_discrete":
                highway_id, idx, speed_limit_mph = example
                error_messages.append(
                    f"Speed limit must be a multiple of {SPEED_LIMIT_INCREMENT_MPH} mph "
                    f"(e.g., segment '{highway_id}', entry {idx}, speed_limit={speed_limit_mph}). "
                    f"Valid values: {AVAILABLE_SPEED_LIMITS_MPH}. "
                    f"Found {count} entry(ies) with this issue."
                )
            elif error_type == "missing_highway_segments":
                error_messages.append(
                    f"Missing required highway segments: {example}. "
                    f"Found {count} missing segment(s)."
                )
            elif error_type == "unknown_highway_segments":
                error_messages.append(
                    f"Unknown highway segments: {example}. "
                    f"Found {count} unknown segment(s)."
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
        Apply highway speed limit control logic.
        
        Only applies speed limits when current_time reaches or exceeds the scheduled time.
        Tracks which speed limits have been applied to avoid duplicate applications.
        
        IMPORTANT: When multiple control modules are used together, the actual step_duration
        may be smaller than this module's remaining duration (if other modules have shorter
        remaining durations). The update_control_state method handles this by recalculating
        next_switch_times based on the actual current_time after step execution.
        
        Args:
            env: SUMOEnv instance
            config: Speed limit configuration
                    Format: {"highway_segment_id": [{"time": <TIME_IN_NEXT_CHECKPOINT_CYCLE>, "speed_limit": <SPEED_IN_MPH>}, ...], ...}
            current_time: Current simulation time
            control_state: Current control state (maintained across steps)
                          If None, will initialize state
            **kwargs: Additional arguments (checkpoint_start_time for converting relative times to absolute times)
            
        Returns:
            Dictionary containing:
                - control_state: Updated control state
                - actions: Dictionary mapping highway_id to speed_limit_mph (only for segments that need to switch now)
                - min_remaining: Time until next scheduled speed limit change
        """
        # Initialize control state if not provided
        if control_state is None:
            control_state = self._initialize_control_state(env, config, current_time)
        
        # Get checkpoint start time (when this config becomes active)
        checkpoint_start_time = kwargs.get("checkpoint_start_time", current_time)

        # Filter invalid segments and warn only once per missing segment
        warned_missing = set(control_state.get("missing_segments_warned", []))
        filtered_config = {}
        for highway_id, schedule in config.items():
            if highway_id in env.highway_dict:
                filtered_config[highway_id] = schedule
            else:
                if highway_id not in warned_missing:
                    print(f"Warning: Highway segment '{highway_id}' not found in environment")
                    warned_missing.add(highway_id)
        config = filtered_config
        
        # Update checkpoint_start_time if this is a new config OR if checkpoint_start_time changed
        # CRITICAL: Even if config is unchanged, we must reinitialize next_switch_times when
        # checkpoint_start_time changes (e.g., after reset_metrics() at new checkpoint interval).
        # This ensures speed limits are reapplied after reset_metrics() rebuilds highway_dict.
        is_new_config = control_state.get("current_config") != config
        is_new_checkpoint = control_state.get("checkpoint_start_time") != checkpoint_start_time
        
        if "checkpoint_start_time" not in control_state or is_new_config or is_new_checkpoint:
            if is_new_config:
                print(f"[HighwaySpeedLimit] New config detected at time {current_time:.0f}s (checkpoint_start: {checkpoint_start_time:.0f}s)")
            elif is_new_checkpoint:
                print(f"[HighwaySpeedLimit] New checkpoint interval detected at time {current_time:.0f}s (checkpoint_start: {checkpoint_start_time:.0f}s)")
            
            control_state["checkpoint_start_time"] = checkpoint_start_time
            control_state["current_config"] = config
            # Reset next_switch_times and reinitialize from config
            control_state["next_switch_times"] = {}
            
            # Reinitialize next_switch_times from the config
            initialized_count = 0
            for highway_id, schedule in config.items():
                if not isinstance(schedule, list) or len(schedule) == 0:
                    continue
                
                # Find the first scheduled change for this segment
                for entry in schedule:
                    if not isinstance(entry, dict):
                        continue
                    
                    relative_time = entry.get("time", 0)
                    speed_limit_mph = entry.get("speed_limit")
                    
                    if speed_limit_mph is None:
                        continue
                    
                    # Validate speed limit
                    if not isinstance(speed_limit_mph, (int, float)):
                        continue
                    
                    # Validate speed limit is in available set
                    if speed_limit_mph not in AVAILABLE_SPEED_LIMITS_MPH:
                        closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - speed_limit_mph))
                        speed_limit_mph = closest_mph
                    
                    # Set next switch time to the first scheduled change
                    absolute_time = checkpoint_start_time + relative_time
                    control_state["next_switch_times"][highway_id] = {
                        "time": absolute_time,
                        "speed_limit": speed_limit_mph,
                        "relative_time": relative_time
                    }
                    initialized_count += 1
                    break  # Only initialize with the first scheduled change
            
            print(f"[HighwaySpeedLimit] Reinitialized {initialized_count} segments (config {'changed' if is_new_config else 'unchanged, reapplying after reset_metrics'})")
        
        # Get next switch times for each segment (absolute time when next speed limit should be applied)
        next_switch_times = control_state.get("next_switch_times", {})

        if next_switch_times:
            for seg_id in list(next_switch_times.keys()):
                if seg_id not in env.highway_dict:
                    next_switch_times.pop(seg_id, None)
        
        # Actions to apply at current time (only segments whose switch time has been reached)
        immediate_actions = {}
        
        # Process each highway segment
        for highway_id, schedule in config.items():
            if not isinstance(schedule, list):
                print(f"Warning: Schedule for '{highway_id}' is not a list: {schedule}")
                continue
            
            # Get next switch time for this segment (if exists)
            next_switch_time = next_switch_times.get(highway_id)
            
            # Check if it's time to apply the next speed limit
            if next_switch_time is not None:
                switch_time = next_switch_time["time"]
                
                # Apply speed limit if current_time has reached or exceeded the switch time
                if current_time >= switch_time:
                    speed_limit_mph = next_switch_time["speed_limit"]
                    immediate_actions[highway_id] = speed_limit_mph
                    # Note: next_switch_time will be updated by update_control_state after this action is applied
        
        # Log immediate actions if any (only log first time or when actions change)
        if immediate_actions and not control_state.get("_logged_first_actions", False):
            print(f"[HighwaySpeedLimit] Applying speed limits at time {current_time:.0f}s:")
            for highway_id, speed_mph in list(immediate_actions.items())[:5]:
                print(f"  - {highway_id}: {speed_mph} mph")
            if len(immediate_actions) > 5:
                print(f"  ... and {len(immediate_actions) - 5} more segments")
            control_state["_logged_first_actions"] = True
        
        # Calculate min_remaining (time until next scheduled speed limit change across all segments)
        min_remaining = float('inf')
        for highway_id, next_switch_info in next_switch_times.items():
            if next_switch_info is not None:
                switch_time = next_switch_info["time"]
                remaining = switch_time - current_time
                if remaining <= 0:
                    # We're at or past the switch time - remaining should be 0
                    remaining = 0.0
                min_remaining = min(min_remaining, remaining)
        
        # Update control state
        control_state["next_switch_times"] = next_switch_times
        control_state["last_update_time"] = current_time
        control_state["missing_segments_warned"] = sorted(warned_missing)
        
        return {
            "control_state": control_state,
            "actions": immediate_actions,
            "min_remaining": min_remaining if min_remaining != float('inf') else float('inf')
        }
    
    def update_control_state(
        self,
        control_state: Dict[str, Any],
        step_duration: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update control state after a simulation step.
        For segments that had speed limits applied, updates next_switch_time to the next scheduled change.
        
        IMPORTANT: This method recalculates next_switch_times based on current_time (after step execution),
        which accounts for the actual step_duration that was executed (which may be smaller
        than this module's remaining duration if other modules had shorter remaining durations).
        
        Args:
            control_state: Current control state
            step_duration: Duration of the simulation step (may be smaller than this module's remaining duration)
            **kwargs: Additional arguments:
                - env: SUMOEnv instance (required)
                - current_time: Current simulation time after step (required)
                - applied_actions: Dictionary of actions that were applied (highway_id -> speed_limit_mph)
            
        Returns:
            Updated control state
        """
        # Get environment and current time from kwargs
        env = kwargs.get("env")
        current_time = kwargs.get("current_time")
        applied_actions = kwargs.get("applied_actions", {})
        
        if env is None or current_time is None:
            # If env or current_time not provided, just return state as-is
            return control_state
        
        # Get checkpoint start time and current config
        checkpoint_start_time = control_state.get("checkpoint_start_time", current_time)
        current_config = control_state.get("current_config", {})
        next_switch_times = control_state.get("next_switch_times", {})
        
        # IMPORTANT: Recalculate next_switch_times for all segments based on current_time (after step execution)
        # This accounts for the actual step_duration that was executed
        for highway_id, schedule in current_config.items():
            if highway_id not in env.highway_dict:
                continue
            
            if not isinstance(schedule, list):
                continue
            
            # Get current switch info
            current_switch_info = next_switch_times.get(highway_id)
            
            # If this segment had an action applied, find the next scheduled change
            if highway_id in applied_actions:
                if current_switch_info is None:
                    continue
                
                current_relative_time = current_switch_info.get("relative_time", 0)
                
                # Find the next scheduled change after the currently applied one
                next_change = None
                for entry in schedule:
                    if not isinstance(entry, dict):
                        continue
                    
                    relative_time = entry.get("time", 0)
                    speed_limit_mph = entry.get("speed_limit")
                    
                    if speed_limit_mph is None:
                        continue
                    
                    # Validate speed limit
                    if not isinstance(speed_limit_mph, (int, float)):
                        continue
                    
                    # Validate speed limit is in available set
                    if speed_limit_mph not in AVAILABLE_SPEED_LIMITS_MPH:
                        closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - speed_limit_mph))
                        speed_limit_mph = closest_mph
                    
                    # Find the first scheduled change that is after the current one
                    if relative_time > current_relative_time:
                        next_change = {
                            "time": checkpoint_start_time + relative_time,
                            "speed_limit": speed_limit_mph,
                            "relative_time": relative_time
                        }
                        break
                
                # Update next_switch_time for this segment
                if next_change is not None:
                    next_switch_times[highway_id] = next_change
                else:
                    # No more scheduled changes for this segment - remove it
                    next_switch_times.pop(highway_id, None)
            else:
                # This segment didn't have an action applied
                # Check if current_time has passed the scheduled switch time (shouldn't happen, but handle it)
                if current_switch_info is not None:
                    switch_time = current_switch_info["time"]
                    if current_time >= switch_time:
                        # This shouldn't happen if apply_control was called correctly,
                        # but handle it by finding the next scheduled change
                        current_relative_time = current_switch_info.get("relative_time", 0)
                        next_change = None
                        for entry in schedule:
                            if not isinstance(entry, dict):
                                continue
                            
                            relative_time = entry.get("time", 0)
                            speed_limit_mph = entry.get("speed_limit")
                            
                            if speed_limit_mph is None:
                                continue
                            
                            if not isinstance(speed_limit_mph, (int, float)):
                                continue
                            
                            if speed_limit_mph not in AVAILABLE_SPEED_LIMITS_MPH:
                                closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - speed_limit_mph))
                                speed_limit_mph = closest_mph
                            
                            if relative_time > current_relative_time:
                                next_change = {
                                    "time": checkpoint_start_time + relative_time,
                                    "speed_limit": speed_limit_mph,
                                    "relative_time": relative_time
                                }
                                break
                        
                        if next_change is not None:
                            next_switch_times[highway_id] = next_change
                        else:
                            next_switch_times.pop(highway_id, None)
        
        # Update control state
        control_state["next_switch_times"] = next_switch_times
        control_state["last_applied_speed_limits"] = applied_actions.copy() if applied_actions else control_state.get("last_applied_speed_limits", {})
        control_state["last_update_time"] = current_time
        
        return control_state
    
    def _initialize_control_state(
        self,
        env: Any,
        config: Dict[str, Any],
        initial_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Initialize control state for highway speed limit control.
        
        Args:
            env: SUMOEnv instance
            config: Speed limit configuration
                    Format: {"highway_segment_id": [{"time": 0, "speed_limit": 65}], ...}
            initial_time: Initial simulation time when state is initialized
            
        Returns:
            Initial control state dictionary
        """
        # Initialize speed limits for all highway segments (in mph)
        # Extract initial speed limits (time=0 entries) from config
        initial_speed_limits = {}
        # Initialize next_switch_times for each segment (first scheduled change)
        next_switch_times = {}
        
        # checkpoint_start_time equals initial_time at initialization
        checkpoint_start_time = initial_time
        
        for highway_id, schedule in config.items():
            if not isinstance(schedule, list) or len(schedule) == 0:
                # Use default from highway object
                if highway_id in env.highway_dict:
                    highway_obj = env.highway_dict[highway_id]
                    features = highway_obj.get_feature()
                    segment_default_speed_limit = features.get('segment_default_speed_limit', 0.0)
                    
                    if segment_default_speed_limit > 0:
                        default_speed_mph = segment_default_speed_limit / MPH_TO_MPS
                        speed_limit_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - default_speed_mph))
                    elif highway_obj.highway_road_ids:
                        first_road_id = highway_obj.highway_road_ids[0]
                        road_info = highway_obj.road_dict.get(first_road_id, {})
                        default_speed_mps = road_info.get("max_speed", MAX_SPEED_LIMIT_MPH * MPH_TO_MPS)
                        default_speed_mph = default_speed_mps / MPH_TO_MPS
                        speed_limit_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - default_speed_mph))
                    else:
                        speed_limit_mph = MAX_SPEED_LIMIT_MPH
                else:
                    speed_limit_mph = MAX_SPEED_LIMIT_MPH
                
                initial_speed_limits[highway_id] = speed_limit_mph
            else:
                # Get speed limit from first entry (time=0)
                first_entry = schedule[0]
                if isinstance(first_entry, dict) and "speed_limit" in first_entry:
                    speed_limit_mph = first_entry["speed_limit"]
                else:
                    speed_limit_mph = MAX_SPEED_LIMIT_MPH
                
                initial_speed_limits[highway_id] = speed_limit_mph
                
                # Initialize next_switch_times: find the first scheduled change (earliest time)
                for entry in schedule:
                    if not isinstance(entry, dict):
                        continue
                    
                    relative_time = entry.get("time", 0)
                    speed_limit_mph = entry.get("speed_limit")
                    
                    if speed_limit_mph is None:
                        continue
                    
                    # Validate speed limit
                    if not isinstance(speed_limit_mph, (int, float)):
                        continue
                    
                    # Validate speed limit is in available set
                    if speed_limit_mph not in AVAILABLE_SPEED_LIMITS_MPH:
                        closest_mph = min(AVAILABLE_SPEED_LIMITS_MPH, key=lambda x: abs(x - speed_limit_mph))
                        speed_limit_mph = closest_mph
                    
                    # Set next switch time to the first scheduled change
                    absolute_time = checkpoint_start_time + relative_time
                    next_switch_times[highway_id] = {
                        "time": absolute_time,
                        "speed_limit": speed_limit_mph,
                        "relative_time": relative_time
                    }
                    break  # Only initialize with the first scheduled change
        
        return {
            "initial_speed_limits": initial_speed_limits,
            "last_applied_speed_limits": initial_speed_limits.copy(),
            "next_switch_times": next_switch_times,  # Initialized with first scheduled change for each segment
            "current_config": config,  # Save config for update_control_state to use
            "checkpoint_start_time": checkpoint_start_time,
            "initial_time": initial_time,
            "last_update_time": initial_time
        }
    
    @staticmethod
    def get_available_speed_limits_mph() -> List[int]:
        """
        Get list of available speed limits in mph.
        
        Returns:
            List of available speed limits in mph
        """
        return AVAILABLE_SPEED_LIMITS_MPH.copy()
    
    @staticmethod
    def get_available_speed_limits_mps() -> List[float]:
        """
        Get list of available speed limits in m/s.
        
        Returns:
            List of available speed limits in m/s
        """
        return AVAILABLE_SPEED_LIMITS_MPS.copy()
    
    @staticmethod
    def mph_to_mps(mph: float) -> float:
        """
        Convert speed from mph to m/s.
        
        Args:
            mph: Speed in miles per hour
            
        Returns:
            Speed in meters per second
        """
        return mph * MPH_TO_MPS
    
    @staticmethod
    def mps_to_mph(mps: float) -> float:
        """
        Convert speed from m/s to mph.
        
        Args:
            mps: Speed in meters per second
            
        Returns:
            Speed in miles per hour
        """
        return mps / MPH_TO_MPS
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for tracking highway speed limit performance.

        Returns:
            Dictionary with initialized metric structures
        """
        return {
            'total_reward': 0.0,
            'travel_times': [],  # List of travel times for vehicles on highways
            'road_occupancies': [],  # List of road occupancies (per road, per step)
            'avg_speeds': [],  # List of average speeds (per step)
            'congestion_ratios': [],  # List of congestion ratios (per step)
            'initial_arrived_count': None,  # Will be set when metrics are first updated (for throughput calculation)
            'highway_vehicle_ids': set()  # Track vehicles that have passed through highway segments
        }
    
    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update training metrics with current step data for highway speed limit control.
        
        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            **kwargs: Additional arguments (e.g., step_duration) - not used by this module
        """
        if reward:
            metrics['total_reward'] += sum(reward)
        
        # Initialize initial_arrived_vehicle_ids on first update (for highway throughput calculation)
        if metrics.get('initial_arrived_vehicle_ids') is None:
            # Get initial highway arrived vehicle IDs at the start of this simulation
            initial_arrived_vehicle_ids = set()
            if hasattr(env, 'get_highway_arrived_vehicle_travel_times'):
                initial_arrived_vehicle_ids = set(env.get_highway_arrived_vehicle_travel_times().keys())
            elif hasattr(env, '_highway_arrived_vehicle_tt'):
                initial_arrived_vehicle_ids = set(env._highway_arrived_vehicle_tt.keys())
            metrics['initial_arrived_vehicle_ids'] = initial_arrived_vehicle_ids
        
        # Collect metrics from all highway segments
        all_road_occupancies = []
        all_avg_speeds = []
        all_congestion_ratios = []
        
        if env and hasattr(env, 'highway_dict'):
            for highway_id, highway_obj in env.highway_dict.items():
                features = highway_obj.get_feature()
                
                # Collect segment-level occupancies (segment_occupancy is the average)
                segment_occupancy = features.get('segment_occupancy', 0.0)
                if segment_occupancy > 0:
                    all_road_occupancies.append(segment_occupancy)
                
                # Collect average speed for this segment (segment-level)
                segment_speed = features.get('segment_speed', 0.0)
                if segment_speed > 0:
                    all_avg_speeds.append(segment_speed)
                
                # Collect congestion ratio (segment-level)
                segment_congestion_ratio = features.get('segment_congestion_ratio', 0.0)
                all_congestion_ratios.append(segment_congestion_ratio)
        
        # Store aggregated metrics
        if all_road_occupancies:
            metrics['road_occupancies'].append(all_road_occupancies)
        
        if all_avg_speeds:
            metrics['avg_speeds'].append(np.mean(all_avg_speeds))
        
        if all_congestion_ratios:
            metrics['congestion_ratios'].append(np.mean(all_congestion_ratios))
        
        # Track vehicles currently on highway segments (for travel time calculation)
        if env and hasattr(env, 'highway_dict'):
            # Collect all highway road IDs
            highway_road_ids = set()
            for highway_obj in env.highway_dict.values():
                highway_road_ids.update(highway_obj.highway_road_ids)
            
            # Get lane to road mapping
            lane_to_road_id = {}
            if hasattr(env, '_lane_to_road_id') and env._lane_to_road_id:
                lane_to_road_id = env._lane_to_road_id
            else:
                # Build lane to road mapping from system_states
                all_lane_ids = set(env.system_states.get("get_lane_vehicles", {}).keys())
                if hasattr(env, 'sumo_net') and env.sumo_net:
                    for lane_id in all_lane_ids:
                        try:
                            lane_obj = env.sumo_net.getLane(lane_id)
                            road_obj = lane_obj.getEdge()
                            road_id = road_obj.getID()
                            if not road_id.startswith(":"):
                                lane_to_road_id[lane_id] = road_id
                        except Exception:
                            pass
            
            # Track vehicles on highway lanes
            lane_vehicles_dict = env.system_states.get("get_lane_vehicles", {})
            for lane_id, vehicle_list in lane_vehicles_dict.items():
                road_id = lane_to_road_id.get(lane_id)
                if road_id in highway_road_ids:
                    for vehicle_id in vehicle_list:
                        metrics['highway_vehicle_ids'].add(vehicle_id)
        
        # Travel times are collected from completed vehicles via env.get_average_travel_time()
        # We'll calculate highway-specific travel time in calculate_final_results
        # Throughput is now calculated as number of vehicles arrived during simulation (in calculate_final_results)
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any
    ) -> Dict[str, float]:
        """
        Calculate final training results and metrics for highway speed limit control.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            
        Returns:
            Dictionary with final metric values
        """
        import numpy as np
        
        # Calculate average travel time only for vehicles that passed through highway segments
        highway_vehicle_ids = metrics.get('highway_vehicle_ids', set())
        
        # Get travel times for all highway vehicles that arrived
        highway_arrived_vehicle_tt = {}
        if hasattr(env, 'get_highway_arrived_vehicle_travel_times'):
            highway_arrived_vehicle_tt = env.get_highway_arrived_vehicle_travel_times()
        elif hasattr(env, '_highway_arrived_vehicle_tt'):
            highway_arrived_vehicle_tt = env._highway_arrived_vehicle_tt
        
        # Filter to only include vehicles that passed through highway segments in this episode
        highway_travel_times = []
        for v_id, tt in highway_arrived_vehicle_tt.items():
            if v_id in highway_vehicle_ids:
                highway_travel_times.append(tt)
        
        # Calculate average travel time for highway vehicles only
        if highway_travel_times:
            avg_travel_time = float(np.mean(highway_travel_times))
        else:
            # Fallback: if no highway vehicles arrived in this episode, return 0
            avg_travel_time = 0.0
        
        # Calculate average road occupancy (most important metric)
        avg_road_occupancy = 0.0
        if metrics.get('road_occupancies'):
            # Flatten all road occupancies across all steps
            all_occupancies = []
            for step_occupancies in metrics['road_occupancies']:
                if isinstance(step_occupancies, list):
                    all_occupancies.extend(step_occupancies)
            if all_occupancies:
                avg_road_occupancy = float(np.mean(all_occupancies))
        
        # Calculate highway throughput: number of highway vehicles that arrived during this simulation
        initial_arrived_vehicle_ids = metrics.get('initial_arrived_vehicle_ids', set())
        final_arrived_vehicle_ids = set(highway_arrived_vehicle_tt.keys())
        
        # Count vehicles that:
        # 1. Passed through highway segments (in highway_vehicle_ids)
        # 2. Arrived during this simulation (in final_arrived_vehicle_ids but not in initial_arrived_vehicle_ids)
        highway_arrived_vehicles = highway_vehicle_ids & final_arrived_vehicle_ids
        highway_new_arrived_vehicles = highway_arrived_vehicles - initial_arrived_vehicle_ids
        throughput = len(highway_new_arrived_vehicles)
        
        # Calculate average speed
        avg_speed = 0.0
        if metrics.get('avg_speeds'):
            avg_speed = float(np.mean(metrics['avg_speeds']))
        
        # Calculate average congestion ratio
        avg_congestion_ratio = 0.0
        if metrics.get('congestion_ratios'):
            avg_congestion_ratio = float(np.mean(metrics['congestion_ratios']))
        
        global_avg_travel_time = env.get_average_travel_time() if hasattr(env, 'get_average_travel_time') else 0.0

        return {
            "reward": float(metrics.get('total_reward', 0.0)),
            "avg_travel_time": float(avg_travel_time),  # Highway-only average travel time
            "avg_road_occupancy": float(avg_road_occupancy),  # Highway-only occupancy
            "throughput": float(throughput),  # Highway-only arrived vehicles
            "avg_speed": float(avg_speed),
            "avg_congestion_ratio": float(avg_congestion_ratio),
            "global_avg_travel_time": float(global_avg_travel_time)
        }
