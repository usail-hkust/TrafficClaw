"""
Bus scheduling control module.
"""

import numpy as np
import traci
from typing import Dict, Any, List, Optional
from .base import ControlModule

from .shared.ptflows_timetable import (
    get_ptflows_path_from_config_path,
    build_bus_timetable_config,
)


# Configuration constraints
CONFIG_CONSTRAINTS = {
    "min_headway": 60,       # Minimum 1 minute between buses
    "max_headway": 600,      # Maximum 10 minutes between buses
    "min_dwell_time": 10,    # Minimum 10 seconds at station
    "max_dwell_time": 60,    # Maximum 1 minute at station
    "default_headway": 300,  # Default 5 minutes between buses
    "default_dwell_time": 20,# Default 20 seconds at station
    "delay_tolerance": 20.0, # Delay tolerance in seconds (delay <= tolerance is considered on-time)
    # Change limits per checkpoint (relative to current config)
    "max_headway_change_ratio": 0.30,  # +/-30%
    "max_dwell_time_change": 10,       # +/-10s per station
    # Holding parameters to reduce bunching
    "holding_buffer_ratio": 0.10,      # Hold if last headway < 90% of target
    "max_holding_time": 30.0           # Max holding in seconds
}

# Flag to control whether the module dispatches buses via code
# When False, buses are dispatched by SUMO flow definitions in ptflows.rou.xml
# This should be False when using pre-generated passenger trips that expect specific bus vehicle IDs
DISPATCH_BUSES_VIA_CODE = True

# Flag to disable SUMO flow-defined bus dispatches when DISPATCH_BUSES_VIA_CODE is True
# This prevents duplicate buses from both flow definitions and code dispatches
DISABLE_FLOW_BUSES = True


class BusSchedulingModule(ControlModule):
    """
    Control module for bus scheduling.

    This module manages dynamic bus dispatching with configurable headway
    and dwell times. It is designed to work with LLM Agent optimization
    at checkpoint intervals.

    Config Structure (New Timetable Format):
        {
            "route_id": {
                "timetable": [
                    {
                        "time_range": [start_sec, end_sec],  # Time range in seconds within an hour
                        "headway": int,                       # Seconds between buses
                        "schedule": [
                            {"station_id": str, "dwell_time": int},
                            ...
                        ]
                    },
                    ...
                ]
            }
        }

    Legacy Config Structure (Still Supported):
        {
            "route_id": {
                "headway": int,
                "schedule": [{"station_id": str, "dwell_time": int}, ...]
            }
        }

    Control State Structure:
        {
            "lines": {
                "route_id": {
                    "time_since_last_dispatch": float,  # Time since last bus dispatched
                    "bus_count": int                     # Number of buses dispatched
                }
            }
        }
    """
    
    DOMAIN_KNOWLEDGE = """Bus scheduling control manages bus fleet dispatch timing and station dwell times to optimize public transit service.

## 0. CRITICAL: Optimization Priority (waiting time FIRST, then energy)

- **PRIMARY OBJECTIVE (MANDATORY):** Minimize avg_passenger_waiting_time (lower is better). This is the TOP priority - waiting time reduction MUST be achieved. This is the primary service quality metric.
- **SECONDARY OBJECTIVE (CONSTRAINED):** Under the constraint of achieving lower waiting time, minimize fuel consumption (total_fuel_consumption_g — lower is better).
- **CRITICAL RULE:** NEVER sacrifice waiting time for fuel savings.
- **Dual Optimization Objective (BOTH must be satisfied for a policy to be "better"):**
  - A policy is only "better" if:
    1. Waiting time is reduced (or at least maintained), AND
    2. Fuel consumption is reduced when possible without increasing waiting time.
  - If waiting time increases, the policy is worse regardless of fuel savings.

## 1. Core Principles and Constraints

- **Headway Configuration Rules:**
  - Headway is the time interval between consecutive bus departures on the same route
  - Minimum headway: 60 seconds (1 minute) - buses cannot depart more frequently
  - Maximum headway: 600 seconds (10 minutes) - buses must depart at least every 10 minutes
  - Default headway: 300 seconds (5 minutes)
  - Maximum headway change per checkpoint: +/-30% of current headway

- **Dwell Time Configuration Rules:**
  - Dwell time is how long a bus stops at each station for passenger boarding/alighting
  - Minimum dwell time: 10 seconds per station
  - Maximum dwell time: 60 seconds per station
  - Default dwell time: 20 seconds per station
  - Maximum dwell time change per checkpoint: +/-10 seconds per station

## 2. Configuration Format

- **Timetable Format (Required):**
  ```json
  {
    "route_id": {
      "timetable": [
        {
          "time_range": [0, 1200],      // 0-20 minutes
          "headway": 300,                // 5 minutes between buses
          "schedule": [
            {"station_id": "stop_1", "dwell_time": 20},
            {"station_id": "stop_2", "dwell_time": 25}
          ]
        },
        {
          "time_range": [1200, 2400],   // 20-40 minutes
          "headway": 240,                // 4 minutes (rush hour)
          "schedule": [...]
        }
      ]
    }
  }
  ```
- Time ranges are in seconds within an hour (0-3600), cycling hourly

## 3. Optimization Strategies

- **Headway Optimization:**
  - Decrease headway (more frequent buses) during peak demand periods to reduce waiting time - THIS IS THE PRIORITY
  - Increase headway (less frequent buses) during low demand periods ONLY if waiting time does not increase
  - Consider passenger waiting counts at stations when adjusting headway
  - ALWAYS prioritize reducing waiting time first; minimize fuel consumption only as a secondary goal when waiting time is already optimized

- **Dwell Time Optimization:**
  - Increase dwell time at high-demand stations with many boardings/alightings
  - Decrease dwell time at low-demand stations to improve travel time
  - Consider station-specific passenger volumes

- **Bunching Prevention (Holding Control):**
  - System automatically holds buses if headway becomes too short (< 90% of target)
  - Maximum holding time: 30 seconds
  - This prevents bus bunching where multiple buses arrive together

## 4. Performance Metrics

- **Primary Metrics (waiting time FIRST, energy SECOND):**
  - **Service (PRIMARY - MANDATORY):** avg_passenger_waiting_time — lower is better (MUST minimize - this is the top priority)
  - **Energy (SECONDARY - OPTIONAL):** total_fuel_consumption_g — lower is better (minimize only if waiting time is reduced/maintained)

- **Secondary Metrics:**
  - Average passenger load ratio (balanced is better, avoid overcrowding)
  - Bus rides completed (higher indicates better service utilization)
  - Headway stability (minimize variance using departure_time from read_bus_states)

## 5. Time-Aware Optimization

- **Peak Hours Strategy:**
  - Morning rush (6:00-11:00): Decrease headway to 60-180 seconds to reduce waiting time - PRIORITY #1
  - Evening rush (16:00-21:00): Decrease headway to 60-180 seconds to reduce waiting time - PRIORITY #1
  - Increase dwell time at major transfer stations to reduce waiting time
  - Only after waiting time is optimized, consider fuel-saving adjustments (avoid unnecessary frequency; use holding and small adjustments)
  - During rush hours, traffic patterns are different from off-peak periods
  - Adjust control strategies based on expected demand for the current time period
  - Use historical data from similar time periods to inform your optimization decisions

- **Off-Peak Strategy:**
  - Midday (10:00-16:00): Standard headway 240-360 seconds - adjust based on waiting time first
  - Night (21:00-6:00): Increase headway to 360-600 seconds ONLY if waiting time does not increase
  - Decrease dwell time at low-volume stations ONLY if it doesn't increase waiting time
  - Focus on waiting time reduction first; fuel efficiency is secondary

## 6. Task Description

- **Step 1: Data Analysis**
  - Analyze historical bus operation data to identify overcrowded buses, long waiting times, and inefficient schedules
  - Use read_bus_states() to analyze historical bus data
  - Consider passenger waiting counts, bus load ratios, and headway stability

- **Step 2: Time Period Consideration**
  - Consider the current time period when making optimization decisions
  - During rush hours (Morning: 6:00-11:00, Evening: 16:00-21:00), traffic patterns are different from off-peak periods
  - Adjust control strategies based on expected demand for the current time period
  - Use historical data from similar time periods to inform your optimization decisions

- **Step 3: Optimization Execution**
  - Optimize bus scheduling (headway and dwell_time) using POLICY_PLANNING action
  - **You MUST optimize BOTH waiting time and fuel consumption simultaneously**
  - Optimize headway (time between buses) and dwell_time (station stop duration) for each route
  - Note: Simulation is automatically executed after POLICY_PLANNING - you will receive results and comparison automatically

- **Step 4: Completion**
  - Complete with FINISH action when satisfied with BOTH waiting time and fuel consumption optimization
  - A policy is only "better" if waiting time is reduced (or maintained) AND fuel consumption is reduced when possible"""
    
    def __init__(self, config_dir_name: Optional[str] = None):
        super().__init__("bus_scheduling", "bus_scheduling.json", config_dir_name=config_dir_name)
    
    def get_default_config(self, env: Any) -> Dict[str, Any]:
        """
        Get default bus scheduling configuration from SUMO environment.
        Uses get_ptflows_path_from_config_path and build_bus_timetable_config to read
        departure settings from ptflows.rou.xml (same directory as sumocfg). Routes not
        present in the XML get a 3-segment fallback with default_headway.
        """
        config_path = getattr(env, "config_path", None)
        if config_path:
            ptflows_path = get_ptflows_path_from_config_path(str(config_path))
            config = build_bus_timetable_config(
                env, ptflows_path, default_dwell=CONFIG_CONSTRAINTS["default_dwell_time"]
            )
        else:
            config = {}

        # Clamp headways from XML to module constraints
        for line_config in config.values():
            for seg in line_config.get("timetable", []):
                h = seg.get("headway")
                if h is not None:
                    seg["headway"] = max(
                        CONFIG_CONSTRAINTS["min_headway"],
                        min(CONFIG_CONSTRAINTS["max_headway"], int(h)),
                    )

        # Fallback for routes in env not in ptflows (no flow in XML)
        for route_id, line in env.bus_lines.items():
            if route_id in config or not line.stations:
                continue
            schedule = [
                {"station_id": str(getattr(sid, "station_id", sid)), "dwell_time": CONFIG_CONSTRAINTS["default_dwell_time"]}
                for sid in line.stations
            ]
            config[route_id] = {
                "timetable": [
                    {"time_range": [0, 1200], "headway": CONFIG_CONSTRAINTS["default_headway"], "schedule": schedule},
                    {"time_range": [1200, 2400], "headway": CONFIG_CONSTRAINTS["default_headway"], "schedule": schedule},
                    {"time_range": [2400, 3600], "headway": CONFIG_CONSTRAINTS["default_headway"], "schedule": schedule},
                ]
            }
        return config

    def validate_config(self, config: Dict[str, Any], reference_config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate bus scheduling configuration.
        Only supports new timetable format.
        
        Args:
            config: Configuration dictionary to validate
            reference_config: Optional reference configuration for comparison (not used for bus scheduling)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Config must be a dictionary"
        
        if len(config) == 0:
            return False, "Config is empty"
        
        for route_id, line_config in config.items():
            if not isinstance(line_config, dict):
                return False, f"Config for route {route_id} must be a dictionary"
            
            # Check for timetable format
            if "timetable" not in line_config:
                return False, f"Missing 'timetable' for route {route_id}. New format required."
            
            # Validate timetable format
            timetable = line_config["timetable"]
            if not isinstance(timetable, list):
                return False, f"timetable for route {route_id} must be a list"
            
            if len(timetable) == 0:
                return False, f"timetable for route {route_id} is empty"
            
            for t_idx, time_segment in enumerate(timetable):
                if not isinstance(time_segment, dict):
                    return False, f"timetable[{t_idx}] in route {route_id} must be a dictionary"
                
                # Validate time_range
                if "time_range" not in time_segment:
                    return False, f"timetable[{t_idx}] in route {route_id} missing 'time_range'"
                
                time_range = time_segment["time_range"]
                if not isinstance(time_range, list) or len(time_range) != 2:
                    return False, f"time_range in timetable[{t_idx}] of route {route_id} must be [start, end]"
                
                if time_range[0] >= time_range[1]:
                    return False, f"Invalid time_range {time_range} in timetable[{t_idx}] of route {route_id}"
                
                # Validate headway
                if "headway" not in time_segment:
                    return False, f"timetable[{t_idx}] in route {route_id} missing 'headway'"
                
                headway = time_segment["headway"]
                if not isinstance(headway, (int, float)):
                    return False, f"Headway in timetable[{t_idx}] of route {route_id} must be a number"
                if headway < CONFIG_CONSTRAINTS["min_headway"]:
                    return False, f"Headway {headway} in timetable[{t_idx}] of route {route_id} is below minimum"
                if headway > CONFIG_CONSTRAINTS["max_headway"]:
                    return False, f"Headway {headway} in timetable[{t_idx}] of route {route_id} exceeds maximum"
                
                # Validate schedule
                if "schedule" not in time_segment:
                    return False, f"timetable[{t_idx}] in route {route_id} missing 'schedule'"
                
                schedule = time_segment["schedule"]
                if not isinstance(schedule, list):
                    return False, f"schedule in timetable[{t_idx}] of route {route_id} must be a list"
                
                if len(schedule) == 0:
                    return False, f"schedule in timetable[{t_idx}] of route {route_id} is empty"
                
                for i, stop in enumerate(schedule):
                    if not isinstance(stop, dict):
                        return False, f"schedule[{i}] in timetable[{t_idx}] of route {route_id} must be a dictionary"
                    if "station_id" not in stop:
                        return False, f"schedule[{i}] in timetable[{t_idx}] of route {route_id} missing 'station_id'"
                    if "dwell_time" not in stop:
                        return False, f"schedule[{i}] in timetable[{t_idx}] of route {route_id} missing 'dwell_time'"
                    
                    dwell_time = stop["dwell_time"]
                    if not isinstance(dwell_time, (int, float)):
                        return False, f"dwell_time at schedule[{i}] in timetable[{t_idx}] of route {route_id} must be a number"
                    if dwell_time < CONFIG_CONSTRAINTS["min_dwell_time"]:
                        return False, f"dwell_time {dwell_time} in timetable[{t_idx}] of route {route_id} is below minimum"
                    if dwell_time > CONFIG_CONSTRAINTS["max_dwell_time"]:
                        return False, f"dwell_time {dwell_time} in timetable[{t_idx}] of route {route_id} exceeds maximum"

                # If reference_config is available, enforce bounded changes
                if isinstance(reference_config, dict):
                    ref_route = reference_config.get(route_id)
                    if isinstance(ref_route, dict):
                        ref_timetable = ref_route.get("timetable")
                        if isinstance(ref_timetable, list) and t_idx < len(ref_timetable):
                            ref_segment = ref_timetable[t_idx]
                            if isinstance(ref_segment, dict):
                                # Headway change limit (relative)
                                ref_headway = ref_segment.get("headway")
                                max_ratio = CONFIG_CONSTRAINTS.get("max_headway_change_ratio")
                                if isinstance(ref_headway, (int, float)) and isinstance(headway, (int, float)):
                                    if max_ratio is not None and ref_headway > 0:
                                        if abs(headway - ref_headway) > max_ratio * ref_headway:
                                            return False, (
                                                f"Headway change too large in timetable[{t_idx}] of route {route_id} "
                                                f"(allowed +/-{max_ratio * 100:.0f}%)"
                                            )
                                # Dwell time change limit (absolute)
                                max_dwell_change = CONFIG_CONSTRAINTS.get("max_dwell_time_change")
                                if max_dwell_change is not None:
                                    ref_schedule = ref_segment.get("schedule")
                                    if isinstance(ref_schedule, list):
                                        ref_dwell_by_station = {}
                                        for ref_stop in ref_schedule:
                                            if isinstance(ref_stop, dict):
                                                station_id = ref_stop.get("station_id")
                                                ref_dwell = ref_stop.get("dwell_time")
                                                if station_id is not None and isinstance(ref_dwell, (int, float)):
                                                    ref_dwell_by_station[station_id] = ref_dwell
                                        for stop in schedule:
                                            if isinstance(stop, dict):
                                                station_id = stop.get("station_id")
                                                dwell_time = stop.get("dwell_time")
                                                if station_id in ref_dwell_by_station and isinstance(dwell_time, (int, float)):
                                                    if abs(dwell_time - ref_dwell_by_station[station_id]) > max_dwell_change:
                                                        return False, (
                                                            f"dwell_time change too large at station {station_id} in "
                                                            f"timetable[{t_idx}] of route {route_id} "
                                                            f"(allowed +/-{max_dwell_change}s)"
                                                        )
        
        return True, None
    
    def _get_current_config(self, line_config: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """
        Get the current configuration for a line based on current time.
        Supports both timetable format and legacy format.
        
        Args:
            line_config: Line configuration (either timetable or legacy format)
            current_time: Current simulation time in seconds
            
        Returns:
            Dictionary with 'headway' and 'schedule' for current time
        """
        # Check if using timetable format
        if "timetable" in line_config:
            timetable = line_config["timetable"]
            max_end = None
            for segment in timetable:
                if not isinstance(segment, dict):
                    continue
                time_range = segment.get("time_range")
                if not isinstance(time_range, list) or len(time_range) != 2:
                    continue
                try:
                    end_val = float(time_range[1])
                except (TypeError, ValueError):
                    continue
                max_end = end_val if max_end is None else max(max_end, end_val)

            if max_end is None:
                max_end = 3600

            if max_end <= 3600:
                time_in_window = current_time % 3600
            elif max_end <= 86400:
                time_in_window = current_time % 86400
            else:
                time_in_window = current_time

            # Find matching time segment
            for time_segment in timetable:
                time_range = time_segment.get("time_range", [])
                if time_range and time_range[0] <= time_in_window < time_range[1]:
                    return {
                        "headway": time_segment["headway"],
                        "schedule": time_segment["schedule"]
                    }

            # Fallback: try absolute time if daily/hourly window didn't match
            if time_in_window != current_time:
                for time_segment in timetable:
                    time_range = time_segment.get("time_range", [])
                    if time_range and time_range[0] <= current_time < time_range[1]:
                        return {
                            "headway": time_segment["headway"],
                            "schedule": time_segment["schedule"]
                        }

            # If no match found (shouldn't happen with valid config), use first segment
            first_segment = timetable[0]
            return {
                "headway": first_segment["headway"],
                "schedule": first_segment["schedule"]
            }
        else:
            # Legacy format: return as-is
            return {
                "headway": line_config.get("headway", CONFIG_CONSTRAINTS["default_headway"]),
                "schedule": line_config.get("schedule", [])
            }
    
    def _initialize_control_state(
        self,
        env: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initialize control state for all bus lines.
        Correctly counts existing buses from SUMO to avoid duplicate IDs after checkpoint loading.
        When DISPATCH_BUSES_VIA_CODE and DISABLE_FLOW_BUSES are both True, disables SUMO flow-defined buses.
        """
        # Disable SUMO flow-defined bus dispatches if the sumocfg was not
        # already rewritten to remove them before startup.
        if (
            DISPATCH_BUSES_VIA_CODE
            and DISABLE_FLOW_BUSES
            and not getattr(env, "transit_flow_vehicles_filtered", False)
        ):
            self._disable_flow_buses(env, config)

        lines_state = {}
        
        for route_id, route_config in config.items():
            line = env.get_bus_line(route_id)
            bus_count = line.bus_count if line else 0
            
            # Get initial config (at time 0)
            current_config = self._get_current_config(route_config, 0.0)
            headway = current_config["headway"]
            
            # Initial time_since_last_dispatch: set to headway to trigger immediate dispatch if no buses
            time_since_last_dispatch = 0.0 if bus_count > 0 else headway
            
            lines_state[route_id] = {
                "time_since_last_dispatch": time_since_last_dispatch,
                "bus_count": bus_count
            }

        return {"lines": lines_state}

    def _disable_flow_buses(self, env: Any, config: Dict[str, Any]) -> None:
        """
        Disable SUMO flow-defined bus dispatches to prevent conflicts with code-dispatched buses.
        Uses TraCI to set flow end time to current time, effectively stopping future dispatches.
        Also removes any existing flow-defined buses that are already in the simulation.
        """
        try:
            current_time = env.traci_conn.simulation.getTime()
            disabled_count = 0

            # Get all flow IDs that match bus routes
            for route_id in config.keys():
                flow_id = route_id  # Flow ID is same as route_id (e.g., "bus_Bx19:0")
                try:
                    # Try to modify the flow's end time to stop future dispatches
                    # Note: Not all SUMO versions support flow modification via TraCI
                    # If this fails, we'll rely on removing vehicles instead
                    env.traci_conn.simulation.setParameter(f"flow.{flow_id}", "end", str(current_time))
                    disabled_count += 1
                except Exception:
                    # Flow modification not supported, will rely on vehicle removal
                    pass

            # Remove ALL existing flow-defined bus vehicles from the simulation
            # This is necessary because flows may have already dispatched vehicles at t=0
            removed_count = 0
            try:
                all_vehicles = env.traci_conn.vehicle.getIDList()
                for veh_id in all_vehicles:
                    # Check if it's a flow-defined bus (format: bus_XXX:N.M)
                    # but NOT a code-dispatched bus (format: ctrl_bus_XXX:N.M)
                    if veh_id.startswith('bus_') and not veh_id.startswith('ctrl_'):
                        try:
                            env.traci_conn.vehicle.remove(veh_id)
                            removed_count += 1
                        except Exception:
                            pass
            except Exception:
                pass

            # Also try to remove pending vehicles (scheduled but not yet inserted)
            try:
                pending_vehicles = env.traci_conn.simulation.getPendingVehicles()
                for veh_id in pending_vehicles:
                    if veh_id.startswith('bus_') and not veh_id.startswith('ctrl_'):
                        try:
                            env.traci_conn.vehicle.remove(veh_id)
                            removed_count += 1
                        except Exception:
                            pass
            except Exception:
                # getPendingVehicles not available in older SUMO versions
                pass

            if disabled_count > 0 or removed_count > 0:
                print(f"[BusScheduling] Disabled {disabled_count} flow definitions, removed {removed_count} existing vehicles")
            else:
                print(f"[BusScheduling] Code-dispatched mode enabled (no flow vehicles to remove)")

        except Exception as e:
            print(f"[BusScheduling] Warning: Failed to disable flow buses: {e}")
            print(f"[BusScheduling] Code-dispatched buses will use 'ctrl_' prefix to avoid conflicts")

    def _remove_flow_buses(self, env: Any) -> None:
        """
        Remove any flow-defined buses currently in the simulation.
        Called on every apply_control to continuously clean up flow-dispatched vehicles.
        """
        try:
            all_vehicles = env.traci_conn.vehicle.getIDList()
            for veh_id in all_vehicles:
                # Remove flow-defined buses (bus_XXX:N.M) but not code-dispatched (ctrl_bus_XXX:N.M)
                if veh_id.startswith('bus_') and not veh_id.startswith('ctrl_'):
                    try:
                        env.traci_conn.vehicle.remove(veh_id)
                    except Exception:
                        pass
        except Exception:
            pass

    def _group_by_first_edge(self, env: Any, pending_dispatches: list) -> Dict[str, list]:
        """
        Group pending dispatches by their first edge to detect potential conflicts.
        
        Args:
            env: SUMO environment
            pending_dispatches: List of pending dispatch info
            
        Returns:
            Dictionary mapping first_edge to list of route_ids
        """
        edge_groups = {}
        
        for dispatch in pending_dispatches:
            route_id = dispatch["route_id"]
            line = env.get_bus_line(route_id)
            
            if line and line.edges:
                first_edge = line.edges[0]
                if first_edge not in edge_groups:
                    edge_groups[first_edge] = []
                edge_groups[first_edge].append(dispatch)
        
        return edge_groups
    
    def _resolve_dispatch_conflicts(
        self,
        env: Any,
        pending_dispatches: list,
        lines_state: Dict[str, Any],
        conflict_delay: float = 20.0
    ) -> list:
        """
        Resolve conflicts when multiple routes want to dispatch on the same edge.
        
        Args:
            env: SUMO environment
            pending_dispatches: List of pending dispatch info
            lines_state: Current state of all routes
            conflict_delay: Delay in seconds for conflicting dispatches (default: 20s)
            
        Returns:
            List of dispatches that should proceed (conflicts resolved)
        """
        # Group by first edge
        edge_groups = self._group_by_first_edge(env, pending_dispatches)
        
        resolved = []
        delayed = []
        
        for first_edge, dispatches in edge_groups.items():
            if len(dispatches) == 1:
                # No conflict, dispatch immediately
                resolved.append(dispatches[0])
            else:
                # Conflict detected: multiple routes want to dispatch on same edge
                # Sort by priority (who waited longer)
                dispatches_sorted = sorted(
                    dispatches,
                    key=lambda d: lines_state[d["route_id"]]["time_since_last_dispatch"],
                    reverse=True
                )
                
                # Allow first one to dispatch
                resolved.append(dispatches_sorted[0])
                
                # Delay others
                for dispatch in dispatches_sorted[1:]:
                    route_id = dispatch["route_id"]
                    # Reduce time_since_last to delay dispatch by conflict_delay seconds
                    lines_state[route_id]["time_since_last_dispatch"] -= conflict_delay
                    delayed.append(route_id)
        
        return resolved
    
    def apply_control(
        self,
        env: Any,
        config: Dict[str, Any],
        current_time: float,
        control_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply bus scheduling control logic with dynamic conflict detection.

        This method is called at each simulation step. It checks if any line
        needs a new bus dispatched based on the configured headway.

        Note: When DISPATCH_BUSES_VIA_CODE is False, this method will not dispatch
        buses but will still track state for metrics collection. In this mode,
        buses are dispatched by SUMO flow definitions in ptflows.rou.xml.

        Args:
            env: SUMOEnv instance
            config: Bus scheduling configuration
            current_time: Current simulation time in seconds
            control_state: Current control state (will be initialized if None)
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary containing:
                - control_state: Updated control state
                - dispatched_buses: List of newly dispatched bus IDs
        """
        # Initialize control state if needed
        if control_state is None:
            control_state = self._initialize_control_state(env, config)

        dispatched_buses = []
        lines_state = control_state["lines"]

        dispatch_actions = {}  # {route_id: dispatch_info}

        # Calculate minimum time until next dispatch
        min_remaining = float('inf')

        # If dispatching is disabled, just return empty results
        # The buses will be dispatched by SUMO flow definitions
        if not DISPATCH_BUSES_VIA_CODE:
            return {
                "control_state": control_state,
                "dispatched_buses": [],
                "dispatch_actions": {},
                "next_dispatch_time": float('inf')
            }

        # Continuously remove flow-defined buses only when startup filtering was not available.
        if DISABLE_FLOW_BUSES and not getattr(env, "transit_flow_vehicles_filtered", False):
            self._remove_flow_buses(env)

        # Step 1: Collect all pending dispatches
        pending_dispatches = []
        
        for route_id, line_config in config.items():
            # Get current config based on time (supports timetable)
            current_config = self._get_current_config(line_config, current_time)
            headway = current_config["headway"]
            schedule = current_config["schedule"]
            line = env.get_bus_line(route_id)
            
            # Skip if line not in state (shouldn't happen with proper initialization)
            if route_id not in lines_state:
                lines_state[route_id] = {
                    "time_since_last_dispatch": headway,  # 立即发车
                    "bus_count": 0
                }
            
            line_state = lines_state[route_id]
            time_since_last = line_state["time_since_last_dispatch"]
            
            # Calculate remaining time until next dispatch
            remaining = headway - time_since_last
            min_remaining = min(min_remaining, remaining)
            
            # Check if it's time to dispatch a new bus
            if time_since_last >= headway:
                # Holding control to reduce bunching (delay if last headway too short)
                if line and hasattr(line, "departure_times") and len(line.departure_times) >= 2:
                    last_headway = line.departure_times[-1] - line.departure_times[-2]
                    buffer_ratio = CONFIG_CONSTRAINTS.get("holding_buffer_ratio", 0.0)
                    max_hold = CONFIG_CONSTRAINTS.get("max_holding_time", 0.0)
                    if last_headway > 0 and buffer_ratio > 0 and max_hold > 0:
                        min_allowed = headway * (1.0 - buffer_ratio)
                        if last_headway < min_allowed:
                            hold_seconds = min(max_hold, min_allowed - last_headway)
                            if hold_seconds > 0:
                                line_state["time_since_last_dispatch"] = max(0.0, headway - hold_seconds)
                                remaining = headway - line_state["time_since_last_dispatch"]
                                min_remaining = min(min_remaining, remaining)
                                continue
                # Add to pending dispatches for conflict detection
                pending_dispatches.append({
                    "route_id": route_id,
                    "line_config": line_config,
                    "current_config": current_config,
                    "line_state": line_state
                })
        
        # Step 2: Resolve conflicts among pending dispatches
        resolved_dispatches = self._resolve_dispatch_conflicts(
            env, pending_dispatches, lines_state, conflict_delay=20.0
        )
        
        # Step 3: Execute resolved dispatches
        for dispatch in resolved_dispatches:
            route_id = dispatch["route_id"]
            current_config = dispatch["current_config"]
            line_state = dispatch["line_state"]
            
            # Prepare dispatch info
            dispatch_info = self._prepare_dispatch(env, route_id, current_config, line_state)
            
            if dispatch_info:
                dispatch_actions[route_id] = dispatch_info
                dispatched_buses.append(dispatch_info["bus_id"])
                # Update state: reset time counter
                line_state["time_since_last_dispatch"] = 0.0  # 重置计时器
                line_state["bus_count"] += 1
                # print(f"[BusScheduling] Prepared dispatch: {dispatch_info['bus_id']} on {route_id} at t={current_time:.0f}s")
        
        return {
            "control_state": control_state,
            "dispatched_buses": dispatched_buses,
            "dispatch_actions": dispatch_actions,  # 返回待执行的发车动作
            "next_dispatch_time": min_remaining  # 返回距离下次发车的最小剩余时间
        }
    
    def _prepare_dispatch(
        self,
        env: Any,
        route_id: str,
        line_config: Dict[str, Any],
        line_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare dispatch information for a new bus.
        Uses 'ctrl_' prefix to distinguish from SUMO flow-defined buses.
        """
        line = env.get_bus_line(route_id)
        if not line:
            return None

        schedule = line_config.get("schedule", [])
        # Use 'ctrl_' prefix to avoid conflict with SUMO flow-defined bus IDs
        bus_id = f"ctrl_{route_id}.{line_state['bus_count']}"

        # Departure position on the first edge (0.0 = edge start, consistent with SUMO flow)
        departure_pos = 0.0

        return {
            "bus_id": bus_id,
            "route_id": route_id,
            "vehicle_type": line_config.get("vehicle_type", "bus"),
            "position": departure_pos,
            "schedule": schedule
        }
    
    def update_control_state(
        self,
        control_state: Dict[str, Any],
        step_duration: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update control state after a simulation step.
        累积每条线路的发车间隔时间。
        
        Args:
            control_state: Current control state
            step_duration: Duration of the simulation step
            **kwargs: Additional arguments (unused)
            
        Returns:
            Updated control state with accumulated time
        """
        if control_state is None:
            return control_state
        
        lines_state = control_state.get("lines", {})
        
        # 累积每条线路的时间
        for line_state in lines_state.values():
            line_state["time_since_last_dispatch"] += step_duration
        
        return control_state
    
    def _count_active_bus_rides(self, env: Any) -> int:
        """Count passengers currently riding buses (in-progress, not yet completed).

        Uses a cached person list to reduce TraCI calls and avoid race conditions.
        """
        active_rides = 0
        try:
            # Get person list once (single TraCI call)
            person_ids = list(env.traci_conn.person.getIDList())

            for person_id in person_ids:
                try:
                    stage = env.traci_conn.person.getStage(person_id)
                    if stage.type == 3:  # Riding
                        vehicle_id = env.traci_conn.person.getVehicle(person_id)
                        if 'bus_' in vehicle_id.lower():
                            active_rides += 1
                except traci.TraCIException:
                    # Person may have left the simulation
                    continue
                except Exception:
                    pass
        except traci.TraCIException:
            # TraCI connection may be closed
            pass
        except Exception:
            pass
        return active_rides

    def _calculate_metrics(self, env: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate delay rate metrics from all bus lines.
        
        Args:
            env: SUMOEnv instance
            config: Bus scheduling configuration
            
        Returns:
            Dictionary containing delay rate metrics
        """
        if not env or not config:
            return {
                "on_time_rate": 1.0,
                "delay_rate": 0.0,
                "avg_delay": 0.0,
                "total_segments": 0
            }
        
        total_on_time = 0
        total_late = 0
        total_delay_sum = 0.0
        total_segments = 0
        
        for route_id in config.keys():
            line = env.get_bus_line(route_id)
            if line:
                delay_stats = line.calculate_delay_rate()
                total_on_time += delay_stats['on_time_rate'] * delay_stats['total_segments']
                total_late += delay_stats['delay_rate'] * delay_stats['total_segments']
                total_delay_sum += delay_stats['avg_delay'] * delay_stats['total_segments']
                total_segments += delay_stats['total_segments']
        
        return {
            "on_time_rate": total_on_time / total_segments if total_segments > 0 else 1.0,
            "delay_rate": total_late / total_segments if total_segments > 0 else 0.0,
            "avg_delay": total_delay_sum / total_segments if total_segments > 0 else 0.0,
            "total_segments": total_segments
        }
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for bus scheduling control.
        
        Returns:
            Dictionary with initialized metric structures
        """
        return {
            'total_reward': 0.0,
            'passenger_waiting_episode': [],  # 每个时间点的总等待乘客数
            'passenger_load_episode': [],      # 每个时间点的平均负载率
            'waiting_time_episode': [],        # 每个时间点的平均等待时间
            'global_waiting_times': [],        # 累积的所有乘客等待时间
            # 能耗和排放指标
            'fuel_consumption_episode': [],    # 每个时间点的总燃油消耗 (ml)
            'total_fuel_consumption': 0.0,     # 累积总燃油消耗
            # 晚点率指标
            'delay_rate_episode': [],          # 每个时间点的晚点率
            'on_time_rate_episode': [],        # 每个时间点的准点率
            'avg_delay_episode': [],           # 每个时间点的平均延迟 (秒)
            # 到达率指标
            'total_arrived_persons': 0,        # 每个checkpoint到达乘客总数（全局）
            'bus_rides': 0                     # 公交乘车人次（仅公交）
        }
    
    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        step_duration: float = 1.0,
        **kwargs
    ) -> None:
        """
        Update metrics using BusStation and BusLine objects.
        
        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            step_duration: Duration of the simulation step (actual time executed)
            **kwargs: Additional arguments - not used by this module
        """
        if reward:
            metrics['total_reward'] += sum(reward)

        # Track active bus rides (in-progress, not yet completed)
        metrics['active_bus_rides'] = self._count_active_bus_rides(env)

        # 1. Average waiting passengers across all bus stations
        waiting_counts = [station.get_waiting_count() for station in env.bus_stations.values()]
        avg_waiting = np.mean(waiting_counts) if waiting_counts else 0.0
        metrics['passenger_waiting_episode'].append(avg_waiting)
        
        # 2. Average load ratio using BusLine objects
        all_load_ratios = []
        for line in env.bus_lines.values():
            all_load_ratios.extend(line.get_load_ratios())
        
        avg_load = sum(all_load_ratios) / len(all_load_ratios) if all_load_ratios else 0.0
        metrics['passenger_load_episode'].append(avg_load)
        
        # 3. Passenger waiting times (bus-only)
        waiting_times = []
        try:
            waiting_ids = set()
            if hasattr(env, 'bus_stops'):
                for stop_id in env.bus_stops:
                    try:
                        waiting_ids.update(env.traci_conn.busstop.getPersonIDs(stop_id))
                    except Exception:
                        pass
            if waiting_ids and hasattr(env, 'waiting_passenger_list'):
                for p_id in waiting_ids:
                    if p_id in env.waiting_passenger_list:
                        waiting_times.append(env.waiting_passenger_list[p_id])
        except Exception:
            waiting_times = []
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        metrics['waiting_time_episode'].append(avg_waiting_time)
        metrics['global_waiting_times'].extend(waiting_times)
        
        # 4. Fuel consumption and emissions for buses
        total_fuel = 0.0  # mg

        try:
            # Get current vehicle list once to validate bus IDs exist
            current_vehicle_ids = set(env.traci_conn.vehicle.getIDList())

            # Iterate through all bus lines and get their buses
            for line in env.bus_lines.values():
                bus_ids = line.get_bus_ids()

                for veh_id in bus_ids:
                    # Skip vehicles that are no longer in the simulation
                    if veh_id not in current_vehicle_ids:
                        continue
                    try:
                        # Get fuel consumption rate (mg/s)
                        fuel_rate = env.traci_conn.vehicle.getFuelConsumption(veh_id)
                        # Convert to fuel consumed in this step (mg)
                        fuel = fuel_rate * step_duration
                        total_fuel += fuel

                    except traci.TraCIException:
                        # Vehicle may have left the network
                        continue
                    except Exception:
                        # Some vehicles might not support these metrics
                        pass

            # Store per-timestep values
            metrics['fuel_consumption_episode'].append(total_fuel)

            # Accumulate totals
            metrics['total_fuel_consumption'] += total_fuel

        except traci.TraCIException:
            # TraCI connection may be closed
            metrics['fuel_consumption_episode'].append(0.0)
        except Exception:
            # If TraCI calls fail, append zeros
            metrics['fuel_consumption_episode'].append(0.0)
        
        # 5. Delay rate metrics
        # Calculate current delay rate from all bus lines
        # Get config from env.enabled_controls
        config = {}
        if hasattr(env, 'enabled_controls') and 'bus_scheduling' in env.enabled_controls:
            config = env.enabled_controls['bus_scheduling'].get('config', {})
        delay_metrics = self._calculate_metrics(env, config)
        metrics['delay_rate_episode'].append(delay_metrics['delay_rate'])
        metrics['on_time_rate_episode'].append(delay_metrics['on_time_rate'])
        metrics['avg_delay_episode'].append(delay_metrics['avg_delay'])
        
        # 6. Passenger arrival tracking
        try:
            arrived_count = env.traci_conn.simulation.getArrivedPersonNumber()
        except:
            arrived_count = 0
        
        metrics['total_arrived_persons'] = metrics.get('total_arrived_persons', 0) + arrived_count
        
        # 7. Bus-specific ride statistics
        try:
            cumulative_bus_rides = int(env.traci_conn.simulation.getParameter("", "device.tripinfo.rideStatistics.bus"))
            # 第一次调用时初始化基准值（不计入增量）
            if '_last_cumulative_bus_rides' not in metrics:
                metrics['_last_cumulative_bus_rides'] = cumulative_bus_rides
            else:
                # 计算增量：当前累积值 - 上次记录的累积值
                incremental_rides = cumulative_bus_rides - metrics['_last_cumulative_bus_rides']
                metrics['bus_rides'] = metrics.get('bus_rides', 0) + incremental_rides
                metrics['_last_cumulative_bus_rides'] = cumulative_bus_rides
        except:
            pass  # Keep previous value if API call fails
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any
    ) -> Dict[str, float]:
        """
        Calculate final training results and metrics for bus scheduling.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            
        Returns:
            Dictionary with final metric values
        """
        waiting_episode = metrics.get('passenger_waiting_episode', [])
        load_episode = metrics.get('passenger_load_episode', [])
        waiting_time_episode = metrics.get('waiting_time_episode', [])
        fuel_episode = metrics.get('fuel_consumption_episode', [])
        delay_rate_episode = metrics.get('delay_rate_episode', [])
        on_time_rate_episode = metrics.get('on_time_rate_episode', [])
        avg_delay_episode = metrics.get('avg_delay_episode', [])
        
        # 到达率指标
        total_arrived = metrics.get('total_arrived_persons', 0)
        bus_rides = metrics.get('bus_rides', 0)
        active_bus_rides = metrics.get('active_bus_rides', 0)

        # avg_passenger_waiting_time: prefer global_waiting_times (all observed waits) over
        # waiting_time_episode (mean of step means). Return None when no data to avoid misleading 0.
        global_waiting = metrics.get('global_waiting_times', [])
        if global_waiting:
            avg_passenger_waiting_time = float(np.mean(global_waiting))
        elif waiting_time_episode:
            avg_passenger_waiting_time = float(np.mean(waiting_time_episode))
        else:
            avg_passenger_waiting_time = None  # No data - avoid misleading 0.0

        return {
            "reward": float(metrics.get('total_reward', 0.0)),
            "avg_passenger_waiting_count": float(np.mean(waiting_episode)) if waiting_episode else 0.0,
            "avg_passenger_load": float(np.mean(load_episode)) if load_episode else 0.0,
            "avg_passenger_waiting_time": avg_passenger_waiting_time,
            # 能耗指标
            "total_fuel_consumption_g": float(metrics.get('total_fuel_consumption', 0.0)) / 1000.0,  # 转换为克 (mg -> g)
            # 晚点率指标
            "avg_delay_rate": float(np.mean(delay_rate_episode)) if delay_rate_episode else 0.0,
            "avg_on_time_rate": float(np.mean(on_time_rate_episode)) if on_time_rate_episode else 1.0,
            "avg_delay_seconds": float(np.mean(avg_delay_episode)) if avg_delay_episode else 0.0,
            # 到达率指标
            "total_arrived_persons": total_arrived,
            "bus_rides": bus_rides,  # 公交乘车人次 (completed rides)
            "active_bus_rides": active_bus_rides  # 当前正在乘坐公交的乘客数 (in-progress)
        }
