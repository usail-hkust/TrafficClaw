import os
import math
import numpy as np
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, List, Optional

import traci


class Highway:
    """
    Represents a single highway segment for speed limit control.
    Handles speed limit control to alleviate congestion on the assigned highway segment.
    
    This class controls a specific set of highway roads (already identified in sumo_env)
    and applies dynamic speed limit control based on traffic conditions.
    """
    
    def __init__(
        self,
        highway_id: str,
        road_ids: List[str],
        dic_traffic_env_conf: Dict[str, Any],
        traci_conn: traci.connection,
        path_to_log: str,
        road_dict: Dict[str, Any],
        adjacency_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes a Highway object for managing speed limit control on a specific highway segment.
        
        Args:
            highway_id (str): Unique identifier for this highway segment.
            road_ids (list): List of road IDs that belong to this highway segment (already identified as highways).
            dic_traffic_env_conf (dict): Traffic environment configuration dictionary.
            traci_conn (traci.connection): Active TraCI connection object.
            path_to_log (str): Path to directory for logging.
            road_dict (dict): Dictionary mapping road_id to road attributes.
            adjacency_info (dict, optional): Information about neighboring highway segments.
        """
        # Store injected dependencies
        self.highway_id = highway_id
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.traci_conn = traci_conn
        self.path_to_log = path_to_log
        self.road_dict = road_dict
        self.adjacency_info = adjacency_info if adjacency_info else {}
        
        # Store road IDs for this highway segment
        self.highway_road_ids = road_ids.copy()
        self.lanes_by_road = {}  # road_id -> list of lane_ids (cached)
        
        # Speed limit control state
        self.current_speed_limits = {}  # road_id -> current_speed_limit (m/s)
        self.default_speed_limits = {}  # road_id -> default_speed_limit (m/s)
        self.speed_limit_history = defaultdict(list)  # road_id -> list of (time, speed_limit) tuples
        
        # Initialize speed limits from road_dict
        # Collect speeds from all roads in the segment
        road_speeds = []
        for road_id in self.highway_road_ids:
            road_info = self.road_dict.get(road_id, {})
            default_speed = road_info.get("max_speed", 33.33)  # Default ~120 km/h if not found
            road_speeds.append(default_speed)
            self.default_speed_limits[road_id] = default_speed
        
        # Calculate average speed for the segment
        # Highway segment's speed limit is set to the average of all roads in the segment
        if road_speeds:
            segment_avg_speed = np.mean(road_speeds)
        else:
            segment_avg_speed = 33.33  # Default ~120 km/h if no roads

        # State variables
        self.dic_lane_vehicle_current_step = defaultdict(list)
        self.dic_lane_speed_current_step = defaultdict(float)
        self.dic_lane_density_current_step = defaultdict(float)
        self.dic_lane_occupancy_current_step = defaultdict(float)
        
        # Feature storage
        self.dic_feature = {}
        
        # Control parameters
        self.min_speed_limit = dic_traffic_env_conf.get("MIN_HIGHWAY_SPEED", 20.0)  # Minimum speed limit (m/s)
        self.max_speed_limit = dic_traffic_env_conf.get("MAX_HIGHWAY_SPEED", 40.0)  # Maximum speed limit (m/s)
        self.speed_limit_step = dic_traffic_env_conf.get("HIGHWAY_SPEED_STEP", 5.0)  # Speed limit adjustment step (m/s)
        self.congestion_threshold = dic_traffic_env_conf.get("HIGHWAY_CONGESTION_THRESHOLD", 0.7)  # Occupancy threshold for congestion

        # Apply the segment speed limit (average of all roads) to all roads in the segment
        # This sets both the state and applies it via TraCI
        self.set_segment_speed_limit(segment_avg_speed, unit="mps")
    
    def __deepcopy__(self, memo):
        """Deep copy implementation for Highway object."""
        if id(self) in memo:
            return memo[id(self)]
        
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['traci_conn', 'road_dict']:
                continue
            setattr(result, k, deepcopy(v, memo))
        
        result.traci_conn = None
        result.road_dict = self.road_dict
        
        return result
    
    def _get_lanes_for_road(self, road_id: str) -> List[str]:
        """
        Gets all lane IDs for a given road using TraCI.
        
        Args:
            road_id (str): Road ID (edge ID in SUMO).
            
        Returns:
            List of lane IDs for this road.
        """
        if road_id in self.lanes_by_road:
            return self.lanes_by_road[road_id]
        
        # Get lanes from TraCI
        lane_ids = []
        try:
            num_lanes = self.road_dict.get(road_id, {}).get("numLanes", 0)
            for lane_idx in range(num_lanes):
                lane_id = f"{road_id}_{lane_idx}"
                # Verify lane exists in simulation
                try:
                    self.traci_conn.lane.getLength(lane_id)
                    lane_ids.append(lane_id)
                except traci.TraCIException:
                    continue
        except Exception as e:
            print(f"Warning: Could not get lanes for road {road_id}: {e}")
        
        # Cache the result
        self.lanes_by_road[road_id] = lane_ids
        return lane_ids
    
    def set_segment_speed_limit(self, speed_limit: float, unit: str = "mps"):
        """
        Sets the same speed limit for all roads in this highway segment.
        This is the primary method for segment-level speed control.
        
        Args:
            speed_limit (float): Speed limit value to apply to all roads in the segment.
            unit (str): Unit of speed_limit, either "mps" (meters per second) or "mph" (miles per hour).
                       Default is "mps" for backward compatibility.
        """
        # Convert from mph to m/s if needed
        if unit == "mph":
            # When setting 65 mph (control module's max for "no congestion"), do NOT reduce below
            # network default. After reset_metrics(), baseline keeps network default (no re-apply);
            # applying 65 mph would incorrectly reduce e.g. 74.5 mph -> 65 mph, making LLM slower.
            if speed_limit == 65 and self.default_speed_limits:
                avg_default_mps = np.mean(list(self.default_speed_limits.values()))
                default_mph = avg_default_mps / 0.44704
                if default_mph > 65:
                    speed_limit = default_mph
            # Conversion: 1 mph = 0.44704 m/s
            speed_limit = speed_limit * 0.44704
        elif unit != "mps":
            print(f"Warning: Unknown unit '{unit}', assuming m/s")
        
        # Clamp speed limit to valid range
        speed_limit = max(self.min_speed_limit, min(self.max_speed_limit, speed_limit))
        
        # Apply speed limit to all roads in the segment
        for road_id in self.highway_road_ids:
            # Get lanes for this road (will cache if not already cached)
            lane_ids = self._get_lanes_for_road(road_id)
            
            # Apply speed limit to all lanes of this road
            for lane_id in lane_ids:
                try:
                    # Set speed limit using TraCI
                    self.traci_conn.lane.setMaxSpeed(lane_id, speed_limit)
                except traci.TraCIException as e:
                    print(f"Warning: Could not set speed limit for lane {lane_id}: {e}")
            
            # Update state
            self.current_speed_limits[road_id] = speed_limit
            
            # Record in history
            current_time = self.get_current_time()
            self.speed_limit_history[road_id].append((current_time, speed_limit))
    
    def update_current_measurements(self, simulator_state: Dict[str, Any]):
        """
        Updates highway state based on the global simulator state.
        Calculates features for the current step.
        
        Args:
            simulator_state (dict): Global simulator state dictionary containing:
                - get_lane_vehicles: dict mapping lane_id to list of vehicle IDs
                - get_vehicle_speed: dict mapping vehicle_id to speed
                - get_lane_length: dict mapping lane_id to length
        """
        # Update vehicle counts per lane
        self.dic_lane_vehicle_current_step = defaultdict(list, simulator_state.get("get_lane_vehicles", {}))
        
        # Calculate aggregated metrics per road
        vehicle_speeds = simulator_state.get("get_vehicle_speed", {})
        lane_lengths = simulator_state.get("get_lane_length", {})
        
        for road_id in self.highway_road_ids:
            lane_ids = self._get_lanes_for_road(road_id)
            
            # Aggregate metrics across all lanes of this road
            total_vehicles = 0
            total_speed = 0.0
            total_length = 0.0
            total_occupancy = 0.0
            
            for lane_id in lane_ids:
                vehicles = self.dic_lane_vehicle_current_step.get(lane_id, [])
                lane_length = lane_lengths.get(lane_id, 100.0)  # Default 100m if not available
                
                total_vehicles += len(vehicles)
                total_length += lane_length
                
                # Calculate average speed for this lane
                lane_speeds = [vehicle_speeds.get(veh_id, 0.0) for veh_id in vehicles]
                if lane_speeds:
                    lane_avg_speed = np.mean(lane_speeds)
                    total_speed += lane_avg_speed * len(vehicles)
                
                # Calculate occupancy (vehicles per meter)
                lane_occupancy = len(vehicles) / lane_length if lane_length > 0 else 0.0
                total_occupancy += lane_occupancy
            
            # Store aggregated metrics
            num_lanes = len(lane_ids)
            if num_lanes > 0:
                # If no vehicles, use current speed limit instead of 0 (vehicles can reach speed limit when road is empty)
                if total_vehicles > 0:
                    self.dic_lane_speed_current_step[road_id] = total_speed / total_vehicles
                else:
                    # Use current speed limit for this road when no vehicles present
                    self.dic_lane_speed_current_step[road_id] = self.current_speed_limits.get(road_id, 0.0)
                
                self.dic_lane_density_current_step[road_id] = total_vehicles / total_length if total_length > 0 else 0.0
                self.dic_lane_occupancy_current_step[road_id] = total_occupancy / num_lanes
            else:
                self.dic_lane_speed_current_step[road_id] = self.current_speed_limits.get(road_id, 0.0)
                self.dic_lane_density_current_step[road_id] = 0.0
                self.dic_lane_occupancy_current_step[road_id] = 0.0
        
        # Update features
        self._update_feature()
    
    def _update_feature(self):
        """
        Calculates and stores various state features for the current time step.
        Features are collected at segment level (averaging all roads in the segment).
        """
        dic_feature = {}
        
        # Collect road-level data first
        road_speeds = [
            self.dic_lane_speed_current_step.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        
        road_densities = [
            self.dic_lane_density_current_step.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        
        road_occupancies = [
            self.dic_lane_occupancy_current_step.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        
        current_speed_limits_list = [
            self.current_speed_limits.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        
        default_speed_limits_list = [
            self.default_speed_limits.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        
        # Calculate segment-level features (averages across all roads in segment)
        dic_feature["segment_speed"] = np.mean(road_speeds) if road_speeds else 0.0
        dic_feature["segment_density"] = np.mean(road_densities) if road_densities else 0.0
        dic_feature["segment_occupancy"] = np.mean(road_occupancies) if road_occupancies else 0.0
        dic_feature["segment_speed_limit"] = np.mean(current_speed_limits_list) if current_speed_limits_list else 0.0
        dic_feature["segment_default_speed_limit"] = np.mean(default_speed_limits_list) if default_speed_limits_list else 0.0
        
        # Calculate segment-level congestion indicator
        congested_roads = [
            1.0 if self.dic_lane_occupancy_current_step.get(road_id, 0.0) > self.congestion_threshold else 0.0
            for road_id in self.highway_road_ids
        ]
        dic_feature["segment_congestion_ratio"] = np.mean(congested_roads) if congested_roads else 0.0
        
        # Calculate segment-level speed ratio (average current speed / average speed limit)
        avg_speed_limit = dic_feature["segment_speed_limit"]
        if avg_speed_limit > 0.1:
            dic_feature["segment_speed_ratio"] = dic_feature["segment_speed"] / avg_speed_limit
        else:
            dic_feature["segment_speed_ratio"] = 0.0
        
        # Calculate segment-level speed pressure (average speed limit - average current speed)
        dic_feature["segment_speed_pressure"] = max(0.0, dic_feature["segment_speed_limit"] - dic_feature["segment_speed"])
        
        # Keep road-level features for backward compatibility (if needed)
        dic_feature["road_speed"] = road_speeds
        dic_feature["road_density"] = road_densities
        dic_feature["road_occupancy"] = road_occupancies
        dic_feature["current_speed_limits"] = current_speed_limits_list
        dic_feature["default_speed_limits"] = default_speed_limits_list
        dic_feature["is_congested"] = congested_roads
        
        # Calculate road-level speed ratios and pressures for backward compatibility
        dic_feature["speed_ratio"] = [
            (road_speeds[i] / max(current_speed_limits_list[i], 0.1))
            for i in range(len(road_speeds))
        ]
        dic_feature["speed_pressure"] = [
            max(0.0, current_speed_limits_list[i] - road_speeds[i])
            for i in range(len(road_speeds))
        ]
        
        # Legacy aggregate statistics (for backward compatibility)
        dic_feature["avg_speed"] = dic_feature["segment_speed"]
        dic_feature["avg_density"] = dic_feature["segment_density"]
        dic_feature["avg_occupancy"] = dic_feature["segment_occupancy"]
        dic_feature["congestion_ratio"] = dic_feature["segment_congestion_ratio"]
        
        self.dic_feature = dic_feature
    
    def get_current_time(self) -> float:
        """Returns the current simulation time."""
        return self.traci_conn.simulation.getTime()
    
    def get_feature(self) -> Dict[str, Any]:
        """Returns the calculated dictionary of features for the current step."""
        return self.dic_feature
    
    def get_highway_road_ids(self) -> List[str]:
        """Returns list of highway road IDs managed by this object."""
        return self.highway_road_ids.copy()
    
    def get_current_speed_limits(self) -> Dict[str, float]:
        """Returns current speed limits for all highway roads."""
        return self.current_speed_limits.copy()
    
    def reset_speed_limits_to_default(self):
        """
        Resets all speed limits to their default values.
        Uses the average default speed limit across all roads in the segment.
        """
        if not self.highway_road_ids:
            return
        
        # Calculate average default speed limit for the segment
        default_limits = [
            self.default_speed_limits.get(road_id, 0.0)
            for road_id in self.highway_road_ids
        ]
        avg_default_limit = np.mean(default_limits) if default_limits else 0.0
        
        if avg_default_limit > 0:
            # Use set_segment_speed_limit to set the same limit for all roads
            self.set_segment_speed_limit(avg_default_limit, unit="mps")

