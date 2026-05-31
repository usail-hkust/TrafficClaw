import os
import math
import numpy as np
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, Optional, List

import traci
import sumolib


class Ramp:
    """
    Represents a single ramp metering location for on-ramp control.
    Handles state updates and 0/1 ramp control (open/closed).
    It dynamically discovers its topology from a SUMO network and controls via TraCI.
    """
    
    def __init__(
        self,
        tls_id: str,
        dic_traffic_env_conf: Dict[str, Any],
        traci_conn: traci.connection,
        sumo_net: sumolib.net.Net,
        path_to_log: str,
        adjacency_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes a Ramp object based on a SUMO traffic light system (TLS) for ramp metering.
        
        Args:
            tls_id (str): The ID of the traffic light system in SUMO (ramp metering TLS).
            dic_traffic_env_conf (dict): The traffic environment configuration dictionary.
            traci_conn (traci.connection): The active TraCI connection object.
            sumo_net (sumolib.net.Net): The pre-parsed sumolib network object.
            path_to_log (str): Path to the directory for logging.
            adjacency_info (dict, optional): Information about neighboring ramps and highway segments.
        """
        # Store injected dependencies
        self.tls_id = tls_id
        self.ramp_id = tls_id
        self.ramp_name = tls_id
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.traci_conn = traci_conn
        self.sumo_net = sumo_net
        self.path_to_log = path_to_log
        self.adjacency_info = adjacency_info if adjacency_info else {}
        
        # Initial Validation: Ensure the TLS ID is valid
        try:
            if self.tls_id not in self.traci_conn.trafficlight.getIDList():
                raise ValueError(f"Ramp TLS ID '{self.tls_id}' not found in the running SUMO simulation.")
        except Exception as e:
            raise ValueError(f"Failed to validate Ramp TLS ID '{self.tls_id}': {e}") from e
        
        # Declare attributes that will be populated by _build_conceptual_model
        self.point = {}
        self.phases = []  # Raw SUMO phase definitions
        self.control_phases = []  # Final list of phase NAMES available to agent
        self.phase_name_2_index = {}  # Map name string to actual SUMO phase index
        self.ramp_open_phase_index = -1  # Phase index for RAMP_OPEN
        self.ramp_closed_phase_index = -1  # Phase index for RAMP_CLOSED
        
        # Ramp roads and lanes
        self.ramp_road_id = None  # The ramp road (incoming to merge junction)
        self.highway_road_id = None  # The highway road (outgoing from merge junction)
        self.ramp_lanes = []  # List of ramp lane IDs
        self.highway_lanes = []  # List of highway lane IDs (at merge point)
        
        # Build the conceptual model from SUMO data
        self._build_conceptual_model()
        
        # State Variables
        self.current_phase_index = 0
        try:
            # Get the initial phase from SUMO
            self.current_phase_index = self.traci_conn.trafficlight.getPhase(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: Could not get initial phase for ramp {self.tls_id}. Defaulting to 0. Error: {e}")
        
        self.default_phase_index = self.current_phase_index
        self.previous_phase_index = self.current_phase_index
        
        # Control state
        self.is_open = self.current_phase_index == self.ramp_open_phase_index
        self.control_history = []  # List of (time, action, phase_index) tuples
        
        # Feature storage (similar to Highway class)
        self.dic_lane_vehicle_current_step = defaultdict(list)
        self.dic_lane_speed_current_step = defaultdict(float)
        self.dic_lane_density_current_step = defaultdict(float)
        self.dic_lane_occupancy_current_step = defaultdict(float)
        self.dic_lane_queue_length_current_step = defaultdict(int)
        self.dic_lane_waiting_time_current_step = defaultdict(float)
        self.dic_feature = {}
        
        # Metrics tracking
        self.arrived_vehicles_count = 0  # Count of vehicles that arrived at destination (for throughput)
        self.arrived_vehicles_history = []  # List of (time, count) tuples for throughput calculation
        
    def _build_conceptual_model(self):
        """
        Orchestrates the discovery of ramp topology (lanes, roads) and traffic light logic (phases).
        """
        # Get basic ramp info from the pre-parsed network object
        try:
            tls_node = self.sumo_net.getNode(self.tls_id)
            self.point = {"x": tls_node.getCoord()[0], "y": tls_node.getCoord()[1]}
        except KeyError:
            # Fallback if TLS node is not found in the static network file
            self.point = {"x": 0, "y": 0}
        
        # Step 1: Discover ramp and highway roads
        self._discover_ramp_and_highway_roads()
        
        # Step 2: Discover SUMO phases
        self._discover_phases()
        
        # Step 3: Set ramp_lanes to controlled lanes from signal analysis
        # Clear ramp_lanes first (may have been populated by _discover_ramp_and_highway_roads())
        self.ramp_lanes = []
        
        # Get controlled lanes from signal phase analysis
        controlled_lanes_info = self.get_controlled_lanes_from_signal()
        controlled_lanes = controlled_lanes_info.get("controlled_lanes", [])
        
        if controlled_lanes:
            self.ramp_lanes = controlled_lanes
            print(f"Ramp {self.tls_id}: Using {len(self.ramp_lanes)} controlled lanes as ramp_lanes")
        else:
            print(f"Warning: Ramp {self.tls_id}: No controlled lanes found from signal analysis")
    
    def _discover_ramp_and_highway_roads(self):
        """
        Discovers the ramp road (incoming) and highway road (outgoing) at the merge junction.
        """
        # Get all lanes controlled by this traffic light
        try:
            controlled_lanes = list(set(self.traci_conn.trafficlight.getControlledLanes(self.tls_id)))
        except traci.TraCIException as e:
            print(f"Warning: Could not get controlled lanes for ramp {self.tls_id}. Error: {e}")
            return
        
        if not controlled_lanes:
            return
        
        # Identify ramp lanes (from ramp road) and highway lanes (from highway road)
        ramp_road_ids = set()
        highway_road_ids = set()
        
        for lane_id in controlled_lanes:
            try:
                lane_obj = self.sumo_net.getLane(lane_id)
                road_obj = lane_obj.getEdge()
                road_id = road_obj.getID()
                
                # Skip internal edges
                if road_id.startswith(":"):
                    continue
                
                # Check if this is a ramp road (motorway_link/trunk_link) or highway road
                road_type = road_obj.getType()
                if road_type and ("motorway_link" in road_type or "trunk_link" in road_type):
                    # This is a ramp road
                    ramp_road_ids.add(road_id)
                    # Note: ramp_lanes will be set from controlled lanes in Step 3 of _build_conceptual_model()
                    # So we don't append here to avoid confusion
                else:
                    # Check speed to determine if it's a highway road
                    try:
                        speed = lane_obj.getSpeed()
                        if speed >= 25.0:  # ~90 km/h, typical highway speed
                            highway_road_ids.add(road_id)
                            self.highway_lanes.append(lane_id)
                    except:
                        pass
            except (KeyError, AttributeError) as e:
                pass
        
        # Set primary ramp and highway road IDs
        if ramp_road_ids:
            self.ramp_road_id = list(ramp_road_ids)[0]  # Use first ramp road found
        if highway_road_ids:
            self.highway_road_id = list(highway_road_ids)[0]  # Use first highway road found
    
    def _discover_phases(self):
        """
        Discovers SUMO phases and maps them to ramp control phases (RAMP_OPEN, RAMP_CLOSED).
        """
        try:
            logic_list = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
            if not logic_list:
                print(f"Warning: No TLS program found for ramp {self.tls_id}")
                return
            
            logic = logic_list[0]
            phases = logic.getPhases()
            
            if not phases:
                print(f"Warning: No phases found for ramp {self.tls_id}")
                return
            
            self.phases = phases
            
            # Map phase names to indices
            for idx, phase in enumerate(phases):
                if phase.name:
                    phase_name_upper = phase.name.upper()
                    self.phase_name_2_index[phase.name] = idx
                    
                    # Identify ramp control phases
                    if "RAMP_OPEN" in phase_name_upper or "OPEN" in phase_name_upper:
                        self.ramp_open_phase_index = idx
                        if "RAMP_OPEN" not in self.control_phases:
                            self.control_phases.append("RAMP_OPEN")
                    elif "RAMP_CLOSED" in phase_name_upper or "CLOSED" in phase_name_upper:
                        self.ramp_closed_phase_index = idx
                        if "RAMP_CLOSED" not in self.control_phases:
                            self.control_phases.append("RAMP_CLOSED")
            
            # Validate that we found both phases
            if self.ramp_open_phase_index == -1 or self.ramp_closed_phase_index == -1:
                print(f"Warning: Ramp {self.tls_id} missing RAMP_OPEN or RAMP_CLOSED phase")
            
        except (traci.TraCIException, IndexError, AttributeError) as e:
            print(f"Warning: Failed to discover phases for ramp {self.tls_id}: {e}")
    
    def get_controlled_lanes_from_signal(self) -> Dict[str, Any]:
        """
        Identifies which lanes are controlled by this ramp metering signal by analyzing the signal phases.
        Returns information about controlled lanes (ramp lanes that can be red/green) and their connections.
        
        Returns:
            Dict containing:
                - controlled_lanes: List of lane IDs that are controlled (can change from red to green)
                - upstream_lanes: Dict mapping controlled lane to its upstream lanes (2 hops)
                - downstream_lanes: Dict mapping controlled lane to its downstream lanes (2 hops)
        """
        result = {
            "controlled_lanes": [],
            "upstream_lanes": {},
            "downstream_lanes": {}
        }
        
        try:
            # Get controlled links using TraCI (same method as intersection.py)
            controlled_links = self.traci_conn.trafficlight.getControlledLinks(self.tls_id)
            if not controlled_links:
                return result
            
            # Get the traffic light logic for phase states
            logic_list = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
            if not logic_list:
                return result
            
            logic = logic_list[0]
            phases = logic.getPhases()
            
            if not phases:
                return result
            
            # Analyze phase states to identify which lanes are controlled
            # A lane is "controlled" if it changes from red ('r') to green ('G') between phases
            controlled_lane_indices = set()
            
            # Compare RAMP_OPEN and RAMP_CLOSED phases
            if self.ramp_open_phase_index >= 0 and self.ramp_closed_phase_index >= 0:
                open_state = phases[self.ramp_open_phase_index].state
                closed_state = phases[self.ramp_closed_phase_index].state
                
                # Find positions where state changes from 'r' to 'G' (or vice versa)
                for i, (open_char, closed_char) in enumerate(zip(open_state, closed_state)):
                    if (closed_char.lower() == 'r' and open_char.upper() == 'G') or \
                       (open_char.lower() == 'r' and closed_char.upper() == 'G'):
                        controlled_lane_indices.add(i)
            
            # Get the actual lane IDs for controlled indices
            # controlled_links is a list of link groups, where each link group contains tuples (from_lane, to_lane, via_lane)
            for lane_idx in controlled_lane_indices:
                if lane_idx < len(controlled_links):
                    link_group = controlled_links[lane_idx]
                    if link_group and len(link_group) > 0:
                        # Each link_group is a list of links, get the first link
                        link = link_group[0]
                        if link and len(link) >= 2:
                            incoming_lane = link[0]  # First element is from_lane_id
                            if incoming_lane and not incoming_lane.startswith(":"):
                                result["controlled_lanes"].append(incoming_lane)
                                
                                # Find upstream lanes (2 hops)
                                result["upstream_lanes"][incoming_lane] = self._get_upstream_lanes(incoming_lane, max_hops=2)
                                
                                # Find downstream lanes (2 hops)
                                result["downstream_lanes"][incoming_lane] = self._get_downstream_lanes(incoming_lane, max_hops=2)
            
        except (traci.TraCIException, IndexError, AttributeError) as e:
            print(f"Warning: Failed to get controlled lanes for ramp {self.tls_id}: {e}")
        
        return result
    
    def _get_upstream_lanes(self, lane_id: str, max_hops: int = 2) -> List[str]:
        """
        Get upstream lanes within max_hops using BFS.
        
        Args:
            lane_id: Starting lane ID
            max_hops: Maximum number of hops to traverse
            
        Returns:
            List of upstream lane IDs
        """
        upstream_lanes = []
        visited = set()
        queue = [(lane_id, 0)]  # (lane_id, hop_count)
        
        while queue:
            current_lane, hops = queue.pop(0)
            
            if current_lane in visited or hops > max_hops:
                continue
            
            visited.add(current_lane)
            
            # Add to result if not the starting lane
            if current_lane != lane_id:
                upstream_lanes.append(current_lane)
            
            # Get predecessors (incoming lanes)
            try:
                lane_obj = self.sumo_net.getLane(current_lane)
                edge_obj = lane_obj.getEdge()
                
                # Get incoming edges
                from_node = edge_obj.getFromNode()
                incoming_edges = from_node.getIncoming()
                
                for incoming_edge in incoming_edges:
                    # Skip internal edges
                    if incoming_edge.getID().startswith(":"):
                        continue
                    
                    # Add all lanes from incoming edge
                    for incoming_lane in incoming_edge.getLanes():
                        incoming_lane_id = incoming_lane.getID()
                        if incoming_lane_id not in visited:
                            queue.append((incoming_lane_id, hops + 1))
            except (KeyError, AttributeError) as e:
                pass
        
        return upstream_lanes
    
    def _get_downstream_lanes(self, lane_id: str, max_hops: int = 2) -> List[str]:
        """
        Get downstream lanes within max_hops using BFS.
        
        Args:
            lane_id: Starting lane ID
            max_hops: Maximum number of hops to traverse
            
        Returns:
            List of downstream lane IDs
        """
        downstream_lanes = []
        visited = set()
        queue = [(lane_id, 0)]  # (lane_id, hop_count)
        
        while queue:
            current_lane, hops = queue.pop(0)
            
            if current_lane in visited or hops > max_hops:
                continue
            
            visited.add(current_lane)
            
            # Add to result if not the starting lane
            if current_lane != lane_id:
                downstream_lanes.append(current_lane)
            
            # Get successors (outgoing lanes)
            try:
                lane_obj = self.sumo_net.getLane(current_lane)
                
                # Get outgoing connections from this lane
                outgoing_connections = lane_obj.getOutgoing()
                
                for connection in outgoing_connections:
                    to_lane = connection.getToLane()
                    to_lane_id = to_lane.getID()
                    
                    # Skip internal lanes (they are intermediate, we want the final destination)
                    if to_lane_id.startswith(":"):
                        # For internal lanes, find their outgoing connections
                        try:
                            internal_outgoing = to_lane.getOutgoing()
                            for internal_conn in internal_outgoing:
                                final_lane = internal_conn.getToLane()
                                final_lane_id = final_lane.getID()
                                if final_lane_id not in visited:
                                    queue.append((final_lane_id, hops + 1))
                        except:
                            pass
                    else:
                        if to_lane_id not in visited:
                            queue.append((to_lane_id, hops + 1))
            except (KeyError, AttributeError) as e:
                pass
        
        return downstream_lanes
    
    def set_ramp_state(self, is_open: bool):
        """
        Sets the ramp state (open or closed).
        
        Args:
            is_open (bool): True to open the ramp (RAMP_OPEN phase), False to close it (RAMP_CLOSED phase).
        """
        target_phase_index = self.ramp_open_phase_index if is_open else self.ramp_closed_phase_index
        
        if target_phase_index == -1:
            print(f"Warning: Cannot set ramp state for {self.tls_id}: phase index not found")
            return
        
        try:
            self.traci_conn.trafficlight.setPhase(self.tls_id, target_phase_index)
            self.current_phase_index = target_phase_index
            self.is_open = is_open
        except traci.TraCIException as e:
            print(f"Error: Failed to set ramp state for {self.tls_id}: {e}")
    
    def get_ramp_state(self) -> bool:
        """
        Gets the current ramp state.
        
        Returns:
            bool: True if ramp is open, False if closed.
        """
        try:
            current_phase = self.traci_conn.trafficlight.getPhase(self.tls_id)
            self.current_phase_index = current_phase
            self.is_open = (current_phase == self.ramp_open_phase_index)
            return self.is_open
        except traci.TraCIException as e:
            print(f"Warning: Could not get ramp state for {self.tls_id}: {e}")
            return self.is_open
    
    def update_current_measurements(self, simulator_state: Dict[str, Any]):
        """
        Updates ramp state based on the global simulator state.
        Calculates features for the current step.
        
        Args:
            simulator_state (dict): Global simulator state dictionary containing:
                - get_lane_vehicles: dict mapping lane_id to list of vehicle IDs
                - get_vehicle_speed: dict mapping vehicle_id to speed
                - get_lane_length: dict mapping lane_id to length
                - get_waiting_vehicles: dict mapping vehicle_id to waiting time (optional)
        """
        # Update vehicle counts per lane
        self.dic_lane_vehicle_current_step = defaultdict(list, simulator_state.get("get_lane_vehicles", {}))
        
        # Calculate metrics for ramp lanes
        vehicle_speeds = simulator_state.get("get_vehicle_speed", {})
        lane_lengths = simulator_state.get("get_lane_length", {})
        waiting_vehicles = simulator_state.get("get_waiting_vehicles", {})
        
        # Aggregate metrics across ramp lanes
        total_vehicles = 0
        total_speed = 0.0
        total_length = 0.0
        total_occupancy = 0.0
        total_queue_length = 0
        total_waiting_time = 0.0
        
        for lane_id in self.ramp_lanes:
            vehicles = self.dic_lane_vehicle_current_step.get(lane_id, [])
            lane_length = lane_lengths.get(lane_id, 100.0)  # Default 100m if not available
            
            total_vehicles += len(vehicles)
            total_length += lane_length
            
            # Calculate average speed for this lane
            lane_speeds = [vehicle_speeds.get(veh_id, 0.0) for veh_id in vehicles]
            lane_avg_speed = 0.0
            if lane_speeds:
                lane_avg_speed = np.mean(lane_speeds)
                total_speed += lane_avg_speed * len(vehicles)
            
            # Calculate occupancy (vehicles per meter)
            lane_occupancy = len(vehicles) / lane_length if lane_length > 0 else 0.0
            total_occupancy += lane_occupancy
            
            # Calculate queue length (vehicles with speed < 0.1 m/s)
            queue_vehicles = [veh_id for veh_id in vehicles if vehicle_speeds.get(veh_id, 0.0) < 0.1]
            lane_queue_length = len(queue_vehicles)
            total_queue_length += lane_queue_length
            
            # Calculate waiting time (sum of waiting times for waiting vehicles in this lane only)
            # Only count vehicles that are actually waiting (speed < 0.1 m/s)
            # This matches the logic used in signal_timing.py
            # Handle both old format ({v_id: waiting_time}) and new format ({v_id: {"time": waiting_time, "lane": lane_id}})
            lane_waiting_time = 0.0
            for veh_id in queue_vehicles:  # Only iterate over waiting vehicles
                waiting_info = waiting_vehicles.get(veh_id, 0.0)
                if isinstance(waiting_info, dict):
                    # New format: extract "time" field
                    lane_waiting_time += waiting_info.get("time", 0.0)
                else:
                    # Old format: waiting_info is directly the waiting time
                    lane_waiting_time += float(waiting_info)
            total_waiting_time += lane_waiting_time
            
            # Store lane-level metrics
            self.dic_lane_speed_current_step[lane_id] = lane_avg_speed
            self.dic_lane_density_current_step[lane_id] = len(vehicles) / lane_length if lane_length > 0 else 0.0
            self.dic_lane_occupancy_current_step[lane_id] = lane_occupancy
            self.dic_lane_queue_length_current_step[lane_id] = lane_queue_length
            self.dic_lane_waiting_time_current_step[lane_id] = lane_waiting_time
        
        # Calculate ramp-level aggregated metrics
        num_ramp_lanes = len(self.ramp_lanes)
        if num_ramp_lanes > 0:
            self.dic_feature["ramp_speed"] = total_speed / total_vehicles if total_vehicles > 0 else 0.0
            self.dic_feature["ramp_density"] = total_vehicles / total_length if total_length > 0 else 0.0
            self.dic_feature["ramp_occupancy"] = total_occupancy / num_ramp_lanes
            self.dic_feature["ramp_queue_length"] = total_queue_length
            self.dic_feature["ramp_waiting_time"] = total_waiting_time
            self.dic_feature["ramp_vehicle_count"] = total_vehicles
        else:
            self.dic_feature["ramp_speed"] = 0.0
            self.dic_feature["ramp_density"] = 0.0
            self.dic_feature["ramp_occupancy"] = 0.0
            self.dic_feature["ramp_queue_length"] = 0
            self.dic_feature["ramp_waiting_time"] = 0.0
            self.dic_feature["ramp_vehicle_count"] = 0
        
        # Add ramp state
        self.dic_feature["is_open"] = self.is_open
    
    def get_feature(self) -> Dict[str, Any]:
        """Returns the calculated dictionary of features for the current step."""
        return self.dic_feature
    
    def get_current_time(self) -> float:
        """Returns the current simulation time."""
        return self.traci_conn.simulation.getTime()
    
    def get_ramp_info(self) -> Dict[str, Any]:
        """
        Returns information about this ramp.
        
        Returns:
            dict: Ramp information including roads, lanes, phases, and current state.
        """
        return {
            "ramp_id": self.ramp_id,
            "tls_id": self.tls_id,
            "point": self.point,
            "ramp_road_id": self.ramp_road_id,
            "highway_road_id": self.highway_road_id,
            "ramp_lanes": self.ramp_lanes,
            "highway_lanes": self.highway_lanes,
            "control_phases": self.control_phases,
            "ramp_open_phase_index": self.ramp_open_phase_index,
            "ramp_closed_phase_index": self.ramp_closed_phase_index,
            "current_phase_index": self.current_phase_index,
            "is_open": self.is_open
        }
    
    def __deepcopy__(self, memo):
        """Deep copy implementation for Ramp objects."""
        if id(self) in memo:
            return memo[id(self)]
        
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['traci_conn', 'sumo_net']:
                continue
            setattr(result, k, deepcopy(v, memo))
        
        result.traci_conn = None
        result.sumo_net = self.sumo_net
        
        return result
