import os
import time
import math
import random
import json
import pickle
import subprocess
import shutil
import numpy as np
from multiprocessing import Process
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional, List
import networkx as nx

import traci
import sumolib

from environment.intersection import Intersection
from environment.highway import Highway
from environment.ramp import Ramp
from environment.subway import SubwayStation, SubwayLine
from environment.bus import BusStation, BusLine

# Global dictionaries (can be part of config or discovered)
location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {v: k for k, v in location_dict.items()}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

# Angles represent the direction of travel (Heading) towards the intersection (calculated via atan2(dy, dx))
angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]  # Eastbound, Northbound, Westbound, Southbound, Eastbound
# Orients map these Headings to their Origin (Standard Convention)
orients = ['W', 'S', 'E', 'N', 'W', 'S', 'E', 'N']

DEFAULT_YELLOW_TIME = 5  # Default yellow time if not specified


def find_free_port(start_port: int = 10000, end_port: int = 60000, max_attempts: int = 100) -> int:
    """
    Find a free port for SUMO TraCI connection.

    This function attempts to find an available port by:
    1. Trying to bind a socket to random ports in the given range
    2. Verifying the port is not in TIME_WAIT state

    Args:
        start_port: Minimum port number to try
        end_port: Maximum port number to try
        max_attempts: Maximum number of attempts before raising an error

    Returns:
        An available port number

    Raises:
        RuntimeError: If no free port can be found after max_attempts
    """
    import socket

    for attempt in range(max_attempts):
        # Generate a random port in the range
        port = random.randint(start_port, end_port)

        try:
            # Try to create a socket and bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Set SO_REUSEADDR to avoid TIME_WAIT issues
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('localhost', port))
                # If we get here, the port is available
                return port
        except OSError:
            # Port is in use, try another one
            continue

    raise RuntimeError(
        f"Could not find a free port after {max_attempts} attempts "
        f"in range [{start_port}, {end_port}]. "
        "Please check for orphaned SUMO processes."
    )


def load_line_station_mapping():
    """
    Load line-station mapping from environment/line_station_mapping.json.
    Returns a dictionary with 'subway' and 'bus' keys mapping route_id to list of station_ids.
    """
    import os
    mapping_file = os.path.join(os.path.dirname(__file__), "line_station_mapping.json")
    
    if not os.path.exists(mapping_file):
        print(f"Warning: line_station_mapping.json not found at {mapping_file}")
        return {"subway": {}, "bus": {}}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load line_station_mapping.json: {e}")
        return {"subway": {}, "bus": {}}


class SUMOEnv:
    """
    Generic SUMO Simulation Environment.

    Handles the overall simulation lifecycle, interacts with the SUMO engine via TraCI,
    manages multiple Intersection objects, and provides an API compatible with
    the original CityFlow-based environment.
    """

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path, inter_phase_mapping=None, control_modules=None, config_dir_name=None, seed=None):
        """
        Initializes the SUMO Environment.

        Args:
            path_to_log (str): Path to the logging directory.
            path_to_work_directory (str): Path to the working directory containing SUMO config files.
            dic_traffic_env_conf (dict): Environment configuration dictionary.
            dic_path (dict): Dictionary containing paths to data files.
            inter_phase_mapping (dict, optional): Dictionary mapping intersection_id to a list
                                   of allowed phase name strings.
            control_modules (list, optional): List of control module names to enable.
                                   Available modules: 'signal_timing', 'subway_scheduling', 'bus_scheduling'
                                   If None, defaults to ['signal_timing']
            config_dir_name (str, optional): Directory name from sumo config path (e.g., "jinan").
                                   Used to determine control config save location.
            seed (int, optional): Random seed for simulation. If provided, will be used for
                                   initialization reset to ensure reproducibility.
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.inter_phase_mapping = inter_phase_mapping if inter_phase_mapping is not None else {}
        self.init_seed = seed  # Store seed for initialization reset

        # Control modules configuration
        if control_modules is None:
            control_modules = ['signal_timing']  # Default to signal control only
        self.control_modules = control_modules
        self.config_dir_name = config_dir_name  # Store config_dir_name for control module initialization
        self.enabled_controls = {}  # Will be populated after reset()

        # Determine config path for module access (used by taxi scheduling for TAZ inference)
        sumocfg_file = dic_traffic_env_conf.get("SUMOCFG_FILE", "map.sumocfg")
        data_path = dic_path.get("PATH_TO_DATA", path_to_work_directory)
        self.config_path = os.path.join(data_path, sumocfg_file)

        # --- SUMO Process and TraCI Connection Management ---
        self.sumo_process = None
        self.traci_conn = None
        self._simulation_running = False
        self._traci_healthy = True  # Flag to track TraCI connection health
        self.sumo_net = None  # Cache for the parsed sumolib network object
        # --- End of SUMO Management ---

        self.roadnet = None # Kept for compatibility, but self.sumo_net is the primary source
        self.roads_data = {}  # Legacy format: road_id -> road_json
        self.intersections_data = {}  # Legacy format: tls_id -> inter_json

        self.intersection_dict = {}  # inter_id -> Intersection object
        self.inter_info_dict = {}  # Compatibility: inter_id -> parsed info dict
        self.id_to_index = {}  # Maps intersection id to index (deprecated, kept for compatibility)
        # list_inter_log removed - no longer needed for RL training
        
        self.highway_dict = {}  # highway_id -> Highway object
        
        self.ramp_dict = {}  # ramp_id -> Ramp object
        self.ramp_info_dict = {}  # ramp_id -> ramp_info dict

        self.subway_stations = {}  # station_id -> SubwayStation
        self.subway_lines = {}     # route_id -> SubwayLine
        
        self.bus_stations = {}     # station_id -> BusStation
        self.bus_lines = {}        # route_id -> BusLine

        # Zone infrastructure (built during runtime initialization)
        self.zone_dict = {}        # zone_id -> zone info dict
        self.zone_graph = None     # NetworkX DiGraph of zone adjacency
        self.transit_graph = None  # NetworkX DiGraph of transit network (routes and stations)
        self.bus_route_info = {}   # route_id -> detailed route info with realtime data
        self._zone_manager = None  # ZoneManager instance
        self._transit_builder = None  # TransitGraphBuilder instance
        self._edge_to_zone = {}    # edge_id -> zone_id mapping
        self._lane_to_zone = {}    # lane_id -> zone_id mapping

        self.list_lanes = []  # List of all unique lane IDs in the network
        self.lane_length = {}  # lane_id -> length
        self.route_stop_mapping = {}  # route_id -> [stop_id1, stop_id2, ...] (cached from route files)
        self.route_travel_times = {}  # route_id -> {stop1|stop2: travel_time, ...} (cached from route files)
        self.subway_routes = []  # List of subway route IDs (cached)
        self.bus_routes = []  # List of bus route IDs (cached)
        self.subway_stops = []  # List of subway stop IDs (cached)
        self.bus_stops = []  # List of bus stop IDs (cached)
        
        # TLS↔node mappings for robust planning (cached)
        self.tls_to_nodes = {}   # TLS id -> [node ids]
        self.node_to_tls = {}    # node id -> TLS id
        
        # Road network graphs cache (built lazily when needed)
        self._road_network_graphs = None
        
        # Highway data (built during runtime data initialization and cached)
        self.highway_info_dict = {}  # road_id -> road_info (only highway roads, subset of road_dict)
        self.highway_subgraph = None  # NetworkX subgraph containing only highway roads
        self.highway_segment_graph = None  # NetworkX DiGraph connecting highway segments (segment_id -> segment_id)
        
        # Ramp graphs (built during build_road_network_graphs)
        self.ramp_lane_graph = None  # NetworkX DiGraph connecting ramps to controlled lanes and their 2-hop neighbors
        self.ramp_lane_dict = {}  # Dictionary mapping lane_id to lane metadata for lanes in ramp_lane_graph

        # ==================== FOUNDATION LAYER (Layer 1) ====================
        # Complete network graphs and dictionaries covering ALL lanes/roads/transit
        self.network_graphs = {
            "lane_graph": None,    # Full lane connectivity graph (all non-internal lanes)
            "road_graph": None,    # Full road connectivity graph (all roads)
            "transit_graph": None  # Full transit network graph (routes + stations)
        }
        self.network_dicts = {
            "lane_dict": {},       # Full lane metadata dictionary
            "road_dict": {},       # Full road metadata dictionary
            "station_dict": {}     # Full station metadata dictionary
        }
        # ====================================================================

        self.system_states = {}  # Stores results from bulk TraCI API calls
        self.current_time = 0.0
        self.waiting_vehicle_list = {}
        self.waiting_passenger_list = {}  # person_id -> waiting_time (for subway/bus passengers)
        self._subscribed_vehicle_ids = set()
        self._last_passenger_update_time = -1.0
        # --- Travel-time aggregator (fallback for SUMO versions without getArrivedMeanTravelTime) ---
        self._depart_time_by_vehicle = {}   # vid -> depart_time
        self._arrived_tt_sum = 0.0          # sum of travel times of arrived vehicles
        self._arrived_count = 0             # count of arrived vehicles
        self._arrived_vehicle_tt = {}       # optional: vid -> travel time (for debugging/analysis)
        # --- Highway-only travel-time aggregator (vehicles that touched highway roads) ---
        self._highway_road_ids = set()
        self._lane_to_road_id = {}
        self._highway_vehicle_ids = set()
        self._highway_arrived_tt_sum = 0.0
        self._highway_arrived_count = 0
        self._highway_arrived_vehicle_tt = {}
        # --- Global travel-time aggregator (persists across reset_metrics) ---
        # These variables accumulate travel times across checkpoint intervals and are NOT reset by reset_metrics()
        # They are only reset by reset() when starting a completely new simulation
        self._global_arrived_tt_sum = 0.0          # Global sum of travel times of all arrived vehicles
        self._global_arrived_count = 0             # Global count of all arrived vehicles
        self._global_arrived_vehicle_tt = {}       # Global: vid -> travel time (for all vehicles across checkpoints)
        self._global_highway_arrived_tt_sum = 0.0  # Global sum of highway vehicle travel times
        self._global_highway_arrived_count = 0     # Global count of highway vehicles
        self._global_highway_arrived_vehicle_tt = {}  # Global: vid -> travel time (for highway vehicles across checkpoints)
        # --- Update timing control ---
        self._last_update_time = -600.0  # Initialize to allow first update immediately
        self._last_highway_update_time = -1.0

        # Port will be dynamically allocated in reset()
        self._current_port = None
        # --- Configuration Validation ---
        if self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15) <= self.dic_traffic_env_conf.get("YELLOW_TIME", DEFAULT_YELLOW_TIME):
            print("Warning: MIN_ACTION_TIME should ideally be greater than YELLOW_TIME.")

        # --- Ensure Log Directory Exists ---
        os.makedirs(self.path_to_log, exist_ok=True)

        # --- Load static network data (cached if available) ---
        self._load_static_network_data()
        
        # --- Load or build inter_info_dict and road_network_graphs ---
        # Try to load from cache first, if not available, run reset() to build them
        self._initialize_runtime_data()

        print("SUMO Environment initialized. Call reset() to start simulation.")

    def _get_static_data_cache_path(self):
        """Get the path to the static data cache file."""
        data_path = self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory)
        net_file_name = self.dic_traffic_env_conf.get("ROADNET_FILE", "map.net.xml")
        # Create cache filename based on network file name
        # Remove extension and any path separators from filename
        base_name = os.path.splitext(os.path.basename(net_file_name))[0]
        cache_filename = f"static_network_data_{base_name}.json"
        cache_path = os.path.join(data_path, cache_filename)
        
        # Ensure directory exists
        os.makedirs(data_path, exist_ok=True)
        
        return cache_path
    
    def _get_runtime_data_cache_path(self):
        """Get the path to the runtime data cache file (inter_info_dict and road_network_graphs)."""
        data_path = self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory)
        net_file_name = self.dic_traffic_env_conf.get("ROADNET_FILE", "map.net.xml")
        # Create cache filename based on network file name
        base_name = os.path.splitext(os.path.basename(net_file_name))[0]
        cache_filename = f"runtime_network_data_{base_name}.pkl"
        cache_path = os.path.join(data_path, cache_filename)
        
        # Ensure directory exists
        os.makedirs(data_path, exist_ok=True)
        
        return cache_path
    
    def _extract_route_stop_mapping(self) -> Dict[str, Any]:
        """
        Extract route-to-stops mapping from route files.
        Parses XML route files to find <route> elements with <stop> children.
        Also extracts travel times between consecutive stops using:
        travel_time = (next_until - next_duration) - current_until
        
        Returns a dictionary with:
        {
            "route_stop_mapping": {route_id: [stop_id1, stop_id2, ...]},
            "route_travel_times": {route_id: {(stop1, stop2): travel_time, ...}},
            "subway_routes": [route_id1, route_id2, ...],
            "bus_routes": [route_id1, route_id2, ...],
            "subway_stops": [stop_id1, stop_id2, ...],
            "bus_stops": [stop_id1, stop_id2, ...]
        }
        """
        import xml.etree.ElementTree as ET
        
        route_stop_mapping = {}
        route_travel_times = {}
        subway_routes = []
        bus_routes = []
        subway_stops_set = set()
        bus_stops_set = set()
        
        # Get route files from SUMO config file
        route_files = []
        sumocfg_file = self.dic_traffic_env_conf.get("SUMOCFG_FILE", "")
        data_path = self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory)
        
        if sumocfg_file:
            sumocfg_path = os.path.join(data_path, sumocfg_file)
            if os.path.exists(sumocfg_path):
                try:
                    tree = ET.parse(sumocfg_path)
                    root = tree.getroot()
                    route_files_elem = root.find('.//route-files')
                    if route_files_elem is not None:
                        route_files_value = route_files_elem.get('value', '')
                        route_files = [f.strip() for f in route_files_value.split(',') if f.strip()]
                        print(f"Found route files in config: {route_files}")
                except Exception as e:
                    print(f"Warning: Failed to parse SUMO config {sumocfg_path}: {e}")
        
        for route_file in route_files:
            route_file_path = os.path.join(data_path, route_file)
            
            if not os.path.exists(route_file_path):
                continue
            
            try:
                tree = ET.parse(route_file_path)
                root = tree.getroot()
                
                # Find all <route> elements
                for route_elem in root.findall('.//route'):
                    route_id = route_elem.get('id')
                    if not route_id:
                        continue
                    
                    # Find all <stop> children with timing information
                    stops = []
                    stop_elements = []
                    for stop_elem in route_elem.findall('stop'):
                        stop_id = stop_elem.get('busStop')
                        if stop_id:
                            stops.append(stop_id)
                            stop_elements.append(stop_elem)
                    
                    if stops:
                        route_stop_mapping[route_id] = stops
                        
                        # Extract travel times between consecutive stops
                        travel_times = {}
                        for i in range(len(stop_elements) - 1):
                            current_stop = stop_elements[i]
                            next_stop = stop_elements[i + 1]
                            
                            current_stop_id = current_stop.get('busStop')
                            next_stop_id = next_stop.get('busStop')
                            
                            # Get timing attributes
                            current_until = current_stop.get('until')
                            current_duration = current_stop.get('duration')
                            next_until = next_stop.get('until')
                            next_duration = next_stop.get('duration')
                            
                            # Calculate travel time if all attributes are present
                            if current_until and current_duration and next_until and next_duration:
                                try:
                                    current_until_val = float(current_until)
                                    current_duration_val = float(current_duration)
                                    next_until_val = float(next_until)
                                    next_duration_val = float(next_duration)
                                    
                                    # Formula: travel_time = (next_arrival) - (current_arrival)
                                    # where arrival = until - duration
                                    travel_time = (next_until_val - next_duration_val) - (current_until_val - current_duration_val)
                                    
                                    # Store as tuple key for easy lookup
                                    travel_times[f"{current_stop_id}|{next_stop_id}"] = travel_time
                                except ValueError:
                                    pass
                        
                        if travel_times:
                            route_travel_times[route_id] = travel_times
                        
                        # Classify route by prefix
                        if route_id.startswith('subway_'):
                            subway_routes.append(route_id)
                            subway_stops_set.update(stops)
                        elif route_id.startswith('bus_'):
                            bus_routes.append(route_id)
                            bus_stops_set.update(stops)
                        
            except Exception as e:
                print(f"Warning: Failed to parse route file {route_file_path}: {e}")
        
        return {
            "route_stop_mapping": route_stop_mapping,
            "route_travel_times": route_travel_times,
            "subway_routes": subway_routes,
            "bus_routes": bus_routes,
            "subway_stops": sorted(list(subway_stops_set)),
            "bus_stops": sorted(list(bus_stops_set))
        }
    
    def _load_static_network_data(self):
        """
        Load static network data from cache if available, otherwise compute and cache it.
        This includes: roads_data, intersections_data, lane_length, route_stop_mapping.
        Note: sumo_net is loaded separately as it's needed for runtime operations.
        """
        cache_path = self._get_static_data_cache_path()
        
        # Try to load static computed data from cache first
        if os.path.exists(cache_path):
            try:
                print(f"Loading static network data from cache: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                self.roads_data = cached_data.get("roads_data", {})
                self.intersections_data = cached_data.get("intersections_data", {})
                self.lane_length = cached_data.get("lane_length", {})
                self.tls_to_nodes = cached_data.get("tls_to_nodes", {})
                self.node_to_tls = cached_data.get("node_to_tls", {})
                self.route_stop_mapping = cached_data.get("route_stop_mapping", {})
                self.route_travel_times = cached_data.get("route_travel_times", {})
                self.subway_routes = cached_data.get("subway_routes", [])
                self.bus_routes = cached_data.get("bus_routes", [])
                self.subway_stops = cached_data.get("subway_stops", [])
                self.bus_stops = cached_data.get("bus_stops", [])
                
                # Update config
                self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(self.intersections_data)
                
                print(f"Loaded cached static data: {len(self.roads_data)} roads, "
                      f"{len(self.intersections_data)} intersections, "
                      f"{len(self.lane_length)} lanes, "
                      f"{len(self.tls_to_nodes)} TLS mappings, "
                      f"{len(self.route_stop_mapping)} route-stop mappings, "
                      f"{len(self.route_travel_times)} routes with travel times, "
                      f"{len(self.subway_routes)} subway routes, "
                      f"{len(self.bus_routes)} bus routes, "
                      f"{len(self.subway_stops)} subway stops, "
                      f"{len(self.bus_stops)} bus stops")
                
                # Still need to load sumo_net for runtime operations (but skip data population)
                if self.sumo_net is None:
                    self._load_roadnet()
                
                return
            except Exception as e:
                print(f"Warning: Failed to load cache from {cache_path}: {e}")
                print("Will recompute static data...")
        
        # Cache not available or failed to load - compute and save
        print("Computing static network data (this may take a while)...")
        
        # Load sumo_net and populate all data
        if self.sumo_net is None:
            self._load_roadnet()
        elif not self.roads_data or not self.intersections_data or not self.tls_to_nodes:
            # If sumo_net is loaded but data is missing, populate it
            self._load_roadnet()
        
        # Get lane lengths (if not already computed)
        if not self.lane_length:
            self._get_lane_length()
        
        # Extract route-stop mapping from route files
        print("Extracting route-stop mappings from route files...")
        mapping_data = self._extract_route_stop_mapping()
        self.route_stop_mapping = mapping_data["route_stop_mapping"]
        self.route_travel_times = mapping_data["route_travel_times"]
        self.subway_routes = mapping_data["subway_routes"]
        self.bus_routes = mapping_data["bus_routes"]
        self.subway_stops = mapping_data["subway_stops"]
        self.bus_stops = mapping_data["bus_stops"]
        
        print(f"Extracted {len(self.route_stop_mapping)} route-stop mappings:")
        print(f"  - {len(self.subway_routes)} subway routes with {len(self.subway_stops)} stops")
        print(f"  - {len(self.bus_routes)} bus routes with {len(self.bus_stops)} stops")
        print(f"  - {len(self.route_travel_times)} routes with travel time data")
        
        # Save to cache
        try:
            # Convert numpy types to Python native types for JSON serialization
            cache_data = {
                "roads_data": self.roads_data,
                "intersections_data": self.intersections_data,
                "lane_length": self.lane_length,
                "tls_to_nodes": self.tls_to_nodes,
                "node_to_tls": self.node_to_tls,
                "route_stop_mapping": self.route_stop_mapping,
                "route_travel_times": self.route_travel_times,
                "subway_routes": self.subway_routes,
                "bus_routes": self.bus_routes,
                "subway_stops": self.subway_stops,
                "bus_stops": self.bus_stops,
                "num_intersections": len(self.intersections_data),
                "cache_version": "1.4"
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"Static network data cached to: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_public_transport(self):
        """
        Initialize subway and bus stations/lines based on cached data.
        This replaces the old infrastructure_ids.py and line_station_mapping.json approach.
        Should be called after _load_static_network_data() and after TraCI is connected.
        """
        if not self.route_stop_mapping:
            print("Warning: route_stop_mapping is empty, skipping public transport initialization")
            return
        
        print("Initializing public transport system from cached data...")
        
        # Initialize subway stations
        for stop_id in self.subway_stops:
            if stop_id not in self.subway_stations:
                try:
                    # Get stop position from TraCI
                    pos = self.traci_conn.busstop.getPosition(stop_id)
                    lane_id = self.traci_conn.busstop.getLaneID(stop_id)
                    
                    self.subway_stations[stop_id] = SubwayStation(
                        traci_conn=self.traci_conn,
                        station_id=stop_id
                    )
                except Exception as e:
                    print(f"Warning: Failed to initialize subway station {stop_id}: {e}")
        
        # Initialize subway lines
        for route_id in self.subway_routes:
            if route_id not in self.subway_lines:
                stops = self.route_stop_mapping.get(route_id, [])
                station_objects = [self.subway_stations.get(stop_id) for stop_id in stops]
                station_objects = [s for s in station_objects if s is not None]
                
                self.subway_lines[route_id] = SubwayLine(
                    traci_conn=self.traci_conn,
                    route_id=route_id,
                    stations=station_objects
                )
        
        # Initialize bus stations
        for stop_id in self.bus_stops:
            if stop_id not in self.bus_stations:
                try:
                    pos = self.traci_conn.busstop.getPosition(stop_id)
                    lane_id = self.traci_conn.busstop.getLaneID(stop_id)
                    
                    self.bus_stations[stop_id] = BusStation(
                        traci_conn=self.traci_conn,
                        station_id=stop_id
                    )
                except Exception as e:
                    print(f"Warning: Failed to initialize bus station {stop_id}: {e}")
        
        # Initialize bus lines
        for route_id in self.bus_routes:
            if route_id not in self.bus_lines:
                stops = self.route_stop_mapping.get(route_id, [])
                station_objects = [self.bus_stations.get(stop_id) for stop_id in stops]
                station_objects = [s for s in station_objects if s is not None]
                
                self.bus_lines[route_id] = BusLine(
                    traci_conn=self.traci_conn,
                    route_id=route_id,
                    stations=station_objects
                )
        
        print(f"Initialized public transport: "
              f"{len(self.subway_stations)} subway stations, "
              f"{len(self.subway_lines)} subway lines, "
              f"{len(self.bus_stations)} bus stations, "
              f"{len(self.bus_lines)} bus lines")
    
    def _initialize_runtime_data(self):
        """
        Initialize inter_info_dict, road_network_graphs, and Foundation Layer in __init__.
        Try to load from cache first, if not available, run reset() to build them.
        Optimized to avoid unnecessary SUMO restart if cache exists.
        """
        cache_path = self._get_runtime_data_cache_path()

        # Step 1: Try to load from cache FIRST (before starting SUMO)
        if os.path.exists(cache_path):
            try:
                print(f"Loading runtime data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                cached_inter_info_dict = cached_data.get("inter_info_dict")
                cached_graphs = cached_data.get("road_network_graphs")
                cached_highway_info = cached_data.get("highway_info_dict", {})
                cached_highway_subgraph = cached_data.get("highway_subgraph")
                cached_highway_segment_graph = cached_data.get("highway_segment_graph")
                cached_ramp_lane_dict = cached_data.get("ramp_lane_dict", {})

                # Load Foundation Layer data
                cached_network_graphs = cached_data.get("network_graphs", {})
                cached_network_dicts = cached_data.get("network_dicts", {})

                if cached_inter_info_dict is not None:
                    self.inter_info_dict = cached_inter_info_dict
                    print(f"Loaded inter_info_dict from cache: {len(self.inter_info_dict)} intersections")
                else:
                    print("Warning: Cache file exists but inter_info_dict not found")

                if cached_graphs is not None:
                    self._road_network_graphs = cached_graphs
                    print(f"Loaded road_network_graphs from cache: {list(self._road_network_graphs.keys())}")

                    # Extract ramp_lane_graph from road_network_graphs
                    if 'ramp_lane_graph' in self._road_network_graphs:
                        self.ramp_lane_graph = self._road_network_graphs['ramp_lane_graph']
                        if self.ramp_lane_graph is not None:
                            print(f"Loaded ramp_lane_graph from cache: {len(self.ramp_lane_graph.nodes())} nodes")
                else:
                    print("Warning: Cache file exists but road_network_graphs not found")

                if cached_highway_info:
                    self.highway_info_dict = cached_highway_info
                    print(f"Loaded highway_info_dict from cache: {len(self.highway_info_dict)} highway roads")

                if cached_highway_subgraph is not None:
                    self.highway_subgraph = cached_highway_subgraph
                    print(f"Loaded highway_subgraph from cache: {len(self.highway_subgraph.nodes())} nodes")

                if cached_highway_segment_graph is not None:
                    self.highway_segment_graph = cached_highway_segment_graph
                    print(f"Loaded highway_segment_graph from cache: {len(self.highway_segment_graph.nodes())} segments")

                if cached_ramp_lane_dict:
                    self.ramp_lane_dict = cached_ramp_lane_dict
                    print(f"Loaded ramp_lane_dict from cache: {len(self.ramp_lane_dict)} lanes")

                # Load Foundation Layer data
                if cached_network_graphs:
                    self.network_graphs = cached_network_graphs
                    for graph_name, graph in self.network_graphs.items():
                        if graph is not None:
                            print(f"Loaded network_graphs['{graph_name}'] from cache: {len(graph.nodes())} nodes")

                if cached_network_dicts:
                    self.network_dicts = cached_network_dicts
                    for dict_name, d in self.network_dicts.items():
                        if d:
                            print(f"Loaded network_dicts['{dict_name}'] from cache: {len(d)} entries")

                # Check if Foundation Layer needs to be built (cache version 1.0 doesn't have it)
                foundation_layer_missing = (
                    not cached_network_graphs or
                    not cached_network_graphs.get("lane_graph") or
                    not cached_network_graphs.get("road_graph")
                )

                if foundation_layer_missing:
                    print("Foundation Layer not found in cache (cache version 1.0). Will rebuild cache...")
                    # Don't return early - fall through to rebuild cache with Foundation Layer

                # If both loaded successfully and Foundation Layer exists, skip reset()
                elif cached_inter_info_dict is not None and cached_graphs is not None:
                    # Build highway data if not loaded from cache
                    if not self.highway_info_dict or self.highway_subgraph is None:
                        self._build_highway_info_dict()
                    # Note: highway_segment_graph requires highway_dict to be populated (done in reset())
                    # It will be built during reset() when highway_dict is created
                    # Build ramp_lane_graph if not loaded from cache (requires ramp_dict to be populated in reset())
                    if self.ramp_lane_graph is None and self.ramp_dict:
                        self._build_ramp_lane_graph()
                    # Initialize control modules (will generate fresh default configs, not load from file)
                    # Note: get_default_config may return empty dict if env not running, but config files may already exist

                    return  # Skip reset() if cache loaded successfully

            except Exception as e:
                print(f"Warning: Failed to load runtime data from cache: {e}")
                print("Will build inter_info_dict and road_network_graphs by running reset()...")

        # Step 2: Cache not available or failed to load - need to run reset() to build them
        # This is the only case where we need to start SUMO during initialization
        print("Runtime data cache not found. Building inter_info_dict and road_network_graphs...")
        print("Running reset() to initialize intersections...")
        self.reset(use_gui=False, seed=self.init_seed)

        # Note: Control modules are now initialized in reset() after all infrastructure is ready

        try:
            # Step 4: Build inter_info_dict (requires intersection_dict to be populated)
            print("Building inter_info_dict...")
            if not self.inter_info_dict:
                self.create_intersection_dict()
                print(f"Created inter_info_dict: {len(self.inter_info_dict)} intersections")

            # Step 5: Build road_network_graphs (requires inter_info_dict, ramp_dict, and highway_dict to be populated)
            # Note: ramp_dict and highway_dict are initialized in reset() which is called earlier
            print("Building road network graphs...")
            if self._road_network_graphs is None:
                self.build_road_network_graphs()
                if self._road_network_graphs:
                    print(f"Built graphs: {list(self._road_network_graphs.keys())}")

            # Step 6: Build highway_info_dict and highway_subgraph (requires road_network_graphs to be populated)
            print("Building highway segments dictionary and subgraph...")
            if not self.highway_info_dict or self.highway_subgraph is None:
                self._build_highway_info_dict()
                if self.highway_info_dict:
                    print(f"Built highway_info_dict: {len(self.highway_info_dict)} highway roads")
                if self.highway_subgraph is not None:
                    print(f"Built highway_subgraph: {len(self.highway_subgraph.nodes())} nodes")

            # Step 6.5: Build highway_segment_graph (requires highway_dict to be populated, done in reset())
            # Note: highway_segment_graph is built in reset() after highway_dict is created

            # Step 6.6: Build Foundation Layer (Layer 1) - complete network graphs and dicts
            print("Building Foundation Layer (Layer 1)...")
            if self.network_graphs.get("lane_graph") is None or self.network_graphs.get("road_graph") is None:
                self._build_foundation_layer()

            # Step 7: Save to cache (inter_info_dict, road_network_graphs, highway_info_dict, highway_subgraph, and Foundation Layer)
            if self.inter_info_dict and self._road_network_graphs:
                self._save_runtime_data_cache()
                print("Successfully built and cached runtime data including Foundation Layer")
            else:
                print("Warning: Failed to build inter_info_dict or road_network_graphs")

        except Exception as e:
            print(f"Warning: Failed to build runtime data during initialization: {e}")
            print("These will be built during the first reset() call")
            import traceback
            traceback.print_exc()

    def _load_roadnet(self):
        """
        Loads and parses the SUMO network file (.net.xml) using sumolib.
        Translates the SUMO network topology into the legacy dictionary formats
        and caches the results. This acts as the Anti-Corruption Layer for static data.
        """
        net_file_name = self.dic_traffic_env_conf.get("ROADNET_FILE", "map.net.xml")
        net_file_path = os.path.join(self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory), net_file_name)

        if not os.path.exists(net_file_path):
            raise FileNotFoundError(f"SUMO network file '{net_file_path}' not found.")

        print(f"Loading SUMO network from: {net_file_path}")
        try:
            # 1. One-Time Parsing and Caching of the sumolib object
            self.sumo_net = sumolib.net.readNet(net_file_path)
        except Exception as e:
            # Catches XML parsing errors and other sumolib issues
            raise ValueError(f"Failed to parse SUMO network file '{net_file_path}': {e}")

        # 2. Translate SUMO topology into legacy formats
        # --- Translate Intersections (Traffic Light Systems) ---
        # Only populate if not already loaded from cache
        if not self.intersections_data or not self.tls_to_nodes:
            self.intersections_data = {}
        
        # TLS↔node mappings for robust planning
        self.tls_to_nodes = {}   # TLS id -> [node ids]
        self.node_to_tls = {}    # node id -> TLS id
        
        # Get TLS from both getTrafficLights() and by checking junction types
        # This ensures we catch ramp metering TLS that may not be returned by getTrafficLights()
        tls_ids_found = set()
        
        # Method 1: Use getTrafficLights() (standard approach)
        for tls in self.sumo_net.getTrafficLights():
            tls_ids_found.add(tls.getID())
        
        # Method 2: Check all junctions with type="traffic_light"
        # This catches TLS that were added by tools like add_ramp_metering.py
        for node in self.sumo_net.getNodes():
            node_type = node.getType()
            if node_type == "traffic_light":
                node_id = node.getID()
                # Check if this node has a TLS (could be same ID or GS_ prefixed)
                possible_tls_ids = [node_id]
                if node_id.startswith("GS_"):
                    possible_tls_ids.append(node_id[3:])
                else:
                    possible_tls_ids.append("GS_" + node_id)
                
                for potential_tls_id in possible_tls_ids:
                    try:
                        tls = self.sumo_net.getTLS(potential_tls_id)
                        if tls:
                            tls_ids_found.add(potential_tls_id)
                            break
                    except:
                        # TLS with this ID doesn't exist
                        continue
                
                # If no TLS object found, still add the node as potential TLS
                # (it will be verified later via TraCI)
                if node_id not in tls_ids_found and not any(pid in tls_ids_found for pid in possible_tls_ids):
                    tls_ids_found.add(node_id)
        
        print(f"Found {len(tls_ids_found)} traffic light systems in network")
        
        # Now process each TLS ID
        for tls_id in tls_ids_found:
            # Try to get TLS object
            tls = None
            try:
                tls = self.sumo_net.getTLS(tls_id)
            except:
                pass
            
            # Nodes controlled by this TLS
            node_ids = []
            if tls:
                nodes = getattr(tls, "getNodes", lambda: [])()
                node_ids = [n.getID() for n in nodes] if nodes else []
            
            # If no nodes found from TLS object, try to find node with matching ID
            if not node_ids:
                try:
                    node = self.sumo_net.getNode(tls_id)
                    if node and node.getType() == "traffic_light":
                        node_ids = [tls_id]
                except:
                    # Try alternative ID formats
                    for alt_id in ([tls_id[3:]] if tls_id.startswith("GS_") else ["GS_" + tls_id]):
                        try:
                            node = self.sumo_net.getNode(alt_id)
                            if node and node.getType() == "traffic_light":
                                node_ids = [alt_id]
                                break
                        except:
                            continue

            # Coordinates
            # Try to get coordinates from nodes
            coord_nodes = []
            if tls:
                nodes = getattr(tls, "getNodes", lambda: [])()
                coord_nodes = nodes if nodes else []
            
            if coord_nodes:
                xs, ys = zip(*(n.getCoord() for n in coord_nodes))
                # Convert to Python native floats (maybe numpy floats)
                x, y = float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
            else:
                # Try to get coordinates from node with matching ID
                coord_found = False
                for potential_node_id in [tls_id] + ([tls_id[3:]] if tls_id.startswith("GS_") else ["GS_" + tls_id]):
                    try:
                        node = self.sumo_net.getNode(potential_node_id)
                        coord = node.getCoord()
                        x, y = float(coord[0]), float(coord[1])
                        coord_found = True
                        break
                    except Exception:
                        continue
                
                if not coord_found:
                    x, y = 0.0, 0.0

            # Save mapping + some useful metadata
            self.tls_to_nodes[tls_id] = node_ids
            for nid in node_ids:
                self.node_to_tls[nid] = tls_id

            # Check if this is a ramp metering TLS (has RAMP in phase names)
            # We'll check this later when traci_conn is available, but for now just store all TLS
            self.intersections_data[tls_id] = {
                "id": tls_id,
                "virtual": False,
                "point": {"x": x, "y": y},
                "controlled_nodes": node_ids,         # NEW
                "graph_node_ids": node_ids or [tls_id]# NEW: how planners can address this TLS in graphs
            }

        # --- Translate Roads (Edges) ---
        # Only populate if not already loaded from cache
        if not self.roads_data:
            self.roads_data = {}
        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"): # Ignore internal edges
                continue
            
            # Convert coordinates to Python native types (may be numpy floats)
            points = [{"x": float(x), "y": float(y)} for x, y in edge.getShape()]
            # Convert speeds to Python native types (may be numpy floats)
            lanes_info = [{"maxSpeed": float(lane.getSpeed())} for lane in edge.getLanes()]

            self.roads_data[edge_id] = {
                "id": edge_id,
                "lanes": lanes_info,
                "startIntersection": edge.getFromNode().getID(),
                "endIntersection": edge.getToNode().getID(),
                "points": points,
            }

        # Update config if intersections_data was populated
        if self.intersections_data:
            self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(self.intersections_data)
            print(f"Found {len(self.roads_data)} roads (edges) and {len(self.intersections_data)} signalized intersections (TLS).")

    def get_lane_speed(self, lane_id):
        """Returns the speed limit of the specified lane."""
        return self.sumo_net.getLane(lane_id).getSpeed()

    def _get_lane_length(self):
        """Calculates and caches the length of each lane from the parsed sumolib network."""
        self.lane_length = {}
        if not self.sumo_net:
            print("Warning: Cannot calculate lane lengths, SUMO network not loaded.")
            return

        for edge in self.sumo_net.getEdges():
            # Include all lanes, even internal junction lanes
            for lane in edge.getLanes():
                # Convert to Python native float (may be numpy float)
                self.lane_length[lane.getID()] = float(lane.getLength())

        print(f"Cached lengths for {len(self.lane_length)} lanes.")


    def reset_metrics(self):
        """
        Reset only the metrics/statistics tracked during simulation without restarting SUMO.
        This is used for checkpoint-based simulations where we want to continue the simulation
        but reset the metrics collection for each checkpoint interval.

        This method performs all the reset operations that reset() does EXCEPT:
        - self.current_time (continues from current simulation time, not reset to 0)
        - Closing and restarting SUMO
        - Reloading the network
        - Re-establishing TraCI connection

        Resets:
        - waiting_vehicle_list: Current waiting vehicles
        - waiting_passenger_list: Current waiting passengers
        - Travel time aggregators (all vehicles and highway-only vehicles) - checkpoint-scoped only
        - Departed vehicle tracking
        - Highway vehicle tracking
        - Update timing control
        - Subscribed vehicle IDs
        - Checkpoint metadata
        - Re-instantiates intersection_dict, highway_dict, ramp_dict with fresh metrics
        
        Does NOT reset (preserves across checkpoints):
        - Global travel time aggregators (_global_* variables) - these accumulate across all checkpoints
        - Use get_global_average_travel_time() to get cumulative statistics
        """
        print("================ Resetting Metrics (No SUMO Restart) ================")

        # Reset waiting lists (these track current state, not cumulative)
        self.waiting_vehicle_list = {}
        self.waiting_passenger_list = {}

        # Reset subscribed vehicle IDs tracking
        self._subscribed_vehicle_ids = set()
        self._last_passenger_update_time = -1.0

        # Reset travel-time aggregate
        self._depart_time_by_vehicle = {}
        self._arrived_tt_sum = 0.0
        self._arrived_count = 0
        self._arrived_vehicle_tt = {}

        # Reset highway travel-time aggregate
        # Note: Unlike reset(), we don't clear _highway_road_ids and _lane_to_road_id
        # as they are network structure that doesn't change
        self._highway_vehicle_ids = set()
        self._highway_arrived_tt_sum = 0.0
        self._highway_arrived_count = 0
        self._highway_arrived_vehicle_tt = {}

        # Note: Global travel-time aggregators (_global_* variables) are NOT reset here
        # They persist across checkpoint intervals and accumulate travel times from the start
        # of the simulation. Use get_global_average_travel_time() to access cumulative statistics.

        # Reset update timing control
        self._last_update_time = -600.0  # Initialize to allow first update immediately
        self._last_highway_update_time = -1.0

        # Reset checkpoint metadata
        self.checkpoint_vehicle_counts = {}
        self.checkpoint_extra_metadata = {}
        self.checkpoint_taxi_state = None

        # Clear and re-instantiate infrastructure dicts with fresh metrics
        print("Re-instantiating infrastructure objects...")

        # Clear old dicts (will be rebuilt with fresh metrics)
        self.highway_dict = {}
        self.ramp_dict = {}

        # Re-initialize Intersection Objects with fresh metrics
        print("Re-initializing intersections...")
        self._initialize_intersections()

        # Re-initialize Ramp Objects with fresh metrics
        print("Re-initializing ramps...")
        self._initialize_ramps()

        # Re-initialize Highway Objects with fresh metrics
        print("Re-initializing highways...")
        self._initialize_highways()
        # Note: _highway_road_ids is already set, no need to rebuild

        # Re-initialize Subway Infrastructure (if enabled)
        self._initialize_subway_infrastructure()

        # Re-initialize Bus Infrastructure (if enabled)
        self._initialize_bus_infrastructure()

        # Re-initialize Zone Infrastructure (TAZ-based organization)
        self._initialize_zone_infrastructure()

        # Re-initialize Control Modules (after all infrastructure is ready)
        print("Re-initializing control modules...")
        self._initialize_control_modules()

        # Update system states to get current state from SUMO
        self._update_system_states()

        # Update initial measurements for all infrastructure objects
        print("Updating initial measurements...")
        for inter in self.intersection_dict.values():
            inter.update_current_measurements(self.system_states)

        for highway in self.highway_dict.values():
            highway.update_current_measurements(self.system_states)

        for ramp in self.ramp_dict.values():
            ramp.update_current_measurements(self.system_states)

        print(f"Metrics reset complete at simulation time {self.current_time:.0f}s")


    def _prepare_simulated_taxi_sumocfg(self, original_sumocfg_path: str) -> str:
        """
        Generate a modified .sumocfg that excludes taxi-related files and vTypes.

        When USE_SIMULATED_TAXI_SYSTEM is True, SUMO must not load any taxi device
        artifacts.  This method:
        1. Removes taxi_fleet.rou.xml and persons.taxi.xml from <route-files>.
        2. Rewrites vtypes.add.xml so that vClass="taxi" becomes vClass="passenger"
           and the has.taxi.device param is dropped.
        3. Points <additional-files> at the rewritten vtypes file.
        4. Writes the modified sumocfg into the work directory and returns its path.
        """
        import xml.etree.ElementTree as ET

        original_dir = os.path.dirname(os.path.abspath(original_sumocfg_path))
        work_dir = self.path_to_work_directory or original_dir
        os.makedirs(work_dir, exist_ok=True)

        tree = ET.parse(original_sumocfg_path)
        root = tree.getroot()

        # --- 1. Filter route-files ---
        taxi_route_files = {"taxi_fleet.rou.xml", "persons.taxi.xml"}
        for input_elem in root.findall("input"):
            route_elem = input_elem.find("route-files")
            if route_elem is not None:
                raw = route_elem.get("value", "")
                filtered = [f.strip() for f in raw.split(",")
                            if f.strip() and f.strip() not in taxi_route_files]
                route_elem.set("value", ",".join(filtered))
                print(f"[SimTaxi-sumocfg] route-files: {raw} -> {','.join(filtered)}")

            # --- 2. Rewrite vtypes.add.xml ---
            add_elem = input_elem.find("additional-files")
            if add_elem is not None:
                add_files = [f.strip() for f in add_elem.get("value", "").split(",") if f.strip()]
                new_add_files = []
                for af in add_files:
                    if "vtypes" in af and af.endswith(".xml"):
                        new_af = self._rewrite_vtypes_file(
                            os.path.join(original_dir, af), work_dir
                        )
                        # Use path relative to work_dir if sumocfg will live there,
                        # otherwise use absolute path.
                        new_add_files.append(os.path.abspath(new_af))
                    else:
                        # Keep absolute path so SUMO can find it regardless of cwd
                        new_add_files.append(os.path.join(original_dir, af))
                add_elem.set("value", ",".join(new_add_files))

        # Make net-file path absolute so it works from any cwd
        for input_elem in root.findall("input"):
            net_elem = input_elem.find("net-file")
            if net_elem is not None:
                net_val = net_elem.get("value", "")
                if not os.path.isabs(net_val):
                    net_elem.set("value", os.path.join(original_dir, net_val))
            # Also make remaining route-files absolute
            route_elem = input_elem.find("route-files")
            if route_elem is not None:
                parts = [p.strip() for p in route_elem.get("value", "").split(",") if p.strip()]
                abs_parts = []
                for p in parts:
                    if not os.path.isabs(p):
                        abs_parts.append(os.path.join(original_dir, p))
                    else:
                        abs_parts.append(p)
                route_elem.set("value", ",".join(abs_parts))

        modified_path = os.path.join(work_dir, "simtaxi_" + os.path.basename(original_sumocfg_path))
        tree.write(modified_path, xml_declaration=True, encoding="UTF-8")
        print(f"[SimTaxi-sumocfg] Written modified sumocfg: {modified_path}")
        return modified_path

    def _prepare_code_dispatched_transit_sumocfg(self, original_sumocfg_path: str) -> str:
        """
        Generate a modified .sumocfg where bus/subway flow vehicles are removed
        from route files before SUMO starts.

        The scheduling modules dispatch their own vehicles. Removing the original
        <flow> definitions at startup avoids repeatedly deleting SUMO generated
        bus_* / subway_* vehicles and prevents "Vehicle is not known" noise from
        SUMO/TraCI.
        """
        import xml.etree.ElementTree as ET

        disabled_types = set()
        if self.control_modules and "bus_scheduling" in self.control_modules:
            disabled_types.add("bus")
        if self.control_modules and "subway_scheduling" in self.control_modules:
            disabled_types.add("subway")

        if not disabled_types:
            return original_sumocfg_path

        self.transit_flow_vehicles_filtered = True

        original_dir = os.path.dirname(os.path.abspath(original_sumocfg_path))
        work_dir = self.path_to_work_directory or original_dir
        os.makedirs(work_dir, exist_ok=True)

        tree = ET.parse(original_sumocfg_path)
        root = tree.getroot()

        def _abs_path(path_value: str) -> str:
            return path_value if os.path.isabs(path_value) else os.path.join(original_dir, path_value)

        for input_elem in root.findall("input"):
            net_elem = input_elem.find("net-file")
            if net_elem is not None:
                net_val = net_elem.get("value", "")
                if net_val:
                    net_elem.set("value", _abs_path(net_val))

            add_elem = input_elem.find("additional-files")
            if add_elem is not None:
                add_files = [p.strip() for p in add_elem.get("value", "").split(",") if p.strip()]
                add_elem.set("value", ",".join(_abs_path(p) for p in add_files))

            route_elem = input_elem.find("route-files")
            if route_elem is None:
                continue

            route_files = [p.strip() for p in route_elem.get("value", "").split(",") if p.strip()]
            rewritten_route_files = []
            for route_file in route_files:
                source_path = _abs_path(route_file)
                rewritten_route_files.append(self._rewrite_transit_flow_route_file(source_path, work_dir, disabled_types))
            route_elem.set("value", ",".join(rewritten_route_files))

        suffix = "_".join(sorted(disabled_types))
        modified_path = os.path.join(work_dir, f"code_transit_{suffix}_" + os.path.basename(original_sumocfg_path))
        tree.write(modified_path, xml_declaration=True, encoding="UTF-8")
        return modified_path

    def _rewrite_transit_flow_route_file(self, route_file_path: str, work_dir: str, disabled_types: set) -> str:
        import xml.etree.ElementTree as ET

        if not os.path.exists(route_file_path):
            return route_file_path

        try:
            tree = ET.parse(route_file_path)
        except Exception:
            return route_file_path

        root = tree.getroot()
        removed = 0
        for parent in root.iter():
            for child in list(parent):
                if child.tag != "flow":
                    continue
                flow_type = (child.get("type") or "").strip().lower()
                route_id = (child.get("route") or child.get("id") or "").strip()
                is_disabled = flow_type in disabled_types
                if not is_disabled and "bus" in disabled_types:
                    is_disabled = route_id.startswith("bus_")
                if not is_disabled and "subway" in disabled_types:
                    is_disabled = route_id.startswith("subway_")
                if is_disabled:
                    parent.remove(child)
                    removed += 1

        if removed == 0:
            return route_file_path

        base_name = os.path.basename(route_file_path)
        rewritten_path = os.path.join(work_dir, f"code_transit_{base_name}")
        tree.write(rewritten_path, xml_declaration=True, encoding="UTF-8")
        return rewritten_path

    def _rewrite_vtypes_file(self, vtypes_path: str, work_dir: str) -> str:
        """
        Rewrite a vtypes.add.xml file:
        - Keep vClass="taxi" so vehicles can use taxi-allowed lanes
        - Set has.taxi.device to "false" to prevent SUMO from assigning the taxi device
        Returns the path to the rewritten file.
        """
        import xml.etree.ElementTree as ET

        if not os.path.exists(vtypes_path):
            print(f"[SimTaxi-vtypes] Warning: vtypes file not found: {vtypes_path}")
            return vtypes_path

        tree = ET.parse(vtypes_path)
        root = tree.getroot()

        modified = False
        for vtype in root.iter("vType"):
            if vtype.get("vClass") == "taxi":
                # Keep vClass="taxi" — changing to "passenger" breaks lane access
                # Instead, explicitly disable the taxi device
                has_device_param = False
                for param in list(vtype.findall("param")):
                    if param.get("key") == "has.taxi.device":
                        param.set("value", "false")
                        has_device_param = True
                        modified = True
                if not has_device_param:
                    p = ET.SubElement(vtype, "param")
                    p.set("key", "has.taxi.device")
                    p.set("value", "false")
                    modified = True

        out_name = "simtaxi_" + os.path.basename(vtypes_path)
        out_path = os.path.join(work_dir, out_name)
        tree.write(out_path, xml_declaration=True, encoding="UTF-8")
        if modified:
            print(f"[SimTaxi-vtypes] Rewrote vtypes (disabled taxi device, kept vClass=taxi): {out_path}")
        else:
            print(f"[SimTaxi-vtypes] No taxi vTypes found, copied as-is: {out_path}")
        return out_path

    def reset(self, use_gui=False, seed=None, load_state_path=None, simulation_begin_time=None):
        """
        Resets the simulation environment.
        - Shuts down any existing SUMO simulation.
        - Uses the provided .sumocfg file directly from PATH_TO_DATA (no copying).
        - Launches a new SUMO instance (with or without GUI) in the sumocfg directory.
        - Establishes a TraCI connection.
        - Initializes Intersection objects.
        - Retrieves initial state.
        
        Note: All paths in the sumocfg file should be relative to the sumocfg file's directory.
        SUMO will run in the sumocfg directory to resolve these relative paths.
        
        Args:
            use_gui (bool): Whether to use sumo-gui.
            seed (int, optional): Random seed for the simulation.
            load_state_path (str, optional): If provided, starts the simulation from this snapshot file.
                                           SUMO will automatically read the simulation time from the checkpoint.
            simulation_begin_time (float, optional): If set and ``load_state_path`` is not used,
                SUMO is started with ``--begin`` at this simulation second (no snapshot). Route demand
                continues from this time; vehicle state is not restored.
        """
        print("================ Starting Environment Reset ================")
        self.close()

        self.current_time = 0.0
        self.waiting_vehicle_list = {}
        self.waiting_passenger_list = {}
        self._subscribed_vehicle_ids = set()
        self._last_passenger_update_time = -1.0
        # Reset travel-time aggregate
        self._depart_time_by_vehicle = {}
        self._arrived_tt_sum = 0.0
        self._arrived_count = 0
        self._arrived_vehicle_tt = {}
        # Reset highway travel-time aggregate
        self._highway_road_ids = set()
        self._lane_to_road_id = {}
        self._highway_vehicle_ids = set()
        self._highway_arrived_tt_sum = 0.0
        self._highway_arrived_count = 0
        self._highway_arrived_vehicle_tt = {}
        # Reset global travel-time aggregate (only reset on full reset, not reset_metrics)
        self._global_arrived_tt_sum = 0.0
        self._global_arrived_count = 0
        self._global_arrived_vehicle_tt = {}
        self._global_highway_arrived_tt_sum = 0.0
        self._global_highway_arrived_count = 0
        self._global_highway_arrived_vehicle_tt = {}
        # Reset update timing control
        self._last_update_time = -600.0  # Initialize to allow first update immediately
        self._last_highway_update_time = -1.0
        
        # Initialize vehicle counts (will be populated from checkpoint metadata if loading state)
        self.checkpoint_vehicle_counts = {}
        # Optional extra metadata from checkpoints (e.g., taxi control state)
        self.checkpoint_extra_metadata = {}
        self.checkpoint_taxi_state = None
        
        # Clear highway dict and ramp dict (will be reinitialized after intersections are created)
        self.highway_dict = {}
        self.ramp_dict = {}

        # 1. Load sumo_net if not already loaded (needed for runtime operations)
        if self.sumo_net is None:
            self._load_roadnet()

        # 2. Use the provided sumocfg file directly (no copying)
        # All paths in sumocfg are relative to the sumocfg file's directory
        data_path = self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory)
        sumocfg_file_name = self.dic_traffic_env_conf.get("SUMOCFG_FILE", "map.sumocfg")
        source_sumocfg_path = os.path.join(data_path, sumocfg_file_name)

        if not os.path.exists(source_sumocfg_path):
            raise FileNotFoundError(
                f"SUMO configuration file not found at source: {source_sumocfg_path}\n"
                f"Please provide a valid .sumocfg file in the PATH_TO_DATA directory."
            )
        
        # Use absolute path to sumocfg file
        sumocfg_path = os.path.abspath(source_sumocfg_path)
        sumocfg_dir = os.path.dirname(sumocfg_path)
        sumocfg_filename = os.path.basename(sumocfg_path)
        self.transit_flow_vehicles_filtered = False

        # If simulated taxi system is enabled, generate a modified sumocfg
        # that excludes taxi files and rewrites taxi vTypes to passenger.
        use_sim_taxi = self.dic_traffic_env_conf.get("USE_SIMULATED_TAXI_SYSTEM", False)
        if use_sim_taxi:
            sumocfg_path = self._prepare_simulated_taxi_sumocfg(sumocfg_path)
            sumocfg_dir = os.path.dirname(sumocfg_path)
            sumocfg_filename = os.path.basename(sumocfg_path)
            print(f"[SimTaxi] Using modified sumocfg (no taxi device): {sumocfg_path}")

        if self.control_modules and (
            "bus_scheduling" in self.control_modules or "subway_scheduling" in self.control_modules
        ):
            sumocfg_path = self._prepare_code_dispatched_transit_sumocfg(sumocfg_path)
            sumocfg_dir = os.path.dirname(sumocfg_path)
            sumocfg_filename = os.path.basename(sumocfg_path)
        elif not use_sim_taxi:
            print(f"Using SUMO configuration file directly (no copying): {sumocfg_path}")

        print(f"SUMO will run in directory: {sumocfg_dir}")
        
        # 3. Prepare and Launch SUMO
        sumo_binary = "sumo" if not use_gui else "sumo-gui"

        if seed is None:
            seed = int(np.random.randint(0, 10000))

        # Find a free port dynamically for each reset
        self._current_port = find_free_port()
        print(f"Using TraCI port: {self._current_port}")

        # 设置仿真时间
        sim_time = self.dic_traffic_env_conf.get("RUN_COUNTS", 3600)
        # Use relative path to sumocfg (since we'll set cwd to sumocfg_dir)
        time_to_teleport = self.dic_traffic_env_conf.get("TIME_TO_TELEPORT", 300)
        sumo_cmd = [
            sumo_binary, "-c", sumocfg_filename,
            "--remote-port", str(self._current_port),
            "--seed", str(seed),
            "--step-length", str(self.dic_traffic_env_conf.get("INTERVAL", 1.0)),
            "--no-warnings", "false",
            "--ignore-route-errors", "true",
            "--time-to-teleport", str(time_to_teleport),
            "--no-step-log", "true"  # Suppress step log messages
        ]

        # Add taxi dispatch algorithms if taxi_scheduling module is enabled
        # Skip when using simulated taxi system — no SUMO taxi device needed.
        if self.control_modules and 'taxi_scheduling' in self.control_modules and not use_sim_taxi:
            taxi_dispatch_algorithm = self.dic_traffic_env_conf.get("TAXI_DISPATCH_ALGORITHM", "traci")
            # Use randomCircling by default - "stop" can cause SUMO crashes when taxis have no destinations
            taxi_idle_algorithm = self.dic_traffic_env_conf.get("TAXI_IDLE_ALGORITHM", "randomCircling")
            sumo_cmd.extend(["--device.taxi.dispatch-algorithm", str(taxi_dispatch_algorithm)])
            sumo_cmd.extend(["--device.taxi.idle-algorithm", str(taxi_idle_algorithm)])
            print(
                "Taxi scheduling enabled: Added --device.taxi.dispatch-algorithm "
                f"{taxi_dispatch_algorithm} and --device.taxi.idle-algorithm {taxi_idle_algorithm}"
            )
        elif use_sim_taxi and self.control_modules and 'taxi_scheduling' in self.control_modules:
            print("[SimTaxi] Skipping SUMO taxi device command-line options (simulated taxi system active)")

        if self.control_modules and (
            'bus_scheduling' in self.control_modules or 'subway_scheduling' in self.control_modules
        ):
            sumo_cmd.extend(["--device.tripinfo.probability", "1"])
            print("Bus/subway scheduling enabled: Added --device.tripinfo.probability 1 for ride statistics")

        # Enable SUMO internal logging for crash diagnostics
        sumo_log_enabled = self.dic_traffic_env_conf.get("SUMO_LOG_ENABLED", True)
        if sumo_log_enabled:
            log_dir = self.path_to_log or sumocfg_dir
            os.makedirs(log_dir, exist_ok=True)
            log_tag = time.strftime("%Y%m%d_%H%M%S")
            sumo_log_path = os.path.join(log_dir, f"sumo_{log_tag}.log")
            sumo_cmd.extend(["--log", sumo_log_path])
            print(f"SUMO log enabled: {sumo_log_path}")

        # Always save transportables and RNG state to keep checkpoint continuity.
        # Required for taxi scheduling checkpoints and safe for other modules.
        sumo_cmd.append("--save-state.transportables")
        sumo_cmd.append("--save-state.rng")
        # Save rail signal constraints so checkpoint restore doesn't fail on driveWay refs
        sumo_cmd.append("--save-state.constraints")

        # MODIFICATION: Add the --load-state and --begin arguments if a path is provided
        # Read checkpoint time from metadata file and add --begin TIME to sumo command
        checkpoint_time = None
        if load_state_path and os.path.exists(load_state_path):
            print(f"Attempting to load simulation from state: {load_state_path}")
            sumo_cmd.extend(["--load-state", load_state_path])

            # Try to read checkpoint time and vehicle counts from metadata file
            metadata_path = os.path.splitext(load_state_path)[0] + "_metadata.json"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        checkpoint_time = metadata.get("checkpoint_time") or metadata.get("sim_time")
                        if checkpoint_time is not None:
                            print(f"Found checkpoint metadata: time={checkpoint_time:.2f}s")
                            sumo_cmd.extend(["--begin", str(checkpoint_time)])
                            # Add --end parameter to ensure simulation runs long enough
                            # Use a very large end time (24 hours = 86400s) to prevent premature termination
                            end_time = max(86400, checkpoint_time + sim_time + 3600)
                            sumo_cmd.extend(["--end", str(end_time)])
                        else:
                            print(f"Warning: checkpoint_time not found in metadata file: {metadata_path}")

                        # Read vehicle counts to maintain continuous vehicle IDs
                        self.checkpoint_vehicle_counts = metadata.get("vehicle_counts", {}) or {}
                        if self.checkpoint_vehicle_counts:
                            print(f"Loaded vehicle counts for {len(self.checkpoint_vehicle_counts)} lines")

                        # Load extra checkpoint metadata (control states/configs, taxi state, etc.)
                        self.checkpoint_extra_metadata = metadata.get("extra", {}) or {}
                        self.checkpoint_taxi_state = self.checkpoint_extra_metadata.get("taxi_state")
                except Exception as e:
                    print(f"Warning: Failed to read checkpoint metadata from {metadata_path}: {e}")
                    self.checkpoint_vehicle_counts = {}
            else:
                print(f"Warning: Metadata file not found: {metadata_path}. Starting from checkpoint time 0.")
        elif load_state_path:
            print(f"Warning: Snapshot file for loading not found at {load_state_path}. Starting new simulation.")
        elif simulation_begin_time is not None and float(simulation_begin_time) > 1e-6:
            t0 = float(simulation_begin_time)
            sumo_cmd.extend(["--begin", str(t0)])
            end_time = max(86400.0, t0 + float(sim_time) + 3600.0)
            sumo_cmd.extend(["--end", str(int(end_time))])
            print(
                f"Starting SUMO at simulation time {t0:.1f}s via --begin (no snapshot); "
                f"--end {int(end_time)}"
            )

        print(f"Launching SUMO with command: {' '.join(sumo_cmd)}")
        sumo_stdout_log = os.path.join(self.path_to_log, "sumo_stdout.log")

        try:
            # Redirect SUMO stdout to log file, stderr to /dev/null to suppress error messages
            # Error messages like "Vehicle X is not known" are expected when removing flow vehicles
            # and should not clutter the console output
            with open(sumo_stdout_log, 'w') as f_out:
                 self.sumo_process = subprocess.Popen(
                     sumo_cmd,
                     stdout=f_out,
                     stderr=subprocess.DEVNULL,  # Suppress all SUMO stderr output
                     cwd=sumocfg_dir  # Run SUMO in the sumocfg directory
                 )

            # Give SUMO a moment to start up or crash
            time.sleep(5)

            # Check if the SUMO process terminated prematurely
            if self.sumo_process.poll() is not None:
                error_message = f"SUMO process terminated unexpectedly. Check SUMO stdout log for details:\n"
                error_message += f"  - STDOUT: {sumo_stdout_log}\n"
                try:
                    with open(sumo_stdout_log, 'r') as f_out_read:
                        log_content = f_out_read.read()
                        if log_content.strip():
                            error_message += f"\n--- SUMO Log Content ---\n{log_content}\n----------------------------"
                except IOError:
                    error_message += "(Could not read log file.)"
                self.close()
                raise RuntimeError(error_message)

            # 4. Establish TraCI Connection
            self.traci_conn = traci.connect(port=self._current_port, numRetries=10)
            # Reset TraCI health on successful (re)connect
            self._traci_healthy = True
            
            # Subscribe to lane-based information once
            for lane_id in self.lane_length.keys():
                self.traci_conn.lane.subscribe(lane_id, [
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST
                ])

            self._simulation_running = True
            print(f"Successfully connected to SUMO (seed: {seed}).")

        except FileNotFoundError:
            raise EnvironmentError(f"'{sumo_binary}' not found. Please ensure SUMO is installed and in your system's PATH.")
        except traci.TraCIException as e:
            error_message = (f"Failed to connect to SUMO via TraCI. This often means SUMO crashed on startup. "
                             f"Please check the SUMO stdout log for errors:\n"
                             f"  - STDOUT: {sumo_stdout_log}\n"
                             f"Original TraCI error: {e}")
            try:
                with open(sumo_stdout_log, 'r') as f_out_read:
                     log_content = f_out_read.read()
                     if log_content.strip():
                         error_message += f"\n\n--- SUMO Log Content ---\n{log_content}\n----------------------------"
            except IOError:
                pass
            self.close()
            raise EnvironmentError(error_message) from e
        except Exception as e:
            self.close()
            raise

        # 5. Initialize Intersection Objects
        self._initialize_intersections()
        
        # 6. Initialize Ramp Objects
        # (ramp_info_dict will be built during initialization)
        self._initialize_ramps()
        
        # 7. Initialize Highway Objects
        # (highway_info_dict should already be built in _initialize_runtime_data())
        self._initialize_highways()
        self._highway_road_ids = self._collect_highway_road_ids()

        # 7.5. Initialize Subway Infrastructure (if enabled)
        self._initialize_subway_infrastructure()

        # 7.6. Initialize Bus Infrastructure (if enabled)
        self._initialize_bus_infrastructure()

        # 7.65. Initialize Zone Infrastructure (TAZ-based organization)
        self._initialize_zone_infrastructure()

        # 7.7. Initialize Control Modules (after all infrastructure is ready)
        print("Initializing control modules...")
        self._initialize_control_modules()

        # 8. Get Initial State from Simulator
        # MODIFICATION: If loading from state, the first step is not needed as SUMO is already at that time.
        # Otherwise, take one step to populate the network with initial vehicles.
        print("Getting initial simulator state...")
        if not load_state_path:
            self.traci_conn.simulationStep()
        self._update_system_states()

        # 10. Update Intersection Measurements
        print("Updating initial intersection measurements...")
        for inter in self.intersection_dict.values():
            inter.update_current_measurements(self.system_states)

        # 11. Update Highway Measurements
        for highway in self.highway_dict.values():
            highway.update_current_measurements(self.system_states)
        
        # 12. Update Ramp Measurements
        for ramp in self.ramp_dict.values():
            ramp.update_current_measurements(self.system_states)
        
        # 12.5. Extract ramp_lane_graph from road_network_graphs if available (after reset())
        if self._road_network_graphs and 'ramp_lane_graph' in self._road_network_graphs:
            self.ramp_lane_graph = self._road_network_graphs['ramp_lane_graph']
            if self.ramp_lane_graph is not None:
                print(f"Loaded ramp_lane_graph from road_network_graphs: {len(self.ramp_lane_graph.nodes())} nodes")

        # 13. Initializing control modules
        print("Initializing control modules (using cached data)...")
        self._initialize_control_modules()

        print("================ Environment Reset Complete ================")
        return None  # State return removed - no longer needed for RL training

    def _update_system_states(self):
        """
        Subscribes to dynamic vehicle data and retrieves all subscription
        results in a single batch.
        Only collects data needed for highway feature calculation (used by traffic_state_collector).
        """
        if not self._simulation_running:
            return

        try:
            # 1. Dynamic Vehicle Subscriptions (per step)
            subscription_mode = self.dic_traffic_env_conf.get("VEHICLE_SUBSCRIPTION_MODE", "departed")
            if isinstance(subscription_mode, str):
                subscription_mode = subscription_mode.lower()
            departed_ids = None
            if subscription_mode == "departed" and self._subscribed_vehicle_ids:
                try:
                    departed_ids = self.traci_conn.simulation.getDepartedIDList()
                    current_vehicle_ids = departed_ids
                except traci.TraCIException:
                    current_vehicle_ids = self.traci_conn.vehicle.getIDList()
            else:
                current_vehicle_ids = self.traci_conn.vehicle.getIDList()

            # Get current active vehicle set for subscription cleanup
            active_vehicle_set = set(self.traci_conn.vehicle.getIDList())

            # Clean up stale subscriptions - remove vehicles that left the simulation
            # This prevents TraCI errors when querying subscription results for departed vehicles
            stale_subscriptions = self._subscribed_vehicle_ids - active_vehicle_set
            if stale_subscriptions:
                self._subscribed_vehicle_ids -= stale_subscriptions

            for vehicle_id in current_vehicle_ids:
                if vehicle_id in self._subscribed_vehicle_ids:
                    continue
                # Only subscribe if vehicle still exists
                if vehicle_id not in active_vehicle_set:
                    continue
                try:
                    self.traci_conn.vehicle.subscribe(vehicle_id, [
                        traci.constants.VAR_SPEED,
                        traci.constants.VAR_LANE_ID,
                        traci.constants.VAR_LANEPOSITION
                    ])
                except traci.TraCIException:
                    # Vehicle may have left the network between list and subscribe.
                    continue
                self._subscribed_vehicle_ids.add(vehicle_id)

            # 2. Batch Data Retrieval - Use domain-specific calls
            vehicle_results = self.traci_conn.vehicle.getAllSubscriptionResults()
            lane_results = self.traci_conn.lane.getAllSubscriptionResults()

            # 3. Data Transformation - Collect data needed for highway and ramp features
            self.system_states = {
                "get_lane_vehicles": defaultdict(list),
                "get_lane_halting_number": {},
                "get_vehicle_speed": {},
                "get_vehicle_lane_position": {},
                "get_lane_length": self.lane_length.copy(),  # Copy cached lane lengths
                "get_waiting_vehicles": self.waiting_vehicle_list.copy(),  # Copy waiting vehicle times
            }
            
            # Process lane subscription results to get the list of vehicles per lane
            for lane_id, data in lane_results.items():
                self.system_states["get_lane_vehicles"][lane_id] = data[traci.constants.LAST_STEP_VEHICLE_ID_LIST]
                halting_n = data.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER)
                if halting_n is not None:
                    self.system_states["get_lane_halting_number"][lane_id] = int(halting_n)
            
            # Process vehicle subscription results to get speeds
            for vehicle_id, data in vehicle_results.items():
                speed = data[traci.constants.VAR_SPEED]
                self.system_states["get_vehicle_speed"][vehicle_id] = speed
                lane_pos = data.get(traci.constants.VAR_LANEPOSITION)
                if lane_pos is not None:
                    self.system_states["get_vehicle_lane_position"][vehicle_id] = lane_pos

            # Update highway vehicle tracking (vehicles on highway roads in this step)
            if not self._highway_road_ids and self.highway_dict:
                self._highway_road_ids = self._collect_highway_road_ids()
            if self.highway_dict and not self._lane_to_road_id:
                self._ensure_lane_to_road_map()

            if self._highway_road_ids and self._lane_to_road_id:
                current_highway_vehicle_ids = set()
                for lane_id, vehicle_list in self.system_states["get_lane_vehicles"].items():
                    road_id = self._lane_to_road_id.get(lane_id)
                    if road_id in self._highway_road_ids:
                        current_highway_vehicle_ids.update(vehicle_list)
                if current_highway_vehicle_ids:
                    self._highway_vehicle_ids.update(current_highway_vehicle_ids)

            # Get current sim time once
            now_time = self.traci_conn.simulation.getTime()

            # --- Update travel-time aggregate ---
            try:
                if departed_ids is None:
                    departed_ids = self.traci_conn.simulation.getDepartedIDList()
                arrived_ids = self.traci_conn.simulation.getArrivedIDList()
            except traci.TraCIException:
                departed_ids, arrived_ids = [], []

            # Record depart times for vehicles that started this step
            for vid in departed_ids:
                self._depart_time_by_vehicle[vid] = now_time

            # For vehicles that finished this step, accumulate travel times
            for vid in arrived_ids:
                depart_time = self._depart_time_by_vehicle.pop(vid, None)
                if depart_time is not None:
                    tt = now_time - depart_time
                    # Update checkpoint-scoped metrics (reset by reset_metrics)
                    self._arrived_tt_sum += tt
                    self._arrived_count += 1
                    self._arrived_vehicle_tt[vid] = tt
                    # Update global metrics (persist across reset_metrics)
                    self._global_arrived_tt_sum += tt
                    self._global_arrived_count += 1
                    self._global_arrived_vehicle_tt[vid] = tt
                    if vid in self._highway_vehicle_ids:
                        # Update checkpoint-scoped highway metrics
                        self._highway_arrived_tt_sum += tt
                        self._highway_arrived_count += 1
                        self._highway_arrived_vehicle_tt[vid] = tt
                        self._highway_vehicle_ids.discard(vid)
                        # Update global highway metrics
                        self._global_highway_arrived_tt_sum += tt
                        self._global_highway_arrived_count += 1
                        self._global_highway_arrived_vehicle_tt[vid] = tt
                # Clean up subscription tracking for arrived vehicles
                self._subscribed_vehicle_ids.discard(vid)
            #     # CRITICAL FIX: Unsubscribe from TraCI to prevent memory leak
            #     try:
            #         self.traci_conn.vehicle.unsubscribe(vid)
            #     except traci.TraCIException:
            #         pass  # Vehicle already gone, subscription auto-removed

            # # Periodically clean up stale subscriptions (every 100 steps)
            # if len(self._subscribed_vehicle_ids) > 0 and int(now_time) % 100 == 0:
            #     current_vehicle_set = set(self.traci_conn.vehicle.getIDList())
            #     stale_subscriptions = self._subscribed_vehicle_ids - current_vehicle_set
            #     for vid in stale_subscriptions:
            #         try:
            #             self.traci_conn.vehicle.unsubscribe(vid)
            #         except traci.TraCIException:
            #             pass
            #         self._subscribed_vehicle_ids.discard(vid)

            # Commit the time for the environment
            self.current_time = now_time

        except (traci.TraCIException, traci.exceptions.FatalTraCIError) as e:
            print(f"TraCI error during state update: {e}. Simulation may have ended.")
            self.close()
            raise RuntimeError(f"TraCI connection lost: {e}") from e

    def _collect_highway_road_ids(self) -> set:
        """Build a set of road IDs that belong to highway segments."""
        road_ids = set()
        if self.highway_dict:
            for highway in self.highway_dict.values():
                road_ids.update(highway.highway_road_ids)
        return road_ids

    def _ensure_lane_to_road_map(self) -> None:
        """Build lane_id -> road_id mapping lazily for highway vehicle tracking."""
        if self._lane_to_road_id:
            return
        lane_to_road = {}
        if self.sumo_net:
            for lane_id in self.lane_length.keys():
                road_id = None
                try:
                    lane_obj = self.sumo_net.getLane(lane_id)
                    road_obj = lane_obj.getEdge()
                    road_id = road_obj.getID()
                except Exception:
                    road_id = None
                if not road_id and "_" in lane_id:
                    parts = lane_id.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        road_id = parts[0]
                if road_id and not road_id.startswith(":"):
                    lane_to_road[lane_id] = road_id
        self._lane_to_road_id = lane_to_road

    def _load_or_create_intersection_dict(self):
        """
        Load inter_info_dict from cache if available, otherwise create it.
        This method checks cache first, and only builds if cache doesn't exist.
        """
        cache_path = self._get_runtime_data_cache_path()
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                print(f"Loading intersection_dict and road_network_graphs from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                cached_inter_info_dict = cached_data.get("inter_info_dict")
                if cached_inter_info_dict is not None:
                    self.inter_info_dict = cached_inter_info_dict
                    print(f"Loaded inter_info_dict from cache: {len(self.inter_info_dict)} intersections")
                    return  # Successfully loaded, no need to build
                else:
                    print("Warning: Cache file exists but inter_info_dict not found, will rebuild...")
            except Exception as e:
                print(f"Warning: Failed to load runtime data from cache: {e}")
                print("Will rebuild inter_info_dict and road_network_graphs...")
        
        # Cache not available or failed to load - create inter_info_dict
        self.create_intersection_dict()
    
    def create_intersection_dict(self):
        """
        Creates the `inter_info_dict` attribute for API compatibility, based on the
        dynamically discovered properties of the Intersection objects.
        """
        self.inter_info_dict = {}
        print(f"Populating inter_info_dict for {len(self.intersection_dict)} intersections...")

        for intersection_obj in self.intersection_dict.values():
            inter_id = intersection_obj.inter_id
            
            agent_intersection_info = {
                "id": inter_id,
                "phases": {},
                "roads": {},
                "control_phases": intersection_obj.control_phases
            }

            for phase_name, sumo_idx in intersection_obj.phase_name_2_cityflow_idx.items():
                if sumo_idx < len(intersection_obj.phases):
                    phase_def = intersection_obj.phases[sumo_idx]
                    agent_intersection_info["phases"][phase_name] = {"time": phase_def.duration, "idx": sumo_idx}

            all_roads = {**intersection_obj.incoming_roads, **intersection_obj.outgoing_roads}
            for road_id, road_obj in all_roads.items():
                is_incoming = road_id in intersection_obj.incoming_roads
                road_type = "incoming" if is_incoming else "outgoing"
                
                location_code = intersection_obj.road_id_2_orient.get(road_type, {}).get(road_id)
                location = location_dict_reverse.get(location_code)

                road_info = {
                    "location": location, "type": road_type,
                    "length": road_obj.getLength(),
                    "max_speed": road_obj.getSpeed(),
                    "num_lanes": len(road_obj.getLanes()),
                    "lanes": defaultdict(list), "go_straight": None,
                    "turn_left": None, "turn_right": None
                }
                
                if is_incoming:
                    for link in intersection_obj.road_links:
                        if link["startRoad"] == road_id:
                            turn_type = link["type"]
                            end_road_id = link["endRoad"]

                            if turn_type == "go_straight":
                                road_info["go_straight"] = end_road_id
                            elif turn_type == "turn_left":
                                road_info["turn_left"] = end_road_id
                            elif turn_type == "turn_right":
                                road_info["turn_right"] = end_road_id

                            for lane_link in link.get("laneLinks", []):
                                start_lane_idx = lane_link.get("startLaneIndex")
                                if start_lane_idx is not None and start_lane_idx not in road_info["lanes"][turn_type]:
                                    road_info["lanes"][turn_type].append(start_lane_idx)
                else:  # This is an outgoing road
                    # Expanded logic for outgoing roads
                    for link in intersection_obj.road_links:
                        if link["endRoad"] == road_id:
                            turn_type = link["type"]
                            end_road_id = link["endRoad"] # This is the same as road_id

                            # This makes the data structure consistent with incoming roads.
                            # The value will be the ID of the outgoing road itself.
                            if turn_type == "go_straight":
                                road_info["go_straight"] = end_road_id
                            elif turn_type == "turn_left":
                                road_info["turn_left"] = end_road_id
                            elif turn_type == "turn_right":
                                road_info["turn_right"] = end_road_id
                            
                            # Populate the lanes field based on the receiving lane index
                            for lane_link in link.get("laneLinks", []):
                                end_lane_idx = lane_link.get("endLaneIndex")
                                if end_lane_idx is not None and end_lane_idx not in road_info["lanes"][turn_type]:
                                    road_info["lanes"][turn_type].append(end_lane_idx)
                
                # Finalize and sort the lane lists for consistency
                road_info["lanes"] = {k: sorted(v) for k, v in road_info["lanes"].items()}
                agent_intersection_info["roads"][road_id] = road_info

            self.inter_info_dict[inter_id] = agent_intersection_info
        
        print(f"Created inter_info_dict: {len(self.inter_info_dict)} intersections")

    def build_road_network_graphs(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build lane-interaction graph and road network graphs from SUMO environment.
        Based on the logic from examples/control_examples/env.py:_create_road_network.
        
        Graphs are built automatically during reset(). This method returns the cached graphs.
        Use force_rebuild=True to rebuild if needed.
        
        Args:
            force_rebuild (bool): If True, rebuilds graphs even if cached. Default False.
        
        Returns:
            Dictionary containing:
                - lane_inter_graph: DiGraph connecting lane_groups to intersections
                - lane_dict: Dictionary mapping lane_id to lane metadata (only lanes connected to intersections)
                - road_graph: DiGraph connecting roads (edges) based on network topology (only intersection-related roads)
                - road_dict: Dictionary mapping road_id to road attributes (only roads connected to intersections: incoming and outgoing)
                - highway_segment_graph: DiGraph connecting highway segments (if available)
                - ramp_lane_graph: DiGraph connecting ramps to controlled lanes and their 2-hop neighbors (if available)
        
        Note: This method only builds graphs for roads and lanes connected to intersections (incoming and outgoing).
        Highway roads are built separately via _build_highway_info_dict() from the entire network.
        Ramps are identified and their controlled lanes are mapped via _build_ramp_lane_graph().
        """
        # Return cached graphs if available and not forcing rebuild
        if self._road_network_graphs is not None and not force_rebuild:
            return self._road_network_graphs
        
        # If graphs haven't been built yet (shouldn't happen if reset() was called), build them now
        if self._road_network_graphs is None:
            print("Warning: Road network graphs not yet built. Building now...")
        
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for building graphs. Install with: pip install networkx")
        
        direction_abbreviation = {
            "go_straight": "T",
            "turn_left": "L",
            "turn_right": "R"
        }
        
        num_intersections = len(self.intersection_dict)
        LI_G = nx.DiGraph()  # Lane-interaction graph
        R_G = nx.DiGraph()  # Road-to-road graph
        lane_dict = {}
        road_dict = {}
        
        # Step 1: Process incoming roads and build lane-interaction connections
        for idx, (name, intersection) in enumerate(self.intersection_dict.items()):
            roads = self.inter_info_dict[name]['roads']
            
            for road_id, road_info in roads.items():
                if road_info['type'] == 'incoming':
                    for direction, lane_list in road_info['lanes'].items():
                        start_lanes = "/".join(f"{road_id}_{lane}" for lane in lane_list)
                        if start_lanes not in LI_G.nodes:
                            LI_G.add_node(start_lanes, length=road_info['length'])
                        
                        LI_G.add_edge(start_lanes, name, 
                                    type=f"{road_info['location'].lower()}-{direction}",
                                    loc_dir=f"{road_info['location'][0].upper()}{direction_abbreviation[direction]}")
                        
                        # Update lane_dict
                        lane_dict.update({
                            f"{road_id}_{lane}": {
                                "road_id": road_id,
                                "location": road_info['location'],
                                "direction": direction,
                                "lane_group": start_lanes
                            } for lane in lane_list
                        })
                        
        
        # Step 2: Process outgoing roads
        for idx, (name, intersection) in enumerate(self.intersection_dict.items()):
            roads = self.inter_info_dict[name]['roads']
            
            for road_id, road_info in roads.items():
                if road_info['type'] == 'outgoing':
                    for direction, lane_list in road_info['lanes'].items():
                        lanes = [f"{road_id}_{lane}" for lane in lane_list]
                        
                        # First, determine the lane_group for this set of lanes
                        # Check if any lane is already in lane_dict
                        existing_lane_group = None
                        for lane in lanes:
                            if lane in lane_dict:
                                existing_lane_group = lane_dict[lane]['lane_group']
                                break
                        
                        # Create the lane_group (use existing if found, otherwise create new)
                        if existing_lane_group:
                            lane_group = existing_lane_group
                        else:
                            lane_group = "/".join(f"{road_id}_{lane}" for lane in lane_list)
                            # Add the lane_group node to the graph if it doesn't exist
                            if lane_group not in LI_G.nodes:
                                # Get road length from road_info if available, otherwise use 0
                                road_length = road_info.get('length', 0.0)
                                LI_G.add_node(lane_group, length=road_length)
                        
                        # Add edge from intersection to lane_group
                        LI_G.add_edge(name, lane_group, 
                                    type=f"{road_info['location'].lower()}-{direction}", 
                                    loc_dir=f"{road_info['location'][0].upper()}{direction_abbreviation[direction]}")
        
                        # Update lane_dict for all lanes in this group
                        lane_dict.update({f"{road_id}_{lane}": {
                            "road_id": road_id,
                            "location": road_info['location'],
                            "direction": direction,
                            "lane_group": lane_group
                        } for lane in lane_list if f"{road_id}_{lane}" not in lane_dict})
        
        # Step 3: Build road graph and road_dict with road attributes
        # Only include roads that are connected to intersections (incoming or outgoing)
        # Collect all road IDs that are connected to intersections
        intersection_related_road_ids = set()
        
        for inter_id, inter_info in self.inter_info_dict.items():
            roads = inter_info.get('roads', {})
            for road_id, road_info in roads.items():
                # Include both incoming and outgoing roads
                if road_info.get('type') in ['incoming', 'outgoing']:
                    intersection_related_road_ids.add(road_id)
        
        # Build road_dict and road_graph only for intersection-related roads
        road_to_node_map = {}  # Maps road_id to (from_node, to_node)
        
        if self.sumo_net is not None:
            # Process only intersection-related roads from SUMO network
            for edge in self.sumo_net.getEdges():
                edge_id = edge.getID()
                # Skip internal edges
                if edge_id.startswith(":"):
                    continue
                
                # Only process roads connected to intersections
                if edge_id not in intersection_related_road_ids:
                    continue
                
                try:
                    # Get road attributes from SUMO network
                    road_type = edge.getType()
                    priority = edge.getPriority()
                    num_lanes = len(edge.getLanes())
                    # Get speed from first lane (typically all lanes have same speed)
                    speed = edge.getLanes()[0].getSpeed() if edge.getLanes() else 0.0
                    # Get road length
                    length = edge.getLength()
                    
                    # Get from/to nodes
                    from_node = edge.getFromNode().getID()
                    to_node = edge.getToNode().getID()
                    
                    # Build road_dict entry
                    road_dict[edge_id] = {
                        "id": edge_id,
                        "type": road_type if road_type else "",
                        "priority": int(priority) if priority is not None else -1,
                        "numLanes": num_lanes,
                        "speed": float(speed),
                        "length": float(length),
                        "from": from_node,
                        "to": to_node,
                    }
                    
                    # Store mapping for efficient graph building
                    road_to_node_map[edge_id] = (from_node, to_node)
                    
                except Exception as e:
                    print(f"Warning: Error processing road {edge_id}: {e}")
                    continue
        
        # Fallback: Also add roads from roads_data if sumo_net is not available
        if (not road_dict or len(road_dict) == 0) and hasattr(self, 'roads_data') and self.roads_data:
            for road_id, road_data in self.roads_data.items():
                if road_id.startswith(":"):
                    continue
                
                # Only process intersection-related roads
                if road_id not in intersection_related_road_ids:
                    continue
                
                # Skip if already in road_dict
                if road_id in road_dict:
                    continue
                
                road_dict[road_id] = {
                    "id": road_id,
                    "type": "",  # Not available in roads_data
                    "priority": -1,  # Not available in roads_data
                    "numLanes": len(road_data.get("lanes", [])),
                    "speed": float(road_data.get("lanes", [{}])[0].get("maxSpeed", 0.0)) if road_data.get("lanes") else 0.0,
                    "from": road_data.get("startIntersection", ""),
                    "to": road_data.get("endIntersection", ""),
                }
        
        # Build road graph connections only for intersection-related roads
        if road_to_node_map:
            # Build a reverse index: to_node -> list of road_ids ending at that node
            node_to_roads = {}
            for road_id, (from_node, to_node) in road_to_node_map.items():
                if to_node not in node_to_roads:
                    node_to_roads[to_node] = []
                node_to_roads[to_node].append(road_id)
            
            # Build road graph: connect roads based on from/to relationships
            # If road A ends at node N and road B starts at node N, add edge A -> B
            for road_id, (from_node, to_node) in road_to_node_map.items():
                # Find all roads ending at this road's start node
                if from_node in node_to_roads:
                    for upstream_road_id in node_to_roads[from_node]:
                        # Only add edges if both roads are intersection-related
                        if upstream_road_id in road_dict and road_id in road_dict:
                            R_G.add_edge(upstream_road_id, road_id)
        
        # Build highway segment graph (requires highway_dict to be populated)
        # Note: highway_dict is populated in reset(), so this may be None if called before reset()
        if self.highway_dict:
            self._build_highway_segment_graph()
        
        # Build ramp-lane graph (requires ramp_dict to be populated)
        # Note: ramp_dict is populated in reset(), so this may be None if called before reset()
        if self.ramp_dict:
            self._build_ramp_lane_graph()
        else:
            self.ramp_lane_graph = None
            self.ramp_lane_dict = {}

        # Cache the results - only intersection-related roads and lanes
        self._road_network_graphs = {
            "lane_inter_graph": LI_G,
            "lane_dict": lane_dict,
            "road_graph": R_G,
            "road_dict": road_dict,  # Only contains intersection-related roads
            "highway_segment_graph": self.highway_segment_graph,
            "ramp_lane_graph": self.ramp_lane_graph
        }
        
        print(f"Built road network graphs: {len(road_dict)} intersection-related roads, {len(lane_dict)} lanes")

        # Sync specialized lane_dict info to Foundation Layer lane_dict
        self._sync_lane_dict_to_foundation_layer(lane_dict)

        # Save to cache after building (if not already cached)
        # Note: This will be called from _load_or_build_road_network_graphs, which handles caching
        # But we also call it here as a safety measure if build_road_network_graphs is called directly
        if hasattr(self, 'inter_info_dict') and self.inter_info_dict:
            self._save_runtime_data_cache()
        
        return self._road_network_graphs

    def _save_runtime_data_cache(self):
        """
        Save inter_info_dict, road_network_graphs, highway_info_dict, highway_subgraph,
        and Foundation Layer data to cache file.
        This is called after building these structures for the first time.
        """
        cache_path = self._get_runtime_data_cache_path()

        try:
            # Prepare data to cache
            cache_data = {
                "inter_info_dict": self.inter_info_dict,
                "road_network_graphs": self._road_network_graphs,
                "highway_info_dict": self.highway_info_dict,
                "highway_subgraph": self.highway_subgraph,
                "highway_segment_graph": self.highway_segment_graph,
                "ramp_lane_dict": getattr(self, 'ramp_lane_dict', {}),
                # Foundation Layer (Layer 1)
                "network_graphs": getattr(self, 'network_graphs', {}),
                "network_dicts": getattr(self, 'network_dicts', {}),
                "cache_version": "2.0"  # Updated version for Foundation Layer
            }

            # Save using pickle (supports networkx graphs and complex data structures)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"Saved runtime network data to cache: {cache_path}")
            print(f"  - inter_info_dict: {len(self.inter_info_dict)} intersections")
            if self._road_network_graphs:
                print(f"  - road_network_graphs: {list(self._road_network_graphs.keys())}")
            if self.highway_info_dict:
                print(f"  - highway_info_dict: {len(self.highway_info_dict)} highway roads")
            if self.highway_subgraph is not None:
                print(f"  - highway_subgraph: {len(self.highway_subgraph.nodes())} nodes")
            if hasattr(self, 'ramp_lane_dict') and self.ramp_lane_dict:
                print(f"  - ramp_lane_dict: {len(self.ramp_lane_dict)} lanes")
            if self.highway_segment_graph is not None:
                print(f"  - highway_segment_graph: {len(self.highway_segment_graph.nodes())} segments")
            # Foundation Layer stats
            if hasattr(self, 'network_graphs') and self.network_graphs:
                for graph_name, graph in self.network_graphs.items():
                    if graph is not None:
                        print(f"  - network_graphs['{graph_name}']: {len(graph.nodes())} nodes")
            if hasattr(self, 'network_dicts') and self.network_dicts:
                for dict_name, d in self.network_dicts.items():
                    if d:
                        print(f"  - network_dicts['{dict_name}']: {len(d)} entries")
        except Exception as e:
            print(f"Warning: Failed to save runtime data cache to {cache_path}: {e}")
            import traceback
            traceback.print_exc()

    def _initialize_control_modules(self):
        """
        Initialize enabled control modules based on self.control_modules.
        Each control module can provide configuration and control logic.
        """
        from control_modules import get_control_module
        
        self.enabled_controls = {}
        
        for module_name in self.control_modules:
            print(f"Initializing control module: {module_name}")
            
            if module_name == 'signal_timing':
                # Initialize traffic signal control module
                module = get_control_module('signal_timing', config_dir_name=self.config_dir_name)
                if module:
                    # Generate fresh default config (not loaded from file)
                    config = module.get_default_config(env=self)

                    self.enabled_controls['signal_timing'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    print(f"Signal control module initialized with {len(config)} intersections")
            
            elif module_name == 'subway_scheduling':
                # Initialize subway scheduling control module
                module = get_control_module('subway_scheduling', config_dir_name=self.config_dir_name)
                if module:
                    # Generate fresh default config (not loaded from file)
                    print(f"Generating default configuration for {module_name}...")
                    config = module.get_default_config(env=self)
                    
                    self.enabled_controls['subway_scheduling'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    print(f"Subway scheduling module initialized with {len(config)} routes")
            
            elif module_name == 'bus_scheduling':
                # Initialize bus scheduling control module
                module = get_control_module('bus_scheduling', config_dir_name=self.config_dir_name)
                if module:
                    # Generate fresh default config (not loaded from file)
                    print(f"Generating default configuration for {module_name}...")
                    config = module.get_default_config(env=self)
                    
                    self.enabled_controls['bus_scheduling'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    print(f"Bus scheduling module initialized with {len(config)} routes")
            
            elif module_name == 'highway_speed_limit':
                # Initialize highway speed limit control module
                module = get_control_module('highway_speed_limit', config_dir_name=self.config_dir_name)
                if module:
                    # Generate fresh default config (not loaded from file)
                    config = module.get_default_config(env=self)
                    
                    self.enabled_controls['highway_speed_limit'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    print(f"Highway speed limit control module initialized with {len(config)} highway segments")
            
            elif module_name == 'ramp_metering':
                # Initialize ramp metering control module
                module = get_control_module('ramp_metering', config_dir_name=self.config_dir_name)
                if module:
                    config = module.get_default_config(env=self)
                    
                    self.enabled_controls['ramp_metering'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    print(f"Ramp metering control module initialized with {len(config)} ramps")

            elif module_name == 'taxi_scheduling':
                # Initialize taxi scheduling control module
                module = get_control_module('taxi_scheduling', config_dir_name=self.config_dir_name)
                if module:
                    config = module.get_default_config(env=self)

                    # Propagate USE_SIMULATED_TAXI_SYSTEM flag and file paths
                    if self.dic_traffic_env_conf.get("USE_SIMULATED_TAXI_SYSTEM", False):
                        config["use_simulated_taxi_system"] = True
                        # Build candidate directories: config dir + net-file dir
                        search_dirs = []
                        cfg_dir = os.path.dirname(os.path.abspath(self.config_path))
                        search_dirs.append(cfg_dir)
                        # Also try the net-file's directory (actual scenario dir)
                        try:
                            import xml.etree.ElementTree as _ET
                            _tree = _ET.parse(self.config_path)
                            _net_el = _tree.find(".//net-file")
                            if _net_el is not None:
                                _net_val = _net_el.get("value", "")
                                if os.path.isabs(_net_val):
                                    _net_dir = os.path.dirname(_net_val)
                                else:
                                    _net_dir = os.path.dirname(os.path.join(cfg_dir, _net_val))
                                if _net_dir and _net_dir not in search_dirs:
                                    search_dirs.append(_net_dir)
                        except Exception:
                            pass
                        for sd in search_dirs:
                            persons_file = os.path.join(sd, "persons.taxi.xml")
                            fleet_file = os.path.join(sd, "taxi_fleet.rou.xml")
                            if os.path.exists(persons_file) and not config.get("reservation_file_path"):
                                config["reservation_file_path"] = persons_file
                            if os.path.exists(fleet_file) and not config.get("taxi_fleet_file_path"):
                                config["taxi_fleet_file_path"] = fleet_file
                        print(f"[SimTaxi] Propagated USE_SIMULATED_TAXI_SYSTEM into module config")
                        print(f"[SimTaxi]   reservation_file_path: {config.get('reservation_file_path')}")
                        print(f"[SimTaxi]   taxi_fleet_file_path: {config.get('taxi_fleet_file_path')}")
                        print(f"[SimTaxi]   search_dirs: {search_dirs}")

                    self.enabled_controls['taxi_scheduling'] = {
                        'module': module,
                        'config': config,
                        'state': None  # Will be initialized on first use
                    }
                    fleet_size = config.get('fleet_size', 0)
                    print(f"Taxi scheduling control module initialized with fleet size: {fleet_size}")
            
            else:
                print(f"Warning: Unknown control module '{module_name}', skipping...")
        
        print(f"Initialized {len(self.enabled_controls)} control modules: {list(self.enabled_controls.keys())}")

    def _is_highway_road(self, road_id: str, road_info: Dict[str, Any]) -> bool:
        """
        Determines if a road is a highway/expressway based on road attributes.
        
        Args:
            road_id (str): Road ID
            road_info (dict): Road information from road_dict
            
        Returns:
            bool: True if the road is a highway/expressway, False otherwise
        """
        # Check by road type keywords
        road_type = road_info.get("type", "").lower()
        highway_keywords = ['highway', 'motorway', 'expressway', 'trunk']
        is_highway_by_type = any(keyword in road_type for keyword in highway_keywords)
        
        # Check by speed threshold (highways typically have higher speeds)
        speed = road_info.get("max_speed", 0.0)
        is_highway_by_speed = speed >= 25.0  # ~90 km/h threshold
        
        return is_highway_by_type or is_highway_by_speed

    # ==================== FOUNDATION LAYER (Layer 1) ====================
    # Complete network graphs and dictionaries covering ALL lanes/roads/transit
    # These serve as the authoritative source from which zone subgraphs and
    # module-specific graphs are derived.
    # ====================================================================

    def _build_foundation_layer(self):
        """
        Build the Foundation Layer (Layer 1) containing complete network graphs and dictionaries.

        This creates:
        - network_graphs["lane_graph"]: Complete lane connectivity graph (all lanes, excluding internal)
        - network_graphs["road_graph"]: Complete road connectivity graph (all roads)
        - network_graphs["transit_graph"]: Complete transit network graph (routes + stations)
        - network_dicts["lane_dict"]: Complete lane metadata dictionary
        - network_dicts["road_dict"]: Complete road metadata dictionary
        - network_dicts["station_dict"]: Complete station metadata dictionary

        Note: Internal lanes (those starting with ':') are excluded from the lane_graph nodes,
        but their connections are tracked via edge attributes.
        """
        print("Building Foundation Layer (Layer 1)...")

        # Initialize network_graphs and network_dicts
        self.network_graphs = {
            "lane_graph": None,
            "road_graph": None,
            "transit_graph": None
        }
        self.network_dicts = {
            "lane_dict": {},
            "road_dict": {},
            "station_dict": {}
        }

        if self.sumo_net is None:
            print("Warning: SUMO network not loaded. Cannot build Foundation Layer.")
            return

        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not available. Cannot build Foundation Layer graphs.")
            return

        # Build complete road graph and road_dict
        self._build_complete_road_graph()

        # Build complete lane graph and lane_dict
        self._build_complete_lane_graph()

        # Build transit graph (uses existing TransitGraphBuilder)
        self._build_complete_transit_graph()

        # Print statistics
        if self.network_graphs["lane_graph"] is not None:
            print(f"Foundation Layer: lane_graph has {len(self.network_graphs['lane_graph'].nodes())} nodes, "
                  f"{len(self.network_graphs['lane_graph'].edges())} edges")
        if self.network_graphs["road_graph"] is not None:
            print(f"Foundation Layer: road_graph has {len(self.network_graphs['road_graph'].nodes())} nodes, "
                  f"{len(self.network_graphs['road_graph'].edges())} edges")
        if self.network_dicts["lane_dict"]:
            print(f"Foundation Layer: lane_dict has {len(self.network_dicts['lane_dict'])} lanes")
        if self.network_dicts["road_dict"]:
            print(f"Foundation Layer: road_dict has {len(self.network_dicts['road_dict'])} roads")

    def _classify_road(self, edge_id: str, road_type: str, speed: float) -> str:
        """
        Classify a road into categories: 'normal', 'highway', 'trunk', or 'ramp'.

        Args:
            edge_id: The road/edge ID
            road_type: SUMO road type string
            speed: Maximum speed in m/s

        Returns:
            Road category string: 'normal', 'highway', 'trunk', or 'ramp'
        """
        road_type_lower = road_type.lower() if road_type else ""

        # Check for ramp
        ramp_keywords = ['ramp', 'on_ramp', 'off_ramp', 'onramp', 'offramp']
        if any(kw in road_type_lower for kw in ramp_keywords):
            return 'ramp'

        # Check for highway/motorway
        highway_keywords = ['highway', 'motorway', 'expressway']
        if any(kw in road_type_lower for kw in highway_keywords):
            return 'highway'

        # Check for trunk roads
        if 'trunk' in road_type_lower:
            return 'trunk'

        # Check by speed (highways typically >= 90 km/h = 25 m/s)
        if speed >= 25.0:
            return 'highway'

        return 'normal'

    def _classify_lane(self, lane_id: str, road_category: str) -> str:
        """
        Classify a lane based on its parent road category.

        Args:
            lane_id: The lane ID
            road_category: The parent road's category

        Returns:
            Lane type string: 'normal', 'highway', or 'ramp'
        """
        # Map road categories to lane types
        if road_category in ['highway', 'trunk']:
            return 'highway'
        elif road_category == 'ramp':
            return 'ramp'
        else:
            return 'normal'

    def _build_complete_road_graph(self):
        """
        Build the complete road graph containing ALL roads in the network.

        Creates:
        - network_graphs["road_graph"]: NetworkX DiGraph with all roads as nodes
        - network_dicts["road_dict"]: Complete road metadata dictionary

        Node attributes:
        - road_type: SUMO type string
        - road_category: 'normal' | 'highway' | 'trunk' | 'ramp'
        - num_lanes: int
        - length: float
        - max_speed: float
        - from_node: str
        - to_node: str
        - zone_id: str | None

        Edge attributes:
        - via_node: str (the junction connecting roads)
        - is_signalized: bool (whether the junction has traffic light)
        """
        import networkx as nx

        road_graph = nx.DiGraph()
        road_dict = {}
        road_to_node_map = {}  # Maps road_id to (from_node, to_node)

        # Get zone mappings if available
        edge_to_zone = getattr(self, '_edge_to_zone', {})

        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()

            # Skip internal edges
            if edge_id.startswith(":"):
                continue

            try:
                # Get road attributes
                road_type = edge.getType() or ""
                priority = edge.getPriority()
                num_lanes = len(edge.getLanes())
                length = edge.getLength()

                # Get max speed from lanes
                if edge.getLanes():
                    lanes = edge.getLanes()
                    try:
                        speeds = [lane.getMaxSpeed() for lane in lanes]
                    except AttributeError:
                        speeds = [lane.getSpeed() for lane in lanes]
                    max_speed = max(speeds) if speeds else 0.0
                else:
                    max_speed = 0.0

                # Get from/to nodes
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()

                # Get lane IDs for this road
                lane_ids = [lane.getID() for lane in edge.getLanes()]

                # Classify road
                road_category = self._classify_road(edge_id, road_type, max_speed)

                # Get zone_id if available
                zone_id = edge_to_zone.get(edge_id)

                # Build road_dict entry
                road_info = {
                    "road_type": road_type,
                    "road_category": road_category,
                    "priority": int(priority) if priority is not None else -1,
                    "num_lanes": num_lanes,
                    "length": float(length),
                    "max_speed": float(max_speed),
                    "from_node": from_node,
                    "to_node": to_node,
                    "zone_id": zone_id,
                    "lanes": lane_ids,
                    "highway_segment_id": None  # Will be populated later if applicable
                }
                road_dict[edge_id] = road_info

                # Add node to graph
                road_graph.add_node(
                    edge_id,
                    road_type=road_type,
                    road_category=road_category,
                    num_lanes=num_lanes,
                    length=float(length),
                    max_speed=float(max_speed),
                    from_node=from_node,
                    to_node=to_node,
                    zone_id=zone_id
                )

                # Store mapping for edge building
                road_to_node_map[edge_id] = (from_node, to_node)

            except Exception as e:
                print(f"Warning: Error processing road {edge_id} in Foundation Layer: {e}")
                continue

        # Build road graph edges based on node connectivity
        # Build reverse index: node_id -> list of roads ending at that node
        node_to_incoming_roads = {}
        for road_id, (from_node, to_node) in road_to_node_map.items():
            if to_node not in node_to_incoming_roads:
                node_to_incoming_roads[to_node] = []
            node_to_incoming_roads[to_node].append(road_id)

        # Get signalized junctions
        signalized_nodes = set()
        for node in self.sumo_net.getNodes():
            if node.getType() == 'traffic_light':
                signalized_nodes.add(node.getID())

        # Connect roads: if road A ends at node N and road B starts at node N, add edge A -> B
        for road_id, (from_node, to_node) in road_to_node_map.items():
            # Find all roads ending at this road's start node
            if from_node in node_to_incoming_roads:
                for upstream_road_id in node_to_incoming_roads[from_node]:
                    is_signalized = from_node in signalized_nodes
                    road_graph.add_edge(
                        upstream_road_id,
                        road_id,
                        via_node=from_node,
                        is_signalized=is_signalized
                    )

        self.network_graphs["road_graph"] = road_graph
        self.network_dicts["road_dict"] = road_dict

    def _build_complete_lane_graph(self):
        """
        Build the complete lane graph containing ALL lanes in the network.
        Excludes internal lanes (those starting with ':') from nodes, but tracks
        connections via internal lanes in edge attributes.

        Creates:
        - network_graphs["lane_graph"]: NetworkX DiGraph with all non-internal lanes as nodes
        - network_dicts["lane_dict"]: Complete lane metadata dictionary

        Node attributes:
        - lane_type: 'normal' | 'highway' | 'ramp'
        - road_id: str
        - length: float
        - max_speed: float
        - zone_id: str | None

        Edge attributes:
        - connection_type: 'direct' | 'via_junction'
        - junction_id: str | None
        - via_internal_lane: str | None (the internal lane used for connection)
        """
        import networkx as nx

        lane_graph = nx.DiGraph()
        lane_dict = {}

        # Get zone mappings if available
        lane_to_zone = getattr(self, '_lane_to_zone', {})
        road_dict = self.network_dicts.get("road_dict", {})

        # First pass: add all non-internal lanes as nodes
        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()

            # Skip internal edges
            if edge_id.startswith(":"):
                continue

            road_info = road_dict.get(edge_id, {})
            road_category = road_info.get("road_category", "normal")

            for lane in edge.getLanes():
                lane_id = lane.getID()

                try:
                    length = lane.getLength()
                    try:
                        max_speed = lane.getMaxSpeed()
                    except AttributeError:
                        max_speed = lane.getSpeed()
                    lane_index = lane.getIndex()

                    # Classify lane
                    lane_type = self._classify_lane(lane_id, road_category)

                    # Get zone_id
                    zone_id = lane_to_zone.get(lane_id)

                    # Build lane_dict entry
                    lane_info = {
                        "road_id": edge_id,
                        "lane_type": lane_type,
                        "lane_index": lane_index,
                        "length": float(length),
                        "max_speed": float(max_speed),
                        "zone_id": zone_id,
                        # Direction/location info (populated later for intersection-connected lanes)
                        "direction": None,
                        "location": None,
                        "lane_group": None,
                        # Control associations
                        "connected_intersections": [],
                        "controlled_by_ramp": None
                    }
                    lane_dict[lane_id] = lane_info

                    # Add node to graph
                    lane_graph.add_node(
                        lane_id,
                        lane_type=lane_type,
                        road_id=edge_id,
                        length=float(length),
                        max_speed=float(max_speed),
                        zone_id=zone_id
                    )

                except Exception as e:
                    print(f"Warning: Error processing lane {lane_id} in Foundation Layer: {e}")
                    continue

        # Second pass: build lane connections
        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()

            # Skip internal edges
            if edge_id.startswith(":"):
                continue

            for lane in edge.getLanes():
                lane_id = lane.getID()

                try:
                    # Get outgoing connections
                    outgoing_connections = lane.getOutgoing()

                    for connection in outgoing_connections:
                        to_lane = connection.getToLane()
                        to_lane_id = to_lane.getID()

                        # If connection goes through an internal lane
                        if to_lane_id.startswith(":"):
                            # Find the final destination lane(s)
                            internal_lane = to_lane
                            internal_lane_id = to_lane_id

                            # Extract junction_id from internal lane ID (format: :junction_id_...)
                            junction_id = None
                            if internal_lane_id.startswith(":"):
                                parts = internal_lane_id[1:].split("_")
                                if parts:
                                    junction_id = parts[0]

                            # Get connections from internal lane to final lane
                            try:
                                internal_outgoing = internal_lane.getOutgoing()
                                for internal_conn in internal_outgoing:
                                    final_lane = internal_conn.getToLane()
                                    final_lane_id = final_lane.getID()

                                    # Skip if final lane is also internal
                                    if final_lane_id.startswith(":"):
                                        continue

                                    # Add edge with connection info
                                    lane_graph.add_edge(
                                        lane_id,
                                        final_lane_id,
                                        connection_type="via_junction",
                                        junction_id=junction_id,
                                        via_internal_lane=internal_lane_id
                                    )
                            except Exception:
                                pass
                        else:
                            # Direct connection (no internal lane)
                            lane_graph.add_edge(
                                lane_id,
                                to_lane_id,
                                connection_type="direct",
                                junction_id=None,
                                via_internal_lane=None
                            )

                except Exception as e:
                    # Silently skip connection errors for individual lanes
                    pass

        self.network_graphs["lane_graph"] = lane_graph
        self.network_dicts["lane_dict"] = lane_dict

        # Enrich lane_dict with intersection connection info
        self._enrich_lane_dict_with_intersections()

    def _enrich_lane_dict_with_intersections(self):
        """
        Enrich lane_dict entries with intersection connection information.
        Uses inter_info_dict if available to add direction, location, lane_group,
        and connected_intersections information.
        """
        if not hasattr(self, 'inter_info_dict') or not self.inter_info_dict:
            return

        lane_dict = self.network_dicts.get("lane_dict", {})

        for inter_id, inter_info in self.inter_info_dict.items():
            roads = inter_info.get('roads', {})

            for road_id, road_info in roads.items():
                location = road_info.get('location', '')
                road_type = road_info.get('type', '')  # 'incoming' or 'outgoing'

                for direction, lane_indices in road_info.get('lanes', {}).items():
                    # Build lane_group string
                    lane_ids = [f"{road_id}_{idx}" for idx in lane_indices]
                    lane_group = "/".join(lane_ids)

                    for lane_id in lane_ids:
                        if lane_id in lane_dict:
                            lane_dict[lane_id]['direction'] = direction
                            lane_dict[lane_id]['location'] = location
                            lane_dict[lane_id]['lane_group'] = lane_group

                            # Add intersection to connected_intersections list
                            if inter_id not in lane_dict[lane_id]['connected_intersections']:
                                lane_dict[lane_id]['connected_intersections'].append(inter_id)

    def _build_complete_transit_graph(self):
        """
        Build the complete transit graph from the TransitGraphBuilder.
        Also builds station_dict from available station information.
        """
        # Check if transit builder is available
        if not hasattr(self, '_transit_builder') or self._transit_builder is None:
            # Try to build transit graph directly if we have the necessary data
            if hasattr(self, 'transit_graph') and self.transit_graph is not None:
                self.network_graphs["transit_graph"] = self.transit_graph
            return

        # Use the transit builder to build the graph
        try:
            transit_graph = self._transit_builder.build_transit_graph()
            self.network_graphs["transit_graph"] = transit_graph
        except Exception as e:
            print(f"Warning: Failed to build transit graph in Foundation Layer: {e}")

        # Build station_dict from bus_stations and subway_stations
        station_dict = {}

        # Add bus stations
        for station_id, station in getattr(self, 'bus_stations', {}).items():
            lane_id = getattr(station, 'lane_id', None)
            zone_id = self._lane_to_zone.get(lane_id) if lane_id else None

            station_dict[station_id] = {
                "station_type": "bus_stop",
                "lane_id": lane_id,
                "zone_id": zone_id,
                "position": getattr(station, 'start_pos', 0.0),
                "routes": []  # Will be populated from transit_graph if available
            }

        # Add subway stations
        for station_id, station in getattr(self, 'subway_stations', {}).items():
            lane_id = getattr(station, 'lane_id', None)
            zone_id = self._lane_to_zone.get(lane_id) if lane_id else None

            station_dict[station_id] = {
                "station_type": "subway_station",
                "lane_id": lane_id,
                "zone_id": zone_id,
                "position": getattr(station, 'start_pos', 0.0),
                "routes": []
            }

        # Enrich with route information from transit_graph
        transit_graph = self.network_graphs.get("transit_graph")
        if transit_graph is not None:
            for node_id in transit_graph.nodes():
                node_data = transit_graph.nodes[node_id]
                if node_data.get('node_type') == 'route':
                    # Find all stations served by this route
                    for _, station_id, edge_data in transit_graph.out_edges(node_id, data=True):
                        if edge_data.get('edge_type') == 'serves':
                            if station_id in station_dict:
                                if node_id not in station_dict[station_id]['routes']:
                                    station_dict[station_id]['routes'].append(node_id)

        self.network_dicts["station_dict"] = station_dict

    def _sync_lane_dict_to_foundation_layer(self, specialized_lane_dict: Dict):
        """
        Sync lane information from specialized lane_dict (Layer 3) to Foundation Layer lane_dict.

        This ensures that intersection-related information (direction, location, lane_group,
        connected_intersections) from the specialized graphs is available in the complete
        Foundation Layer lane_dict.

        Args:
            specialized_lane_dict: Lane dictionary from build_road_network_graphs()
                                   containing only intersection-related lanes
        """
        if not hasattr(self, 'network_dicts') or not self.network_dicts.get("lane_dict"):
            return

        foundation_lane_dict = self.network_dicts["lane_dict"]

        for lane_id, lane_info in specialized_lane_dict.items():
            if lane_id in foundation_lane_dict:
                # Update Foundation Layer with specialized info
                foundation_lane_dict[lane_id]['direction'] = lane_info.get('direction')
                foundation_lane_dict[lane_id]['location'] = lane_info.get('location')
                foundation_lane_dict[lane_id]['lane_group'] = lane_info.get('lane_group')

                # Add to connected_intersections if not already present
                # Note: This requires inferring the intersection from the lane_group relationship
                # which is done in _enrich_lane_dict_with_intersections

    def _sync_ramp_info_to_foundation_layer(self):
        """
        Sync ramp control information to Foundation Layer lane_dict.

        Updates lanes in the Foundation Layer that are controlled by ramp metering
        with the controlling ramp ID.
        """
        if not hasattr(self, 'network_dicts') or not self.network_dicts.get("lane_dict"):
            return

        if not hasattr(self, 'ramp_dict') or not self.ramp_dict:
            return

        foundation_lane_dict = self.network_dicts["lane_dict"]

        for ramp_id, ramp_obj in self.ramp_dict.items():
            # Get controlled lanes from the ramp
            lane_info = ramp_obj.get_controlled_lanes_from_signal()
            controlled_lanes = lane_info.get("controlled_lanes", [])

            for lane_id in controlled_lanes:
                if lane_id in foundation_lane_dict:
                    foundation_lane_dict[lane_id]['controlled_by_ramp'] = ramp_id

    def _build_highway_info_dict(self):
        """
        Identifies all highway/expressway roads from the entire SUMO network and builds highway_info_dict and highway_subgraph.
        highway_info_dict contains all highway roads from the entire network (not just intersection-related roads).
        highway_subgraph is a NetworkX graph containing only highway roads and their connections.
        Both are cached as runtime data.
        Highway objects are created later in reset() when traci_conn is available.
        
        Note: This method builds highway data independently from road_dict, which only contains intersection-related roads.
        """
        self.highway_info_dict = {}
        self.highway_subgraph = None
        
        if self.sumo_net is None:
            print("Warning: SUMO network not loaded. Cannot identify highway segments.")
            return
        
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not available. Cannot build highway subgraph.")
            nx = None
        
        # Build highway_info_dict and highway graph from entire SUMO network
        highway_road_ids = []
        highway_road_to_node_map = {}  # Maps road_id to (from_node, to_node)
        
        # Step 1: Identify all highway roads from entire network
        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()
            # Skip internal edges
            if edge_id.startswith(":"):
                continue
            
            try:
                # Get road attributes from SUMO network
                road_type = edge.getType()
                priority = edge.getPriority()
                num_lanes = len(edge.getLanes())
                # Get max speed from lanes (use getMaxSpeed if available, otherwise getSpeed)
                if edge.getLanes():
                    lanes = edge.getLanes()
                    # Try getMaxSpeed first, fallback to getSpeed
                    try:
                        speeds = [lane.getMaxSpeed() for lane in lanes]
                    except AttributeError:
                        # getMaxSpeed not available, use getSpeed instead
                        speeds = [lane.getSpeed() for lane in lanes]
                    speed = max(speeds) if speeds else 0.0  # Use maximum speed among all lanes
                else:
                    speed = 0.0
                length = edge.getLength()
                
                # Get from/to nodes
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()
                
                # Build road_info dict
                road_info = {
                    "id": edge_id,
                    "type": road_type if road_type else "",
                    "priority": int(priority) if priority is not None else -1,
                    "numLanes": num_lanes,
                    "max_speed": float(speed),
                    "length": float(length),
                    "from": from_node,
                    "to": to_node,
                }
                
                # Check if this is a highway road
                if self._is_highway_road(edge_id, road_info):
                    # Categorize highway: main (speed >= 90km/h = 25.0 m/s) or trunk (ramp/trunk)
                    speed_mps = float(speed)
                    if speed_mps >= 25.0:  # 90 km/h = 25.0 m/s
                        road_info["cate"] = "main"
                    else:
                        road_info["cate"] = "trunk"
                    
                    highway_road_ids.append(edge_id)
                    self.highway_info_dict[edge_id] = road_info
                    highway_road_to_node_map[edge_id] = (from_node, to_node)
                    
            except Exception as e:
                print(f"Warning: Error processing road {edge_id} for highway identification: {e}")
                continue
        
        if not highway_road_ids:
            print("No highway/expressway roads found in the network.")
            return
        
        print(f"Found {len(highway_road_ids)} highway/expressway roads (from entire network)")
        
        # Step 2: Build highway_subgraph (NetworkX graph containing only highway roads)
        if nx is not None:
            # Create a new graph for highways
            highway_graph = nx.DiGraph()
            
            # Add all highway roads as nodes
            for road_id in highway_road_ids:
                highway_graph.add_node(road_id)
            
            # Build connections between highway roads based on from/to relationships
            # Build a reverse index: to_node -> list of road_ids ending at that node
            node_to_highway_roads = {}
            for road_id, (from_node, to_node) in highway_road_to_node_map.items():
                if to_node not in node_to_highway_roads:
                    node_to_highway_roads[to_node] = []
                node_to_highway_roads[to_node].append(road_id)
            
            # Build highway graph: connect highways based on from/to relationships
            # If highway A ends at node N and highway B starts at node N, add edge A -> B
            for road_id, (from_node, to_node) in highway_road_to_node_map.items():
                if from_node in node_to_highway_roads:
                    for upstream_road_id in node_to_highway_roads[from_node]:
                        if upstream_road_id in highway_road_ids and road_id in highway_road_ids:
                            highway_graph.add_edge(upstream_road_id, road_id)
            
            self.highway_subgraph = highway_graph
            print(f"Built highway_subgraph: {len(self.highway_subgraph.nodes())} nodes, {len(self.highway_subgraph.edges())} edges")
        else:
            print("Warning: networkx not available. Cannot build highway_subgraph.")
    
    def _should_skip_intersection(self, tls_id: str) -> Tuple[bool, str]:
        """
        Check if an intersection should be skipped based on phase criteria.
        
        Args:
            tls_id: Traffic light system ID
            
        Returns:
            Tuple of (should_skip, reason)
            should_skip: True if intersection should be skipped, False otherwise
            reason: Reason for skipping (empty string if not skipping)
        """
        try:
            # Get the traffic light program
            logic_list = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            if not logic_list:
                return True, "no TLS program found"
            
            logic = logic_list[0]
            phases = logic.getPhases()
            
            if not phases:
                return True, "no phases found"
            
            # Check if any phase name contains "RAMP" - skip ramp metering intersections
            # These are controlled separately and should not be managed by signal timing control
            has_ramp_phase = any(
                phase.name and "RAMP" in phase.name.upper() 
                for phase in phases 
                if phase.name
            )
            if has_ramp_phase:
                return True, "ramp metering intersection (contains RAMP in phase names)"
            
            # Check if only one phase exists
            if len(phases) == 1:
                return True, "only one phase"
            
            # Check if any phase has no name - if so, skip this intersection
            any_phase_nameless = any(not phase.name or phase.name.strip() == "" for phase in phases)
            if any_phase_nameless:
                return True, "at least one phase has no name"
            
            return False, ""
            
        except (traci.TraCIException, IndexError, AttributeError) as e:
            # If we can't get phase information, skip this intersection
            return True, f"failed to get phase info: {e}"

    def _initialize_intersections(self):
        """
        Creates Intersection objects from intersections_data.
        This is called in reset() after traci_conn is available.
        Skips intersections with only one phase, all phases without names, or ramp metering intersections.
        """
        self.intersection_dict = {}
        self.id_to_index = {}
        # list_inter_log removed - no longer needed for RL training
        print(f"Creating Intersection objects for {len(self.intersections_data)} intersections...")
        
        # Pass the global lane_length dict to the intersection for use in feature calcs
        self.dic_traffic_env_conf["lane_length"] = self.lane_length
        
        skipped_count = 0
        skipped_intersections = []
        
        for idx, (inter_id, _) in enumerate(self.intersections_data.items()):
            # Check if this intersection should be skipped
            should_skip, reason = self._should_skip_intersection(inter_id)
            if should_skip:
                skipped_count += 1
                skipped_intersections.append((inter_id, reason))
                continue
            
            custom_phases = self.inter_phase_mapping.get(inter_id)
            try:
                intersection = Intersection(
                    tls_id=inter_id,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    traci_conn=self.traci_conn,
                    sumo_net=self.sumo_net,
                    path_to_log=self.path_to_log,
                    custom_phase_list=custom_phases
                )
                self.intersection_dict[inter_id] = intersection
                # Use actual index (deprecated, kept for compatibility)
                actual_idx = len(self.intersection_dict) - 1
                self.id_to_index[inter_id] = actual_idx
                # list_inter_log removed - no longer needed for RL training
            except Exception as e:
                print(f"ERROR: Failed to initialize Intersection object for {inter_id}: {e}")
                self.close()
                raise
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} intersections:")
            for inter_id, reason in skipped_intersections[:10]:  # Show first 10 skipped intersections
                print(f"  - {inter_id}: {reason}")
            if len(skipped_intersections) > 10:
                print(f"  ... and {len(skipped_intersections) - 10} more")
        
        print(f"Successfully initialized {len(self.intersection_dict)} intersections")
    
    def _build_ramp_info_dict(self):
        """
        Identifies ramp metering TLS (traffic light systems) from intersections_data.
        A ramp TLS is identified by having phase names containing "RAMP".
        This builds ramp_info_dict which contains ramp metadata.
        """
        self.ramp_info_dict = {}
        
        if not self.traci_conn:
            print("Warning: TraCI connection not available. Cannot identify ramps.")
            return
        
        if not self.intersections_data:
            print("Warning: intersections_data not available. Cannot identify ramps.")
            return
        
        ramp_count = 0
        
        for tls_id, inter_data in self.intersections_data.items():
            try:
                # Check if this TLS has RAMP phases
                logic_list = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                if not logic_list:
                    continue
                
                logic = logic_list[0]
                phases = logic.getPhases()
                
                if not phases:
                    continue
                
                # Check if any phase name contains "RAMP"
                has_ramp_phase = any(
                    phase.name and "RAMP" in phase.name.upper() 
                    for phase in phases 
                    if phase.name
                )
                
                if has_ramp_phase:
                    # This is a ramp metering TLS
                    ramp_info = {
                        "ramp_id": tls_id,
                        "tls_id": tls_id,
                        "point": inter_data.get("point", {"x": 0, "y": 0}),
                        "controlled_nodes": inter_data.get("controlled_nodes", []),
                        "graph_node_ids": inter_data.get("graph_node_ids", [tls_id])
                    }
                    self.ramp_info_dict[tls_id] = ramp_info
                    ramp_count += 1
            except (traci.TraCIException, IndexError, AttributeError) as e:
                # Skip TLS that can't be queried
                continue
        
        print(f"Found {ramp_count} ramp metering TLS")
    
    def _initialize_ramps(self):
        """
        Creates Ramp objects from ramp_info_dict.
        This is called in reset() after traci_conn is available.
        """
        self.ramp_dict = {}
        
        # Build ramp_info_dict if not already built
        if not self.ramp_info_dict:
            self._build_ramp_info_dict()
        
        if not self.ramp_info_dict:
            print("No ramp metering TLS found. Skipping ramp initialization.")
            return
        
        print(f"Initializing {len(self.ramp_info_dict)} ramps...")
        
        for ramp_id, ramp_info in self.ramp_info_dict.items():
            try:
                ramp = Ramp(
                    tls_id=ramp_id,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    traci_conn=self.traci_conn,
                    sumo_net=self.sumo_net,
                    path_to_log=self.path_to_log,
                    adjacency_info=None  # Can be enhanced later if needed
                )
                self.ramp_dict[ramp_id] = ramp
            except Exception as e:
                print(f"Warning: Failed to initialize Ramp {ramp_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Successfully initialized {len(self.ramp_dict)} ramps")
    
    def _initialize_highways(self):
        """
        Creates Highway objects from highway_info_dict and highway_subgraph.
        Groups connected highway roads into segments using highway_subgraph.
        This is called in reset() after traci_conn is available.
        """
        self.highway_dict = {}
        
        # Ensure highway_info_dict and highway_subgraph are available
        if not self.highway_info_dict:
            print("Warning: highway_info_dict not built yet. Building now...")
            self._build_highway_info_dict()
        
        if not self.highway_info_dict:
            print("No highway roads found. Skipping highway initialization.")
            return
        
        # Filter to only include main roads (speed >= 90km/h), exclude trunk/ramp roads
        main_road_ids = [
            road_id for road_id, road_info in self.highway_info_dict.items()
            if road_info.get("cate") == "main"
        ]
        
        if not main_road_ids:
            print("No main highway roads (speed >= 90km/h) found. Skipping highway initialization.")
            return
        
        print(f"Found {len(main_road_ids)} main highway roads (speed >= 90km/h)")
        
        # Build subgraph for main roads only
        if self.highway_subgraph is not None:
            # Create a subgraph containing only main roads
            main_highway_graph = self.highway_subgraph.subgraph(main_road_ids)
            
            # Use community detection to group highway roads into segments
            # This allows segments to have connections while maintaining internal cohesion
            try:
                # Try to use Louvain community detection (if available)
                import networkx.algorithms.community as nx_comm
                
                # Convert to undirected graph for community detection
                undirected_graph = main_highway_graph.to_undirected()
                
                # Use Louvain algorithm for community detection
                # This groups densely connected roads while allowing inter-segment connections
                communities = nx_comm.louvain_communities(undirected_graph, seed=42)
                connected_components = [list(community) for community in communities]
                print(f"Used Louvain community detection to partition {len(main_road_ids)} main roads into {len(connected_components)} segments")
                
            except (ImportError, AttributeError) as e:
                print(f"Louvain algorithm not available ({e}), trying alternative methods...")
                try:
                    # Fallback 1: Use greedy modularity communities
                    import networkx.algorithms.community as nx_comm
                    undirected_graph = main_highway_graph.to_undirected()
                    communities = nx_comm.greedy_modularity_communities(undirected_graph)
                    connected_components = [list(community) for community in communities]
                    print(f"Used greedy modularity to partition {len(main_road_ids)} main roads into {len(connected_components)} segments")
                    
                except Exception as e2:
                    print(f"Community detection not available ({e2}), using spatial clustering...")
                    # Fallback 2: Use spatial clustering based on road positions
                    connected_components = self._cluster_highways_spatially(main_road_ids, main_highway_graph)
        else:
            # If no highway_subgraph, treat each main road as a separate segment
            connected_components = [[road_id] for road_id in main_road_ids]

        # Normalize and sort components for deterministic segment IDs
        normalized_components = []
        for component in connected_components:
            sorted_component = sorted(component)
            if sorted_component:
                normalized_components.append(sorted_component)
        connected_components = sorted(normalized_components, key=lambda c: c[0])
        
        # Create Highway object for each connected component (segment)
        for segment_idx, road_ids in enumerate(connected_components):
            highway_id = f"highway_segment_{segment_idx}"
            
            try:
                highway = Highway(
                    highway_id=highway_id,
                    road_ids=list(road_ids),
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    traci_conn=self.traci_conn,
                    path_to_log=self.path_to_log,
                    road_dict=self.highway_info_dict,  # Use highway_info_dict (subset of road_dict)
                    adjacency_info=None  # Can be enhanced later if needed
                )
                self.highway_dict[highway_id] = highway
            except Exception as e:
                print(f"Warning: Failed to initialize Highway {highway_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Initialized {len(self.highway_dict)} highway segments")
    
    def _cluster_highways_spatially(self, main_road_ids, main_highway_graph, max_segment_size=20):
        """
        Fallback method: Cluster highway roads spatially when community detection is not available.
        Groups nearby roads into segments while respecting connectivity.
        
        Args:
            main_road_ids: List of main highway road IDs
            main_highway_graph: NetworkX graph of main highway roads
            max_segment_size: Maximum number of roads per segment
            
        Returns:
            List of road ID lists (segments)
        """
        # Get road positions (center points)
        road_positions = {}
        for road_id in main_road_ids:
            road_info = self.highway_info_dict.get(road_id, {})
            # Try to get road center position from SUMO network
            try:
                if self.sumo_net:
                    edge = self.sumo_net.getEdge(road_id)
                    shape = edge.getShape()
                    # Calculate center point
                    if shape:
                        center_x = sum(p[0] for p in shape) / len(shape)
                        center_y = sum(p[1] for p in shape) / len(shape)
                        road_positions[road_id] = (center_x, center_y)
            except Exception as e:
                print(f"Warning: Could not get position for road {road_id}: {e}")
                continue
        
        if not road_positions:
            print("Warning: No road positions available. Falling back to connected components.")
            return [list(component) for component in nx.weakly_connected_components(main_highway_graph)]
        
        # Use simple spatial clustering: group roads by proximity
        # Start with connected components, then merge nearby small segments
        connected_components = list(nx.weakly_connected_components(main_highway_graph))
        
        # If components are already reasonable size, use them
        if len(connected_components) <= 10 and all(len(comp) >= 3 for comp in connected_components):
            return [list(comp) for comp in connected_components]
        
        # Otherwise, try to merge small components or split large ones
        segments = []
        for component in connected_components:
            component_list = list(component)
            if len(component_list) <= max_segment_size:
                segments.append(component_list)
            else:
                # Split large component into smaller segments
                # Simple approach: breadth-first traversal with size limit
                visited = set()
                for start_node in component_list:
                    if start_node not in visited:
                        segment = []
                        queue = [start_node]
                        while queue and len(segment) < max_segment_size:
                            node = queue.pop(0)
                            if node not in visited:
                                visited.add(node)
                                segment.append(node)
                                # Add unvisited neighbors
                                for neighbor in main_highway_graph.neighbors(node):
                                    if neighbor not in visited and neighbor in component:
                                        queue.append(neighbor)
                        if segment:
                            segments.append(segment)
        
        print(f"Used spatial clustering to partition {len(main_road_ids)} main roads into {len(segments)} segments")
        return segments
    
    def _build_highway_segment_graph(self):
        """
        Builds a NetworkX DiGraph connecting highway segments based on road connectivity.
        This graph represents spatial relationships between highway segments for LLM analysis.
        
        Nodes are highway segment IDs (e.g., 'highway_segment_0').
        Edges represent connections: if segment A's exit roads connect to segment B's entry roads, add edge A -> B.
        
        Method:
        1. For each segment, build a subgraph containing only its roads
        2. Find all exit roads (roads with no successors within the segment)
        3. Find all entry roads (roads with no predecessors within the segment)
        4. Connect segments if exit roads of one segment connect to entry roads of another
        """
        if not self.highway_dict:
            print("No highway segments found. Cannot build highway segment graph.")
            return
        
        if self.highway_subgraph is None:
            print("Warning: highway_subgraph not available. Cannot build highway segment graph.")
            return
        
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not available. Cannot build highway segment graph.")
            return
        
        # Create a new directed graph for highway segments
        segment_graph = nx.DiGraph()
        
        # Add all highway segments as nodes
        for highway_id in self.highway_dict.keys():
            segment_graph.add_node(highway_id)
        
        # Build subgraph for each segment and find boundary roads
        segment_exit_roads = {}  # segment_id -> set of (road_id, to_node)
        segment_entry_roads = {}  # segment_id -> set of (road_id, from_node)
        
        for highway_id, highway_obj in self.highway_dict.items():
            if not highway_obj.highway_road_ids:
                continue
            
            # Build subgraph for this segment (only roads within this segment)
            segment_road_ids = set(highway_obj.highway_road_ids)
            segment_subgraph = self.highway_subgraph.subgraph(segment_road_ids)
            
            # Find exit roads: roads with no successors within the segment
            exit_roads = set()
            for road_id in segment_road_ids:
                # Check if this road has any successors within the segment
                successors_in_segment = set(segment_subgraph.successors(road_id)) & segment_road_ids
                if not successors_in_segment:
                    # This is an exit road (no successors within segment)
                    road_info = self.highway_info_dict.get(road_id, {})
                    to_node = road_info.get("to")
                    if to_node:
                        exit_roads.add((road_id, to_node))
            
            # Find entry roads: roads with no predecessors within the segment
            entry_roads = set()
            for road_id in segment_road_ids:
                # Check if this road has any predecessors within the segment
                predecessors_in_segment = set(segment_subgraph.predecessors(road_id)) & segment_road_ids
                if not predecessors_in_segment:
                    # This is an entry road (no predecessors within segment)
                    road_info = self.highway_info_dict.get(road_id, {})
                    from_node = road_info.get("from")
                    if from_node:
                        entry_roads.add((road_id, from_node))
            
            segment_exit_roads[highway_id] = exit_roads
            segment_entry_roads[highway_id] = entry_roads
        
        # Build connections between segments
        # Method 1: Check if segment A's exit roads connect to segment B's entry roads via highway_subgraph
        # For each segment A's exit road, check its successors in highway_subgraph
        # If a successor belongs to segment B, add edge A -> B
        segment_road_to_segment = {}  # Map road_id to segment_id for quick lookup
        for highway_id, highway_obj in self.highway_dict.items():
            for road_id in highway_obj.highway_road_ids:
                segment_road_to_segment[road_id] = highway_id
        
        connections_via_subgraph = 0
        for segment_a_id, exit_roads_a in segment_exit_roads.items():
            for exit_road_id, exit_to_node in exit_roads_a:
                # Check all successors of this exit road in highway_subgraph
                if exit_road_id in self.highway_subgraph:
                    for successor_road_id in self.highway_subgraph.successors(exit_road_id):
                        # Check if this successor belongs to a different segment
                        successor_segment_id = segment_road_to_segment.get(successor_road_id)
                        if successor_segment_id and successor_segment_id != segment_a_id:
                            # Segment A's exit road connects to segment B via highway_subgraph
                            if not segment_graph.has_edge(segment_a_id, successor_segment_id):
                                segment_graph.add_edge(segment_a_id, successor_segment_id)
                                connections_via_subgraph += 1
        
        # Method 2: Also check node-based connections (for cases where segments connect via non-highway roads)
        # If segment A's exit road ends at node N and segment B's entry road starts at node N, add edge A -> B
        connections_via_node = 0
        for segment_a_id, exit_roads_a in segment_exit_roads.items():
            for exit_road_id, exit_to_node in exit_roads_a:
                # Find all segments whose entry roads start at this node
                for segment_b_id, entry_roads_b in segment_entry_roads.items():
                    if segment_a_id == segment_b_id:
                        continue  # Skip self-connections
                    
                    # Skip if already connected via Method 1
                    if segment_graph.has_edge(segment_a_id, segment_b_id):
                        continue
                    
                    for entry_road_id, entry_from_node in entry_roads_b:
                        if exit_to_node == entry_from_node:
                            # Segment A's exit road connects to segment B's entry road via node
                            segment_graph.add_edge(segment_a_id, segment_b_id)
                            connections_via_node += 1
                            break  # Found connection, no need to check other entry roads for this segment
        
        self.highway_segment_graph = segment_graph
        
        # Print detailed statistics
        print(f"Built highway_segment_graph: {len(segment_graph.nodes())} segments, {len(segment_graph.edges())} connections")
        print(f"  - Connections via highway_subgraph: {connections_via_subgraph}")
        print(f"  - Connections via node matching: {connections_via_node}")
        
    
    def _build_ramp_lane_graph(self):
        """
        Builds a NetworkX DiGraph connecting ramps to their controlled lanes and upstream/downstream lanes.
        This graph represents the detailed lane-level connections for ramp metering control using real network topology.
        
        Nodes are:
            - Ramp IDs (from ramp_dict) - with node_type="ramp"
            - Lane IDs (all lanes in the graph) - no node_type (same lane can have different relationships to different ramps)
        
        Edges represent real network connections:
            - ramp -> controlled_lane (edge_type="controls")
            - controlled_lane -> 1-hop downstream lanes (edge_type="leads_to")
            - 1-hop downstream -> 2-hop downstream (edge_type="leads_to")
            - 1-hop upstream -> controlled_lane (edge_type="feeds_into")
            - 2-hop upstream -> 1-hop upstream (edge_type="feeds_into")
        """
        if not self.ramp_dict:
            print("No ramps found. Cannot build ramp-lane graph.")
            self.ramp_lane_graph = None
            self.ramp_lane_dict = {}
            return
        
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not available. Cannot build ramp-lane graph.")
            self.ramp_lane_graph = None
            return
        
        # Create a new directed graph for ramp-lane connections
        ramp_lane_graph = nx.DiGraph()
        
        # Add all ramps as nodes
        for ramp_id in self.ramp_dict.keys():
            ramp_lane_graph.add_node(ramp_id, node_type="ramp")
        
        # Build connections for each ramp
        total_controlled_lanes = 0
        total_edges = 0
        
        for ramp_id, ramp_obj in self.ramp_dict.items():
            # Get controlled lanes from the ramp signal
            lane_info = ramp_obj.get_controlled_lanes_from_signal()
            controlled_lanes = lane_info.get("controlled_lanes", [])
            
            total_controlled_lanes += len(controlled_lanes)
            
            # Add controlled lanes as nodes (without node_type)
            for controlled_lane in controlled_lanes:
                if not ramp_lane_graph.has_node(controlled_lane):
                    ramp_lane_graph.add_node(controlled_lane)
                
                # Add edge: ramp -> controlled_lane
                if not ramp_lane_graph.has_edge(ramp_id, controlled_lane):
                    ramp_lane_graph.add_edge(ramp_id, controlled_lane, edge_type="controls")
                    total_edges += 1
                
                # Build downstream connections layer by layer (real network topology)
                self._add_downstream_lanes_recursive(
                    ramp_lane_graph, controlled_lane, self.sumo_net, max_hops=2, current_hop=0
                )
                
                # Build upstream connections layer by layer (real network topology)
                self._add_upstream_lanes_recursive(
                    ramp_lane_graph, controlled_lane, self.sumo_net, max_hops=2, current_hop=0
                )
        
        self.ramp_lane_graph = ramp_lane_graph
        
        # Build ramp_lane_dict: extract metadata for all lanes in ramp_lane_graph
        self.ramp_lane_dict = {}
        for lane_id in ramp_lane_graph.nodes():
            # Skip ramp nodes
            if ramp_lane_graph.nodes[lane_id].get('node_type') == 'ramp':
                continue
            
            # Extract lane metadata from sumo_net
            lane_info = {}
            try:
                lane_obj = self.sumo_net.getLane(lane_id)
                edge_obj = lane_obj.getEdge()
                road_id = edge_obj.getID()
                
                # Skip internal edges
                if not road_id.startswith(":"):
                    lane_info['road_id'] = road_id
                    
                    # Try to get direction from outgoing connections
                    direction = ''
                    outgoing_connections = lane_obj.getOutgoing()
                    if outgoing_connections:
                        # Get direction from first connection
                        connection = outgoing_connections[0]
                        sumo_direction = connection.getDirection()
                        # Map SUMO direction to our direction format
                        if sumo_direction == 's' or sumo_direction == 'straight':
                            direction = 'go_straight'
                        elif sumo_direction == 'l' or sumo_direction == 'left':
                            direction = 'turn_left'
                        elif sumo_direction == 'r' or sumo_direction == 'right':
                            direction = 'turn_right'
                    
                    lane_info['direction'] = direction
                    lane_info['location'] = []  # Location not easily determined for ramp lanes
                    lane_info['lane_group'] = lane_id  # Use lane_id as lane_group for ramp lanes
                    
                    self.ramp_lane_dict[lane_id] = lane_info
            except (KeyError, AttributeError):
                # If lane not found in sumo_net, create minimal entry
                # Extract road_id from lane_id format (road_id_lane_index)
                if '_' in lane_id:
                    parts = lane_id.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        lane_info['road_id'] = parts[0]
                        lane_info['direction'] = ''
                        lane_info['location'] = []
                        lane_info['lane_group'] = lane_id
                        self.ramp_lane_dict[lane_id] = lane_info
        
        print(f"Built ramp_lane_graph: {len(ramp_lane_graph.nodes())} nodes ({len(self.ramp_dict)} ramps, {total_controlled_lanes} controlled lanes), {len(ramp_lane_graph.edges())} connections")
        print(f"Built ramp_lane_dict: {len(self.ramp_lane_dict)} lanes")

        # Sync ramp control info to Foundation Layer
        self._sync_ramp_info_to_foundation_layer()
    
    def _add_downstream_lanes_recursive(self, graph, from_lane_id, sumo_net, max_hops, current_hop):
        """
        Recursively add downstream lanes following real network topology.
        
        Args:
            graph: NetworkX DiGraph to add edges to
            from_lane_id: Starting lane ID
            sumo_net: SUMO network object
            max_hops: Maximum number of hops to traverse
            current_hop: Current hop count
        """
        if current_hop >= max_hops:
            return
        
        try:
            lane_obj = sumo_net.getLane(from_lane_id)
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
                            if not final_lane_id.startswith(":"):
                                # Add node if not exists
                                if not graph.has_node(final_lane_id):
                                    graph.add_node(final_lane_id)
                                
                                # Add edge: from_lane -> final_lane
                                if not graph.has_edge(from_lane_id, final_lane_id):
                                    graph.add_edge(from_lane_id, final_lane_id, edge_type="leads_to")
                                
                                # Recursively add further downstream lanes
                                self._add_downstream_lanes_recursive(
                                    graph, final_lane_id, sumo_net, max_hops, current_hop + 1
                                )
                    except:
                        pass
                else:
                    # Add node if not exists
                    if not graph.has_node(to_lane_id):
                        graph.add_node(to_lane_id)
                    
                    # Add edge: from_lane -> to_lane
                    if not graph.has_edge(from_lane_id, to_lane_id):
                        graph.add_edge(from_lane_id, to_lane_id, edge_type="leads_to")
                    
                    # Recursively add further downstream lanes
                    self._add_downstream_lanes_recursive(
                        graph, to_lane_id, sumo_net, max_hops, current_hop + 1
                    )
        except (KeyError, AttributeError):
            pass
    
    def _add_upstream_lanes_recursive(self, graph, to_lane_id, sumo_net, max_hops, current_hop):
        """
        Recursively add upstream lanes following real network topology.
        
        Args:
            graph: NetworkX DiGraph to add edges to
            to_lane_id: Target lane ID
            sumo_net: SUMO network object
            max_hops: Maximum number of hops to traverse
            current_hop: Current hop count
        """
        if current_hop >= max_hops:
            return
        
        try:
            lane_obj = sumo_net.getLane(to_lane_id)
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
                    
                    # Add node if not exists
                    if not graph.has_node(incoming_lane_id):
                        graph.add_node(incoming_lane_id)
                    
                    # Add edge: incoming_lane -> to_lane
                    if not graph.has_edge(incoming_lane_id, to_lane_id):
                        graph.add_edge(incoming_lane_id, to_lane_id, edge_type="feeds_into")
                    
                    # Recursively add further upstream lanes
                    self._add_upstream_lanes_recursive(
                        graph, incoming_lane_id, sumo_net, max_hops, current_hop + 1
                    )
        except (KeyError, AttributeError):
            pass
    
    def get_highway_segment_graph(self):
        """
        Returns the highway segment graph for LLM analysis.
        
        Returns:
            NetworkX DiGraph: Graph connecting highway segments, or None if not built
        """
        return self.highway_segment_graph

    def _initialize_subway_infrastructure(self):
        """
        Initialize subway stations and lines from cached data.
        Only initializes if subway_scheduling is in control_modules.
        """
        if 'subway_scheduling' not in self.control_modules:
            self.subway_stations = {}
            self.subway_lines = {}
            return
        
        if not self.subway_routes or not self.subway_stops:
            print("Warning: No subway routes/stops found in cache. Skipping subway initialization.")
            self.subway_stations = {}
            self.subway_lines = {}
            return
        
        print("Initializing subway infrastructure from cached data...")
        
        # Initialize subway stations
        self.subway_stations = {}
        for stop_id in self.subway_stops:
            try:
                self.subway_stations[stop_id] = SubwayStation(stop_id, self.traci_conn)
            except Exception as e:
                print(f"Warning: Failed to initialize subway station {stop_id}: {e}")
        
        print(f"Initialized {len(self.subway_stations)} subway stations.")
        
        # Initialize subway lines with station assignments from cache
        self.subway_lines = {}
        for route_id in self.subway_routes:
            try:
                # Get stations from cached mapping
                station_ids = self.route_stop_mapping.get(route_id, [])
                
                line = SubwayLine(
                    route_id=route_id,
                    traci_conn=self.traci_conn,
                    vehicle_type="subway",
                    stations=station_ids,
                    env=self
                )
                
                self.subway_lines[route_id] = line
                print(f"  {route_id}: Loaded {len(line.stations)} stations from cache")
            except Exception as e:
                print(f"Warning: Failed to initialize subway line {route_id}: {e}")
        
        print(f"Initialized {len(self.subway_lines)} subway lines with station assignments.")

    def _initialize_bus_infrastructure(self):
        """
        Initialize bus stations and lines from cached data.
        Only initializes if bus_scheduling is in control_modules.
        """
        if 'bus_scheduling' not in self.control_modules:
            self.bus_stations = {}
            self.bus_lines = {}
            return
        
        if not self.bus_routes or not self.bus_stops:
            print("Warning: No bus routes/stops found in cache. Skipping bus initialization.")
            self.bus_stations = {}
            self.bus_lines = {}
            return
        
        print("Initializing bus infrastructure from cached data...")
        
        # Initialize bus stations
        self.bus_stations = {}
        for stop_id in self.bus_stops:
            try:
                self.bus_stations[stop_id] = BusStation(stop_id, self.traci_conn)
            except Exception as e:
                print(f"Warning: Failed to initialize bus station {stop_id}: {e}")
        
        print(f"Initialized {len(self.bus_stations)} bus stations.")
        
        # Initialize bus lines with station assignments from cache
        self.bus_lines = {}
        for route_id in self.bus_routes:
            try:
                line = BusLine(route_id, self.traci_conn, vehicle_type="bus")
                
                # Get stations from cached mapping
                station_ids = self.route_stop_mapping.get(route_id, [])
                line.stations = station_ids
                
                self.bus_lines[route_id] = line
                print(f"  {route_id}: Loaded {len(line.stations)} stations from cache")
            except Exception as e:
                print(f"Warning: Failed to initialize bus line {route_id}: {e}")
        
        print(f"Initialized {len(self.bus_lines)} bus lines with station assignments.")

    def get_subway_line(self, route_id: str):
        """Returns the SubwayLine object for the given route ID."""
        return self.subway_lines.get(route_id)

    def get_subway_station(self, station_id: str):
        """Returns the SubwayStation object for the given station ID."""
        return self.subway_stations.get(station_id)

    def get_bus_line(self, route_id: str):
        """Returns the BusLine object for the given route ID."""
        return self.bus_lines.get(route_id)

    def get_bus_station(self, station_id: str):
        """Returns the BusStation object for the given station ID."""
        return self.bus_stations.get(station_id)

    def _initialize_zone_infrastructure(self):
        """
        Initialize zone-based infrastructure organization.

        This method builds:
        - zone_dict: Zone dictionary mapping zone_id to infrastructure info
        - zone_graph: NetworkX DiGraph of zone adjacency relationships
        - transit_graph: NetworkX DiGraph of transit network (routes and stations)
        - bus_route_info: Detailed bus route information with realtime data

        Requires TAZ_FILE_PATH in configuration and bus/subway infrastructure to be initialized first.
        """
        from environment.zone import ZoneManager
        from environment.transit_graph import TransitGraphBuilder

        # Get TAZ file path from configuration
        taz_file_path = self.dic_traffic_env_conf.get('TAZ_FILE_PATH')

        if not taz_file_path:
            print("Note: TAZ_FILE_PATH not configured. Zone infrastructure disabled.")
            return

        import os
        if not os.path.exists(taz_file_path):
            print(f"Warning: TAZ file not found at {taz_file_path}. Zone infrastructure disabled.")
            return

        print(f"Initializing zone infrastructure from {taz_file_path}...")

        try:
            # 1. Initialize ZoneManager and build zone dictionary
            self._zone_manager = ZoneManager(taz_file_path, self.sumo_net, self.traci_conn)

            # Collect highway edges from highway_dict
            highway_edges = set()
            if self.highway_dict:
                for highway in self.highway_dict.values():
                    if hasattr(highway, 'edges'):
                        highway_edges.update(highway.edges)

            # Collect ramp TLS to lanes mapping
            ramp_tls_lanes = {}
            if self.ramp_dict:
                for ramp_id, ramp in self.ramp_dict.items():
                    if hasattr(ramp, 'controlled_lanes'):
                        ramp_tls_lanes[ramp_id] = ramp.controlled_lanes

            # Build zone dictionary with all infrastructure mappings
            self.zone_dict = self._zone_manager.build_zone_dict(
                highway_edges=highway_edges if highway_edges else None,
                ramp_tls_lanes=ramp_tls_lanes if ramp_tls_lanes else None,
                subway_stations=self.subway_stations if self.subway_stations else None,
                bus_stations=self.bus_stations if self.bus_stations else None
            )

            # Get lane-to-zone mapping for transit graph
            self._edge_to_zone = self._zone_manager.get_edge_to_zone_mapping()
            self._lane_to_zone = self._zone_manager.get_lane_to_zone_mapping()

            # Build zone adjacency graph
            self.zone_graph = self._zone_manager.build_zone_graph(self.zone_dict)

            print(f"Built zone dictionary with {len(self.zone_dict)} zones.")
            print(f"Built zone graph with {self.zone_graph.number_of_nodes()} nodes and {self.zone_graph.number_of_edges()} edges.")

            # 2. Initialize TransitGraphBuilder and build transit structures
            if self.bus_lines or self.subway_lines:
                self._transit_builder = TransitGraphBuilder(
                    bus_lines=self.bus_lines if self.bus_lines else None,
                    subway_lines=self.subway_lines if self.subway_lines else None,
                    bus_stations=self.bus_stations if self.bus_stations else None,
                    subway_stations=self.subway_stations if self.subway_stations else None,
                    zone_dict=self.zone_dict,
                    traci_conn=self.traci_conn,
                    lane_to_zone=self._lane_to_zone
                )

                # Build transit graph
                self.transit_graph = self._transit_builder.build_transit_graph()
                print(f"Built transit graph with {self.transit_graph.number_of_nodes()} nodes and {self.transit_graph.number_of_edges()} edges.")

                # Build bus route info with static data
                self.bus_route_info = self._transit_builder.build_bus_route_info(self.sumo_net)
                print(f"Built bus route info for {len(self.bus_route_info)} routes.")
            else:
                print("No bus/subway lines available. Transit graph not built.")

        except Exception as e:
            print(f"Error initializing zone infrastructure: {e}")
            import traceback
            traceback.print_exc()
            # Reset to empty state on error
            self.zone_dict = {}
            self.zone_graph = None
            self.transit_graph = None
            self.bus_route_info = {}

    def update_bus_realtime_data(self):
        """
        Update real-time bus operation data.

        Call this method during each control cycle to refresh:
        - Active buses per route
        - Current headway
        - Segment travel times and delays
        - Station passenger load
        """
        if self._transit_builder and self.bus_route_info:
            current_step = int(self.traci_conn.simulation.getTime())
            self._transit_builder.update_realtime_data(self.bus_route_info, current_step)

    def get_zone_for_edge(self, edge_id: str):
        """Get the zone ID containing the given edge."""
        return self._edge_to_zone.get(edge_id)

    def get_zone_for_lane(self, lane_id: str):
        """Get the zone ID containing the given lane."""
        return self._lane_to_zone.get(lane_id)

    def get_zone_infrastructure(self, zone_id: str, infra_type: str = 'all'):
        """
        Query infrastructure for a specific zone.

        Args:
            zone_id: Zone ID to query
            infra_type: Type of infrastructure ('all', 'lanes', 'intersections',
                        'highways', 'ramps', 'transit')

        Returns:
            Dict containing requested infrastructure information
        """
        from environment.zone import get_zone_infrastructure
        return get_zone_infrastructure(self.zone_dict, zone_id, infra_type)

    def get_zones_by_infrastructure(self, infra_id: str, infra_type: str):
        """
        Find zones containing a specific infrastructure element.

        Args:
            infra_id: Infrastructure element ID
            infra_type: Type of infrastructure ('edge', 'lane', 'intersection',
                        'highway', 'ramp', 'subway_station', 'bus_stop')

        Returns:
            List of zone IDs containing the infrastructure
        """
        from environment.zone import get_zones_by_infrastructure
        return get_zones_by_infrastructure(self.zone_dict, infra_id, infra_type)

    def step(self, action_dict, min_action_time=15):
        """
        Controls different transportation facilities and advances the simulation.
        
        Args:
            action_dict (dict): Dictionary mapping control type to control actions.
                               Format: {
                                   'signal_timing': {intersection_id: action_idx, ...},
                                   'highway_speed_limit': {
                                       'highway_id': {
                                           'speed_limit_mph': 55
                                       }, ...
                                   },
                                   'highway': {'highway_id': speed_limit, ...},  # Legacy format: speed_limit in m/s
                                   'subway_scheduling': {...},  # TODO
                                   'bus_scheduling': {...},     # TODO
                               }
            min_action_time (float): Duration of simulation step in seconds.
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not self._simulation_running:
            return None, None, True, {"error": "Simulation not running."}
        
        # Check if we should update measurements (at least 600 seconds since last update)
        # This reduces update frequency to improve performance
        current_time = self.traci_conn.simulation.getTime()

        # 1. Process signal timing: update measurements and apply signal control actions
        signal_timing_actions = {}
        if 'signal_timing' in action_dict and 'signal_timing' in self.enabled_controls:
            signal_timing_actions = action_dict['signal_timing']
        
        if signal_timing_actions:
            for inter in self.intersection_dict.values():
                inter.update_current_measurements(self.system_states)
        
        # Update ramp measurements
        if self.ramp_dict:
            for ramp in self.ramp_dict.values():
                ramp.update_current_measurements(self.system_states)
        
        for inter_id in signal_timing_actions:
            # Apply signal control action if available (always apply, regardless of update window)
            action = signal_timing_actions.get(inter_id, -1)
            if inter_id in self.intersection_dict:
                self.intersection_dict[inter_id].set_signal(action, action_pattern="set")
        
        # 2. Process highways: apply speed limit control actions
        highway_config = {}
        if 'highway_speed_limit' in action_dict and 'highway_speed_limit' in self.enabled_controls:
            highway_config = action_dict['highway_speed_limit']
        
        for highway_id, speed_limit_mph in highway_config.items():
            # Apply speed limit control action if available (always apply, regardless of update window)
            # Use set_segment_speed_limit to set the same speed limit for all roads in the segment
            # Config format: {segment_id: speed_limit_mph}
            if highway_id in self.highway_dict:
                self.highway_dict[highway_id].set_segment_speed_limit(speed_limit_mph, unit="mph")

        # 3. Execute one-time control actions BEFORE the simulation step
        # Subway Scheduling: Dispatch trains once per step() call
        if 'subway_scheduling' in action_dict and 'subway_scheduling' in self.enabled_controls:
            subway_actions = action_dict['subway_scheduling']
            if "dispatch_actions" in subway_actions:
                for line_id, dispatch_info in subway_actions["dispatch_actions"].items():
                    line = self.get_subway_line(line_id)
                    if line:
                        line.dispatch_train(
                            position=dispatch_info["position"],
                            schedule=dispatch_info["schedule"],
                            vehicle_type=dispatch_info.get("vehicle_type")
                        )
        
        # 4. Bus Scheduling: Dispatch buses once per step() call
        if 'bus_scheduling' in action_dict and 'bus_scheduling' in self.enabled_controls:
            bus_actions = action_dict['bus_scheduling']
            if "dispatch_actions" in bus_actions:
                for line_id, dispatch_info in bus_actions["dispatch_actions"].items():
                    line = self.get_bus_line(line_id)
                    if line:
                        line.dispatch_bus(
                            position=dispatch_info["position"],
                            schedule=dispatch_info["schedule"],
                            vehicle_type=dispatch_info.get("vehicle_type")
                        )
        
        # 5. Ramp Metering Control
        ramp_metering_actions = {}
        if 'ramp_metering' in action_dict and 'ramp_metering' in self.enabled_controls:
            ramp_metering_actions = action_dict['ramp_metering']
        
        if ramp_metering_actions:
            for ramp_id, is_open in ramp_metering_actions.items():
                # Apply ramp state change if available (always apply, regardless of update window)
                if ramp_id in self.ramp_dict:
                    self.ramp_dict[ramp_id].set_ramp_state(is_open)
        
        # Check vehicle arrivals for delay rate calculation (before simulationStep)
        # This checks the state from the previous step's simulation advancement
        self._check_travel_times()

        # Execute simulation step for the full duration (single step instead of multiple inner steps)
        # simulationStep(target_time) steps to the target simulation time (in seconds)
        try:
            # Step through simulation directly for the full duration
            # Direct phase switching: phases switch directly without yellow/red transitions
            self.traci_conn.simulationStep(self.current_time + min_action_time)
        except (traci.TraCIException, traci.exceptions.FatalTraCIError) as e:
            print(f'TraCI error during step: {e}. Simulation may have ended.')
            self.close()
            return None, None, True, {"error": str(e)}

        # Collect data once at the beginning of step (before action execution)
        # Update previous measurements and collect current system states for intersections
        # Note: These data collections are primarily used by intersection control
        # Only update if at least 600 seconds have passed since last update to improve performance
        try:
            self._update_system_states()
            self._update_waiting_vehicles(min_action_time)
            # Always update passenger waiting times (needed for accurate tracking)
            self._update_waiting_passengers()
        except RuntimeError as e:
            if self._simulation_running:
                self.close()
            return None, None, True, {"error": str(e)}

        # Refresh highway measurements only when needed (throttled + module-aware)
        if self._should_update_highway_measurements(highway_config):
            for highway in self.highway_dict.values():
                highway.update_current_measurements(self.system_states)
            self._last_highway_update_time = self.current_time

        # Only end simulation early if no dynamic scheduling modules are active
        # Dynamic modules (subway_scheduling, bus_scheduling, taxi_scheduling) can affect vehicle presence over time
        has_dynamic_scheduling = False
        if hasattr(self, 'enabled_controls') and self.enabled_controls:
            dynamic_modules = ['subway_scheduling', 'bus_scheduling', 'taxi_scheduling', 'bus_scheduling']
            has_dynamic_scheduling = any(m in self.enabled_controls for m in dynamic_modules)
        
        # if self.traci_conn.simulation.getMinExpectedNumber() == 0 and not has_dynamic_scheduling:
        #     print("Simulation ended: No more vehicles expected.")
        #     self.close()

        if not self._simulation_running: 
            done = True
        else:
            done = False
        
        info = {}
        return None, current_time, done, info  # State and reward removed - no longer needed for RL training

    def _update_waiting_vehicles(self, step_duration):
        """
        Updates the waiting time for vehicles with speed < 0.1 m/s.
        Tracks vehicle position (lane) and resets waiting time if vehicle moves to a new lane.
        This matches the behavior of the original CityFlow implementation.
        
        Args:
            step_duration (float): Duration of the simulation step in seconds.
                                  This should match the min_action_time used in step().
        """
        interval = step_duration
        current_vehicle_speeds = self.system_states.get("get_vehicle_speed", {})
        
        # Build vehicle -> lane mapping from get_lane_vehicles
        vehicle_to_lane = {}
        for lane_id, vehicle_list in self.system_states.get("get_lane_vehicles", {}).items():
            for v_id in vehicle_list:
                vehicle_to_lane[v_id] = lane_id

        # Update or remove vehicles already in waiting list
        for v_id in list(self.waiting_vehicle_list.keys()):
            # Remove if vehicle no longer exists or speed >= 0.1
            if v_id not in current_vehicle_speeds or current_vehicle_speeds[v_id] >= 0.1:
                del self.waiting_vehicle_list[v_id]
            else:
                # Vehicle is still waiting (speed < 0.1)
                current_lane = vehicle_to_lane.get(v_id)
                
                # Check if vehicle data structure needs migration (backward compatibility)
                if isinstance(self.waiting_vehicle_list[v_id], dict):
                    # New format: {v_id: {"time": waiting_time, "lane": lane_id}}
                    waiting_data = self.waiting_vehicle_list[v_id]
                    stored_lane = waiting_data.get("lane")
                    
                    if current_lane is None:
                        # Vehicle lane info not available, keep waiting time
                        waiting_data["time"] += interval
                    elif stored_lane != current_lane:
                        # Vehicle moved to a new lane, reset waiting time
                        waiting_data["time"] = interval
                        waiting_data["lane"] = current_lane
                    else:
                        # Same lane, increment waiting time
                        waiting_data["time"] += interval
                else:
                    # Old format: {v_id: waiting_time} - migrate to new format
                    old_waiting_time = self.waiting_vehicle_list[v_id]
                    self.waiting_vehicle_list[v_id] = {
                        "time": old_waiting_time + interval,
                        "lane": current_lane
                    }

        # Add new waiting vehicles
        for v_id, speed in current_vehicle_speeds.items():
            if v_id not in self.waiting_vehicle_list and speed < 0.1:
                current_lane = vehicle_to_lane.get(v_id)
                self.waiting_vehicle_list[v_id] = {
                    "time": interval,
                    "lane": current_lane
                }
    
    def _update_waiting_passengers(self):
        """
        Updates the waiting time for passengers at subway/bus stations.
        Tracks passengers who are waiting at bus stops and accumulates their waiting time.
        Uses station-based detection for reliability.
        """
        if not self.traci_conn:
            return

        enabled_modules = set()
        if hasattr(self, "enabled_controls") and self.enabled_controls:
            enabled_modules = set(self.enabled_controls.keys())
        elif self.control_modules:
            enabled_modules = set(self.control_modules)
        if "bus_scheduling" not in enabled_modules and "subway_scheduling" not in enabled_modules:
            return
        
        update_interval = self.dic_traffic_env_conf.get(
            "WAITING_PASSENGER_INTERVAL",
            self.dic_traffic_env_conf.get("INTERVAL", 1.0)
        )
        if update_interval is None or update_interval <= 0:
            update_interval = self.dic_traffic_env_conf.get("INTERVAL", 1.0)

        now_time = self.current_time
        if self._last_passenger_update_time >= 0:
            if now_time - self._last_passenger_update_time < update_interval:
                return
            interval = now_time - self._last_passenger_update_time
        else:
            interval = self.dic_traffic_env_conf.get("INTERVAL", 1.0)
        self._last_passenger_update_time = now_time
        
        # Get all current waiting passengers by checking all bus/subway stops
        current_waiting_passengers = set()
        try:
            # Collect all stop IDs from subway and bus infrastructure
            all_stops = []
            if hasattr(self, 'subway_stops'):
                all_stops.extend(self.subway_stops)
            if hasattr(self, 'bus_stops'):
                all_stops.extend(self.bus_stops)
            
            # Get passengers at each stop
            for stop_id in all_stops:
                try:
                    person_ids = self.traci_conn.busstop.getPersonIDs(stop_id)
                    current_waiting_passengers.update(person_ids)
                except:
                    pass
        except:
            pass
        
        # Remove passengers who are no longer waiting (boarded or left)
        for p_id in list(self.waiting_passenger_list.keys()):
            if p_id not in current_waiting_passengers:
                del self.waiting_passenger_list[p_id]
            else:
                self.waiting_passenger_list[p_id] += interval
        
        # Add new passengers who just started waiting
        for p_id in current_waiting_passengers:
            if p_id not in self.waiting_passenger_list:
                self.waiting_passenger_list[p_id] = interval

    def _should_update_highway_measurements(self, highway_actions: Dict[str, Any]) -> bool:
        """
        Decide whether to refresh highway measurements this step.
        Uses HIGHWAY_MEASUREMENT_INTERVAL (default: 60s). Set <=0 to update every step.
        """
        if not self.highway_dict:
            return False

        enabled_modules = set()
        if hasattr(self, "enabled_controls") and self.enabled_controls:
            enabled_modules = set(self.enabled_controls.keys())
        elif self.control_modules:
            enabled_modules = set(self.control_modules)

        if "highway_speed_limit" not in enabled_modules:
            return False

        interval = self.dic_traffic_env_conf.get("HIGHWAY_MEASUREMENT_INTERVAL", 60.0)
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = 60.0

        if interval <= 0:
            return True

        if highway_actions:
            return True

        if self._last_highway_update_time < 0:
            return True

        return (self.current_time - self._last_highway_update_time) >= interval
    
    def _check_travel_times(self):
        """
        Check vehicle arrivals at stations and calculate delay rate.
        Delegates to each line's check_travel_times method.
        """
        if not self.traci_conn:
            return
        
        try:
            vehicle_ids = self.traci_conn.vehicle.getIDList()
        except traci.TraCIException:
            vehicle_ids = []

        # Check subway lines
        for line in self.subway_lines.values():
            travel_times_cache = self.route_travel_times.get(line.route_id, {})
            line.check_travel_times(travel_times_cache, vehicle_ids=vehicle_ids)
        
        # Check bus lines
        for line in self.bus_lines.values():
            travel_times_cache = self.route_travel_times.get(line.route_id, {})
            line.check_travel_times(travel_times_cache, vehicle_ids=vehicle_ids)
    
    def close(self):
        """
        Closes the TraCI connection and terminates the SUMO subprocess.
        """
        if self._simulation_running and self.traci_conn:
            try:
                self.traci_conn.close()
            except (traci.TraCIException, traci.exceptions.FatalTraCIError):
                pass
            finally:
                self.traci_conn = None
                self._simulation_running = False

        if self.sumo_process:
            try:
                if self.sumo_process.poll() is None:
                    self.sumo_process.terminate()
                    self.sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sumo_process.kill()
            except Exception: pass
            finally:
                self.sumo_process = None
        
        print("SUMO Environment closed.")

    def __del__(self):
        """Ensures the simulation is closed when the object is garbage collected."""
        self.close()

    # ==========================================================================
    # API Methods (State-aware wrappers)
    # ==========================================================================

    # get_feature(), get_state(), and get_reward() methods removed - no longer needed for RL training

    def get_current_time(self):
        """Returns the current simulation time."""
        if not self._simulation_running or self.traci_conn is None:
            return self.current_time
        try:
            return self.traci_conn.simulation.getTime()
        except (traci.TraCIException, traci.exceptions.FatalTraCIError, AttributeError):
            # Connection may have closed unexpectedly; return cached time.
            self._simulation_running = False
            self._traci_healthy = False
            self.traci_conn = None
            return self.current_time

    def is_traci_healthy(self) -> bool:
        """Check if the TraCI connection is healthy.

        Returns False if the connection has been corrupted (e.g., by UnicodeDecodeError)
        or closed unexpectedly. Once unhealthy, the connection cannot recover and
        the simulation should be terminated.
        """
        return self._traci_healthy and self._simulation_running and self.traci_conn is not None

    def mark_traci_unhealthy(self):
        """Mark the TraCI connection as unhealthy.

        This should be called when a TraCI protocol error is detected (e.g., UnicodeDecodeError).
        Once marked unhealthy, is_traci_healthy() will return False.
        """
        self._traci_healthy = False
        print("Warning: TraCI connection marked as unhealthy due to protocol error")

    def get_vehicle_count(self):
        """Gets the total number of running vehicles from SUMO."""
        if not self._simulation_running: return 0
        try: return self.traci_conn.vehicle.getIDCount()
        except traci.TraCIException: return 0

    def get_vehicles(self, include_waiting=False):
        """Gets a list of all vehicle IDs from SUMO."""
        if not self._simulation_running: return []
        try: return self.traci_conn.vehicle.getIDList()
        except traci.TraCIException: return []

    def get_lane_vehicle_count(self):
        """Gets vehicle count per lane from the last step's cached state."""
        if not self._simulation_running: return {}
        return {
            lane: len(vehicles)
            for lane, vehicles in self.system_states.get("get_lane_vehicles", {}).items()
        }

    def get_lane_vehicles(self):
        """Gets vehicle IDs per lane from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_lane_vehicles", {}).copy()

    def get_vehicle_info(self, vehicle_id):
        """Gets detailed information for a specific vehicle from SUMO."""
        default_info = {"running": "false"}
        if not self._simulation_running or vehicle_id not in self.system_states.get("get_vehicle_speed", {}):
            return default_info
        try:
            return {"running": "true", "drivable": self.traci_conn.vehicle.getLaneID(vehicle_id)}
        except traci.TraCIException: return default_info

    def get_vehicle_speed(self):
        """Gets speed for all vehicles from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_vehicle_speed", {}).copy()

    def get_leader(self, vehicle_id):
        """Gets the leader of a specific vehicle from SUMO."""
        if not self._simulation_running: return ""
        try:
            leader_info = self.traci_conn.vehicle.getLeader(vehicle_id)
            return leader_info[0] if leader_info else ""
        except traci.TraCIException: return ""

    def get_average_travel_time(self):
        """Gets the average travel time of vehicles that have finished their trips (version-agnostic)."""
        return float(self._arrived_tt_sum / self._arrived_count) if self._arrived_count > 0 else 0.0

    def get_highway_average_travel_time(self):
        """Gets the average travel time of vehicles that have traveled on highway roads."""
        return float(self._highway_arrived_tt_sum / self._highway_arrived_count) if self._highway_arrived_count > 0 else 0.0

    def get_highway_arrived_vehicle_travel_times(self):
        """Returns {vehicle_id: travel_time} for arrived vehicles that used highway roads."""
        return dict(self._highway_arrived_vehicle_tt)

    def get_highway_arrived_count(self):
        """Gets the number of arrived vehicles that have traveled on highway roads."""
        return int(self._highway_arrived_count)

    def get_arrived_vehicle_travel_times(self):
        """Returns {vehicle_id: travel_time} for all vehicles that have arrived so far."""
        return dict(self._arrived_vehicle_tt)

    def get_global_average_travel_time(self):
        """
        Gets the global average travel time across all checkpoints.
        This metric persists across reset_metrics() calls and accumulates travel times
        from the start of the simulation until reset() is called.
        """
        return float(self._global_arrived_tt_sum / self._global_arrived_count) if self._global_arrived_count > 0 else 0.0

    def get_global_highway_average_travel_time(self):
        """
        Gets the global average travel time for highway vehicles across all checkpoints.
        This metric persists across reset_metrics() calls and accumulates travel times
        from the start of the simulation until reset() is called.
        """
        return float(self._global_highway_arrived_tt_sum / self._global_highway_arrived_count) if self._global_highway_arrived_count > 0 else 0.0

    def get_global_arrived_vehicle_travel_times(self):
        """
        Returns {vehicle_id: travel_time} for all vehicles that have arrived across all checkpoints.
        This includes vehicles from all checkpoint intervals and persists across reset_metrics() calls.
        """
        return dict(self._global_arrived_vehicle_tt)

    def get_global_highway_arrived_vehicle_travel_times(self):
        """
        Returns {vehicle_id: travel_time} for all highway vehicles that have arrived across all checkpoints.
        This includes vehicles from all checkpoint intervals and persists across reset_metrics() calls.
        """
        return dict(self._global_highway_arrived_vehicle_tt)

    def get_global_arrived_count(self):
        """
        Gets the global count of arrived vehicles across all checkpoints.
        This count persists across reset_metrics() calls.
        """
        return int(self._global_arrived_count)

    def get_global_highway_arrived_count(self):
        """
        Gets the global count of arrived highway vehicles across all checkpoints.
        This count persists across reset_metrics() calls.
        """
        return int(self._global_highway_arrived_count)

    def get_average_waiting_time(self):
        """Gets the average waiting time of all currently waiting vehicles (speed < 0.1 m/s)."""
        if not self.waiting_vehicle_list:
            return 0.0
        
        waiting_times = []
        for v_id, time_info in self.waiting_vehicle_list.items():
            if isinstance(time_info, dict):
                # New format: extract "time" field
                waiting_times.append(time_info.get("time", 0.0))
            else:
                # Old format: time_info is directly the waiting time
                waiting_times.append(float(time_info))
        
        return float(np.mean(waiting_times)) if waiting_times else 0.0

    def get_waiting_vehicle_count(self):
        """Gets the number of currently waiting vehicles (speed < 0.1 m/s)."""
        return len(self.waiting_vehicle_list)

    def set_tl_phase(self, intersection_id, phase_id):
        """Sets the traffic light phase for a specific intersection ID."""
        if not self._simulation_running: return
        try: self.traci_conn.trafficlight.setPhase(intersection_id, phase_id)
        except traci.TraCIException as e: print(f"TraCIException setting phase for {intersection_id}: {e}")

    def set_vehicle_speed(self, vehicle_id, speed):
        """Sets the speed for a specific vehicle."""
        if not self._simulation_running: return
        try: self.traci_conn.vehicle.setSpeed(vehicle_id, speed)
        except traci.TraCIException as e: print(f"TraCIException setting speed for vehicle {vehicle_id}: {e}")

    def set_vehicle_route(self, vehicle_id, route):
        """Changes the route of a specific vehicle."""
        if not self._simulation_running: return False
        try:
            self.traci_conn.vehicle.setRoute(vehicle_id, route)
            return True
        except traci.TraCIException as e:
            print(f"TraCIException setting route for vehicle {vehicle_id}: {e}")
            return False

    def set_random_seed(self, seed):
        """No-op for SUMO. The seed must be set at simulation start via reset()."""
        print("Warning: `set_random_seed` has no effect after the simulation has started. Provide a seed to `env.reset()`.")

    def snapshot(self, path=None, extra_metadata: Optional[Dict[str, Any]] = None):
        """
        Takes a snapshot of the current simulation state.
        
        Saves both the SUMO checkpoint file (.xml) and a metadata file (.json)
        containing the simulation time for proper checkpoint resumption.
        
        Returns:
            str: Path to the snapshot file, or None if failed
        """
        if not self._simulation_running:
            print("Warning: Cannot take snapshot. Simulation not running.")
            return None
        try:
            snapshot_path = path if path else os.path.join(self.path_to_log, f"snapshot_{self.current_time}.xml")
            # Ensure directory exists
            snapshot_dir = os.path.dirname(snapshot_path)
            if snapshot_dir and not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir, exist_ok=True)
            
            # Get current simulation time BEFORE saving state
            snapshot_time = self.traci_conn.simulation.getTime()
            
            # Save SUMO state snapshot
            self.traci_conn.simulation.saveState(snapshot_path)
            
            # Verify file was actually created
            import time
            time.sleep(0.1)  # Brief wait for file system sync
            if not os.path.exists(snapshot_path):
                print(f"ERROR: SUMO saveState() did not create file: {snapshot_path}")
                print(f"  Check SUMO logs and TraCI connection status.")
                return None
            
            # Save metadata file with checkpoint time
            metadata_path = os.path.splitext(snapshot_path)[0] + "_metadata.json"
            metadata = {
                "checkpoint_time": float(snapshot_time),
                "checkpoint_file": os.path.basename(snapshot_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save vehicle counts for each line to maintain continuous vehicle IDs
            vehicle_counts = {}
            if hasattr(self, 'subway_lines'):
                for line_id, line in self.subway_lines.items():
                    vehicle_counts[line_id] = line.train_count
            if hasattr(self, 'bus_lines'):
                for line_id, line in self.bus_lines.items():
                    vehicle_counts[line_id] = line.bus_count
            
            if vehicle_counts:
                metadata["vehicle_counts"] = vehicle_counts

            if extra_metadata:
                metadata["extra"] = extra_metadata
            
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                # print(f"Snapshot saved: {snapshot_path} (time: {snapshot_time:.2f}s)")
            except Exception as e:
                print(f"Warning: Failed to save snapshot metadata: {e}")
                # Still return snapshot path even if metadata save failed
            return snapshot_path
        except traci.TraCIException as e:
            print(f"TraCIException taking snapshot: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"Unexpected error taking snapshot: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_state_inplace(self, path: str) -> bool:
        """
        Loads a simulation state by restarting SUMO with --load-state.

        Note: traci.load() does not properly support --load-state parameter,
        so we close the current connection and restart SUMO using traci.start().

        Args:
            path (str): Path to the snapshot file (.xml)

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(path):
            print(f"Error: Snapshot file not found: {path}")
            return False

        # Build SUMO command with --load-state (similar to initial startup)
        sumocfg_path = os.path.abspath(self.config_path)
        sumocfg_dir = os.path.dirname(sumocfg_path)
        self.transit_flow_vehicles_filtered = False

        # If simulated taxi system is enabled, use modified sumocfg
        use_sim_taxi = self.dic_traffic_env_conf.get("USE_SIMULATED_TAXI_SYSTEM", False)
        if use_sim_taxi:
            sumocfg_path = self._prepare_simulated_taxi_sumocfg(sumocfg_path)
            sumocfg_dir = os.path.dirname(sumocfg_path)

        if self.control_modules and (
            "bus_scheduling" in self.control_modules or "subway_scheduling" in self.control_modules
        ):
            sumocfg_path = self._prepare_code_dispatched_transit_sumocfg(sumocfg_path)
            sumocfg_dir = os.path.dirname(sumocfg_path)

        sumo_binary = "sumo-gui" if self.dic_traffic_env_conf.get("USE_GUI", False) else "sumo"
        time_to_teleport = self.dic_traffic_env_conf.get("TIME_TO_TELEPORT", 300)

        sumo_cmd = [
            sumo_binary,
            "-c", sumocfg_path,
            "--load-state", path,
            "--step-length", str(self.dic_traffic_env_conf.get("INTERVAL", 1.0)),
            "--no-step-log", "true",
            "--no-warnings", "false",
            "--ignore-route-errors", "true",
            "--time-to-teleport", str(time_to_teleport),
            "--save-state.transportables",
            "--save-state.rng",
            "--save-state.constraints",
            "--railsignal-moving-block",
        ]

        # Bus/subway scheduling needs tripinfo for ride stats.
        if self.control_modules and any(m in self.control_modules for m in ("bus_scheduling", "subway_scheduling")):
            sumo_cmd.extend(["--device.tripinfo.probability", "1"])

        # Add taxi dispatch algorithms if taxi scheduling is enabled
        # Skip when using simulated taxi system — no SUMO taxi device needed.
        if self.control_modules and 'taxi_scheduling' in self.control_modules and not use_sim_taxi:
            taxi_dispatch_algorithm = self.dic_traffic_env_conf.get("TAXI_DISPATCH_ALGORITHM", "traci")
            # Use randomCircling by default - "stop" can cause SUMO crashes when taxis have no destinations
            taxi_idle_algorithm = self.dic_traffic_env_conf.get("TAXI_IDLE_ALGORITHM", "randomCircling")
            sumo_cmd.extend(["--device.taxi.dispatch-algorithm", str(taxi_dispatch_algorithm)])
            sumo_cmd.extend(["--device.taxi.idle-algorithm", str(taxi_idle_algorithm)])
        elif use_sim_taxi and self.control_modules and 'taxi_scheduling' in self.control_modules:
            print("[SimTaxi] Skipping SUMO taxi device options in load_state_inplace")

        # Try to read checkpoint time from metadata file for explicit --begin.
        metadata_path = os.path.splitext(path)[0] + "_metadata.json"
        self.checkpoint_vehicle_counts = {}
        self.checkpoint_extra_metadata = {}
        self.checkpoint_taxi_state = None
        sim_time = self.dic_traffic_env_conf.get("RUN_COUNTS", 3600)
        checkpoint_time = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                checkpoint_time = metadata.get("checkpoint_time") or metadata.get("sim_time")
                if checkpoint_time is not None:
                    sumo_cmd.extend(["--begin", str(checkpoint_time)])
                    # Add --end parameter to ensure simulation runs long enough
                    # Use a very large end time (24 hours = 86400s) to prevent premature termination
                    end_time = max(86400, checkpoint_time + sim_time + 3600)
                    sumo_cmd.extend(["--end", str(end_time)])
                self.checkpoint_vehicle_counts = metadata.get("vehicle_counts", {}) or {}
                if self.checkpoint_vehicle_counts:
                    print(f"Loaded vehicle counts for {len(self.checkpoint_vehicle_counts)} lines")
                self.checkpoint_extra_metadata = metadata.get("extra", {}) or {}
                self.checkpoint_taxi_state = self.checkpoint_extra_metadata.get("taxi_state")
            except Exception as e:
                print(f"Warning: Failed to read checkpoint metadata from {metadata_path}: {e}")

        # Allocate a fresh TraCI port for the restarted SUMO process.
        self._current_port = find_free_port()
        sumo_cmd.extend(["--remote-port", str(self._current_port)])

        # Prepare SUMO stdout/stderr logs for this load-state run.
        log_dir = self.path_to_log or os.getcwd()
        os.makedirs(log_dir, exist_ok=True)
        if checkpoint_time is not None:
            log_tag = f"loadstate_{int(checkpoint_time)}"
        else:
            log_tag = f"loadstate_{int(time.time())}"
        sumo_stdout_log = os.path.join(log_dir, f"sumo_{log_tag}_stdout.log")
        sumo_stderr_log = os.path.join(log_dir, f"sumo_{log_tag}_stderr.log")
        sumo_log_path = os.path.join(log_dir, f"sumo_{log_tag}.log")
        if self.dic_traffic_env_conf.get("SUMO_LOG_ENABLED", True):
            sumo_cmd.extend(["--log", sumo_log_path])

        # Close the current TraCI connection before restarting SUMO.
        if self.traci_conn:
            try:
                self.traci_conn.close()
            except (traci.TraCIException, traci.exceptions.FatalTraCIError):
                pass
            self.traci_conn = None

        # Note: We don't need to terminate sumo_process separately because
        # traci.close() sends a shutdown command to SUMO. However, for safety
        # we still clean up the process reference.
        self.sumo_process = None
        self._simulation_running = False

        try:
            # Start SUMO process with explicit stdout/stderr logs.
            with open(sumo_stdout_log, "w") as f_out, open(sumo_stderr_log, "w") as f_err:
                self.sumo_process = subprocess.Popen(
                    sumo_cmd,
                    stdout=f_out,
                    stderr=f_err,
                    cwd=sumocfg_dir
                )

            # Give SUMO time to start or fail fast.
            time.sleep(5)
            if self.sumo_process.poll() is not None:
                # Check if failure is due to driveWay error — retry with moving-block
                stderr_content = ""
                try:
                    with open(sumo_stderr_log, "r") as f_err_read:
                        stderr_content = f_err_read.read()
                except IOError:
                    pass

                is_driveway_error = "Unknown driveWay" in stderr_content
                if is_driveway_error and "--railsignal-moving-block" not in sumo_cmd:
                    print("[load_state] SUMO failed with driveWay error, retrying with --railsignal-moving-block ...")
                    sumo_cmd.append("--railsignal-moving-block")
                    self._current_port = find_free_port()
                    # Update port in sumo_cmd
                    for i, arg in enumerate(sumo_cmd):
                        if arg == "--remote-port" and i + 1 < len(sumo_cmd):
                            sumo_cmd[i + 1] = str(self._current_port)
                            break
                    retry_stdout = sumo_stdout_log.replace(".log", "_retry.log")
                    retry_stderr = sumo_stderr_log.replace(".log", "_retry.log")
                    with open(retry_stdout, "w") as f_out2, open(retry_stderr, "w") as f_err2:
                        self.sumo_process = subprocess.Popen(
                            sumo_cmd,
                            stdout=f_out2,
                            stderr=f_err2,
                            cwd=sumocfg_dir
                        )
                    time.sleep(5)
                    if self.sumo_process.poll() is not None:
                        print("[load_state] Retry also failed. Check logs:")
                        print(f"  - STDERR: {retry_stderr}")
                        self.close()
                        return False
                else:
                    error_message = (
                        "SUMO process terminated unexpectedly during load-state. "
                        "Check SUMO logs for details:\n"
                        f"  - STDOUT: {sumo_stdout_log}\n"
                        f"  - STDERR: {sumo_stderr_log}\n"
                    )
                    try:
                        with open(sumo_stdout_log, "r") as f_out_read:
                            log_content = f_out_read.read()
                            if log_content.strip():
                                error_message += f"\n--- SUMO STDOUT ---\n{log_content}\n------------------"
                    except IOError:
                        error_message += "\n(Could not read stdout log.)"
                    if stderr_content.strip():
                        error_message += f"\n--- SUMO STDERR ---\n{stderr_content}\n------------------"
                    print(error_message)
                    self.close()
                    return False

            # Establish TraCI connection.
            self.traci_conn = traci.connect(port=self._current_port, numRetries=10)
            self._simulation_running = True
            # Reset TraCI health on successful load-state restart
            self._traci_healthy = True
            print(f"SUMO restarted with checkpoint. Time: {self.traci_conn.simulation.getTime()}")

        except traci.TraCIException as e:
            error_message = (
                "Failed to connect to SUMO via TraCI after load-state. "
                "Check SUMO logs for details:\n"
                f"  - STDOUT: {sumo_stdout_log}\n"
                f"  - STDERR: {sumo_stderr_log}\n"
                f"Original TraCI error: {e}"
            )
            print(error_message)
            self.close()
            return False
        except Exception as e:
            print(f"Error restarting SUMO with checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self.close()
            return False

        # Reset dynamic state caches
        self.system_states = {}
        self.current_time = 0.0
        self.waiting_vehicle_list = {}
        self.waiting_passenger_list = {}
        self._subscribed_vehicle_ids = set()
        self._last_passenger_update_time = -1.0
        self._depart_time_by_vehicle = {}
        self._arrived_tt_sum = 0.0
        self._arrived_count = 0
        self._arrived_vehicle_tt = {}
        self._highway_vehicle_ids = set()
        self._highway_arrived_tt_sum = 0.0
        self._highway_arrived_count = 0
        self._highway_arrived_vehicle_tt = {}
        self._last_update_time = -600.0
        self._last_highway_update_time = -1.0

        # Refresh bus/subway counters for continuous IDs after loading.
        if self.checkpoint_vehicle_counts:
            if hasattr(self, "bus_lines"):
                for route_id, line in self.bus_lines.items():
                    if route_id in self.checkpoint_vehicle_counts:
                        line.bus_count = self.checkpoint_vehicle_counts[route_id]
            if hasattr(self, "subway_lines"):
                for route_id, line in self.subway_lines.items():
                    if route_id in self.checkpoint_vehicle_counts:
                        line.train_count = self.checkpoint_vehicle_counts[route_id]

        # Ensure lane subscriptions remain active after reload.
        try:
            for lane_id in self.lane_length.keys():
                self.traci_conn.lane.subscribe(lane_id, [
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST
                ])
        except traci.TraCIException as e:
            print(f"Warning: Failed to subscribe lanes after load: {e}")

        # Update traci_conn references in all sub-objects after connection restart
        # This is critical because traci.start() creates a new connection, but sub-objects
        # still hold references to the old (now closed) connection object
        for inter in self.intersection_dict.values():
            inter.traci_conn = self.traci_conn
        for highway in self.highway_dict.values():
            highway.traci_conn = self.traci_conn
        for ramp in self.ramp_dict.values():
            ramp.traci_conn = self.traci_conn
        if hasattr(self, 'bus_lines') and self.bus_lines:
            for line in self.bus_lines.values():
                line.traci_conn = self.traci_conn
        if hasattr(self, 'subway_lines') and self.subway_lines:
            for line in self.subway_lines.values():
                line.traci_conn = self.traci_conn
        if hasattr(self, 'bus_stations') and self.bus_stations:
            for station in self.bus_stations.values():
                station.traci_conn = self.traci_conn
        if hasattr(self, 'subway_stations') and self.subway_stations:
            for station in self.subway_stations.values():
                station.traci_conn = self.traci_conn

        # Refresh system state snapshots
        self._update_system_states()
        for inter in self.intersection_dict.values():
            inter.update_current_measurements(self.system_states)
        for highway in self.highway_dict.values():
            highway.update_current_measurements(self.system_states)
        for ramp in self.ramp_dict.values():
            ramp.update_current_measurements(self.system_states)

        print(f"Checkpoint loaded in-place. Current simulation time: {self.get_current_time():.2f}s")
        return True

    def load_from_file(self, path, use_gui=None, seed=None):
        """
        Loads a simulation state from a snapshot file by restarting SUMO.
        
        Reads the checkpoint time from the metadata file (snapshot_*_metadata.json)
        and uses --begin TIME to start SUMO from the correct simulation time.
        
        Args:
            path (str): Path to the snapshot file (.xml)
            use_gui (bool, optional): Whether to use sumo-gui. If None, uses current setting.
            seed (int, optional): Random seed for the simulation. If None, uses random seed.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(path):
            print(f"Error: Snapshot file not found: {path}")
            return False
        
        # Determine use_gui: use provided value or current setting
        if use_gui is None:
            # Try to detect from current process (not perfect, but better than nothing)
            use_gui = False  # Default to False for checkpoint resumption
        
        # Restart SUMO with checkpoint
        # SUMO will automatically read the simulation time from the checkpoint file
        print(f"Restarting SUMO to load checkpoint from: {path}")
        try:
            self.reset(use_gui=use_gui, seed=seed, load_state_path=path)
            print(f"Checkpoint loaded successfully. Current simulation time: {self.get_current_time():.2f}s")
            return True
        except Exception as e:
            print(f"Error restarting SUMO with checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _post_load_reset(self):
        """Resets internal state after loading a snapshot."""
        print("Resetting internal state after loading snapshot...")
        self.current_time = self.traci_conn.simulation.getTime()
        self._update_system_states()
        for inter in self.intersection_dict.values():
            inter.update_current_measurements(self.system_states)
        self._update_waiting_vehicles()

    # log(), batch_log(), and bulk_log_multi_process() methods removed - no longer needed for RL training

    @staticmethod
    def end_engine():
        """Placeholder method indicating simulation end."""
        print("================ SUMO Process End ================")

    def get_road_network_graphs(self) -> Dict[str, Any]:
        """
        Get the road network graphs.
        """
        return self._road_network_graphs

    def get_controlled_intersection_ids(self) -> List[str]:
        """
        TLS ids that have an ``Intersection`` instance in ``intersection_dict``.

        These are exactly the intersections constructed in ``_initialize_intersections`` (after
        ``_should_skip_intersection`` filtering). RL / signal baselines must only issue
        ``signal_timing`` actions for this set so TraCI targets match initialized controllers.
        """
        return sorted(self.intersection_dict.keys())

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation for SUMOEnv.
        This method creates a copy of the Python-side environment state,
        while nullifying attributes that cannot or should not be copied,
        such as the live TraCI connection and the SUMO process handle.
        The static network object (`sumo_net`) is shared by reference.
        """
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['sumo_process', 'traci_conn', 'sumo_net']:
                continue
            # Perform a deepcopy on all other attributes
            setattr(result, k, deepcopy(v, memo))

        # Manually handle the skipped attributes for the new copy
        result.sumo_process = None
        result.traci_conn = None
        result.sumo_net = self.sumo_net  # Share the reference to the static network data

        return result
