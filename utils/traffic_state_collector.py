import json
import traceback
import traci
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterable
from datetime import datetime
import sys
from environment.sumo_env import SUMOEnv

# Add project root to path for imports
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from utils.id_utils import generate_file_prefix as _generate_file_prefix


def _get_workspace_root() -> Path:
    """Get the workspace root directory."""
    return workspace_root


def _ensure_traffic_states_dir() -> Path:
    """Ensure the traffic_states directory exists and return its path."""
    workspace_root = _get_workspace_root()
    traffic_states_dir = workspace_root / "records" / "traffic_states"
    traffic_states_dir.mkdir(parents=True, exist_ok=True)
    return traffic_states_dir


def collect_lane_conditions(
    traci, 
    sim_time: float,
    lane_dict: Optional[Dict[str, Any]] = None,
    lane_inter_graph: Optional[Any] = None,
    lane_ids: Optional[Iterable[str]] = None,
    system_states: Optional[Dict[str, Any]] = None,
) -> dict[Any, Any]:
    """
    Collect traffic conditions for all lanes at the current simulation time.
    
    Args:
        traci: TraCI connection object
        sim_time: Current simulation time in seconds
        lane_dict: Optional dictionary mapping lane_id to lane metadata (direction, location, lane_group)
        lane_inter_graph: Optional networkx graph for getting start/end intersections
        system_states: Optional env.system_states dict (uses subscription results when available)
        
    Returns:
        Dictionary mapping lane_id to traffic condition dictionary
        Format: {lane_id: {traffic_condition_dict}, ...}
    """
    # Direction abbreviation mapping
    direction_abbreviation = {
        "go_straight": "T",
        "turn_left": "L",
        "turn_right": "R"
    }
    traffic_conditions_dict = {}

    lane_vehicle_map = None
    speed_map = None
    waiting_map = None
    lane_pos_map = None
    active_vehicle_ids = None
    use_subscription = False
    if system_states:
        lane_vehicle_map = system_states.get("get_lane_vehicles")
        speed_map = system_states.get("get_vehicle_speed")
        waiting_map = system_states.get("get_waiting_vehicles")
        lane_pos_map = system_states.get("get_vehicle_lane_position")
        if isinstance(lane_vehicle_map, dict) and isinstance(speed_map, dict):
            use_subscription = True
            if speed_map:
                active_vehicle_ids = set(speed_map.keys())
            elif lane_vehicle_map:
                # Subscription data incomplete; fall back to direct queries for this collection pass.
                use_subscription = False
    if active_vehicle_ids is None:
        try:
            active_vehicle_ids = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            active_vehicle_ids = None
    
    try:
        # Determine lane IDs to scan
        if lane_ids is not None:
            lane_iter = lane_ids
        elif lane_dict:
            lane_iter = lane_dict.keys()
        else:
            lane_iter = []

        # Get all lane IDs
        for lane_id in lane_iter:
            if lane_dict is not None and lane_id not in lane_dict:
                continue
            try:
                # Get basic lane information
                max_speed = traci.lane.getMaxSpeed(lane_id)
                
                # Get lane length from traci
                lane_length = traci.lane.getLength(lane_id)
                
                # Get vehicle information
                if use_subscription:
                    lane_vehicles = list(lane_vehicle_map.get(lane_id, []) or [])
                else:
                    lane_vehicles = list(traci.lane.getLastStepVehicleIDs(lane_id))
                if active_vehicle_ids is not None:
                    lane_vehicles = [vid for vid in lane_vehicles if vid in active_vehicle_ids]
                vehicle_count = len(lane_vehicles)
                
                # Get lane statistics
                occupancy = traci.lane.getLastStepOccupancy(lane_id)  # percentage
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                halting_number = traci.lane.getLastStepHaltingNumber(lane_id)
                
                # Get arrival rate information
                # SUMO time step is typically 1 second, so entering vehicles per step = arrival rate in veh/s
                try:
                    entering_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                    # Assuming 1 second time step, arrival_rate = entering_vehicles / 1.0
                    # For more accurate calculation, we can use a rolling window, but for now use per-step rate
                    arrival_rate = float(entering_vehicles)  # vehicles per second (assuming 1s time step)
                except (AttributeError, Exception) as e:
                    # Fallback: estimate arrival rate from vehicle count changes if entering vehicles not available
                    entering_vehicles = 0
                    arrival_rate = 0.0
                
                # Get vehicle details and calculate queue length in one pass
                # Calculate queue length based on vehicle subscription results (vehicles with speed < 0.1 m/s)
                # This matches the logic used in intersection.py for consistency
                vehicle_details = []
                lane_waiting_count = 0
                waiting_vehicles = []
                waiting_times = []
                
                for veh_id in lane_vehicles:
                    if use_subscription:
                        veh_speed = speed_map.get(veh_id)
                        if veh_speed is None:
                            continue
                        veh_pos = None
                        if lane_pos_map is not None:
                            veh_pos = lane_pos_map.get(veh_id)
                        veh_waiting = 0.0
                        if waiting_map is not None and veh_id in waiting_map:
                            waiting_info = waiting_map.get(veh_id)
                            if isinstance(waiting_info, dict):
                                veh_waiting = float(waiting_info.get("time", 0.0))
                            elif waiting_info is not None:
                                veh_waiting = float(waiting_info)
                    else:
                        try:
                            veh_speed = traci.vehicle.getSpeed(veh_id)
                            veh_pos = traci.vehicle.getLanePosition(veh_id)
                            veh_waiting = traci.vehicle.getWaitingTime(veh_id)
                        except traci.TraCIException:
                            # Vehicle may have left the network between list and query
                            continue
                        except (AttributeError, Exception):
                            # If query fails, skip this vehicle
                            continue

                    vehicle_details.append({
                        'id': veh_id,
                        'speed': veh_speed,
                        'position': veh_pos,
                        'waiting_time': veh_waiting
                    })

                    # If vehicle speed is below threshold, count as waiting vehicle
                    if veh_speed < 0.1:
                        lane_waiting_count += 1
                        waiting_vehicles.append(veh_id)
                        waiting_times.append(veh_waiting)
                
                # Fallback: if no vehicles found or all queries failed, use halting_number
                if not lane_vehicles and halting_number > 0:
                    lane_waiting_count = halting_number
                
                # Calculate average waiting time for waiting vehicles
                avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
                
                # Calculate density (vehicles per meter)
                lane_density = vehicle_count / lane_length if lane_length > 0 else 0.0
                
                # Calculate queue density
                queue_density = lane_waiting_count / lane_length if lane_length > 0 else 0.0
                
                # Count moving vehicles
                moving_vehicles = vehicle_count - halting_number
                
                # Extract road ID (everything before last underscore)
                road_id = "_".join(lane_id.split('_')[:-1]) if '_' in lane_id else lane_id
                
                # Get lane metadata from lane_dict if available
                direction = ''
                location = []
                lane_group = ''
                start_intersection = ''
                end_intersection = ''
                loc_dir = ''
                
                if lane_dict and lane_id in lane_dict:
                    lane_info = lane_dict[lane_id]
                    direction = lane_info.get('direction', '')
                    location = lane_info.get('location', [])
                    lane_group = lane_info.get('lane_group', '')
                    
                    # Calculate loc_dir if we have location and direction
                    if location and direction:
                        location_str = location[0] if isinstance(location, list) else str(location)
                        direction_abbr = direction_abbreviation.get(direction, '')
                        if location_str and direction_abbr:
                            loc_dir = f"{location_str[0].upper()}{direction_abbr}"
                    
                    # Get start and end intersections from lane_inter_graph using lane_group
                    # Note: lane_inter_graph connects lane_groups to intersections, not individual lanes
                    if lane_inter_graph and lane_group and lane_group in lane_inter_graph.nodes():
                        # Get predecessors (start intersections) - intersections that feed into this lane_group
                        start_intersections = [inter for inter in lane_inter_graph.predecessors(lane_group)]
                        start_intersection = start_intersections[0] if start_intersections else ''
                        
                        # Get successors (end intersections) - intersections this lane_group feeds into
                        end_intersections = [inter for inter in lane_inter_graph.successors(lane_group)]
                        end_intersection = end_intersections[0] if end_intersections else ''
                
                # Calculate cell occupancy (divide lane into 4 cells)
                # Cell 0 is near intersection, cell 3 is far from intersection
                cell_occupancy = [0, 0, 0, 0]
                if lane_length > 0:
                    cell_length = lane_length / 4
                    for veh_detail in vehicle_details:
                        pos = veh_detail.get('position')
                        if pos is None:
                            continue
                        cell_idx = min(int(pos / cell_length), 3)
                        cell_occupancy[cell_idx] += 1
                
                # Calculate throughput potential (vehicles that could pass per second)
                # Based on max speed and lane capacity
                throughput_potential = (max_speed / lane_length) * vehicle_count if lane_length > 0 else 0.0
                
                # Create traffic condition dictionary
                traffic_condition = {
                    'lane_id': lane_id,
                    'road_id': road_id,
                    'direction': direction,
                    'location': location,
                    'start_intersection': start_intersection,
                    'end_intersection': end_intersection,
                    'loc_dir': loc_dir,
                    'lane_group': lane_group,
                    'vehicle_count': vehicle_count,
                    'queue_length': lane_waiting_count,
                    'queue_density': queue_density,
                    'moving_vehicles': moving_vehicles,
                    'average_speed': mean_speed,
                    'average_waiting_time': avg_waiting_time,
                    'cell_occupancy': cell_occupancy,
                    'vehicle_details': vehicle_details,
                    'lane_density': lane_density,
                    'throughput_potential': throughput_potential,
                    'lane_length': lane_length,
                    'occupancy': occupancy,  # percentage
                    'halting_number': halting_number,
                    'max_speed': max_speed,
                    'arrival_rate': arrival_rate,  # vehicles per second
                    'entering_vehicles': entering_vehicles,  # number of vehicles entering in last step
                    'simulation_time': sim_time
                }
                
                traffic_conditions_dict[lane_id] = traffic_condition
                
            except Exception as e:
                print(f"Warning: Could not collect data for lane {lane_id}: {e}")
                continue
        
    except Exception as e:
        print(f"Error collecting lane conditions: {e}")
        traceback.print_exc()
    
    return traffic_conditions_dict


def collect_highway_conditions(
    env: SUMOEnv,
    sim_time: float
) -> Dict[str, Dict[str, Any]]:
    """
    Collect traffic conditions for all highway segments at the current simulation time.
    
    Args:
        env: SUMOEnv instance with initialized highways
        sim_time: Current simulation time in seconds
        
    Returns:
        Dictionary mapping highway_segment_id to highway condition dictionary
        Format: {highway_segment_id: {highway_condition_dict}, ...}
    """
    highway_conditions_dict = {}
    
    try:
        if not env.highway_dict:
            return highway_conditions_dict
        
        for highway_id, highway_obj in env.highway_dict.items():
            try:
                # Get features from highway object
                features = highway_obj.get_feature()
                
                # Get current speed limits
                current_speed_limits = highway_obj.get_current_speed_limits()
                
                # Get highway road IDs
                highway_road_ids = highway_obj.get_highway_road_ids()
                
                # Create highway condition dictionary
                # Features are collected at segment level (averages across all roads in segment)
                highway_condition = {
                    'highway_id': highway_id,
                    'simulation_time': sim_time,
                    'road_ids': highway_road_ids,
                    'num_roads': len(highway_road_ids),
                    'current_speed_limits': current_speed_limits,  # road_id -> speed_limit (m/s)
                    'default_speed_limits': highway_obj.default_speed_limits,  # road_id -> default_speed_limit (m/s)
                    # Segment-level features (averages across all roads in segment)
                    'segment_speed': features.get('segment_speed', 0.0),  # Average speed across segment
                    'segment_density': features.get('segment_density', 0.0),  # Average density across segment
                    'segment_occupancy': features.get('segment_occupancy', 0.0),  # Average occupancy across segment
                    'segment_speed_limit': features.get('segment_speed_limit', 0.0),  # Average speed limit across segment
                    'segment_default_speed_limit': features.get('segment_default_speed_limit', 0.0),  # Average default speed limit across segment
                    'segment_congestion_ratio': features.get('segment_congestion_ratio', 0.0),  # Ratio of congested roads in segment
                    'segment_speed_ratio': features.get('segment_speed_ratio', 0.0),  # segment_speed / segment_speed_limit
                    'segment_speed_pressure': features.get('segment_speed_pressure', 0.0),  # segment_speed_limit - segment_speed
                }
                
                highway_conditions_dict[highway_id] = highway_condition
                
            except Exception as e:
                print(f"Warning: Could not collect data for highway segment {highway_id}: {e}")
                continue
        
    except Exception as e:
        print(f"Error collecting highway conditions: {e}")
        traceback.print_exc()
    
    return highway_conditions_dict


def collect_ramp_lane_conditions(
    env: SUMOEnv,
    sim_time: float
) -> Dict[str, Dict[str, Any]]:
    """
    Collect traffic conditions for all lanes in ramp_lane_graph at the current simulation time.
    
    Args:
        env: SUMOEnv instance with initialized ramps and ramp_lane_graph
        sim_time: Current simulation time in seconds
        
    Returns:
        Dictionary mapping lane_id to lane condition dictionary
        Format: {lane_id: {lane_condition_dict}, ...}
        Each lane_condition_dict contains:
            - lane_id: str
            - simulation_time: float
            - lane-level traffic metrics (from collect_lane_conditions)
    """
    ramp_lane_conditions_dict = {}
    
    try:
        if not hasattr(env, 'ramp_lane_graph') or env.ramp_lane_graph is None:
            return ramp_lane_conditions_dict
        
        if not env.ramp_dict:
            return ramp_lane_conditions_dict
        
        # Get ramp_lane_dict (contains metadata for lanes in ramp_lane_graph)
        ramp_lane_dict = getattr(env, 'ramp_lane_dict', {})
        
        # If ramp_lane_dict is empty, return empty dict
        if not ramp_lane_dict:
            return ramp_lane_conditions_dict
        
        # Collect lane conditions only for lanes in ramp_lane_graph (using ramp_lane_dict)
        # collect_lane_conditions will only process lanes that are in ramp_lane_dict
        lane_conditions = collect_lane_conditions(
            env.traci_conn,
            sim_time,
            lane_dict=ramp_lane_dict,  # Use ramp_lane_dict instead of lane_dict (only contains lanes in ramp_lane_graph)
            lane_inter_graph=None,  # Not needed for ramp lanes
            system_states=getattr(env, "system_states", None)
        )

        # Process all lane nodes in ramp_lane_graph (skip ramp nodes)
        for lane_id in ramp_lane_dict.keys():
            try:
                # Get lane condition data (if available)
                lane_condition = lane_conditions.get(lane_id, {})
                
                # Create ramp lane condition dictionary (without relationship field)
                ramp_lane_condition = {
                    'lane_id': lane_id,
                    'simulation_time': sim_time,
                    # Lane-level traffic metrics from collect_lane_conditions
                    'vehicle_count': lane_condition.get('vehicle_count', 0),
                    'queue_length': lane_condition.get('queue_length', 0),
                    'queue_density': lane_condition.get('queue_density', 0.0),
                    'moving_vehicles': lane_condition.get('moving_vehicles', 0),
                    'average_speed': lane_condition.get('average_speed', 0.0),
                    'average_waiting_time': lane_condition.get('average_waiting_time', 0.0),
                    'cell_occupancy': lane_condition.get('cell_occupancy', [0, 0, 0, 0]),
                    'lane_density': lane_condition.get('lane_density', 0.0),  # vehicles per meter
                    'occupancy': lane_condition.get('occupancy', 0.0),  # percentage
                    'lane_length': lane_condition.get('lane_length', 0.0),
                    'halting_number': lane_condition.get('halting_number', 0),
                    'max_speed': lane_condition.get('max_speed', 0.0),
                    'arrival_rate': lane_condition.get('arrival_rate', 0.0),
                    'road_id': lane_condition.get('road_id', ''),
                    'direction': lane_condition.get('direction', ''),
                    'location': lane_condition.get('location', []),
                    'start_intersection': lane_condition.get('start_intersection', ''),
                    'end_intersection': lane_condition.get('end_intersection', ''),
                }
                
                ramp_lane_conditions_dict[lane_id] = ramp_lane_condition
                
            except Exception as e:
                print(f"Warning: Could not collect data for ramp lane {lane_id}: {e}")
                continue
        
    except Exception as e:
        print(f"Error collecting ramp lane conditions: {e}")
        traceback.print_exc()
    
    return ramp_lane_conditions_dict
def collect_subway_conditions(
    env: SUMOEnv,
    sim_time: float
) -> Optional[Dict[str, Any]]:
    """
    Collect subway system state data organized by line.
    Uses SubwayLine.get_state() for basic state, then adds detailed train/station data.
    
    Args:
        env: SUMOEnv instance with initialized subway infrastructure
        sim_time: Current simulation time in seconds
        
    Returns:
        Dictionary containing subway state data organized by line, or None if no subway infrastructure exists
        Format: {"lines": {line_id: {line_data}, ...}}
    """
    # Check if environment has subway infrastructure
    if not hasattr(env, 'subway_stations') or not hasattr(env, 'subway_lines'):
        return None
    
    if not env.subway_stations or not env.subway_lines:
        return None

    if getattr(env, "traci_conn", None) is None:
        return None
    if hasattr(env, "is_traci_healthy") and not env.is_traci_healthy():
        return None
    
    subway_data = {"lines": {}}
    
    # Get subway config from enabled_controls to read headway
    subway_config = {}
    if hasattr(env, 'enabled_controls') and 'subway_scheduling' in env.enabled_controls:
        subway_config = env.enabled_controls['subway_scheduling'].get('config', {})
    
    # Organize data by line
    for line_id, line in env.subway_lines.items():
        # Get basic state from SubwayLine.get_state() (similar to Highway pattern)
        line_state = line.get_state()
        
        # Get headway from config (always up-to-date)
        line_config = subway_config.get(line_id, {})
        headway = line_config.get('headway', 300.0)
        
        # Get delay rate statistics from SubwayLine (uses default tolerance from method)
        delay_stats = line.calculate_delay_rate()
        
        # Build line_data using state from SubwayLine
        line_data = {
            "active_trains": line_state["active_trains"],
            "headway": headway,
            "station_count": line_state["station_count"],
            "trains": {},
            "stations": {},
            "delay_stats": {
                "on_time_rate": round(delay_stats.get("on_time_rate", 1.0), 4),
                "delay_rate": round(delay_stats.get("delay_rate", 0.0), 4),
                "avg_delay": round(delay_stats.get("avg_delay", 0.0), 2),
                "total_segments": delay_stats.get("total_segments", 0)
            }
        }
        
        # Get train_ids and load_ratios from state
        train_ids = line_state["train_ids"]
        load_ratios = line_state["load_ratios"]

        # Get current vehicle list once for validation
        try:
            current_vehicle_ids = set(env.traci_conn.vehicle.getIDList())
        except traci.exceptions.FatalTraCIError as exc:
            if hasattr(env, "mark_traci_unhealthy"):
                env.mark_traci_unhealthy()
            print(f"Warning: Failed to collect subway vehicles: {exc}")
            current_vehicle_ids = set()
        except traci.TraCIException:
            current_vehicle_ids = set()

        # Collect train details for this line
        for i, train_id in enumerate(train_ids):
            # Skip trains that are no longer in the simulation
            if train_id not in current_vehicle_ids:
                continue
            try:
                # Basic info
                current_edge = env.traci_conn.vehicle.getRoadID(train_id)
                speed = env.traci_conn.vehicle.getSpeed(train_id)
                
                # Get vehicle type and capacity
                vehicle_type = env.traci_conn.vehicle.getTypeID(train_id)
                capacity = env.traci_conn.vehicletype.getPersonCapacity(vehicle_type)
                
                # Passenger info
                load_ratio = load_ratios[i] if i < len(load_ratios) else 0.0
                passenger_count = int(load_ratio * capacity)
                
                # Time info
                departure_time = env.traci_conn.vehicle.getDeparture(train_id)
                travel_time = sim_time - departure_time
                
                # Next stop info
                next_station = ""
                dwell_duration = 0.0
                try:
                    stops = env.traci_conn.vehicle.getStops(train_id, limit=1)
                    if stops:
                        next_stop = stops[0]
                        next_station = next_stop.stoppingPlaceID
                        dwell_duration = next_stop.duration
                except:
                    pass
                
                line_data["trains"][train_id] = {
                    "departure_time": round(departure_time, 2),
                    "travel_time": round(travel_time, 2),
                    "current_edge": current_edge,
                    "speed": round(speed, 2),
                    "passenger_count": passenger_count,
                    "capacity": capacity,
                    "load_ratio": round(load_ratio, 2),
                    "next_station": next_station,
                    "next_station_dwell_time": round(dwell_duration, 2)
                }
            except:
                # Train might have been removed or not yet spawned
                pass
        
        # Collect station details for this line
        line_stations = line_state["stations"]
        for station_id in line_stations:
            if station_id in env.subway_stations:
                station = env.subway_stations[station_id]
                waiting_count = station.get_waiting_count()
                
                # Get passenger waiting times at this station
                station_waiting_times = []
                if hasattr(env, 'waiting_passenger_list'):
                    try:
                        person_ids = env.traci_conn.busstop.getPersonIDs(station_id)
                        for p_id in person_ids:
                            if p_id in env.waiting_passenger_list:
                                station_waiting_times.append(env.waiting_passenger_list[p_id])
                    except:
                        pass
                
                # Calculate waiting time statistics
                avg_wait = sum(station_waiting_times) / len(station_waiting_times) if station_waiting_times else 0.0
                max_wait = max(station_waiting_times) if station_waiting_times else 0.0
                
                # Waiting time distribution
                wait_distribution = {"0-60s": 0, "60-180s": 0, "180-300s": 0, ">300s": 0}
                for wt in station_waiting_times:
                    if wt <= 60:
                        wait_distribution["0-60s"] += 1
                    elif wt <= 180:
                        wait_distribution["60-180s"] += 1
                    elif wt <= 300:
                        wait_distribution["180-300s"] += 1
                    else:
                        wait_distribution[">300s"] += 1
                
                line_data["stations"][station_id] = {
                    "waiting_count": waiting_count,
                    "avg_waiting_time": round(avg_wait, 2),
                    "max_waiting_time": round(max_wait, 2),
                    "waiting_time_distribution": wait_distribution
                }
        
        subway_data["lines"][line_id] = line_data
    
    return subway_data


def collect_bus_conditions(
    env: SUMOEnv,
    sim_time: float
) -> Optional[Dict[str, Any]]:
    """
    Collect bus system state data organized by line.
    Uses BusLine.get_state() for basic state, then adds detailed bus/station data.
    
    Args:
        env: SUMOEnv instance with initialized bus infrastructure
        sim_time: Current simulation time in seconds
        
    Returns:
        Dictionary containing bus state data organized by line, or None if no bus infrastructure exists
        Format: {"lines": {line_id: {line_data}, ...}}
    """
    # Check if environment has bus infrastructure
    if not hasattr(env, 'bus_stations') or not hasattr(env, 'bus_lines'):
        return None
    
    if not env.bus_stations or not env.bus_lines:
        return None

    if getattr(env, "traci_conn", None) is None:
        return None
    if hasattr(env, "is_traci_healthy") and not env.is_traci_healthy():
        return None
    
    bus_data = {"lines": {}}
    
    # Get bus config from enabled_controls to read headway
    bus_config = {}
    if hasattr(env, 'enabled_controls') and 'bus_scheduling' in env.enabled_controls:
        bus_config = env.enabled_controls['bus_scheduling'].get('config', {})
    
    # Organize data by line
    for line_id, line in env.bus_lines.items():
        # Get basic state from BusLine.get_state() (similar to Highway pattern)
        line_state = line.get_state()
        
        # Get headway from config (supports timetable format)
        line_config = bus_config.get(line_id, {})
        headway = line_config.get('headway')
        timetable = line_config.get('timetable')
        if isinstance(timetable, list) and timetable:
            time_in_hour = sim_time % 3600
            matched = None
            for segment in timetable:
                time_range = segment.get('time_range')
                if isinstance(time_range, list) and len(time_range) == 2:
                    if time_range[0] <= time_in_hour < time_range[1]:
                        matched = segment
                        break
            if matched is None:
                matched = timetable[0]
            headway = matched.get('headway', headway)
        if headway is None:
            headway = 180.0
        
        # Get delay rate statistics from BusLine (uses default tolerance from method)
        delay_stats = line.calculate_delay_rate()
        
        # Build line_data using state from BusLine
        line_data = {
            "active_buses": line_state["active_buses"],
            "headway": headway,
            "station_count": line_state["station_count"],
            "buses": {},
            "stations": {},
            "delay_stats": {
                "on_time_rate": round(delay_stats.get("on_time_rate", 1.0), 4),
                "delay_rate": round(delay_stats.get("delay_rate", 0.0), 4),
                "avg_delay": round(delay_stats.get("avg_delay", 0.0), 2),
                "total_segments": delay_stats.get("total_segments", 0)
            }
        }
        
        # Get bus_ids and load_ratios from state
        bus_ids = line_state["bus_ids"]
        load_ratios = line_state["load_ratios"]

        # Get current vehicle list once for validation
        try:
            current_vehicle_ids = set(env.traci_conn.vehicle.getIDList())
        except traci.exceptions.FatalTraCIError as exc:
            if hasattr(env, "mark_traci_unhealthy"):
                env.mark_traci_unhealthy()
            print(f"Warning: Failed to collect bus vehicles: {exc}")
            current_vehicle_ids = set()
        except traci.TraCIException:
            current_vehicle_ids = set()

        # Collect bus details for this line
        for i, bus_id in enumerate(bus_ids):
            # Skip buses that are no longer in the simulation
            if bus_id not in current_vehicle_ids:
                continue
            try:
                # Basic info
                current_edge = env.traci_conn.vehicle.getRoadID(bus_id)
                speed = env.traci_conn.vehicle.getSpeed(bus_id)
                
                # Get vehicle type and capacity
                vehicle_type = env.traci_conn.vehicle.getTypeID(bus_id)
                capacity = env.traci_conn.vehicletype.getPersonCapacity(vehicle_type)
                
                # Passenger info
                load_ratio = load_ratios[i] if i < len(load_ratios) else 0.0
                passenger_count = int(load_ratio * capacity)
                
                # Time info
                departure_time = env.traci_conn.vehicle.getDeparture(bus_id)
                travel_time = sim_time - departure_time
                
                # Next stop info
                next_station = ""
                dwell_duration = 0.0
                try:
                    stops = env.traci_conn.vehicle.getStops(bus_id, limit=1)
                    if stops:
                        next_stop = stops[0]
                        next_station = next_stop.stoppingPlaceID
                        dwell_duration = next_stop.duration
                except:
                    pass
                
                line_data["buses"][bus_id] = {
                    "departure_time": round(departure_time, 2),
                    "travel_time": round(travel_time, 2),
                    "current_edge": current_edge,
                    "speed": round(speed, 2),
                    "passenger_count": passenger_count,
                    "capacity": capacity,
                    "load_ratio": round(load_ratio, 2),
                    "next_station": next_station,
                    "next_station_dwell_time": round(dwell_duration, 2)
                }
            except:
                # Bus might have been removed or not yet spawned
                pass
        
        # Collect station details for this line
        line_stations = line_state["stations"]
        for station_id in line_stations:
            if station_id in env.bus_stations:
                station = env.bus_stations[station_id]
                waiting_count = station.get_waiting_count()
                
                # Get passenger waiting times at this station
                station_waiting_times = []
                if hasattr(env, 'waiting_passenger_list'):
                    try:
                        person_ids = env.traci_conn.busstop.getPersonIDs(station_id)
                        for p_id in person_ids:
                            if p_id in env.waiting_passenger_list:
                                station_waiting_times.append(env.waiting_passenger_list[p_id])
                    except:
                        pass
                
                # Calculate waiting time statistics
                avg_wait = sum(station_waiting_times) / len(station_waiting_times) if station_waiting_times else 0.0
                max_wait = max(station_waiting_times) if station_waiting_times else 0.0
                
                # Waiting time distribution
                wait_distribution = {"0-60s": 0, "60-180s": 0, "180-300s": 0, ">300s": 0}
                for wt in station_waiting_times:
                    if wt <= 60:
                        wait_distribution["0-60s"] += 1
                    elif wt <= 180:
                        wait_distribution["60-180s"] += 1
                    elif wt <= 300:
                        wait_distribution["180-300s"] += 1
                    else:
                        wait_distribution[">300s"] += 1
                
                line_data["stations"][station_id] = {
                    "waiting_count": waiting_count,
                    "avg_waiting_time": round(avg_wait, 2),
                    "max_waiting_time": round(max_wait, 2),
                    "waiting_time_distribution": wait_distribution
                }
        
        bus_data["lines"][line_id] = line_data
    
    return bus_data


def init_traffic_states_file(
    simulation_id: Optional[str] = None,
    config_name: Optional[str] = None,
    llm_name: Optional[str] = None,
    control_modules: Optional[List[str]] = None
) -> Path:
    """
    Initialize a new traffic states file for streaming data.
    
    Args:
        simulation_id: Optional simulation ID to include in filename
        config_name: Config directory name (e.g., "jinan") for file naming
        llm_name: LLM model name (e.g., "deepseek-v3.2") for file naming
        control_modules: List of control module names for file naming
        
    Returns:
        Path to the created file
    """
    traffic_states_dir = _ensure_traffic_states_dir()
    
    # Generate file prefix
    file_prefix = _generate_file_prefix(
        config_name=config_name,
        llm_name=llm_name,
        control_modules=control_modules
    )
    
    # Create filename with prefix, timestamp and optional simulation ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if simulation_id:
        filename = f"{file_prefix}_traffic_states_{simulation_id}_{timestamp}.jsonl"
    else:
        filename = f"{file_prefix}_traffic_states_{timestamp}.jsonl"
    
    filepath = traffic_states_dir / filename
    
    # Create file with metadata header (as first JSON line)
    metadata = {
        "type": "metadata",
        "simulation_id": simulation_id,
        "start_time": timestamp,
        "format_version": "1.0",
        "description": "Streaming traffic state data in JSON Lines format"
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
        f.write('\n')
    
    print(f"Initialized traffic states file: {filepath}")
    return filepath


class TrafficStateCollector:
    """
    Manages traffic state collection during simulation.
    Collects snapshots at regular intervals and saves to JSON Lines file at checkpoints.
    """
    
    def __init__(
        self,
        env: SUMOEnv,
        traffic_states_filepath: Path,
        interval: float = 300.0,
        lane_dict: Optional[Dict[str, Any]] = None,
        lane_inter_graph: Optional[Any] = None,
        simulation_id: Optional[str] = None,
        lane_ids: Optional[Iterable[str]] = None
    ):
        """
        Initialize traffic state collector.
        
        Args:
            env: SUMOEnv instance
            traffic_states_filepath: Path to JSON Lines file for saving traffic states
            interval: Write interval in seconds (default: 300)
            lane_dict: Optional dictionary mapping lane_id to lane metadata
            lane_inter_graph: Optional networkx graph for getting start/end intersections
            simulation_id: Optional simulation ID to tag snapshots (for filtering during read)
        """
        self.env = env
        self.traffic_states_filepath = traffic_states_filepath
        self.interval = interval
        self.lane_dict = lane_dict
        self.lane_inter_graph = lane_inter_graph
        self.simulation_id = simulation_id  # Store simulation_id for tagging snapshots
        self.lane_ids = list(lane_ids) if lane_ids is not None else None
        # Accumulate snapshots for current checkpoint (will be saved at checkpoint time)
        self.pending_snapshots: List[Dict[str, Any]] = []
        self.last_collection_time = -interval
        self.snapshot_count = 0
    
    def collect(self, sim_time: float) -> bool:
        """
        Collect traffic states if interval has elapsed.
        Collects lane/highway/ramp/subway/bus states.
        Snapshots are accumulated and will be saved at checkpoint time.
        
        Args:
            sim_time: Current simulation time in seconds
            
        Returns:
            True if collection was performed, False otherwise
        """
        if hasattr(self.env, "is_traci_healthy") and not self.env.is_traci_healthy():
            return False
        if getattr(self.env, "traci_conn", None) is None:
            return False
        if sim_time - self.last_collection_time >= self.interval:
            try:
                # Collect lane conditions (for intersections)
                traffic_conditions = collect_lane_conditions(
                    self.env.traci_conn,
                    sim_time,
                    lane_dict=self.lane_dict,
                    lane_inter_graph=self.lane_inter_graph,
                    lane_ids=self.lane_ids,
                    system_states=getattr(self.env, "system_states", None)
                )

                # Collect highway conditions
                highway_conditions = collect_highway_conditions(
                    self.env,
                    sim_time
                )
                
                # Collect ramp lane conditions (from ramp_lane_graph)
                ramp_lane_conditions = collect_ramp_lane_conditions(
                    self.env,
                    sim_time
                )
                
                # Collect subway data
                subway_data = collect_subway_conditions(self.env, sim_time)
                
                # Collect bus data
                bus_data = collect_bus_conditions(self.env, sim_time)

                # Store traffic states as a dictionary with separate keys for different state types
                traffic_states = {
                    "lane_states": traffic_conditions,
                    "highway_states": highway_conditions,
                    "ramp_lane_states": ramp_lane_conditions,
                    "subway": subway_data,
                    "bus": bus_data
                }
                
                has_data = (
                    traffic_states.get("lane_states")
                    or traffic_states.get("highway_states")
                    or traffic_states.get("ramp_lane_states")
                    or traffic_states.get("subway")
                    or traffic_states.get("bus")
                )
                
                if has_data:
                    # Create snapshot record (but don't save yet - will save at checkpoint)
                    lane_states = traffic_states.get("lane_states", {})
                    highway_states = traffic_states.get("highway_states", {})
                    ramp_lane_states = traffic_states.get("ramp_lane_states", {})

                    snapshot = {
                        "simulation_time": sim_time,
                        "timestamp": datetime.now().isoformat(),
                        "lane_count": len(lane_states),
                        "highway_count": len(highway_states),
                        "ramp_lane_count": len(ramp_lane_states),
                        "traffic_states": traffic_states
                    }

                    # Accumulate snapshot for checkpoint save
                    self.pending_snapshots.append(snapshot)
                    self.last_collection_time = sim_time
                    return True
            except traci.exceptions.FatalTraCIError as e:
                if hasattr(self.env, "mark_traci_unhealthy"):
                    self.env.mark_traci_unhealthy()
                print(f"Warning: Failed to collect traffic states: {e}")
            except Exception as e:
                print(f"Warning: Failed to collect traffic states: {e}")
                traceback.print_exc()
        return False

    def collect_if_needed(self, sim_time: float) -> bool:
        """
        Backward-compatible alias for collect().

        Some scripts call collect_if_needed; keep behavior identical to collect.
        """
        return self.collect(sim_time)
    
    def save_checkpoint_snapshots(self, checkpoint_time: float) -> bool:
        """
        Save all accumulated snapshots for the current checkpoint.
        This should be called when a checkpoint is reached.
        
        Args:
            checkpoint_time: The simulation time when checkpoint was reached
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pending_snapshots:
            return True  # No snapshots to save
        
        try:
            success = append_checkpoint_snapshots(
                self.traffic_states_filepath,
                self.pending_snapshots,
                checkpoint_time,
                simulation_id=self.simulation_id
            )
            if success:
                self.snapshot_count += len(self.pending_snapshots)
                self.pending_snapshots = []  # Clear accumulated snapshots
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to save checkpoint snapshots: {e}")
            traceback.print_exc()
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "snapshot_count": self.snapshot_count,
            "pending_snapshots": len(self.pending_snapshots),
            "last_collection_time": self.last_collection_time,
            "filepath": str(self.traffic_states_filepath)
        }


def append_checkpoint_snapshots(
    filepath: Path,
    snapshots: List[Dict[str, Any]],
    checkpoint_time: float,
    simulation_id: Optional[str] = None
) -> bool:
    """
    Append all snapshots from a checkpoint to a JSON Lines file.
    All snapshots collected during a checkpoint interval are saved together.
    Data is organized by checkpoint_time as the primary key for efficient checkpoint-based retrieval.
    
    Args:
        filepath: Path to the JSON Lines file
        snapshots: List of snapshot dictionaries, each containing:
            - "simulation_time": float
            - "timestamp": str
            - "lane_count": int
            - "highway_count": int
            - "ramp_count": int
            - "traffic_states": Dict with lane_states, highway_states, ramp_states, subway, bus
        checkpoint_time: The simulation time when checkpoint was reached
        simulation_id: Optional simulation ID to tag this checkpoint record (for filtering during read)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not snapshots:
            return True  # No snapshots to save
        
        # Create a checkpoint record containing all snapshots from this checkpoint
        checkpoint_record = {
            "type": "checkpoint",
            "checkpoint_time": checkpoint_time,  # Primary key for checkpoint-based retrieval
            "timestamp": datetime.now().isoformat(),
            "snapshot_count": len(snapshots),
            "simulation_id": simulation_id,  # Tag checkpoint with simulation_id for filtering
            "snapshots": snapshots  # List of all snapshots collected during this checkpoint interval
        }
        
        # Append as a single JSON line (streaming write)
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(checkpoint_record, f, ensure_ascii=False)
            f.write('\n')
            f.flush()  # Ensure data is written immediately
        
        return True
        
    except Exception as e:
        print(f"Error appending checkpoint snapshots: {e}")
        traceback.print_exc()
        return False


def _read_traffic_states_file(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal helper function to read traffic states files and return raw snapshots.
    When simulation_id is provided, reads only from the matching file (single experiment).
    Supports checkpoint format (checkpoint records with multiple snapshots).
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_signal_timing")
                    If provided, searches for files matching "{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
                    If None, searches for files matching "*_traffic_states_{simulation_id}_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required. Directly locates and reads from the matching file (single experiment).
        
    Returns:
        Dictionary containing metadata and raw snapshots
    """
    result = {
        "metadata": None,
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": 0,
        "filtered_snapshots": 0
    }
    
    try:
        # Find all matching traffic states files
        traffic_states_dir = _ensure_traffic_states_dir()
        
        # If simulation_id is provided, directly locate the matching file(s)
        # File format: "{file_prefix}_traffic_states_{simulation_id}_{timestamp}.jsonl"
        if simulation_id:
            # Search for files containing the simulation_id
            if file_prefix:
                # More precise pattern: "{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
                pattern = f"{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
            else:
                # Search for any file containing this simulation_id
                pattern = f"*_traffic_states_{simulation_id}_*.jsonl"
            
            jsonl_files = list(traffic_states_dir.glob(pattern))
        else:
            # No simulation_id provided - require it for single experiment reading
            raise ValueError("simulation_id is required for reading traffic states. "
                           "Each experiment's snapshots are stored in a single file identified by simulation_id.")
        
        if not jsonl_files:
            error_msg = f"No traffic states files found in {traffic_states_dir}"
            if simulation_id:
                error_msg += f" with simulation_id '{simulation_id}'"
            if file_prefix:
                error_msg += f" with prefix '{file_prefix}'"
            raise FileNotFoundError(error_msg)
        
        # Sort files by modification time (newest first) for consistent processing
        jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Collect all snapshots from matching files
        all_snapshots = []
        metadata_collected = False
        
        # Process each file
        for filepath in jsonl_files:
            if not filepath.exists():
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # Handle metadata (first line of file)
                            if data.get("type") == "metadata":
                                # If simulation_id was provided, verify it matches
                                metadata_simulation_id = data.get("simulation_id")
                                if simulation_id is not None and metadata_simulation_id != simulation_id:
                                    # Skip this file if simulation_id doesn't match
                                    break  # Break from reading this file
                                
                                if not metadata_collected:
                                    result["metadata"] = data
                                    metadata_collected = True
                                continue
                            
                            # Handle new checkpoint format (checkpoint records with multiple snapshots)
                            if data.get("type") == "checkpoint":
                                # If simulation_id was provided, verify it matches
                                checkpoint_simulation_id = data.get("simulation_id")
                                if simulation_id is not None and checkpoint_simulation_id != simulation_id:
                                    continue  # Skip checkpoint records with different simulation_id
                                
                                checkpoint_snapshots = data.get("snapshots", [])
                                
                                # Process each snapshot in the checkpoint
                                for snapshot in checkpoint_snapshots:
                                    sim_time = snapshot.get("simulation_time")
                                    if sim_time is None:
                                        continue
                                    
                                    # Exact time match (takes precedence) - return immediately
                                    if exact_time is not None:
                                        if abs(sim_time - exact_time) < 0.1:
                                            result["snapshots"] = [snapshot]
                                            result["snapshots_by_time"] = {sim_time: snapshot}
                                            result["total_snapshots"] = 1
                                            result["filtered_snapshots"] = 1
                                            return result
                                    
                                    # Collect snapshot for later filtering and sorting
                                    all_snapshots.append(snapshot)
                                continue
                            
                            # Skip unknown record types
                            continue
                                
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_num + 1} in {filepath}: {e}")
                            continue
            except Exception as e:
                print(f"Warning: Error reading file {filepath}: {e}")
                continue
        
        # If exact_time was specified but not found, return empty result
        if exact_time is not None:
            result["total_snapshots"] = 0
            result["filtered_snapshots"] = 0
            return result
        
        # Sort all snapshots by simulation_time
        all_snapshots.sort(key=lambda s: s.get("simulation_time", 0))
        
        # Apply time range filter and max_snapshots limit
        snapshot_count = 0
        filtered_count = 0
        
        for snapshot in all_snapshots:
            sim_time = snapshot.get("simulation_time")
            if sim_time is None:
                continue
            
            # Apply time range filter
            if start_time is not None and sim_time < start_time:
                continue
            if end_time is not None and sim_time > end_time:
                break  # Snapshots are sorted, so we can break here
            
            filtered_count += 1
            
            # Apply max_snapshots limit
            if max_snapshots is None or snapshot_count < max_snapshots:
                result["snapshots"].append(snapshot)
                result["snapshots_by_time"][sim_time] = snapshot
                snapshot_count += 1
            else:
                # If we've reached max_snapshots and have end_time, we can break
                if end_time is not None:
                    break
        
        result["total_snapshots"] = snapshot_count
        result["filtered_snapshots"] = filtered_count
        return result
        
    except Exception as e:
        return {"error": f"Error reading files: {str(e)}"}


def read_lane_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read lane traffic states from JSON Lines files organized by simulation_time.
    This function reads files line by line to avoid loading everything into memory.
    Supports time-based retrieval: exact time match, time range, or all snapshots.
    
    When simulation_id is provided, reads only from the matching file (single experiment).
    All snapshots from the same experiment are stored in one file, so no cross-file reading is needed.
    Raises FileNotFoundError if no traffic states files are found.
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_signal_timing")
                    If provided, searches for files matching "{file_prefix}_traffic_states_*.jsonl"
                    If None, searches all files matching "*_traffic_states_*.jsonl" or "traffic_states_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required for single experiment reading. Directly locates and reads from the matching file.
        
    Returns:
        Dictionary containing metadata and lane traffic states organized by simulation_time
        Format: {
            "metadata": {...},
            "snapshots": [
                {
                    "simulation_time": float,
                    "timestamp": str,
                    "lane_count": int,
                    "lane_states": {lane_id: condition_dict, ...}
                },
                ...
            ],
            "snapshots_by_time": {sim_time: snapshot_dict, ...},
            "total_snapshots": int,
            "filtered_snapshots": int
        }
        
    Raises:
        FileNotFoundError: If no traffic states files are found in records/traffic_states directory
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)
    
    if "error" in raw_result:
        return raw_result
    
    # Extract lane_states from each snapshot
    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }
    
    for snapshot in raw_result["snapshots"]:
        # Extract lane_states from traffic_states
        traffic_states = snapshot.get("traffic_states", {})
        lane_states = traffic_states.get("lane_states", {})
        
        # Create lane-focused snapshot
        lane_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "lane_count": len(lane_states),
            "lane_states": lane_states
        }
        
        result["snapshots"].append(lane_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = lane_snapshot
    
    return result


def read_highway_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read highway traffic states from JSON Lines files organized by simulation_time.
    This function reads files line by line to avoid loading everything into memory.
    Supports time-based retrieval: exact time match, time range, or all snapshots.
    
    When simulation_id is provided, reads only from the matching file (single experiment).
    All snapshots from the same experiment are stored in one file, so no cross-file reading is needed.
    Raises FileNotFoundError if no traffic states files are found.
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_signal_timing")
                    If provided, searches for files matching "{file_prefix}_traffic_states_*.jsonl"
                    If None, searches all files matching "*_traffic_states_*.jsonl" or "traffic_states_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required for single experiment reading. Directly locates and reads from the matching file.
        
    Returns:
        Dictionary containing metadata and highway traffic states organized by simulation_time
        Format: {
            "metadata": {...},
            "snapshots": [
                {
                    "simulation_time": float,
                    "timestamp": str,
                    "highway_count": int,
                    "highway_states": {highway_id: condition_dict, ...}
                },
                ...
            ],
            "snapshots_by_time": {sim_time: snapshot_dict, ...},
            "total_snapshots": int,
            "filtered_snapshots": int
        }
        
    Raises:
        FileNotFoundError: If no traffic states files are found in records/traffic_states directory
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)
    
    if "error" in raw_result:
        return raw_result
    
    # Extract highway_states from each snapshot
    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }
    
    for snapshot in raw_result["snapshots"]:
        # Extract highway_states from traffic_states
        traffic_states = snapshot.get("traffic_states", {})
        highway_states = traffic_states.get("highway_states", {})
        
        # Create highway-focused snapshot
        highway_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "highway_count": len(highway_states),
            "highway_states": highway_states
        }
        
        result["snapshots"].append(highway_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = highway_snapshot
    
    return result


def read_ramp_lane_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read ramp lane traffic states from JSON Lines files organized by simulation_time.
    This function reads files line by line to avoid loading everything into memory.
    Supports time-based retrieval: exact time match, time range, or all snapshots.
    
    Reads lane-level data from ramp_lane_graph (controlled_lane, upstream_lane, downstream_lane nodes).
    When simulation_id is provided, reads only from the matching file (single experiment).
    All snapshots from the same experiment are stored in one file, so no cross-file reading is needed.
    Raises FileNotFoundError if no traffic states files are found.
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_ramp_metering")
                    If provided, searches for files matching "{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
                    If None, searches all files matching "*_traffic_states_{simulation_id}_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required for single experiment reading. Directly locates and reads from the matching file.
        
    Returns:
        Dictionary containing metadata and ramp lane traffic states organized by simulation_time
        Format: {
            "metadata": {...},
            "snapshots": [
                {
                    "simulation_time": float,
                    "timestamp": str,
                    "ramp_lane_count": int,
                    "ramp_lane_states": {lane_id: lane_condition_dict, ...}
                },
                ...
            ],
            "snapshots_by_time": {sim_time: snapshot_dict, ...},
            "total_snapshots": int,
            "filtered_snapshots": int
        }
        Each lane_condition_dict contains:
            - lane_id: str
            - node_type: 'controlled_lane', 'upstream_lane', or 'downstream_lane'
            - associated_ramp_id: str
            - simulation_time: float
            - lane-level traffic metrics (occupancy, density, speed, queue_length, etc.)
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)
    
    if "error" in raw_result:
        return raw_result
    
    # Extract ramp_lane_states from each snapshot
    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }
    
    for snapshot in raw_result["snapshots"]:
        # Extract ramp_lane_states from traffic_states
        traffic_states = snapshot.get("traffic_states", {})
        ramp_lane_states = traffic_states.get("ramp_lane_states", {})
        
        # Create ramp lane-focused snapshot
        ramp_lane_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "ramp_lane_count": len(ramp_lane_states),
            "ramp_lane_states": ramp_lane_states
        }
        
        result["snapshots"].append(ramp_lane_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = ramp_lane_snapshot
    
    return result


def read_subway_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    line_id: Optional[str] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read subway traffic states from JSON Lines files organized by simulation_time.
    This function reads files line by line to avoid loading everything into memory.
    Supports time-based retrieval: exact time match, time range, or all snapshots.
    
    When simulation_id is provided, reads only from the matching file (single experiment).
    All snapshots from the same experiment are stored in one file, so no cross-file reading is needed.
    Raises FileNotFoundError if no traffic states files are found.
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        line_id: Optional line ID to filter results (e.g., "route_line1_eastbound")
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_subway_scheduling")
                    If provided, searches for files matching "{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
                    If None, searches all files matching "*_traffic_states_{simulation_id}_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required for single experiment reading. Directly locates and reads from the matching file.
        
    Returns:
        Dictionary containing metadata and subway traffic states organized by simulation_time
        Format: {
            "metadata": {...},
            "snapshots": [
                {
                    "simulation_time": float,
                    "timestamp": str,
                    "subway_line_count": int,
                    "subway": {
                        "lines": {
                            "line_id": {
                                "active_trains": int,
                                "headway": float,
                                "trains": {...},
                                "stations": {...}
                            }
                        }
                    }
                },
                ...
            ],
            "snapshots_by_time": {sim_time: snapshot_dict, ...},
            "total_snapshots": int,
            "filtered_snapshots": int
        }
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)
    
    if "error" in raw_result:
        return raw_result
    
    # Extract subway data from each snapshot
    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }
    
    for snapshot in raw_result["snapshots"]:
        # Extract subway from traffic_states
        traffic_states = snapshot.get("traffic_states", {})
        subway_data = traffic_states.get("subway")
        
        # Filter by line_id if specified
        if line_id and subway_data and "lines" in subway_data:
            filtered_lines = {line_id: subway_data["lines"][line_id]} if line_id in subway_data["lines"] else {}
            subway_data = {"lines": filtered_lines} if filtered_lines else None
        
        # Create subway-focused snapshot
        subway_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "subway_line_count": len(subway_data.get("lines", {})) if subway_data else 0,
            "subway": subway_data
        }
        
        result["snapshots"].append(subway_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = subway_snapshot
    
    return result


def read_bus_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    line_id: Optional[str] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read bus traffic states from JSON Lines files organized by simulation_time.
    This function reads files line by line to avoid loading everything into memory.
    Supports time-based retrieval: exact time match, time range, or all snapshots.
    
    When simulation_id is provided, reads only from the matching file (single experiment).
    All snapshots from the same experiment are stored in one file, so no cross-file reading is needed.
    Raises FileNotFoundError if no traffic states files are found.
    
    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        line_id: Optional line ID to filter results (e.g., "route_bus1")
        file_prefix: Optional file prefix to filter files (e.g., "jinan_deepseek-v3.2_bus_scheduling")
                    If provided, searches for files matching "{file_prefix}_traffic_states_{simulation_id}_*.jsonl"
                    If None, searches all files matching "*_traffic_states_{simulation_id}_*.jsonl"
        simulation_id: Simulation ID to filter files (only reads from file containing this simulation_id)
                       Required for single experiment reading. Directly locates and reads from the matching file.
        
    Returns:
        Dictionary containing metadata and bus traffic states organized by simulation_time
        Format: {
            "metadata": {...},
            "snapshots": [
                {
                    "simulation_time": float,
                    "timestamp": str,
                    "bus_line_count": int,
                    "bus": {
                        "lines": {
                            "line_id": {
                                "active_buses": int,
                                "headway": float,
                                "buses": {...},
                                "stations": {...}
                            }
                        }
                    }
                },
                ...
            ],
            "snapshots_by_time": {sim_time: snapshot_dict, ...},
            "total_snapshots": int,
            "filtered_snapshots": int
        }
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)
    
    if "error" in raw_result:
        return raw_result
    
    # Extract bus data from each snapshot
    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }
    
    for snapshot in raw_result["snapshots"]:
        # Extract bus from traffic_states
        traffic_states = snapshot.get("traffic_states", {})
        bus_data = traffic_states.get("bus")
        
        # Filter by line_id if specified
        if line_id and bus_data and "lines" in bus_data:
            filtered_lines = {line_id: bus_data["lines"][line_id]} if line_id in bus_data["lines"] else {}
            bus_data = {"lines": filtered_lines} if filtered_lines else None
        
        # Create bus-focused snapshot
        bus_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "bus_line_count": len(bus_data.get("lines", {})) if bus_data else 0,
            "bus": bus_data
        }
        
        result["snapshots"].append(bus_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = bus_snapshot

    return result


def read_taxi_traffic_states(
    max_snapshots: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exact_time: Optional[float] = None,
    file_prefix: Optional[str] = None,
    simulation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read taxi traffic states from JSON Lines files organized by simulation_time.

    Args:
        max_snapshots: Maximum number of snapshots to read (None for all)
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        file_prefix: Optional file prefix to filter files
        simulation_id: Simulation ID to filter files

    Returns:
        Dictionary containing metadata and taxi traffic states organized by simulation_time
    """
    raw_result = _read_traffic_states_file(max_snapshots, start_time, end_time, exact_time, file_prefix, simulation_id)

    if "error" in raw_result:
        return raw_result

    result = {
        "metadata": raw_result["metadata"],
        "snapshots": [],
        "snapshots_by_time": {},
        "total_snapshots": raw_result["total_snapshots"],
        "filtered_snapshots": raw_result["filtered_snapshots"]
    }

    for snapshot in raw_result["snapshots"]:
        traffic_states = snapshot.get("traffic_states", {})
        taxi_data = traffic_states.get("taxi")

        taxi_snapshot = {
            "simulation_time": snapshot.get("simulation_time"),
            "timestamp": snapshot.get("timestamp"),
            "fleet_size": taxi_data.get("fleet_size", 0) if taxi_data else 0,
            "idle_count": taxi_data.get("idle_count", 0) if taxi_data else 0,
            "pickup_count": taxi_data.get("pickup_count", 0) if taxi_data else 0,
            "occupied_count": taxi_data.get("occupied_count", 0) if taxi_data else 0,
            "pending_reservations": taxi_data.get("pending_reservations", 0) if taxi_data else 0,
            "utilization_rate": taxi_data.get("utilization_rate", 0.0) if taxi_data else 0.0
        }

        result["snapshots"].append(taxi_snapshot)
        sim_time = snapshot.get("simulation_time")
        if sim_time is not None:
            result["snapshots_by_time"][sim_time] = taxi_snapshot

    return result
