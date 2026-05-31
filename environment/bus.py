"""
Bus infrastructure classes for the SUMO simulation environment.
Provides BusStation and BusLine classes for managing bus operations.
"""

from typing import Dict, List, Optional
import traci


class BusStation:
    """
    Represents a bus station in the simulation environment.
    Encapsulates TraCI calls for station-specific operations.
    """
    def __init__(self, station_id: str, traci_conn):
        self.station_id = station_id
        self.traci_conn = traci_conn
        self.lane_id = traci_conn.busstop.getLaneID(station_id)
        self.start_pos = traci_conn.busstop.getStartPos(station_id)
    
    def get_waiting_count(self) -> int:
        """Returns the number of passengers waiting at this station."""
        try:
            return self.traci_conn.busstop.getPersonCount(self.station_id)
        except:
            return 0

    def get_state(self) -> dict:
        return {
            "station_id": self.station_id,
            "waiting_count": self.get_waiting_count()
        }


class BusLine:
    """
    Represents a bus line (route) as fixed infrastructure.
    Encapsulates TraCI calls for line-wide operations like dispatching.
    """
    def __init__(self, route_id: str, traci_conn, vehicle_type: str = "bus", stations: list = None, env = None):
        self.route_id = route_id
        self.traci_conn = traci_conn
        self.vehicle_type = vehicle_type
        self.edges = traci_conn.route.getEdges(route_id)
        
        # Initialize bus_count to maintain continuous vehicle IDs
        # Priority: 1) checkpoint metadata, 2) existing vehicles, 3) default 0
        self.bus_count = 0
        
        # Try to read from checkpoint metadata first (for continuous IDs across checkpoints)
        if env and hasattr(env, 'checkpoint_vehicle_counts') and env.checkpoint_vehicle_counts:
            if route_id in env.checkpoint_vehicle_counts:
                self.bus_count = env.checkpoint_vehicle_counts[route_id]
        else:
            # Fallback: check existing vehicles (prevents duplicate IDs in current session)
            existing_buses = 0
            try:
                for veh_id in traci_conn.vehicle.getIDList():
                    if traci_conn.vehicle.getRouteID(veh_id) == route_id:
                        # Extract the count from vehicle ID format: route_id.count
                        if '.' in veh_id:
                            try:
                                count = int(veh_id.split('.')[-1])
                                existing_buses = max(existing_buses, count + 1)
                            except:
                                pass
            except:
                pass
            self.bus_count = existing_buses
        
        self.stations = stations if stations is not None else []  # List of station IDs on this line
        self.env = env
        
        # History for adaptive normalization
        self.waiting_time_history = []
        self.fuel_history = []
        self.departure_times = []
        
        # Delay rate tracking
        self.vehicle_last_stop = {}  # {vehicle_id: (stop_id, time)}
        self.delay_stats = {
            'on_time_count': 0,
            'late_count': 0,
            'total_delay': 0.0,
            'total_segments': 0
        }

    def dispatch_bus(self, position: float, schedule: list, vehicle_type: str = None) -> bool:
        """
        Executes the dispatch of a bus on this line with a specific schedule.
        Vehicle ID is auto-generated as ctrl_route_id.0, ctrl_route_id.1, etc.
        The 'ctrl_' prefix distinguishes code-dispatched buses from flow-defined buses.

        Args:
            position: Starting position on the first edge
            schedule: List of {"station_id": str, "dwell_time": int}
            vehicle_type: Optional override for vehicle type
        """
        try:
            # When SUMO flow definitions were removed before startup, reuse the
            # original flow ID pattern so explicit person rides that reference
            # bus_M15:0.1-style vehicle IDs can still board code-dispatched buses.
            if self.env is not None and getattr(self.env, "transit_flow_vehicles_filtered", False):
                bus_id = f"{self.route_id}.{self.bus_count}"
            else:
                bus_id = f"ctrl_{self.route_id}.{self.bus_count}"
            
            # Use provided vehicle_type or default to instance's type
            v_type = vehicle_type if vehicle_type else self.vehicle_type
            
            # 1. Add vehicle
            # Remove 'bus_' prefix from route_id for line attribute to match passenger definitions
            line_name = self.route_id.replace("bus_", "") if self.route_id.startswith("bus_") else self.route_id
            
            self.traci_conn.vehicle.add(
                vehID=bus_id,
                routeID=self.route_id,
                typeID=v_type,
                depart="now",
                line=line_name
            )
            
            # 2. Move to position
            # Note: Unlike subway (single-lane tracks), bus routes often have multi-lane roads.
            # Lane _0 is often pedestrian-only, lane _1 is for vehicles.
            # We try _1 first, fallback to letting SUMO auto-place if it fails.
            # Only move after the vehicle is inserted into the simulation.
            # Moving too early can trigger a fatal TraCI error.
            should_move = False
            try:
                should_move = bus_id in self.traci_conn.vehicle.getIDList()
            except Exception:
                should_move = False

            if should_move and self.edges:
                edge_id = self.edges[0]
                lane_count = 0
                try:
                    lane_count = self.traci_conn.edge.getLaneNumber(edge_id)
                except Exception:
                    lane_count = 0

                lane_candidates = []
                if lane_count >= 2:
                    lane_candidates.append(1)
                if lane_count >= 1:
                    lane_candidates.append(0)

                for lane_idx in lane_candidates:
                    lane_id = f"{edge_id}_{lane_idx}"
                    try:
                        self.traci_conn.vehicle.moveTo(bus_id, lane_id, position)
                        break
                    except Exception:
                        continue
            
            # 3. Set stops
            for stop in schedule:
                try:
                    self.traci_conn.vehicle.setBusStop(
                        vehID=bus_id,
                        stopID=stop["station_id"],
                        duration=stop["dwell_time"]
                    )
                except:
                    continue
            
            self.bus_count += 1
            try:
                depart_time = self.traci_conn.simulation.getTime()
                self.departure_times.append(float(depart_time))
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[BusLine] Error dispatching {bus_id}: {e}")
            return False

    def get_bus_ids(self) -> list:
        """Returns the IDs of all buses currently on this line.

        Uses cached vehicle list to reduce TraCI calls and avoid race conditions
        where vehicles leave between getIDList and subsequent queries.
        """
        bus_ids = []
        try:
            # Get all vehicles currently in the simulation (single TraCI call)
            all_vehicle_ids = set(self.traci_conn.vehicle.getIDList())

            # Filter by route - only query vehicles that exist
            for veh_id in all_vehicle_ids:
                try:
                    if self.traci_conn.vehicle.getRouteID(veh_id) == self.route_id:
                        bus_ids.append(veh_id)
                except traci.TraCIException:
                    # Vehicle may have left the network between getIDList and getRouteID
                    continue
        except traci.TraCIException:
            # TraCI connection may be closed
            pass
        except Exception:
            pass
        return bus_ids
    
    def get_load_ratios(self) -> list:
        """Returns the load ratios of all buses currently on this line.

        Uses the bus IDs from get_bus_ids() to avoid redundant TraCI calls
        and race conditions.
        """
        load_ratios = []
        try:
            # Reuse get_bus_ids to get validated vehicle list
            bus_ids = self.get_bus_ids()

            for veh_id in bus_ids:
                try:
                    passenger_count = self.traci_conn.vehicle.getPersonNumber(veh_id)
                    capacity = self.traci_conn.vehicletype.getPersonCapacity(self.traci_conn.vehicle.getTypeID(veh_id))
                    if capacity > 0:
                        load_ratios.append(passenger_count / capacity)
                except traci.TraCIException:
                    # Vehicle may have left the network
                    continue
        except traci.TraCIException:
            # TraCI connection may be closed
            pass
        except Exception:
            pass
        return load_ratios
    
    def get_state(self) -> dict:
        """
        Get current state of the bus line.
        Returns basic state information for traffic state collection.
        """
        bus_ids = self.get_bus_ids()
        load_ratios = self.get_load_ratios()
        
        return {
            "active_buses": len(bus_ids),
            "bus_ids": bus_ids,
            "load_ratios": load_ratios,
            "station_count": len(self.stations),
            "stations": [s.get('station_id', s) if isinstance(s, dict) else s for s in self.stations]
        }
    
    def get_reward(self, dic_reward_info: dict) -> float:
        """
        Calculate reward for bus line based on multiple metrics.
        All metrics are normalized to [0, 1] range.
        
        Args:
            dic_reward_info: Dictionary containing reward weights and parameters:
                - waiting_time_weight: Weight for waiting time (default 0.4)
                - load_ratio_weight: Weight for load ratio (default 0.3)
                - operation_cost_weight: Weight for operation cost (default 0.2)
                - headway_stability_weight: Weight for headway stability (default 0.1)
                - ideal_load_ratio: Target load ratio (default 0.70, lower than subway)
                - use_adaptive_normalization: Use adaptive thresholds (default True)
                - max_waiting_time: Max waiting time threshold (default 300.0)
                - max_fuel: Max fuel consumption threshold in ml (default 5000.0)
                - target_headway: Target headway in seconds (default 180.0)
                
        Returns:
            Reward value (higher is better)
        """
        import numpy as np
        
        # Extract weights
        waiting_time_weight = dic_reward_info.get("waiting_time_weight", 0.4)
        load_ratio_weight = dic_reward_info.get("load_ratio_weight", 0.3)
        operation_cost_weight = dic_reward_info.get("operation_cost_weight", 0.2)
        headway_stability_weight = dic_reward_info.get("headway_stability_weight", 0.1)
        
        use_adaptive = dic_reward_info.get("use_adaptive_normalization", True)
        
        # 1. Waiting time score (lower waiting time = higher reward)
        waiting_time_score = 0.0
        if waiting_time_weight > 0:
            waiting_times = self._get_station_waiting_times()
            if waiting_times:
                avg_waiting_time = np.mean(waiting_times)
                
                # Adaptive normalization
                if use_adaptive:
                    self._update_waiting_time_stats(avg_waiting_time)
                    max_waiting_time = self._get_adaptive_max_waiting_time(
                        default=dic_reward_info.get("max_waiting_time", 300.0)
                    )
                else:
                    max_waiting_time = dic_reward_info.get("max_waiting_time", 300.0)
                
                # Normalize to [0, 1]
                normalized_wait = min(avg_waiting_time / max_waiting_time, 1.0) if max_waiting_time > 0 else 0.0
                # Convert to reward: 1 - normalized (lower waiting = higher reward)
                waiting_time_score = (1.0 - normalized_wait) * waiting_time_weight
        
        # 2. Load ratio score (closer to ideal = higher reward)
        load_ratio_score = 0.0
        if load_ratio_weight > 0:
            load_ratios = self.get_load_ratios()
            if load_ratios:
                avg_load_ratio = np.mean(load_ratios)
                ideal_load = dic_reward_info.get("ideal_load_ratio", 0.70)  # Lower than subway
                
                # Calculate deviation (similar to Highway speed_limit_deviation)
                if ideal_load > 0:
                    deviation = abs(avg_load_ratio - ideal_load) / ideal_load
                    normalized_deviation = min(deviation, 1.0)
                    # Convert to reward: 1 - deviation
                    load_ratio_score = (1.0 - normalized_deviation) * load_ratio_weight
        
        # 3. Operation cost score (lower fuel consumption = higher reward)
        operation_cost_score = 0.0
        if operation_cost_weight > 0:
            fuel = self._get_fuel_consumption()
            
            if fuel is not None:
                # Adaptive normalization
                if use_adaptive:
                    self._update_fuel_stats(fuel)
                    max_fuel = self._get_adaptive_max_fuel(
                        default=dic_reward_info.get("max_fuel", 5000.0)
                    )
                else:
                    max_fuel = dic_reward_info.get("max_fuel", 5000.0)
                
                # Normalize to [0, 1]
                normalized_cost = min(fuel / max_fuel, 1.0) if max_fuel > 0 else 0.0
                # Convert to reward: 1 - normalized (lower fuel = higher reward)
                operation_cost_score = (1.0 - normalized_cost) * operation_cost_weight
        
        # 4. Headway stability score (more stable = higher reward)
        headway_stability_score = 0.0
        if headway_stability_weight > 0:
            if len(self.departure_times) > 1:
                actual_headways = np.diff(self.departure_times)
                headway_std = np.std(actual_headways)
                target_headway = dic_reward_info.get("target_headway", 180.0)
                
                # Normalize: std relative to target headway
                if target_headway > 0:
                    normalized_std = min(headway_std / target_headway, 1.0)
                    # Convert to reward: 1 - normalized (lower std = higher reward)
                    headway_stability_score = (1.0 - normalized_std) * headway_stability_weight
        
        # Total reward (higher is better)
        total_reward = (
            waiting_time_score +
            load_ratio_score +
            operation_cost_score +
            headway_stability_score
        )
        
        return total_reward
    
    def _get_station_waiting_times(self) -> list:
        """Get average waiting times at all stations."""
        import numpy as np
        waiting_times = []
        
        if not self.env or not hasattr(self.env, 'bus_stations'):
            return waiting_times
        
        for station_id in self.stations:
            if station_id in self.env.bus_stations:
                if hasattr(self.env, 'waiting_passenger_list'):
                    try:
                        person_ids = self.traci_conn.busstop.getPersonIDs(station_id)
                        station_waits = [
                            self.env.waiting_passenger_list[p_id]
                            for p_id in person_ids
                            if p_id in self.env.waiting_passenger_list
                        ]
                        if station_waits:
                            waiting_times.append(np.mean(station_waits))
                    except:
                        pass
        
        return waiting_times
    
    def _update_waiting_time_stats(self, waiting_time: float):
        """Update waiting time history for adaptive normalization."""
        self.waiting_time_history.append(waiting_time)
        # Keep last 100 data points
        if len(self.waiting_time_history) > 100:
            self.waiting_time_history.pop(0)
    
    def _get_adaptive_max_waiting_time(self, default: float = 300.0) -> float:
        """Get adaptive max waiting time threshold using historical maximum."""
        if len(self.waiting_time_history) < 10:
            return default
        return max(self.waiting_time_history)
    
    def _get_fuel_consumption(self) -> float:
        """Get total fuel consumption for all buses on this line."""
        total_fuel = 0.0
        try:
            for veh_id in self.get_bus_ids():
                try:
                    # Get fuel consumption from TraCI
                    fuel = self.traci_conn.vehicle.getFuelConsumption(veh_id)
                    total_fuel += abs(fuel)
                except traci.TraCIException:
                    # Vehicle may have left the network
                    continue
        except:
            return None
        return total_fuel
    
    def _update_fuel_stats(self, fuel: float):
        """Update fuel consumption history for adaptive normalization."""
        self.fuel_history.append(fuel)
        if len(self.fuel_history) > 100:
            self.fuel_history.pop(0)
    
    def _get_adaptive_max_fuel(self, default: float = 5000.0) -> float:
        """Get adaptive max fuel threshold using historical maximum."""
        if len(self.fuel_history) < 10:
            return default
        return max(self.fuel_history)
    
    def check_travel_times(
        self,
        travel_times_cache: dict,
        tolerance: float = 10.0,
        vehicle_ids: Optional[list] = None
    ):
        """
        Check vehicle arrivals at stations and calculate delay rate.
        Uses SUMO's recorded arrival times from StopData to detect completed stops.
        
        Args:
            travel_times_cache: Dictionary of scheduled travel times {segment_key: time}
            tolerance: Tolerance threshold in seconds (-tolerance <= delay <= tolerance is considered on-time)
        """
        if vehicle_ids is None:
            try:
                vehicle_ids = self.traci_conn.vehicle.getIDList()
            except:
                vehicle_ids = []

        for vehicle_id in vehicle_ids:
            # Check if vehicle belongs to this line
            # Support both flow-defined (bus_X:N.M) and code-dispatched (ctrl_bus_X:N.M) buses
            if not (vehicle_id.startswith(self.route_id) or vehicle_id.startswith(f"ctrl_{self.route_id}")):
                continue
            
            try:
                # Get current stops
                stops = self.traci_conn.vehicle.getStops(vehicle_id)
                if not stops:
                    continue
                
                current_stop = stops[0].stoppingPlaceID
                current_arrival = stops[0].arrival
                
                # Only process if vehicle has arrived at this stop (arrival > 0)
                if current_arrival > 0:
                    if vehicle_id in self.vehicle_last_stop:
                        prev_stop, prev_arrival = self.vehicle_last_stop[vehicle_id]
                        
                        # Check if this is a new stop
                        if current_stop != prev_stop:
                            # Verify that prev_stop and current_stop are adjacent
                            is_adjacent = self._are_stops_adjacent(prev_stop, current_stop)
                            
                            if is_adjacent:
                                # Calculate actual travel time using SUMO's recorded arrival times
                                actual_travel_time = current_arrival - prev_arrival
                                
                                # Get scheduled travel time from cache
                                segment_key = f"{prev_stop}|{current_stop}"
                                scheduled_travel_time = travel_times_cache.get(segment_key)
                                
                                if scheduled_travel_time is not None:
                                    # Calculate delay
                                    delay = actual_travel_time - scheduled_travel_time
                                    
                                    # Update statistics
                                    self.delay_stats['total_segments'] += 1
                                    self.delay_stats['total_delay'] += delay
                                    
                                    # Check if within tolerance range (both early and late)
                                    if abs(delay) <= tolerance:
                                        self.delay_stats['on_time_count'] += 1
                                    else:
                                        self.delay_stats['late_count'] += 1
                            else:
                                # Non-adjacent stops, treat as on-time (skip delay calculation)
                                self.delay_stats['total_segments'] += 1
                                self.delay_stats['on_time_count'] += 1
                            
                            # Update last stop
                            self.vehicle_last_stop[vehicle_id] = (current_stop, current_arrival)
                    else:
                        # First stop - just record
                        self.vehicle_last_stop[vehicle_id] = (current_stop, current_arrival)
            except:
                pass
    
    def _are_stops_adjacent(self, stop1: str, stop2: str) -> bool:
        """
        Check if two stops are adjacent in the station list.
        
        Args:
            stop1: First stop ID
            stop2: Second stop ID
            
        Returns:
            True if stops are adjacent, False otherwise
        """
        if not self.stations or len(self.stations) < 2:
            return True  # If no station list, assume adjacent
        
        try:
            # Extract station_id from stations list (handle both dict and string formats)
            station_ids = []
            for station in self.stations:
                if isinstance(station, dict):
                    station_ids.append(station.get('station_id', ''))
                else:
                    station_ids.append(str(station))
            
            # Find indices
            if stop1 in station_ids and stop2 in station_ids:
                idx1 = station_ids.index(stop1)
                idx2 = station_ids.index(stop2)
                return abs(idx2 - idx1) == 1
            else:
                return True  # If stops not in list, assume adjacent (fallback)
        except:
            return True  # On error, assume adjacent
    
    def calculate_delay_rate(self, tolerance: float = 10.0) -> dict:
        """
        Calculate delay rate based on accumulated statistics.
        
        Args:
            tolerance: Tolerance threshold in seconds (delay <= tolerance is considered on-time)
        
        Returns:
            {
                'on_time_rate': float,
                'delay_rate': float,
                'avg_delay': float,
                'total_segments': int
            }
        """
        total = self.delay_stats['total_segments']
        
        if total == 0:
            return {
                'on_time_rate': 1.0,
                'delay_rate': 0.0,
                'avg_delay': 0.0,
                'total_segments': 0
            }
        
        on_time_count = self.delay_stats['on_time_count']
        avg_delay = self.delay_stats['total_delay'] / total
        
        return {
            'on_time_rate': on_time_count / total,
            'delay_rate': 1.0 - (on_time_count / total),
            'avg_delay': avg_delay,
            'total_segments': total
        }
