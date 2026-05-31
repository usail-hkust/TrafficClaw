"""
Taxi scheduling control module.
Implements LLM-controlled taxi dispatch and repositioning for taxi fleet management.

Uses a simulated taxi system built on ordinary TraCI vehicle control:
1. Dispatch taxis to waiting passengers (simulated reservations)
2. Reposition idle taxis to high-demand areas

Key TraCI APIs used:
- traci.vehicle.setSpeed(vehID, speed) - Freeze (0) / release (-1) taxi movement
- traci.vehicle.setRoute(vehID, edges) - Set taxi route for pickup/dropoff/idle
- traci.vehicle.changeTarget(vehID, edgeID) - Reposition idle taxi
- traci.simulation.findRoute(fromEdge, toEdge) - Compute routes for dispatch
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import copy
import traceback
import traci
import xml.etree.ElementTree as ET
import os
import random
from .base import ControlModule


# ============================================================================
# Simulated Taxi System Data Structures
# These replace SUMO's native taxi API with ordinary vehicle control
# ============================================================================

class SimulatedTaxiState:
    """Taxi state constants for simulated taxi system."""
    IDLE = 0        # Empty and available for dispatch
    PICKUP = 1      # En route to pick up passenger
    OCCUPIED = 2    # Carrying passenger to destination


@dataclass
class SimulatedReservation:
    """
    Simulated reservation data structure.
    Replaces SUMO's native Reservation object.
    """
    id: str                              # Unique reservation ID
    person_ids: List[str]                # Associated person IDs (from XML)
    from_edge: str                       # Pickup edge
    to_edge: str                         # Dropoff edge
    depart_time: float                   # Time when reservation becomes active
    reservation_time: float              # Same as depart_time for compatibility
    state: int = 1                       # State bitmask: 1=new, 4=assigned, 8=picked_up, 16=completed
    assigned_taxi: Optional[str] = None  # Assigned taxi ID
    pickup_time: Optional[float] = None  # Time when passenger was picked up
    dropoff_time: Optional[float] = None # Time when passenger was dropped off


@dataclass
class SimulatedTaxi:
    """
    Simulated taxi data structure.
    Tracks taxi state without using SUMO's taxi device.
    """
    id: str                                      # Vehicle ID
    state: int = SimulatedTaxiState.IDLE         # Current state
    current_reservation_id: Optional[str] = None # Currently serving reservation
    route_stage: str = "idle"                    # "idle", "to_pickup", "at_pickup", "to_dropoff"
    pickup_edge: Optional[str] = None            # Target pickup edge
    dropoff_edge: Optional[str] = None           # Target dropoff edge
    spawn_time: float = 0.0                      # Time when taxi was added to simulation
    dropoff_route_set: bool = False              # Whether dropoff route was successfully set
    stage_start_time: float = 0.0                # Time when current stage started


class TaxiSchedulingModule(ControlModule):
    """Control module for LLM-controlled taxi dispatch and repositioning."""
    
    DOMAIN_KNOWLEDGE = """Taxi scheduling control manages taxi fleet dispatch and idle taxi repositioning in urban networks.

- **Decision Types:**
  1. **Dispatch**: Assign waiting passengers (reservations) to available taxis
  2. **Repositioning**: Move idle taxis to high-demand areas to reduce future wait times

- **Taxi States:**
  - State 0 (Idle): Taxi is empty and available for dispatch
  - State 1 (Pickup): Taxi is en route to pick up a passenger
  - State 2 (Occupied): Taxi is carrying a passenger
  - State 3 (Full): Taxi has reached maximum capacity

- **Reservation States:**
  - Reservation.state is commonly a bit mask:
    - 1: new / unassigned
    - 4: assigned / awaiting pickup
    - 8: picked up / on board
  - Treat bit 8 as "picked up" for wait/travel metrics.

- **Configuration Format:**
  - Format: {"dispatch_decisions": [...], "reposition_decisions": [...]}
  - dispatch_decisions: List of {"taxi_id": str, "reservation_ids": [str]}
  - reposition_decisions: List of {"taxi_id": str, "target_edge": str} or {"taxi_id": str, "target_taz": str}

- **Optimization Strategies:**
  - **Dispatch Strategy:**
    - Assign nearest idle taxi to each waiting reservation to minimize pickup time
    - Consider taxi battery/fuel levels if available
    - Prioritize reservations with longer wait times
  
  - **Repositioning Strategy:**
    - Analyze historical demand patterns by zone and time
    - Move idle taxis to zones with expected high demand
    - Avoid clustering all taxis in one area
    - Balance taxi distribution across the network

- **Online Dispatch Policy (Recommended):**
  - Use `dispatch_strategy` set to `"online_eta"`, `"online"`, or `"online_matching_eta"` instead of hand-crafting `dispatch_decisions` for the whole window
  - Tune `online_dispatch_cost_weights` (keys: `eta`, `income`, `recent_orders`) to trade off pickup speed vs fairness
    - `eta`: Weight for estimated time to arrival (pickup speed)
    - `income`: Weight for taxi cumulative income (fairness)
    - `recent_orders`: Weight for recent order count (fairness)
  - Tune `online_dispatch_candidate_k` / `online_dispatch_max_reservations` to control computation workload
  - Fleet fairness signals are available in `taxi_fleet_state['taxi_details'][taxi_id]`: `cumulative_income`, `recent_order_count`

- **Online Reposition Policy (Optional):**
  - Use `reposition_strategy` set to `"auto_taz_balance"`, `"taz_balance"`, or `"auto"` to reposition based on TAZ demand/supply
  - Tune `auto_reposition_*` parameters:
    - `auto_reposition_max_ratio`: Fraction of idle taxis per cycle
    - `auto_reposition_cooldown_seconds`: Per-taxi cooldown period
    - `auto_reposition_critical_only`: Only reposition to TAZs with demand>0 & supply==0
    - `auto_reposition_top_taz`: Number of top TAZs to consider when critical_only=False
  - Tune `reposition_interval_seconds` to control reposition frequency

- **Policy Output Format:**
  - In POLICY_PLANNING, define `taxi_config = current_taxi_config.copy()` and set `config = {"taxi_scheduling": taxi_config}` (do NOT use return)
  - `taxi_config` should include:
    - `dispatch_strategy`: `"online_eta"` | `"llm"` | `"greedy"`
    - `online_dispatch_cost_weights`: `{"eta": float, "income": float, "recent_orders": float}`
    - `reposition_strategy`: `"auto_taz_balance"` | `"llm"`
    - Optional overrides: `dispatch_decisions`, `reposition_decisions` (for special cases only)
  - Complete with FINISH action when satisfied

- **Performance Metrics:**
  - Average passenger wait time (MOST IMPORTANT)
  - Average pickup time (time from dispatch to pickup)
  - Fleet utilization (occupied taxis / total taxis)
  - Service rate (served reservations / total reservations)
  - Empty vehicle miles traveled (lower is better)

- **Time-Aware Optimization:**
  - Morning rush (7:00-9:00): Position taxis in residential areas
  - Daytime (9:00-17:00): Position taxis in commercial/business districts
  - Evening rush (17:00-19:00): Position taxis in commercial areas
  - Night (19:00-7:00): Position taxis near entertainment/airport areas"""
    
    def __init__(self, config_dir_name: Optional[str] = None):
        super().__init__("taxi_scheduling", "taxi_scheduling.json", config_dir_name=config_dir_name)
        # TAZ data structures
        self.taz_edges = {}  # taz_id -> [edge_ids]
        self.edge_to_taz = {}  # edge_id -> taz_id
        self.taz_stats = {}  # taz_id -> {demand: int, matches: int}

        # Simulated taxi system data structures
        self._simulated_taxis: Dict[str, SimulatedTaxi] = {}
        self._simulated_reservations: Dict[str, SimulatedReservation] = {}
        self._pending_reservations_from_file: List[SimulatedReservation] = []
        self._simulated_system_initialized: bool = False
        self._reservation_file_loaded: bool = False
        self._taxi_fleet_file_loaded: bool = False

        # Load TAZ definitions if config_dir_name is provided
        if config_dir_name:
             pass

    # ========================================================================
    # Simulated Taxi System Methods
    # ========================================================================

    def _init_simulated_taxi_system(
        self,
        env: Any,
        config: Dict[str, Any],
        control_state: Dict[str, Any],
        current_time: float
    ) -> None:
        """
        Initialize the simulated taxi system.
        Called once at the start of simulation when use_simulated_taxi_system is enabled.
        """
        if self._simulated_system_initialized:
            return

        print("[SimulatedTaxi] Initializing simulated taxi system...")

        # Load reservations from file
        reservation_file = config.get("reservation_file_path")
        if reservation_file and not self._reservation_file_loaded:
            self._load_reservations_from_file(reservation_file, config)

        # Load or adopt taxi fleet
        taxi_fleet_file = config.get("taxi_fleet_file_path")
        if taxi_fleet_file and not self._taxi_fleet_file_loaded:
            self._load_taxi_fleet_from_file(taxi_fleet_file, env, config, current_time)
        elif config.get("adopt_existing_vehicles", True):
            self._adopt_existing_vehicles(env, config, current_time)

        self._simulated_system_initialized = True

        # Set initial_fleet_size so auto-replenish knows the target
        fleet_count = len(self._simulated_taxis)
        if fleet_count > 0:
            config["initial_fleet_size"] = fleet_count
            config["fleet_size"] = fleet_count
            control_state["initial_fleet_size"] = fleet_count
            print(f"[SimulatedTaxi] Set initial_fleet_size={fleet_count}")

        print(f"[SimulatedTaxi] Initialized: {fleet_count} taxis, "
              f"{len(self._pending_reservations_from_file)} pending reservations, "
              f"module_id={id(self)}")

    def _load_reservations_from_file(self, file_path: str, config: Dict[str, Any]) -> None:
        """
        Load reservation definitions from persons.taxi.xml file.

        Expected format:
        <person id="taxi_p_13" depart="156.38">
            <ride from="660225539#0" to="587061863#2" lines="taxi" />
        </person>
        """
        if not os.path.exists(file_path):
            print(f"[SimulatedTaxi] Warning: Reservation file not found: {file_path}")
            return

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            reservations = []
            for idx, person in enumerate(root.findall('person')):
                person_id = person.get('id', f'person_{idx}')
                depart_str = person.get('depart', '0')
                try:
                    depart_time = float(depart_str)
                except ValueError:
                    depart_time = 0.0

                # Find ride element
                ride = person.find('ride')
                if ride is None:
                    continue

                from_edge = ride.get('from', '')
                to_edge = ride.get('to', '')
                lines = ride.get('lines', '')

                # Only process taxi rides
                if 'taxi' not in lines.lower():
                    continue

                if not from_edge or not to_edge:
                    continue

                # Create reservation with index as ID (matches SUMO behavior)
                res_id = str(len(reservations))
                res = SimulatedReservation(
                    id=res_id,
                    person_ids=[person_id],
                    from_edge=from_edge,
                    to_edge=to_edge,
                    depart_time=depart_time,
                    reservation_time=depart_time,
                    state=1  # New/unassigned
                )
                reservations.append(res)

            # Sort by depart time
            reservations.sort(key=lambda r: r.depart_time)
            self._pending_reservations_from_file = reservations
            self._reservation_file_loaded = True

            print(f"[SimulatedTaxi] Loaded {len(reservations)} reservations from {file_path}")

        except Exception as e:
            print(f"[SimulatedTaxi] Error loading reservations: {e}")
            traceback.print_exc()

    def _load_taxi_fleet_from_file(
        self,
        file_path: str,
        env: Any,
        config: Dict[str, Any],
        current_time: float
    ) -> None:
        """
        Load taxi fleet definitions from taxi_fleet.rou.xml file.
        Creates simulated taxi entries for each vehicle.

        When USE_SIMULATED_TAXI_SYSTEM is True the taxi_fleet.rou.xml is NOT
        loaded by SUMO (it was removed from the sumocfg).  So we must also
        inject the vehicles into the running simulation via TraCI.

        The file typically contains single-edge routes on pedestrian edges
        (pickup locations).  Since taxi vClass cannot depart on pedestrian
        lanes, we use TAZ edges to find valid road edges for spawning.
        """
        if not os.path.exists(file_path):
            print(f"[SimulatedTaxi] Warning: Taxi fleet file not found: {file_path}")
            return

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            traci_conn = getattr(env, 'traci_conn', None)
            use_sim_taxi = bool(config.get("use_simulated_taxi_system", False))
            if not use_sim_taxi and hasattr(env, 'dic_traffic_env_conf'):
                use_sim_taxi = env.dic_traffic_env_conf.get("USE_SIMULATED_TAXI_SYSTEM", False)

            # Build pool of valid spawn edges from TAZ data
            spawn_edges = []
            if use_sim_taxi and self.taz_edges:
                for edges in self.taz_edges.values():
                    spawn_edges.extend(edges)
            if not spawn_edges and use_sim_taxi and traci_conn is not None:
                # Fallback: use all edges currently in the network
                try:
                    spawn_edges = list(traci_conn.edge.getIDList())
                    # Filter out internal edges (start with ':')
                    spawn_edges = [e for e in spawn_edges if not e.startswith(':')]
                except Exception:
                    pass
            # Pre-filter: only keep edges where at least one lane allows taxi vClass
            if spawn_edges and use_sim_taxi and traci_conn is not None:
                filtered = []
                for eid in spawn_edges:
                    try:
                        lane_count = traci_conn.edge.getLaneNumber(eid)
                        for li in range(lane_count):
                            allowed = traci_conn.lane.getAllowed(f"{eid}_{li}")
                            # Empty allowed list means all classes permitted
                            if not allowed or "taxi" in allowed:
                                filtered.append(eid)
                                break
                    except Exception:
                        pass
                print(f"[SimulatedTaxi] Spawn edge pool: {len(spawn_edges)} total -> {len(filtered)} taxi-allowed")
                spawn_edges = filtered

            # Route-verify spawn edges: only keep edges that can actually route to another edge
            if spawn_edges and use_sim_taxi and traci_conn is not None and len(spawn_edges) > 10:
                verified = []
                sample = spawn_edges[:300]
                for edge in sample:
                    tgt = random.choice(spawn_edges)
                    if tgt == edge and len(spawn_edges) > 1:
                        tgt = random.choice([e for e in spawn_edges if e != edge])
                    try:
                        r = traci_conn.simulation.findRoute(edge, tgt, vType="taxi")
                        if r and getattr(r, "edges", None):
                            verified.append(edge)
                    except Exception:
                        pass
                if verified:
                    print(f"[SimulatedTaxi] Route-verified spawn edges: {len(sample)} tested -> {len(verified)} valid")
                    spawn_edges = verified
                else:
                    print("[SimulatedTaxi] WARNING: route verification found 0 valid edges, keeping original pool")

            added_via_traci = 0
            failed_count = 0
            vehicles_in_file = list(root.findall('vehicle'))

            for vehicle in vehicles_in_file:
                veh_id = vehicle.get('id', '')
                if not veh_id:
                    continue

                # Register in simulated taxi dict
                taxi = SimulatedTaxi(
                    id=veh_id,
                    state=SimulatedTaxiState.IDLE,
                    spawn_time=current_time
                )
                self._simulated_taxis[veh_id] = taxi

                # If simulated taxi system is active, inject vehicle via TraCI
                if use_sim_taxi and traci_conn is not None and spawn_edges:
                    injected = False
                    # Try up to 10 random spawn edges to place the taxi
                    for _attempt in range(10):
                        src = random.choice(spawn_edges)
                        try:
                            # Use multi-edge route via findRoute (single-edge routes
                            # fail vClass=taxi depart validation on many edges)
                            target = random.choice(spawn_edges)
                            if target == src and len(spawn_edges) > 1:
                                target = random.choice([e for e in spawn_edges if e != src])
                            try:
                                route_result = traci_conn.simulation.findRoute(src, target, vType="taxi")
                                route_edges = list(route_result.edges) if route_result and route_result.edges else []
                            except Exception:
                                route_edges = []
                            if not route_edges:
                                continue  # skip this spawn edge, try another
                            route_id = f"_simtaxi_route_{veh_id}"
                            traci_conn.route.add(route_id, route_edges)
                            traci_conn.vehicle.add(
                                veh_id,
                                routeID=route_id,
                                typeID="taxi",
                                depart="now",
                                departSpeed="0",
                            )
                            # Freeze the taxi so it stays in the simulation
                            traci_conn.vehicle.setSpeed(veh_id, 0)
                            added_via_traci += 1
                            injected = True
                            break
                        except Exception:
                            # Clean up failed route if vehicle wasn't added
                            continue
                    if not injected:
                        failed_count += 1
                        # Keep in dict as IDLE; replenish will retry later

            self._taxi_fleet_file_loaded = True
            print(f"[SimulatedTaxi] Loaded {len(self._simulated_taxis)} taxis from {file_path}")
            if use_sim_taxi:
                print(f"[SimulatedTaxi] Injected {added_via_traci} vehicles via TraCI "
                      f"(failed={failed_count})")

        except Exception as e:
            print(f"[SimulatedTaxi] Error loading taxi fleet: {e}")
            traceback.print_exc()

    def _adopt_existing_vehicles(
        self,
        env: Any,
        config: Dict[str, Any],
        current_time: float
    ) -> None:
        """
        Adopt existing vehicles in the simulation as simulated taxis.
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return

        try:
            vehicle_ids = list(env.traci_conn.vehicle.getIDList())
            vtype_filter = config.get("simulated_taxi_vtype_filter", "taxi")

            for veh_id in vehicle_ids:
                try:
                    vtype = env.traci_conn.vehicle.getTypeID(veh_id)
                    if vtype_filter and vtype_filter not in vtype.lower():
                        continue
                except Exception:
                    continue

                if veh_id not in self._simulated_taxis:
                    taxi = SimulatedTaxi(
                        id=veh_id,
                        state=SimulatedTaxiState.IDLE,
                        spawn_time=current_time
                    )
                    self._simulated_taxis[veh_id] = taxi

            print(f"[SimulatedTaxi] Adopted {len(self._simulated_taxis)} existing vehicles")

        except Exception as e:
            print(f"[SimulatedTaxi] Error adopting vehicles: {e}")

    def _activate_pending_reservations(
        self,
        current_time: float,
        config: Dict[str, Any]
    ) -> List[SimulatedReservation]:
        """
        Activate reservations whose depart_time has been reached.
        Returns list of newly activated reservations.
        """
        newly_activated = []

        pending_count = len(self._pending_reservations_from_file)
        # Always log periodically to track current_time progression
        last_log_t = getattr(self, "_activate_last_log_t", -999)
        if pending_count > 0 and (current_time - last_log_t) >= 60:
            first_departs = sorted(r.depart_time for r in list(self._pending_reservations_from_file)[:10])
            # print(f"[SimTaxi-DEBUG] _activate_pending: t={current_time:.1f}s, "
            #       f"pending={pending_count}, active={len(self._simulated_reservations)}, "
            #       f"first_departs={first_departs[:5]}")
            self._activate_last_log_t = current_time

        for res in list(self._pending_reservations_from_file):
            if res.depart_time <= current_time:
                if res.id not in self._simulated_reservations:
                    self._simulated_reservations[res.id] = res
                    newly_activated.append(res)
                self._pending_reservations_from_file.remove(res)

        return newly_activated

    def _get_simulated_taxi_fleet(self, state: int = -1) -> List[str]:
        """
        Get taxi IDs by state. Replaces traci.vehicle.getTaxiFleet().

        Args:
            state: -1=all, 0=idle, 1=pickup, 2=occupied

        Returns:
            List of taxi IDs matching the state filter
        """
        result = []
        for taxi_id, taxi in self._simulated_taxis.items():
            if state == -1:
                result.append(taxi_id)
            elif state == 0 and taxi.state == SimulatedTaxiState.IDLE:
                result.append(taxi_id)
            elif state == 1 and taxi.state == SimulatedTaxiState.PICKUP:
                result.append(taxi_id)
            elif state == 2 and taxi.state == SimulatedTaxiState.OCCUPIED:
                result.append(taxi_id)
        return result

    def _get_simulated_reservations(
        self,
        only_pending: bool = True
    ) -> List[SimulatedReservation]:
        """
        Get simulated reservations. Replaces traci.person.getTaxiReservations().

        Args:
            only_pending: If True, only return reservations not yet completed

        Returns:
            List of SimulatedReservation objects
        """
        result = []
        for res in self._simulated_reservations.values():
            if only_pending:
                # Not completed (state != 16)
                if (res.state & 16) == 0:
                    result.append(res)
            else:
                result.append(res)
        return result

    def _execute_simulated_dispatch(
        self,
        traci_conn: Any,
        taxi_id: str,
        reservation_id: str,
        control_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Execute dispatch using ordinary vehicle control API.
        Replaces traci.vehicle.dispatchTaxi().
        """
        res = self._simulated_reservations.get(reservation_id)
        taxi = self._simulated_taxis.get(taxi_id)

        if res is None:
            return False, f"Reservation {reservation_id} not found"
        if taxi is None:
            return False, f"Taxi {taxi_id} not found"
        if taxi.state != SimulatedTaxiState.IDLE:
            return False, f"Taxi {taxi_id} not idle (state={taxi.state})"

        try:
            # Get current taxi position
            current_edge = traci_conn.vehicle.getRoadID(taxi_id)
            if not current_edge or current_edge.startswith(":"):
                return False, "Cannot get taxi current edge"

            # Calculate route to pickup - multi-strategy fallback
            route = None
            effective_pickup_edge = res.from_edge

            # Strategy 1: direct findRoute with vType="taxi"
            try:
                route = traci_conn.simulation.findRoute(current_edge, res.from_edge, vType="taxi")
            except Exception:
                pass
            if not route or not route.edges:
                # Strategy 1b: direct findRoute without vType
                try:
                    route = traci_conn.simulation.findRoute(current_edge, res.from_edge)
                except Exception:
                    pass

            if not route or not route.edges:
                # Strategy 2: convertRoad to find nearest reachable edge
                effective_pickup_edge = None
                try:
                    shape = traci_conn.lane.getShape(f"{res.from_edge}_0")
                    if shape:
                        mid = shape[len(shape) // 2]
                        nearest = traci_conn.simulation.convertRoad(mid[0], mid[1])
                        if nearest and nearest[0] and not nearest[0].startswith(":"):
                            alt_edge = nearest[0]
                            route = traci_conn.simulation.findRoute(current_edge, alt_edge, vType="taxi")
                            if route and route.edges:
                                effective_pickup_edge = alt_edge
                except Exception:
                    pass

            if not route or not route.edges:
                # Strategy 3: search nearby offsets (±200m) for a reachable edge
                effective_pickup_edge = None
                try:
                    shape = traci_conn.lane.getShape(f"{res.from_edge}_0")
                    if shape:
                        mid = shape[len(shape) // 2]
                        for dx, dy in [(200, 0), (-200, 0), (0, 200), (0, -200),
                                       (150, 150), (-150, -150)]:
                            try:
                                nearby = traci_conn.simulation.convertRoad(
                                    mid[0] + dx, mid[1] + dy)
                                if nearby and nearby[0] and not nearby[0].startswith(":"):
                                    alt = nearby[0]
                                    r = traci_conn.simulation.findRoute(
                                        current_edge, alt, vType="taxi")
                                    if r and r.edges:
                                        route = r
                                        effective_pickup_edge = alt
                                        break
                            except Exception:
                                continue
                except Exception:
                    pass

            if not effective_pickup_edge:
                effective_pickup_edge = res.from_edge
            if not route or not route.edges:
                return False, f"No route from {current_edge} to {res.from_edge}"

            # Release the taxi from speed freeze so it can move
            traci_conn.vehicle.setSpeed(taxi_id, -1)

            # Set route to pickup location
            traci_conn.vehicle.setRoute(taxi_id, route.edges)

            # Store pickup position for proximity-based detection
            pickup_pos = None
            try:
                shape = traci_conn.lane.getShape(f"{res.from_edge}_0")
                if shape:
                    mid = shape[len(shape) // 2]
                    pickup_pos = (float(mid[0]), float(mid[1]))
            except Exception:
                pass

            # Update taxi state
            taxi.state = SimulatedTaxiState.PICKUP
            taxi.current_reservation_id = reservation_id
            taxi.route_stage = "to_pickup"
            taxi.pickup_edge = effective_pickup_edge
            taxi.dropoff_edge = res.to_edge
            taxi.pickup_pos = pickup_pos  # for proximity detection
            taxi.dropoff_route_set = False
            taxi.stage_start_time = traci_conn.simulation.getTime()

            # Update reservation state
            res.state = 4  # Assigned
            res.assigned_taxi = taxi_id

            return True, "Success"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def _update_taxi_trip_progress(
        self,
        traci_conn: Any,
        current_time: float,
        control_state: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Monitor taxi trip progress and update states.
        Called every control cycle.
        """
        events = {"pickups": [], "dropoffs": [], "removed": []}
        arrival_threshold = float(config.get("arrival_distance_threshold", 50.0))

        # Get current vehicle list to check which taxis still exist
        try:
            current_vehicle_ids = set(traci_conn.vehicle.getIDList())
        except Exception:
            current_vehicle_ids = set()

        # Clean up taxis that have left the simulation
        # Grace period: don't remove taxis that were just spawned (they need
        # one simulation step to appear in SUMO's vehicle list)
        grace_seconds = float(config.get("spawn_grace_seconds", 60.0))
        for taxi_id in list(self._simulated_taxis.keys()):
            if taxi_id not in current_vehicle_ids:
                taxi_obj = self._simulated_taxis.get(taxi_id)
                if taxi_obj and (current_time - taxi_obj.spawn_time) < grace_seconds:
                    continue  # too young to remove, may not have appeared yet
                # Don't remove IDLE taxis that were never dispatched - they may
                # not have been injected into SUMO yet (e.g. after checkpoint reload)
                if taxi_obj and taxi_obj.state == SimulatedTaxiState.IDLE and not taxi_obj.route_stage:
                    continue
                # If taxi was in to_dropoff stage, count as completed dropoff
                if taxi_obj and taxi_obj.route_stage == "to_dropoff" and taxi_obj.current_reservation_id:
                    res = self._simulated_reservations.get(taxi_obj.current_reservation_id)
                    if res and res.state != 16:
                        pickup_time = res.pickup_time or current_time
                        travel_time = current_time - pickup_time
                        res.state = 16
                        res.dropoff_time = current_time
                        control_state["passenger_dropoffs"] = control_state.get("passenger_dropoffs", 0) + 1
                        control_state.setdefault("passenger_travel_times", []).append(travel_time)
                        fare = self._estimate_fare(travel_time, control_state)
                        taxi_income = control_state.setdefault("taxi_income", {})
                        taxi_income[taxi_id] = float(taxi_income.get(taxi_id, 0.0)) + fare
                        control_state["total_income"] = float(control_state.get("total_income", 0.0)) + fare
                        events["dropoffs"].append({
                            "taxi_id": taxi_id, "reservation_id": res.id, "travel_time": travel_time
                        })
                        # print(f"[SimTaxi-DEBUG] DROPOFF (vehicle removed): taxi={taxi_id}, "
                        #       f"res={res.id}, t={current_time:.0f}s, travel={travel_time:.1f}s")
                # If taxi was in to_pickup stage, reset reservation so it can be re-assigned
                if taxi_obj and taxi_obj.route_stage == "to_pickup" and taxi_obj.current_reservation_id:
                    res = self._simulated_reservations.get(taxi_obj.current_reservation_id)
                    if res and res.state == 4:
                        res.state = 1  # Reset to unassigned
                        res.assigned_taxi = None
                        # print(f"[SimTaxi-DEBUG] RESET reservation (taxi removed during pickup): "
                        #       f"taxi={taxi_id}, res={res.id}, t={current_time:.0f}s")
                removed_taxi = self._simulated_taxis.pop(taxi_id, None)
                if removed_taxi:
                    events["removed"].append({"taxi_id": taxi_id})

        for taxi_id, taxi in list(self._simulated_taxis.items()):
            if taxi.state == SimulatedTaxiState.IDLE:
                continue

            # Check if taxi still exists
            try:
                current_edge = traci_conn.vehicle.getRoadID(taxi_id)
            except Exception:
                continue

            if not current_edge:
                continue

            res = self._simulated_reservations.get(taxi.current_reservation_id)
            if res is None:
                taxi.state = SimulatedTaxiState.IDLE
                taxi.route_stage = "idle"
                taxi.current_reservation_id = None
                continue

            # Check pickup arrival - edge match OR proximity OR route completed OR timeout
            if taxi.route_stage == "to_pickup":
                arrived = False
                if current_edge == taxi.pickup_edge:
                    arrived = True
                elif hasattr(taxi, "pickup_pos") and taxi.pickup_pos:
                    try:
                        pos = traci_conn.vehicle.getPosition(taxi_id)
                        dx = pos[0] - taxi.pickup_pos[0]
                        dy = pos[1] - taxi.pickup_pos[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < arrival_threshold:
                            arrived = True
                    except Exception:
                        pass
                if not arrived:
                    try:
                        route_idx = traci_conn.vehicle.getRouteIndex(taxi_id)
                        route = traci_conn.vehicle.getRoute(taxi_id)
                        speed = traci_conn.vehicle.getSpeed(taxi_id)
                        if route and route_idx >= len(route) - 1 and speed < 0.5:
                            arrived = True
                    except Exception:
                        pass

                # Pickup timeout: force-arrive after 600s
                pickup_stage_dur = current_time - getattr(taxi, "stage_start_time", current_time)
                if not arrived and pickup_stage_dur > 600.0:
                    # print(f"[SimTaxi-DEBUG] TIMEOUT pickup: taxi={taxi_id}, "
                    #       f"stuck for {pickup_stage_dur:.0f}s, force-arriving")
                    arrived = True

                # Reroute stuck taxis heading to pickup
                if not arrived and pickup_stage_dur > 120.0:
                    try:
                        speed = traci_conn.vehicle.getSpeed(taxi_id)
                        if speed < 0.1:
                            last_rr = getattr(taxi, "_last_reroute_t", 0.0)
                            if current_time - last_rr > 120.0:
                                traci_conn.vehicle.rerouteTraveltime(taxi_id)
                                taxi._last_reroute_t = current_time
                    except Exception:
                        pass

                if arrived:
                    self._handle_pickup_arrival(
                        traci_conn, taxi, res, current_time, control_state, events
                    )

            # Check dropoff arrival - edge match OR proximity OR route completed OR timeout
            elif taxi.route_stage == "to_dropoff":
                # If dropoff route was never set (e.g. taxi was on junction edge), retry
                if not getattr(taxi, "dropoff_route_set", False):
                    self._try_set_dropoff_route(traci_conn, taxi, res)

                arrived = False
                # 1) Exact edge match
                if current_edge == taxi.dropoff_edge:
                    arrived = True

                # 2) Proximity-based detection using cached dropoff_pos
                if not arrived:
                    dropoff_pos = getattr(taxi, "dropoff_pos", None)
                    if dropoff_pos:
                        try:
                            pos = traci_conn.vehicle.getPosition(taxi_id)
                            dx = pos[0] - dropoff_pos[0]
                            dy = pos[1] - dropoff_pos[1]
                            dist = (dx * dx + dy * dy) ** 0.5
                            if dist < arrival_threshold:
                                arrived = True
                        except Exception:
                            pass

                # 3) Route completion detection (reached last edge and stopped)
                if not arrived:
                    try:
                        route_idx = traci_conn.vehicle.getRouteIndex(taxi_id)
                        route = traci_conn.vehicle.getRoute(taxi_id)
                        speed = traci_conn.vehicle.getSpeed(taxi_id)
                        if route and route_idx >= len(route) - 1 and speed < 0.5:
                            arrived = True
                    except Exception:
                        pass

                # 4) Timeout: force-complete after 600s in to_dropoff
                stage_duration = current_time - getattr(taxi, "stage_start_time", current_time)
                if not arrived and stage_duration > 600.0:
                    # print(f"[SimTaxi-DEBUG] TIMEOUT dropoff: taxi={taxi_id}, "
                    #       f"stuck for {stage_duration:.0f}s, force-completing")
                    arrived = True

                # 5) Reroute stuck taxis: if stuck > 120s, try rerouting
                if not arrived and stage_duration > 120.0:
                    try:
                        speed = traci_conn.vehicle.getSpeed(taxi_id)
                        if speed < 0.1:
                            last_reroute = getattr(taxi, "_last_reroute_t", 0.0)
                            if current_time - last_reroute > 120.0:
                                traci_conn.vehicle.rerouteTraveltime(taxi_id)
                                taxi._last_reroute_t = current_time
                    except Exception:
                        pass

                if arrived:
                    self._handle_dropoff_arrival(
                        traci_conn, taxi, res, current_time, control_state, events
                    )

        return events

    def _handle_pickup_arrival(
        self,
        traci_conn: Any,
        taxi: SimulatedTaxi,
        res: SimulatedReservation,
        current_time: float,
        control_state: Dict[str, Any],
        events: Dict[str, List]
    ) -> None:
        """Handle taxi arrival at pickup location."""
        # Update states
        taxi.state = SimulatedTaxiState.OCCUPIED
        taxi.route_stage = "to_dropoff"
        res.state = 8  # Picked up
        res.pickup_time = current_time

        # Record metrics
        control_state["passenger_pickups"] = control_state.get("passenger_pickups", 0) + 1
        wait_time = current_time - res.reservation_time
        control_state.setdefault("passenger_wait_times", []).append(wait_time)
        # print(f"[SimTaxi-DEBUG] PICKUP: taxi={taxi.id}, res={res.id}, t={current_time:.0f}s, wait={wait_time:.1f}s, dropoff_edge={res.to_edge}")

        # Set route to dropoff - handle pedestrian-only dropoff edges and junction edges
        taxi.dropoff_route_set = False
        taxi.stage_start_time = current_time
        self._try_set_dropoff_route(traci_conn, taxi, res)

        events["pickups"].append({
            "taxi_id": taxi.id,
            "reservation_id": res.id,
            "wait_time": wait_time
        })

    def _try_set_dropoff_route(
        self,
        traci_conn: Any,
        taxi: SimulatedTaxi,
        res: SimulatedReservation,
    ) -> bool:
        """Try to set the dropoff route for a taxi. Returns True if successful."""
        try:
            current_edge = traci_conn.vehicle.getRoadID(taxi.id)
            if not current_edge or current_edge.startswith(":"):
                return False  # On junction edge, retry later

            route = traci_conn.simulation.findRoute(current_edge, res.to_edge)
            effective_dropoff = res.to_edge
            if not route or not route.edges:
                # Dropoff edge may be pedestrian-only; find nearest reachable
                try:
                    shape = traci_conn.lane.getShape(f"{res.to_edge}_0")
                    if shape:
                        mid = shape[len(shape) // 2]
                        nearest = traci_conn.simulation.convertRoad(mid[0], mid[1])
                        if nearest and nearest[0] and not nearest[0].startswith(":"):
                            route = traci_conn.simulation.findRoute(current_edge, nearest[0])
                            if route and route.edges:
                                effective_dropoff = nearest[0]
                except Exception:
                    pass

            if route and route.edges:
                traci_conn.vehicle.setRoute(taxi.id, route.edges)
                taxi.dropoff_edge = effective_dropoff
                taxi.dropoff_route_set = True
                # Store dropoff position for proximity detection
                try:
                    shape = traci_conn.lane.getShape(f"{effective_dropoff}_0")
                    if shape:
                        mid = shape[len(shape) // 2]
                        taxi.dropoff_pos = (float(mid[0]), float(mid[1]))
                except Exception:
                    taxi.dropoff_pos = None
                # print(f"[SimTaxi-DEBUG] Dropoff route set: taxi={taxi.id}, "
                #     #   f"edges={len(route.edges)}, effective_dropoff={effective_dropoff}")
                return True
            else:
                # print(f"[SimTaxi-DEBUG] Dropoff route FAILED: taxi={taxi.id}, to_edge={res.to_edge}")
                return False
        except Exception as e:
            print(f"[SimulatedTaxi] Error setting dropoff route: {e}")
            return False

    def _handle_dropoff_arrival(
        self,
        traci_conn: Any,
        taxi: SimulatedTaxi,
        res: SimulatedReservation,
        current_time: float,
        control_state: Dict[str, Any],
        events: Dict[str, List]
    ) -> None:
        """Handle taxi arrival at dropoff location."""
        # Calculate travel time
        pickup_time = res.pickup_time or current_time
        travel_time = current_time - pickup_time

        # Update states
        taxi.state = SimulatedTaxiState.IDLE
        taxi.route_stage = "idle"
        taxi.current_reservation_id = None
        taxi.pickup_edge = None
        taxi.dropoff_edge = None
        taxi.dropoff_route_set = False
        taxi.stage_start_time = current_time

        res.state = 16  # Completed
        res.dropoff_time = current_time

        # print(f"[SimTaxi-DEBUG] DROPOFF: taxi={taxi.id}, res={res.id}, "
        #       f"t={current_time:.0f}s, travel={travel_time:.1f}s")

        # Record metrics
        control_state["passenger_dropoffs"] = control_state.get("passenger_dropoffs", 0) + 1
        control_state.setdefault("passenger_travel_times", []).append(travel_time)

        # Update income
        fare = self._estimate_fare(travel_time, control_state)
        taxi_income = control_state.setdefault("taxi_income", {})
        taxi_income[taxi.id] = float(taxi_income.get(taxi.id, 0.0)) + fare
        control_state["total_income"] = float(control_state.get("total_income", 0.0)) + fare

        events["dropoffs"].append({
            "taxi_id": taxi.id,
            "reservation_id": res.id,
            "travel_time": travel_time
        })

        # Keep taxi alive in SUMO: reset route and freeze speed to prevent removal
        try:
            current_edge = traci_conn.vehicle.getRoadID(taxi.id)
            if current_edge and not current_edge.startswith(":"):
                traci_conn.vehicle.setRoute(taxi.id, [current_edge])
                traci_conn.vehicle.setSpeed(taxi.id, 0)
        except Exception:
            pass

        # Check if taxi is in a cul-de-sac and move it out
        try:
            current_edge = traci_conn.vehicle.getRoadID(taxi.id)
            if current_edge and self._edge_is_dead_end(traci_conn, current_edge):
                escape_edge = self._find_escape_edge_from_dead_end(
                    traci_conn, current_edge, taxi.id
                )
                if escape_edge:
                    traci_conn.vehicle.changeTarget(taxi.id, escape_edge)
        except Exception:
            pass

    def _add_simulated_taxi(
        self,
        taxi_id: str,
        current_time: float
    ) -> SimulatedTaxi:
        """Add a new taxi to the simulated fleet."""
        taxi = SimulatedTaxi(
            id=taxi_id,
            state=SimulatedTaxiState.IDLE,
            spawn_time=current_time
        )
        self._simulated_taxis[taxi_id] = taxi
        return taxi

    @staticmethod
    def _normalize_reservation_id(reservation_id: Any) -> str:
        """
        Normalize reservation IDs to strings for TraCI.
        SUMO/TraCI reservation IDs are typically strings like "0", "1", ...
        """
        return str(reservation_id)

    @staticmethod
    def _reservation_state_int(state: Any) -> Optional[int]:
        """Best-effort conversion of reservation state to int."""
        if state is None:
            return None
        try:
            return int(state)
        except Exception:
            return None

    def _reservation_is_picked_up(self, state: Any) -> bool:
        """
        Determine whether a reservation is already picked up (passenger on board).

        NOTE: In SUMO (1.20+), Reservation.state is an int that commonly appears as:
          - 1: new/unassigned
          - 4: assigned/awaiting pickup
          - 8: picked up / on board
        Some versions encode this as bit flags, so we treat '8' as the picked-up bit.
        """
        state_int = self._reservation_state_int(state)
        if state_int is None:
            return False
        return (state_int & 8) != 0

    def _reservation_is_dispatchable(self, state: Any, config: Dict[str, Any]) -> bool:
        """
        Determine whether a reservation is eligible for dispatch.
        """
        if self._reservation_is_picked_up(state):
            return False
        state_int = self._reservation_state_int(state)
        if state_int is None:
            return True
        # Skip already assigned reservations.
        if (state_int & 4) != 0:
            return False
        # Skip retrieved reservations (state bit 2) ONLY if explicitly enabled.
        # NOTE: getTaxiReservations(0) marks reservations as retrieved (state=2) after the first call.
        # If we skip state=2 by default, we will starve dispatch after the first cycle.
        force_skip = bool(config.get("force_skip_retrieved_reservations", False))
        skip_retrieved = bool(config.get("skip_retrieved_reservations", False))
        if (force_skip or skip_retrieved) and (state_int & 2) != 0:
            return False
        return True

    def _taxi_is_mature_for_dispatch(
        self,
        taxi_id: str,
        current_time: float,
        control_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> bool:
        """Avoid dispatching a taxi immediately after it is spawned."""
        try:
            min_age = float(config.get("min_dispatch_taxi_age", 1.0))
        except Exception:
            min_age = 1.0
        if min_age <= 0:
            return True

        spawn_times = control_state.get("taxi_spawn_times", {})
        if not isinstance(spawn_times, dict):
            return True
        spawn_time = spawn_times.get(taxi_id)
        if spawn_time is None:
            return True
        try:
            return (float(current_time) - float(spawn_time)) >= min_age
        except Exception:
            return True

    def _safe_get_taxi_reservations(self, traci_conn: Any, max_retries: int = 3, env: Any = None, use_simulated: bool = None) -> List[Any]:
        """
        Get pending taxi reservations from the simulated reservation system.

        Signature preserved for API compatibility; parameters other than self are unused.

        Returns:
            List of SimulatedReservation objects with pending state.
        """
        # Always use simulated reservation system (native SUMO taxi API removed)
        return self._get_simulated_reservations(only_pending=True)

    def _safe_get_taxi_fleet(self, traci_conn: Any, state: int = -1, max_retries: int = 3, env: Any = None, use_simulated: bool = None) -> List[str]:
        """
        Get taxi fleet from the simulated taxi system.

        Signature preserved for API compatibility; only `state` is used.

        Args:
            state: Taxi state filter (-1=all, 0=idle, 1=pickup, 2=occupied)

        Returns:
            List of taxi IDs, or empty list on failure
        """
        # Always use simulated taxi fleet (native SUMO taxi API removed)
        result = self._get_simulated_taxi_fleet(state)
        # if not result and len(self._simulated_taxis) > 0:
        #     print(f"[SimTaxi-DEBUG] _safe_get_taxi_fleet: state={state}, "
        #           f"simulated_taxis={len(self._simulated_taxis)}, "
        #           f"initialized={self._simulated_system_initialized}, "
        #           f"result={len(result)}, id(self)={id(self)}")
        return result

    def _get_cached_taxi_allowed_edges(self, env: Any) -> Optional[Set[str]]:
        """
        Cache and return edges that allow taxi class.

        This avoids repeated per-edge getAllowed calls every control cycle.
        """
        if getattr(self, "_taxi_allowed_edges_cache", None) is not None:
            return self._taxi_allowed_edges_cache
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return None
        try:
            edge_ids = list(env.traci_conn.edge.getIDList())
        except Exception:
            return None
        allowed_edges: Set[str] = set()
        for edge_id in edge_ids:
            if not edge_id or edge_id.startswith(":"):
                continue
            allowed = None
            try:
                allowed = env.traci_conn.edge.getAllowed(edge_id)
            except Exception:
                allowed = None
            # Only allow edges that explicitly allow taxis (or are unrestricted).
            if allowed and "taxi" not in allowed:
                continue
            allowed_edges.add(edge_id)
        self._taxi_allowed_edges_cache = allowed_edges
        return allowed_edges

    @staticmethod
    def _parse_taxi_customers(customers: Any) -> List[str]:
        """
        Parse SUMO taxi customers parameter into a list of reservation ids (strings).
        SUMO commonly returns a whitespace-separated string, but we handle a few variants.
        """
        if customers is None:
            return []
        if isinstance(customers, (list, tuple, set)):
            return [str(x) for x in customers]

        try:
            s = str(customers).strip()
        except Exception:
            return []

        if not s:
            return []

        # Remove simple bracket wrappers
        s = s.strip().strip("[]")
        # Replace commas with whitespace then split
        parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
        return [str(p) for p in parts]

    def _prune_recent_order_times(
        self,
        order_times: List[float],
        current_time: float,
        window_seconds: float,
    ) -> List[float]:
        if not order_times:
            return []
        cutoff = float(current_time) - float(window_seconds)
        return [float(t) for t in order_times if float(t) >= cutoff]

    def _get_recent_order_count(
        self,
        control_state: Dict[str, Any],
        taxi_id: str,
        current_time: float,
        window_seconds: float,
    ) -> int:
        recent_map = control_state.setdefault("taxi_recent_order_times", {})
        raw_times = recent_map.get(taxi_id) or []
        pruned = self._prune_recent_order_times(raw_times, current_time, window_seconds)
        recent_map[taxi_id] = pruned
        return len(pruned)

    def _record_taxi_order_time(self, control_state: Dict[str, Any], taxi_id: str, when: float) -> None:
        recent_map = control_state.setdefault("taxi_recent_order_times", {})
        order_times = list(recent_map.get(taxi_id) or [])
        order_times.append(float(when))
        window_seconds = float(control_state.get("recent_order_window_seconds", 1800))
        recent_map[taxi_id] = self._prune_recent_order_times(order_times, when, window_seconds)

    def _estimate_fare(self, travel_time_seconds: float, control_state: Dict[str, Any]) -> float:
        base = float(control_state.get("fare_base", 3.0))
        per_sec = float(control_state.get("fare_per_second", 0.01))
        fare = base + float(travel_time_seconds) * per_sec
        return max(0.0, float(fare))

    def _sync_reservation_to_taxi_mapping(
        self,
        traci: Any,
        control_state: Dict[str, Any],
        active_reservation_ids: Optional[set] = None,
        env: Any = None,
    ) -> None:
        """
        Best-effort mapping from reservation_id -> taxi_id, inferred from taxi customers list.
        Useful after checkpoint load or if the module started tracking mid-episode.
        """
        mapping = control_state.setdefault("reservation_to_taxi", {})
        # Use safe method to avoid protocol corruption
        all_taxis = self._safe_get_taxi_fleet(traci, state=-1, env=env)
        if not all_taxis:
            return

        for taxi_id in all_taxis:
            try:
                customers = traci.vehicle.getParameter(taxi_id, "device.taxi.customers")
            except Exception:
                continue

            for rid in self._parse_taxi_customers(customers):
                res_id = self._normalize_reservation_id(rid)
                if active_reservation_ids is not None and res_id not in active_reservation_ids:
                    continue
                if not res_id:
                    continue
                # Prefer the latest observed mapping
                mapping[res_id] = taxi_id

    def _get_reservation_pickup_position(self, traci: Any, reservation_obj: Any) -> Optional[Tuple[float, float]]:
        """Get pickup position for a reservation."""
        try:
            # Handle SimulatedReservation - use from_edge to get position via lane shape
            if isinstance(reservation_obj, SimulatedReservation):
                from_edge = reservation_obj.from_edge
                if from_edge:
                    try:
                        lane_id = f"{from_edge}_0"
                        shape = traci.lane.getShape(lane_id)
                        if shape and len(shape) > 0:
                            mid_idx = len(shape) // 2
                            return (float(shape[mid_idx][0]), float(shape[mid_idx][1]))
                    except Exception:
                        pass
                return None

            # Handle SUMO Reservation - use person position
            persons = list(reservation_obj.persons) if hasattr(reservation_obj, "persons") else []
            if not persons:
                return None
            pos = traci.person.getPosition(persons[0])
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return (float(pos[0]), float(pos[1]))
            return None
        except Exception:
            return None

    @staticmethod
    def _edge_is_dead_end(traci: Any, edge_id: Optional[str]) -> bool:
        """
        Check if an edge is a dead end (cul-de-sac) that taxis should avoid.

        A dead end is an edge that:
        1. Is an internal junction edge (starts with ':')
        2. Has no outgoing edges (true dead end)
        3. All outgoing edges lead back to the same edge (loop/turnaround)
        """
        if not edge_id or str(edge_id).startswith(":"):
            return True
        try:
            outgoing = traci.edge.getOutgoing(edge_id)
        except Exception:
            return True
        if not outgoing:
            return True
        # Check if all outgoing edges are internal junctions or lead back to self
        valid_outgoing = []
        for out_edge_data in outgoing:
            # outgoing is a list of tuples: (edge_id, via_lane)
            if isinstance(out_edge_data, tuple):
                out_edge_id = out_edge_data[0]
            else:
                out_edge_id = str(out_edge_data)
            if out_edge_id and not str(out_edge_id).startswith(":") and out_edge_id != edge_id:
                valid_outgoing.append(out_edge_id)
        return len(valid_outgoing) == 0

    def _find_escape_edge_from_dead_end(
        self,
        traci: Any,
        current_edge: str,
        taxi_id: str
    ) -> Optional[str]:
        """
        Find a nearby non-dead-end edge for taxi to escape from cul-de-sac.
        Uses BFS to find the nearest valid edge.
        """
        if not current_edge:
            return None

        visited = {current_edge}
        queue = [current_edge]
        max_depth = 10

        for _ in range(max_depth):
            if not queue:
                break
            edge = queue.pop(0)
            try:
                # Check incoming edges (to go back)
                incoming = traci.edge.getIncoming(edge)
                for in_edge_data in incoming:
                    if isinstance(in_edge_data, tuple):
                        in_edge_id = in_edge_data[0]
                    else:
                        in_edge_id = str(in_edge_data)

                    if not in_edge_id or in_edge_id in visited:
                        continue
                    if str(in_edge_id).startswith(":"):
                        continue

                    visited.add(in_edge_id)

                    # Check if this edge is not a dead end
                    if not self._edge_is_dead_end(traci, in_edge_id):
                        # Verify route exists
                        route = traci.simulation.findRoute(current_edge, in_edge_id)
                        if route and route.edges:
                            return in_edge_id

                    queue.append(in_edge_id)
            except Exception:
                continue

        # Fallback: use TAZ edges if available
        if self.taz_edges:
            for taz_id, edges in self.taz_edges.items():
                for edge in edges[:3]:
                    if not self._edge_is_dead_end(traci, edge):
                        try:
                            route = traci.simulation.findRoute(current_edge, edge)
                            if route and route.edges:
                                return edge
                        except Exception:
                            continue
        return None

    @staticmethod
    def _route_is_reachable(traci: Any, from_edge: Optional[str], to_edge: Optional[str]) -> bool:
        if not from_edge or not to_edge:
            return False
        # Internal edges (":" prefix) can break findRoute; allow target selection
        # and let changeTarget validate at execution time.
        if str(from_edge).startswith(":"):
            return True
        try:
            try:
                route = traci.simulation.findRoute(from_edge, to_edge, vType="taxi")
            except TypeError:
                route = traci.simulation.findRoute(from_edge, to_edge)
        except Exception:
            # Fallback: try without vType if first attempt fails for any reason
            try:
                route = traci.simulation.findRoute(from_edge, to_edge)
            except Exception:
                return False
        edges = getattr(route, "edges", None)
        if not edges:
            return False
        length = getattr(route, "length", None)
        if length is not None and float(length) <= 0:
            return False
        return True

    def _select_reposition_edge(
        self,
        traci: Any,
        taxi_edge: Optional[str],
        taz_edges: List[str],
    ) -> Optional[str]:
        if not taz_edges:
            return None
        if taxi_edge and str(taxi_edge).startswith(":"):
            taxi_edge = None
        candidates = list(taz_edges)
        random.shuffle(candidates)
        for edge_id in candidates:
            if self._edge_is_dead_end(traci, edge_id):
                continue
            if taxi_edge and not self._route_is_reachable(traci, taxi_edge, edge_id):
                continue
            return edge_id
        return None

    def _generate_greedy_dispatch_decisions(
        self,
        traci: Any,
        active_reservations: Dict[str, Any],
        idle_taxis_set: set,
        control_state: Dict[str, Any],
        config: Dict[str, Any],
        current_time: float,
        exclude_reservation_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Greedy dispatcher with fairness tie-breaker:
        - Primary: distance to pickup position
        - If distances are within a threshold: prefer lower cumulative income and fewer recent orders
        """
        # print(f"[GreedyDispatch-DEBUG] ENTER: active_reservations={len(active_reservations)}, idle_taxis_set={len(idle_taxis_set)}")
        if not active_reservations or not idle_taxis_set:
            # print(f"[GreedyDispatch-DEBUG] EARLY EXIT: no reservations or no idle taxis")
            return []

        # Expire _failed_route_pairs every 600s so previously-failed pairs get retried
        _fp_clear_interval = 600.0
        _fp_last_clear = control_state.get("_failed_pairs_clear_time", 0)
        if current_time - _fp_last_clear > _fp_clear_interval:
            control_state["_failed_route_pairs"] = set()
            control_state["_failed_pairs_clear_time"] = current_time

        tie_threshold = float(config.get("greedy_distance_tie_threshold", 50.0))
        weights = config.get("greedy_score_weights") or {}
        w_dist = float(weights.get("distance", 0.7))
        w_income = float(weights.get("income", 0.2))
        w_recent = float(weights.get("recent_orders", 0.1))
        window_seconds = float(config.get("recent_order_window_seconds", control_state.get("recent_order_window_seconds", 1800)))

        # Cache taxi positions/income
        taxi_positions: Dict[str, Tuple[float, float]] = {}
        taxi_income = control_state.setdefault("taxi_income", {})
        for taxi_id in list(idle_taxis_set):
            try:
                pos = traci.vehicle.getPosition(taxi_id)
            except Exception:
                continue
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                taxi_positions[taxi_id] = (float(pos[0]), float(pos[1]))
            taxi_income.setdefault(taxi_id, 0.0)

        available_taxis = set(taxi_positions.keys())
        # print(f"[GreedyDispatch-DEBUG] taxi_positions={len(taxi_positions)}, available_taxis={len(available_taxis)}")
        if not available_taxis:
            # print(f"[GreedyDispatch-DEBUG] EARLY EXIT: no taxi positions found")
            return []

        exclude_reservation_ids = {self._normalize_reservation_id(r) for r in (exclude_reservation_ids or set())}

        # Choose reservations that are dispatchable (not picked up, not already assigned/retrieved)
        dispatchable: List[Tuple[str, Any]] = []
        excluded_count = 0
        not_dispatchable_count = 0
        for rid, res in active_reservations.items():
            if self._normalize_reservation_id(rid) in exclude_reservation_ids:
                excluded_count += 1
                continue
            state = getattr(res, "state", None)
            if not self._reservation_is_dispatchable(state, config):
                not_dispatchable_count += 1
                continue
            # NOTE: No longer skip reservations based on unreachable edges.
            # Enhanced fallback routing in _execute_simulated_dispatch handles
            # pedestrian-only or disconnected edges via convertRoad + nearby search.
            dispatchable.append((rid, res))
        # print(f"[GreedyDispatch-DEBUG] dispatchable={len(dispatchable)}, excluded={excluded_count}, not_dispatchable={not_dispatchable_count}")

        # Sort by reservation time - handle both SUMO and Simulated reservations
        def get_res_time_greedy(item):
            r = item[1]
            if isinstance(r, SimulatedReservation):
                return r.reservation_time or 0
            return getattr(r, "reservationTime", 0)
        dispatchable.sort(key=get_res_time_greedy)

        decisions: List[Dict[str, Any]] = []
        no_pickup_pos_count = 0
        for rid, res in dispatchable:
            if not available_taxis:
                break

            pickup_pos = self._get_reservation_pickup_position(traci, res)
            if pickup_pos is None:
                no_pickup_pos_count += 1
                if no_pickup_pos_count <= 3:
                    from_edge = getattr(res, "from_edge", None) or getattr(res, "fromEdge", None)
                    # print(f"[GreedyDispatch-DEBUG] No pickup pos for rid={rid}, from_edge={from_edge}, type={type(res).__name__}")
                continue

            # Compute distances to all available taxis
            failed_pairs = control_state.get("_failed_route_pairs", set())
            norm_rid = self._normalize_reservation_id(rid)
            dists: List[Tuple[str, float]] = []
            for taxi_id in available_taxis:
                # Skip taxis that previously failed routing for this reservation
                if (taxi_id, norm_rid) in failed_pairs:
                    continue
                taxi_pos = taxi_positions.get(taxi_id)
                if taxi_pos is None:
                    continue
                dx = taxi_pos[0] - pickup_pos[0]
                dy = taxi_pos[1] - pickup_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                dists.append((taxi_id, dist))

            if not dists:
                continue

            min_dist = min(d for _, d in dists)
            candidates = [(t, d) for t, d in dists if d <= min_dist + tie_threshold]

            # Fast path: unique nearest
            chosen_taxi = None
            chosen_dist = None
            if len(candidates) == 1:
                chosen_taxi, chosen_dist = candidates[0]
            else:
                incomes = [float(taxi_income.get(t, 0.0)) for t, _ in candidates]
                min_income = min(incomes)
                max_income = max(incomes)
                income_range = max(max_income - min_income, 1e-6)

                recents = [
                    float(self._get_recent_order_count(control_state, t, current_time, window_seconds))
                    for t, _ in candidates
                ]
                min_recent = min(recents)
                max_recent = max(recents)
                recent_range = max(max_recent - min_recent, 1e-6)

                best_score = None
                for (t, d), income, recent in zip(candidates, incomes, recents):
                    dist_norm = (d - min_dist) / max(tie_threshold, 1e-6)
                    income_norm = (income - min_income) / income_range
                    recent_norm = (recent - min_recent) / recent_range
                    score = w_dist * dist_norm + w_income * income_norm + w_recent * recent_norm

                    if best_score is None or score < best_score - 1e-12:
                        best_score = score
                        chosen_taxi = t
                        chosen_dist = d
                    elif best_score is not None and abs(score - best_score) <= 1e-12:
                        # Tie-break by distance
                        if chosen_dist is None or d < chosen_dist:
                            chosen_taxi = t
                            chosen_dist = d

            if chosen_taxi is None:
                continue

            decisions.append({"taxi_id": chosen_taxi, "reservation_ids": [rid]})
            available_taxis.remove(chosen_taxi)

        # print(f"[GreedyDispatch-DEBUG] EXIT: decisions={len(decisions)}, no_pickup_pos={no_pickup_pos_count}")
        return decisions

    def _estimate_pickup_eta_seconds(
        self,
        traci: Any,
        taxi_edge: Optional[str],
        pickup_edge: Optional[str],
        taxi_pos: Optional[Tuple[float, float]],
        pickup_pos: Optional[Tuple[float, float]],
        assumed_speed_mps: float,
    ) -> Optional[float]:
        """
        Estimate time (seconds) for an idle taxi to reach a reservation pickup.

        Priority:
        1) Route ETA via traci.simulation.findRoute(current_edge, pickup_edge)
        2) Euclidean fallback: distance / assumed_speed_mps (requires positions)
        """
        if taxi_edge and pickup_edge:
            try:
                route = traci.simulation.findRoute(taxi_edge, pickup_edge)
                eta = getattr(route, "travelTime", None)
                if eta is not None:
                    eta_f = float(eta)
                    if eta_f >= 0.0:
                        return eta_f
            except Exception:
                pass

        if taxi_pos is None or pickup_pos is None:
            return None

        dx = float(taxi_pos[0]) - float(pickup_pos[0])
        dy = float(taxi_pos[1]) - float(pickup_pos[1])
        dist = (dx * dx + dy * dy) ** 0.5
        speed = max(float(assumed_speed_mps), 1e-6)
        return float(dist) / speed

    def _generate_online_eta_dispatch_decisions(
        self,
        traci: Any,
        active_reservations: Dict[str, Any],
        idle_taxis_set: set,
        control_state: Dict[str, Any],
        config: Dict[str, Any],
        current_time: float,
        exclude_reservation_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Online dispatcher:
        - Uses route ETA (findRoute) as primary cost (more realistic than Euclidean distance).
        - Always includes fairness terms (income + recent orders), not only in tie cases.
        - Runs every control cycle, so it naturally handles new reservations appearing inside a window.
        """
        if not active_reservations or not idle_taxis_set:
            return []

        # Expire _failed_route_pairs every 600s so previously-failed pairs get retried
        _fp_clear_interval = 600.0
        _fp_last_clear = control_state.get("_failed_pairs_clear_time", 0)
        if current_time - _fp_last_clear > _fp_clear_interval:
            control_state["_failed_route_pairs"] = set()
            control_state["_failed_pairs_clear_time"] = current_time

        weights = config.get("online_dispatch_cost_weights") or {}
        w_eta = float(weights.get("eta", 0.7))
        w_income = float(weights.get("income", 0.2))
        w_recent = float(weights.get("recent_orders", 0.1))
        candidate_k = int(config.get("online_dispatch_candidate_k", 15))
        max_reservations = int(config.get("online_dispatch_max_reservations", 25))
        assumed_speed_mps = float(config.get("online_dispatch_assumed_speed_mps", 8.0))

        window_seconds = float(config.get("recent_order_window_seconds", control_state.get("recent_order_window_seconds", 1800)))
        taxi_income = control_state.setdefault("taxi_income", {})

        taxi_positions: Dict[str, Tuple[float, float]] = {}
        taxi_edges: Dict[str, str] = {}
        for taxi_id in list(idle_taxis_set):
            try:
                pos = traci.vehicle.getPosition(taxi_id)
                edge = traci.vehicle.getRoadID(taxi_id)
            except Exception:
                continue
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                taxi_positions[taxi_id] = (float(pos[0]), float(pos[1]))
                taxi_edges[taxi_id] = str(edge) if edge is not None else ""
                taxi_income.setdefault(taxi_id, 0.0)

        available_taxis = set(taxi_positions.keys())
        if not available_taxis:
            return []

        exclude_reservation_ids = {self._normalize_reservation_id(r) for r in (exclude_reservation_ids or set())}

        dispatchable: List[Tuple[str, Any]] = []
        for rid, res in active_reservations.items():
            if self._normalize_reservation_id(rid) in exclude_reservation_ids:
                continue
            state = getattr(res, "state", None)
            if not self._reservation_is_dispatchable(state, config):
                continue
            # NOTE: No longer skip reservations based on unreachable edges.
            # Enhanced fallback routing handles pedestrian-only / disconnected edges.
            dispatchable.append((rid, res))

        # Sort by reservation time - handle both SUMO and Simulated reservations
        def get_res_time_online(item):
            r = item[1]
            if isinstance(r, SimulatedReservation):
                return r.reservation_time or 0
            return getattr(r, "reservationTime", 0)
        dispatchable.sort(key=get_res_time_online)
        pending_count = len(dispatchable)
        if pending_count > 0:
            # Ensure online limits cover current pending demand.
            if max_reservations < pending_count:
                max_reservations = pending_count
            if candidate_k < pending_count:
                candidate_k = pending_count
        if max_reservations > 0:
            dispatchable = dispatchable[:max_reservations]

        decisions: List[Dict[str, Any]] = []
        for rid, res in dispatchable:
            if not available_taxis:
                break

            # Handle both SUMO and Simulated reservations
            if isinstance(res, SimulatedReservation):
                pickup_edge = res.from_edge
            else:
                pickup_edge = getattr(res, "fromEdge", None)
            pickup_pos = self._get_reservation_pickup_position(traci, res)

            # Candidate selection by Euclidean distance (cheap), then score by ETA+fairness.
            failed_pairs = control_state.get("_failed_route_pairs", set())
            norm_rid = self._normalize_reservation_id(rid)
            candidates: List[str]
            if pickup_pos is not None:
                dists: List[Tuple[str, float]] = []
                for taxi_id in available_taxis:
                    if (taxi_id, norm_rid) in failed_pairs:
                        continue
                    taxi_pos = taxi_positions.get(taxi_id)
                    if taxi_pos is None:
                        continue
                    dx = taxi_pos[0] - pickup_pos[0]
                    dy = taxi_pos[1] - pickup_pos[1]
                    dists.append((taxi_id, (dx * dx + dy * dy) ** 0.5))
                dists.sort(key=lambda x: x[1])
                if candidate_k > 0:
                    candidates = [t for t, _ in dists[: min(candidate_k, len(dists))]]
                else:
                    candidates = [t for t, _ in dists]
            else:
                # If pickup position isn't available, fall back to considering a bounded subset
                # of available taxis (route ETA will still guide selection).
                candidates = list(available_taxis)
                if candidate_k > 0 and len(candidates) > candidate_k:
                    candidates = random.sample(candidates, candidate_k)

            scored: List[Tuple[str, float, float, float, float]] = []
            for taxi_id in candidates:
                taxi_pos = taxi_positions.get(taxi_id)
                taxi_edge = taxi_edges.get(taxi_id) or ""
                eta = self._estimate_pickup_eta_seconds(
                    traci=traci,
                    taxi_edge=taxi_edge,
                    pickup_edge=pickup_edge,
                    taxi_pos=taxi_pos,
                    pickup_pos=pickup_pos,
                    assumed_speed_mps=assumed_speed_mps,
                )
                if eta is None:
                    continue
                income = float(taxi_income.get(taxi_id, 0.0))
                recent = float(self._get_recent_order_count(control_state, taxi_id, current_time, window_seconds))
                scored.append((taxi_id, float(eta), income, recent, 0.0))

            if not scored:
                continue

            etas = [x[1] for x in scored]
            incomes = [x[2] for x in scored]
            recents = [x[3] for x in scored]

            min_eta, max_eta = min(etas), max(etas)
            min_income, max_income = min(incomes), max(incomes)
            min_recent, max_recent = min(recents), max(recents)

            eta_range = max(max_eta - min_eta, 1e-6)
            income_range = max(max_income - min_income, 1e-6)
            recent_range = max(max_recent - min_recent, 1e-6)

            best_taxi = None
            best_score = None
            best_eta = None
            for taxi_id, eta, income, recent, _ in scored:
                eta_norm = (eta - min_eta) / eta_range
                income_norm = (income - min_income) / income_range
                recent_norm = (recent - min_recent) / recent_range
                score = w_eta * eta_norm + w_income * income_norm + w_recent * recent_norm

                if best_score is None or score < best_score - 1e-12:
                    best_score = score
                    best_taxi = taxi_id
                    best_eta = eta
                elif best_score is not None and abs(score - best_score) <= 1e-12:
                    # Tie-break by ETA
                    if best_eta is None or eta < best_eta:
                        best_taxi = taxi_id
                        best_eta = eta

            if best_taxi is None:
                continue

            decisions.append({"taxi_id": best_taxi, "reservation_ids": [rid]})
            available_taxis.remove(best_taxi)

        return decisions

    def _generate_auto_reposition_decisions(
        self,
        traci: Any,
        active_reservations: Dict[str, Any],
        idle_taxis_set: set,
        control_state: Dict[str, Any],
        config: Dict[str, Any],
        current_time: float,
        reserved_taxis: Optional[set] = None,
        env: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Online reposition heuristic (TAZ imbalance):
        - Targets TAZs with pending demand and low idle supply (critical zones first).
        - Respects a per-taxi cooldown and a max reposition ratio per cycle.
        """
        if not idle_taxis_set:
            return []
        if not self.taz_edges or not self.edge_to_taz:
            return []
        if not bool(config.get("reposition_enabled", True)):
            return []

        interval = float(config.get("reposition_interval_seconds", 300))
        last_reposition_time = float(control_state.get("last_reposition_time", control_state.get("initial_time", 0.0)) or 0.0)
        if current_time - last_reposition_time < interval:
            return []

        max_ratio = float(config.get("auto_reposition_max_ratio", 0.05))
        max_per_cycle = int(config.get("auto_reposition_max_per_cycle", 20))
        cooldown_seconds = float(config.get("auto_reposition_cooldown_seconds", 600))
        critical_only = bool(config.get("auto_reposition_critical_only", False))

        # Skip repositioning if fleet is already tight on idle taxis
        # NOTE: Lowered threshold from 0.15 to 0.02 to allow repositioning even with high utilization
        # Use safe method to avoid protocol corruption
        all_taxis = self._safe_get_taxi_fleet(traci, state=-1, env=env)
        if all_taxis:
            idle_ratio = float(len(idle_taxis_set)) / max(len(all_taxis), 1)
            min_idle_ratio = float(config.get("min_idle_taxis_ratio", 0.02))
            # Allow at least 1 taxi to reposition even when idle ratio is very low
            if idle_ratio < min_idle_ratio and len(idle_taxis_set) < 1:
                return []

        reserved_taxis = set(reserved_taxis or set())

        # Candidate taxis: idle, not reserved for dispatch, and not recently repositioned
        taxi_last_reposition = control_state.setdefault("taxi_last_reposition_time", {})
        candidates: List[str] = []
        for taxi_id in idle_taxis_set:
            if taxi_id in reserved_taxis:
                continue
            last = float(taxi_last_reposition.get(taxi_id, -1e18))
            if current_time - last < cooldown_seconds:
                continue
            candidates.append(taxi_id)

        if not candidates:
            return []

        # Demand per TAZ from pending reservations
        demand: Dict[str, int] = {}
        for rid, res in (active_reservations or {}).items():
            state = getattr(res, "state", None)
            if self._reservation_is_picked_up(state):
                continue
            # Handle both SUMO and Simulated reservations
            if isinstance(res, SimulatedReservation):
                from_edge = res.from_edge
            else:
                from_edge = getattr(res, "fromEdge", None)
            taz_id = self.edge_to_taz.get(from_edge)
            if not taz_id:
                continue
            demand[taz_id] = int(demand.get(taz_id, 0)) + 1

        if not demand:
            return []

        # Supply per TAZ from idle taxis
        supply: Dict[str, int] = {}
        taxi_taz: Dict[str, str] = {}
        for taxi_id in candidates:
            try:
                edge = traci.vehicle.getRoadID(taxi_id)
            except Exception:
                continue
            taz_id = self.edge_to_taz.get(edge)
            if not taz_id:
                continue
            taxi_taz[taxi_id] = taz_id
            supply[taz_id] = int(supply.get(taz_id, 0)) + 1

        if not taxi_taz:
            return []

        critical_taz = [(taz, d) for taz, d in demand.items() if d > 0 and int(supply.get(taz, 0)) == 0]
        critical_taz.sort(key=lambda x: x[1], reverse=True)

        target_taz_list: List[str] = [t for t, _ in critical_taz]
        if not target_taz_list and not critical_only:
            ratios = []
            for taz, d in demand.items():
                s = int(supply.get(taz, 0))
                ratios.append((taz, float(d) / float(s + 1)))
            ratios.sort(key=lambda x: x[1], reverse=True)
            top_n = int(config.get("auto_reposition_top_taz", 5))
            target_taz_list = [t for t, _ in ratios[: max(0, top_n)]]

        if not target_taz_list:
            return []

        # How many taxis to reposition this cycle
        n = int(len(candidates) * max_ratio) if max_ratio > 0 else 0
        n = max(0, min(n, max_per_cycle, len(candidates)))
        if n <= 0:
            return []

        # Prefer moving taxis from surplus zones (high supply, low demand)
        def surplus_score(taxi_id: str) -> int:
            taz = taxi_taz.get(taxi_id)
            if not taz:
                return -10**9
            return int(supply.get(taz, 0)) - int(demand.get(taz, 0))

        candidates_sorted = sorted(candidates, key=surplus_score, reverse=True)
        selected_taxis = candidates_sorted[:n]

        decisions: List[Dict[str, Any]] = []
        for idx, taxi_id in enumerate(selected_taxis):
            target_taz = target_taz_list[idx % len(target_taz_list)]
            decisions.append({"taxi_id": taxi_id, "target_taz": target_taz})
            taxi_last_reposition[taxi_id] = float(current_time)

        control_state["last_reposition_time"] = float(current_time)
        return decisions

    def load_taz_definitions(self, taz_file_path: str) -> bool:
        """
        Load TAZ definitions from XML file.
        
        Args:
            taz_file_path: Path to the .taz.xml file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(taz_file_path):
                print(f"Warning: TAZ file not found: {taz_file_path}")
                return False
                
            tree = ET.parse(taz_file_path)
            root = tree.getroot()
            
            self.taz_edges = {}
            self.edge_to_taz = {}
            
            for taz in root.findall('taz'):
                taz_id = taz.get('id')
                edges_str = taz.get('edges', '')
                edges = edges_str.split()
                
                if taz_id and edges:
                    self.taz_edges[taz_id] = edges
                    for edge in edges:
                        self.edge_to_taz[edge] = taz_id
                        
            # Initialize stats for loaded TAZs
            for taz_id in self.taz_edges:
                self.taz_stats[taz_id] = {
                    "demand_history": [],  # List of (time, count) tuples
                    "match_history": [],   # List of (time, count) tuples
                    "current_demand": 0,
                    "current_matches": 0
                }
                
            print(f"Loaded {len(self.taz_edges)} TAZ definitions from {taz_file_path}")
            return True
            
        except Exception as e:
            print(f"Error loading TAZ definitions: {e}")
            traceback.print_exc()
            return False

    def update_taz_stats(self, env: Any, current_time: float) -> None:
        """
        Update TAZ statistics (demand and matches).
        
        Args:
            env: SUMOEnv instance
            current_time: Current simulation time
        """
        if not self.taz_edges:
            return
            
        # Get new reservations to track demand
        reservations = self.get_pending_reservations(env)
        if "reservations" in reservations:
            for res in reservations["reservations"]:
                # We need a way to identify NEW demand. 
                # Since get_pending_reservations returns all currently pending, we might overcount
                # if we just add them all. Ideally we track by reservation ID.
                # However, for now let's just use a simplified approach:
                # In a real system we'd listen to 'reservation created' events.
                # Here, we can perhaps just check 'reservation_time'.
                pass
        
        # NOTE: A robust implementation requires tracking state between steps to only count NEW events.
        # For simplicity in this iteration, we will rely on aggregated snapshots or 
        # let the LLM see the instantaneous "pending count" per TAZ.
        
        # But we DO need to populate 'supply' updates.
        pass

    def get_taz_stats(self, env: Any) -> Dict[str, Any]:
        """
        Get current TAZ statistics including supply, demand, and matches.
        
        Args:
            env: SUMOEnv instance
            
        Returns:
            Dictionary of TAZ stats
        """
        stats = {}
        
        # Base init
        for taz_id in self.taz_edges:
            stats[taz_id] = {
                "demand": 0, # Pending reservations originating here
                "supply": 0, # Idle taxis currently here
                "total_taxis": 0, # Total taxis here (any state)
                "incoming_supply": 0 # Taxis moving to this TAZ
            }
            
        if not self.taz_edges:
            return stats

        # Update Supply (Taxis)
        fleet = self.get_taxi_fleet_state(env)
        if "taxi_details" in fleet:
            for taxi_id, details in fleet["taxi_details"].items():
                if isinstance(details, dict):
                    edge = details.get("current_edge")
                    taz_id = self.edge_to_taz.get(edge)
                    
                    if taz_id:
                        stats[taz_id]["total_taxis"] += 1
                        if details.get("state") == "0" or details.get("state") == "idle": # 0 is idle
                            stats[taz_id]["supply"] += 1
        
        # Update Demand (Reservations)
        reservations = self.get_pending_reservations(env)
        if "reservations" in reservations:
            for res in reservations["reservations"]:
                origin_edge = res.get("from_edge")
                taz_id = self.edge_to_taz.get(origin_edge)
                if taz_id:
                    stats[taz_id]["demand"] += 1
                    
        return stats

    
    def get_default_config(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate default configuration for taxi scheduling.
        
        Args:
            env: SUMOEnv instance with initialized TraCI connection
            
        Returns:
            Dictionary with default taxi scheduling configuration
        """
        config = {
            # Cleanup-only mode: if True, run stale reservation cleanup and skip dispatch/reposition.
            # dispatch_enabled=False has the same effect.
            "cleanup_only": False,
            "dispatch_enabled": True,
            "dispatch_strategy": "llm",  # "llm", "greedy", or online variants like "online_eta"
            # If dispatch_strategy is "llm" but no dispatch_decisions are provided,
            # optionally fall back to a greedy, fairness-aware dispatcher.
            "fallback_to_greedy_dispatch": True,
            # Greedy dispatch tuning (used when dispatch_strategy=="greedy" OR fallback triggers)
            "greedy_distance_tie_threshold": 50.0,  # meters; treat taxis within this margin as "similar distance"
            "greedy_score_weights": {"distance": 0.7, "income": 0.2, "recent_orders": 0.1},
            # Online dispatch tuning (used when dispatch_strategy in {"online","online_eta","online_matching_eta"})
            # Cost = w_eta * ETA_to_pickup + w_income * income + w_recent * recent_orders (normalized per reservation)
            "online_dispatch_cost_weights": {"eta": 0.7, "income": 0.2, "recent_orders": 0.1},
            "online_dispatch_candidate_k": 15,          # bound ETA computations per reservation
            "online_dispatch_max_reservations": 25,     # bound per-cycle workload
            "online_dispatch_assumed_speed_mps": 8.0,   # fallback when route ETA is unavailable
            "recent_order_window_seconds": 1800,  # "recent" window for order-count fairness
            # Income model (proxy): fare = base + travel_time * per_second
            "fare_base": 3.0,
            "fare_per_second": 0.01,
            "reposition_enabled": True,
            # Reposition strategy:
            # - "llm": only follow explicit reposition_decisions
            # - "auto_taz_balance"/"taz_balance": generate online reposition decisions each interval
            "reposition_strategy": "llm",
            "fallback_to_auto_reposition": True,
            "reposition_interval_seconds": 120,  # Reposition check interval
            "min_idle_taxis_ratio": 0.02,  # Minimum ratio of idle taxis before repositioning (lowered from 0.15)
            # Auto-reposition tuning (when reposition_strategy is auto/online)
            "auto_reposition_max_ratio": 0.10,          # fraction of idle taxis per cycle
            "auto_reposition_max_per_cycle": 20,        # hard cap per cycle
            "auto_reposition_cooldown_seconds": 180,    # per-taxi cooldown
            "auto_reposition_critical_only": False,      # only demand>0 & supply==0 TAZs
            "auto_reposition_top_taz": 5,               # used when critical_only=False
            "high_demand_zones": [],  # Will be populated based on demand analysis
            "last_update_time": 0,
            # Dispatch retry safety: prevent stale dispatch decisions from being retried forever
            # (especially important when running with checkpoint save/load).
            "dispatch_retry_max_seconds": 900,   # Drop queued dispatch after this age (s)
            "dispatch_retry_max_attempts": 200,  # Drop queued dispatch after this many cycles
            # Avoid dispatching a taxi in the same step it was spawned (SUMO 1.20.0 issue #15016)
            "min_dispatch_taxi_age": 1.0,  # seconds
            # Reservation polling / decode error resilience
            "reservation_poll_interval_seconds": 15.0,  # Min seconds between full reservation fetches
            "reservation_decode_error_cooldown_seconds": 10.0,  # Pause fetching after decode errors
            "reservation_decode_error_log_interval_seconds": 10.0,  # Throttle decode error logs
            # Dispatch workload caps
            "max_dispatches_per_cycle": 200,  # Max pending dispatch entries processed per cycle
            "max_route_checks_per_cycle": 200,  # Max route computations per cycle
            # Load shedding / backoff under high load (helps avoid SUMO instability)
            "load_shedding_enabled": True,
            "load_shedding_pending_threshold": 400,
            "load_shedding_no_route_threshold": 25,
            "load_shedding_replenish_no_route_threshold": 25,
            "load_shedding_backoff_factor": 0.5,
            "load_shedding_min_dispatches": 20,
            "load_shedding_min_replenish": 10,
            "load_shedding_cooldown_seconds": 300,
            "load_shedding_next_action_time": 120.0,
            # Reservation state handling
            # NOTE: getTaxiReservations(0) flips reservations into state=2 after the first call.
            # If we skip state=2 by default, dispatch starves after the first cycle.
            "skip_retrieved_reservations": False,
            # Hard safety override to always skip retrieved reservations (use only if needed).
            "force_skip_retrieved_reservations": False,
            # Auto-replenish: automatically add new taxis when fleet size decreases
            "auto_replenish_enabled": True,       # Enable automatic taxi replenishment
            "max_replenish_per_cycle": 50,        # Maximum taxis to add per control cycle
            # "target_fleet_size": N,             # Target fleet size (defaults to initial fleet size)
            # Initial fleet size: Set on first cycle, preserved across checkpoints
            # This is the authoritative source for fleet size after checkpoint load
            "initial_fleet_size": None,
            # Keepalive settings - DISABLED by default to prevent SUMO crashes
            # Only enable if using "stop" idle algorithm and taxis are being removed
            "keepalive_enabled": False,           # Disabled by default - can crash SUMO
            "keepalive_interval_seconds": 120,    # Periodic refresh interval for idle taxis
            "max_keepalive_per_cycle": 50,        # Limit taxis refreshed per cycle to avoid overwhelming SUMO
            # Reservation cleanup settings - reduced from 600 to 300 to prevent accumulation
            # that can trigger SUMO crash (GitHub issue #9805)
            "max_retrieved_reservation_age": 300, # Max seconds a reservation can stay in 'retrieved' state
            # ========================================================================
            # Simulated Taxi System Settings
            # Use ordinary vehicle control API instead of SUMO taxi API
            # ========================================================================
            "use_simulated_taxi_system": False,    # Enable simulated taxi mode
            "reservation_file_path": None,         # Path to persons.taxi.xml file
            "taxi_fleet_file_path": None,          # Path to taxi_fleet.rou.xml file
            "simulated_taxi_vtype_filter": "taxi", # Vehicle type filter for adopting existing vehicles
            "adopt_existing_vehicles": True,       # Adopt existing taxi vehicles in simulation
            "arrival_distance_threshold": 50.0,    # Distance threshold for arrival detection (meters)
        }
        
        if env is not None and hasattr(env, 'traci_conn') and env.traci_conn is not None:
            try:
                # Get initial fleet info using safe method
                all_taxis = self._safe_get_taxi_fleet(env.traci_conn, state=-1, env=env)
                config["fleet_size"] = len(all_taxis)
                print(f"Initialized taxi scheduling with fleet size: {len(all_taxis)}")
            except Exception as e:
                print(f"Warning: Could not get taxi fleet info: {e}")
                config["fleet_size"] = 0
        else:
            config["fleet_size"] = 0
        
        return config
    
    def validate_config(
        self, 
        config: Dict[str, Any], 
        reference_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate taxi scheduling configuration.
        
        Args:
            config: Configuration dictionary to validate
            reference_config: Optional reference configuration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        # Validate dispatch_decisions if present
        dispatch_decisions = config.get("dispatch_decisions", [])
        if dispatch_decisions:
            if not isinstance(dispatch_decisions, list):
                return False, "dispatch_decisions must be a list"
            
            for i, decision in enumerate(dispatch_decisions):
                if not isinstance(decision, dict):
                    return False, f"dispatch_decisions[{i}] must be a dictionary"
                if "taxi_id" not in decision:
                    return False, f"dispatch_decisions[{i}] missing 'taxi_id'"
                if "reservation_ids" not in decision:
                    return False, f"dispatch_decisions[{i}] missing 'reservation_ids'"
                if not isinstance(decision["reservation_ids"], list):
                    return False, f"dispatch_decisions[{i}] 'reservation_ids' must be a list"
        
        # Validate reposition_decisions if present
        reposition_decisions = config.get("reposition_decisions", [])
        if reposition_decisions:
            if not isinstance(reposition_decisions, list):
                return False, "reposition_decisions must be a list"
            
            for i, decision in enumerate(reposition_decisions):
                if not isinstance(decision, dict):
                    return False, f"reposition_decisions[{i}] must be a dictionary"
                if "taxi_id" not in decision:
                    return False, f"reposition_decisions[{i}] missing 'taxi_id'"
                if "target_edge" not in decision and "target_taz" not in decision:
                    return False, f"reposition_decisions[{i}] must specify 'target_edge' or 'target_taz'"
        
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
        Apply taxi scheduling control logic.
        
        This function processes dispatch and repositioning decisions from the LLM agent.
        
        Args:
            env: SUMOEnv instance with TraCI connection
            config: Taxi scheduling configuration with decisions
            current_time: Current simulation time
            control_state: Current control state (maintained across steps)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dictionary containing:
                - control_state: Updated control state
                - actions: Dictionary with dispatch and reposition actions
                - next_action_time: Time until next control action
        """
        # Initialize control state if not provided
        if control_state is None:
            control_state = self._initialize_control_state(env, config, current_time)

        # ========================================================================
        # Ensure TAZ definitions are loaded BEFORE simulated taxi init (needed for spawn edges)
        if not self.taz_edges and not getattr(self, "_taz_load_attempted", False):
            self._taz_load_attempted = True
            taz_path = None
            search_dirs = []
            if hasattr(env, "config_path") and env.config_path:
                cfg_dir = os.path.dirname(os.path.abspath(env.config_path))
                search_dirs.append(cfg_dir)
                # Also try net-file's directory (actual scenario dir)
                try:
                    import xml.etree.ElementTree as _ET
                    _tree = _ET.parse(env.config_path)
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
                candidate = os.path.join(sd, "districts.taz.xml")
                if os.path.exists(candidate):
                    taz_path = candidate
                    break
            if taz_path:
                self.load_taz_definitions(taz_path)
                print(f"TaxiSchedulingModule: TAZ edges loaded={len(self.taz_edges)} from {taz_path}")
            else:
                print(f"TaxiSchedulingModule: TAZ file not found in {search_dirs}")

        # Simulated Taxi System Initialization and Update
        # ========================================================================
        use_simulated = bool(config.get("use_simulated_taxi_system", False))
        # Fallback: check env-level flag if config doesn't have it
        if not use_simulated and hasattr(env, 'dic_traffic_env_conf'):
            use_simulated = bool(env.dic_traffic_env_conf.get("USE_SIMULATED_TAXI_SYSTEM", False))
            if use_simulated:
                # Propagate into config so subsequent checks work
                config["use_simulated_taxi_system"] = True
                # Also propagate file paths if missing - search config dir + net-file dir
                if not config.get("reservation_file_path") or not config.get("taxi_fleet_file_path"):
                    fb_dirs = []
                    cfg_path = getattr(env, 'config_path', '') or ''
                    if cfg_path:
                        fb_dirs.append(os.path.dirname(os.path.abspath(cfg_path)))
                        try:
                            import xml.etree.ElementTree as _ET2
                            _t2 = _ET2.parse(cfg_path)
                            _ne2 = _t2.find(".//net-file")
                            if _ne2 is not None:
                                _nv2 = _ne2.get("value", "")
                                if os.path.isabs(_nv2):
                                    _nd2 = os.path.dirname(_nv2)
                                else:
                                    _nd2 = os.path.dirname(os.path.join(fb_dirs[0], _nv2))
                                if _nd2 and _nd2 not in fb_dirs:
                                    fb_dirs.append(_nd2)
                        except Exception:
                            pass
                    for sd in fb_dirs:
                        pf = os.path.join(sd, "persons.taxi.xml")
                        ff = os.path.join(sd, "taxi_fleet.rou.xml")
                        if os.path.exists(pf) and not config.get("reservation_file_path"):
                            config["reservation_file_path"] = pf
                        if os.path.exists(ff) and not config.get("taxi_fleet_file_path"):
                            config["taxi_fleet_file_path"] = ff
                print(f"[SimTaxi] Detected USE_SIMULATED_TAXI_SYSTEM from env, propagated to config")
        # if use_simulated and not self._simulated_system_initialized:
        #     print(f"[SimTaxi-DEBUG] apply_control entry: use_simulated={use_simulated}, "
        #           f"initialized={self._simulated_system_initialized}, "
        #           f"taxis={len(self._simulated_taxis)}, module_id={id(self)}")
        if use_simulated:
            # Initialize simulated system if not already done
            if not self._simulated_system_initialized:
                self._init_simulated_taxi_system(env, config, control_state, current_time)

            # Activate pending reservations based on current time
            newly_activated = self._activate_pending_reservations(current_time, config)
            # if newly_activated:
            #     print(f"[SimTaxi-DEBUG] Activated {len(newly_activated)} reservations at t={current_time:.0f}s, "
            #           f"total active={len(self._simulated_reservations)}, "
            #           f"pending_from_file={len(self._pending_reservations_from_file)}")
            if newly_activated:
                control_state["_newly_activated_reservations"] = len(newly_activated)

            # Update taxi trip progress (check arrivals)
            trip_events = self._update_taxi_trip_progress(
                env.traci_conn, current_time, control_state, config
            )
            if trip_events.get("pickups") or trip_events.get("dropoffs"):
                control_state["_last_trip_events"] = trip_events

        # NOTE: track_reservation_events() is called AFTER caching reservations (see below)
        # to ensure it uses fresh data, not stale cache from previous cycle

        # Ensure pending_dispatches exists in control_state
        if "pending_dispatches" not in control_state:
            control_state["pending_dispatches"] = []
            
        # Ensure completed_dispatches exists in control_state
        if "completed_dispatches" not in control_state:
            control_state["completed_dispatches"] = []

        # Ensure completed_repositions exists in control_state (avoid repeating the same changeTarget every step)
        if "completed_repositions" not in control_state:
            control_state["completed_repositions"] = []

        load_shedding_active = False

        actions = {
            "dispatch": [],
            "reposition": []
        }

        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            print("Warning: TraCI connection not available")
            return {
                "control_state": control_state,
                "actions": actions,
                "next_action_time": 60.0
            }

        # Check if TraCI connection is healthy (not corrupted)
        if hasattr(env, 'is_traci_healthy') and not env.is_traci_healthy():
            print("Warning: TraCI connection is unhealthy, skipping taxi control")
            return {
                "control_state": control_state,
                "actions": actions,
                "next_action_time": 60.0
            }

        # CRITICAL: Always use env.traci_conn directly, never cache it in a local variable
        # After checkpoint/metrics reset, the traci_conn reference may change
        # Using env.traci_conn ensures we always have the current valid connection

        # Validate TraCI connection before proceeding
        try:
            sumo_time = env.traci_conn.simulation.getTime()
        except (traci.TraCIException, traci.exceptions.FatalTraCIError) as e:
            print(f"Warning: TraCI connection closed: {e}")
            if hasattr(env, 'mark_traci_unhealthy'):
                env.mark_traci_unhealthy()
            return {
                "control_state": control_state,
                "actions": actions,
                "next_action_time": 60.0
            }
        except UnicodeDecodeError as e:
            print(f"CRITICAL: TraCI protocol corruption in apply_control: {e}")
            if hasattr(env, 'mark_traci_unhealthy'):
                env.mark_traci_unhealthy()
            return {
                "control_state": control_state,
                "actions": actions,
                "next_action_time": 60.0
            }
        except Exception as e:
            print(f"Warning: Unexpected error validating TraCI: {e}")
            pass

        # (TAZ loading moved earlier, before simulated taxi init)

        # Dispatch retry bounds (guardrails against infinite retries)
        # Enforce defaults as minimums to avoid premature drops.
        dispatch_retry_max_seconds = float(config.get("dispatch_retry_max_seconds", 900))
        dispatch_retry_max_attempts = int(config.get("dispatch_retry_max_attempts", 200))
        dispatch_retry_max_seconds = max(dispatch_retry_max_seconds, 900.0)
        dispatch_retry_max_attempts = max(dispatch_retry_max_attempts, 200)

        # ========================================================================
        # CRITICAL FIX: Cache TraCI data ONCE at the start of control cycle
        # This prevents repeated TraCI calls that can cause protocol buffer corruption
        # NOTE: Always pass env to safe methods so they can use env.traci_conn directly
        # ========================================================================

        # Cache active reservations using safe method with retry logic + poll interval/cooldown
        self._current_sim_time = current_time
        self._decode_error_log_interval_seconds = float(
            config.get("reservation_decode_error_log_interval_seconds", 10.0)
        )
        poll_interval = float(config.get("reservation_poll_interval_seconds", 5.0))
        cooldown_seconds = float(config.get("reservation_decode_error_cooldown_seconds", 10.0))

        last_fetch_time = control_state.get("_last_reservation_fetch_time")
        fetch_disabled_until = control_state.get("_reservation_fetch_disabled_until")
        last_good_reservations = control_state.get("_last_good_reservations")

        use_cached = False
        if fetch_disabled_until is not None and current_time < float(fetch_disabled_until):
            use_cached = True
        elif last_fetch_time is not None and poll_interval > 0:
            if (current_time - float(last_fetch_time)) < poll_interval:
                use_cached = True

        if use_cached:
            if isinstance(last_good_reservations, list):
                cached_reservation_list = list(last_good_reservations)
            else:
                cached_reservation_list = []
        else:
            cached_reservation_list = self._safe_get_taxi_reservations(env.traci_conn, env=env)
            if getattr(self, "_last_decode_error", False):
                if cooldown_seconds > 0:
                    control_state["_reservation_fetch_disabled_until"] = current_time + cooldown_seconds
                if isinstance(last_good_reservations, list):
                    cached_reservation_list = list(last_good_reservations)
            else:
                control_state["_last_reservation_fetch_time"] = current_time
                control_state["_last_good_reservations"] = list(cached_reservation_list)
                control_state.pop("_reservation_fetch_disabled_until", None)
        active_reservations: Dict[str, Any] = {}
        for res in cached_reservation_list:
            active_reservations[self._normalize_reservation_id(res.id)] = res
        active_reservation_ids = set(active_reservations.keys())

        # DEBUG: trace reservation and taxi counts for dispatch
        # if use_simulated and (len(active_reservations) > 0 or len(self._simulated_reservations) > 0):
        #     print(f"[SimTaxi-DEBUG] t={current_time:.0f}s: active_reservations={len(active_reservations)}, "
        #           f"simulated_reservations={len(self._simulated_reservations)}, "
        #           f"simulated_taxis={len(self._simulated_taxis)}, "
        #           f"cached_list={len(cached_reservation_list)}")

        # Store cached data in control_state for use by other methods in this cycle
        control_state["_cached_reservations"] = cached_reservation_list
        control_state["_cached_reservation_dict"] = active_reservations

        # TRACK EVENTS (Metrics Logic Hook) - called AFTER caching to use fresh data
        self.track_reservation_events(env, control_state, None)

        # Clean up stale reservations (state 2 stuck for too long) to prevent SUMO crashes
        self._cleanup_stale_reservations(
            env=env,
            control_state=control_state,
            config=config,
            current_time=current_time,
            cached_reservations=cached_reservation_list
        )

        cleanup_only = bool(config.get("cleanup_only", False)) or not bool(config.get("dispatch_enabled", True))
        if cleanup_only:
            if not control_state.get("_cleanup_only_logged"):
                print("[TaxiScheduling] cleanup_only enabled: skipping dispatch/reposition; stale reservation cleanup active")
                control_state["_cleanup_only_logged"] = True
            control_state["last_update_time"] = current_time
            return {
                "control_state": control_state,
                "actions": actions,
                "next_action_time": 60.0
            }

        # Cache idle taxis using safe method
        idle_taxis_set = set(self._safe_get_taxi_fleet(env.traci_conn, state=0, env=env))
        dispatch_idle_taxis_set = {
            taxi_id
            for taxi_id in idle_taxis_set
            if self._taxi_is_mature_for_dispatch(taxi_id, current_time, control_state, config)
        }

        # Ensure fairness tracking structures exist and keep key parameters in sync
        control_state.setdefault("reservation_to_taxi", {})
        control_state.setdefault("taxi_income", {})
        control_state.setdefault("taxi_recent_order_times", {})
        control_state.setdefault("taxi_spawn_times", {})
        try:
            control_state["fare_base"] = float(config.get("fare_base", control_state.get("fare_base", 3.0)))
            control_state["fare_per_second"] = float(config.get("fare_per_second", control_state.get("fare_per_second", 0.01)))
            control_state["recent_order_window_seconds"] = float(
                config.get("recent_order_window_seconds", control_state.get("recent_order_window_seconds", 1800))
            )
        except Exception:
            pass

        # Best-effort reservation->taxi mapping sync (helps with checkpoint boundary cases)
        self._sync_reservation_to_taxi_mapping(env.traci_conn, control_state, active_reservation_ids, env=env)
        
        # 1. Add NEW dispatch decisions from config to pending queue
        current_pending_signatures = {
            (
                d["taxi_id"],
                tuple(self._normalize_reservation_id(r) for r in d.get("reservation_ids", []))
            )
            for d in control_state["pending_dispatches"]
        }
        
        # We store completed signatures as list of lists/tuples, need consistent format
        # Let's use set of tuples for fast lookup locally
        completed_signatures = {
            (
                d["taxi_id"],
                tuple(self._normalize_reservation_id(r) for r in d.get("reservation_ids", []))
            )
            for d in control_state["completed_dispatches"]
        }

        dispatch_decisions = config.get("dispatch_decisions", []) or []
        dispatch_strategy = str(config.get("dispatch_strategy", "llm")).lower()

        online_dispatch_strategies = {
            "online",
            "online_eta",
            "online_matching_eta",
        }

        # Avoid re-dispatching reservations that are already queued for dispatch attempts
        reservations_in_flight = {
            self._normalize_reservation_id(r)
            for d in (control_state.get("pending_dispatches") or [])
            for r in (d.get("reservation_ids") or [])
        }

        if dispatch_strategy in online_dispatch_strategies:
            # Treat config-provided dispatch_decisions as optional high-priority overrides,
            # then fill the remaining demand online using ETA+fairness.
            normalized_overrides: List[Dict[str, Any]] = []
            used_taxis: set = set()
            used_reservations: set = set()
            for decision in dispatch_decisions:
                taxi_id = decision.get("taxi_id")
                reservation_ids = [self._normalize_reservation_id(r) for r in (decision.get("reservation_ids") or [])]
                if not taxi_id or not reservation_ids:
                    continue
                normalized_overrides.append({"taxi_id": taxi_id, "reservation_ids": reservation_ids})
                used_taxis.add(taxi_id)
                used_reservations.update(reservation_ids)

            # Avoid assigning taxis already queued for dispatch attempts
            taxis_in_flight = {
                d.get("taxi_id")
                for d in (control_state.get("pending_dispatches") or [])
                if isinstance(d, dict) and d.get("taxi_id")
            }
            candidate_idle_taxis = set(dispatch_idle_taxis_set) - used_taxis - taxis_in_flight

            remaining_reservations = {
                rid: res for rid, res in active_reservations.items() if rid not in used_reservations
            }
            auto_decisions = self._generate_online_eta_dispatch_decisions(
                traci=env.traci_conn,
                active_reservations=remaining_reservations,
                idle_taxis_set=candidate_idle_taxis,
                control_state=control_state,
                config=config,
                current_time=current_time,
                exclude_reservation_ids=reservations_in_flight,
            )

            dispatch_decisions = normalized_overrides + auto_decisions

            if reservations_in_flight:
                filtered = []
                for d in dispatch_decisions:
                    rids = [self._normalize_reservation_id(r) for r in d.get("reservation_ids", [])]
                    if any(r in reservations_in_flight for r in rids):
                        continue
                    filtered.append(d)
                dispatch_decisions = filtered

        elif dispatch_strategy == "greedy" or (
            dispatch_strategy == "llm"
            and bool(config.get("fallback_to_greedy_dispatch", True))
            and not dispatch_decisions
        ):
            greedy_decisions = self._generate_greedy_dispatch_decisions(
                traci=env.traci_conn,
                active_reservations=active_reservations,
                idle_taxis_set=dispatch_idle_taxis_set,
                control_state=control_state,
                config=config,
                current_time=current_time,
                exclude_reservation_ids=reservations_in_flight,
            )
            if reservations_in_flight:
                filtered = []
                for d in greedy_decisions:
                    rids = [self._normalize_reservation_id(r) for r in d.get("reservation_ids", [])]
                    if any(r in reservations_in_flight for r in rids):
                        continue
                    filtered.append(d)
                dispatch_decisions = filtered
            else:
                dispatch_decisions = greedy_decisions
        
        for decision in dispatch_decisions:
            taxi_id = decision.get("taxi_id")
            reservation_ids = [self._normalize_reservation_id(r) for r in decision.get("reservation_ids", [])]
            
            if taxi_id and reservation_ids:
                signature = (taxi_id, tuple(reservation_ids))
                
                # Check if already pending OR already completed
                if signature not in current_pending_signatures and signature not in completed_signatures:
                    control_state["pending_dispatches"].append({
                        "taxi_id": taxi_id,
                        "reservation_ids": reservation_ids,
                        "added_time": current_time,
                        "attempts": 0
                    })
                    current_pending_signatures.add(signature)

        # Load shedding / backoff when the system is under heavy load
        if bool(config.get("load_shedding_enabled", False)):
            pending_count = len(control_state.get("pending_dispatches") or [])
            last_no_route = int(control_state.get("_last_dispatch_no_route_count", 0))
            last_replenish_no_route = int(control_state.get("_last_replenish_no_route_count", 0))
            pending_threshold = int(config.get("load_shedding_pending_threshold", 400))
            no_route_threshold = int(config.get("load_shedding_no_route_threshold", 25))
            replenish_no_route_threshold = int(config.get("load_shedding_replenish_no_route_threshold", 25))
            overload_now = (
                pending_count >= pending_threshold
                or last_no_route >= no_route_threshold
                or last_replenish_no_route >= replenish_no_route_threshold
            )
            if overload_now:
                cooldown = float(config.get("load_shedding_cooldown_seconds", 300))
                control_state["_load_shedding_until"] = float(current_time) + max(0.0, cooldown)
                control_state["_load_shedding_reason"] = {
                    "pending": pending_count,
                    "last_no_route": last_no_route,
                    "last_replenish_no_route": last_replenish_no_route,
                }
            load_shedding_until = float(control_state.get("_load_shedding_until", 0.0))
            load_shedding_active = float(current_time) < load_shedding_until
            prev_active = bool(control_state.get("_load_shedding_active", False))
            if load_shedding_active and not prev_active:
                print(f"[TaxiScheduling] Load shedding activated (pending={pending_count})")
            elif not load_shedding_active and prev_active:
                print("[TaxiScheduling] Load shedding cleared")
            control_state["_load_shedding_active"] = load_shedding_active

        # 2. Process Passing Dispatches (Retry Logic)
        remaining_dispatches = []

        # Cache person IDs ONCE before the loop to avoid repeated TraCI calls
        cached_person_ids = None
        person_id_check_failed = False
        try:
            cached_person_ids = set(env.traci_conn.person.getIDList())
        except Exception as e:
            # If we can't get person list, defer dispatch for stability (avoid missing-person crash)
            cached_person_ids = None
            person_id_check_failed = True
        dispatch_no_route_count = 0

        min_dispatch_taxi_age = float(config.get("min_dispatch_taxi_age", 1.0))
        max_dispatches = int(config.get("max_dispatches_per_cycle", 200))
        if load_shedding_active:
            backoff = float(config.get("load_shedding_backoff_factor", 0.5))
            min_dispatches = int(config.get("load_shedding_min_dispatches", 20))
            max_dispatches = max(min_dispatches, int(max_dispatches * backoff))
        max_route_checks = int(config.get("max_route_checks_per_cycle", 200))
        route_checks_remaining = None if max_route_checks <= 0 else max_route_checks
        deferred_dispatches = []
        pending_dispatches = list(control_state["pending_dispatches"])
        if max_dispatches > 0 and len(pending_dispatches) > max_dispatches:
            deferred_dispatches = pending_dispatches[max_dispatches:]
            pending_dispatches = pending_dispatches[:max_dispatches]
        for pending in pending_dispatches:
            taxi_id = pending["taxi_id"]
            reservation_ids = [self._normalize_reservation_id(r) for r in pending.get("reservation_ids", [])]
            pending["reservation_ids"] = reservation_ids
            pending["attempts"] += 1
            
            success = False
            drop = False
            
            # Drop stale retries (prevents dispatching phantom reservation IDs forever)
            added_time = pending.get("added_time", current_time)
            age_seconds = float(current_time - float(added_time)) if added_time is not None else 0.0
            if pending["attempts"] > dispatch_retry_max_attempts or age_seconds > dispatch_retry_max_seconds:
                drop = True
                actions["dispatch"].append({
                    "taxi_id": taxi_id,
                    "reservations": reservation_ids,
                    "success": False,
                    "reason": f"Dropped stale dispatch (age={age_seconds:.0f}s, attempts={pending['attempts']})",
                    "attempts": pending["attempts"],
                })

            # Guard: avoid dispatching a taxi immediately after spawn
            if not drop and not self._taxi_is_mature_for_dispatch(taxi_id, current_time, control_state, config):
                actions["dispatch"].append({
                    "taxi_id": taxi_id,
                    "reservations": reservation_ids,
                    "success": False,
                    "reason": f"Taxi too new for dispatch (min_age={min_dispatch_taxi_age:.0f}s)",
                    "attempts": pending["attempts"],
                })
                remaining_dispatches.append(pending)
                continue
            # Pre-check taxi idleness: dispatchTaxi generally expects an idle taxi
            if not drop and taxi_id not in idle_taxis_set:
                actions["dispatch"].append({
                    "taxi_id": taxi_id,
                    "reservations": reservation_ids,
                    "success": False,
                    "reason": "Taxi not idle (queued)",
                    "attempts": pending["attempts"],
                })
            # Pre-check reservation existence: only dispatch to currently active reservations
            elif not drop:
                missing = [r for r in reservation_ids if r not in active_reservation_ids]
                if missing:
                    actions["dispatch"].append({
                        "taxi_id": taxi_id,
                        "reservations": reservation_ids,
                        "success": False,
                        "reason": f"Reservation(s) not active yet (queued): {missing[:3]}",
                        "attempts": pending["attempts"],
                    })
                else:
                    # If we cannot validate person presence, defer dispatch for stability
                    if person_id_check_failed:
                        actions["dispatch"].append({
                            "taxi_id": taxi_id,
                            "reservations": reservation_ids,
                            "success": False,
                            "reason": "Skipped dispatch: person list unavailable",
                            "attempts": pending["attempts"],
                        })
                        remaining_dispatches.append(pending)
                        continue

                    # Guard: skip non-dispatchable reservations (assigned/retrieved/picked up)
                    non_dispatchable = []
                    for r in reservation_ids:
                        res_obj = active_reservations.get(r)
                        if res_obj is None:
                            continue
                        state_val = getattr(res_obj, "state", None)
                        if not self._reservation_is_dispatchable(state_val, config):
                            non_dispatchable.append((r, state_val))
                    if non_dispatchable:
                        preview = ", ".join(
                            f"{rid}:{state}" for rid, state in non_dispatchable[:3]
                        )
                        drop = True
                        actions["dispatch"].append({
                            "taxi_id": taxi_id,
                            "reservations": reservation_ids,
                            "success": False,
                            "reason": f"Dropped dispatch: reservation not dispatchable (state) {preview}",
                            "attempts": pending["attempts"],
                        })
                    else:
                        # CRITICAL FIX: Check if persons in reservation still exist in simulation
                        # This prevents SUMO crash when person was teleported but reservation remains
                        # (Known SUMO bug: GitHub issues #9733, #15016)
                        persons_missing = False
                        missing_person_ids = []

                        # Use cached person IDs (fetched ONCE before the loop)
                        if cached_person_ids is not None:
                            for r in reservation_ids:
                                res_obj = active_reservations.get(r)
                                if res_obj is not None:
                                    person_ids = list(res_obj.persons) if hasattr(res_obj, 'persons') else []
                                    for pid in person_ids:
                                        if pid not in cached_person_ids:
                                            persons_missing = True
                                            missing_person_ids.append(pid)

                        if persons_missing:
                            # Person was teleported/removed - drop this dispatch to prevent crash
                            drop = True
                            actions["dispatch"].append({
                                "taxi_id": taxi_id,
                                "reservations": reservation_ids,
                                "success": False,
                                "reason": f"Dropped dispatch: person(s) no longer in simulation (teleported?): {missing_person_ids[:3]}",
                                "attempts": pending["attempts"],
                            })
                            print(f"[TaxiScheduling] PREVENTED CRASH: Skipped dispatch for reservation with missing persons: {missing_person_ids[:3]}")
                        else:
                            taxi_edge = None
                            try:
                                taxi_edge = env.traci_conn.vehicle.getRoadID(taxi_id)
                                if taxi_edge and taxi_edge.startswith(":"):
                                    taxi_edge = None
                            except Exception:
                                taxi_edge = None

                            if not taxi_edge:
                                actions["dispatch"].append({
                                    "taxi_id": taxi_id,
                                    "reservations": reservation_ids,
                                    "success": False,
                                    "reason": "Skipped dispatch: missing taxi edge",
                                    "attempts": pending["attempts"],
                                })
                                remaining_dispatches.append(pending)
                                continue

                            route_cache: Dict[Tuple[str, str], Optional[bool]] = {}
                            route_budget_exhausted = False

                            def _route_exists(start_edge: Optional[str], end_edge: Optional[str]) -> Optional[bool]:
                                nonlocal route_checks_remaining, route_budget_exhausted
                                if not start_edge or not end_edge:
                                    return None
                                if start_edge.startswith(":") or end_edge.startswith(":"):
                                    return False
                                if route_checks_remaining is not None:
                                    if route_checks_remaining <= 0:
                                        route_budget_exhausted = True
                                        return None
                                    route_checks_remaining -= 1
                                key = (start_edge, end_edge)
                                if key in route_cache:
                                    return route_cache[key]
                                try:
                                    # Use findRoute without vType to match _execute_simulated_dispatch behavior
                                    route_result = env.traci_conn.simulation.findRoute(start_edge, end_edge)
                                    has_edges = bool(route_result and getattr(route_result, "edges", None))
                                    if not has_edges and use_simulated:
                                        # Fallback 1: try alt end edge
                                        alt_end = _find_nearest_reachable(end_edge)
                                        if alt_end and alt_end != end_edge:
                                            r2 = env.traci_conn.simulation.findRoute(start_edge, alt_end)
                                            has_edges = bool(r2 and getattr(r2, "edges", None))
                                    if not has_edges and use_simulated:
                                        # Fallback 2: try alt start edge
                                        alt_start = _find_nearest_reachable(start_edge)
                                        if alt_start and alt_start != start_edge:
                                            r3 = env.traci_conn.simulation.findRoute(alt_start, end_edge)
                                            has_edges = bool(r3 and getattr(r3, "edges", None))
                                            if not has_edges and alt_end and alt_end != end_edge:
                                                r4 = env.traci_conn.simulation.findRoute(alt_start, alt_end)
                                                has_edges = bool(r4 and getattr(r4, "edges", None))
                                    route_cache[key] = has_edges
                                    return has_edges
                                except Exception:
                                    route_cache[key] = None
                                    return None

                            def _find_nearest_reachable(edge_id: str) -> Optional[str]:
                                """Find nearest vehicle-reachable edge for a pedestrian-only edge."""
                                try:
                                    shape = env.traci_conn.lane.getShape(f"{edge_id}_0")
                                    if shape:
                                        mid = shape[len(shape) // 2]
                                        nearest = env.traci_conn.simulation.convertRoad(mid[0], mid[1])
                                        if nearest and nearest[0] and not nearest[0].startswith(":"):
                                            return nearest[0]
                                except Exception:
                                    pass
                                return None

                            route_failed = False
                            route_unknown = False
                            missing_route_desc = None

                            for r in reservation_ids:
                                res_obj = active_reservations.get(r)
                                if res_obj is None:
                                    continue
                                # Handle both SUMO and Simulated reservations
                                if isinstance(res_obj, SimulatedReservation):
                                    from_edge = res_obj.from_edge
                                    to_edge = res_obj.to_edge
                                else:
                                    from_edge = getattr(res_obj, "fromEdge", None)
                                    to_edge = getattr(res_obj, "toEdge", None)
                                if not from_edge:
                                    route_unknown = True
                                    missing_route_desc = "missing pickup edge"
                                    break
                                if not to_edge:
                                    route_unknown = True
                                    missing_route_desc = "missing dropoff edge"
                                    break

                                # Taxi -> pickup
                                res = _route_exists(taxi_edge, from_edge)
                                if res is None:
                                    route_unknown = True
                                    missing_route_desc = f"{taxi_edge}->{from_edge}"
                                    break
                                if res is False:
                                    route_failed = True
                                    missing_route_desc = f"taxi->pickup: {taxi_edge}->{from_edge}"
                                    break

                                # Pickup -> dropoff
                                res = _route_exists(from_edge, to_edge)
                                if res is None:
                                    route_unknown = True
                                    missing_route_desc = f"{from_edge}->{to_edge}"
                                    break
                                if res is False:
                                    route_failed = True
                                    missing_route_desc = f"pickup->dropoff: {from_edge}->{to_edge}"
                                    break

                            if route_budget_exhausted:
                                actions["dispatch"].append({
                                    "taxi_id": taxi_id,
                                    "reservations": reservation_ids,
                                    "success": False,
                                    "reason": "Skipped dispatch: route check budget exhausted",
                                    "attempts": pending["attempts"],
                                })
                                remaining_dispatches.append(pending)
                                continue

                            if route_unknown:
                                actions["dispatch"].append({
                                    "taxi_id": taxi_id,
                                    "reservations": reservation_ids,
                                    "success": False,
                                    "reason": "Skipped dispatch: route check failed",
                                    "attempts": pending["attempts"],
                                })
                                remaining_dispatches.append(pending)
                                continue

                            if route_failed and not use_simulated:
                                drop = True
                                dispatch_no_route_count += 1
                                if not missing_route_desc:
                                    missing_route_desc = "unknown"
                                actions["dispatch"].append({
                                    "taxi_id": taxi_id,
                                    "reservations": reservation_ids,
                                    "success": False,
                                    "reason": f"Dropped dispatch: no route {missing_route_desc}",
                                    "attempts": pending["attempts"],
                                })
                                continue
                            # if route_failed and use_simulated:
                            #     # For simulated system, let _execute_simulated_dispatch handle it
                            #     # It has its own fallback logic for disconnected edges
                            #     print(f"[SimTaxi-DEBUG] Route pre-check failed ({missing_route_desc}), but trying simulated dispatch anyway")
                            try:
                                # Try dispatch - use simulated system if enabled
                                if use_simulated:
                                    # Use simulated dispatch for first reservation
                                    sim_success, sim_reason = self._execute_simulated_dispatch(
                                        env.traci_conn, taxi_id, reservation_ids[0], control_state
                                    )
                                    if sim_success:
                                        success = True
                                    else:
                                        # Record failed taxi-reservation pair so greedy dispatch skips it next time
                                        if "No route" in sim_reason or "no route" in sim_reason.lower():
                                            failed_pairs = control_state.setdefault("_failed_route_pairs", set())
                                            for r in reservation_ids:
                                                failed_pairs.add((taxi_id, self._normalize_reservation_id(r)))
                                            # NOTE: Removed permanent unreachable-edge blacklist.
                                            # Edges that fail routing are NOT blacklisted anymore;
                                            # instead, _execute_simulated_dispatch uses enhanced
                                            # fallback routing (convertRoad + nearby search).
                                            drop = True  # Don't re-queue; let reservation be re-assigned to different taxi
                                        actions["dispatch"].append({
                                            "taxi_id": taxi_id,
                                            "reservations": reservation_ids,
                                            "success": False,
                                            "reason": f"Simulated dispatch failed: {sim_reason}",
                                            "attempts": pending["attempts"],
                                        })
                                else:
                                    # Native SUMO taxi API removed; simulated system must be active
                                    print(f"[TaxiScheduling] ERROR: dispatchTaxi called but simulated system not active for taxi={taxi_id}")
                                    drop = True
                            except Exception as e:
                                error_msg = str(e)
                                # Default: keep queued until TTL is reached (above)
                                actions["dispatch"].append({
                                    "taxi_id": taxi_id,
                                    "reservations": reservation_ids,
                                    "success": False,
                                    "reason": f"Error: {error_msg} (queued)",
                                    "attempts": pending["attempts"],
                                })
            
            if success:
                # Mark as completed
                control_state["completed_dispatches"].append({
                    "taxi_id": taxi_id,
                    "reservation_ids": reservation_ids,
                    "completed_time": current_time
                })
                # Fairness bookkeeping: remember assignment and recent order time
                mapping = control_state.setdefault("reservation_to_taxi", {})
                for r in reservation_ids:
                    mapping[self._normalize_reservation_id(r)] = taxi_id
                self._record_taxi_order_time(control_state, taxi_id, current_time)
                control_state["total_dispatches"] = control_state.get("total_dispatches", 0) + 1
                actions["dispatch"].append({
                    "taxi_id": taxi_id,
                    "reservations": reservation_ids,
                    "success": True,
                    "attempts": pending["attempts"]
                })
            else:
                if not drop:
                    remaining_dispatches.append(pending)
        
        # Update pending list (append deferred entries not processed this cycle)
        control_state["pending_dispatches"] = remaining_dispatches + deferred_dispatches
        control_state["_last_dispatch_no_route_count"] = dispatch_no_route_count
        if dispatch_no_route_count > 0:
            control_state["dispatch_no_route_count"] = control_state.get("dispatch_no_route_count", 0) + dispatch_no_route_count
            print(f"[TaxiScheduling] Dispatch: skipped {dispatch_no_route_count} assignments due to no route")

        # Process repositioning decisions from config
        # Repositioning is usually "fire and forget" but we don't want to spam it either?
        # If target has been set, should we set it again?
        # Ideally, check if target is already set.
        # But for now, let's leave Repositioning as is (it simply calls changeTarget). 
        # Calling changeTarget to same edge repeatedly is generally harmless (idempotent-ish).
        
        completed_reposition_signatures = {
            (d.get("taxi_id"), d.get("target_edge"))
            for d in control_state.get("completed_repositions", [])
            if isinstance(d, dict)
        }

        reposition_decisions = config.get("reposition_decisions", []) or []
        if reposition_decisions:
            filtered_reposition = []
            for decision in reposition_decisions:
                if not isinstance(decision, dict):
                    continue
                taxi_id = decision.get("taxi_id")
                if not taxi_id or taxi_id not in idle_taxis_set:
                    continue
                target_taz = decision.get("target_taz")
                target_edge = decision.get("target_edge")
                if target_taz:
                    if target_taz not in self.taz_edges:
                        continue
                elif not target_edge:
                    continue
                filtered_reposition.append(decision)
            reposition_decisions = filtered_reposition
            config["reposition_decisions"] = reposition_decisions
        reposition_strategy = str(config.get("reposition_strategy", "llm")).lower()
        auto_reposition_strategies = {
            "online",
            "auto",
            "auto_taz_balance",
            "taz_balance",
        }

        if bool(config.get("reposition_enabled", True)) and (
            reposition_strategy in auto_reposition_strategies
            or (reposition_strategy == "llm" and bool(config.get("fallback_to_auto_reposition", False)) and not reposition_decisions)
        ):
            reserved_for_dispatch = {
                d.get("taxi_id")
                for d in (dispatch_decisions or [])
                if isinstance(d, dict) and d.get("taxi_id")
            }
            reserved_for_dispatch |= {
                d.get("taxi_id")
                for d in (control_state.get("pending_dispatches") or [])
                if isinstance(d, dict) and d.get("taxi_id")
            }

            auto_repositions = self._generate_auto_reposition_decisions(
                traci=traci,
                active_reservations=active_reservations,
                idle_taxis_set=idle_taxis_set,
                control_state=control_state,
                config=config,
                current_time=current_time,
                reserved_taxis=reserved_for_dispatch,
                env=env,
            )

            if auto_repositions:
                existing = {
                    d.get("taxi_id")
                    for d in reposition_decisions
                    if isinstance(d, dict) and d.get("taxi_id")
                }
                for d in auto_repositions:
                    if d.get("taxi_id") in existing:
                        continue
                    reposition_decisions.append(d)
                    existing.add(d.get("taxi_id"))

        for decision in reposition_decisions:
            taxi_id = decision.get("taxi_id")
            target_edge = decision.get("target_edge")
            target_taz = decision.get("target_taz")

            taxi_edge = None
            if taxi_id:
                try:
                    taxi_edge = env.traci_conn.vehicle.getRoadID(taxi_id)
                except Exception:
                    taxi_edge = None

            # Resolve TAZ to edge if needed
            if target_taz and not target_edge:
                if target_taz in self.taz_edges:
                    edges = self.taz_edges[target_taz]
                    if edges:
                        target_edge = self._select_reposition_edge(env.traci_conn, taxi_edge, edges)
                        if not target_edge:
                            actions["reposition"].append({
                                "taxi_id": taxi_id,
                                "target_taz": target_taz,
                                "success": False,
                                "reason": "No reachable non-dead-end edge in target TAZ"
                            })
                            continue
                    else:
                        actions["reposition"].append({
                            "taxi_id": taxi_id,
                            "target_taz": target_taz,
                            "success": False,
                            "reason": f"TAZ {target_taz} has no edges"
                        })
                        continue
                else:
                    actions["reposition"].append({
                        "taxi_id": taxi_id,
                        "target_taz": target_taz,
                        "success": False,
                        "reason": f"Unknown TAZ: {target_taz}"
                    })
                    continue

            if taxi_id and target_edge:
                if self._edge_is_dead_end(env.traci_conn, target_edge):
                    actions["reposition"].append({
                        "taxi_id": taxi_id,
                        "target_edge": target_edge,
                        "success": False,
                        "reason": "Target edge is a dead-end"
                    })
                    continue
                if taxi_edge and not self._route_is_reachable(env.traci_conn, taxi_edge, target_edge):
                    actions["reposition"].append({
                        "taxi_id": taxi_id,
                        "target_edge": target_edge,
                        "success": False,
                        "reason": "Target edge unreachable from current edge"
                    })
                    continue

                # Additional validation: skip if target edge is a dead end
                if self._edge_is_dead_end(env.traci_conn, target_edge):
                    actions["reposition"].append({
                        "taxi_id": taxi_id,
                        "target_edge": target_edge,
                        "success": False,
                        "reason": "Target edge is a dead end (cul-de-sac)"
                    })
                    continue

                # De-duplicate successful repositions (taxis keep their target until changed)
                signature = (taxi_id, target_edge)
                if signature in completed_reposition_signatures:
                    continue
                try:
                    # Verify taxi is idle before repositioning (use cached idle_taxis_set instead of new TraCI call)
                    if taxi_id in idle_taxis_set:
                        # Release the taxi from speed freeze so it can move
                        try:
                            env.traci_conn.vehicle.setSpeed(taxi_id, -1)
                        except Exception:
                            pass

                        # Change taxi's target to reposition
                        env.traci_conn.vehicle.changeTarget(taxi_id, target_edge)
                        actions["reposition"].append({
                            "taxi_id": taxi_id,
                            "target_edge": target_edge,
                            "success": True
                        })
                        control_state["completed_repositions"].append({
                            "taxi_id": taxi_id,
                            "target_edge": target_edge,
                            "completed_time": current_time
                        })
                        completed_reposition_signatures.add(signature)
                    else:
                        actions["reposition"].append({
                            "taxi_id": taxi_id,
                            "target_edge": target_edge,
                            "success": False,
                            "reason": "Taxi not idle"
                        })
                except Exception as e:
                    actions["reposition"].append({
                        "taxi_id": taxi_id,
                        "target_edge": target_edge,
                        "success": False,
                        "reason": str(e)
                    })

        # Enhanced Taxi keepalive logic: prevent taxis from being removed by SUMO
        # This is critical when using "stop" idle algorithm - taxis need explicit targets
        # IMPORTANT: Keepalive is DISABLED by default because:
        # 1. Most idle algorithms (randomCircling, etc.) don't need it
        # 2. Sending many changeTarget commands can crash SUMO
        # 3. Only enable if using "stop" idle algorithm and taxis are being removed
        # Bulk keepalive runs only at:
        # 1. Checkpoint start (first call in checkpoint)
        # 2. Every 900s within the checkpoint
        try:
            keepalive_enabled = bool(config.get("keepalive_enabled", False))  # Disabled by default
            if keepalive_enabled:
                # Check if we should run bulk keepalive
                # Only run at checkpoint start and every 900s
                last_bulk_keepalive = control_state.get("_last_bulk_keepalive_time", 0)
                bulk_keepalive_interval = 900  # Run every 900s

                should_run_bulk_keepalive = False
                if last_bulk_keepalive == 0:
                    # First call in this checkpoint - run keepalive
                    should_run_bulk_keepalive = True
                elif current_time - last_bulk_keepalive >= bulk_keepalive_interval:
                    # 900s has passed since last bulk keepalive
                    should_run_bulk_keepalive = True

                if should_run_bulk_keepalive:
                    # Use safe method to avoid protocol corruption
                    all_taxis = self._safe_get_taxi_fleet(env.traci_conn, state=-1, env=env)
                    taz_edges_list = list(self.taz_edges.values()) if self.taz_edges else []
                    all_edges = [e for edges in taz_edges_list for e in edges] if taz_edges_list else []

                    # If no TAZ edges, try to get edges from network
                    if not all_edges:
                        # Fallback: get edges from all taxis in the fleet
                        for taxi_id in all_taxis:
                            try:
                                edge = env.traci_conn.vehicle.getRoadID(taxi_id)
                                if edge and not edge.startswith(":"):
                                    all_edges.append(edge)
                            except Exception:
                                continue
                        # Also try to get edges from the network
                        if not all_edges:
                            try:
                                all_edges = list(env.traci_conn.edge.getIDList())[:100]
                            except Exception:
                                pass

                    keepalive_count = 0
                    # CRITICAL: Limit the number of taxis refreshed per cycle to avoid overwhelming SUMO
                    max_keepalive_per_cycle = int(config.get("max_keepalive_per_cycle", 50))

                    # Refresh idle taxis at checkpoint boundaries (limited to avoid overwhelming SUMO)
                    for taxi_id in all_taxis:
                        if keepalive_count >= max_keepalive_per_cycle:
                            break  # Stop after reaching limit
                        if taxi_id not in idle_taxis_set:
                            continue  # Only refresh idle taxis

                        try:
                            if all_edges:
                                # Try multiple times to find a valid target
                                for attempt in range(10):
                                    new_target = random.choice(all_edges)
                                    if new_target.startswith(":"):
                                        continue
                                    try:
                                        env.traci_conn.vehicle.changeTarget(taxi_id, new_target)
                                        keepalive_count += 1
                                        break
                                    except Exception:
                                        continue
                        except Exception:
                            continue

                    control_state["_last_bulk_keepalive_time"] = current_time

        except Exception as e:
            # Keepalive is best-effort, don't fail the whole control cycle
            print(f"[TaxiScheduling] Keepalive error: {e}")
            pass

        # Auto-replenish taxis: detect if fleet size has decreased and add new taxis
        # to maintain the target fleet size
        try:
            replenish_enabled = bool(config.get("auto_replenish_enabled", True))
            if replenish_enabled:
                # Use safe method to avoid protocol corruption
                current_fleet = self._safe_get_taxi_fleet(env.traci_conn, state=-1, env=env)
                current_fleet_size = len(current_fleet)

                # Get initial_fleet_size: prefer config (preserved across checkpoints), then control_state, then current
                # This ensures initial_fleet_size survives checkpoint load/restore cycles
                config_initial = config.get("initial_fleet_size")
                state_initial = control_state.get("initial_fleet_size")

                if config_initial is not None and int(config_initial) > 0:
                    initial_fleet_size = int(config_initial)
                elif state_initial is not None and int(state_initial) > 0:
                    initial_fleet_size = int(state_initial)
                else:
                    initial_fleet_size = current_fleet_size
                    # Store in config for future checkpoint cycles (authoritative source)
                    config["initial_fleet_size"] = initial_fleet_size

                # Always update control_state to keep it in sync
                control_state["initial_fleet_size"] = initial_fleet_size
                if "total_taxis_replenished" not in control_state:
                    control_state["total_taxis_replenished"] = 0

                target_fleet_size = int(config.get("target_fleet_size") or initial_fleet_size)
                fleet_deficit = target_fleet_size - current_fleet_size

                # Only replenish if there's a deficit
                if fleet_deficit > 0:
                    # Collect valid edges for spawning new taxis
                    taz_edges_list = list(self.taz_edges.values()) if self.taz_edges else []
                    spawn_edges = [e for edges in taz_edges_list for e in edges] if taz_edges_list else []

                    # DEBUG: Log spawn edges status
                    # if not spawn_edges:
                    #     print(f"[TaxiScheduling] Replenish: No TAZ edges, trying fallback...")

                    if not spawn_edges:
                        # Fallback: try to get edges from existing taxis (any state)
                        for taxi_id in current_fleet:
                            try:
                                edge = env.traci_conn.vehicle.getRoadID(taxi_id)
                                if edge and not edge.startswith(":"):
                                    spawn_edges.append(edge)
                            except Exception:
                                continue

                    # Additional fallback: get edges from network
                    if not spawn_edges:
                        try:
                            network_edges = list(env.traci_conn.edge.getIDList())
                            # Filter out internal edges
                            spawn_edges = [e for e in network_edges if not e.startswith(":")][:200]
                            # print(f"[TaxiScheduling] Replenish: Using {len(spawn_edges)} network edges as fallback")
                        except Exception as e:
                            print(f"[TaxiScheduling] Replenish: Failed to get network edges: {e}")

                    if spawn_edges:
                        # Filter to edges that allow taxi/passenger (avoid invalid depart edges)
                        filtered_edges = []
                        invalid_edge_count = 0
                        allowed_edges = self._get_cached_taxi_allowed_edges(env)
                        if allowed_edges is not None:
                            for edge_id in spawn_edges:
                                if not edge_id or edge_id.startswith(":") or edge_id not in allowed_edges:
                                    invalid_edge_count += 1
                                    continue
                                filtered_edges.append(edge_id)
                        else:
                            for edge_id in spawn_edges:
                                if not edge_id or edge_id.startswith(":"):
                                    invalid_edge_count += 1
                                    continue
                                allowed = None
                                try:
                                    allowed = env.traci_conn.edge.getAllowed(edge_id)
                                except Exception:
                                    allowed = None
                                if allowed and "taxi" not in allowed:
                                    invalid_edge_count += 1
                                    continue
                                filtered_edges.append(edge_id)
                        if invalid_edge_count > 0:
                            control_state["replenish_invalid_edge_count"] = control_state.get("replenish_invalid_edge_count", 0) + invalid_edge_count
                            # print(f"[TaxiScheduling] Replenish: filtered {invalid_edge_count} edges not taxi-allowed")
                        spawn_edges = filtered_edges

                    # Apply cached valid spawn edges to reduce no-route failures
                    if spawn_edges:
                        cached_valid_spawns = control_state.get("_cached_valid_spawn_edges")
                        cache_time = control_state.get("_cached_valid_spawn_edges_time", 0)
                        cache_expired = (current_time - cache_time) > 1800.0
                        if cached_valid_spawns is None or cache_expired:
                            # Pre-validate: test findRoute from each spawn edge to a random target
                            valid = []
                            sample = spawn_edges[:200]  # limit validation count
                            for edge in sample:
                                try:
                                    tgt = random.choice(spawn_edges)
                                    r = env.traci_conn.simulation.findRoute(edge, tgt, vType="taxi")
                                    if r and getattr(r, "edges", None):
                                        valid.append(edge)
                                except Exception:
                                    pass
                            control_state["_cached_valid_spawn_edges"] = valid
                            control_state["_cached_valid_spawn_edges_time"] = current_time
                            # if valid:
                            #     print(f"[TaxiScheduling] Replenish: cached {len(valid)}/{len(sample)} valid spawn edges")
                            # else:
                            #     print("[TaxiScheduling] Replenish: WARNING no valid spawn edges found in cache pass")
                            if valid:
                                spawn_edges = valid
                        else:
                            if cached_valid_spawns:
                                spawn_edges = cached_valid_spawns

                    if spawn_edges:
                        # Limit replenishment per cycle to avoid overwhelming the simulation
                        max_replenish_per_cycle = int(config.get("max_replenish_per_cycle", 50))
                        if load_shedding_active:
                            backoff = float(config.get("load_shedding_backoff_factor", 0.5))
                            min_replenish = int(config.get("load_shedding_min_replenish", 10))
                            max_replenish_per_cycle = max(min_replenish, int(max_replenish_per_cycle * backoff))
                        taxis_to_add = min(fleet_deficit, max_replenish_per_cycle)

                        # Get existing taxi IDs to avoid conflicts
                        existing_ids = set(current_fleet)

                        # Find the next available taxi ID
                        next_id = control_state.get("next_replenish_id", 0)

                        taxis_added = 0
                        attempts = 0
                        dead_end_count = 0
                        max_attempts = taxis_to_add * 5  # Allow more failures to ensure deficit is filled
                        last_error = None
                        no_route_count = 0

                        while taxis_added < taxis_to_add and attempts < max_attempts:
                            attempts += 1
                            # Generate unique taxi ID
                            new_taxi_id = f"taxi_replenish_{next_id}"
                            next_id += 1

                            if new_taxi_id in existing_ids:
                                continue

                            # Select random spawn/target edges and ensure a valid route exists
                            spawn_edge = random.choice(spawn_edges)
                            target_edge = random.choice(spawn_edges)
                            if len(spawn_edges) > 1 and target_edge == spawn_edge:
                                target_edge = random.choice([e for e in spawn_edges if e != spawn_edge])

                            # Skip dead end check for replenish - just try to add
                            # The taxi will be removed if it can't move, but that's better than not adding at all
                            # if self._edge_is_dead_end(traci_conn, spawn_edge):
                            #     dead_end_count += 1
                            #     continue

                            try:
                                # Add new taxi to simulation
                                # Create a valid route (avoid single-edge routes that cannot depart)
                                try:
                                    try:
                                        route_result = env.traci_conn.simulation.findRoute(spawn_edge, target_edge, vType="taxi")
                                    except TypeError:
                                        route_result = env.traci_conn.simulation.findRoute(spawn_edge, target_edge)
                                    route_edges = list(route_result.edges) if route_result and getattr(route_result, "edges", None) else []
                                except Exception:
                                    route_edges = []
                                if not route_edges:
                                    no_route_count += 1
                                    continue
                                route_id = f"route_{new_taxi_id}"
                                env.traci_conn.route.add(route_id, route_edges)

                                # Add vehicle with taxi type
                                env.traci_conn.vehicle.add(
                                    vehID=new_taxi_id,
                                    routeID=route_id,
                                    typeID="taxi",
                                    depart="now",
                                    departPos="random",
                                    departSpeed="0"
                                )

                                # Freeze the new taxi and register it in simulated system
                                env.traci_conn.vehicle.setSpeed(new_taxi_id, 0)
                                self._simulated_taxis[new_taxi_id] = SimulatedTaxi(
                                    id=new_taxi_id,
                                    state=SimulatedTaxiState.IDLE,
                                    spawn_time=current_time,
                                )

                                taxis_added += 1
                                existing_ids.add(new_taxi_id)
                                spawn_times = control_state.setdefault("taxi_spawn_times", {})
                                if isinstance(spawn_times, dict):
                                    spawn_times[new_taxi_id] = current_time

                            except Exception as e:
                                # Failed to add taxi, try next edge
                                last_error = str(e)
                                continue

                        control_state["next_replenish_id"] = next_id
                        control_state["total_taxis_replenished"] = control_state.get("total_taxis_replenished", 0) + taxis_added
                        control_state["_last_replenish_no_route_count"] = no_route_count
                        if no_route_count > 0:
                            control_state["replenish_no_route_count"] = control_state.get("replenish_no_route_count", 0) + no_route_count
                            # print(f"[TaxiScheduling] Replenish: skipped {no_route_count} spawn attempts due to no route")

                        if taxis_added > 0:
                            # Update metrics
                            control_state["total_replenished"] = control_state.get("total_replenished", 0) + taxis_added

                            # print(f"[TaxiScheduling] Auto-replenished {taxis_added} taxis "
                            #       f"(deficit: {fleet_deficit}, target: {target_fleet_size}, "
                            #       f"current: {current_fleet_size + taxis_added})")

                            # Add to actions for tracking
                            if "replenish" not in actions:
                                actions["replenish"] = []
                            actions["replenish"].append({
                                "taxis_added": taxis_added,
                                "deficit": fleet_deficit,
                                "target_fleet_size": target_fleet_size,
                                "current_fleet_size": current_fleet_size + taxis_added
                            })
        except Exception as e:
            # Auto-replenish is best-effort, don't fail the whole control cycle
            print(f"[TaxiScheduling] Replenish error: {e}")
            import traceback
            traceback.print_exc()

        # Update control state
        control_state["last_update_time"] = current_time
        
        next_action_time = 60.0
        if load_shedding_active:
            next_action_time = float(config.get("load_shedding_next_action_time", 120.0))
        return {
            "control_state": control_state,
            "actions": actions,
            "next_action_time": next_action_time  # Check every 60 seconds (longer under load)
        }
    
    def update_control_state(
        self,
        control_state: Dict[str, Any],
        step_duration: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update control state after a simulation step.
        
        Args:
            control_state: Current control state
            step_duration: Duration of the simulation step
            **kwargs: Additional arguments (unused)
            
        Returns:
            Updated control state
        """
        # Increment elapsed time
        control_state["elapsed_time"] = control_state.get("elapsed_time", 0) + step_duration
        return control_state
    
    def _initialize_control_state(
        self,
        env: Any,
        config: Dict[str, Any],
        initial_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Initialize control state for taxi scheduling.
        
        Args:
            env: SUMOEnv instance
            config: Taxi scheduling configuration
            initial_time: Initial simulation time
            
        Returns:
            Initial control state dictionary
        """
        # Try to load TAZs if not loaded
        if not self.taz_edges:
            if hasattr(env, 'config_path'):
                # Infer taz path from config path
                config_dir = os.path.dirname(env.config_path)
                taz_path = os.path.join(config_dir, "districts.taz.xml")
                print(f"TaxiSchedulingModule: Attempting to load TAZs from {taz_path}")
                if os.path.exists(taz_path):
                    self.load_taz_definitions(taz_path)
                else:
                    print(f"TaxiSchedulingModule: TAZ file not found at {taz_path}")
            else:
                print("TaxiSchedulingModule: env.config_path not found, cannot load TAZs.")

        state = {
            "initial_time": initial_time,
            "last_update_time": initial_time,
            "elapsed_time": 0,
            "total_dispatches": 0,
            "fleet_size": config.get("fleet_size", 0),
            "reposition_interval": config.get("reposition_interval_seconds", 300),
            "last_reposition_time": initial_time,
            "reservation_tracker": {},  # Tracks state of every reservation: {res_id: {state, reservation_time, ...}}
            "known_reservations": set(),  # Set of known reservation IDs for fast lookup
            # Cross-checkpoint fairness state
            "reservation_to_taxi": {},  # reservation_id -> taxi_id (best-effort)
            "taxi_income": {},  # taxi_id -> cumulative income (proxy fare)
            "taxi_recent_order_times": {},  # taxi_id -> [dispatch_time, ...] within a rolling window
            "taxi_spawn_times": {},  # taxi_id -> spawn_time (for dispatch age guard)
            "fare_base": float(config.get("fare_base", 3.0)),
            "fare_per_second": float(config.get("fare_per_second", 0.01)),
            "recent_order_window_seconds": float(config.get("recent_order_window_seconds", 1800)),
        }
        # Get initial fleet state if TraCI is available
        if hasattr(env, 'traci_conn') and env.traci_conn is not None:
            try:
                traci = env.traci_conn
                # Use safe methods to avoid protocol corruption
                all_taxis = self._safe_get_taxi_fleet(traci, state=-1, env=env)
                idle_taxis = self._safe_get_taxi_fleet(traci, state=0, env=env)
                state["fleet_size"] = len(all_taxis)
                state["initial_idle_count"] = len(idle_taxis)
                print(f"Taxi scheduling initialized: {len(all_taxis)} taxis, {len(idle_taxis)} idle")
            except Exception as e:
                print(f"Warning: Could not initialize taxi fleet state: {e}")
        
        return state

    def restore_control_state_from_checkpoint(
        self,
        env: Any,
        control_state: Optional[Dict[str, Any]],
        checkpoint_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Restore taxi control state from checkpoint metadata and resync with SUMO.

        Args:
            env: SUMOEnv instance with TraCI connection
            control_state: Existing in-memory control state (may be None)
            checkpoint_state: Control state loaded from checkpoint metadata (may be None)

        Returns:
            Restored and synced control state
        """
        # Start from checkpoint state if available; otherwise keep current.
        if checkpoint_state:
            state = copy.deepcopy(checkpoint_state)
        else:
            state = copy.deepcopy(control_state) if control_state else {}

        # Ensure required structures exist.
        state.setdefault("pending_dispatches", [])
        state.setdefault("completed_dispatches", [])
        state.setdefault("completed_repositions", [])
        state.setdefault("reservation_tracker", {})

        known_res = state.get("known_reservations")
        if isinstance(known_res, list):
            state["known_reservations"] = set(known_res)
        elif known_res is None:
            state["known_reservations"] = set()

        state.setdefault("reservation_to_taxi", {})
        state.setdefault("taxi_income", {})
        state.setdefault("taxi_recent_order_times", {})
        state.setdefault("taxi_spawn_times", {})

        # CRITICAL: Preserve initial_fleet_size from checkpoint to maintain fleet size across restarts
        # This is essential for auto-replenish to work correctly after checkpoint load
        if checkpoint_state and checkpoint_state.get("initial_fleet_size"):
            state["initial_fleet_size"] = int(checkpoint_state["initial_fleet_size"])
            print(f"[TaxiScheduling] Restored initial_fleet_size from checkpoint: {state['initial_fleet_size']}")
        elif control_state and control_state.get("initial_fleet_size"):
            state["initial_fleet_size"] = int(control_state["initial_fleet_size"])
            print(f"[TaxiScheduling] Preserved initial_fleet_size from control_state: {state['initial_fleet_size']}")

        # Merge any missing keys from in-memory state.
        if control_state:
            for key, value in control_state.items():
                if key not in state:
                    state[key] = value

        if hasattr(env, "traci_conn") and env.traci_conn is not None:
            traci = env.traci_conn
            # Adopt any active reservations that predate this checkpoint.
            self._sync_reservations(env, state)
            # Rebuild reservation -> taxi mapping from live vehicle states.
            self._sync_reservation_to_taxi_mapping(traci, state, None, env=env)

            # Prune stale pending dispatches if their reservations are no longer active.
            # Use safe method to avoid protocol corruption
            active_reservations = self._safe_get_taxi_reservations(traci, env=env)
            active_ids = {
                self._normalize_reservation_id(r.id)
                for r in active_reservations
            } if active_reservations else None

            if active_ids is not None and state.get("pending_dispatches"):
                kept = []
                removed = 0
                for item in state.get("pending_dispatches", []):
                    res_ids = [
                        self._normalize_reservation_id(r)
                        for r in (item.get("reservation_ids") or [])
                    ]
                    if not res_ids or any(rid in active_ids for rid in res_ids):
                        kept.append(item)
                    else:
                        removed += 1
                if removed > 0:
                    print(f"TaxiSchedulingModule: pruned {removed} stale pending dispatches after load.")
                state["pending_dispatches"] = kept

        # CRITICAL: Force taxi fleet reload after checkpoint restore.
        # _simulated_taxis dict is not persisted in checkpoints, so it will be
        # empty after restore. Reset the loaded flag so _load_taxi_fleet_from_file
        # is called again on the next step.
        self._taxi_fleet_file_loaded = False

        # Sync any taxis already present in SUMO into _simulated_taxis
        # to avoid duplicate injection when the fleet file is reloaded.
        if hasattr(env, "traci_conn") and env.traci_conn is not None:
            try:
                existing = self._safe_get_taxi_fleet(env.traci_conn, state=-1, env=env)
                current_time = env.traci_conn.simulation.getTime()
                synced = 0
                for tid in existing:
                    if tid not in self._simulated_taxis:
                        self._simulated_taxis[tid] = SimulatedTaxi(
                            id=tid,
                            state=SimulatedTaxiState.IDLE,
                            spawn_time=current_time,
                        )
                        synced += 1
                if synced > 0:
                    print(f"[TaxiScheduling] Checkpoint restore: synced {synced} existing SUMO taxis into _simulated_taxis")
            except Exception as e:
                print(f"[TaxiScheduling] Checkpoint restore: failed to sync existing taxis: {e}")

        # Clear stale caches that may reference edges/pairs from before checkpoint
        state.pop("_unreachable_edges", None)
        state.pop("_edge_route_fail_counts", None)
        state.pop("_failed_route_pairs", None)
        state.pop("_cached_valid_spawn_edges", None)

        return state

    def get_taxi_fleet_state(self, env: Any, control_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get current state of the taxi fleet.

        Args:
            env: SUMOEnv instance with TraCI connection

        Returns:
            Dictionary containing fleet state information
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return {"error": "TraCI connection not available"}

        traci_conn = env.traci_conn

        # Validate TraCI connection before making calls
        try:
            traci_conn.simulation.getTime()
        except (traci.TraCIException, traci.exceptions.FatalTraCIError):
            return {"error": "TraCI connection closed"}
        except Exception:
            pass

        try:
            # Get taxi lists by state using safe methods
            all_taxis = self._safe_get_taxi_fleet(traci_conn, state=-1, env=env)
            idle_taxis = self._safe_get_taxi_fleet(traci_conn, state=0, env=env)
            pickup_taxis = self._safe_get_taxi_fleet(traci_conn, state=1, env=env)
            occupied_taxis = self._safe_get_taxi_fleet(traci_conn, state=2, env=env)

            # Get current vehicle list once for validation
            current_vehicle_ids = set(traci_conn.vehicle.getIDList())

            # Get taxi details - only query taxis that exist
            taxi_details = {}
            for taxi_id in all_taxis:
                # Skip taxis that are no longer in the simulation
                if taxi_id not in current_vehicle_ids:
                    continue
                try:
                    # Use simulated state if available
                    if self._simulated_system_initialized and taxi_id in self._simulated_taxis:
                        sim_taxi = self._simulated_taxis[taxi_id]
                        state_str = str(sim_taxi.state)
                        customers = sim_taxi.current_reservation_id or ""
                    else:
                        state_str = traci_conn.vehicle.getParameter(taxi_id, "device.taxi.state")
                        customers = traci_conn.vehicle.getParameter(taxi_id, "device.taxi.customers")

                    taxi_details[taxi_id] = {
                        "state": state_str,
                        "customers": customers,
                        "current_edge": traci_conn.vehicle.getRoadID(taxi_id),
                        "current_taz": self.edge_to_taz.get(traci_conn.vehicle.getRoadID(taxi_id), "unknown"),
                        "position": traci_conn.vehicle.getPosition(taxi_id),
                        "speed": traci_conn.vehicle.getSpeed(taxi_id)
                    }
                except traci.TraCIException:
                    # Taxi may have left the network
                    continue
                except Exception as e:
                    taxi_details[taxi_id] = {"error": str(e)}

            # Attach fairness state if available
            if control_state:
                taxi_income = control_state.get("taxi_income", {}) or {}
                recent_map = control_state.get("taxi_recent_order_times", {}) or {}
                window_seconds = float(control_state.get("recent_order_window_seconds", 1800))
                now = float(env.get_current_time())
                for taxi_id in list(taxi_details.keys()):
                    details = taxi_details.get(taxi_id)
                    if not isinstance(details, dict) or "error" in details:
                        continue
                    details["cumulative_income"] = float(taxi_income.get(taxi_id, 0.0))
                    pruned = self._prune_recent_order_times(recent_map.get(taxi_id) or [], now, window_seconds)
                    recent_map[taxi_id] = pruned
                    details["recent_order_count"] = len(pruned)

            return {
                "fleet_size": len(all_taxis),
                "idle_count": len(idle_taxis),
                "pickup_count": len(pickup_taxis),
                "occupied_count": len(occupied_taxis),
                "idle_taxis": idle_taxis,
                "pickup_taxis": pickup_taxis,
                "occupied_taxis": occupied_taxis,
                "taxi_details": taxi_details,
                "utilization_rate": (len(pickup_taxis) + len(occupied_taxis)) / max(len(all_taxis), 1)
            }
        except (traci.TraCIException, traci.exceptions.FatalTraCIError) as e:
            return {"error": f"TraCI error: {e}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_pending_reservations(self, env: Any, include_person_positions: bool = False, cached_reservations: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Get pending taxi reservations (waiting passengers).

        Args:
            env: SUMOEnv instance with TraCI connection
            include_person_positions: Whether to include pickup position info
            cached_reservations: Optional pre-fetched reservations list to avoid redundant TraCI calls

        Returns:
            Dictionary containing reservation information
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return {"error": "TraCI connection not available", "pending_count": 0, "reservations": []}

        traci_conn = env.traci_conn
        # Reuse cached checkpoint pending reservations if available (to avoid repeated TraCI calls)
        if not include_person_positions and cached_reservations is None:
            cached = None
            if isinstance(env, dict):
                cached = env.get("_cached_checkpoint_pending")
            elif hasattr(env, "enabled_controls"):
                controls = getattr(env, "enabled_controls", None)
                if isinstance(controls, dict):
                    taxi_state = controls.get("taxi_scheduling", {}).get("state")
                    if isinstance(taxi_state, dict):
                        cached = taxi_state.get("_cached_checkpoint_pending")
            if isinstance(cached, dict) and "reservations" in cached:
                return cached

        # Check if TraCI connection is still valid before making calls
        try:
            # Quick validation: check if the connection object has _connections attribute
            if hasattr(traci_conn, '_connections') and not traci_conn._connections:
                return {"error": "TraCI connection closed", "pending_count": 0, "reservations": []}
            # Additional validation: try a lightweight TraCI call
            traci_conn.simulation.getTime()
        except (traci.TraCIException, traci.exceptions.FatalTraCIError):
            return {"error": "TraCI connection closed", "pending_count": 0, "reservations": []}
        except Exception:
            pass

        try:
            # Use cached reservations if provided, otherwise fetch with safe method
            if cached_reservations is not None:
                reservations = cached_reservations
            else:
                reservations = self._safe_get_taxi_reservations(traci_conn, env=env)

            # Get current person list once for validation
            current_person_ids = set()
            if include_person_positions:
                try:
                    current_person_ids = set(traci_conn.person.getIDList())
                except traci.TraCIException:
                    current_person_ids = set()

            reservation_list = []
            for res in reservations:
                # Handle both SUMO Reservation objects and SimulatedReservation objects
                if isinstance(res, SimulatedReservation):
                    res_id = res.id
                    res_state = res.state
                    person_ids = res.person_ids
                    from_edge = res.from_edge
                    to_edge = res.to_edge
                    depart_time = res.depart_time
                    reservation_time = res.reservation_time
                else:
                    res_id = self._normalize_reservation_id(res.id)
                    res_state = getattr(res, "state", None)
                    person_ids = list(res.persons) if hasattr(res, 'persons') else []
                    from_edge = res.fromEdge
                    to_edge = res.toEdge
                    depart_time = res.depart if hasattr(res, 'depart') else None
                    reservation_time = res.reservationTime if hasattr(res, 'reservationTime') else None

                # Pending = not yet picked up (still waiting for taxi)
                if self._reservation_is_picked_up(res_state):
                    continue

                # Optional pickup position info (for LLM distance ranking)
                pickup_position = None
                pickup_road_id = None
                if include_person_positions and person_ids:
                    person_id = person_ids[0]
                    if person_id in current_person_ids:
                        try:
                            pickup_position = list(traci_conn.person.getPosition(person_id))
                            pickup_road_id = traci_conn.person.getRoadID(person_id)
                        except traci.TraCIException:
                            pickup_position = None
                            pickup_road_id = None
                        except Exception:
                            pickup_position = None
                            pickup_road_id = None

                reservation_list.append({
                    "id": res_id,
                    "state": res_state,
                    "person_ids": person_ids,
                    "from_edge": from_edge,
                    "to_edge": to_edge,
                    "depart_time": depart_time,
                    "reservation_time": reservation_time,
                    "origin_taz": self.edge_to_taz.get(from_edge, "unknown"),
                    "pickup_position": pickup_position,
                    "pickup_road_id": pickup_road_id
                })

            return {
                "pending_count": len(reservation_list),
                "reservations": reservation_list
            }
        except (traci.TraCIException, traci.exceptions.FatalTraCIError) as e:
            error_str = str(e)
            # Check for SUMO connection closed error specifically
            if "Connection closed" in error_str or "FatalTraCIError" in error_str:
                print("Warning: SUMO connection closed unexpectedly")
                # Mark env as unhealthy if method is available
                if hasattr(env, 'mark_traci_unhealthy'):
                    env.mark_traci_unhealthy()
            return {"error": "SUMO closed", "pending_count": 0, "reservations": []}
        except UnicodeDecodeError as e:
            # This indicates TraCI protocol buffer corruption - CRITICAL ERROR
            # The connection is now in an unrecoverable state
            print(f"CRITICAL: TraCI protocol data corruption detected: {e}")
            print("The TraCI connection is now in an unrecoverable state.")
            # Mark env as unhealthy so simulation can be terminated gracefully
            if hasattr(env, 'mark_traci_unhealthy'):
                env.mark_traci_unhealthy()
            return {"error": "TraCI protocol error", "pending_count": 0, "reservations": []}
        except Exception as e:
            error_str = str(e)
            traceback.print_exc()
            return {"error": error_str, "pending_count": 0, "reservations": []}
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for tracking taxi scheduling performance.
        
        Returns:
            Dictionary with initialized metric structures
        """
        return {
            "total_dispatches": 0,
            "successful_dispatches": 0,
            "failed_dispatches": 0,
            "total_replenished": 0,  # Track auto-replenished taxis
            "wait_times": [],
            "pickup_times": [],
            "fleet_utilizations": [],
            "idle_counts": [],
            "pending_reservation_counts": [],
            "passenger_pickups": 0,
            "passenger_dropoffs": 0,
            "passenger_wait_times": [],
            "passenger_travel_times": [],
            "unserved_reservations": 0
        }

    def _cleanup_stale_reservations(
        self,
        env: Any,
        control_state: Dict[str, Any],
        config: Dict[str, Any],
        current_time: float,
        cached_reservations: Optional[List[Any]] = None,
    ) -> int:
        """
        Clean up stale reservations that have been in 'retrieved' state (state 2) for too long.

        These reservations can accumulate and eventually cause SUMO to crash.
        This method marks them for re-dispatch by removing them from pending dispatches
        and tracking them for potential cancellation.

        Args:
            env: SUMOEnv instance
            control_state: Control state dictionary
            config: Configuration dictionary
            current_time: Current simulation time
            cached_reservations: Optional pre-fetched reservations list

        Returns:
            Number of stale reservations cleaned up
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return 0

        traci_conn = env.traci_conn
        max_age = float(config.get("max_retrieved_reservation_age", 600))

        # Use cached reservations if provided
        if cached_reservations is not None:
            reservations = cached_reservations
        else:
            reservations = self._safe_get_taxi_reservations(traci_conn, env=env)

        # Initialize stale reservation tracker
        if "stale_reservation_first_seen" not in control_state:
            control_state["stale_reservation_first_seen"] = {}
        stale_tracker = control_state["stale_reservation_first_seen"]

        cleaned_count = 0
        state_2_count = 0

        for res in reservations:
            # Handle both reservation objects and strings
            if isinstance(res, str):
                continue
            try:
                res_id = self._normalize_reservation_id(res.id)
                state = getattr(res, "state", None)
            except AttributeError:
                continue
            state_int = self._reservation_state_int(state)

            # Check for state 2 (retrieved) reservations
            # State 2 means the reservation was retrieved but taxi hasn't picked up yet
            if state_int is not None and (state_int & 2) != 0 and (state_int & 8) == 0:
                state_2_count += 1

                # Track when we first saw this reservation in state 2
                if res_id not in stale_tracker:
                    stale_tracker[res_id] = current_time

                first_seen = stale_tracker[res_id]
                age = current_time - first_seen

                if age >= max_age:
                    # This reservation has been stuck too long
                    # CRITICAL FIX (SUMO bug #9805): Do NOT re-dispatch stale reservations
                    # Re-dispatching can trigger SUMO crash due to internal state corruption.
                    # Instead, just clean up our tracking and let the reservation expire naturally.

                    # Remove from pending dispatches if present
                    pending = control_state.get("pending_dispatches", [])
                    new_pending = []
                    for item in pending:
                        item_res_ids = [self._normalize_reservation_id(r) for r in (item.get("reservation_ids") or [])]
                        if res_id not in item_res_ids:
                            new_pending.append(item)
                    control_state["pending_dispatches"] = new_pending

                    # Remove from reservation_to_taxi mapping
                    if res_id in control_state.get("reservation_to_taxi", {}):
                        del control_state["reservation_to_taxi"][res_id]

                    # Track as unserved for metrics
                    control_state["unserved_reservations"] = control_state.get("unserved_reservations", 0) + 1
                    cleaned_count += 1
                    print(f"[TaxiScheduling] Cleaned stale reservation {res_id} (age={age:.0f}s) - NOT re-dispatching to prevent SUMO crash")

                    # Remove from stale tracker since we've handled it
                    if res_id in stale_tracker:
                        del stale_tracker[res_id]
            else:
                # Not in state 2 anymore, remove from tracker
                if res_id in stale_tracker:
                    del stale_tracker[res_id]

        if cleaned_count > 0:
            print(f"[TaxiScheduling] Cleaned up {cleaned_count} stale reservations "
                  f"(state_2_count: {state_2_count}, max_age: {max_age}s)")

        return cleaned_count

    def _sync_reservations(self, env: Any, control_state: Dict[str, Any], cached_reservations: Optional[List[Any]] = None) -> None:
        """
        Sync reservation tracker with current SUMO state.
        Critically, this detects reservations that existed before this checkpoint (boundary case)
        and adopts them so they aren't lost.

        Args:
            env: SUMOEnv instance
            control_state: Control state dictionary
            cached_reservations: Optional pre-fetched reservations list to avoid redundant TraCI calls
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return

        traci = env.traci_conn

        # Initialize trackers if missing
        if "reservation_tracker" not in control_state:
            control_state["reservation_tracker"] = {}
        if "known_reservations" not in control_state:
            control_state["known_reservations"] = set()

        # Use cached reservations if provided, otherwise fetch (with retry)
        if cached_reservations is not None:
            active_reservations = cached_reservations
        else:
            # IMPORTANT: traci.person.getTaxiReservations(onlyNew) does NOT filter by reservation state.
            # onlyNew=0 returns ALL currently active reservations.
            active_reservations = self._safe_get_taxi_reservations(traci, env=env)

        current_time = env.get_current_time()

        for res in active_reservations:
            # Handle both reservation objects and strings
            if isinstance(res, str):
                # Skip strings - we need actual reservation objects
                continue

            # Handle both SUMO Reservation and SimulatedReservation
            if isinstance(res, SimulatedReservation):
                res_id = res.id
                res_reservation_time = res.reservation_time
                res_state = res.state
                res_persons = res.person_ids
            else:
                try:
                    res_id = self._normalize_reservation_id(res.id)
                except AttributeError:
                    continue
                res_reservation_time = getattr(res, "reservationTime", None)
                res_state = getattr(res, "state", None)
                res_persons = list(res.persons) if hasattr(res, "persons") else []

            if res_id not in control_state["known_reservations"]:
                # Found a reservation we didn't know about! (The Boundary Case)
                # Adopt it.
                control_state["reservation_tracker"][res_id] = {
                    "id": res_id,
                    "reservation_time": res_reservation_time,
                    "first_seen_time": current_time,
                    "state": res_state,
                    "pickup_time": None,
                    "dropoff_time": None,
                    "persons": res_persons
                }
                control_state["known_reservations"].add(res_id)
                # print(f"DEBUG: Adopted existing reservation {res_id} from t={res.reservationTime}")

    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update metrics with current step data.
        
        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            **kwargs: Additional arguments (e.g., step_duration) - not used by this module
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return
            
        # Access control state for persistent tracking.
        # NOTE: update_metrics receives 'metrics' dict, but we really need 'control_state' 
        # to track reservation lifecycle across steps.
        # However, run_controlled_simulation calls this.
        # We might need to rely on the module having reference to control_state? 
        # No, the interface is stateless.
        # The 'metrics' dict is reset every checkpoint if not careful, 
        # BUT 'control_state' is passed to 'apply_control'.
        # 'update_metrics' is called in simulation loop.
        
        # We have a design issue here: update_metrics doesn't get control_state.
        # We need to move reservation tracking to apply_control OR 
        # monkey-patch/update the interface. 
        # Current cleanest way: We can fetch control_state if we access it via the env or if we
        # simply use 'apply_control' to do the heavy lifting of state tracking, 
        # and 'update_metrics' just snapshots simple gauges.
        
        # actually, the metrics dict passed here IS persistent for the checkpoint duration.
        # But we need persistence ACROSS checkpoints for reservation lifecycle.
        # So tracking MUST happen in 'control_state' inside 'apply_control'.
        
        # However, we want to pop "passenger_wait_times" into the metrics for reporting.
        # So:
        # 1. apply_control does the lifecycle logic and updates 'control_state'.
        # 2. apply_control pushes "events" (like pickup_complete) to a temporary list in control_state.
        # 3. update_metrics reads those events and moves them to 'metrics'.
        
        # Wait, 'apply_control' is only called when we strictly apply control.
        # If simulation runs without control steps (e.g. baseline), we miss data?
        # But here we are running WITH taxi control.
        
        # Let's do the tracking in this method but we need a way to store "State" 
        # that persists. 'self' persists for the life of the python object (the simulation run).
        # So we can use 'self.reservations_tracker' initialized in __init__ 
        # IF the module instance is reused. 
        # In run_taxi_scheduling, 'env' is recreated, but 'control_configs' persist?
        # Simulation_utils: "module = get_control_module(module_name)". 
        # This re-instantiates the module if it's not cached? 
        # No, registry usually keeps instances or classes.
        
        # Let's rely on 'apply_control' being called.
        # We will add a helper 'track_reservation_events' called from apply_control.
        
        pass

    def track_reservation_events(self, env: Any, control_state: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Updates reservation states and metrics. Called from apply_control.

        Uses cached reservation data from control_state if available to avoid redundant TraCI calls.
        """
        # Guard: Check if TraCI connection is valid
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return

        traci = env.traci_conn

        # Guard: Check if TraCI connection is healthy
        if hasattr(env, 'is_traci_healthy') and not env.is_traci_healthy():
            return

        try:
            current_time = env.get_current_time()
        except Exception:
            return

        # Use cached reservations if available (set by apply_control)
        cached_reservations = control_state.get("_cached_reservations")

        # Sync first to catch boundary reservations (pass cached data to avoid redundant TraCI call)
        self._sync_reservations(env, control_state, cached_reservations=cached_reservations)

        tracker = control_state.get("reservation_tracker", {})
        if not tracker:
            return

        # Use cached reservation dict if available, otherwise fetch safely
        current_reservations: Dict[str, Any] = control_state.get("_cached_reservation_dict")
        if current_reservations is None:
            # Fallback: fetch with safe method
            current_reservations = {}
            for res in self._safe_get_taxi_reservations(traci, env=env):
                res_id = self._normalize_reservation_id(res.id)
                current_reservations[res_id] = res

        # Best-effort sync of reservation->taxi mapping for active reservations
        self._sync_reservation_to_taxi_mapping(traci, control_state, set(current_reservations.keys()), env=env)

        for res_id, info in list(tracker.items()):
            # Update last seen regardless of current existence
            info["last_seen"] = current_time

            res = current_reservations.get(res_id)
            if res is None:
                # Reservation disappeared. If we had already detected pickup, treat as dropoff completion.
                if info.get("pickup_time") is not None and info.get("dropoff_time") is None:
                    info["dropoff_time"] = current_time
                    pickup_time = info.get("pickup_time", current_time)
                    travel_time = current_time - pickup_time

                    control_state["passenger_dropoffs"] = control_state.get("passenger_dropoffs", 0) + 1
                    control_state.setdefault("passenger_travel_times", []).append(travel_time)

                    # Income update (proxy fare). Fleet-level income should be credited for
                    # every completed trip; per-taxi income is best-effort when the reservation
                    # to taxi mapping is available.
                    fare = self._estimate_fare(travel_time, control_state)
                    control_state["total_income"] = float(control_state.get("total_income", 0.0)) + fare
                    taxi_id = (control_state.get("reservation_to_taxi") or {}).get(res_id)
                    if taxi_id:
                        taxi_income = control_state.setdefault("taxi_income", {})
                        taxi_income[taxi_id] = float(taxi_income.get(taxi_id, 0.0)) + fare
                    # Clean up mapping to avoid unbounded growth
                    try:
                        control_state.get("reservation_to_taxi", {}).pop(res_id, None)
                    except Exception:
                        pass
                continue

            # Handle both SUMO Reservation and SimulatedReservation
            if isinstance(res, SimulatedReservation):
                current_state = res.state
                res_reservation_time = res.reservation_time
            else:
                current_state = getattr(res, "state", None)
                res_reservation_time = getattr(res, "reservationTime", None)

            info["state"] = current_state

            # Keep reservation_time in sync
            if info.get("reservation_time") is None and res_reservation_time is not None:
                info["reservation_time"] = res_reservation_time

            # Pickup detection (transition into "picked up" state)
            if info.get("pickup_time") is None and self._reservation_is_picked_up(current_state):
                info["pickup_time"] = current_time
                reservation_time = info.get("reservation_time", current_time)
                wait_time = current_time - reservation_time

                control_state["passenger_pickups"] = control_state.get("passenger_pickups", 0) + 1
                control_state.setdefault("passenger_wait_times", []).append(wait_time)

    def update_metrics(self, metrics: Dict[str, Any], env: Any, reward: Optional[List[float]] = None, **kwargs) -> None:
        """
        Legacy update_metrics - we just gather gauges here.
        Event tracking is done in apply_control now via track_reservation_events.
        
        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            **kwargs: Additional arguments (e.g., step_duration) - not used by this module
        """
        if not hasattr(env, 'traci_conn') or env.traci_conn is None:
            return
        
        try:
            fleet_state = self.get_taxi_fleet_state(env)
            if "error" not in fleet_state:
                metrics["fleet_utilizations"].append(fleet_state.get("utilization_rate", 0))
                metrics["idle_counts"].append(fleet_state.get("idle_count", 0))
                
            reservations = self.get_pending_reservations(env)
            if "error" not in reservations:
                metrics["pending_reservation_counts"].append(reservations.get("pending_count", 0))
                
        except Exception as e:
            print(f"Warning: Failed to update taxi metrics: {e}")
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any,
        control_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate final results and metrics for taxi scheduling.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            control_state: Optional control state dictionary containing cumulative counters
            
        Returns:
            Dictionary with final metric values
        """
        import numpy as np
        
        if control_state is None:
            control_state = {}

        results = {}
        
        # Use counters from control_state if available (more accurate for cumulative events)
        if control_state:
            results["total_dispatches"] = control_state.get("total_dispatches", 0)
            # Count successful dispatches from completed_dispatches list if available
            if "completed_dispatches" in control_state:
                results["successful_dispatches"] = len(control_state["completed_dispatches"])
            else:
                 # Fallback to metrics if not in control_state (unlikely given logic)
                results["successful_dispatches"] = metrics.get("successful_dispatches", 0)
        else:
            # Fallback to step-wise metrics (might be inaccurate for sparse events)
            results["total_dispatches"] = metrics.get("total_dispatches", 0)
            results["successful_dispatches"] = metrics.get("successful_dispatches", 0)
        
        # Calculate averages
        if metrics.get("fleet_utilizations"):
            results["avg_fleet_utilization"] = float(np.mean(metrics["fleet_utilizations"]))
        else:
            results["avg_fleet_utilization"] = 0.0
        
        if metrics.get("idle_counts"):
            results["avg_idle_count"] = float(np.mean(metrics["idle_counts"]))
        else:
            results["avg_idle_count"] = 0.0
            
        # PASSENGER METRICS
        results["passenger_pickups"] = control_state.get("passenger_pickups", 0)
        results["passenger_dropoffs"] = control_state.get("passenger_dropoffs", 0)
        
        wait_times = control_state.get("passenger_wait_times", [])
        if wait_times:
            results["avg_wait_time"] = float(np.mean(wait_times))
            results["max_wait_time"] = float(np.max(wait_times))
        else:
            results["avg_wait_time"] = None # Indicating NO DATA
            
        travel_times = control_state.get("passenger_travel_times", [])
        if travel_times:
            results["avg_travel_time"] = float(np.mean(travel_times))
        else:
            results["avg_travel_time"] = None
        
        if metrics.get("pending_reservation_counts"):
            results["avg_pending_reservations"] = float(np.mean(metrics["pending_reservation_counts"]))
        else:
            results["avg_pending_reservations"] = 0.0

        # FAIRNESS / INCOME METRICS (cross-checkpoint)
        income_map = control_state.get("taxi_income", {}) or {}
        incomes = [float(v) for v in income_map.values()] if income_map else []
        fleet_total_income = float(control_state.get("total_income", 0.0) or 0.0)
        if incomes:
            vals = np.array(incomes, dtype=float)
            results["total_income"] = float(max(fleet_total_income, float(np.sum(vals))))
            results["avg_income_per_taxi"] = float(np.mean(vals))
            results["income_std"] = float(np.std(vals))

            if float(np.sum(vals)) <= 0.0:
                results["income_gini"] = 0.0
            else:
                sorted_vals = np.sort(vals)
                n = sorted_vals.size
                cum_vals = np.cumsum(sorted_vals)
                gini = (n + 1 - 2 * np.sum(cum_vals) / cum_vals[-1]) / n
                results["income_gini"] = float(gini)
        else:
            results["total_income"] = fleet_total_income
            results["avg_income_per_taxi"] = 0.0
            results["income_std"] = 0.0
            results["income_gini"] = None

        return results
