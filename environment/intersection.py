import os
import time
import math
import numpy as np
from multiprocessing import Process
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any

import traci
import sumolib

# Global dictionaries (can be part of config or discovered)
location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {v: k for k, v in location_dict.items()}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

# Angles represent the direction of travel (Heading) towards the intersection (calculated via atan2(dy, dx))
angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]  # Eastbound, Northbound, Westbound, Southbound, Eastbound
# Orients map these Headings to their Origin (Standard Convention)
orients = ['W', 'S', 'E', 'N', 'W', 'S', 'E', 'N']

DEFAULT_YELLOW_TIME = 5  # Default yellow time if not specified

class Intersection:
    """
    Represents a single intersection in the simulation environment, adapted for SUMO.
    Handles state updates and signal control for this intersection.
    It dynamically discovers its topology from a SUMO network and controls via TraCI.
    """
    # _MOVEMENT_TO_PRESSURE_IDX_MAP removed - no longer needed for RL training

    def __init__(self, tls_id, dic_traffic_env_conf, traci_conn, sumo_net, path_to_log, custom_phase_list=None):
        """
        Initializes an Intersection object based on a SUMO traffic light system (TLS).

        Args:
            tls_id (str): The ID of the traffic light system in SUMO.
            dic_traffic_env_conf (dict): The traffic environment configuration dictionary.
            traci_conn (traci.connection): The active TraCI connection object.
            sumo_net (sumolib.net.Net): The pre-parsed sumolib network object.
            path_to_log (str): Path to the directory for logging.
            adjacency_info (dict): Information about neighboring intersections.
            custom_phase_list (list, optional): A list of phase name strings restricting the agent's choices.
        """
        # 1. Store injected dependencies and set compatibility attributes
        self.tls_id = tls_id
        self.inter_id = tls_id
        self.inter_name = tls_id
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.traci_conn = traci_conn
        self.sumo_net = sumo_net
        self.path_to_log = path_to_log
        self.custom_phase_list = custom_phase_list

        # 2. Initial Validation: Ensure the TLS ID is valid
        try:
            if self.tls_id not in self.traci_conn.trafficlight.getIDList():
                raise ValueError(f"TLS ID '{self.tls_id}' not found in the running SUMO simulation.")
        except Exception as e:
             raise ValueError(f"Failed to validate TLS ID '{self.tls_id}': {e}") from e

        # --- Declare attributes that will be populated by _build_conceptual_model ---
        self.virtual = False
        self.point = {}
        self.phases = [] # Raw SUMO phase definitions
        self.control_phases = [] # Final list of phase NAMES available to agent
        self.all_control_phase_names = [] # All detected phase NAMES
        self.phase_name_2_cityflow_idx = {} # Map name string to actual SUMO phase index
        self.green_phases = [] # List of actual SUMO green phase indices

        self.incoming_roads = {}
        self.outgoing_roads = {}
        self.list_entering_lanes = [] # Padded, canonical lists
        self.list_exiting_lanes = [] # Padded, canonical lists
        self.lane_to_road = {}
        self.list_lanes = []
        self.road_id_2_orient = {}
        self.action_2_phase_index = {} # action_idx (0,1..) -> SUMO phase_idx
        self.phase_index_2_action_idx = {}

        self.yellow_time = self.dic_traffic_env_conf.get("YELLOW_TIME", DEFAULT_YELLOW_TIME)
        self.yellow_phase_index = -1 # Placeholder for logging/internal state
        self.yellow_all_red_phase_index = -1 # Index of YELLOW_ALL_RED phase
        self.yellow_all_red_duration = 0 # Duration of YELLOW_ALL_RED phase in seconds
        self.pending_target_phase_index = -1 # Target phase to switch to after YELLOW_ALL_RED
        self.yellow_all_red_start_time = 0 # Time when YELLOW_ALL_RED phase started

        # 3. Build the conceptual model from SUMO data
        self._build_conceptual_model()

        # --- State Variables ---
        # Queue count removed - now calculated directly from system_states in signal_timing.py

        # Feature storage removed - no longer needed for RL training

        # --- Signal Timing ---
        self.current_phase_index = 0
        try:
             # Get the initial phase from SUMO
             self.current_phase_index = self.traci_conn.trafficlight.getPhase(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: Could not get initial phase for {self.tls_id}. Defaulting to 0. Error: {e}")

        self.default_phase_index = self.current_phase_index
        self.previous_phase_index = self.current_phase_index
        self.next_phase_to_set_index = self.current_phase_index

        # for SUMO to complete its internal yellow/red transition.
        self.is_in_transition = False
        self.fixed_time_cycle_index = 0

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['traci_conn', 'sumo_net', 'incoming_roads', 'outgoing_roads']:
                continue
            setattr(result, k, deepcopy(v, memo))

        result.traci_conn = None
        result.sumo_net = self.sumo_net
        result.incoming_roads = self.incoming_roads.copy()
        result.outgoing_roads = self.outgoing_roads.copy()

        return result

    def _build_conceptual_model(self):
        """
        Orchestrates the discovery of intersection topology (lanes, roads, orientations)
        and traffic light logic (phases, movements) from the SUMO simulation.
        """
        # Get basic intersection info from the pre-parsed network object
        try:
            tls_node = self.sumo_net.getNode(self.tls_id)
            self.point = {"x": tls_node.getCoord()[0], "y": tls_node.getCoord()[1]}
        except KeyError:
            # Fallback if TLS node is not found in the static network file
            self.point = {"x": 0, "y": 0}

        # Step 1: Discover all connected lanes and roads
        self._discover_lanes_and_roads()

        # Step 2: Determine road orientations based on geometry
        self._calculate_road_orientations()

        # Step 3: Build the legacy `road_links` structure.
        self._build_road_links()

        # Step 4: Create padded, canonical lane lists for feature calculation.
        self._create_canonical_lane_lists()

        # Step 5: Discover SUMO phases and map them to canonical names
        self._discover_phases_and_movements()

        # Step 6: Apply custom phase filtering if provided.
        self._apply_custom_phase_mapping()

    def _discover_lanes_and_roads(self):
        """
        Populates lane and road lists using TraCI and the sumolib network object.
        """
        # Get all lanes controlled by this traffic light
        try:
            # Note: getControlledLanes returns ALL lanes, including internal ones.
            self.list_entering_lanes = list(set(self.traci_conn.trafficlight.getControlledLanes(self.tls_id)))
        except traci.TraCIException as e:
            print(f"Warning: Could not get controlled lanes for {self.tls_id}. Error: {e}")
            self.list_entering_lanes = []
            return

        if not self.list_entering_lanes:
            return

        # Discover incoming/outgoing roads and all lanes
        temp_outgoing_lanes = set()

        for lane_id in self.list_entering_lanes:
            try:
                lane_obj = self.sumo_net.getLane(lane_id)
                road_obj = lane_obj.getEdge()
                road_id = road_obj.getID()

                # Ignore internal edges
                if not road_id.startswith(":"):
                    self.incoming_roads[road_id] = road_obj
                    self.lane_to_road[lane_id] = road_id

                # Find corresponding outgoing lanes from this incoming lane (sumolib)
                outgoing_lanes = lane_obj.getOutgoingLanes()  # list[sumolib.net.Lane]

                for outgoing_lane in outgoing_lanes:
                    outgoing_lane_id = outgoing_lane.getID()
                    outgoing_road_obj = outgoing_lane.getEdge()
                    outgoing_road_id = outgoing_road_obj.getID()

                    # Ensure it's not an internal junction edge
                    if not outgoing_road_id.startswith(":"):
                        temp_outgoing_lanes.add(outgoing_lane_id)
                        self.outgoing_roads[outgoing_road_id] = outgoing_road_obj
                        self.lane_to_road[outgoing_lane_id] = outgoing_road_id
            except (KeyError, AttributeError) as e:
                pass

        self.list_exiting_lanes = sorted(list(temp_outgoing_lanes))
        self.list_lanes = sorted(list(set(self.list_entering_lanes + self.list_exiting_lanes)))

    def _calculate_road_orientations(self):
        """
        Calculates and assigns canonical orientations (N, S, E, W) to incoming and
        outgoing roads based on their geometry from the sumolib network.
        """
        def _calculate_for(roads_dict, is_incoming):
            center_x, center_y = self.point['x'], self.point['y']
            roads_with_angle = []

            for road_id, road_obj in roads_dict.items():
                shape = road_obj.getShape()
                if len(shape) < 2:
                    continue

                point_x, point_y = shape[-2] if is_incoming else shape[1]

                dx = center_x - point_x
                dy = center_y - point_y
                angle = math.atan2(dy, dx)
                if angle < 0: angle += 2 * math.pi

                # Find the closest cardinal direction
                orient_angle_diffs = np.abs(np.subtract(angles, angle))
                orient_index = np.argmin(orient_angle_diffs)
                orient = orients[orient_index]
                orient_angle_diff = orient_angle_diffs[orient_index]

                roads_with_angle.append(
                    {'id': road_id, 'angle': angle, 'orient': orient, 'angle_diff': orient_angle_diff}
                )

            if not roads_with_angle:
                return {}

            roads_with_angle.sort(key=lambda x: x['angle'])
            # Use the min angle diff for starting point
            min_orient_road = min(roads_with_angle, key=lambda x: x['angle_diff'])
            min_orient_road_index = roads_with_angle.index(min_orient_road)
            roads_with_angle = roads_with_angle[min_orient_road_index:] + roads_with_angle[:min_orient_road_index]

            if not self._decide_road_orient(roads_with_angle):
                # 即使失败，也返回初始的方向分配
                return {road['id']: road.get('orient') for road in roads_with_angle}

            result = {road['id']: road.get('orient') for road in roads_with_angle}
            return result

        self.road_id_2_orient['incoming'] = _calculate_for(self.incoming_roads, is_incoming=True)
        self.road_id_2_orient['outgoing'] = _calculate_for(self.outgoing_roads, is_incoming=False)

    def _build_road_links(self):
        """
        Builds the `road_links` structure by relying on SUMO's connection definitions (via sumolib)
        """
        self.road_links = []
        links_grouped = defaultdict(list)

        try:
            controlled_links = self.traci_conn.trafficlight.getControlledLinks(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: getControlledLinks failed for {self.tls_id}: {e}")
            return

        # Mapping SUMO internal directions ('s', 'l', 'r', etc.) to standardized turn types
        SUMO_DIR_MAP = {
            's': 'go_straight',
            't': 'turn_left',  # 't' (turn) is often treated as slight left/through
            'l': 'turn_left',
            'L': 'turn_left',
            'r': 'turn_right',
            'R': 'turn_right',
        }

        for link_group in controlled_links:
            if not link_group:
                continue
            from_lane_id, to_lane_id, _via = link_group[0]

            if from_lane_id.startswith(":") or to_lane_id.startswith(":"):
                continue

            try:
                from_lane_obj = self.sumo_net.getLane(from_lane_id)
                to_lane_obj = self.sumo_net.getLane(to_lane_id)
                from_road_id = from_lane_obj.getEdge().getID()
                to_road_id = to_lane_obj.getEdge().getID()

                # Find the specific connection object between these two lanes in sumolib
                connection = None
                for conn in from_lane_obj.getOutgoing():
                    if conn.getToLane() == to_lane_obj:
                        connection = conn
                        break

                if connection is None:
                    continue

                # Get the direction attribute from the connection
                sumo_direction = connection.getDirection()
                turn_type = SUMO_DIR_MAP.get(sumo_direction)

                if turn_type is None:
                    continue # Skip unknown/unsupported types (e.g., U-turns 'u')

                # We still need the orientation (now correctly calculated) to categorize the movement
                f_or = self.road_id_2_orient['incoming'].get(from_road_id)

                if not f_or:
                    continue

                start_lane_idx = from_lane_obj.getIndex()
                end_lane_idx = to_lane_obj.getIndex()

                links_grouped[(from_road_id, to_road_id, turn_type)].append({
                    "startLaneIndex": start_lane_idx,
                    "endLaneIndex": end_lane_idx
                })

            except (KeyError, AttributeError) as e:
                continue

        for (start_road, end_road, turn_type), lane_links in links_grouped.items():
            # Remove duplicate lane links and sort for stability
            unique_lane_links = [dict(t) for t in {tuple(d.items()) for d in lane_links}]
            # Sort by startLaneIndex primarily for consistency
            unique_lane_links.sort(key=lambda x: (x['startLaneIndex'], x.get('endLaneIndex', -1)))
            self.road_links.append({
                "startRoad": start_road,
                "endRoad": end_road,
                "type": turn_type,
                "laneLinks": unique_lane_links
            })

    def _create_canonical_lane_lists(self):
        """
        Creates 12-element, canonically ordered (W,E,N,S approach, 3 lanes each)
        lists for entering and exiting lanes. Non-existent lanes are filled with None.
        This is required for compatibility with feature calculation methods that expect a fixed-size input.
        """
        # Ensure we have a valid road_id_2_orient map
        if not self.road_id_2_orient or not 'incoming' in self.road_id_2_orient:
            print(f"Warning: {self.inter_id} Missing orientation data for canonical lane list creation. Using default empty lists.")
            self.list_entering_lanes = [None] * 12
            self.list_exiting_lanes = [None] * 12
            return

        padded_entering_lanes = [None] * 12
        padded_exiting_lanes = [None] * 12

        # --- Pad Entering Lanes ---
        lanes_by_orient = defaultdict(list)
        for road_id, orient in self.road_id_2_orient.get('incoming', {}).items():
            road_obj = self.incoming_roads.get(road_id)
            if road_obj:
                for lane_obj in road_obj.getLanes():
                    lanes_by_orient[orient].append(lane_obj.getID())

        orient_map = {'W': 0, 'E': 3, 'N': 6, 'S': 9}
        for orient, offset in orient_map.items():
            lanes = sorted(lanes_by_orient.get(orient, []))
            for i in range(3):
                if i < len(lanes):
                    padded_entering_lanes[offset + i] = lanes[i]

        # --- Pad Exiting Lanes ---
        exiting_lanes_by_orient = defaultdict(list)
        for road_id, orient in self.road_id_2_orient.get('outgoing', {}).items():
             road_obj = self.outgoing_roads.get(road_id)
             if road_obj:
                for lane_obj in road_obj.getLanes():
                    exiting_lanes_by_orient[orient].append(lane_obj.getID())

        for orient, offset in orient_map.items():
            lanes = sorted(exiting_lanes_by_orient.get(orient, []))
            for i in range(3):
                if i < len(lanes):
                    padded_exiting_lanes[offset + i] = lanes[i]

        # Overwrite the instance attributes with the padded, canonical lists
        self.list_entering_lanes = padded_entering_lanes
        self.list_exiting_lanes = padded_exiting_lanes

        # Update the list_lanes to include all padded lanes
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # Remove None values from the list_lanes
        self.list_lanes = [lane for lane in self.list_lanes if lane is not None]


    def _discover_phases_and_movements(self):
        """
        Discovers the traffic light program from SUMO, identifies green phases,
        and translates them into canonical movement-based names (e.g., "ETWT")
        by READING THE PHASE NAME ATTRIBUTE.
        """
        try:
            # Get the first (active) program definition
            logic = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            self.phases = logic.getPhases()
        except (traci.TraCIException, IndexError) as e:
            print(f"Warning: Could not get TLS logic for '{self.tls_id}'. Phase control disabled. Error: {e}")
            return

        phase_dict = {}  # cityflow_idx -> set(movement_names_like_WT)
        phase_name_to_idx_map = {}
        temp_control_phase_names = []

        for phase_idx, phase in enumerate(self.phases):
            # Check for YELLOW_ALL_RED phase and record its index and duration
            phase_name_str = phase.name
            if phase_name_str == "YELLOW_ALL_RED":
                self.yellow_all_red_phase_index = phase_idx
                self.yellow_all_red_duration = phase.duration
                # print(f"Found YELLOW_ALL_RED phase for {self.tls_id}: index={phase_idx}, duration={phase.duration}s")
                continue

            # A phase is green if it has a 'G' or 'g' and is not a short transition phase.
            if ('G' in phase.state or 'g' in phase.state) and phase.minDur > 2:
                self.green_phases.append(phase_idx)
                # 1. Read the phase name directly from the object. This is the name
                #    set by convert_sumo_roadnet_phases.py script.
                # phase_name_str already set above

                # 2. Skip phases that are unnamed.
                #    This makes the logic robust and ignores transitional phases.
                if not phase_name_str:
                    continue

                # 3. Create a set of movements from the name for the legacy `phase_dict`
                #    This assumes names are pairs of characters, e.g., "ETWT" -> {"ET", "WT"}
                movement_names = set()
                if len(phase_name_str) % 2 == 0:
                    for i in range(0, len(phase_name_str), 2):
                        movement_names.add(phase_name_str[i:i + 2])

                if not movement_names:
                    continue

                phase_dict[phase_idx] = movement_names

                # 4. Use the clean, correct name for agent control.
                if phase_name_str not in phase_name_to_idx_map:
                    temp_control_phase_names.append(phase_name_str)
                    phase_name_to_idx_map[phase_name_str] = phase_idx


        self.phase_index_2_phase_name = phase_dict
        self.all_control_phase_names = sorted(list(set(temp_control_phase_names)))
        self.phase_name_2_cityflow_idx = phase_name_to_idx_map
        if self.green_phases:
            # Set a reasonable default green phase
            if self.all_control_phase_names:
                first_phase_name = self.all_control_phase_names[0]
                self.default_phase_index = self.phase_name_2_cityflow_idx.get(first_phase_name, self.green_phases[0])
            else:
                 self.default_phase_index = self.green_phases[0]

    # Helper methods for road orientation calculation (used in _calculate_road_orientations)
    def _get_opposite_road(self, cur_road, roads_with_angle, orients_taken={}):
        """
        Finds the opposite road for a given road based on its orientation.
        """
        for orient, road_idx in orients_taken.items():
            road = roads_with_angle[road_idx]
            if road['orient'] and cur_road['id'] != road['id'] and abs(
                    abs(cur_road['angle'] - road['angle']) - math.pi) < math.pi / 8:
                return road_idx
        return -1

    def _decide_road_orient(self, roads_with_angle, last_road=None, orients_taken={}, cur_index=0):
        if cur_index == len(roads_with_angle):
            return True

        cur_road = roads_with_angle[cur_index]

        my_possible_orients = []
        cur_possible_dir_index = 0 if last_road is None else (orients.index(last_road['orient']) + 1)
        for i in range(cur_possible_dir_index, cur_possible_dir_index + 4):
            if orients[i % 4] in orients_taken or (not last_road is None and orients[i % 4] == last_road['orient']):
                break
            my_possible_orients.append(orients[i % 4])

        if len(my_possible_orients) == 0:
            return False

        my_fav_orient_index = my_possible_orients.index(cur_road['orient']) if cur_road[
                                                                                   'orient'] in my_possible_orients else 0
        opposite_road_index = self._get_opposite_road(cur_road, roads_with_angle, orients_taken)
        if opposite_road_index != -1:
            opposite_orient = orients[(orients.index(roads_with_angle[opposite_road_index]['orient']) + 2) % 4]
            if opposite_orient in my_possible_orients:
                my_fav_orient_index = my_possible_orients.index(opposite_orient)
        my_possible_orients = my_possible_orients[my_fav_orient_index:] + my_possible_orients[:my_fav_orient_index]

        for dir in my_possible_orients:
            cur_road['orient'] = dir
            _orients_taken = orients_taken.copy()
            _orients_taken[dir] = cur_index
            if self._decide_road_orient(roads_with_angle, cur_road, _orients_taken, cur_index + 1):
                return True

        return False

    def _apply_custom_phase_mapping(self):
        """
        Filters the detected phases based on self.custom_phase_list.
        Sets the final self.control_phases and self.action_2_phase_index.
        """
        source_phase_names = self.all_control_phase_names
        name_to_idx_map = self.phase_name_2_cityflow_idx
        final_control_phase_names = []

        if self.custom_phase_list is not None:
            available_names_set = set(source_phase_names)
            for name in self.custom_phase_list:
                if name in available_names_set:
                    if name not in final_control_phase_names:
                        final_control_phase_names.append(name)
                else:
                    print(f"  Warning: Custom phase '{name}' not found in detected phases {source_phase_names}.")

        else:
            final_control_phase_names = source_phase_names

        if not final_control_phase_names and self.green_phases:
            final_control_phase_names = []
            for idx in self.green_phases:
                dummy_name = f"Index_{idx}"
                final_control_phase_names.append(dummy_name)
                name_to_idx_map[dummy_name] = idx

        self.control_phases = final_control_phase_names
        self.action_2_phase_index = {}
        self.phase_index_2_action_idx = {}
        allowed_sumo_indices = set()

        for i, phase_name in enumerate(self.control_phases):
            sumo_idx = name_to_idx_map.get(phase_name)
            if sumo_idx is not None:
                self.action_2_phase_index[i] = sumo_idx
                self.phase_index_2_action_idx[sumo_idx] = i
                allowed_sumo_indices.add(sumo_idx)

        original_green_phases = list(self.green_phases)
        self.green_phases = [idx for idx in original_green_phases if idx in allowed_sumo_indices]


    def set_signal(self, action, action_pattern):
        """
        Sets the traffic signal phase for the intersection in SUMO.

        Direct phase switching: phases switch directly without yellow/red transitions.
        """
        if not self.phases:
            return

        if action_pattern == "set":
            if action == -1:
                target_phase_index = self.current_phase_index
            elif not self.control_phases or not self.action_2_phase_index:
                target_phase_index = self.current_phase_index
            elif action >= 0:
                num_actions = len(self.control_phases)
                if num_actions > 0:
                    actual_action_index = action % num_actions
                    default_sumo_idx = self.action_2_phase_index.get(0, self.current_phase_index)
                    target_phase_index = self.action_2_phase_index.get(actual_action_index, default_sumo_idx)
                else:
                    target_phase_index = self.current_phase_index
            else:
                target_phase_index = self.current_phase_index

        elif action_pattern == "switch":
            if action == 0:
                target_phase_index = self.current_phase_index
            elif action == 1:
                # Switch to next phase in SUMO program
                target_phase_index = (self.current_phase_index + 1) % len(self.phases)
            else:
                target_phase_index = self.current_phase_index
        else:
            target_phase_index = self.current_phase_index

        # If the determined target phase is different from the current one, command SUMO.
        if target_phase_index != self.current_phase_index and target_phase_index != -1:
            try:
                # Directly switch to target phase (no yellow/red transitions)
                self.traci_conn.trafficlight.setPhase(self.tls_id, target_phase_index)

                # Update internal state to reflect the commanded action
                self.previous_phase_index = self.current_phase_index
                self.current_phase_index = target_phase_index

            except traci.TraCIException as e:
                print(f"TraCIException setting phase {target_phase_index} for {self.inter_name}. Error: {e}")

    def update_current_measurements(self, simulator_state):
        """
        Updates intersection state based on the global simulator state.
        Only updates data needed for signal control and metrics collection.
        """
        # --- Update Phase Duration and Sync Current Phase ---
        try:
            # Sync internal phase index with SUMO's current phase
            self.current_phase_index = self.traci_conn.trafficlight.getPhase(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: Could not sync phase index for {self.tls_id}. Error: {e}")
            pass # Keep previous phase index if sync fails

    def _get_four_phase(self):
        """
        Get the control phases for each intersection.
        """
        available_phases = self.control_phases
        default_allowed_four_phases = ['ETWT', 'NTST', 'ELWL', 'NLSL']
        filtered_2_eight = {}
        eight_2_filtered = {}
        # Use a dictionary to store phases by their default index
        phase_dict = {}

        for idx, phase in enumerate(available_phases):
            if len(phase) == 2:
                for f_idx, f_phase in enumerate(default_allowed_four_phases):
                    if phase in f_phase and f_idx not in filtered_2_eight:
                        filtered_2_eight[f_idx] = idx
                        eight_2_filtered[idx] = f_idx
                        phase_dict[f_idx] = phase

            elif phase in default_allowed_four_phases:
                f_idx = default_allowed_four_phases.index(phase)
                if f_idx not in filtered_2_eight:
                    filtered_2_eight[f_idx] = idx
                    eight_2_filtered[idx] = f_idx
                    phase_dict[f_idx] = phase

        # Build filtered_phases in the order of default_allowed_four_phases
        filtered_phases = [phase_dict[f_idx] for f_idx in sorted(phase_dict.keys())]

        return filtered_2_eight, eight_2_filtered, filtered_phases

    def get_current_time(self):
        """Returns the current simulation time."""
        return self.traci_conn.simulation.getTime()

    # get_feature(), get_state(), and get_reward() methods removed - no longer needed for RL training
