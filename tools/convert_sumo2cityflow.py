"""
Convert traffic configurations between SUMO and CityFlow.

This module provides bidirectional conversion between SUMO and CityFlow formats:
- SUMO to CityFlow: Convert .net.xml and .rou.xml to CityFlow roadnet.json and flow.json
- CityFlow to SUMO: Convert roadnet.json and flow.json to SUMO .net.xml, .rou.xml, and .sumocfg

Part of the SUMO-to-CityFlow conversion code is borrowed from CityFlow:
https://github.com/cityflow-project/CityFlow

Usage:
    # SUMO to CityFlow
    python convert_sumo2cityflow.py --typ s2c --or_sumonet <path> --cityflownet <path> ...
    
    # CityFlow to SUMO
    python convert_sumo2cityflow.py --typ c2s --or_cityflownet <path> --sumonet <path> ...
"""

import os
import sys
import argparse
import json
import copy
import math
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from math import atan2, pi

import sympy
from mpmath import degrees, radians
import xml.etree.cElementTree as ET
import xml.dom.minidom

# ============================================================================
# SUMO Import Setup (Linux only)
# ============================================================================

def _setup_sumo_imports():
    """Setup SUMO library imports for Linux platform."""
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
        return traci, tc, sumolib, Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(sumo_tools)
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
            return traci, tc, sumolib, Connection
        else:
            raise EnvironmentError(
                "SUMO libraries not found. Please set SUMO_HOME environment variable "
                "or install traci as a Python module."
            )

# Import SUMO libraries
try:
    traci, tc, sumolib, Connection = _setup_sumo_imports()
except EnvironmentError as e:
    print(f"Error: {e}")
    sys.exit(1)

# ============================================================================
# Constants
# ============================================================================

U_TURN_AS = "turn_left"
DEBUG = False
TRUE_CORRECTION_LANE = True
SUMO_PROGRAM = True

# Global dictionaries for phase tracking
all_phase_dict = {}
node_outgoing_dict = {}

# ============================================================================
# Utility Functions
# ============================================================================

def get_direction_from_connection(connection):
    """Get direction string from SUMO connection object.
    
    Args:
        connection: SUMO connection object
        
    Returns:
        str: Direction type ('go_straight', 'turn_left', 'turn_right', 'turn_u')
    """
    direction_map = {
        Connection.LINKDIR_STRAIGHT: "go_straight",
        Connection.LINKDIR_TURN: "turn_u",
        Connection.LINKDIR_LEFT: "turn_left",
        Connection.LINKDIR_RIGHT: "turn_right",
        Connection.LINKDIR_PARTLEFT: "turn_left",
        Connection.LINKDIR_PARTRIGHT: "turn_right",
    }
    return direction_map[connection.getDirection()]


def point_tuple_to_dict(point_tuple):
    """Convert (x, y) tuple to dictionary format.
    
    Args:
        point_tuple: Tuple of (x, y) coordinates
        
    Returns:
        dict: Dictionary with 'x' and 'y' keys
    """
    return {"x": point_tuple[0], "y": point_tuple[1]}


def get_filename(netfile, typ='', need_path=True):
    """Generate SUMO filename based on type.
    
    Args:
        netfile: Original filename
        typ: File type ('net', 'nod', 'edg', 'tll', 'rou', 'trip', 'sumocfg')
        need_path: Whether to include full path
        
    Returns:
        str: Generated filename
    """
    filepath, filename_all = os.path.split(netfile)
    filename = filename_all.split('.')
    
    if typ != 'sumocfg':
        if need_path:
            file_res = os.path.join(filepath, filename[0] + '.' + typ + '.xml')
        else:
            file_res = filename[0] + '.' + typ + '.xml'
    else:
        file_res = os.path.join(filepath, filename[0] + '.sumocfg')
    return file_res


def _get_direction(road):
    """Get direction angle of a road.
    
    Args:
        road: Road dictionary with 'points' key
        
    Returns:
        float: Direction angle in radians
    """
    x = road["points"][-2]["x"] - road["points"][-1]["x"]
    y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


def get_start_idx(lists):
    """Calculate start index for phase mapping.
    
    Args:
        lists: Dictionary of phase mappings
        
    Returns:
        dict: Dictionary with (start_idx, count) tuples
    """
    new_lists = {}
    for key, value in lists.items():
        k, v = list(value.keys())[0], list(value.values())[0]
        start_idx = sum([x_v for _, x in lists.items() for x_k, x_v in x.items() if x_k < k])
        new_lists[key] = (start_idx, v)
    return new_lists


def judg_turn_u(roadlink, roads_data):
    """Judge whether a roadlink is a U-turn.
    
    Args:
        roadlink: Roadlink dictionary with 'startRoad' and 'endRoad'
        roads_data: List of road dictionaries
        
    Returns:
        bool: True if U-turn, False otherwise
    """
    start_info = []
    end_info = []
    count = 0
    
    for road in roads_data:
        if road['id'] == roadlink['startRoad']:
            start_info.append(road['startIntersection'])
            start_info.append(road['endIntersection'])
            count += 1
        if road['id'] == roadlink['endRoad']:
            end_info.append(road['startIntersection'])
            end_info.append(road['endIntersection'])
            count += 1
        if count == 2:
            break
    
    if start_info[1] == end_info[0] and start_info[0] == end_info[1]:
        return True  # U-turn
    return False


def sort_roads(roadnet):
    """Sort roads according to NESW direction for each intersection.
    
    Args:
        roadnet: Roadnet dictionary with 'roads' key
        
    Returns:
        dict: Dictionary mapping intersection ID to ordered road IDs
    """
    intersections = {}
    directions = {}
    
    for road in roadnet["roads"]:
        iid = road["endIntersection"]
        if iid not in intersections:
            intersections[iid] = []
            directions[iid] = []
        intersections[iid].append(road)
        directions[iid].append(_get_direction(road))
    
    ordered = {}
    for i, d in zip(intersections.items(), directions.items()):
        assert len(i[1]) == len(d[1])
        order = sorted(range(len(i[1])), key=lambda x: (d[1][x], i[1][x]))
        ordered[i[0]] = [i[1][x]['id'] for x in order]
    return ordered


def filter_roadlinks_by_startedge(roadLinks, lane_id):
    """Filter roadlinks by starting edge and lane.
    
    Args:
        roadLinks: List of roadlink dictionaries
        lane_id: Lane ID string (format: 'edge_id_lane_index')
        
    Returns:
        list: List of (index, roadlink) tuples
    """
    result = []
    edge_id, lane_index = lane_id.rsplit("_", 1)
    
    for index, roadlink in enumerate(roadLinks):
        lane_index_list = []
        for laneLink in roadlink['laneLinks']:
            lane_index_list.append(laneLink['startLaneIndex'])
        lane_index_list = list(set(lane_index_list))
        
        if roadlink['startRoad'] == edge_id and int(lane_index) in lane_index_list:
            result.append((index, roadlink))
    return result

# ============================================================================
# SUMO to CityFlow Conversion Functions
# ============================================================================

def process_edge(edge):
    """Process SUMO edge to generate CityFlow lane information.
    
    Args:
        edge: SUMO edge object
        
    Returns:
        edge: Modified edge object with CityFlow lane information
    """
    lanes = []
    
    if TRUE_CORRECTION_LANE:
        for inx, lane in enumerate(reversed(edge.getLanes())):
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_from_connection(outgoing)
                to_lane = outgoing.getToLane()
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = direction
                lanes.append(new_lane)
            
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    else:
        for lane in edge.getLanes():
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_from_connection(outgoing)
                to_lane = outgoing.getToLane()
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._direction = direction
                lanes.append(new_lane)
            
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    
    edge._cityflow_lanes = lanes[::-1]
    return edge


def _cityflow_get_lane_index_in_edge(lane, edge):
    """Get lane index within edge for CityFlow.
    
    Args:
        lane: Lane object
        edge: Edge object with _cityflow_lanes attribute
        
    Returns:
        int: Lane index
    """
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return i
    raise Exception('Lane in edge not found')


def _cityflow_get_lane_index_in_edge_cor(lane, edge):
    """Get corrected lane index within edge for CityFlow.
    
    Args:
        lane: Lane object
        edge: Edge object with _cityflow_lanes attribute
        
    Returns:
        int: Corrected lane index
    """
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return _lane._cityflow_lane_inx
    raise Exception('Lane in edge not found')


def _is_node_virtual(node, tls_dict):
    """Check if node is virtual (not controlled by traffic light).
    
    Args:
        node: SUMO node object
        tls_dict: Dictionary of traffic light systems
        
    Returns:
        bool: True if virtual, False otherwise
    """
    edges = [edge for edge in node.getIncoming() + node.getOutgoing()]
    ids = list(set([e.getFromNode().getID() for e in edges] + 
                   [e.getToNode().getID() for e in edges]))
    
    # Virtual nodes have <= 2 connected roads or no traffic light
    if len(ids) <= 2 or (node.getID() not in tls_dict and 
                         'GS_' + node.getID() not in tls_dict):
        return True
        return False


def group_connections_by_start_end(connections):
    """Group connections by start road, end road, and direction.
    
    Args:
        connections: List of SUMO connection objects
        
    Returns:
        dict: Dictionary mapping connection key to list of connections
    """
    connection_group_result = defaultdict(list)
    for connection in connections:
        start_road = connection.getFrom()
        end_road = connection.getTo()
        direction = get_direction_from_connection(connection)
        key = "{}|{}|{}".format(start_road.getID(), end_road.getID(), direction)
        connection_group_result[key].append(connection)
    return connection_group_result


def calc_edge_compass_angle_no_modify(edge):
    """Calculate edge compass angle without modifying edge object.
    
    Args:
        edge: SUMO edge object
        
    Returns:
        float: Angle in degrees
    """
    north_ray = sympy.Ray((0, 0), (0, 1))
    edge_ray = sympy.Ray(*edge.getShape()[:2])
    angle = north_ray.closing_angle(edge_ray)
    angle = (angle + 2 * sympy.pi) % (2 * sympy.pi)
    return float(degrees(angle))


def get_road_direction_from_angle(angle):
    """Convert compass angle to cardinal direction.
    
    Args:
        angle: Angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        
    Returns:
        str: Cardinal direction ('N', 'E', 'S', 'W')
    """
    # Normalize angle to 0-360 range
    angle = angle % 360
    
    # Define direction ranges (45 degrees each, centered on cardinal directions)
    if 315 <= angle or angle < 45:
        return 'N'  # North
    elif 45 <= angle < 135:
        return 'E'  # East
    elif 135 <= angle < 225:
        return 'S'  # South
    elif 225 <= angle < 315:
        return 'W'  # West
    else:
        return 'N'  # Default to North


def get_road_direction_mapping(node, net):
    """Get mapping of road IDs to their cardinal directions for a given intersection.
    
    Args:
        node: SUMO node object representing the intersection
        net: SUMO network object
        
    Returns:
        dict: Mapping of road ID to cardinal direction ('N', 'E', 'S', 'W')
    """
    road_directions = {}
    node_coord = node.getCoord()
    
    # Process incoming roads (roads ending at this intersection)
    for edge in node.getIncoming():
        # Calculate direction from road start to intersection
        edge_shape = edge.getShape()
        if len(edge_shape) >= 2:
            # Use the last two points to determine direction approaching the intersection
            start_point = edge_shape[-2]
            end_point = edge_shape[-1]
            
            # Calculate angle from start to end point
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.degrees(math.atan2(dx, dy))
            
            # Convert to compass direction
            direction = get_road_direction_from_angle(angle)
            road_directions[edge.getID()] = direction
    
    # Process outgoing roads (roads starting from this intersection)
    for edge in node.getOutgoing():
        # Calculate direction from intersection to road end
        edge_shape = edge.getShape()
        if len(edge_shape) >= 2:
            # Use the first two points to determine direction leaving the intersection
            start_point = edge_shape[0]
            end_point = edge_shape[1]
            
            # Calculate angle from start to end point
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.degrees(math.atan2(dx, dy))
            
            # Convert to compass direction
            direction = get_road_direction_from_angle(angle)
            road_directions[edge.getID()] = direction
    
    return road_directions


def generate_phase_name(available_roadlinks, roadlinks, road_directions):
    """Generate meaningful phase name based on available roadlinks and directions.
    
    Only includes through (T) and left turn (L) movements, excluding right turns (R) and U-turns (U).
    
    Args:
        available_roadlinks: List of roadlink indices that are green in this phase
        roadlinks: List of all roadlink dictionaries for the intersection
        road_directions: Dictionary mapping road ID to cardinal direction
        
    Returns:
        str: Phase name (e.g., 'ETWT', 'NTST', 'ELWL', etc.)
    """
    if not available_roadlinks:
        return "ALL_RED"
    
    # Collect movement information (only T and L, exclude R and U)
    movements = []
    
    for idx in available_roadlinks:
        if idx < len(roadlinks):
            roadlink = roadlinks[idx]
            start_road = roadlink['startRoad']
            end_road = roadlink['endRoad']
            movement_type = roadlink['type']
            
            # Skip right turns and U-turns
            if movement_type == 'turn_right' or movement_type == 'turn_u':
                continue
            
            # Get directions
            start_dir = road_directions.get(start_road, 'X')
            end_dir = road_directions.get(end_road, 'X')
            
            # Determine movement type abbreviation (only T and L)
            if movement_type == 'go_straight':
                move_abbr = 'T'  # Through/straight
            elif movement_type == 'turn_left':
                move_abbr = 'L'  # Left turn
            else:
                move_abbr = 'T'  # Default to through
            
            # Create movement descriptor
            movement = f"{start_dir}{move_abbr}"
            movements.append(movement)
    
    # Sort movements for consistent naming
    movements = sorted(list(set(movements)))
    
    # Generate phase name
    if len(movements) == 0:
        return "ALL_RED"
    elif len(movements) == 1:
        return movements[0]
    else:
        # Group by movement type (only T and L)
        through_moves = [m for m in movements if m.endswith('T')]
        left_moves = [m for m in movements if m.endswith('L')]
        
        # Create combined name
        name_parts = []
        if through_moves:
            name_parts.extend(through_moves)
        if left_moves:
            name_parts.extend(left_moves)
        
        return ''.join(name_parts) if name_parts else "MIXED"


def process_intersection_simple_phase(intersection):
    """Generate simple phase for intersection (all green or all red).
    
    Args:
        intersection: Intersection dictionary
        
    Returns:
        intersection: Modified intersection with phase information
    """
    if intersection['virtual']:
        return intersection
    
    all_green = {
        "time": 30,
        "availableRoadLinks": intersection['trafficLight']['roadLinkIndices']
    }
    intersection['trafficLight']['lightphases'] = [all_green]
    return intersection


def node_to_intersection(node, tls_dict, edge_dict, net=None):
    """Convert SUMO node to CityFlow intersection format.
    
    Args:
        node: SUMO node object
        tls_dict: Dictionary of traffic light systems
        edge_dict: Dictionary mapping edge ID to lanes
        net: SUMO network object (optional, for direction mapping)
        
    Returns:
        dict: Intersection dictionary in CityFlow format
    """
    node_type = node.getType()
    node_coord = node.getCoord()
    is_virtual = _is_node_virtual(node, tls_dict)
    
    intersection = {
        "id": node.getID(),
        "point": {"x": node_coord[0], "y": node_coord[1]},
        "width": 0 if is_virtual else 15,
        "roads": [edge.getID() for edge in node.getIncoming() + node.getOutgoing()],
        "roadLinks": [],
        "trafficLight": {
            "roadLinkIndices": [],
            "lightphases": []
        },
        "virtual": False,
        "gt_virtual": is_virtual,
    }

    # Process connections into roadLinks
    connections_group = group_connections_by_start_end(node.getConnections())
    roadLinks = intersection['roadLinks']
    
    for k, v in connections_group.items():
        connection_template = v[0]
        start_road = connection_template.getFrom()
        end_road = connection_template.getTo()
        raw_roadlink_type = get_direction_from_connection(connection_template)
        
        roadLink = {
            "type": raw_roadlink_type,
            "startRoad": start_road.getID(),
            "endRoad": end_road.getID(),
            "direction": 0,
            "laneLinks": []
        }
        
        if roadLink["type"] == "turn_u":
            roadLink["type"] = U_TURN_AS

        # Process lane links
        for start_lane in reversed(start_road._cityflow_lanes):
            if start_lane._direction != raw_roadlink_type:
                continue
            
            if TRUE_CORRECTION_LANE:
                for end_inx, end_lane in enumerate(reversed(end_road._lanes)):
                    start_point = point_tuple_to_dict(start_lane.getShape()[-1])
                    end_point = point_tuple_to_dict(end_lane.getShape()[0])
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge_cor(start_lane, start_road),
                        "endLaneIndex": end_inx,
                        "points": [start_point, end_point]
                    }
                    roadLink["laneLinks"].append(path)
            else:
                for end_lane in end_road._cityflow_lanes:
                    start_point = point_tuple_to_dict(start_lane.getShape()[-1])
                    end_point = point_tuple_to_dict(end_lane.getShape()[0])
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge(start_lane, start_road),
                        "endLaneIndex": _cityflow_get_lane_index_in_edge(end_lane, end_road),
                        "points": [start_point, end_point]
                    }
                    roadLink["laneLinks"].append(path)
        
        roadLinks.append(roadLink)

    # Add roadLink indices to traffic light
    for i, _ in enumerate(intersection["roadLinks"]):
        intersection["trafficLight"]["roadLinkIndices"].append(i)

    # Handle different node types
    if node_type in ['dead_end', 'priority', 'right_before_left']:
        intersection = process_intersection_simple_phase(intersection)

    # Process traffic light phases
    if node_type in ['traffic_light', 'traffic_light_right_on_red']:
        print(f"Processing traffic light: {node.getID()}")
        
        if SUMO_PROGRAM:
            all_phase = []
            nodeid = node.getID()
            tlnodeid = nodeid
            
            if nodeid not in tls_dict:
                tlnodeid = 'GS_' + tlnodeid
            
            # Get road direction mapping if network is available
            road_directions = {}
            if net:
                road_directions = get_road_direction_mapping(node, net)
            
            # Map green phases to lanes
            G_to_lane_dict = {}
            for connec in tls_dict[tlnodeid]._connections:
                G_to_lane_dict[connec[-1]] = connec[0].getID()

            # Process each phase
            for idx_phase in tls_dict[tlnodeid]._programs['0']._phases:
                phase, duration = idx_phase.state, idx_phase.duration
                lane_list = []
                
                for i, alpha in enumerate(phase):
                    if (alpha == 'G' or alpha == 'g' or alpha == 's') and i in G_to_lane_dict:
                        lane_list.append(G_to_lane_dict[i])

                # Convert lane IDs to CityFlow format
                lane_list_ = []
                for lane in lane_list:
                    edge_id, lane_id = lane.rsplit("_", 1)
                    lane_id = int(lane_id)
                    lane_ = edge_id + '_' + str(len(edge_dict[edge_id]) - lane_id - 1)
                    lane_list_.append(lane_)

                # Find roadlink indices for these lanes
                index_list = []
                for _lane in lane_list_:
                    index_roadlink_list = filter_roadlinks_by_startedge(roadLinks, _lane)
                    index_list += [item[0] for item in index_roadlink_list]
                
                # Generate phase name if we have direction mapping
                phase_name = None
                if road_directions and roadLinks:
                    phase_name = generate_phase_name(list(set(index_list)), roadLinks, road_directions)
                
                phase_dict = {
                    'availableRoadLinks': list(set(index_list)),
                    'time': duration
                }
                
                # Add phase name if available
                if phase_name:
                    phase_dict['name'] = phase_name
                
                all_phase.append(phase_dict)
            
            intersection["trafficLight"]["lightphases"] = all_phase

    return intersection


def get_final_intersections(net, tls_dict, edge_dict):
    """Get all intersections in CityFlow format.
    
    Args:
        net: SUMO network object
        tls_dict: Dictionary of traffic light systems
        edge_dict: Dictionary mapping edge ID to lanes
        
    Returns:
        list: List of intersection dictionaries
    """
    final_intersections = []
    net_nodes = sorted(net.getNodes(), key=lambda n: n.getID())
    
    for node in net_nodes:
        intersection = node_to_intersection(node, tls_dict, edge_dict, net)
        if intersection["roads"]:
            final_intersections.append(intersection)

    return final_intersections


def get_final_roads(net):
    """Get all roads in CityFlow format.
    
    Args:
        net: SUMO network object
        
    Returns:
        list: List of road dictionaries
    """
    final_roads = []
    edges = net.getEdges()
    
    for edge in edges:
        start_intersection = edge.getFromNode()
        start_coord = start_intersection.getCoord()
        end_intersection = edge.getToNode()
        end_coord = end_intersection.getCoord()
        tmp_points = edge.getShape()
        
        points = [{"x": start_coord[0], "y": start_coord[1]}]
        for i in range(1, len(tmp_points) - 1):
            points.append({"x": tmp_points[i][0], "y": tmp_points[i][1]})
        points.append({"x": end_coord[0], "y": end_coord[1]})
        
        road = {
            "id": edge.getID(),
            "points": points,
            "lanes": [],
            "startIntersection": start_intersection.getID(),
            "endIntersection": end_intersection.getID(),
        }
        
        if DEBUG:
            road['_compass_angle'] = calc_edge_compass_angle_no_modify(edge)
        
        if TRUE_CORRECTION_LANE:
            for _v in edge._lanes:
                road["lanes"].append({
                    "width": _v._width,
                    "maxSpeed": _v._speed
                })
        else:
            for _v in edge._cityflow_lanes:
                road["lanes"].append({
                    "width": _v._width,
                    "maxSpeed": _v._speed
                })
        
        final_roads.append(road)
    
    return final_roads


def sumo2cityflow_net(args):
    """Convert SUMO network file to CityFlow roadnet format.
    
    Args:
        args: Arguments object with file paths
    """
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join('.', args.or_sumonet)
    cityflowfile = os.path.join('.', args.cityflownet)
    
    print(f"Converting SUMO net file: {args.or_sumonet}")
    
    # Read SUMO network
    net = sumolib.net.readNet(sumofile, withPrograms=True)
    
    # Process edges
    for edge in net.getEdges():
        process_edge(edge)
    
    # Build TLS dictionary
    tls_dict = {tls.getID(): tls for tls in net.getTrafficLights()}
    print(f'Processing {len(tls_dict)} traffic lights')
    
    # Build edge dictionary
    edge_dict = {edge_.getID(): edge_._lanes for edge_ in net.getEdges()}
    
    # Convert intersections and roads
    final_intersections = get_final_intersections(net, tls_dict, edge_dict)
    final_roads = get_final_roads(net)
    
    # Write CityFlow roadnet
    result = {
        "intersections": final_intersections,
        "roads": final_roads
    }
    
    with open(cityflowfile, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("CityFlow net file generated successfully!")


def sumo2cityflow_flow(args):
    """Convert SUMO traffic flow file to CityFlow flow format.
    
    Args:
        args: Arguments object with file paths
    """
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join('.', args.or_sumotraffic)
    cityflowfile = os.path.join('.', args.cityflowtraffic)
    sumocfg = os.path.join('.', args.sumocfg)

    print(f"Converting SUMO flow file: {args.or_sumotraffic}")
    
    tree = ET.parse(sumofile)
    root = tree.getroot()
    
    # Handle trip.xml -> rou.xml conversion
    if root.find('trip') is not None and 'rou' in sumofile:
        src = sumofile
        dst = get_filename(sumofile, typ='trip')
        try:
            os.rename(src, dst)
            print('Renamed file to trip.xml')
        except Exception as e:
            print(f'Rename failed: {e}')
            return
        
        sumofile = get_filename(sumofile, typ='rou')
        sumonet = os.path.join('.', args.or_sumonet)
        cmd = f"duarouter --route-files={dst} --net-file={sumonet} --output-file={sumofile}"
        os.system(cmd)
        print("SUMO rou file generated successfully!")

        # Re-parse the generated file
    tree = ET.parse(sumofile)
    root = tree.getroot()
    
    # Read time range from config
    tree_cfg = ET.parse(sumocfg)
    root_cfg = tree_cfg.getroot()
    start_time = int(root_cfg.find('time').find('begin').attrib['value'])
    end_time = int(root_cfg.find('time').find('end').attrib['value'])
    assert end_time - start_time == 3600, "Expected 1 hour simulation"
    
    # Default vehicle parameters
    length = 5.0
    width = 1.8
    maxPosAcc = 2.6
    maxNegAcc = 4.5
    minGap = 2.5
    
    # Read vehicle type if available
    vtype_elem = root.find('vType')
    if vtype_elem is not None:
        length = float(vtype_elem.attrib.get('length', 5.0))
        width = float(vtype_elem.attrib.get('width', 1.8))
        maxPosAcc = float(vtype_elem.attrib.get('accel', 2.6))
        maxNegAcc = float(vtype_elem.attrib.get('decel', 4.5))
        minGap = float(vtype_elem.attrib.get('minGap', 2.5))
    
    # Convert vehicles to flows
    flows = []
    for obj in root.iter('vehicle'):
        route_elem = obj.find('route')
        if route_elem is None:
            continue
        
        routes = route_elem.attrib['edges'].split()
        if len(routes) < 2:
            continue
        
        depart_time = int(float(obj.attrib['depart']))
        flows.append({
            "vehicle": {
                "length": length,
                "width": width,
                "maxPosAcc": maxPosAcc,
                "maxNegAcc": maxNegAcc,
                "usualPosAcc": maxPosAcc,
                "usualNegAcc": maxNegAcc,
                "minGap": minGap,
                "maxSpeed": 13.39,
                "headwayTime": 1.5
            },
            "route": routes,
            "interval": 5.0,
            "startTime": depart_time - start_time,
            "endTime": depart_time - start_time
        })
    
    # Write CityFlow flow file
    with open(cityflowfile, "w") as f:
        json.dump(flows, f, indent=2)
    
    print("CityFlow flow file generated successfully!")

# ============================================================================
# CityFlow to SUMO Conversion Functions
# ============================================================================

def cityflow2sumo_flow(args):
    """Convert CityFlow traffic flow file to SUMO routes format.
    
    Args:
        args: Arguments object with file paths
    """
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join('.', args.sumotraffic)
    cityflowfile = os.path.join('.', args.or_cityflowtraffic)
    
    print(f"Converting CityFlow flow file: {args.or_cityflowtraffic}")
    
    # Read CityFlow flow data
    with open(cityflowfile, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    # Sort by start time
    data = sorted(data, key=lambda x: x['startTime'])

    # Create SUMO routes XML
    doc = xml.dom.minidom.Document()
    root = doc.createElement('routes')
    root.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.setAttribute('xsi:noNamespaceSchemaLocation',
                      'http://sumo.dlr.de/xsd/routes_file.xsd')
    doc.appendChild(root)
    
    # Add vehicle type
    node_vtype = doc.createElement('vType')
    node_vtype.setAttribute('id', 'pkw')
    node_vtype.setAttribute('length', '5.0')
    node_vtype.setAttribute('width', '2.0')
    node_vtype.setAttribute('minGap', '2.5')
    node_vtype.setAttribute('maxSpeed', '11.111')
    node_vtype.setAttribute('accel', '2.0')
    node_vtype.setAttribute('decel', '4.5')
    root.appendChild(node_vtype)

    # Add vehicles
    for idx, info in enumerate(data):
        vehicle = info['vehicle']
        route = info['route']
        startTime = info['startTime']
        
        node_vehicle = doc.createElement('vehicle')
        node_vehicle.setAttribute('id', str(idx))
        node_vehicle.setAttribute('depart', str(startTime))

        node_route = doc.createElement('route')
        node_route.setAttribute('edges', ' '.join(route))
        node_vehicle.appendChild(node_route)
        root.appendChild(node_vehicle)

    # Write SUMO routes file
    with open(sumofile, 'w', encoding="utf-8") as fp:
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    
    print("SUMO flow file generated successfully!")


def cityflow2sumo_net(args):
    """Convert CityFlow roadnet to SUMO network format.
    
    Args:
        args: Arguments object with file paths
    """
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join('.', args.sumonet)
    cityflowfile = os.path.join('.', args.or_cityflownet)
    
    print(f"Converting CityFlow net file: {args.or_cityflownet}")

    sumo_node = get_filename(sumofile, 'nod')
    sumo_edge = get_filename(sumofile, 'edg')
    sumo_con = get_filename(sumofile, 'con')
    sumo_tll = get_filename(sumofile, 'tll')

    # Read CityFlow roadnet
    with open(cityflowfile, 'r', encoding="utf-8") as f:
        data = json.load(f)

    ordered_roads = sort_roads(data)

    # Create XML documents
    doc_node = xml.dom.minidom.Document()
    root_node = doc_node.createElement('nodes')
    root_node.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_node.setAttribute('xsi:noNamespaceSchemaLocation',
                           'http://sumo.dlr.de/xsd/nodes_file.xsd')
    doc_node.appendChild(root_node)

    doc_con = xml.dom.minidom.Document()
    root_con = doc_con.createElement('connections')
    root_con.setAttribute('version', '1.1')
    root_con.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_con.setAttribute('xsi:noNamespaceSchemaLocation',
                          'http://sumo.dlr.de/xsd/connections_file.xsd')
    doc_con.appendChild(root_con)

    doc_tll = xml.dom.minidom.Document()
    root_tll = doc_tll.createElement('tlLogics')
    root_tll.setAttribute('version', '1.1')
    root_tll.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_tll.setAttribute('xsi:noNamespaceSchemaLocation',
                          'http://sumo.dlr.de/xsd/tllogic_file.xsd')
    doc_tll.appendChild(root_tll)

    # Process intersections
    for inter in data['intersections']:
        # Create node
        node = doc_node.createElement('node')
        node.setAttribute('id', inter['id'])
        node.setAttribute('x', str(inter['point']['x']))
        node.setAttribute('y', str(inter['point']['y']))
        node.setAttribute('type', 'priority' if inter['virtual'] else 'traffic_light_right_on_red')
        root_node.appendChild(node)

        # Group and sort roadlinks
        road_group = []
        sortorder = {"turn_right": 0, "go_straight": 1, "turn_left": 2, "turn_u": 3}
        
        for idx, items in groupby(inter['roadLinks'], key=itemgetter('startRoad')):
            l_items = list(items)
            # Detect U-turns
            for x in l_items:
                if x['type'] == 'turn_left' and judg_turn_u(x, data['roads']):
                        x.update({'type': 'turn_u'})
            # Sort by turn type
            sorted_items = sorted(l_items, key=lambda x: sortorder[x['type']])
            road_group += sorted_items
        
        # Reorder by start road
        sorted_road_group = [[] for _ in range(len(road_group))]
        for x in road_group:
            idx = ordered_roads[inter['id']].index(x['startRoad'])
            sorted_road_group[idx].append(x)
        
        road_group = []
        for x in sorted_road_group:
            if x:
                road_group += x
        
        # Create phase mapping
        phase_dic = {}
        for idx, x in enumerate(inter['roadLinks']):
            dst_idx = road_group.index(x)
            phase_dic[idx] = {dst_idx: len(road_group[dst_idx]['laneLinks'])}
        phase_dic = get_start_idx(phase_dic)
        phase_num_all = sum(len(i['laneLinks']) for i in inter['roadLinks'])
        
        # Process connections
        for idx, link in enumerate(inter['roadLinks']):
            start_num_lanes = [len(x['lanes']) for x in data['roads'] 
                             if x['id'] == link['startRoad']][0] - 1
            end_num_lanes = [len(x['lanes']) for x in data['roads'] 
                           if x['id'] == link['endRoad']][0] - 1
            
            for lanelink in link['laneLinks']:
                con = doc_con.createElement('connection')
                con.setAttribute('from', link['startRoad'])
                con.setAttribute('to', link['endRoad'])
                # SUMO lane order is opposite to CityFlow
                con.setAttribute('fromLane', str(abs(start_num_lanes - lanelink['startLaneIndex'])))
                con.setAttribute('toLane', str(abs(end_num_lanes - lanelink['endLaneIndex'])))
                root_con.appendChild(con)
        
        # Process traffic lights
        if not inter['virtual']:
            tll = doc_node.createElement('tlLogic')
            tll.setAttribute('id', inter['id'])
            tll.setAttribute('type', 'static')
            tll.setAttribute('programID', '0')
            tll.setAttribute('offset', '0')
            
            # Get road direction mapping for this intersection
            road_directions = {}
            for road_id in inter['roads']:
                # Find the road in the roads data
                road_data = next((r for r in data['roads'] if r['id'] == road_id), None)
                if road_data:
                    # Calculate direction based on road geometry
                    points = road_data['points']
                    if len(points) >= 2:
                        # Determine if this road is incoming or outgoing to the intersection
                        if road_data['endIntersection'] == inter['id']:
                            # Incoming road - use direction approaching intersection
                            start_point = points[-2]
                            end_point = points[-1]
                        else:
                            # Outgoing road - use direction leaving intersection
                            start_point = points[0]
                            end_point = points[1]
                        
                        # Calculate angle
                        dx = end_point['x'] - start_point['x']
                        dy = end_point['y'] - start_point['y']
                        angle = math.degrees(math.atan2(dx, dy))
                        
                        # Convert to compass direction
                        direction = get_road_direction_from_angle(angle)
                        road_directions[road_id] = direction
            
            yellow_state = ['r'] * phase_num_all
            
            for idx, light in enumerate(inter['trafficLight']['lightphases']):
                state = ['r'] * phase_num_all
                
                if idx != 0 and light['availableRoadLinks'] is not None:
                    # Green or yellow phase
                    single_phase = ['y'] if light['time'] <= 5 else ['G']
                    
                    for act_roadlink in light['availableRoadLinks']:
                        start_idx, count = phase_dic[act_roadlink]
                        state[start_idx:start_idx + count] = single_phase * count
                    
                    # Generate phase name
                    phase_name = generate_phase_name(light['availableRoadLinks'], inter['roadLinks'], road_directions)
                    
                    # Add green/yellow phase
                    phase = doc_node.createElement('phase')
                    phase.setAttribute('duration', str(light['time']))
                    phase.setAttribute('state', ''.join(state))
                    phase.setAttribute('name', phase_name)
                    tll.appendChild(phase)
                    root_tll.appendChild(tll)
                    
                    # Add yellow phase after green
                    phase_y = doc_node.createElement('phase')
                    phase_y.setAttribute('duration', '5')
                    phase_y.setAttribute('state', ''.join(yellow_state))
                    phase_y.setAttribute('name', 'YELLOW_ALL_RED')
                    tll.appendChild(phase_y)
                    root_tll.appendChild(tll)
                
                # Handle initial yellow phase
                if light['time'] <= 5:
                    assert idx == 0
                    for act_roadlink in light['availableRoadLinks']:
                        start_idx, count = phase_dic[act_roadlink]
                        yellow_state[start_idx:start_idx + count] = ['s'] * count
    
    # Write node, connection, and traffic light files
    with open(sumo_node, 'w', encoding="utf-8") as fp:
        doc_node.writexml(fp, addindent='\t', newl='\n', encoding="utf-8")
    
    with open(sumo_con, 'w', encoding="utf-8") as fp:
        doc_con.writexml(fp, addindent='\t', newl='\n', encoding="utf-8")
    
    with open(sumo_tll, 'w', encoding="utf-8") as fp:
        doc_tll.writexml(fp, addindent='\t', newl='\n', encoding="utf-8")

    print("SUMO node, connections and tll files generated successfully!")

    # Generate edge file
    doc_edge = xml.dom.minidom.Document()
    root_edge = doc_edge.createElement('edges')
    root_edge.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_edge.setAttribute('xsi:noNamespaceSchemaLocation',
                           'http://sumo.dlr.de/xsd/edges_file.xsd')
    doc_edge.appendChild(root_edge)
    
    for road in data['roads']:
        edge = doc_edge.createElement('edge')
        edge.setAttribute('id', road['id'])
        edge.setAttribute('from', road['startIntersection'])
        edge.setAttribute('to', road['endIntersection'])
        edge.setAttribute('numLanes', str(len(road['lanes'])))
        edge.setAttribute('speed', '11.111')
        edge.setAttribute('priority', '-1')
        root_edge.appendChild(edge)
    
    with open(sumo_edge, 'w', encoding="utf-8") as fp:
        doc_edge.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    
    print("SUMO edge file generated successfully!")
    
    # Generate net.xml using netconvert
    cmd = (f"netconvert --node-files={sumo_node} --edge-files={sumo_edge} "
           f"--connection-files={sumo_con} --tllogic-files={sumo_tll} "
           f"--output-file={sumofile}")
    
    res = os.system(cmd)
    if res == 0:
        print("SUMO net file generated successfully!")
    else:
        pass


def cityflow2sumo_cfg(args):
    """Generate SUMO configuration file.
    
    Args:
        args: Arguments object with file paths
    """
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join('.', args.sumonet)
    sumo_cfg = get_filename(sumofile, typ='sumocfg')
    sumo_net = get_filename(sumofile, typ='net', need_path=False)
    sumo_route = get_filename(sumofile, typ='rou', need_path=False)
    
    print(f"Generating SUMO config file: {sumo_cfg}")

    doc = xml.dom.minidom.Document()
    root = doc.createElement('configuration')
    doc.appendChild(root)

    # Input files
    input_file = doc.createElement('input')
    input_net = doc.createElement('net-file')
    input_net.setAttribute('value', sumo_net)
    input_file.appendChild(input_net)
    
    input_route = doc.createElement('route-files')
    input_route.setAttribute('value', sumo_route)
    input_file.appendChild(input_route)
    root.appendChild(input_file)
    
    # Time settings
    time = doc.createElement('time')
    begin = doc.createElement('begin')
    begin.setAttribute('value', '0')
    time.appendChild(begin)
    
    end = doc.createElement('end')
    end.setAttribute('value', '3600')
    time.appendChild(end)
    root.appendChild(time)

    # Write config file
    with open(sumo_cfg, 'w', encoding="utf-8") as fp:
        doc.writexml(fp, addindent='\t', newl='\n', encoding="utf-8")

    print("SUMO cfg file generated successfully!")

# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Convert traffic configurations between SUMO and CityFlow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SUMO to CityFlow
  python convert_sumo2cityflow.py --typ s2c \\
      --or_sumonet cologne3/cologne3.net.xml \\
      --cityflownet cologne3/cologne3_roadnet.json \\
      --or_sumotraffic cologne3/cologne3.rou.xml \\
      --cityflowtraffic cologne3/cologne3_flow.json \\
      --sumocfg cologne3/cologne3.sumocfg
  
  # CityFlow to SUMO
  python convert_sumo2cityflow.py --typ c2s \\
      --or_cityflownet ../data/Jinan/3_4/roadnet_3_4.json \\
      --sumonet ../data/Jinan/3_4/jinan.net.xml \\
      --or_cityflowtraffic ../data/Jinan/3_4/anon_3_4_jinan_synthetic_24h_6000.json \\
      --sumotraffic ../data/Jinan/3_4/jinan_synthetic_24h_6000.rou.xml
        """
    )
    
    parser.add_argument(
        "--typ", type=str, default='s2c',
        choices=['c2s', 's2c'],
        help='Conversion type: c2s (CityFlow to SUMO) or s2c (SUMO to CityFlow)'
    )
    
    # SUMO to CityFlow arguments
    parser.add_argument(
        "--or_sumonet", type=str,
        default='cologne3/cologne3.net.xml',
        help='Input SUMO network file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--cityflownet", type=str,
        default='cologne3/cologne3_roadnet_red.json',
        help='Output CityFlow roadnet file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--or_sumotraffic", type=str,
        default='cologne3/cologne3.rou.xml',
        help='Input SUMO traffic/route file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--cityflowtraffic", type=str,
        default='cologne3/cologne3_flow.json',
        help='Output CityFlow flow file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--sumocfg", type=str,
        default='cologne3/cologne3.sumocfg',
        help='Input SUMO config file (relative to data/raw_data)'
    )
    
    # CityFlow to SUMO arguments
    parser.add_argument(
        "--or_cityflownet", type=str,
        default='hangzhou_1x1_bc-tyc_18041610_1h/roadnet.json',
        help='Input CityFlow roadnet file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--sumonet", type=str,
        default='hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.net.xml',
        help='Output SUMO network file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--or_cityflowtraffic", type=str,
        default='hangzhou_1x1_bc-tyc_18041610_1h/flow.json',
        help='Input CityFlow flow file (relative to data/raw_data)'
    )
    parser.add_argument(
        "--sumotraffic", type=str,
        default='hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.rou.xml',
        help='Output SUMO traffic/route file (relative to data/raw_data)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.typ == 'c2s':
        # CityFlow to SUMO conversion
        print("=" * 60)
        print("Converting CityFlow to SUMO")
        print("=" * 60)
        cityflow2sumo_net(args)
        cityflow2sumo_flow(args)
        cityflow2sumo_cfg(args)
        print("=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
    else:
        # SUMO to CityFlow conversion
        print("=" * 60)
        print("Converting SUMO to CityFlow")
        print("=" * 60)
        sumo2cityflow_net(args)
        sumo2cityflow_flow(args)
        print("=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)


if __name__ == '__main__':
    main()
