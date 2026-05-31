#!/usr/bin/env python3
"""
Add phase names and standardize durations to SUMO network traffic lights.

This script processes multiple SUMO network files and:
1. Adds phase names based on movement directions (e.g., ETWT, NTST, ELWL)
2. Standardizes phase durations: 5s for all-red/all-yellow, 300s for others
3. Updates the net.xml files in place or to a new location

Usage:
    python tools/add_phase_names_to_sumo_net.py --regions inner_brooklyn inner_queens
    python tools/add_phase_names_to_sumo_net.py --all-regions --backup
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import math

# Try to import sumolib
try:
    import sumolib
    from sumolib.net import Connection
except ImportError:
    if "SUMO_HOME" in os.environ:
        sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(sumo_tools)
        import sumolib
        from sumolib.net import Connection
    else:
        raise EnvironmentError(
            "SUMO libraries not found. Please set SUMO_HOME environment variable "
            "or install sumolib."
        )


# Constants
GREEN_DURATION = 300  # Standard green phase duration in seconds
YELLOW_RED_DURATION = 5  # Duration for all-red/all-yellow phases in seconds

# All regions to process
ALL_REGIONS = [
    "Inner_Brooklyn",
    "Upper_Manhattan",
    "Inner_Queens",
    "JFK",
    "LGA",
    "Manhattan_Core",
    "Middle_Queens",
    "Northern_Bronx",
    "Outer_Brooklyn",
    "Outer_Queens",
    "Southern_Bronx",
    "Staten_Island",
]


def get_road_direction_from_angle(angle: float) -> str:
    """Convert compass angle to cardinal direction.
    
    Args:
        angle: Angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        
    Returns:
        str: Cardinal direction ('N', 'E', 'S', 'W')
    """
    angle = angle % 360
    
    if 315 <= angle or angle < 45:
        return 'N'
    elif 45 <= angle < 135:
        return 'E'
    elif 135 <= angle < 225:
        return 'S'
    elif 225 <= angle < 315:
        return 'W'
    else:
        return 'N'


def get_road_direction_mapping(node, net) -> Dict[str, str]:
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
        edge_shape = edge.getShape()
        if len(edge_shape) >= 2:
            start_point = edge_shape[-2]
            end_point = edge_shape[-1]
            
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.degrees(math.atan2(dx, dy))
            
            direction = get_road_direction_from_angle(angle)
            road_directions[edge.getID()] = direction
    
    # Process outgoing roads (roads starting from this intersection)
    for edge in node.getOutgoing():
        edge_shape = edge.getShape()
        if len(edge_shape) >= 2:
            start_point = edge_shape[0]
            end_point = edge_shape[1]
            
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.degrees(math.atan2(dx, dy))
            
            direction = get_road_direction_from_angle(angle)
            road_directions[edge.getID()] = direction
    
    return road_directions


def get_direction_from_connection(connection) -> str:
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


def generate_phase_name_from_connections(
    green_connections: List,
    road_directions: Dict[str, str],
    node
) -> str:
    """Generate phase name based on green connections and road directions.
    
    Only includes through (T) and left turn (L) movements, excluding right turns (R) and U-turns (U).
    
    Args:
        green_connections: List of Connection objects that are green in this phase
        road_directions: Dictionary mapping road ID to cardinal direction
        node: SUMO node object
        
    Returns:
        str: Phase name (e.g., 'ETWT', 'NTST', 'ELWL', etc.)
    """
    if not green_connections:
        return "ALL_RED"
    
    movements = []
    
    for conn in green_connections:
        # Verify connection is a valid Connection object
        if not hasattr(conn, 'getFrom') or not hasattr(conn, 'getTo'):
            continue
        
        try:
            from_edge = conn.getFrom()
            to_edge = conn.getTo()
            
            if from_edge is None or to_edge is None:
                continue
            
            movement_type = get_direction_from_connection(conn)
            
            # Skip right turns and U-turns
            if movement_type == 'turn_right' or movement_type == 'turn_u':
                continue
            
            # Get directions
            from_edge_id = from_edge.getID() if hasattr(from_edge, 'getID') else str(from_edge)
            to_edge_id = to_edge.getID() if hasattr(to_edge, 'getID') else str(to_edge)
            
            start_dir = road_directions.get(from_edge_id, 'X')
            end_dir = road_directions.get(to_edge_id, 'X')
            
            # Determine movement type abbreviation
            if movement_type == 'go_straight':
                move_abbr = 'T'  # Through/straight
            elif movement_type == 'turn_left':
                move_abbr = 'L'  # Left turn
            else:
                move_abbr = 'T'  # Default to through
            
            # Create movement descriptor
            movement = f"{start_dir}{move_abbr}"
            movements.append(movement)
        except Exception as e:
            # Skip invalid connections
            continue
    
    # Sort movements for consistent naming
    movements = sorted(list(set(movements)))
    
    # Generate phase name
    if len(movements) == 0:
        return "ALL_RED"
    elif len(movements) == 1:
        return movements[0]
    else:
        # Group by movement type
        through_moves = [m for m in movements if m.endswith('T')]
        left_moves = [m for m in movements if m.endswith('L')]
        
        # Create combined name
        name_parts = []
        if through_moves:
            name_parts.extend(through_moves)
        if left_moves:
            name_parts.extend(left_moves)
        
        return ''.join(name_parts) if name_parts else "MIXED"


def is_all_red_or_yellow(state: str) -> bool:
    """Check if phase state is all red or all yellow.
    
    Args:
        state: Phase state string (e.g., 'rrrrGGGggrrrrGGGgg')
        
    Returns:
        bool: True if all characters are 'r', 'y', or 's' (red/yellow)
    """
    return all(c in 'rys' for c in state)


def classify_connection(conn, road_directions: Dict[str, str]) -> Tuple[str, str, str]:
    """Classify a connection by direction and movement type.
    
    Args:
        conn: SUMO Connection object
        road_directions: Dictionary mapping road ID to cardinal direction
        
    Returns:
        Tuple of (from_dir, movement_type, connection_type)
        movement_type: 'T' (through), 'L' (left), 'R' (right), 'U' (u-turn)
    """
    try:
        if not hasattr(conn, 'getFrom') or not hasattr(conn, 'getTo'):
            return 'X', 'T', 'unknown'
        
        from_edge = conn.getFrom()
        to_edge = conn.getTo()
        
        if from_edge is None or to_edge is None:
            return 'X', 'T', 'unknown'
        
        from_edge_id = from_edge.getID() if hasattr(from_edge, 'getID') else str(from_edge)
        from_dir = road_directions.get(from_edge_id, 'X')
        
        direction = get_direction_from_connection(conn)
        
        if direction == 'go_straight':
            movement = 'T'
        elif direction == 'turn_left':
            movement = 'L'
        elif direction == 'turn_right':
            movement = 'R'
        elif direction == 'turn_u':
            movement = 'U'
        else:
            movement = 'T'  # Default
        
        return from_dir, movement, direction
    except:
        return 'X', 'T', 'unknown'


def build_standard_phases(
    signal_to_connection: Dict[int, Connection],
    road_directions: Dict[str, str],
    num_signals: int
) -> List[Tuple[str, str]]:
    """Build standard four-phase configuration based on available movements.
    
    Args:
        signal_to_connection: Mapping from signal index to connection
        road_directions: Dictionary mapping road ID to cardinal direction
        num_signals: Total number of signals
        
    Returns:
        List of (phase_name, state_string) tuples
    """
    phases = []
    
    # Collect available movements by direction
    movements_by_dir = {
        'E': {'T': False, 'L': False, 'U': False},
        'W': {'T': False, 'L': False, 'U': False},
        'N': {'T': False, 'L': False, 'U': False},
        'S': {'T': False, 'L': False, 'U': False}
    }
    
    # Analyze all connections
    for signal_idx, conn in signal_to_connection.items():
        from_dir, movement, _ = classify_connection(conn, road_directions)
        if from_dir in movements_by_dir:
            if movement in ['T', 'L']:
                movements_by_dir[from_dir][movement] = True
            elif movement == 'U':
                movements_by_dir[from_dir]['U'] = True
                # If there's U but no L, treat U as L for phase naming
                if not movements_by_dir[from_dir]['L']:
                    movements_by_dir[from_dir]['L'] = True
    
    # Build standard phases
    # ETWT: East Through + West Through
    if movements_by_dir['E']['T'] and movements_by_dir['W']['T']:
        phases.append(('ETWT', build_phase_state('ETWT', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['E']['T']:
        phases.append(('ET', build_phase_state('ET', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['W']['T']:
        phases.append(('WT', build_phase_state('WT', signal_to_connection, road_directions, num_signals)))
    
    # ELWL: East Left + West Left
    if movements_by_dir['E']['L'] and movements_by_dir['W']['L']:
        phases.append(('ELWL', build_phase_state('ELWL', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['E']['L']:
        phases.append(('EL', build_phase_state('EL', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['W']['L']:
        phases.append(('WL', build_phase_state('WL', signal_to_connection, road_directions, num_signals)))
    
    # NTST: North Through + South Through
    if movements_by_dir['N']['T'] and movements_by_dir['S']['T']:
        phases.append(('NTST', build_phase_state('NTST', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['N']['T']:
        phases.append(('NT', build_phase_state('NT', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['S']['T']:
        phases.append(('ST', build_phase_state('ST', signal_to_connection, road_directions, num_signals)))
    
    # NLSL: North Left + South Left
    if movements_by_dir['N']['L'] and movements_by_dir['S']['L']:
        phases.append(('NLSL', build_phase_state('NLSL', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['N']['L']:
        phases.append(('NL', build_phase_state('NL', signal_to_connection, road_directions, num_signals)))
    elif movements_by_dir['S']['L']:
        phases.append(('SL', build_phase_state('SL', signal_to_connection, road_directions, num_signals)))
    
    return phases


def build_phase_state(
    phase_name: str,
    signal_to_connection: Dict[int, Connection],
    road_directions: Dict[str, str],
    num_signals: int
) -> str:
    """Build phase state string for a given phase name.
    
    Args:
        phase_name: Phase name (e.g., 'ETWT', 'ELWL', 'NTST', 'NLSL', 'ALL_RED_ALL_YELLOW')
        signal_to_connection: Mapping from signal index to connection
        road_directions: Dictionary mapping road ID to cardinal direction
        num_signals: Total number of signals
        
    Returns:
        State string (e.g., 'GGGrrrrrrGGGGGGrrr...')
    """
    state = ['r'] * num_signals
    
    if phase_name == 'ALL_RED_ALL_YELLOW':
        # All directions are 'r', right turns are 's'
        # For signal indices without mapped connections, keep them as 'r'
        for signal_idx, conn in signal_to_connection.items():
            if signal_idx >= num_signals:
                print(f"    Warning: Signal index {signal_idx} exceeds num_signals {num_signals}, skipping")
                continue
            _, movement, _ = classify_connection(conn, road_directions)
            if movement == 'R':
                state[signal_idx] = 's'
            else:
                state[signal_idx] = 'r'
        # All unmapped indices remain 'r' (already initialized)
    else:
        # Parse phase name to determine which movements are allowed
        # Phase names like: ETWT (East Through + West Through), ELWL (East Left + West Left), etc.
        allowed_movements = set()
        
        # Parse phase name: pairs of (direction, movement_type)
        i = 0
        while i < len(phase_name):
            if i + 1 < len(phase_name):
                dir_char = phase_name[i]
                move_char = phase_name[i + 1]
                allowed_movements.add((dir_char, move_char))
                i += 2
            else:
                break
        
        # Set green for allowed movements, right turns, and U-turns (when left turns are allowed)
        for signal_idx, conn in signal_to_connection.items():
            if signal_idx >= num_signals:
                print(f"    Warning: Signal index {signal_idx} exceeds num_signals {num_signals}, skipping")
                continue
                
            from_dir, movement, _ = classify_connection(conn, road_directions)
            
            # Right turns are always green
            if movement == 'R':
                state[signal_idx] = 'G'
            # U-turns are allowed when left turns are allowed (merged with left)
            # This handles three cases:
            # 1. Only L exists: L connections are allowed
            # 2. Only U exists: U connections are allowed (marked as L in phase name)
            # 3. Both L and U exist: Both L and U connections are allowed when phase name contains L
            # Note: If phase name contains L for this direction, U turn is always allowed
            # The else branch handles cases where phase name doesn't contain L (e.g., ETWT, NTST - through phases only)
            elif movement == 'U':
                if (from_dir, 'L') in allowed_movements:
                    # Phase name contains L for this direction, allow U turn
                    # This works for both "only U" and "both L and U" cases
                    state[signal_idx] = 'G'
                else:
                    # Phase name doesn't contain L (e.g., through-only phases like ETWT, NTST)
                    # U turn should be prohibited in through-only phases
                    state[signal_idx] = 'r'
            # Check if this movement is allowed
            elif (from_dir, movement) in allowed_movements:
                state[signal_idx] = 'G'
            else:
                state[signal_idx] = 'r'
    
    return ''.join(state)


def validate_phase_state(state: str, tls_id: str, phase_name: str) -> bool:
    """Validate that a phase state string is valid for SUMO.
    
    Args:
        state: Phase state string
        tls_id: Traffic light system ID (for logging)
        phase_name: Phase name (for logging)
        
    Returns:
        True if valid, False otherwise
    """
    if not state:
        print(f"    Error: Empty state string for TLS {tls_id}, phase {phase_name}")
        return False
    
    # Check that all characters are valid SUMO signal states
    valid_chars = set('rRgGyYoOuUsS')
    invalid_chars = set(state) - valid_chars
    if invalid_chars:
        print(f"    Error: Invalid characters {invalid_chars} in state for TLS {tls_id}, phase {phase_name}")
        return False
    
    return True


def process_tl_logic(
    tl_logic_elem: ET.Element,
    tls_id: str,
    net: sumolib.net.Net,
    tls_obj: sumolib.net.TLS
) -> Tuple[int, int, bool, bool]:
    """Process a single tlLogic element and rebuild phases from scratch.
    
    Args:
        tl_logic_elem: XML element for tlLogic
        tls_id: Traffic light system ID
        net: SUMO network object
        tls_obj: SUMO TLS object
        
    Returns:
        Tuple of (phases_created, old_phases_removed, standard_phases_created, is_single_phase)
        standard_phases_created: True if standard phases were successfully created
        is_single_phase: True if intersection has only one phase (before or after processing)
    """
    # Get the node/junction for this TLS
    node = None
    
    # Try to find node by TLS ID
    try:
        node = net.getNode(tls_id)
    except:
        pass
    
    if node is None:
        # Try alternative TLS ID format
        if tls_id.startswith('GS_'):
            node_id = tls_id[3:]
            try:
                node = net.getNode(node_id)
            except:
                pass
        else:
            # Try GS_ prefix
            try:
                node = net.getNode('GS_' + tls_id)
            except:
                pass
    
    # If still not found, try to get from TLS object
    if node is None and hasattr(tls_obj, 'getNodes'):
        try:
            nodes = tls_obj.getNodes()
            if nodes:
                node = nodes[0]
        except:
            pass
    
    if node is None:
        print(f"  Warning: Could not find node for TLS {tls_id}")
        return 0, 0, False, False
    
    # Get the actual state string length from existing phases BEFORE processing
    # This is the ONLY authoritative source for num_signals
    old_phases = list(tl_logic_elem.findall('phase'))
    old_phases_count = len(old_phases)
    
    if not old_phases:
        print(f"  Warning: TLS {tls_id} has no existing phases, cannot infer signal count safely, skipping")
        return 0, 0, False, False
    
    first_state = old_phases[0].get('state', '')
    if not first_state:
        print(f"  Warning: TLS {tls_id} has empty phase state, skipping")
        return 0, 0, False, False
    
    # Check if original intersection has only one phase
    is_single_phase_original = (old_phases_count == 1)
    
    # Use the actual state string length as the ONLY source of truth
    num_signals = len(first_state)
    
    # Get road direction mapping
    road_directions = get_road_direction_mapping(node, net)
    
    # Build mapping from signal index to connection
    signal_to_connection = {}
    
    # Get all connections from the node
    node_connections = node.getConnections()
    
    if hasattr(tls_obj, '_connections'):
        tls_connections = tls_obj._connections
        for conn_tuple in tls_connections:
            if len(conn_tuple) >= 2:
                signal_idx = conn_tuple[-1]  # Last element is always signal index
                lane = conn_tuple[0]  # First element is always lane
                
                # CRITICAL: Skip signal indices that are out of range
                if signal_idx >= num_signals:
                    print(f"    Warning: TLS {tls_id} has signal index {signal_idx} >= num_signals {num_signals}, skipping this connection")
                    continue
                
                # Try to get connection from tuple if available
                connection = None
                if len(conn_tuple) >= 3:
                    # Check if second element is a Connection object
                    potential_conn = conn_tuple[1]
                    if hasattr(potential_conn, 'getFrom') and hasattr(potential_conn, 'getTo'):
                        connection = potential_conn
                
                # If not found in tuple, search in node connections
                if connection is None:
                    for conn in node_connections:
                        try:
                            if conn.getFromLane() == lane:
                                connection = conn
                                break
                        except:
                            # Skip if getFromLane() fails
                            continue
                
                # Verify connection is valid before storing
                if connection is not None and hasattr(connection, 'getFrom') and hasattr(connection, 'getTo'):
                    signal_to_connection[signal_idx] = connection
    
    if not signal_to_connection:
        print(f"  Warning: No valid connections found for TLS {tls_id} (num_signals={num_signals})")
        return 0, 0, False, is_single_phase_original
    
    # Validate all signal indices are within range
    max_idx_in_dict = max(signal_to_connection.keys()) if signal_to_connection else -1
    if max_idx_in_dict >= num_signals:
        print(f"  Error: TLS {tls_id} has signal index {max_idx_in_dict} >= num_signals {num_signals}, this should not happen after filtering")
        return 0, 0, False, is_single_phase_original
    
    # Debug output: show signal indices for troubleshooting
    signal_indices = sorted(signal_to_connection.keys())
    if len(signal_indices) < num_signals:
        print(f"    Info: TLS {tls_id} has {len(signal_indices)} connections mapped (indices: {signal_indices}), num_signals={num_signals}")
    
    # Build standard phases BEFORE removing old phases
    # This allows us to preserve original phases if standard phases cannot be built
    standard_phases = build_standard_phases(signal_to_connection, road_directions, num_signals)
    
    # Handle case when no standard phases can be built (e.g., all directions are 'X' or only right turns)
    # In this case, preserve the original phases
    if len(standard_phases) == 0:
        print(f"    Warning: TLS {tls_id} - No standard phases could be built (likely all directions are 'X' or only right turns)")
        print(f"    Preserving original phases without modification")
        return 0, 0, False, is_single_phase_original  # No phases created, no phases removed, no standard phases
    
    # Remove all existing phases only if we can build standard phases
    for phase in old_phases:
        tl_logic_elem.remove(phase)
    
    phases_created = 0
    
    # If only one phase is available, use the actual phase state (not all green)
    if len(standard_phases) == 1:
        # Use the actual phase state instead of forcing all green
        phase_name, phase_state = standard_phases[0]
        
        # Validate phase state
        if not validate_phase_state(phase_state, tls_id, phase_name):
            print(f"    Error: Invalid phase state for TLS {tls_id}, phase {phase_name}")
            return 0, old_phases_count, False, True  # Single phase (only one standard phase built)
        
        phase_elem = ET.SubElement(tl_logic_elem, 'phase')
        phase_elem.set('duration', str(GREEN_DURATION))
        phase_elem.set('state', phase_state)
        phase_elem.set('name', phase_name)
        phases_created += 1
        print(f"    Single phase detected: {phase_name} (state length: {len(phase_state)}) -> Using actual phase state (no ALL_RED_ALL_YELLOW)")
        return phases_created, old_phases_count, True, True  # Single phase (only one standard phase built)
    else:
        # Add standard green phases
        for phase_name, state in standard_phases:
            # Validate phase state
            if not validate_phase_state(state, tls_id, phase_name):
                print(f"    Error: Invalid phase state for TLS {tls_id}, phase {phase_name}")
                continue
            
            phase_elem = ET.SubElement(tl_logic_elem, 'phase')
            phase_elem.set('duration', str(GREEN_DURATION))
            phase_elem.set('state', state)
            phase_elem.set('name', phase_name)
            phases_created += 1
        
        # Add ALL_RED_ALL_YELLOW phase (only for multi-phase intersections)
        all_red_state = build_phase_state('ALL_RED_ALL_YELLOW', signal_to_connection, road_directions, num_signals)
        
        # Validate all-red phase state
        if not validate_phase_state(all_red_state, tls_id, 'ALL_RED_ALL_YELLOW'):
            print(f"    Error: Invalid ALL_RED_ALL_YELLOW phase state for TLS {tls_id}")
        else:
            all_red_elem = ET.SubElement(tl_logic_elem, 'phase')
            all_red_elem.set('duration', str(YELLOW_RED_DURATION))
            all_red_elem.set('state', all_red_state)
            all_red_elem.set('name', 'ALL_RED_ALL_YELLOW')
            phases_created += 1
        
        return phases_created, old_phases_count, True, False  # Multiple phases (not single phase)


def process_net_file(net_file_path: Path, backup: bool = False, output_suffix: str = "_with_phases") -> Tuple[bool, int, int, int]:
    """Process a single SUMO network file.
    
    Args:
        net_file_path: Path to the net.xml file
        backup: Whether to create a backup before modifying
        output_suffix: Suffix to add to output filename (default: "_with_phases")
        
    Returns:
        Tuple of (success, total_intersections, standard_phases_intersections, single_phase_intersections)
        success: True if successful, False otherwise
        total_intersections: Total number of traffic light intersections
        standard_phases_intersections: Number of intersections with standard phases created
        single_phase_intersections: Number of intersections with only one phase
    """
    if not net_file_path.exists():
        print(f"Error: File not found: {net_file_path}")
        return False, 0, 0, 0
    
    print(f"\nProcessing: {net_file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = net_file_path.with_suffix('.net.xml.backup')
        shutil.copy2(net_file_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Generate output filename
    output_path = net_file_path.parent / f"{net_file_path.stem}.xml"
    print(f"  Output file: {output_path}")
    
    # Read network using sumolib
    try:
        net = sumolib.net.readNet(str(net_file_path), withPrograms=True)
    except Exception as e:
        print(f"  Error reading network with sumolib: {e}")
        return False, 0, 0, 0
    
    # Build TLS dictionary
    tls_dict = {tls.getID(): tls for tls in net.getTrafficLights()}
    print(f"  Found {len(tls_dict)} traffic light systems")
    
    # Parse XML for modification
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    
    total_phases_created = 0
    total_old_phases_removed = 0
    tls_processed = 0
    standard_phases_intersections = 0
    single_phase_intersections = 0
    
    # Count total intersections (all tlLogic elements)
    all_tl_logic_elements = list(root.findall('tlLogic'))
    total_intersections = len(all_tl_logic_elements)
    
    # Process each tlLogic element
    for tl_logic_elem in all_tl_logic_elements:
        tls_id = tl_logic_elem.get('id')
        
        if tls_id not in tls_dict:
            # Try alternative ID format
            if tls_id.startswith('GS_'):
                alt_id = tls_id[3:]
            else:
                alt_id = 'GS_' + tls_id
            
            if alt_id not in tls_dict:
                print(f"  Warning: TLS {tls_id} not found in network")
                continue
            
            tls_obj = tls_dict[alt_id]
        else:
            tls_obj = tls_dict[tls_id]
        
        try:
            created, removed, standard_created, is_single_phase = process_tl_logic(tl_logic_elem, tls_id, net, tls_obj)
            total_phases_created += created
            total_old_phases_removed += removed
            if standard_created:
                standard_phases_intersections += 1
            if is_single_phase:
                single_phase_intersections += 1
            tls_processed += 1
        except Exception as e:
            print(f"  Error processing TLS {tls_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Processed {tls_processed} TLS, created {total_phases_created} new phases, removed {total_old_phases_removed} old phases")
    
    # Write updated XML to new file
    try:
        ET.indent(root, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"  ✓ Successfully saved to {output_path}")
        return True, total_intersections, standard_phases_intersections, single_phase_intersections
    except Exception as e:
        print(f"  ✗ Error writing file: {e}")
        return False, total_intersections, standard_phases_intersections, single_phase_intersections


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Add phase names and standardize durations to SUMO network traffic lights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific regions
  python tools/add_phase_names_to_sumo_net.py --regions inner_brooklyn inner_queens
  
  # Process all regions with backup
  python tools/add_phase_names_to_sumo_net.py --all-regions --backup
  
  # Process single region
  python tools/add_phase_names_to_sumo_net.py --regions manhattan_core
        """
    )
    
    ap.add_argument(
        "--regions",
        nargs="+",
        # default=["manhattan_core"],
        help="List of regions to process (e.g., inner_brooklyn inner_queens)"
    )
    ap.add_argument(
        "--all-regions",
        default=True,
        action="store_true",
        help="Process all available regions"
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup files before modifying"
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="_with_phases",
        help="Suffix to add to output filename (default: '_with_phases')"
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default="../sumo_config_highway",
        help="Base directory containing region folders (default: sumo_config)"
    )
    
    args = ap.parse_args()
    
    # Determine regions to process
    if args.all_regions:
        regions = ALL_REGIONS
    elif args.regions:
        regions = args.regions
    else:
        print("Error: Must specify --regions or --all-regions")
        return 1
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1
    
    print("=" * 60)
    print("Adding Phase Names to SUMO Network Files")
    print("=" * 60)
    print(f"Regions to process: {len(regions)}")
    print(f"Backup mode: {args.backup}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    # Statistics for each network
    network_stats = []
    
    for region in regions:
        net_file = base_dir / region / f"{region}.net.xml"
        
        if not net_file.exists():
            print(f"\n⚠ Skipping {region}: {net_file} not found")
            fail_count += 1
            continue
        
        success, total_intersections, standard_phases_intersections, single_phase_intersections = process_net_file(
            net_file, backup=args.backup, output_suffix=args.output_suffix
        )
        
        network_stats.append({
            'region': region,
            'success': success,
            'total_intersections': total_intersections,
            'standard_phases_intersections': standard_phases_intersections,
            'single_phase_intersections': single_phase_intersections
        })
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)
    
    # Print statistics for each network
    print("\n" + "=" * 80)
    print("路口统计 (Intersection Statistics)")
    print("=" * 80)
    print(f"{'路网 (Network)':<25} {'总路口数':<12} {'标准相位路口数':<15} {'单相位路口数':<15} {'标准相位比例':<12}")
    print("-" * 80)
    
    total_all_intersections = 0
    total_all_standard = 0
    total_all_single_phase = 0
    
    for stat in network_stats:
        region = stat['region']
        total = stat['total_intersections']
        standard = stat['standard_phases_intersections']
        single_phase = stat['single_phase_intersections']
        ratio = f"{standard/total*100:.1f}%" if total > 0 else "N/A"
        
        total_all_intersections += total
        total_all_standard += standard
        total_all_single_phase += single_phase
        
        print(f"{region:<25} {total:<12} {standard:<15} {single_phase:<15} {ratio:<12}")
    
    print("-" * 80)
    overall_ratio = f"{total_all_standard/total_all_intersections*100:.1f}%" if total_all_intersections > 0 else "N/A"
    print(f"{'总计 (Total)':<25} {total_all_intersections:<12} {total_all_standard:<15} {total_all_single_phase:<15} {overall_ratio:<12}")
    print("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

