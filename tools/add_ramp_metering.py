#!/usr/bin/env python3
"""
Add ramp metering (traffic lights) to highway on-ramps in SUMO network.

This tool:
1. Identifies "ramp → highway" merge junctions (on-ramps) in the network
2. Converts those junctions to traffic_light type
3. Generates simple 0/1 ramp control TLS logic:
   - Phase 1 (RAMP_CLOSED): Ramp red (0 - closed), mainline green
   - Phase 2 (RAMP_OPEN): Ramp green (1 - open), mainline green
4. Uses sumolib (similar to add_phase_names_to_sumo_net.py) to correctly map
   signal indices to connections for accurate phase state generation

Usage:
    python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --generate-tl-logic
    python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --run-netconvert
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List

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


def build_edge_info(root: ET.Element) -> Dict[str, Dict[str, any]]:
    """
    Build edge information dictionary from network XML.
    
    Args:
        root: Root element of the network XML
        
    Returns:
        Dictionary mapping edge_id to {type, speed, ...}
    """
    edge_info = {}
    
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        
        edge_id = edge.get("id")
        edge_type = edge.get("type")
        
        # Get speed from first lane
        lane = edge.find("lane")
        speed = float(lane.get("speed")) if lane is not None else None
        
        edge_info[edge_id] = {
            "type": edge_type,
            "speed": speed
        }
    
    return edge_info


def identify_ramp_to_highway_junctions(
    root: ET.Element,
    edge_info: Dict[str, Dict[str, any]],
    min_highway_speed: float = 22.0,
    ramp_types: Optional[Set[str]] = None
) -> Set[str]:
    """
    Identify junctions where ramps merge into highways.
    
    Args:
        root: Root element of the network XML
        edge_info: Edge information dictionary
        min_highway_speed: Minimum speed (m/s) to consider as highway (default: 22.0 ≈ 80 km/h)
        ramp_types: Set of edge types to consider as ramps
                   (default: {"highway.motorway_link", "highway.trunk_link"})
        
    Returns:
        Set of junction IDs where ramps merge into highways
    """
    if ramp_types is None:
        ramp_types = {"highway.motorway_link", "highway.trunk_link"}
    
    ramp_to_main_junctions = set()
    
    # Method 1: Use connection's toJunction attribute
    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        to_junction = conn.get("toJunction")
        
        if not from_edge or not to_edge or not to_junction:
            continue
        
        if from_edge not in edge_info or to_edge not in edge_info:
            continue
        
        from_type = edge_info[from_edge]["type"]
        to_speed = edge_info[to_edge]["speed"]
        
        # Check if this is a ramp → highway connection
        if (
            from_type in ramp_types
            and to_speed is not None
            and to_speed >= min_highway_speed
        ):
            ramp_to_main_junctions.add(to_junction)
    
    # Method 2: If toJunction is not available, use via lane to find junction
    # This is a fallback method
    if not ramp_to_main_junctions:
        print("Warning: No junctions found via toJunction, trying alternative method...")
        # Build junction → edges mapping
        junction_edges = {}
        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            if edge.get("function") == "internal":
                continue
            from_junction = edge.get("from")
            to_junction = edge.get("to")
            if from_junction:
                if from_junction not in junction_edges:
                    junction_edges[from_junction] = {"incoming": [], "outgoing": []}
                junction_edges[from_junction]["outgoing"].append(edge_id)
            if to_junction:
                if to_junction not in junction_edges:
                    junction_edges[to_junction] = {"incoming": [], "outgoing": []}
                junction_edges[to_junction]["incoming"].append(edge_id)
        
        # Find junctions with ramp incoming and highway outgoing
        for junction_id, edges in junction_edges.items():
            incoming_types = {edge_info.get(e, {}).get("type") for e in edges["incoming"] if e in edge_info}
            outgoing_speeds = {edge_info.get(e, {}).get("speed") for e in edges["outgoing"] if e in edge_info and edge_info[e].get("speed")}
            
            has_ramp_incoming = any(t in ramp_types for t in incoming_types)
            has_highway_outgoing = any(s is not None and s >= min_highway_speed for s in outgoing_speeds)
            
            if has_ramp_incoming and has_highway_outgoing:
                ramp_to_main_junctions.add(junction_id)
    
    return ramp_to_main_junctions


def convert_junctions_to_traffic_light(
    root: ET.Element,
    junction_ids: Set[str]
) -> int:
    """
    Convert specified junctions to traffic_light type.
    
    Args:
        root: Root element of the network XML
        junction_ids: Set of junction IDs to convert
        
    Returns:
        Number of junctions converted
    """
    converted_count = 0
    
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if junction_id in junction_ids:
            current_type = junction.get("type")
            if current_type != "traffic_light":
                junction.set("type", "traffic_light")
                converted_count += 1
                print(f"  Converted junction {junction_id} from {current_type} to traffic_light")
            else:
                print(f"  Junction {junction_id} is already traffic_light")
    
    return converted_count


def generate_simple_tl_logic(
    root: ET.Element,
    junction_ids: Set[str],
    edge_info: Dict[str, Dict[str, any]],
    net: Optional[sumolib.net.Net],
    ramp_types: Optional[Set[str]] = None,
    default_green_duration: int = 3000,
    default_red_duration: int = 3000
) -> int:
    """
    Generate simple TLS logic directly in XML for ramp metering.
    
    This creates a basic 2-phase logic for 0/1 ramp control:
    - Phase 1 (RAMP_CLOSED): Ramp red (0 - closed), mainline green
    - Phase 2 (RAMP_OPEN): Ramp green (1 - open), mainline green
    
    Uses sumolib to correctly map signal indices to connections, similar to
    add_phase_names_to_sumo_net.py.
    
    Args:
        root: Root element of the network XML
        junction_ids: Set of junction IDs to generate TLS for
        edge_info: Edge information dictionary
        net: SUMO network object from sumolib
        ramp_types: Set of edge types to consider as ramps
        default_green_duration: Default green phase duration in seconds (default: 300)
        default_red_duration: Default red phase duration in seconds (default: 5)
        
    Returns:
        Number of TLS logic elements generated
    """
    if ramp_types is None:
        ramp_types = {"highway.motorway_link", "highway.trunk_link"}
    
    generated_count = 0
    
    # Find or create tlLogic elements
    existing_tl_logic = {tl.get("id"): tl for tl in root.findall("tlLogic")}
    
    # Get TLS dictionary from network if available
    tls_dict = {}
    if net is not None:
        tls_dict = {tls.getID(): tls for tls in net.getTrafficLights()}
    
    for junction_id in junction_ids:
        # Check if tlLogic already exists
        if junction_id in existing_tl_logic:
            print(f"  TLS logic already exists for {junction_id}, skipping")
            continue
        
        # Get the node/junction for this TLS
        node = None
        if net is not None:
            try:
                node = net.getNode(junction_id)
            except:
                # Try alternative TLS ID format
                if junction_id.startswith('GS_'):
                    node_id = junction_id[3:]
                    try:
                        node = net.getNode(node_id)
                    except:
                        pass
                else:
                    try:
                        node = net.getNode('GS_' + junction_id)
                    except:
                        pass
        
        if node is None and net is not None:
            print(f"  Warning: Could not find node for junction {junction_id}, skipping")
            continue
        
        # If net is None, we'll use a fallback method
        if net is None:
            print(f"  Warning: sumolib network not available for {junction_id}, using fallback method")
        
        # Get TLS object if available
        tls_obj = tls_dict.get(junction_id)
        if tls_obj is None:
            # Try alternative ID format
            if junction_id.startswith('GS_'):
                tls_obj = tls_dict.get(junction_id[3:])
            else:
                tls_obj = tls_dict.get('GS_' + junction_id)
        
        # Build mapping from signal index to connection
        signal_to_connection = {}
        signal_to_is_ramp = {}
        num_signals = 0
        ramp_signals = []
        mainline_signals = []
        
        if net is not None and node is not None:
            # Use sumolib method (more accurate)
            # Get all connections from the node
            node_connections = node.getConnections()
            
            # Identify ramp connections and mainline connections
            ramp_connections = []
            mainline_connections = []
            
            for conn in node_connections:
                try:
                    from_edge = conn.getFrom()
                    if from_edge is None:
                        continue
                    
                    from_edge_id = from_edge.getID()
                    from_type = edge_info.get(from_edge_id, {}).get("type")
                    
                    if from_type in ramp_types:
                        ramp_connections.append(conn)
                    else:
                        # Check if it's a mainline (high speed)
                        from_speed = edge_info.get(from_edge_id, {}).get("speed", 0)
                        if from_speed >= 22.0:
                            mainline_connections.append(conn)
                except Exception as e:
                    continue
            
            if not ramp_connections:
                print(f"  Warning: No ramp connections found for {junction_id}, skipping TLS generation")
                continue
            
            # Build mapping from signal index to connection using TLS object
            if tls_obj and hasattr(tls_obj, '_connections'):
                tls_connections = tls_obj._connections
                for conn_tuple in tls_connections:
                    if len(conn_tuple) >= 2:
                        signal_idx = conn_tuple[-1]  # Last element is signal index
                        lane = conn_tuple[0]  # First element is lane
                        
                        # Try to get connection from tuple if available
                        connection = None
                        if len(conn_tuple) >= 3:
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
                                    continue
                        
                        if connection is not None:
                            signal_to_connection[signal_idx] = connection
                            # Check if this connection is from a ramp
                            try:
                                from_edge = connection.getFrom()
                                if from_edge:
                                    from_edge_id = from_edge.getID()
                                    from_type = edge_info.get(from_edge_id, {}).get("type")
                                    signal_to_is_ramp[signal_idx] = from_type in ramp_types
                            except:
                                signal_to_is_ramp[signal_idx] = False
            else:
                # Fallback: map connections directly
                # This is less accurate but works when TLS object is not available
                for idx, conn in enumerate(node_connections):
                    signal_to_connection[idx] = conn
                    try:
                        from_edge = conn.getFrom()
                        if from_edge:
                            from_edge_id = from_edge.getID()
                            from_type = edge_info.get(from_edge_id, {}).get("type")
                            signal_to_is_ramp[idx] = from_type in ramp_types
                    except:
                        signal_to_is_ramp[idx] = False
        else:
            # Fallback method: use XML connections directly
            # Build junction -> incoming edges mapping
            junction_incoming = {}
            for edge in root.findall("edge"):
                if edge.get("function") == "internal":
                    continue
                edge_id = edge.get("id")
                to_junction = edge.get("to")
                if to_junction == junction_id:
                    if junction_id not in junction_incoming:
                        junction_incoming[junction_id] = []
                    junction_incoming[junction_id].append(edge_id)
            
            # Get incoming edges for this junction
            incoming_edges = junction_incoming.get(junction_id, [])
            
            # Identify ramp edges
            ramp_edges = [
                e for e in incoming_edges
                if e in edge_info and edge_info[e].get("type") in ramp_types
            ]
            
            if not ramp_edges:
                print(f"  Warning: No ramp edges found for {junction_id}, skipping TLS generation")
                continue
            
            # Count lanes for ramp and mainline
            ramp_lanes = 0
            mainline_lanes = 0
            
            for edge_id in ramp_edges:
                edge_elem = root.find(f"./edge[@id='{edge_id}']")
                if edge_elem is not None:
                    ramp_lanes += len(edge_elem.findall("lane"))
            
            # Identify mainline edges
            mainline_edges = [
                e for e in incoming_edges
                if e in edge_info
                and edge_info[e].get("type") not in ramp_types
                and edge_info[e].get("speed", 0) >= 22.0
            ]
            
            for edge_id in mainline_edges:
                edge_elem = root.find(f"./edge[@id='{edge_id}']")
                if edge_elem is not None:
                    mainline_lanes += len(edge_elem.findall("lane"))
            
            # Create simple mapping: mainline signals first, then ramp signals
            total_lanes = ramp_lanes + mainline_lanes
            if total_lanes == 0:
                print(f"  Warning: No lanes found for {junction_id}, skipping")
                continue
            
            # Simple mapping: mainline lanes get indices 0 to mainline_lanes-1, ramp lanes get mainline_lanes to total_lanes-1
            for idx in range(mainline_lanes):
                signal_to_is_ramp[idx] = False
            for idx in range(mainline_lanes, total_lanes):
                signal_to_is_ramp[idx] = True
            
            num_signals = total_lanes
            ramp_signals = list(range(mainline_lanes, total_lanes))
            mainline_signals = list(range(mainline_lanes))
        
        # Determine number of signals and signal lists
        if net is not None and node is not None:
            if not signal_to_connection:
                print(f"  Warning: No signal-to-connection mapping found for {junction_id}, skipping")
                continue
            
            num_signals = max(signal_to_connection.keys()) + 1 if signal_to_connection else 0
            if num_signals == 0:
                print(f"  Warning: No signals found for {junction_id}, skipping")
                continue
            
            # Count ramp and mainline signals
            ramp_signals = [idx for idx, is_ramp in signal_to_is_ramp.items() if is_ramp]
            mainline_signals = [idx for idx, is_ramp in signal_to_is_ramp.items() if not is_ramp]
        else:
            # Fallback method: num_signals, ramp_signals, mainline_signals already set above
            if num_signals == 0:
                print(f"  Warning: No signals found for {junction_id}, skipping")
                continue
        
        # Create tlLogic element
        tl_logic = ET.SubElement(root, "tlLogic")
        tl_logic.set("id", junction_id)
        tl_logic.set("type", "static")
        tl_logic.set("programID", "0")
        tl_logic.set("offset", "0")
        
        # Phase 1: RAMP_CLOSED (0) - Ramp red, mainline and others green
        # For ramp metering at merge junctions:
        # - Ramp signals: red (controlled to meter flow)
        # - Mainline/through signals: green (maintain highway flow)
        # - Default for unclassified: green (avoid blocking)
        phase1_state = ['G'] * num_signals  # Default: green for all
        for idx in ramp_signals:
            if idx < num_signals:
                phase1_state[idx] = 'r'  # Only ramp signals are red when closed
        
        phase1 = ET.SubElement(tl_logic, "phase")
        phase1.set("duration", str(default_green_duration))
        phase1.set("state", ''.join(phase1_state))
        phase1.set("name", "RAMP_CLOSED")
        
        # Phase 2: RAMP_OPEN (1) - All green
        phase2_state = ['G'] * num_signals
        
        phase2 = ET.SubElement(tl_logic, "phase")
        phase2.set("duration", str(default_red_duration))
        phase2.set("state", ''.join(phase2_state))
        phase2.set("name", "RAMP_OPEN")
        
        generated_count += 1
        print(f"  Generated TLS logic for {junction_id}: {len(ramp_signals)} ramp signals, {len(mainline_signals)} mainline signals, {num_signals} total signals")
    
    return generated_count


def check_connection_conflicts(
    connections: List,
    signal_to_connection: Dict[int, Connection]
) -> Dict[int, Set[int]]:
    """
    Check for conflicts between connections at a junction using SUMO's built-in conflict detection.
    Two connections conflict if they cross each other's paths or have incompatible movements.
    
    Args:
        connections: List of Connection objects from sumolib
        signal_to_connection: Dictionary mapping signal index to Connection object
        
    Returns:
        Dictionary mapping signal index to set of conflicting signal indices
    """
    conflicts = {}
    
    # Build reverse mapping: connection -> signal index
    connection_to_signal = {}
    for signal_idx, conn in signal_to_connection.items():
        connection_to_signal[id(conn)] = signal_idx
    
    # Check conflicts for each connection
    for signal_idx, conn in signal_to_connection.items():
        conflicts[signal_idx] = set()
        
        try:
            # Use SUMO's built-in getFoes() method if available
            if hasattr(conn, 'getFoes'):
                foes = conn.getFoes()
                # foes is a list of conflicting connections
                for foe_conn in foes:
                    foe_signal_idx = connection_to_signal.get(id(foe_conn))
                    if foe_signal_idx is not None and foe_signal_idx != signal_idx:
                        conflicts[signal_idx].add(foe_signal_idx)
            else:
                # Fallback: manual conflict detection based on connection geometry
                from_edge_a = conn.getFrom()
                to_edge_a = conn.getTo()
                from_lane_a = conn.getFromLane()
                to_lane_a = conn.getToLane()
                
                if not all([from_edge_a, to_edge_a, from_lane_a, to_lane_a]):
                    continue
                
                direction_a = conn.getDirection()
                
                # Check against all other connections
                for other_idx, other_conn in signal_to_connection.items():
                    if signal_idx >= other_idx:  # Avoid duplicate checks
                        continue
                    
                    try:
                        from_edge_b = other_conn.getFrom()
                        to_edge_b = other_conn.getTo()
                        from_lane_b = other_conn.getFromLane()
                        to_lane_b = other_conn.getToLane()
                        
                        if not all([from_edge_b, to_edge_b, from_lane_b, to_lane_b]):
                            continue
                        
                        direction_b = other_conn.getDirection()
                        
                        # Check for conflicts based on connection patterns
                        is_conflict = False
                        
                        # Get edge positions to determine if movements are opposing
                        from_pos_a = from_edge_a.getFromNode().getCoord() if hasattr(from_edge_a, 'getFromNode') else None
                        to_pos_a = to_edge_a.getToNode().getCoord() if hasattr(to_edge_a, 'getToNode') else None
                        from_pos_b = from_edge_b.getFromNode().getCoord() if hasattr(from_edge_b, 'getFromNode') else None
                        to_pos_b = to_edge_b.getToNode().getCoord() if hasattr(to_edge_b, 'getToNode') else None
                        
                        # Rule 1: Opposing straight movements conflict
                        if direction_a == 's' and direction_b == 's':
                            # Check if they are opposing (from opposite directions to opposite directions)
                            if from_edge_a.getID() != from_edge_b.getID() and to_edge_a.getID() != to_edge_b.getID():
                                # Additional check: ensure they are truly opposing by checking if paths cross
                                is_conflict = True
                        
                        # Rule 2: Left turn conflicts with opposing straight (classic intersection conflict)
                        if (direction_a in ['l', 'L'] and direction_b == 's') or (direction_a == 's' and direction_b in ['l', 'L']):
                            # Only if from different edges (opposing directions)
                            if from_edge_a.getID() != from_edge_b.getID():
                                is_conflict = True
                        
                        # Rule 2b: Two left turns from opposing directions may conflict
                        if direction_a in ['l', 'L'] and direction_b in ['l', 'L']:
                            if from_edge_a.getID() != from_edge_b.getID():
                                is_conflict = True
                        
                        # Rule 3: Crossing paths (from different directions to different destinations)
                        if from_edge_a.getID() != from_edge_b.getID() and to_edge_a.getID() != to_edge_b.getID():
                            # Different incoming and outgoing edges - likely crossing
                            # This is a conservative rule - any two movements from different sources 
                            # going to different destinations are assumed to potentially conflict
                            if direction_a in ['s', 'l', 'L', 't'] and direction_b in ['s', 'l', 'L', 't']:
                                is_conflict = True
                        
                        # Rule 4: Same incoming edge, different outgoing edges (diverge, no conflict)
                        if from_edge_a.getID() == from_edge_b.getID() and to_edge_a.getID() != to_edge_b.getID():
                            is_conflict = False  # Diverging movements don't conflict
                        
                        # Rule 5: Different incoming edges, same outgoing edge (merge)
                        # This is generally safe for ramp merging, but we need to check if the 
                        # incoming edges are actually merging at an acute angle (safe) vs. 
                        # crossing paths at obtuse angles (potentially unsafe)
                        if from_edge_a.getID() != from_edge_b.getID() and to_edge_a.getID() == to_edge_b.getID():
                            # For now, consider merging as no conflict (typical for highway ramps)
                            # In reality, this depends on the merge angle and traffic dynamics
                            is_conflict = False  # Merging movements (like ramp merging) don't conflict
                        
                        if is_conflict:
                            conflicts[signal_idx].add(other_idx)
                            if other_idx not in conflicts:
                                conflicts[other_idx] = set()
                            conflicts[other_idx].add(signal_idx)
                            
                    except (AttributeError, KeyError):
                        continue
                        
        except (AttributeError, KeyError):
            continue
    
    return conflicts


def report_conflict_summary(
    junction_id: str,
    all_conflicts: Dict[int, Set[int]],
    signal_to_connection_map: Dict[int, Connection],
    ramp_signals: List[int],
    mainline_signals: List[int]
) -> Tuple[bool, bool]:
    """
    Report a detailed summary of conflicts at a junction.
    
    Args:
        junction_id: Junction ID
        all_conflicts: Dictionary mapping signal index to set of conflicting signal indices
        signal_to_connection_map: Dictionary mapping signal index to Connection object
        ramp_signals: List of ramp signal indices
        mainline_signals: List of mainline signal indices
        
    Returns:
        Tuple of (has_mainline_conflicts, has_ramp_mainline_conflicts)
    """
    # Helper function to get connection description
    def describe_connection(conn_idx):
        """Get a human-readable description of a connection."""
        if conn_idx not in signal_to_connection_map:
            return f"signal_{conn_idx}"
        conn = signal_to_connection_map[conn_idx]
        try:
            from_edge = conn.getFrom()
            to_edge = conn.getTo()
            direction = conn.getDirection()
            from_id = from_edge.getID() if from_edge else "?"
            to_id = to_edge.getID() if to_edge else "?"
            dir_map = {'s': 'straight', 'l': 'left', 'r': 'right', 'L': 'left', 'R': 'right', 't': 'turn'}
            dir_str = dir_map.get(direction, direction) if direction else "unknown"
            return f"{from_id} -> {to_id} ({dir_str})"
        except:
            return f"signal_{conn_idx}"
    
    print(f"  冲突检测报告 (Conflict Detection Report) for {junction_id}:")
    print(f"  " + "=" * 70)
    
    # Check conflicts among mainline signals
    mainline_signal_conflicts = []
    for i in mainline_signals:
        for j in mainline_signals:
            if i < j and j in all_conflicts.get(i, set()):
                mainline_signal_conflicts.append((i, j))
    
    has_mainline_conflicts = len(mainline_signal_conflicts) > 0
    
    if mainline_signal_conflicts:
        print(f"  ⚠ 主线车道冲突 (Mainline Conflicts): {len(mainline_signal_conflicts)} 个")
        for idx, (i, j) in enumerate(mainline_signal_conflicts[:5]):
            print(f"    [{idx+1}] Signal {i} <-> Signal {j}")
            print(f"        {describe_connection(i)}")
            print(f"        {describe_connection(j)}")
        if len(mainline_signal_conflicts) > 5:
            print(f"    ... 还有 {len(mainline_signal_conflicts)-5} 个冲突")
        print(f"  ⚠ 警告: 当前配时方案将所有主线设为绿灯,但存在冲突!")
        print(f"  ⚠ 这可能导致对向来车或交叉路径的车辆相撞")
    else:
        print(f"  ✓ 主线车道之间无冲突")
    
    # Check if ramp signals conflict with mainline signals
    ramp_mainline_conflicts = []
    for ramp_idx in ramp_signals:
        for mainline_idx in mainline_signals:
            if mainline_idx in all_conflicts.get(ramp_idx, set()) or ramp_idx in all_conflicts.get(mainline_idx, set()):
                ramp_mainline_conflicts.append((ramp_idx, mainline_idx))
    
    has_ramp_mainline_conflicts = len(ramp_mainline_conflicts) > 0
    
    if ramp_mainline_conflicts:
        print(f"  ⚠ 匝道-主线冲突 (Ramp-Mainline Conflicts): {len(ramp_mainline_conflicts)} 个")
        for idx, (ramp_idx, mainline_idx) in enumerate(ramp_mainline_conflicts[:5]):
            print(f"    [{idx+1}] Ramp Signal {ramp_idx} <-> Mainline Signal {mainline_idx}")
            print(f"        匝道: {describe_connection(ramp_idx)}")
            print(f"        主线: {describe_connection(mainline_idx)}")
        if len(ramp_mainline_conflicts) > 5:
            print(f"    ... 还有 {len(ramp_mainline_conflicts)-5} 个冲突")
        print(f"  ⚠ 警告: 匝道与主线轨迹相交而非合流,不适合匝道控制")
    else:
        print(f"  ✓ 匝道与主线之间无冲突 (正常合流)")
    
    print(f"  " + "=" * 70)
    
    return has_mainline_conflicts, has_ramp_mainline_conflicts


def convert_existing_tls_to_ramp_metering(
    root: ET.Element,
    junction_ids: Set[str],
    edge_info: Dict[str, Dict[str, any]],
    net: Optional[sumolib.net.Net],
    ramp_types: Optional[Set[str]] = None,
    default_green_duration: int = 300,
    default_red_duration: int = 5,
    check_conflicts: bool = True
) -> int:
    """
    Convert existing traffic light signals at ramp merge junctions to ramp metering logic.
    
    This function:
    1. Finds existing tlLogic elements for the specified junctions
    2. Identifies which signals control ramp lanes vs mainline lanes
    3. Checks for conflicts between connections (optional)
    4. Replaces the existing phase logic with ramp metering logic (RAMP_OPEN/RAMP_CLOSED)
    
    Args:
        root: Root element of the network XML
        junction_ids: Set of junction IDs that are ramp merge points with existing TLS
        edge_info: Edge information dictionary
        net: SUMO network object from sumolib
        ramp_types: Set of edge types to consider as ramps
        default_green_duration: Default green phase duration in seconds
        default_red_duration: Default red phase duration in seconds
        check_conflicts: Whether to check for conflicts between connections
        
    Returns:
        Number of TLS logic elements converted
    """
    if ramp_types is None:
        ramp_types = {"highway.motorway_link", "highway.trunk_link"}
    
    converted_count = 0
    
    # Get TLS dictionary from network if available
    tls_dict = {}
    if net is not None:
        tls_dict = {tls.getID(): tls for tls in net.getTrafficLights()}
    
    # Find existing tlLogic elements
    existing_tl_logic = {tl.get("id"): tl for tl in root.findall("tlLogic")}
    
    for junction_id in junction_ids:
        # Check if tlLogic exists for this junction
        if junction_id not in existing_tl_logic:
            print(f"  Warning: No existing tlLogic found for {junction_id}, skipping conversion")
            continue
        
        tl_logic = existing_tl_logic[junction_id]
        
        # Get the node/junction for this TLS
        node = None
        if net is not None:
            try:
                node = net.getNode(junction_id)
            except:
                # Try alternative TLS ID format
                if junction_id.startswith('GS_'):
                    node_id = junction_id[3:]
                    try:
                        node = net.getNode(node_id)
                    except:
                        pass
                else:
                    try:
                        node = net.getNode('GS_' + junction_id)
                    except:
                        pass
        
        if node is None and net is not None:
            print(f"  Warning: Could not find node for junction {junction_id}, skipping")
            continue
        
        # Get TLS object if available
        tls_obj = tls_dict.get(junction_id)
        if tls_obj is None:
            # Try alternative ID format
            if junction_id.startswith('GS_'):
                tls_obj = tls_dict.get(junction_id[3:])
            else:
                tls_obj = tls_dict.get('GS_' + junction_id)
        
        # Build mapping from signal index to connection
        signal_to_is_ramp = {}
        ramp_signals = []
        mainline_signals = []
        
        if net is not None and node is not None:
            # Use sumolib method (more accurate)
            # Get all connections from the node
            node_connections = node.getConnections()
            
            # Identify ramp connections and mainline connections
            ramp_connections = []
            mainline_connections = []
            
            for conn in node_connections:
                try:
                    from_edge = conn.getFrom()
                    if from_edge is None:
                        continue
                    
                    from_edge_id = from_edge.getID()
                    from_type = edge_info.get(from_edge_id, {}).get("type")
                    
                    if from_type in ramp_types:
                        ramp_connections.append(conn)
                    else:
                        # Check if it's a mainline (high speed)
                        from_speed = edge_info.get(from_edge_id, {}).get("speed", 0)
                        if from_speed >= 22.0:
                            mainline_connections.append(conn)
                except Exception as e:
                    continue
            
            if not ramp_connections:
                print(f"  Warning: No ramp connections found for {junction_id}, skipping conversion")
                continue
            
            # Build mapping from signal index to connection using TLS object
            if tls_obj and hasattr(tls_obj, '_connections'):
                tls_connections = tls_obj._connections
                for conn_tuple in tls_connections:
                    if len(conn_tuple) >= 2:
                        signal_idx = conn_tuple[-1]  # Last element is signal index
                        lane = conn_tuple[0]  # First element is lane
                        
                        # Try to get connection from tuple if available
                        connection = None
                        if len(conn_tuple) >= 3:
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
                                    continue
                        
                        if connection is not None:
                            # Check if this connection is from a ramp
                            try:
                                from_edge = connection.getFrom()
                                if from_edge:
                                    from_edge_id = from_edge.getID()
                                    from_type = edge_info.get(from_edge_id, {}).get("type")
                                    if from_type in ramp_types:
                                        signal_to_is_ramp[signal_idx] = True
                                        ramp_signals.append(signal_idx)
                                    else:
                                        signal_to_is_ramp[signal_idx] = False
                                        mainline_signals.append(signal_idx)
                            except:
                                signal_to_is_ramp[signal_idx] = False
                                mainline_signals.append(signal_idx)
        
        # Get number of signals from existing phases
        existing_phases = tl_logic.findall("phase")
        if not existing_phases:
            print(f"  Warning: No phases found in existing tlLogic for {junction_id}, skipping")
            continue
        
        # Get number of signals from first phase state
        first_phase = existing_phases[0]
        first_state = first_phase.get("state", "")
        num_signals = len(first_state)
        
        if not ramp_signals:
            print(f"  Warning: Could not identify ramp signals for {junction_id}, skipping conversion")
            continue
        
        # Build signal_to_connection mapping for conflict checking
        signal_to_connection_map = {}
        if tls_obj and hasattr(tls_obj, '_connections'):
            for conn_tuple in tls_obj._connections:
                if len(conn_tuple) >= 2:
                    signal_idx = conn_tuple[-1]
                    lane = conn_tuple[0]
                    
                    # Find the connection object
                    connection = None
                    if len(conn_tuple) >= 3:
                        potential_conn = conn_tuple[1]
                        if hasattr(potential_conn, 'getFrom'):
                            connection = potential_conn
                    
                    if connection is None:
                        for conn in node_connections:
                            try:
                                if conn.getFromLane() == lane:
                                    connection = conn
                                    break
                            except:
                                continue
                    
                    if connection is not None:
                        signal_to_connection_map[signal_idx] = connection
        
        # Check for conflicts between connections if requested
        should_skip = False
        if check_conflicts and signal_to_connection_map:
            all_conflicts = check_connection_conflicts(node_connections, signal_to_connection_map)
            
            # Report conflict summary
            has_mainline_conflicts, has_ramp_mainline_conflicts = report_conflict_summary(
                junction_id,
                all_conflicts,
                signal_to_connection_map,
                ramp_signals,
                mainline_signals
            )
            
            # Skip conversion if there are warnings
            if has_mainline_conflicts or has_ramp_mainline_conflicts:
                should_skip = True
                print(f"  ⚠ 跳过转换 {junction_id}: 检测到冲突警告，不适合转换为匝道控制")
                if has_mainline_conflicts:
                    print(f"  ⚠ 安全建议: 考虑使用标准交叉口信号控制,而非简单匝道控制")
                if has_ramp_mainline_conflicts:
                    print(f"  ⚠ 安全建议: 路网设计可能不适合匝道控制场景")
        
        # Skip conversion if warnings detected
        if should_skip:
            continue
        
        # Remove all existing phases
        for phase in existing_phases:
            tl_logic.remove(phase)
        
        # Create new ramp metering phases
        # Phase 1: RAMP_CLOSED - Ramp red, mainline green
        phase1_state = ['G'] * num_signals  # Default: green for all
        for idx in ramp_signals:
            if idx < num_signals:
                phase1_state[idx] = 'r'  # Only ramp signals are red when closed
        
        phase1 = ET.SubElement(tl_logic, "phase")
        phase1.set("duration", str(default_red_duration))
        phase1.set("state", ''.join(phase1_state))
        phase1.set("name", "RAMP_CLOSED")
        
        # Phase 2: RAMP_OPEN - All green
        phase2_state = ['G'] * num_signals
        
        phase2 = ET.SubElement(tl_logic, "phase")
        phase2.set("duration", str(default_green_duration))
        phase2.set("state", ''.join(phase2_state))
        phase2.set("name", "RAMP_OPEN")
        
        # Update TLS type to static if needed
        if tl_logic.get("type") != "static":
            tl_logic.set("type", "static")
        
        converted_count += 1
        print(f"  Converted TLS logic for {junction_id}: {len(ramp_signals)} ramp signals, {len(mainline_signals)} mainline signals, {num_signals} total signals")
    
    return converted_count


def process_net_file(
    net_file_path: Path,
    min_highway_speed: float = 22.0,
    ramp_types: Optional[Set[str]] = None,
    generate_tl_logic: bool = False,
    run_netconvert: bool = False,
    tls_type: str = "actuated",
    dry_run: bool = False,
    backup: bool = False,
    output_suffix: Optional[str] = "_with_ramp_tls",
    output_path: Optional[Path] = None,
    convert_existing: bool = False
) -> Tuple[bool, int, int]:
    """
    Process a single SUMO network file to add ramp metering.
    
    Args:
        net_file_path: Path to the input net.xml file
        min_highway_speed: Minimum speed (m/s) to consider as highway
        ramp_types: Set of edge types to consider as ramps
        generate_tl_logic: Whether to generate TLS logic directly in XML
        run_netconvert: Whether to run netconvert after conversion
        tls_type: TLS type for netconvert
        dry_run: Only identify junctions without modifying
        backup: Whether to create a backup before modifying
        output_suffix: Suffix to add to output filename
        
    Returns:
        Tuple of (success, converted_count, generated_count)
        success: True if successful, False otherwise
        converted_count: Number of junctions converted to traffic_light
        generated_count: Number of TLS logic elements generated
    """
    if not net_file_path.exists():
        print(f"Error: File not found: {net_file_path}")
        return False, 0, 0
    
    print(f"\nProcessing: {net_file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = net_file_path.with_suffix('.net.xml.ramp_backup')
        import shutil
        shutil.copy2(net_file_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Generate output filename
    if output_path is None:
        if output_suffix:
            output_path = net_file_path.parent / f"{net_file_path.stem}{output_suffix}.net.xml"
        else:
            output_path = net_file_path.parent / f"{net_file_path.stem}.xml"
    print(f"  Output file: {output_path}")
    
    # Read network file
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Error: Failed to parse XML file: {e}")
        return False, 0, 0
    
    print(f"  Building edge information...")
    edge_info = build_edge_info(root)
    print(f"  Found {len(edge_info)} edges")
    
    print(f"  Identifying ramp → highway junctions...")
    print(f"    Minimum highway speed: {min_highway_speed} m/s")
    print(f"    Ramp types: {ramp_types}")
    
    ramp_junctions = identify_ramp_to_highway_junctions(
        root,
        edge_info,
        min_highway_speed=min_highway_speed,
        ramp_types=ramp_types or {"highway.motorway_link", "highway.trunk_link"}
    )
    
    # Handle existing traffic lights
    existing_tl_logic = {tl.get("id"): tl for tl in root.findall("tlLogic")}
    existing_tl_logic_ids = set(existing_tl_logic.keys())
    junctions_to_exclude = set()
    junctions_to_convert = set()
    
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if junction_id in ramp_junctions:
            current_type = junction.get("type")
            # If it's already a traffic_light with existing tlLogic
            if current_type == "traffic_light" and junction_id in existing_tl_logic_ids:
                if convert_existing:
                    # Convert existing signal to ramp metering
                    junctions_to_convert.add(junction_id)
                    print(f"  Will convert existing TLS at {junction_id} to ramp metering")
                else:
                    # Exclude it (normal intersection signal)
                    junctions_to_exclude.add(junction_id)
                    print(f"  Excluding {junction_id}: already has traffic_light with tlLogic (normal intersection)")
    
    ramp_junctions = ramp_junctions - junctions_to_exclude - junctions_to_convert
    
    if ramp_junctions:
        print(f"  Found {len(ramp_junctions)} ramp merge junctions (will not convert, only using existing traffic lights):")
        for j_id in sorted(ramp_junctions):
            print(f"    - {j_id}")
    
    if junctions_to_convert:
        print(f"  Found {len(junctions_to_convert)} existing traffic lights to convert to ramp metering:")
        for j_id in sorted(junctions_to_convert):
            print(f"    - {j_id}")
    
    if junctions_to_exclude:
        print(f"  Excluded {len(junctions_to_exclude)} junctions that are already normal intersection signals:")
        for j_id in sorted(junctions_to_exclude):
            print(f"    - {j_id}")
    
    if dry_run:
        print("  [DRY RUN] No modifications made.")
        return True, len(junctions_to_convert), 0
    
    # Only use existing traffic lights for conversion, skip converting new junctions
    if not junctions_to_convert:
        print("  No existing traffic lights found at ramp junctions. Nothing to convert.")
        return True, 0, 0
    
    converted_count = 0
    generated_count = 0
    
    # Convert existing traffic lights to ramp metering
    print(f"  Converting {len(junctions_to_convert)} existing traffic lights to ramp metering...")
    # Read network using sumolib for proper connection mapping
    try:
        net = sumolib.net.readNet(str(net_file_path), withPrograms=True)
    except Exception as e:
        print(f"    Error reading network with sumolib: {e}")
        print("    Falling back to XML-only method (may be less accurate)")
        net = None
    
    converted_count = convert_existing_tls_to_ramp_metering(
        root,
        junctions_to_convert,
        edge_info,
        net=net,
        ramp_types=ramp_types or {"highway.motorway_link", "highway.trunk_link"},
        default_green_duration=300,
        default_red_duration=5,
        check_conflicts=True  # Enable conflict detection
    )
    print(f"  Converted {converted_count} existing TLS logic elements to ramp metering")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write modified network
    print(f"  Writing modified network to: {output_path}")
    try:
        # Pretty print with indentation
        ET.indent(root, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print("  ✓ Network file written successfully")
    except Exception as e:
        print(f"  ✗ Failed to write output file: {e}")
        return False, converted_count, generated_count
    
    # Optionally run netconvert
    if run_netconvert:
        # Use a temporary intermediate file
        intermediate_path = output_path.with_suffix(".intermediate.net.xml")
        
        # First, copy the modified file to intermediate
        import shutil
        shutil.copy2(output_path, intermediate_path)
        
        # Run netconvert
        success = run_netconvert(intermediate_path, output_path, tls_type=tls_type)
        
        # Clean up intermediate file
        if intermediate_path.exists():
            intermediate_path.unlink()
        
        if not success:
            print("  Warning: netconvert failed, but the network file with traffic_light junctions has been saved.")
            print(f"  You can run netconvert manually later:")
            print(f"    netconvert --sumo-net-file {output_path} --output-file {output_path} --tls.guess --tls.default-type {tls_type}")
            return False, converted_count, generated_count
    
    return True, converted_count, generated_count


def run_netconvert(
    input_net: Path,
    output_net: Path,
    tls_type: str = "actuated"
) -> bool:
    """
    Run netconvert to generate TLS logic for traffic light junctions.
    
    Why netconvert is needed:
    - Simply changing junction type to "traffic_light" is not enough
    - SUMO requires <tlLogic> elements that define:
      * Phase sequences (which lanes get green/red/yellow)
      * Phase durations
      * State strings (e.g., "GGGrrrrrr..." matching controlled lanes)
      * Lane-to-signal-index mappings
    - Manually generating <tlLogic> is complex because you need to:
      * Analyze all incoming/outgoing lanes for each junction
      * Determine controlled lane order (must match state string length)
      * Handle conflict detection and phase compatibility
    - netconvert automatically does all this analysis and generates proper TLS logic
    
    Alternative approaches:
    1. Use --generate-tl-logic: Generate simple 2-phase logic (mainline green, ramp controlled)
    2. Control via TraCI at runtime: No <tlLogic> needed, control dynamically
    
    Args:
        input_net: Input network file path
        output_net: Output network file path
        tls_type: TLS type (default: "actuated")
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "netconvert",
        "--sumo-net-file", str(input_net),
        "--output-file", str(output_net),
        "--tls.guess",
        f"--tls.default-type", tls_type
    ]
    
    print(f"\nRunning netconvert to generate TLS logic...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ netconvert completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ netconvert failed with exit code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ netconvert not found. Please ensure SUMO is installed and in PATH.")
        print("  You can skip this step and run netconvert manually later.")
        return False


# All regions to process (can be customized)
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


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Add ramp metering (traffic lights) to highway on-ramps in SUMO network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode: convert ramp junctions to traffic lights
  python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml
  
  # Single file with TLS logic generation
  python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --generate-tl-logic
  
  # Batch mode: process specific regions
  python tools/add_ramp_metering.py --regions mzw jinan --generate-tl-logic
  
  # Batch mode: process all regions
  python tools/add_ramp_metering.py --all-regions --generate-tl-logic --backup
  
  # Custom speed threshold (m/s)
  python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --min-speed 25.0
  
  # Custom TLS type
  python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --run-netconvert --tls-type static
  
  # Convert existing traffic lights to ramp metering
  python tools/add_ramp_metering.py --input net.xml --output net_with_ramp_tls.net.xml --convert-existing
        """
    )
    
    # Single file mode arguments
    ap.add_argument(
        "--input",
        help="Input SUMO network file (.net.xml) - for single file mode"
    )
    ap.add_argument(
        "--output",
        help="Output SUMO network file (.net.xml) - for single file mode"
    )
    
    # Batch mode arguments
    ap.add_argument(
        "--regions",
        nargs="+",
        help="List of regions to process in batch mode (e.g., mzw jinan)"
    )
    ap.add_argument(
        "--all-regions",
        action="store_true",
        default=True,
        help="Process all available regions in batch mode"
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default="../sumo_config_highway",
        help="Base directory containing region folders (default: sumo_config)"
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup files before modifying (batch mode only)"
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        help="Suffix to add to output filename in batch mode (default: '_with_ramp_tls')"
    )
    ap.add_argument(
        "--min-speed",
        type=float,
        default=22.0,
        help="Minimum speed (m/s) to consider as highway (default: 22.0 ≈ 80 km/h)"
    )
    ap.add_argument(
        "--ramp-types",
        nargs="+",
        default=["highway.motorway_link", "highway.trunk_link"],
        help="Edge types to consider as ramps (default: highway.motorway_link highway.trunk_link)"
    )
    ap.add_argument(
        "--generate-tl-logic",
        action="store_true",
        help="Generate simple 0/1 ramp control TLS logic directly in XML. "
             "Creates two phases: RAMP_CLOSED (ramp red) and RAMP_OPEN (ramp green). "
             "Uses sumolib to correctly map signal indices to connections. "
             "If not set, you can use --run-netconvert or control via TraCI at runtime."
    )
    ap.add_argument(
        "--run-netconvert",
        action="store_true",
        help="Run netconvert to generate TLS logic after conversion. "
             "Note: netconvert will generate full TLS logic automatically."
    )
    ap.add_argument(
        "--tls-type",
        default="actuated",
        choices=["static", "actuated", "delay_based"],
        help="TLS type for netconvert (default: actuated)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only identify junctions without modifying the network"
    )
    ap.add_argument(
        "--convert-existing",
        action="store_true",
        default=True,
        help="Convert existing traffic lights at ramp merge junctions to ramp metering logic. "
             "Instead of creating new traffic lights, this modifies existing ones to use "
             "RAMP_OPEN/RAMP_CLOSED phases. Use this when ramps already have traffic lights."
    )
    
    args = ap.parse_args()
    
    # Determine mode: single file or batch
    is_batch_mode = args.all_regions or args.regions is not None
    
    if is_batch_mode:
        # Batch mode: process multiple network files
        if args.all_regions:
            regions = ALL_REGIONS
        elif args.regions:
            regions = args.regions
        else:
            print("Error: Must specify --regions or --all-regions for batch mode")
            return 1
        
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"Error: Base directory not found: {base_dir}")
            return 1
        
        print("=" * 60)
        print("Adding Ramp Metering to SUMO Network Files")
        print("=" * 60)
        print(f"Regions to process: {len(regions)}")
        print(f"Backup mode: {args.backup}")
        print(f"Generate TLS logic: {args.generate_tl_logic}")
        print(f"Run netconvert: {args.run_netconvert}")
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
            
            success, converted_count, generated_count = process_net_file(
                net_file,
                min_highway_speed=args.min_speed,
                ramp_types=set(args.ramp_types),
                generate_tl_logic=args.generate_tl_logic,
                run_netconvert=args.run_netconvert,
                tls_type=args.tls_type,
                dry_run=args.dry_run,
                backup=args.backup,
                output_suffix=args.output_suffix,
                convert_existing=args.convert_existing
            )
            
            network_stats.append({
                'region': region,
                'success': success,
                'converted_count': converted_count,
                'generated_count': generated_count
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
        print("匝道控制统计 (Ramp Metering Statistics)")
        print("=" * 80)
        print(f"{'路网 (Network)':<25} {'转换路口数':<15} {'生成TLS数':<15} {'状态':<15}")
        print("-" * 80)
        
        total_converted = 0
        total_generated = 0
        
        for stat in network_stats:
            region = stat['region']
            converted = stat['converted_count']
            generated = stat['generated_count']
            status = "✓ Success" if stat['success'] else "✗ Failed"
            
            total_converted += converted
            total_generated += generated
            
            print(f"{region:<25} {converted:<15} {generated:<15} {status:<15}")
        
        print("-" * 80)
        print(f"{'总计 (Total)':<25} {total_converted:<15} {total_generated:<15} {'':<15}")
        print("=" * 80)
        
        return 0 if fail_count == 0 else 1
    
    else:
        # Single file mode
        if not args.input or not args.output:
            print("Error: Must specify --input and --output for single file mode, or use --regions/--all-regions for batch mode")
            return 1
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1
        
        success, converted_count, generated_count = process_net_file(
            input_path,
            min_highway_speed=args.min_speed,
            ramp_types=set(args.ramp_types),
            generate_tl_logic=args.generate_tl_logic,
            run_netconvert=args.run_netconvert,
            tls_type=args.tls_type,
            dry_run=args.dry_run,
            backup=False,
            output_suffix=None,
            output_path=output_path,
            convert_existing=args.convert_existing
        )
        
        if success and not args.dry_run:
            print(f"\n✓ Successfully added ramp metering to {converted_count} junctions")
            print(f"  Output file: {output_path}")
        
        if not args.run_netconvert and not args.generate_tl_logic and not args.dry_run:
            print("\nNote: TLS logic not generated. You have two options:")
            print("  1. Run netconvert to generate full TLS logic:")
            print(f"     netconvert --sumo-net-file {output_path} --output-file {output_path} --tls.guess --tls.default-type {args.tls_type}")
            print("  2. Control traffic lights dynamically via TraCI at runtime")
            print("     (The junctions are already traffic_light type, ready for TraCI control)")
        
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

