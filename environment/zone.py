"""
Zone Manager for Traffic Analysis Zone (TAZ) based infrastructure organization.

This module provides functionality to:
- Parse TAZ XML files and build zone dictionaries
- Map infrastructure (lanes, intersections, highways, ramps, transit) to zones
- Build zone adjacency graphs
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx


class ZoneManager:
    """
    Manages zone-based infrastructure organization.

    Single zone loading mode - processes the TAZ file for the current simulation scenario.
    Each zone contains references to all infrastructure elements within its boundaries.
    """

    def __init__(self, taz_file_path: str, sumo_net, traci_conn=None):
        """
        Initialize ZoneManager.

        Args:
            taz_file_path: Path to the districts.taz.xml file
            sumo_net: SUMO network object (sumolib.net.Net)
            traci_conn: TraCI connection (optional, for runtime queries)
        """
        self.taz_file_path = taz_file_path
        self.sumo_net = sumo_net
        self.traci_conn = traci_conn

        # Internal mappings built during zone_dict construction
        self._edge_to_zone: Dict[str, str] = {}
        self._lane_to_zone: Dict[str, str] = {}

    def build_zone_dict(self,
                        highway_edges: Optional[Set[str]] = None,
                        ramp_tls_lanes: Optional[Dict[str, List[str]]] = None,
                        subway_stations: Optional[Dict] = None,
                        bus_stations: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Build zone dictionary from TAZ file.

        Args:
            highway_edges: Set of edge IDs that are highways (optional)
            ramp_tls_lanes: Dict mapping ramp TLS ID to controlled lane IDs (optional)
            subway_stations: Dict of subway station objects (optional)
            bus_stations: Dict of bus station objects (optional)

        Returns:
            Dict mapping zone_id to zone information
        """
        # Parse TAZ file
        raw_zones = self._parse_taz_file()

        # Build zone dictionary with infrastructure mappings
        zone_dict = {}

        for zone_id, zone_data in raw_zones.items():
            edges = zone_data['edges']
            shape = zone_data['shape']

            # Derive lanes from edges
            lanes = self._get_lanes_from_edges(edges)

            # Map intersections (junctions at edge endpoints)
            intersections = self._get_intersections_from_edges(edges)

            # Map highways
            highway_segments = []
            zone_highway_edges = []
            if highway_edges:
                zone_highway_edges = [e for e in edges if e in highway_edges]
                highway_segments = self._get_highway_segments(zone_highway_edges)

            # Map ramps
            ramps = []
            ramp_lanes = []
            if ramp_tls_lanes:
                for tls_id, tls_lanes in ramp_tls_lanes.items():
                    # Check if any ramp lane is in this zone
                    for lane in tls_lanes:
                        if lane in lanes:
                            if tls_id not in ramps:
                                ramps.append(tls_id)
                            if lane not in ramp_lanes:
                                ramp_lanes.append(lane)

            # Build edge and lane to zone mappings
            for edge in edges:
                self._edge_to_zone[edge] = zone_id
            for lane in lanes:
                self._lane_to_zone[lane] = zone_id

            # Map subway stations
            zone_subway_stations = []
            if subway_stations:
                for station_id, station in subway_stations.items():
                    station_lane = getattr(station, 'lane_id', None)
                    if station_lane and station_lane in lanes:
                        zone_subway_stations.append(station_id)

            # Map bus stops
            zone_bus_stops = []
            if bus_stations:
                for station_id, station in bus_stations.items():
                    station_lane = getattr(station, 'lane_id', None)
                    if station_lane and station_lane in lanes:
                        zone_bus_stops.append(station_id)

            # Calculate statistics
            total_highway_length = self._calculate_highway_length(zone_highway_edges)

            zone_dict[zone_id] = {
                "zone_id": zone_id,
                "shape": self._parse_shape(shape),
                "edges": edges,
                "lanes": lanes,
                "intersections": intersections,
                "highway_segments": highway_segments,
                "highway_edges": zone_highway_edges,
                "ramps": ramps,
                "ramp_lanes": ramp_lanes,
                "subway_stations": zone_subway_stations,
                "bus_stops": zone_bus_stops,
                "stats": {
                    "total_lanes": len(lanes),
                    "total_intersections": len(intersections),
                    "total_highway_length_m": total_highway_length,
                    "total_bus_stops": len(zone_bus_stops),
                    "total_subway_stations": len(zone_subway_stations)
                }
            }

        return zone_dict

    def build_zone_graph(self, zone_dict: Dict) -> nx.DiGraph:
        """
        Build zone adjacency graph based on shared boundary edges.

        Args:
            zone_dict: Zone dictionary built by build_zone_dict()

        Returns:
            NetworkX DiGraph with zones as nodes and adjacency relationships as edges
        """
        graph = nx.DiGraph()

        # Add zone nodes with attributes
        for zone_id, zone_info in zone_dict.items():
            graph.add_node(
                zone_id,
                node_type='zone',
                edge_count=len(zone_info['edges']),
                intersection_count=len(zone_info['intersections']),
                has_highway=len(zone_info['highway_edges']) > 0,
                has_ramp=len(zone_info['ramps']) > 0,
                has_transit=(len(zone_info['subway_stations']) > 0 or
                           len(zone_info['bus_stops']) > 0)
            )

        # Find adjacent zones based on shared junctions/edges
        zone_ids = list(zone_dict.keys())
        for i, zone1_id in enumerate(zone_ids):
            zone1_edges = set(zone_dict[zone1_id]['edges'])
            zone1_junctions = set(zone_dict[zone1_id]['intersections'])

            for zone2_id in zone_ids[i+1:]:
                zone2_edges = set(zone_dict[zone2_id]['edges'])
                zone2_junctions = set(zone_dict[zone2_id]['intersections'])

                shared_junctions = zone1_junctions & zone2_junctions

                # Find edges that connect between zones
                shared_edges = self._find_connecting_edges(
                    zone1_edges, zone2_edges, shared_junctions
                )

                if shared_junctions or shared_edges:
                    # Add bidirectional edges
                    graph.add_edge(
                        zone1_id, zone2_id,
                        edge_type='adjacent',
                        shared_junctions=list(shared_junctions),
                        shared_edges=list(shared_edges)
                    )
                    graph.add_edge(
                        zone2_id, zone1_id,
                        edge_type='adjacent',
                        shared_junctions=list(shared_junctions),
                        shared_edges=list(shared_edges)
                    )

        return graph

    def get_edge_to_zone_mapping(self) -> Dict[str, str]:
        """Get edge_id -> zone_id mapping."""
        return self._edge_to_zone.copy()

    def get_lane_to_zone_mapping(self) -> Dict[str, str]:
        """Get lane_id -> zone_id mapping."""
        return self._lane_to_zone.copy()

    def get_zone_for_edge(self, edge_id: str) -> Optional[str]:
        """Get zone ID for a given edge."""
        return self._edge_to_zone.get(edge_id)

    def get_zone_for_lane(self, lane_id: str) -> Optional[str]:
        """Get zone ID for a given lane."""
        return self._lane_to_zone.get(lane_id)

    def _parse_taz_file(self) -> Dict[str, Dict]:
        """
        Parse TAZ XML file.

        Returns:
            Dict mapping zone_id to raw zone data (edges, shape)
        """
        try:
            tree = ET.parse(self.taz_file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse TAZ file {self.taz_file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"TAZ file not found: {self.taz_file_path}")

        zones = {}
        for taz in root.findall('.//taz'):
            zone_id = taz.get('id')
            if zone_id is None:
                continue

            edges_str = taz.get('edges', '')
            edges = edges_str.split() if edges_str else []
            shape = taz.get('shape', '')

            zones[zone_id] = {
                'edges': edges,
                'shape': shape
            }

        return zones

    def _parse_shape(self, shape_str: str) -> List[Tuple[float, float]]:
        """Parse shape string into list of coordinate tuples."""
        if not shape_str:
            return []

        coords = []
        for point in shape_str.split():
            try:
                parts = point.split(',')
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    coords.append((x, y))
            except (ValueError, IndexError):
                continue

        return coords

    def _get_lanes_from_edges(self, edges: List[str]) -> List[str]:
        """Get all lane IDs from a list of edges."""
        lanes = []
        for edge_id in edges:
            edge = self.sumo_net.getEdge(edge_id) if self.sumo_net else None
            if edge:
                for lane in edge.getLanes():
                    lanes.append(lane.getID())
            else:
                # Fallback: assume standard lane naming convention
                # SUMO typically names lanes as edge_id_0, edge_id_1, etc.
                lanes.append(f"{edge_id}_0")

        return lanes

    def _get_intersections_from_edges(self, edges: List[str]) -> List[str]:
        """Get intersection/junction IDs at edge endpoints."""
        intersections = set()

        for edge_id in edges:
            edge = self.sumo_net.getEdge(edge_id) if self.sumo_net else None
            if edge:
                # Get from and to nodes
                from_node = edge.getFromNode()
                to_node = edge.getToNode()

                if from_node:
                    node_id = from_node.getID()
                    # Only include traffic light controlled junctions
                    if from_node.getType() == 'traffic_light':
                        intersections.add(node_id)

                if to_node:
                    node_id = to_node.getID()
                    if to_node.getType() == 'traffic_light':
                        intersections.add(node_id)

        return list(intersections)

    def _get_highway_segments(self, highway_edges: List[str]) -> List[str]:
        """
        Group highway edges into logical segments.

        For simplicity, each highway edge is treated as its own segment.
        More sophisticated grouping can be added later.
        """
        # For now, return edge IDs as segment IDs
        return [f"seg_{edge}" for edge in highway_edges]

    def _calculate_highway_length(self, highway_edges: List[str]) -> float:
        """Calculate total length of highway edges in meters."""
        total_length = 0.0

        for edge_id in highway_edges:
            edge = self.sumo_net.getEdge(edge_id) if self.sumo_net else None
            if edge:
                total_length += edge.getLength()

        return total_length

    def _find_connecting_edges(self,
                               zone1_edges: Set[str],
                               zone2_edges: Set[str],
                               shared_junctions: Set[str]) -> List[str]:
        """Find edges that connect two zones through shared junctions."""
        connecting = []

        if not self.sumo_net:
            return connecting

        for junction_id in shared_junctions:
            junction = self.sumo_net.getNode(junction_id)
            if not junction:
                continue

            # Get edges connected to this junction
            for edge in junction.getIncoming():
                edge_id = edge.getID()
                if edge_id in zone1_edges or edge_id in zone2_edges:
                    connecting.append(edge_id)

            for edge in junction.getOutgoing():
                edge_id = edge.getID()
                if edge_id in zone1_edges or edge_id in zone2_edges:
                    connecting.append(edge_id)

        return list(set(connecting))

    # ==================== ZONE LAYER ENHANCEMENTS (Layer 2) ====================
    # Methods for building zone subgraphs and boundary analysis
    # ==========================================================================

    def build_zone_subgraphs(self,
                              zone_dict: Dict,
                              network_graphs: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Build subgraphs for each zone from the Foundation Layer graphs.

        Creates deep copies of subgraphs for each zone so modifications
        don't affect the underlying Foundation Layer graphs.

        Args:
            zone_dict: Zone dictionary built by build_zone_dict()
            network_graphs: Foundation Layer network_graphs containing:
                - lane_graph: Complete lane connectivity graph
                - road_graph: Complete road connectivity graph
                - transit_graph: Complete transit network graph

        Returns:
            Dict mapping zone_id to subgraphs dict:
            {
                zone_id: {
                    "lane_subgraph": nx.DiGraph,
                    "road_subgraph": nx.DiGraph,
                    "transit_subgraph": nx.DiGraph
                }
            }

        Also updates zone_dict in place with 'subgraphs' key for each zone.
        """
        if network_graphs is None:
            print("Warning: network_graphs not provided. Cannot build zone subgraphs.")
            return {}

        lane_graph = network_graphs.get("lane_graph")
        road_graph = network_graphs.get("road_graph")
        transit_graph = network_graphs.get("transit_graph")

        zone_subgraphs = {}

        for zone_id, zone_info in zone_dict.items():
            subgraphs = {}

            # Build lane_subgraph
            if lane_graph is not None:
                zone_lanes = set(zone_info.get('lanes', []))
                # Filter to only include lanes that exist in the graph
                valid_lanes = zone_lanes & set(lane_graph.nodes())
                if valid_lanes:
                    # Create deep copy of subgraph
                    subgraphs["lane_subgraph"] = lane_graph.subgraph(valid_lanes).copy()
                else:
                    subgraphs["lane_subgraph"] = nx.DiGraph()
            else:
                subgraphs["lane_subgraph"] = None

            # Build road_subgraph
            if road_graph is not None:
                zone_edges = set(zone_info.get('edges', []))
                # Filter to only include roads that exist in the graph
                valid_edges = zone_edges & set(road_graph.nodes())
                if valid_edges:
                    # Create deep copy of subgraph
                    subgraphs["road_subgraph"] = road_graph.subgraph(valid_edges).copy()
                else:
                    subgraphs["road_subgraph"] = nx.DiGraph()
            else:
                subgraphs["road_subgraph"] = None

            # Build transit_subgraph (includes stations in the zone and their routes)
            if transit_graph is not None:
                zone_stations = set(zone_info.get('subway_stations', []) +
                                   zone_info.get('bus_stops', []))

                # Also include routes that serve these stations
                zone_transit_nodes = set(zone_stations)
                for node_id in transit_graph.nodes():
                    node_data = transit_graph.nodes[node_id]
                    if node_data.get('node_type') == 'route':
                        # Check if this route serves any station in the zone
                        for _, station_id, edge_data in transit_graph.out_edges(node_id, data=True):
                            if edge_data.get('edge_type') == 'serves' and station_id in zone_stations:
                                zone_transit_nodes.add(node_id)
                                break

                if zone_transit_nodes:
                    # Create deep copy of subgraph
                    valid_nodes = zone_transit_nodes & set(transit_graph.nodes())
                    subgraphs["transit_subgraph"] = transit_graph.subgraph(valid_nodes).copy()
                else:
                    subgraphs["transit_subgraph"] = nx.DiGraph()
            else:
                subgraphs["transit_subgraph"] = None

            zone_subgraphs[zone_id] = subgraphs

            # Update zone_dict in place
            zone_dict[zone_id]["subgraphs"] = subgraphs

        return zone_subgraphs

    def analyze_zone_boundaries(self,
                                 zone_dict: Dict,
                                 network_graphs: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Analyze zone boundary information including entry/exit roads and boundary intersections.

        Args:
            zone_dict: Zone dictionary built by build_zone_dict()
            network_graphs: Foundation Layer network_graphs containing road_graph

        Returns:
            Dict mapping zone_id to boundary info:
            {
                zone_id: {
                    "entry_roads": [road_id, ...],      # Roads entering the zone
                    "exit_roads": [road_id, ...],       # Roads exiting the zone
                    "boundary_intersections": [...]      # Intersections at zone boundary
                }
            }

        Also updates zone_dict in place with 'boundary' key for each zone.
        """
        zone_boundaries = {}

        for zone_id, zone_info in zone_dict.items():
            zone_edges = set(zone_info.get('edges', []))
            zone_lanes = set(zone_info.get('lanes', []))

            entry_roads = []
            exit_roads = []
            boundary_intersections = set()

            # For each edge in the zone, check its from/to nodes
            for edge_id in zone_edges:
                edge = self.sumo_net.getEdge(edge_id) if self.sumo_net else None
                if not edge:
                    continue

                from_node = edge.getFromNode()
                to_node = edge.getToNode()

                # Check incoming edges to see if they're from outside the zone
                if from_node:
                    for incoming_edge in from_node.getIncoming():
                        incoming_id = incoming_edge.getID()
                        if incoming_id not in zone_edges and not incoming_id.startswith(':'):
                            # This edge comes from outside the zone
                            if edge_id not in entry_roads:
                                entry_roads.append(edge_id)
                            boundary_intersections.add(from_node.getID())
                            break

                # Check outgoing edges to see if they go outside the zone
                if to_node:
                    for outgoing_edge in to_node.getOutgoing():
                        outgoing_id = outgoing_edge.getID()
                        if outgoing_id not in zone_edges and not outgoing_id.startswith(':'):
                            # This edge leads to outside the zone
                            if edge_id not in exit_roads:
                                exit_roads.append(edge_id)
                            boundary_intersections.add(to_node.getID())
                            break

            boundary_info = {
                "entry_roads": entry_roads,
                "exit_roads": exit_roads,
                "boundary_intersections": list(boundary_intersections)
            }

            zone_boundaries[zone_id] = boundary_info

            # Update zone_dict in place
            zone_dict[zone_id]["boundary"] = boundary_info

        return zone_boundaries

    def build_enhanced_zone_graph(self,
                                   zone_dict: Dict,
                                   network_graphs: Optional[Dict] = None) -> nx.DiGraph:
        """
        Build enhanced zone adjacency graph with boundary road information.

        This is an enhanced version of build_zone_graph() that includes
        boundary road capacity information on edges.

        Args:
            zone_dict: Zone dictionary with boundary information
            network_graphs: Foundation Layer network_graphs (optional, for capacity calculation)

        Returns:
            NetworkX DiGraph with enhanced edge attributes:
            - edge_type: 'adjacent'
            - shared_junctions: List[str]
            - shared_edges: List[str]
            - boundary_roads: List[str]  # Roads that cross the boundary
            - boundary_capacity: float    # Total capacity of boundary roads (veh/hr estimate)
        """
        # First build the basic graph
        graph = self.build_zone_graph(zone_dict)

        road_graph = network_graphs.get("road_graph") if network_graphs else None

        # Enhance edges with boundary road information
        for zone1_id, zone2_id, edge_data in list(graph.edges(data=True)):
            zone1_info = zone_dict.get(zone1_id, {})
            zone2_info = zone_dict.get(zone2_id, {})

            zone1_boundary = zone1_info.get('boundary', {})
            zone2_boundary = zone2_info.get('boundary', {})

            # Find boundary roads (roads that exit zone1 to zone2 or vice versa)
            boundary_roads = []

            # Check zone1's exit roads that lead to zone2
            zone2_edges = set(zone2_info.get('edges', []))
            for exit_road in zone1_boundary.get('exit_roads', []):
                # Check if this exit road connects to zone2
                edge = self.sumo_net.getEdge(exit_road) if self.sumo_net else None
                if edge:
                    to_node = edge.getToNode()
                    if to_node:
                        for outgoing in to_node.getOutgoing():
                            if outgoing.getID() in zone2_edges:
                                if exit_road not in boundary_roads:
                                    boundary_roads.append(exit_road)
                                break

            # Calculate boundary capacity (simplified: num_lanes * 1800 veh/hr/lane)
            boundary_capacity = 0.0
            for road_id in boundary_roads:
                edge = self.sumo_net.getEdge(road_id) if self.sumo_net else None
                if edge:
                    num_lanes = len(edge.getLanes())
                    boundary_capacity += num_lanes * 1800  # Standard capacity estimate

            # Update edge attributes
            graph.edges[zone1_id, zone2_id]['boundary_roads'] = boundary_roads
            graph.edges[zone1_id, zone2_id]['boundary_capacity'] = boundary_capacity

        return graph


def get_zone_infrastructure(zone_dict: Dict, zone_id: str, infra_type: str = 'all') -> Dict:
    """
    Query infrastructure for a specific zone.

    Args:
        zone_dict: Zone dictionary
        zone_id: Zone ID to query
        infra_type: Type of infrastructure ('all', 'lanes', 'intersections',
                    'highways', 'ramps', 'transit')

    Returns:
        Dict containing requested infrastructure information
    """
    if zone_id not in zone_dict:
        return {"error": f"Zone {zone_id} not found"}

    zone_info = zone_dict[zone_id]

    if infra_type == 'all':
        return zone_info
    elif infra_type == 'lanes':
        return {"zone_id": zone_id, "lanes": zone_info['lanes']}
    elif infra_type == 'intersections':
        return {"zone_id": zone_id, "intersections": zone_info['intersections']}
    elif infra_type == 'highways':
        return {
            "zone_id": zone_id,
            "highway_segments": zone_info['highway_segments'],
            "highway_edges": zone_info['highway_edges']
        }
    elif infra_type == 'ramps':
        return {
            "zone_id": zone_id,
            "ramps": zone_info['ramps'],
            "ramp_lanes": zone_info['ramp_lanes']
        }
    elif infra_type == 'transit':
        return {
            "zone_id": zone_id,
            "subway_stations": zone_info['subway_stations'],
            "bus_stops": zone_info['bus_stops']
        }
    else:
        return {"error": f"Unknown infrastructure type: {infra_type}"}


def get_zones_by_infrastructure(zone_dict: Dict, infra_id: str, infra_type: str) -> List[str]:
    """
    Find zones containing a specific infrastructure element.

    Args:
        zone_dict: Zone dictionary
        infra_id: Infrastructure element ID
        infra_type: Type of infrastructure ('edge', 'lane', 'intersection',
                    'highway', 'ramp', 'subway_station', 'bus_stop')

    Returns:
        List of zone IDs containing the infrastructure
    """
    matching_zones = []

    type_to_key = {
        'edge': 'edges',
        'lane': 'lanes',
        'intersection': 'intersections',
        'highway': 'highway_edges',
        'ramp': 'ramps',
        'subway_station': 'subway_stations',
        'bus_stop': 'bus_stops'
    }

    key = type_to_key.get(infra_type)
    if not key:
        return []

    for zone_id, zone_info in zone_dict.items():
        if infra_id in zone_info.get(key, []):
            matching_zones.append(zone_id)

    return matching_zones
