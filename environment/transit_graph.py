"""
Transit Graph Builder for bus and subway network organization.

This module provides functionality to:
- Build transit network graphs connecting routes and stations
- Build detailed bus route information with static and real-time data
- Update real-time bus operation data during simulation
"""

from typing import Dict, List, Optional, Set, Any
import networkx as nx


class TransitGraphBuilder:
    """
    Builds and manages transit network graphs and bus route information.

    Creates a graph representation of the transit network where:
    - Nodes represent routes and stations
    - Edges represent service relationships and stop sequences
    """

    def __init__(self,
                 bus_lines: Optional[Dict] = None,
                 subway_lines: Optional[Dict] = None,
                 bus_stations: Optional[Dict] = None,
                 subway_stations: Optional[Dict] = None,
                 zone_dict: Optional[Dict] = None,
                 traci_conn=None,
                 lane_to_zone: Optional[Dict[str, str]] = None):
        """
        Initialize TransitGraphBuilder.

        Args:
            bus_lines: Dict of BusLine objects keyed by route_id
            subway_lines: Dict of SubwayLine objects keyed by line_id
            bus_stations: Dict of BusStation objects keyed by station_id
            subway_stations: Dict of SubwayStation objects keyed by station_id
            zone_dict: Zone dictionary for station-to-zone mapping
            traci_conn: TraCI connection for real-time data
            lane_to_zone: Pre-built lane_id -> zone_id mapping (optional)
        """
        self.bus_lines = bus_lines or {}
        self.subway_lines = subway_lines or {}
        self.bus_stations = bus_stations or {}
        self.subway_stations = subway_stations or {}
        self.zone_dict = zone_dict or {}
        self.traci_conn = traci_conn
        self.lane_to_zone = lane_to_zone or {}

        # Build station to zone mapping if not provided
        if not self.lane_to_zone and self.zone_dict:
            self._build_lane_to_zone_from_zone_dict()

    def _build_lane_to_zone_from_zone_dict(self):
        """Build lane_to_zone mapping from zone_dict."""
        for zone_id, zone_info in self.zone_dict.items():
            for lane_id in zone_info.get('lanes', []):
                self.lane_to_zone[lane_id] = zone_id

    def _get_station_zone(self, station) -> str:
        """Get zone ID for a station based on its lane."""
        lane_id = getattr(station, 'lane_id', None)
        if lane_id and lane_id in self.lane_to_zone:
            return self.lane_to_zone[lane_id]
        return 'unknown'

    def build_transit_graph(self) -> nx.DiGraph:
        """
        Build transit network graph.

        Node types:
        - route: Bus or subway route (node_type='route', route_type='bus'|'subway')
        - station: Bus stop or subway station (node_type='station', station_type='bus_stop'|'subway_station')

        Edge types:
        - serves: Route -> Station (edge_type='serves', stop_index)
        - next_stop: Station -> Station (edge_type='next_stop', route_id)

        Returns:
            NetworkX DiGraph representing the transit network
        """
        graph = nx.DiGraph()

        # Add bus route nodes and their stations
        for route_id, bus_line in self.bus_lines.items():
            # Add route node
            graph.add_node(
                route_id,
                node_type='route',
                route_type='bus',
                direction=getattr(bus_line, 'direction', 'unknown'),
                station_count=len(getattr(bus_line, 'stations', []))
            )

            # Get station list from bus line
            stations = getattr(bus_line, 'stations', [])

            # Add station nodes and edges
            prev_station_id = None
            for idx, station in enumerate(stations):
                station_id = getattr(station, 'station_id', str(station))

                # Add station node if not exists
                if not graph.has_node(station_id):
                    zone_id = 'unknown'
                    if station_id in self.bus_stations:
                        zone_id = self._get_station_zone(self.bus_stations[station_id])
                    elif hasattr(station, 'lane_id'):
                        zone_id = self._get_station_zone(station)

                    graph.add_node(
                        station_id,
                        node_type='station',
                        station_type='bus_stop',
                        zone_id=zone_id
                    )

                # Add route -> station edge
                graph.add_edge(
                    route_id, station_id,
                    edge_type='serves',
                    stop_index=idx
                )

                # Add station -> next_station edge
                if prev_station_id:
                    graph.add_edge(
                        prev_station_id, station_id,
                        edge_type='next_stop',
                        route_id=route_id,
                        stop_index=idx
                    )

                prev_station_id = station_id

        # Add subway line nodes and their stations
        for line_id, subway_line in self.subway_lines.items():
            # Add route node
            graph.add_node(
                line_id,
                node_type='route',
                route_type='subway',
                direction=getattr(subway_line, 'direction', 'unknown'),
                station_count=len(getattr(subway_line, 'stations', []))
            )

            # Get station list from subway line
            stations = getattr(subway_line, 'stations', [])

            # Add station nodes and edges
            prev_station_id = None
            for idx, station in enumerate(stations):
                station_id = getattr(station, 'station_id', str(station))

                # Add station node if not exists
                if not graph.has_node(station_id):
                    zone_id = 'unknown'
                    if station_id in self.subway_stations:
                        zone_id = self._get_station_zone(self.subway_stations[station_id])
                    elif hasattr(station, 'lane_id'):
                        zone_id = self._get_station_zone(station)

                    graph.add_node(
                        station_id,
                        node_type='station',
                        station_type='subway_station',
                        zone_id=zone_id
                    )

                # Add route -> station edge
                graph.add_edge(
                    line_id, station_id,
                    edge_type='serves',
                    stop_index=idx
                )

                # Add station -> next_station edge
                if prev_station_id:
                    graph.add_edge(
                        prev_station_id, station_id,
                        edge_type='next_stop',
                        route_id=line_id,
                        stop_index=idx
                    )

                prev_station_id = station_id

        return graph

    def build_bus_route_info(self, sumo_net=None) -> Dict[str, Dict]:
        """
        Build detailed bus route information with static data.

        Args:
            sumo_net: SUMO network object for edge length calculations

        Returns:
            Dict mapping route_id to detailed route information
        """
        bus_route_info = {}

        for route_id, bus_line in self.bus_lines.items():
            # Extract basic route information
            edges = getattr(bus_line, 'edges', [])
            stations = getattr(bus_line, 'stations', [])
            direction = getattr(bus_line, 'direction', 'unknown')

            # Calculate total route length
            total_length = 0.0
            if sumo_net and edges:
                for edge_id in edges:
                    edge = sumo_net.getEdge(edge_id)
                    if edge:
                        total_length += edge.getLength()

            # Build station info list
            station_info_list = []
            cumulative_length = 0.0

            for idx, station in enumerate(stations):
                station_id = getattr(station, 'station_id', str(station))
                lane_id = getattr(station, 'lane_id', '')
                zone_id = 'unknown'

                if station_id in self.bus_stations:
                    zone_id = self._get_station_zone(self.bus_stations[station_id])
                elif hasattr(station, 'lane_id'):
                    zone_id = self._get_station_zone(station)

                # Get station position along route
                start_pos = getattr(station, 'start_pos', 0.0)

                station_info_list.append({
                    "station_id": station_id,
                    "zone_id": zone_id,
                    "position_m": cumulative_length + start_pos,
                    "stop_index": idx
                })

                cumulative_length += start_pos

            # Build segment info
            segments = []
            scheduled_runtime = 0.0
            avg_speed = 10.0  # Default 10 m/s (36 km/h)

            for i in range(len(station_info_list) - 1):
                from_station = station_info_list[i]
                to_station = station_info_list[i + 1]

                segment_length = to_station['position_m'] - from_station['position_m']
                if segment_length < 0:
                    segment_length = abs(segment_length)

                travel_time = segment_length / avg_speed if avg_speed > 0 else 0
                scheduled_runtime += travel_time

                # Count intersections in segment (simplified)
                intersections_count = 0

                segments.append({
                    "from_station": from_station['station_id'],
                    "to_station": to_station['station_id'],
                    "length_m": segment_length,
                    "scheduled_travel_time_s": travel_time,
                    "intersections_count": intersections_count
                })

            # Identify zones served
            zones_served = list(set(
                s['zone_id'] for s in station_info_list
                if s['zone_id'] != 'unknown'
            ))

            # Find transfer stations (stations shared with other routes)
            transfer_stations = self._find_transfer_stations(route_id, station_info_list)

            # Get scheduled headway from bus line
            scheduled_headway = getattr(bus_line, 'headway', 180.0)

            bus_route_info[route_id] = {
                # Static information
                "route_id": route_id,
                "route_name": route_id,
                "direction": direction,
                "edges": edges,
                "total_length_m": total_length,
                "scheduled_runtime_s": scheduled_runtime,
                "stations": station_info_list,
                "segments": segments,
                "zones_served": zones_served,
                "transfer_stations": transfer_stations,

                # Real-time data (initialized with defaults)
                "realtime": {
                    "current_headway_s": scheduled_headway,
                    "scheduled_headway_s": scheduled_headway,
                    "active_buses": [],
                    "bus_count": 0,
                    "segment_status": [
                        {
                            "from_station": seg["from_station"],
                            "to_station": seg["to_station"],
                            "current_travel_time_s": seg["scheduled_travel_time_s"],
                            "delay_s": 0.0,
                            "buses_in_segment": []
                        }
                        for seg in segments
                    ],
                    "station_load": [
                        {
                            "station_id": s["station_id"],
                            "waiting_passengers": 0,
                            "boarding_rate": 0.0,
                            "alighting_rate": 0.0
                        }
                        for s in station_info_list
                    ],
                    "last_update_step": 0
                }
            }

        return bus_route_info

    def _find_transfer_stations(self,
                                current_route_id: str,
                                station_info_list: List[Dict]) -> List[Dict]:
        """Find transfer stations that connect to other routes."""
        transfer_stations = []

        # Build a map of station_id -> routes that serve it
        station_to_routes: Dict[str, List[str]] = {}

        for route_id, bus_line in self.bus_lines.items():
            stations = getattr(bus_line, 'stations', [])
            for station in stations:
                station_id = getattr(station, 'station_id', str(station))
                if station_id not in station_to_routes:
                    station_to_routes[station_id] = []
                station_to_routes[station_id].append(route_id)

        # Also include subway lines
        for line_id, subway_line in self.subway_lines.items():
            stations = getattr(subway_line, 'stations', [])
            for station in stations:
                station_id = getattr(station, 'station_id', str(station))
                if station_id not in station_to_routes:
                    station_to_routes[station_id] = []
                station_to_routes[station_id].append(line_id)

        # Check each station in current route
        for station_info in station_info_list:
            station_id = station_info['station_id']
            routes = station_to_routes.get(station_id, [])

            # Find connecting routes (exclude current route)
            connecting_bus_routes = [
                r for r in routes
                if r != current_route_id and r in self.bus_lines
            ]
            connecting_subway_lines = [
                r for r in routes
                if r != current_route_id and r in self.subway_lines
            ]

            if connecting_bus_routes or connecting_subway_lines:
                transfer_stations.append({
                    "station_id": station_id,
                    "connecting_routes": connecting_bus_routes,
                    "connecting_subway_lines": connecting_subway_lines
                })

        return transfer_stations

    def update_realtime_data(self,
                             bus_route_info: Dict[str, Dict],
                             current_step: int = 0) -> None:
        """
        Update real-time bus operation data.

        Called during simulation to update:
        - Active buses on each route
        - Current headway
        - Segment travel times and delays
        - Station load information

        Args:
            bus_route_info: Bus route info dictionary to update
            current_step: Current simulation step
        """
        if not self.traci_conn:
            return

        for route_id, route_info in bus_route_info.items():
            if route_id not in self.bus_lines:
                continue

            bus_line = self.bus_lines[route_id]
            realtime = route_info['realtime']

            # Update active buses
            try:
                # Get bus IDs from bus line
                active_bus_ids = []
                if hasattr(bus_line, 'get_bus_ids'):
                    active_bus_ids = list(bus_line.get_bus_ids())
                elif hasattr(bus_line, 'buses'):
                    active_bus_ids = list(bus_line.buses.keys())

                realtime['active_buses'] = active_bus_ids
                realtime['bus_count'] = len(active_bus_ids)

                # Calculate current headway if we have multiple buses
                if len(active_bus_ids) >= 2:
                    # Get departure times and calculate average headway
                    departure_times = []
                    for bus_id in active_bus_ids:
                        if hasattr(bus_line, 'buses') and bus_id in bus_line.buses:
                            bus = bus_line.buses[bus_id]
                            dep_time = getattr(bus, 'departure_time', None)
                            if dep_time is not None:
                                departure_times.append(dep_time)

                    if len(departure_times) >= 2:
                        departure_times.sort()
                        headways = [
                            departure_times[i+1] - departure_times[i]
                            for i in range(len(departure_times) - 1)
                        ]
                        if headways:
                            realtime['current_headway_s'] = sum(headways) / len(headways)

                # Update segment status with bus positions
                for seg_status in realtime['segment_status']:
                    seg_status['buses_in_segment'] = []

                # Track which buses are in which segments
                for bus_id in active_bus_ids:
                    try:
                        # Get current edge of bus
                        if hasattr(bus_line, 'buses') and bus_id in bus_line.buses:
                            bus = bus_line.buses[bus_id]
                            current_edge = getattr(bus, 'current_edge', None)
                            next_station = getattr(bus, 'next_station_id', None)

                            # Find which segment this bus is in
                            for seg_status in realtime['segment_status']:
                                if seg_status['to_station'] == next_station:
                                    seg_status['buses_in_segment'].append(bus_id)
                                    break
                    except Exception:
                        pass

                # Update station load
                for station_load in realtime['station_load']:
                    station_id = station_load['station_id']
                    if station_id in self.bus_stations:
                        station = self.bus_stations[station_id]
                        if hasattr(station, 'get_waiting_count'):
                            station_load['waiting_passengers'] = station.get_waiting_count()

            except Exception as e:
                # Log error but don't crash
                pass

            realtime['last_update_step'] = current_step


def get_route_segment_info(bus_route_info: Dict,
                           route_id: str,
                           segment_index: Optional[int] = None) -> Dict:
    """
    Query segment information for a bus route.

    Args:
        bus_route_info: Bus route info dictionary
        route_id: Route ID to query
        segment_index: Optional specific segment index (None for all segments)

    Returns:
        Dict containing segment information
    """
    if route_id not in bus_route_info:
        return {"error": f"Route {route_id} not found"}

    route = bus_route_info[route_id]
    segments = route.get('segments', [])
    realtime_segments = route.get('realtime', {}).get('segment_status', [])

    if segment_index is not None:
        if 0 <= segment_index < len(segments):
            static = segments[segment_index]
            realtime = realtime_segments[segment_index] if segment_index < len(realtime_segments) else {}
            return {
                "route_id": route_id,
                "segment_index": segment_index,
                "static": static,
                "realtime": realtime
            }
        else:
            return {"error": f"Segment index {segment_index} out of range (0-{len(segments)-1})"}

    # Return all segments
    combined_segments = []
    for i, seg in enumerate(segments):
        realtime = realtime_segments[i] if i < len(realtime_segments) else {}
        combined_segments.append({
            "segment_index": i,
            "static": seg,
            "realtime": realtime
        })

    return {
        "route_id": route_id,
        "total_segments": len(segments),
        "segments": combined_segments
    }


def get_route_stations(bus_route_info: Dict, route_id: str) -> Dict:
    """
    Query station information for a bus route.

    Args:
        bus_route_info: Bus route info dictionary
        route_id: Route ID to query

    Returns:
        Dict containing station information
    """
    if route_id not in bus_route_info:
        return {"error": f"Route {route_id} not found"}

    route = bus_route_info[route_id]

    return {
        "route_id": route_id,
        "stations": route.get('stations', []),
        "transfer_stations": route.get('transfer_stations', []),
        "zones_served": route.get('zones_served', [])
    }


def get_route_realtime_status(bus_route_info: Dict, route_id: str) -> Dict:
    """
    Query real-time status for a bus route.

    Args:
        bus_route_info: Bus route info dictionary
        route_id: Route ID to query

    Returns:
        Dict containing real-time status
    """
    if route_id not in bus_route_info:
        return {"error": f"Route {route_id} not found"}

    route = bus_route_info[route_id]

    return {
        "route_id": route_id,
        "realtime": route.get('realtime', {})
    }
