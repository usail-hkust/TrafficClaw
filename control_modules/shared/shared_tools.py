"""
Shared tools for cross-module traffic analysis.

Provides common analysis functions that can be used across all control modules.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


def analyze_zone_traffic(
    zone_dict: Dict[str, Any],
    lane_states: Dict[str, Dict[str, Any]],
    zone_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze traffic at the zone/TAZ level.

    Args:
        zone_dict: Dictionary mapping zone_id to zone info (lanes, intersections)
        lane_states: Dictionary mapping lane_id to lane traffic state
        zone_ids: Optional list of specific zones to analyze (None = all zones)

    Returns:
        Dictionary mapping zone_id to aggregated traffic metrics:
        {
            "zone_id": {
                "total_vehicles": int,
                "avg_queue_length": float,
                "avg_speed": float,
                "avg_waiting_time": float,
                "congestion_level": str,  # "free_flow", "moderate", "congested", "severe"
                "intersection_count": int,
                "lane_count": int,
            }
        }
    """
    results = {}

    zones_to_analyze = zone_ids if zone_ids else list(zone_dict.keys())

    for zone_id in zones_to_analyze:
        if zone_id not in zone_dict:
            continue

        zone_info = zone_dict[zone_id]
        zone_lanes = zone_info.get("lanes", [])
        zone_intersections = zone_info.get("intersections", [])

        # Aggregate metrics
        total_vehicles = 0
        total_queue_length = 0
        total_speed = 0
        total_waiting_time = 0
        lane_count_with_data = 0

        for lane_id in zone_lanes:
            if lane_id not in lane_states:
                continue

            state = lane_states[lane_id]
            lane_count_with_data += 1

            total_vehicles += state.get("vehicle_count", 0)
            total_queue_length += state.get("queue_length", 0)
            total_speed += state.get("average_speed", 0)
            total_waiting_time += state.get("average_waiting_time", 0)

        # Calculate averages
        if lane_count_with_data > 0:
            avg_queue_length = total_queue_length / lane_count_with_data
            avg_speed = total_speed / lane_count_with_data
            avg_waiting_time = total_waiting_time / lane_count_with_data
        else:
            avg_queue_length = 0.0
            avg_speed = 0.0
            avg_waiting_time = 0.0

        # Determine congestion level
        if avg_queue_length < 3 and avg_speed > 10:
            congestion_level = "free_flow"
        elif avg_queue_length < 8 and avg_speed > 5:
            congestion_level = "moderate"
        elif avg_queue_length < 15 and avg_speed > 2:
            congestion_level = "congested"
        else:
            congestion_level = "severe"

        results[zone_id] = {
            "total_vehicles": total_vehicles,
            "avg_queue_length": avg_queue_length,
            "avg_speed": avg_speed,
            "avg_waiting_time": avg_waiting_time,
            "congestion_level": congestion_level,
            "intersection_count": len(zone_intersections),
            "lane_count": len(zone_lanes),
            "lane_count_with_data": lane_count_with_data,
        }

    return results


def identify_congestion_hotspots(
    lane_states: Dict[str, Dict[str, Any]],
    queue_threshold: int = 10,
    speed_threshold: float = 3.0,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Identify congestion hotspots in the network.

    Args:
        lane_states: Dictionary mapping lane_id to lane traffic state
        queue_threshold: Queue length threshold for congestion
        speed_threshold: Speed threshold (m/s) below which is congested
        top_n: Number of top hotspots to return

    Returns:
        List of hotspot dictionaries sorted by severity:
        [
            {
                "lane_id": str,
                "intersection": str,
                "queue_length": int,
                "average_speed": float,
                "waiting_time": float,
                "severity_score": float,
            }
        ]
    """
    hotspots = []

    for lane_id, state in lane_states.items():
        queue_length = state.get("queue_length", 0)
        avg_speed = state.get("average_speed", 0)
        waiting_time = state.get("average_waiting_time", 0)

        # Check if this qualifies as a hotspot
        is_hotspot = (queue_length >= queue_threshold or
                      (avg_speed <= speed_threshold and queue_length > 0))

        if is_hotspot:
            # Calculate severity score (higher = worse)
            severity = (
                queue_length * 2.0 +
                max(0, 10 - avg_speed) * 1.5 +
                min(waiting_time / 60, 5) * 1.0
            )

            hotspots.append({
                "lane_id": lane_id,
                "intersection": state.get("end_intersection", "unknown"),
                "queue_length": queue_length,
                "average_speed": avg_speed,
                "waiting_time": waiting_time,
                "severity_score": severity,
                "direction": state.get("direction", "unknown"),
                "loc_dir": state.get("loc_dir", "unknown"),
            })

    # Sort by severity and return top N
    hotspots.sort(key=lambda x: x["severity_score"], reverse=True)
    return hotspots[:top_n]


def calculate_network_metrics(
    lane_states: Dict[str, Dict[str, Any]],
    module_metrics: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Calculate network-wide traffic metrics.

    Args:
        lane_states: Dictionary mapping lane_id to lane traffic state
        module_metrics: Optional per-module metrics for additional context

    Returns:
        Dictionary of network-wide metrics
    """
    if not lane_states:
        return {
            "total_vehicles": 0,
            "avg_speed": 0.0,
            "avg_queue_length": 0.0,
            "avg_waiting_time": 0.0,
            "network_throughput_potential": 0.0,
            "congestion_index": 0.0,
        }

    total_vehicles = 0
    total_speed = 0.0
    total_queue = 0.0
    total_waiting = 0.0
    total_throughput = 0.0
    lane_count = len(lane_states)

    for state in lane_states.values():
        total_vehicles += state.get("vehicle_count", 0)
        total_speed += state.get("average_speed", 0)
        total_queue += state.get("queue_length", 0)
        total_waiting += state.get("average_waiting_time", 0)
        total_throughput += state.get("throughput_potential", 0)

    avg_speed = total_speed / lane_count if lane_count > 0 else 0.0
    avg_queue = total_queue / lane_count if lane_count > 0 else 0.0
    avg_waiting = total_waiting / lane_count if lane_count > 0 else 0.0
    network_throughput = total_throughput / lane_count if lane_count > 0 else 0.0

    # Calculate congestion index (0-1, higher = more congested)
    # Based on queue length and speed relative to expected free-flow
    expected_speed = 15.0  # m/s
    speed_ratio = min(1.0, avg_speed / expected_speed) if expected_speed > 0 else 0.0
    queue_factor = min(1.0, avg_queue / 20.0)  # Normalize by max expected queue

    congestion_index = 0.5 * (1 - speed_ratio) + 0.5 * queue_factor

    result = {
        "total_vehicles": total_vehicles,
        "avg_speed": avg_speed,
        "avg_queue_length": avg_queue,
        "avg_waiting_time": avg_waiting,
        "network_throughput_potential": network_throughput,
        "congestion_index": congestion_index,
        "lane_count": lane_count,
    }

    # Add module-specific summary if available
    if module_metrics:
        for module_name, metrics in module_metrics.items():
            if "avg_travel_time" in metrics:
                result[f"{module_name}_travel_time"] = metrics["avg_travel_time"]
            if "avg_queue_len" in metrics:
                result[f"{module_name}_queue_len"] = metrics["avg_queue_len"]

    return result


def aggregate_by_intersection(
    lane_states: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate lane metrics by intersection.

    Args:
        lane_states: Dictionary mapping lane_id to lane traffic state

    Returns:
        Dictionary mapping intersection_id to aggregated metrics
    """
    intersection_data = defaultdict(lambda: {
        "lanes": [],
        "total_vehicles": 0,
        "total_queue": 0,
        "total_speed": 0,
        "total_waiting": 0,
        "direction_metrics": defaultdict(lambda: {"queue": 0, "vehicles": 0, "speed": 0, "count": 0})
    })

    for lane_id, state in lane_states.items():
        end_inter = state.get("end_intersection")
        if not end_inter:
            continue

        data = intersection_data[end_inter]
        data["lanes"].append(lane_id)
        data["total_vehicles"] += state.get("vehicle_count", 0)
        data["total_queue"] += state.get("queue_length", 0)
        data["total_speed"] += state.get("average_speed", 0)
        data["total_waiting"] += state.get("average_waiting_time", 0)

        # Track by direction
        loc_dir = state.get("loc_dir", "unknown")
        dir_data = data["direction_metrics"][loc_dir]
        dir_data["queue"] += state.get("queue_length", 0)
        dir_data["vehicles"] += state.get("vehicle_count", 0)
        dir_data["speed"] += state.get("average_speed", 0)
        dir_data["count"] += 1

    # Calculate averages
    results = {}
    for inter_id, data in intersection_data.items():
        lane_count = len(data["lanes"])
        if lane_count == 0:
            continue

        # Find busiest direction
        busiest_dir = None
        max_queue = 0
        direction_summary = {}

        for dir_name, dir_data in data["direction_metrics"].items():
            if dir_data["count"] > 0:
                dir_queue = dir_data["queue"] / dir_data["count"]
                direction_summary[dir_name] = {
                    "avg_queue": dir_queue,
                    "avg_speed": dir_data["speed"] / dir_data["count"],
                    "total_vehicles": dir_data["vehicles"],
                }
                if dir_data["queue"] > max_queue:
                    max_queue = dir_data["queue"]
                    busiest_dir = dir_name

        results[inter_id] = {
            "lane_count": lane_count,
            "total_vehicles": data["total_vehicles"],
            "avg_queue_length": data["total_queue"] / lane_count,
            "avg_speed": data["total_speed"] / lane_count,
            "avg_waiting_time": data["total_waiting"] / lane_count,
            "busiest_direction": busiest_dir,
            "direction_summary": dict(direction_summary),
        }

    return results


def get_directional_demand(
    lane_states: Dict[str, Dict[str, Any]],
    intersection_id: str
) -> Dict[str, Dict[str, float]]:
    """
    Get demand by direction for a specific intersection.

    Args:
        lane_states: Dictionary mapping lane_id to lane traffic state
        intersection_id: ID of the intersection to analyze

    Returns:
        Dictionary mapping direction (loc_dir) to demand metrics
    """
    demand = defaultdict(lambda: {"queue": 0, "vehicles": 0, "arrival_rate": 0, "count": 0})

    for lane_id, state in lane_states.items():
        if state.get("end_intersection") != intersection_id:
            continue

        loc_dir = state.get("loc_dir", "unknown")
        demand[loc_dir]["queue"] += state.get("queue_length", 0)
        demand[loc_dir]["vehicles"] += state.get("vehicle_count", 0)
        demand[loc_dir]["arrival_rate"] += state.get("arrival_rate", 0)
        demand[loc_dir]["count"] += 1

    # Calculate averages
    results = {}
    for dir_name, data in demand.items():
        if data["count"] > 0:
            results[dir_name] = {
                "avg_queue": data["queue"] / data["count"],
                "total_vehicles": data["vehicles"],
                "avg_arrival_rate": data["arrival_rate"] / data["count"],
                "lane_count": data["count"],
            }

    return results
