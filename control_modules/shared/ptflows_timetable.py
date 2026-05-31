"""
Build bus/subway timetable config from real ptflows.rou.xml.

ptflows.rou.xml contains <flow> elements with period (headway in seconds),
begin, end, route. We convert these to the timetable format expected by
control_modules/bus_scheduling and control_modules/subway_scheduling.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

# Default dwell times when building schedule from env (match control_modules defaults)
SUBWAY_DEFAULT_DWELL = 30
BUS_DEFAULT_DWELL = 20


def parse_ptflows_flows(ptflows_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse ptflows.rou.xml and extract flow definitions by type (subway, bus).

    Returns:
        {"subway": {route_id: {"begin": float, "end": float, "period": float}},
         "bus": {route_id: {...}}}
    """
    result = {"subway": {}, "bus": {}}
    if not ptflows_path.exists():
        return result
    try:
        tree = ET.parse(ptflows_path)
        root = tree.getroot()
    except Exception:
        return result

    for flow in root.findall(".//flow"):
        flow_type = (flow.get("type") or "").strip().lower()
        if flow_type not in ("subway", "bus"):
            continue
        route_id = flow.get("route")
        if not route_id:
            continue
        begin_s = flow.get("begin")
        end_s = flow.get("end")
        period_s = flow.get("period")
        if not period_s:
            continue
        try:
            begin = float(begin_s) if begin_s else 0.0
            end = float(end_s) if end_s else 86400.0
            period = float(period_s)
            if period <= 0:
                continue
        except (TypeError, ValueError):
            continue
        result[flow_type][route_id] = {"begin": begin, "end": end, "period": period}

    return result


def get_ptflows_path_from_config_path(config_path: str) -> Path:
    """Return ptflows.rou.xml path: same directory as sumocfg (sumo_config/REGION/ptflows.rou.xml)."""
    p = Path(config_path).resolve()
    return p.parent / "ptflows.rou.xml"


def _schedule_from_subway_line(line: Any, default_dwell: int = SUBWAY_DEFAULT_DWELL) -> list:
    """Build schedule list from env subway line (SubwayStation objects)."""
    schedule = []
    for station in (line.stations or []):
        if hasattr(station, "station_id"):
            station_id = station.station_id
        else:
            station_id = str(station)
        schedule.append({"station_id": station_id, "dwell_time": default_dwell})
    return schedule


def _schedule_from_bus_line(line: Any, default_dwell: int = BUS_DEFAULT_DWELL) -> list:
    """Build schedule list from env bus line (list of station_id strings)."""
    schedule = []
    for station_id in (line.stations or []):
        sid = getattr(station_id, "station_id", station_id)
        schedule.append({"station_id": str(sid), "dwell_time": default_dwell})
    return schedule


def build_subway_timetable_config(
    env: Any,
    ptflows_path: Path,
    default_dwell: int = SUBWAY_DEFAULT_DWELL,
) -> Dict[str, Any]:
    """
    Build subway scheduling config from ptflows.rou.xml (real timetable).

    Uses period as headway, one segment [0, 3600] per route. Schedule (stations)
    comes from env.subway_lines.
    """
    flows = parse_ptflows_flows(ptflows_path)
    route_flows = flows.get("subway", {})
    config = {}
    for route_id, line in getattr(env, "subway_lines", {}).items():
        if not getattr(line, "stations", None):
            continue
        schedule = _schedule_from_subway_line(line, default_dwell)
        if not schedule:
            continue
        # Use real headway from ptflows if available, else skip (no flow for this route)
        if route_id not in route_flows:
            continue
        period = route_flows[route_id]["period"]
        config[route_id] = {
            "timetable": [
                {
                    "time_range": [0, 3600],
                    "headway": int(period),
                    "schedule": schedule,
                }
            ]
        }
    return config


def build_bus_timetable_config(
    env: Any,
    ptflows_path: Path,
    default_dwell: int = BUS_DEFAULT_DWELL,
) -> Dict[str, Any]:
    """
    Build bus scheduling config from ptflows.rou.xml (real timetable).

    Uses period as headway, one segment [0, 3600] per route. Schedule (stations)
    comes from env.bus_lines.
    """
    flows = parse_ptflows_flows(ptflows_path)
    route_flows = flows.get("bus", {})
    config = {}
    for route_id, line in getattr(env, "bus_lines", {}).items():
        if not getattr(line, "stations", None):
            continue
        schedule = _schedule_from_bus_line(line, default_dwell)
        if not schedule:
            continue
        if route_id not in route_flows:
            continue
        period = route_flows[route_id]["period"]
        config[route_id] = {
            "timetable": [
                {
                    "time_range": [0, 3600],
                    "headway": int(period),
                    "schedule": schedule,
                }
            ]
        }
    return config
