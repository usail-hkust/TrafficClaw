#!/usr/bin/env python3
"""
Compute key statistics of SUMO environments for paper tables.

Generates network scale (junctions, edges, lanes, traffic lights, lane-km)
and demand dynamics (trips, vehicles, flows, persons, simulation duration)
from .net.xml and route files. Outputs LaTeX table for inclusion in papers.

Usage:
    # From SUMO config (reads net + route files from sumocfg)
    python tools/statistics_network_and_demand.py --config sumo_config/jinan/jinan.sumocfg
    python tools/statistics_network_and_demand.py --config sumo_config/Upper_Manhattan/Upper_Manhattan.sumocfg

    # Multiple environments
    python tools/statistics_network_and_demand.py \\
        --config sumo_config/jinan/jinan.sumocfg \\
        --config sumo_config/Upper_Manhattan/Upper_Manhattan.sumocfg \\
        --output table.tex

    # From net file only (no demand stats)
    python tools/statistics_network_and_demand.py --net sumo_config/jinan/jinan.net.xml
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_sumocfg(config_path: Path) -> Tuple[Optional[str], List[str], float, float]:
    """Parse SUMO config to get net file, route files, begin/end time."""
    if not config_path.exists():
        return None, [], 0.0, 86400.0
    tree = ET.parse(config_path)
    root = tree.getroot()
    net_file = None
    route_files = []
    begin_time = 0.0
    end_time = 86400.0

    for inp in root.findall("input"):
        net_elem = inp.find("net-file")
        if net_elem is not None:
            net_file = net_elem.get("value")
        route_elem = inp.find("route-files")
        if route_elem is not None:
            raw = route_elem.get("value", "")
            route_files = [f.strip() for f in raw.split(",") if f.strip()]

    for time_elem in root.findall("time"):
        begin_elem = time_elem.find("begin")
        end_elem = time_elem.find("end")
        if begin_elem is not None:
            try:
                begin_time = float(begin_elem.get("value", 0))
            except ValueError:
                pass
        if end_elem is not None:
            try:
                end_time = float(end_elem.get("value", 86400))
            except ValueError:
                pass

    return net_file, route_files, begin_time, end_time


def _compute_network_stats(net_path: Path) -> Dict[str, Any]:
    """Compute network statistics from .net.xml using sumolib."""
    try:
        import sumolib
    except ImportError:
        raise ImportError("sumolib is required. Install with: pip install sumolib")

    if not net_path.exists():
        return {"error": f"Network file not found: {net_path}"}

    net = sumolib.net.readNet(str(net_path))

    # Basic counts
    nodes = list(net.getNodes())
    edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    lanes = []
    total_lane_length = 0.0
    highway_edges = 0
    highway_lanes = 0
    ramp_edges = 0
    ramp_lanes = 0

    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        edge_type = getattr(edge, "getType", lambda: "")() or ""
        for lane in edge.getLanes():
            lane_id = lane.getID()
            length = float(lane.getLength())
            lanes.append(lane_id)
            total_lane_length += length
            if edge_type in ("motorway", "motorway_link", "trunk", "trunk_link"):
                if "link" in edge_type:
                    ramp_edges += 1
                    ramp_lanes += 1
                else:
                    highway_edges += 1
                    highway_lanes += 1

    # Deduplicate highway/ramp edge counts (we're counting per-lane above, fix: count edges once)
    highway_edges = 0
    ramp_edges = 0
    highway_lanes = 0
    ramp_lanes = 0
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        edge_type = getattr(edge, "getType", lambda: "")() or ""
        n_lanes = len(edge.getLanes())
        if edge_type in ("motorway", "trunk"):
            highway_edges += 1
            highway_lanes += n_lanes
        elif edge_type in ("motorway_link", "trunk_link"):
            ramp_edges += 1
            ramp_lanes += n_lanes

    # Traffic lights
    tls_count = len(list(net.getTrafficLights()))

    return {
        "junctions": len(nodes),
        "edges": len(edges),
        "lanes": len(lanes),
        "traffic_lights": tls_count,
        "lane_km": total_lane_length / 1000.0,
        "highway_edges": highway_edges,
        "highway_lanes": highway_lanes,
        "ramp_edges": ramp_edges,
        "ramp_lanes": ramp_lanes,
    }


# Route file to demand category mapping.
# PT = public transportation (bus + subway). Includes persons.pt, ptflows (bus/subway vehicles).
# Categories: background, taxi, pt, walk. Unknown files default to background.
_ROUTE_FILE_CATEGORY: Dict[str, str] = {
    "taxi_fleet.rou.xml": "taxi",
    "persons.taxi.xml": "taxi",
    "routes.background.rou.xml": "background",
    "routes.rou.xml": "background",
    "persons.pt.rou.xml": "pt",
    "persons.walk.rou.xml": "walk",
    "ptflows.rou.xml": "pt",
}


def _parse_route_file(rou_path: Path) -> Dict[str, Any]:
    """Parse route file for demand statistics (trips, vehicles, flows, persons)."""
    stats = {
        "trips": 0,
        "vehicles": 0,
        "flows": 0,
        "flow_vehicles_est": 0,
        "persons": 0,
    }
    if not rou_path.exists():
        return stats

    try:
        tree = ET.parse(rou_path)
        root = tree.getroot()
    except ET.ParseError:
        return stats

    # Trips
    for _ in root.findall(".//trip"):
        stats["trips"] += 1

    # Vehicles (explicit vehicle elements)
    for _ in root.findall(".//vehicle"):
        stats["vehicles"] += 1

    # Flows: estimate vehicles from begin, end, period/vehsPerHour
    for flow in root.findall(".//flow"):
        stats["flows"] += 1
        begin = float(flow.get("begin", 0))
        end = float(flow.get("end", 86400))
        period = flow.get("period")
        vehs_per_hour = flow.get("vehsPerHour")
        number = flow.get("number")
        if number is not None:
            stats["flow_vehicles_est"] += int(number)
        elif period is not None:
            try:
                p = float(period)
                if p > 0:
                    stats["flow_vehicles_est"] += int((end - begin) / p)
            except ValueError:
                pass
        elif vehs_per_hour is not None:
            try:
                vph = float(vehs_per_hour)
                if vph > 0:
                    stats["flow_vehicles_est"] += int((end - begin) / 3600.0 * vph)
            except ValueError:
                pass

    # Persons
    for _ in root.findall(".//person"):
        stats["persons"] += 1

    return stats


def _aggregate_demand(
    config_dir: Path,
    route_files: List[str],
    begin: float,
    end: float,
) -> Dict[str, Any]:
    """Aggregate demand stats by category: background, taxi, PT (bus+subway), walk."""
    categories = ("background", "taxi", "pt", "walk")
    by_cat: Dict[str, Dict[str, int]] = {c: {"trips": 0, "vehicles": 0, "flows": 0, "flow_vehicles_est": 0, "persons": 0} for c in categories}

    for rf in route_files:
        path = config_dir / rf
        filename = Path(rf).name
        cat = _ROUTE_FILE_CATEGORY.get(filename, "background")  # default unknown to background
        if cat not in by_cat:
            by_cat[cat] = {"trips": 0, "vehicles": 0, "flows": 0, "flow_vehicles_est": 0, "persons": 0}
        s = _parse_route_file(path)
        for k in by_cat[cat]:
            by_cat[cat][k] += s[k]

    # Build totals and per-category demand
    total = {
        "background_trips": by_cat["background"]["trips"],
        "background_vehicles": by_cat["background"]["vehicles"],
        "background_flows": by_cat["background"]["flows"],
        "background_flow_veh": by_cat["background"]["flow_vehicles_est"],
        "background_demand": by_cat["background"]["trips"] + by_cat["background"]["vehicles"] + by_cat["background"]["flow_vehicles_est"],
        "taxi_vehicles": by_cat["taxi"]["vehicles"],
        "taxi_persons": by_cat["taxi"]["persons"],
        "taxi_demand": by_cat["taxi"]["vehicles"] + by_cat["taxi"]["persons"],
        "pt_vehicles": by_cat["pt"]["vehicles"],
        "pt_flows": by_cat["pt"]["flows"],
        "pt_flow_veh": by_cat["pt"]["flow_vehicles_est"],
        "pt_persons": by_cat["pt"]["persons"],
        "pt_demand": by_cat["pt"]["vehicles"] + by_cat["pt"]["persons"] + by_cat["pt"]["flow_vehicles_est"],
        "walk_persons": by_cat["walk"]["persons"],
        "walk_demand": by_cat["walk"]["persons"],
        "sim_duration_h": (end - begin) / 3600.0,
    }
    total["total_demand_est"] = (
        total["background_demand"] + total["taxi_demand"] + total["pt_demand"] + total["walk_demand"]
    )
    return total


def _format_number(x: Any) -> str:
    """Format number for LaTeX (use comma as thousands separator in text)."""
    if isinstance(x, float):
        if x >= 1000:
            return f"{x:,.0f}"
        if x == int(x):
            return str(int(x))
        return f"{x:.2f}"
    return str(x)


def _format_latex_number(x: Any) -> str:
    """Format number for LaTeX table (no commas in LaTeX, use \\num{} if siunitx available)."""
    if isinstance(x, float):
        if x >= 1000:
            return f"{x:,.0f}".replace(",", "\\,")
        if x == int(x):
            return str(int(x))
        return f"{x:.2f}"
    return str(x)


def compute_statistics(
    config_path: Optional[Path] = None,
    net_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Compute network and demand statistics.

    Returns:
        (network_stats, demand_stats, env_name)
    """
    if config_path is not None:
        config_dir = config_path.parent
        env_name = config_dir.name
        net_file, route_files, begin, end = _parse_sumocfg(config_path)
        if net_file is None:
            raise FileNotFoundError(f"Could not parse net file from {config_path}")
        net_path = config_dir / net_file
        demand_stats = _aggregate_demand(config_dir, route_files, begin, end)
    else:
        if net_path is None:
            raise ValueError("Either --config or --net must be provided")
        env_name = net_path.stem
        demand_stats = {
            "background_demand": 0,
            "taxi_demand": 0,
            "pt_demand": 0,
            "walk_demand": 0,
            "sim_duration_h": 0,
            "total_demand_est": 0,
        }

    net_stats = _compute_network_stats(net_path)
    if "error" in net_stats:
        raise FileNotFoundError(net_stats["error"])

    return net_stats, demand_stats, env_name


def print_human_readable(
    env_name: str,
    net_stats: Dict[str, Any],
    demand_stats: Dict[str, Any],
) -> None:
    """Print statistics in human-readable form."""
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print("=" * 60)
    print("Network scale:")
    print(f"  Junctions:        {net_stats.get('junctions', 0):,}")
    print(f"  Edges:            {net_stats.get('edges', 0):,}")
    print(f"  Lanes:            {net_stats.get('lanes', 0):,}")
    print(f"  Traffic lights:   {net_stats.get('traffic_lights', 0):,}")
    print(f"  Lane-km:          {net_stats.get('lane_km', 0):.2f}")
    if net_stats.get("highway_edges", 0) > 0:
        print(f"  Highway edges:    {net_stats['highway_edges']:,}")
        print(f"  Highway lanes:    {net_stats['highway_lanes']:,}")
    if net_stats.get("ramp_edges", 0) > 0:
        print(f"  Ramp edges:       {net_stats['ramp_edges']:,}")
        print(f"  Ramp lanes:       {net_stats['ramp_lanes']:,}")
    print("Demand dynamics:")
    print(f"  Background:       {demand_stats.get('background_demand', 0):,} (trips+vehicles+flows from routes.background.rou.xml)")
    print(f"  Taxi:             {demand_stats.get('taxi_demand', 0):,} (taxi_fleet + persons.taxi)")
    print(f"  PT (bus+subway):  {demand_stats.get('pt_demand', 0):,} (ptflows + persons.pt)")
    print(f"  Walk:             {demand_stats.get('walk_demand', 0):,} (persons.walk)")
    print(f"  Sim. duration:    {demand_stats.get('sim_duration_h', 0):.2f} h")
    print(f"  Total demand:     {demand_stats.get('total_demand_est', 0):,}")


def _latex_escape(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", "\\_")


def to_latex_table(
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    caption: str = "Key statistics of the environments, including network scale and demand dynamics.",
    label: str = "tab:env-statistics",
) -> str:
    """Generate LaTeX table from list of (env_name, net_stats, demand_stats).
    Requires \\usepackage{booktabs} in preamble for \\toprule, \\midrule, \\bottomrule."""
    # Columns: Environment, Junctions, Edges, Lanes, TLS, Lane-km, BG, Taxi, PT, Walk, Total, Duration
    cols = [
        ("Environment", "l", lambda r: _latex_escape(r[0])),
        ("Junctions", "r", lambda r: _format_latex_number(r[1].get("junctions", 0))),
        ("Edges", "r", lambda r: _format_latex_number(r[1].get("edges", 0))),
        ("Lanes", "r", lambda r: _format_latex_number(r[1].get("lanes", 0))),
        ("TLS", "r", lambda r: _format_latex_number(r[1].get("traffic_lights", 0))),
        ("Lane-km", "r", lambda r: f"{r[1].get('lane_km', 0):.2f}"),
        ("BG", "r", lambda r: _format_latex_number(r[2].get("background_demand", 0))),
        ("Taxi", "r", lambda r: _format_latex_number(r[2].get("taxi_demand", 0))),
        ("PT", "r", lambda r: _format_latex_number(r[2].get("pt_demand", 0))),
        ("Walk", "r", lambda r: _format_latex_number(r[2].get("walk_demand", 0))),
        ("Total", "r", lambda r: _format_latex_number(r[2].get("total_demand_est", 0))),
        ("Duration (h)", "r", lambda r: f"{r[2].get('sim_duration_h', 0):.2f}"),
    ]

    header = " & ".join(c[0] for c in cols) + " \\\\"
    colspec = "".join(c[1] for c in cols)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in rows:
        cells = [str(c[2](row)) for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute network and demand statistics for SUMO environments"
    )
    parser.add_argument(
        "--config",
        action="append",
        type=str,
        default=["sumo_config/Upper_Manhattan/Upper_Manhattan.sumocfg", "sumo_config/Inner_Queens/Inner_Queens.sumocfg"],
        help="Path to SUMO config (.sumocfg). Can be repeated for multiple envs.",
    )
    parser.add_argument(
        "--net",
        type=str,
        help="Path to network file (.net.xml). Used when --config is not provided.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output LaTeX table to file.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Key statistics of the environments, including network scale and demand dynamics.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:env-statistics",
        help="LaTeX table label.",
    )
    args = parser.parse_args()

    configs = args.config or []
    net_path = Path(args.net) if args.net else None

    if not configs and net_path is None:
        parser.error("Either --config or --net must be provided")

    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    if configs:
        for cfg in configs:
            cfg_path = Path(cfg).resolve()
            try:
                net_stats, demand_stats, env_name = compute_statistics(config_path=cfg_path)
                rows.append((env_name, net_stats, demand_stats))
                print_human_readable(env_name, net_stats, demand_stats)
            except Exception as e:
                print(f"Error processing {cfg}: {e}", file=sys.stderr)
                return 1
    else:
        try:
            net_stats, demand_stats, env_name = compute_statistics(net_path=net_path)
            rows.append((env_name, net_stats, demand_stats))
            print_human_readable(env_name, net_stats, demand_stats)
        except Exception as e:
            print(f"Error processing {net_path}: {e}", file=sys.stderr)
            return 1

    if rows:
        latex = to_latex_table(rows, caption=args.caption, label=args.label)
        print("\n" + "=" * 60)
        print("LaTeX table:")
        print("=" * 60)
        print(latex)

        if args.output:
            out_path = Path(args.output)
            out_path.write_text(latex, encoding="utf-8")
            print(f"\nTable written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
