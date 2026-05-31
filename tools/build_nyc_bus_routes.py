#!/usr/bin/env python3
"""
Build a NYC bus PT routes file from the merged bus timetable JSON.

Why this exists:
  - Data/timetables/merged_bus_timetable.json contains many bus lines.
  - The repo's sumo_network/bus_routes_complete.rou.xml currently only includes B42 because earlier
    conversions used a global --max-vehicles limit that got consumed by the first line.

This script:
  1) converts timetable JSON -> SUMO bus routes/vehicles (.rou.xml) using convert_timetable_to_sumo.py
     (supports per-line limits and depart time window filtering)
  2) fills <route edges="..."> to ensure all planned stops are downstream, using:
       tools/fill_pt_route_edges_and_prune.py
     This avoids SUMO aborting with "busStop ... is not downstream the current route".
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


def main() -> int:
    ap = argparse.ArgumentParser(description="Build NYC bus routes (.rou.xml) from merged_bus_timetable.json")
    ap.add_argument(
        "--timetable",
        default="Data/timetables/merged_bus_timetable.json",
        help="Merged bus timetable JSON (default: Data/timetables/merged_bus_timetable.json)",
    )
    ap.add_argument(
        "--stops-mapping",
        default="sumo_output/bus_stops_route_mapping.csv",
        help="CSV mapping stop_id->sumo_stop_id (default: sumo_output/bus_stops_route_mapping.csv)",
    )
    ap.add_argument(
        "--mapped-stops",
        default="sumo_network/bus_stops_mapped.add.xml",
        help="Mapped SUMO busStops add.xml with edge param (default: sumo_network/bus_stops_mapped.add.xml)",
    )
    ap.add_argument(
        "--out",
        default="sumo_network/bus_routes_all_complete.rou.xml",
        help="Output PT routes file (default: sumo_network/bus_routes_all_complete.rou.xml)",
    )
    ap.add_argument(
        "--network",
        default="sumo_network/newyork.net.xml",
        help="SUMO net file used for filling route edges (default: sumo_network/newyork.net.xml)",
    )
    ap.add_argument(
        "--routing-threads",
        type=int,
        default=1,
        help="Threads for duarouter when filling route edges (default: 1; higher may use much more memory).",
    )

    ap.add_argument("--depart-begin", type=int, help="Only include vehicles with depart >= this (sec)")
    ap.add_argument("--depart-end", type=int, help="Only include vehicles with depart <= this (sec)")
    ap.add_argument(
        "--max-vehicles-per-line",
        type=int,
        default=10,
        help="Max vehicles per bus line (default: 10). Use a larger number for denser schedules.",
    )
    ap.add_argument(
        "--service-policy",
        choices=["all", "first", "largest", "most_trips"],
        default="largest",
        help="Pick one service group per line during conversion (default: largest).",
    )
    ap.add_argument("--service-date", help="Prefer the service group that includes this date (YYYY-MM-DD or YYYY/MM/DD)")
    ap.add_argument("--service-key", help="Exact service key to use for every line (advanced)")
    ap.add_argument(
        "--max-vehicles-total",
        type=int,
        default=None,
        help="Optional global cap across all lines (default: unlimited)",
    )
    ap.add_argument(
        "--tmp",
        default="sumo_output/bus_routes_all_tmp.rou.xml",
        help="Temporary intermediate routes file (default: sumo_output/bus_routes_all_tmp.rou.xml)",
    )
    args = ap.parse_args()

    timetable = Path(args.timetable)
    stops_mapping = Path(args.stops_mapping)
    mapped_stops = Path(args.mapped_stops)
    tmp = Path(args.tmp)
    out = Path(args.out)

    tmp.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) timetable -> routes/vehicles (edges=PLACEHOLDER)
    cmd1: List[str] = [
        "python3",
        str(Path(__file__).resolve().parents[1] / "convert_timetable_to_sumo.py"),
        "-t",
        str(timetable),
        "-s",
        str(stops_mapping),
        "-o",
        str(tmp),
        "-m",
        "bus",
        "--max-vehicles-per-line",
        str(int(args.max_vehicles_per_line)),
        "--service-policy",
        str(args.service_policy),
    ]
    if args.service_date:
        cmd1 += ["--service-date", str(args.service_date)]
    if args.service_key:
        cmd1 += ["--service-key", str(args.service_key)]
    if args.max_vehicles_total is not None:
        cmd1 += ["--max-vehicles", str(int(args.max_vehicles_total))]
    if args.depart_begin is not None:
        cmd1 += ["--depart-begin", str(int(args.depart_begin))]
    if args.depart_end is not None:
        cmd1 += ["--depart-end", str(int(args.depart_end))]
    subprocess.run(cmd1, check=True)

    # 2) rewrite route edges using mapped stop->edge
    cmd2 = [
        "python3",
        str(Path(__file__).resolve().parent / "fill_pt_route_edges_and_prune.py"),
        "--net",
        str(Path(args.network)),
        "--routes",
        str(tmp),
        "--additional",
        str(mapped_stops),
        "--output",
        str(out),
        "--method",
        "duarouter",
        "--routing-threads",
        str(int(args.routing_threads)),
    ]
    subprocess.run(cmd2, check=True)

    print(f"OK: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
