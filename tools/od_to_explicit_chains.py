#!/usr/bin/env python3
"""
OD (edge->edge) -> explicit public-transport person chains (walk/ride legs) -> SUMO persons routes.

This tool is designed for this repo's workflow:
  - input: OD JSON file (only fromEdge/toEdge/depart/personId)
  - planning: uses SUMO intermodal routing via TraCI:
      traci.simulation.findIntermodalRoute(fromEdge, toEdge, modes="public", depart=...)
    which returns a sequence of stages (walk/ride). We convert those stages into explicit legs:
      walk (edge->busStop), ride (busStop->busStop, lines=[line]), ..., walk (busStop->edge)
  - output:
      1) explicit chains JSON (one plan per person)
      2) persons.rou.xml with <person><walk>/<ride lines="...">... for SUMO simulation

Notes / limitations:
  - findIntermodalRoute returns the currently fastest route. If this fastest route violates your
    constraints (e.g. too many transfers) there is no guarantee SUMO returns an alternative without
    doing a more complex k-best / constrained search. This tool implements "relaxation" by widening
    acceptance thresholds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class OD:
    person_id: str
    depart: int
    from_edge: str
    to_edge: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_ods(obj: Any) -> List[OD]:
    """
    Accept either:
      - a list of OD objects
      - {"ods": [...], "defaults": {...}}
    Each OD object: {personId, depart, fromEdge, toEdge}
    """
    defaults: Dict[str, Any] = {}
    if isinstance(obj, dict) and "ods" in obj:
        defaults = obj.get("defaults") or {}
        items = obj["ods"]
    else:
        items = obj

    if not isinstance(items, list) or not items:
        raise ValueError("OD JSON must be a non-empty list or an object with key 'ods' (a non-empty list)")

    ods: List[OD] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"OD #{idx} must be an object")
        person_id = str(it.get("personId") or defaults.get("personId") or "").strip()
        if not person_id:
            person_id = f"p{idx}"
        depart = it.get("depart", defaults.get("depart", 0))
        from_edge = str(it.get("fromEdge") or it.get("originEdge") or "").strip()
        to_edge = str(it.get("toEdge") or it.get("destinationEdge") or "").strip()
        if not from_edge or not to_edge:
            raise ValueError(f"OD #{idx} missing fromEdge/toEdge")
        ods.append(OD(person_id=person_id, depart=int(depart), from_edge=from_edge, to_edge=to_edge))
    return ods


def _stages_to_plan(
    stages: Sequence[Any],
    person_id: str,
    depart: int,
    from_edge: str,
    to_edge: str,
) -> Dict[str, Any]:
    """
    Convert TraCI Stage objects from findIntermodalRoute to this repo's explicit chain JSON schema.
    """
    legs: List[Dict[str, Any]] = []
    current_ref: Dict[str, str] = {"edge": from_edge}

    for idx, st in enumerate(stages):
        st_type = getattr(st, "type", None)
        line = str(getattr(st, "line", "") or "")
        dest_stop = str(getattr(st, "destStop", "") or "")

        is_ride = st_type == 3 or (line != "" and dest_stop != "")
        is_walk = st_type == 2 or (line == "" and st_type is not None)

        if is_ride:
            if "busStop" not in current_ref:
                raise ValueError(
                    f"Cannot build ride leg at stage #{idx}: expected current_ref to be busStop, got {current_ref}"
                )
            if not dest_stop:
                raise ValueError(f"Ride stage #{idx} has empty destStop; cannot create busStop destination")
            legs.append(
                {
                    "type": "ride",
                    "from": {"busStop": current_ref["busStop"]},
                    "to": {"busStop": dest_stop},
                    "lines": [line] if line else [],
                }
            )
            current_ref = {"busStop": dest_stop}
            continue

        if is_walk:
            if dest_stop:
                to_ref = {"busStop": dest_stop}
                legs.append({"type": "walk", "from": dict(current_ref), "to": to_ref})
                current_ref = {"busStop": dest_stop}
            else:
                to_ref = {"edge": to_edge}
                legs.append({"type": "walk", "from": dict(current_ref), "to": to_ref})
                current_ref = {"edge": to_edge}
            continue

        raise ValueError(f"Unsupported stage at index {idx}: type={st_type} line={line!r} destStop={dest_stop!r}")

    plan = {"personId": person_id, "depart": depart, "legs": legs, "meta": {"fromEdge": from_edge, "toEdge": to_edge}}
    return plan


def _plan_stats_from_stages(stages: Sequence[Any]) -> Dict[str, Any]:
    ride_count = 0
    first_walk_len: Optional[float] = None
    last_walk_len: Optional[float] = None

    for i, st in enumerate(stages):
        st_type = getattr(st, "type", None)
        line = str(getattr(st, "line", "") or "")
        dest_stop = str(getattr(st, "destStop", "") or "")
        length = float(getattr(st, "length", 0.0) or 0.0)

        is_ride = st_type == 3 or (line != "" and dest_stop != "")
        is_walk = st_type == 2 or (line == "" and st_type is not None)

        if is_ride:
            ride_count += 1
        if is_walk and i == 0 and dest_stop:
            first_walk_len = length
        if is_walk and i == len(stages) - 1 and not dest_stop:
            last_walk_len = length

    transfers = max(0, ride_count - 1)
    total_cost = float(sum(float(getattr(s, "cost", 0.0) or 0.0) for s in stages))
    total_travel_time = float(sum(float(getattr(s, "travelTime", 0.0) or 0.0) for s in stages))

    return {
        "rideCount": ride_count,
        "transferCount": transfers,
        "accessWalkLength": first_walk_len,
        "egressWalkLength": last_walk_len,
        "totalCost": total_cost,
        "totalTravelTime": total_travel_time,
    }


def _meets_constraints(
    stats: Dict[str, Any],
    max_access_egress_walk_m: Optional[float],
    max_transfers: Optional[int],
) -> bool:
    if max_transfers is not None and int(stats["transferCount"]) > int(max_transfers):
        return False
    if max_access_egress_walk_m is not None:
        a = stats.get("accessWalkLength")
        e = stats.get("egressWalkLength")
        if a is not None and float(a) > float(max_access_egress_walk_m):
            return False
        if e is not None and float(e) > float(max_access_egress_walk_m):
            return False
    return True


def _relaxation_grid(walk_limits_m: Sequence[float], max_transfers_seq: Sequence[int]) -> List[Tuple[Optional[float], Optional[int]]]:
    out: List[Tuple[Optional[float], Optional[int]]] = []
    for t in max_transfers_seq:
        for w in walk_limits_m:
            out.append((float(w), int(t)))
    return out


def _filter_pt_if_requested(
    out_dir: Path,
    routes_files: Sequence[Path],
    begin: int,
    end: int,
    filter_pt: bool,
) -> List[Path]:
    if not filter_pt:
        return [p.resolve() for p in routes_files]

    filtered: List[Path] = []
    for idx, src in enumerate(routes_files):
        dst = out_dir / f"pt_{idx}.rou.xml"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "filter_pt_routes.py"),
            "--input",
            str(src),
            "--output",
            str(dst),
            "--all-lines",
            "--begin",
            str(begin),
            "--end",
            str(end),
        ]
        subprocess.run(cmd, check=True)
        filtered.append(dst)
    return filtered


def _render_persons_routes(plans: Sequence[Dict[str, Any]]) -> ET.ElementTree:
    """
    Render a routes file that contains multiple <person> elements with <walk>/<ride> stages.
    """
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    root.append(ET.Comment("Generated by tools/od_to_explicit_chains.py"))

    for plan in plans:
        person = ET.SubElement(root, "person")
        person.set("id", str(plan["personId"]))
        person.set("depart", str(int(plan["depart"])))

        for leg in plan["legs"]:
            leg_type = leg["type"]
            from_ref = leg["from"]
            to_ref = leg["to"]

            if leg_type == "walk":
                w = ET.SubElement(person, "walk")
                if "edge" in from_ref:
                    w.set("from", str(from_ref["edge"]))
                if "busStop" in to_ref:
                    w.set("busStop", str(to_ref["busStop"]))
                elif "edge" in to_ref:
                    w.set("to", str(to_ref["edge"]))
                else:
                    raise ValueError(f"walk leg to must contain busStop or edge, got {to_ref}")
            elif leg_type == "ride":
                r = ET.SubElement(person, "ride")
                lines = leg.get("lines") or []
                r.set("lines", " ".join(str(x) for x in lines))
                if "busStop" in to_ref:
                    r.set("busStop", str(to_ref["busStop"]))
                elif "edge" in to_ref:
                    r.set("to", str(to_ref["edge"]))
                else:
                    raise ValueError(f"ride leg to must contain busStop or edge, got {to_ref}")
            else:
                raise ValueError(f"Unknown leg type {leg_type!r}")

    ET.indent(root, space="  ", level=0)
    return ET.ElementTree(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="OD(JSON) -> explicit PT chains(JSON) + persons.rou.xml using SUMO intermodal routing")
    parser.add_argument("--od", required=True, help="OD JSON file path")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--scenario", help="Scenario name from --scenario-config (toy/nyc)")
    parser.add_argument(
        "--scenario-config",
        default=str(Path(__file__).resolve().parent / "pt_scenarios.json"),
        help="Scenario config JSON path",
    )

    parser.add_argument("--net", help="SUMO .net.xml path (overrides scenario)")
    parser.add_argument("--routes", action="append", default=[], help="PT routes .rou.xml (repeatable; overrides scenario if given)")
    parser.add_argument("--additional", action="append", default=[], help="PT stops additional .add.xml (repeatable; overrides scenario if given)")

    parser.add_argument("--modes", default="public", help='Intermodal modes string for findIntermodalRoute (default: "public")')
    parser.add_argument("--walk-limit", type=float, action="append", default=[1000.0, 2000.0, 5000.0], help="Access/egress walk length limit in meters (repeatable, relaxed in order)")
    parser.add_argument("--max-transfers", type=int, action="append", default=[2, 3, 4], help="Max transfers (repeatable, relaxed in order)")

    parser.add_argument("--pt-window-before", type=int, default=3600, help="Filter PT routes begin = minDepart - this (sec)")
    parser.add_argument("--pt-window-after", type=int, default=7200, help="Filter PT routes end = maxDepart + this (sec)")
    parser.add_argument("--no-filter-pt", action="store_true", help="Do not filter PT routes by time window")

    parser.add_argument("--sumo", default="sumo", help="SUMO binary to start for TraCI routing")
    parser.add_argument(
        "--ignore-route-errors",
        action="store_true",
        help="Pass SUMO option --ignore-route-errors (SUMO CLI) to avoid aborting on inconsistent routes",
    )
    parser.add_argument("--walk-factor", type=float, default=-1.0, help="walkFactor for findIntermodalRoute (default: SUMO default)")
    parser.add_argument("--pType", default="", help="Pedestrian type id for routing (optional)")
    args = parser.parse_args()

    od_path = Path(args.od)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ods = _parse_ods(_load_json(od_path))

    scenario_net: Optional[str] = None
    scenario_routes: List[str] = []
    scenario_additional: List[str] = []
    if args.scenario:
        cfg = _load_json(Path(args.scenario_config))
        if args.scenario not in cfg:
            raise SystemExit(f"ERROR: unknown scenario {args.scenario!r} in {args.scenario_config}")
        scenario_net = str(cfg[args.scenario].get("net") or "")
        scenario_routes = [str(x) for x in (cfg[args.scenario].get("routes") or [])]
        scenario_additional = [str(x) for x in (cfg[args.scenario].get("additional") or [])]

    net_arg = args.net or scenario_net
    routes_arg = args.routes or scenario_routes
    additional_arg = args.additional or scenario_additional

    if not net_arg:
        raise SystemExit("ERROR: missing --net (or --scenario with a net configured)")
    if not routes_arg:
        raise SystemExit("ERROR: missing --routes (or --scenario with routes configured)")

    begin = min(o.depart for o in ods) - int(args.pt_window_before)
    end = max(o.depart for o in ods) + int(args.pt_window_after)
    begin = max(0, begin)

    routes_files = [Path(p) for p in routes_arg]
    additional_files = [Path(p) for p in additional_arg]
    net_path = Path(net_arg)

    filtered_routes = _filter_pt_if_requested(
        out_dir=out_dir,
        routes_files=routes_files,
        begin=begin,
        end=end,
        filter_pt=not args.no_filter_pt,
    )

    # Start SUMO with TraCI for routing queries.
    import traci  # type: ignore

    sumo_cmd = [
        args.sumo,
        "--no-step-log",
        "true",
        "--step-length",
        "1",
        "--begin",
        str(begin),
        "--end",
        str(end),
        "-n",
        str(net_path),
        "-r",
        ",".join(str(p) for p in filtered_routes),
    ]
    if additional_files:
        sumo_cmd += ["-a", ",".join(str(p) for p in additional_files)]
    if args.ignore_route_errors:
        sumo_cmd += ["--ignore-route-errors"]

    traci.start(sumo_cmd)
    plans: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "odFile": str(od_path),
        "net": str(net_path),
        "routes": [str(p) for p in routes_files],
        "additional": [str(p) for p in additional_files],
        "timeWindow": {"begin": begin, "end": end},
        "constraints": {"walkLimits": args.walk_limit, "maxTransfers": args.max_transfers},
        "results": [],
    }

    try:
        relax = _relaxation_grid(args.walk_limit, args.max_transfers)
        for od in ods:
            item: Dict[str, Any] = {
                "personId": od.person_id,
                "depart": od.depart,
                "fromEdge": od.from_edge,
                "toEdge": od.to_edge,
                "status": "error",
            }
            try:
                stages = traci.simulation.findIntermodalRoute(
                    od.from_edge,
                    od.to_edge,
                    args.modes,
                    depart=float(od.depart),
                    walkFactor=float(args.walk_factor),
                    pType=str(args.pType),
                )
                stages = tuple(stages)
                stats = _plan_stats_from_stages(stages)

                accepted: Optional[Tuple[float, int]] = None
                for walk_limit, max_transfers in relax:
                    if _meets_constraints(stats, max_access_egress_walk_m=walk_limit, max_transfers=max_transfers):
                        accepted = (float(walk_limit), int(max_transfers))
                        break

                if accepted is None:
                    item["status"] = "no_solution_under_constraints"
                    item["stats"] = stats
                    report["results"].append(item)
                    continue

                plan = _stages_to_plan(
                    stages=stages,
                    person_id=od.person_id,
                    depart=od.depart,
                    from_edge=od.from_edge,
                    to_edge=od.to_edge,
                )
                plan["meta"]["routing"] = {
                    "modes": args.modes,
                    "acceptedWalkLimit": accepted[0],
                    "acceptedMaxTransfers": accepted[1],
                    "stats": stats,
                }
                plans.append(plan)
                item["status"] = "ok"
                item["acceptedWalkLimit"] = accepted[0]
                item["acceptedMaxTransfers"] = accepted[1]
                item["stats"] = stats
                report["results"].append(item)
            except Exception as e:
                item["error"] = str(e)
                report["results"].append(item)
    finally:
        traci.close()

    chains_path = out_dir / "explicit_chains.json"
    persons_path = out_dir / "persons.rou.xml"
    report_path = out_dir / "report.json"

    _dump_json(chains_path, plans)
    _dump_json(report_path, report)

    tree = _render_persons_routes(plans)
    tree.write(persons_path, encoding="utf-8", xml_declaration=True)

    print(json.dumps({"outputDir": str(out_dir), "chains": str(chains_path), "persons": str(persons_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
