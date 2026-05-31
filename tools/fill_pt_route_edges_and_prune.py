#!/usr/bin/env python3
"""
Fill <route edges="..."> for PT routes based on a stop sequence and a SUMO network, and prune invalid data.

Motivation:
  - Routes generated from timetables start with edges="PLACEHOLDER" and a stop sequence param.
  - SUMO requires that each vehicle's planned stops are downstream of its route edges; otherwise it aborts.

This tool:
  1) reads a mapped PT stop file (.add.xml) to get busStop -> edge mapping (from param key="edge" or lane)
  2) reads a PT routes file (.rou.xml) with <route><param key="stops_sequence" .../>
  3) uses sumolib on the provided network to compute a connected edge path that visits stop-edges in order
  4) drops routes that cannot be constructed and drops vehicles that reference dropped routes

Works well on time-windowed PT subsets (hundreds of routes), not on full-day citywide timetables without batching.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET
import subprocess
import uuid


def _clone(elem: ET.Element) -> ET.Element:
    copied = ET.Element(elem.tag, dict(elem.attrib))
    copied.text = elem.text
    copied.tail = elem.tail
    for child in list(elem):
        copied.append(_clone(child))
    return copied


def _edge_from_lane(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id


def _load_stop_edge_map(additional_files: Sequence[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in additional_files:
        root = ET.parse(p).getroot()
        for bs in root.findall("busStop"):
            bs_id = (bs.get("id") or "").strip()
            if not bs_id:
                continue
            edge_param = bs.find("param[@key='edge']")
            if edge_param is not None and edge_param.get("value"):
                mapping[bs_id] = str(edge_param.get("value"))
                continue
            lane = (bs.get("lane") or "").strip()
            if lane:
                mapping[bs_id] = _edge_from_lane(lane)
    return mapping


def _load_stop_startpos_map(additional_files: Sequence[Path]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for p in additional_files:
        root = ET.parse(p).getroot()
        for bs in root.findall("busStop"):
            bs_id = (bs.get("id") or "").strip()
            if not bs_id:
                continue
            sp = (bs.get("startPos") or "").strip()
            if not sp:
                continue
            try:
                mapping[bs_id] = float(sp)
            except Exception:
                continue
    return mapping


def _load_stop_posrange_map(additional_files: Sequence[Path]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    mapping: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for p in additional_files:
        root = ET.parse(p).getroot()
        for bs in root.findall("busStop"):
            bs_id = (bs.get("id") or "").strip()
            if not bs_id:
                continue
            sp = (bs.get("startPos") or "").strip()
            ep = (bs.get("endPos") or "").strip()
            start: Optional[float] = None
            end: Optional[float] = None
            try:
                if sp:
                    start = float(sp)
            except Exception:
                start = None
            try:
                if ep:
                    end = float(ep)
            except Exception:
                end = None
            if start is not None or end is not None:
                mapping[bs_id] = (start, end)
    return mapping


def _stops_sequence(route_elem: ET.Element) -> List[str]:
    p = route_elem.find("param[@key='stops_sequence']")
    if p is None:
        return []
    v = (p.get("value") or "").strip()
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Stats:
    routes_total: int = 0
    routes_kept: int = 0
    routes_dropped_missing_stop: int = 0
    routes_dropped_no_path: int = 0
    routes_dropped_duarouter_missing: int = 0
    routes_dropped_not_downstream: int = 0
    vehicles_total: int = 0
    vehicles_kept: int = 0
    vehicles_dropped_route_missing: int = 0
    segments_total: int = 0
    segments_cached: int = 0


def _shortest_path_edges_cached(net, from_edge: str, to_edge: str, cache: Dict[Tuple[str, str], Optional[List[str]]]) -> Optional[List[str]]:
    key = (from_edge, to_edge)
    if key in cache:
        return cache[key]
    try:
        from_obj = net.getEdge(from_edge)
        to_obj = net.getEdge(to_edge)
        res = net.getShortestPath(from_obj, to_obj)
        if not res:
            cache[key] = None
            return None
        edges, _cost = res
        out = [e.getID() for e in edges]
        cache[key] = out
        return out
    except Exception:
        cache[key] = None
        return None


def _dedupe_consecutive(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _is_stop_sequence_downstream(
    *,
    route_edges: List[str],
    stop_ids: List[str],
    stop_to_edge: Dict[str, str],
    stop_to_startpos: Dict[str, float],
    stop_to_posrange: Dict[str, Tuple[Optional[float], Optional[float]]],
    eps: float = 1e-3,
) -> bool:
    if not route_edges or len(stop_ids) < 2:
        return False

    prev_edge: Optional[str] = None
    prev_pos_end: Optional[float] = None
    idx = -1

    for sid in stop_ids:
        e = stop_to_edge.get(sid)
        if not e:
            return False

        # find the next occurrence of the edge after the last matched edge index
        start = idx if e == prev_edge else idx + 1
        found = None
        for j in range(start, len(route_edges)):
            if route_edges[j] == e:
                found = j
                break
        if found is None:
            return False

        pos_start, pos_end = stop_to_posrange.get(sid, (None, None))
        if pos_start is None:
            pos_start = stop_to_startpos.get(sid)
        if pos_end is None:
            pos_end = pos_start

        if prev_edge == e and found == idx and prev_pos_end is not None and pos_start is not None:
            # still on the same edge occurrence; next stop must begin after the previous stop end
            if pos_start + eps < prev_pos_end:
                return False

        idx = found
        prev_edge = e
        prev_pos_end = pos_end

    return True


def _duarouter_compute_route_edges(
    *,
    net_file: Path,
    route_id_to_stop_edges: Dict[str, List[str]],
    vclass: str,
    duarouter_bin: str,
    routing_threads: Optional[int],
    work_dir: Path,
) -> Dict[str, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    trips_file = work_dir / f"pt_fill_{tag}.trips.xml"
    out_file = work_dir / f"pt_fill_{tag}.rou.xml"

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.SubElement(root, "vType", {"id": "pt_tmp", "vClass": vclass})

    for rid, stop_edges in sorted(route_id_to_stop_edges.items()):
        edges = _dedupe_consecutive(stop_edges)
        if len(edges) < 2:
            continue
        trip_attrib = {
            "id": rid,
            "type": "pt_tmp",
            "depart": "0",
            "from": edges[0],
            "to": edges[-1],
        }
        via_edges = edges[1:-1]
        if via_edges:
            trip_attrib["via"] = " ".join(via_edges)
        ET.SubElement(root, "trip", trip_attrib)

    ET.indent(root, space="  ", level=0)
    ET.ElementTree(root).write(trips_file, encoding="utf-8", xml_declaration=True)

    cmd = [
        duarouter_bin,
        "--net-file",
        str(net_file),
        "--route-files",
        str(trips_file),
        "--output-file",
        str(out_file),
        "--ignore-errors",
        "--no-step-log",
    ]
    if routing_threads is not None:
        cmd += ["--routing-threads", str(int(routing_threads))]

    subprocess.run(cmd, check=True)

    out_root = ET.parse(out_file).getroot()
    named_routes: Dict[str, str] = {}
    for r in out_root.findall("route"):
        rid = (r.get("id") or "").strip()
        edges = (r.get("edges") or "").strip()
        if rid and edges:
            named_routes[rid] = edges

    route_edges: Dict[str, str] = {}
    for v in out_root.findall("vehicle"):
        vid = (v.get("id") or "").strip()
        edges = ""
        rchild = v.find("route")
        if rchild is not None and rchild.get("edges"):
            edges = str(rchild.get("edges") or "").strip()
        if not edges:
            rdist = v.find("routeDistribution")
            if rdist is not None:
                r2 = rdist.find("route")
                if r2 is not None and r2.get("edges"):
                    edges = str(r2.get("edges") or "").strip()
        if not edges:
            ref = (v.get("route") or "").strip()
            if ref and ref in named_routes:
                edges = named_routes[ref]
        if vid and edges:
            route_edges[vid] = edges

    # best-effort cleanup (keep files for debugging if desired)
    try:
        trips_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        out_file.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    return route_edges


def fill_edges_and_prune(
    net_file: Path,
    routes_file: Path,
    additional_files: Sequence[Path],
    output_file: Path,
    *,
    method: str = "sumolib",
    vclass: str = "bus",
    duarouter_bin: str = "duarouter",
    routing_threads: Optional[int] = None,
) -> Stats:
    stop_to_edge = _load_stop_edge_map(additional_files)
    stop_to_startpos = _load_stop_startpos_map(additional_files)
    stop_to_posrange = _load_stop_posrange_map(additional_files)
    stats = Stats()

    root = ET.parse(routes_file).getroot()
    vtypes = [_clone(e) for e in root.findall("vType")]
    routes = {r.get("id"): r for r in root.findall("route") if r.get("id")}
    vehicles = [v for v in root.findall("vehicle")]

    stats.routes_total = len(routes)
    stats.vehicles_total = len(vehicles)

    valid_routes: Dict[str, ET.Element] = {}
    route_id_to_stop_edges: Dict[str, List[str]] = {}
    route_id_to_stop_ids: Dict[str, List[str]] = {}
    for rid, r in routes.items():
        seq = _stops_sequence(r)
        if len(seq) < 2:
            stats.routes_dropped_missing_stop += 1
            continue
        stop_edges: List[str] = []
        for sid in seq:
            e = stop_to_edge.get(sid)
            if not e:
                stop_edges = []
                break
            stop_edges.append(e)
        if len(stop_edges) < 2:
            stats.routes_dropped_missing_stop += 1
            continue
        route_id_to_stop_edges[rid] = stop_edges
        route_id_to_stop_ids[rid] = seq

    if method == "sumolib":
        import sumolib  # type: ignore

        net = sumolib.net.readNet(str(net_file))
        cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
        for rid, stop_edges in route_id_to_stop_edges.items():
            full: List[str] = []
            ok = True
            for i in range(len(stop_edges) - 1):
                a = stop_edges[i]
                b = stop_edges[i + 1]
                stats.segments_total += 1
                if (a, b) in cache:
                    stats.segments_cached += 1
                seg = _shortest_path_edges_cached(net, a, b, cache)
                if not seg:
                    ok = False
                    break
                if i == 0:
                    full.extend(seg)
                else:
                    full.extend(seg[1:])
            if not ok or not full:
                stats.routes_dropped_no_path += 1
                continue

            stop_ids = route_id_to_stop_ids.get(rid, [])
            if not _is_stop_sequence_downstream(
                route_edges=full,
                stop_ids=stop_ids,
                stop_to_edge=stop_to_edge,
                stop_to_startpos=stop_to_startpos,
                stop_to_posrange=stop_to_posrange,
            ):
                stats.routes_dropped_not_downstream += 1
                continue

            r2 = _clone(routes[rid])
            r2.set("edges", " ".join(full))
            valid_routes[rid] = r2
            stats.routes_kept += 1
    elif method == "duarouter":
        computed = _duarouter_compute_route_edges(
            net_file=net_file,
            route_id_to_stop_edges=route_id_to_stop_edges,
            vclass=vclass,
            duarouter_bin=duarouter_bin,
            routing_threads=routing_threads,
            work_dir=output_file.parent,
        )
        for rid, stop_edges in route_id_to_stop_edges.items():
            edges = computed.get(rid, "").strip()
            if not edges:
                stats.routes_dropped_duarouter_missing += 1
                continue

            edge_list = [x for x in edges.split() if x]
            stop_ids = route_id_to_stop_ids.get(rid, [])
            if not _is_stop_sequence_downstream(
                route_edges=edge_list,
                stop_ids=stop_ids,
                stop_to_edge=stop_to_edge,
                stop_to_startpos=stop_to_startpos,
                stop_to_posrange=stop_to_posrange,
            ):
                stats.routes_dropped_not_downstream += 1
                continue

            r2 = _clone(routes[rid])
            r2.set("edges", edges)
            valid_routes[rid] = r2
            stats.routes_kept += 1
    else:
        raise ValueError(f"Unknown method: {method}")

    valid_route_ids: Set[str] = set(valid_routes.keys())
    kept_vehicles: List[ET.Element] = []
    for v in vehicles:
        rid = (v.get("route") or "").strip()
        if rid and rid in valid_route_ids:
            kept_vehicles.append(_clone(v))
            stats.vehicles_kept += 1
        else:
            stats.vehicles_dropped_route_missing += 1

    out_root = ET.Element("routes")
    out_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    out_root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    out_root.append(
        ET.Comment(
            f"Filled edges from {routes_file.name} using {net_file.name}; keptRoutes={stats.routes_kept} keptVehicles={stats.vehicles_kept}"
        )
    )

    for vt in vtypes:
        out_root.append(vt)
    for rid in sorted(valid_routes.keys()):
        out_root.append(valid_routes[rid])
    for v in kept_vehicles:
        out_root.append(v)

    ET.indent(out_root, space="  ", level=0)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out_root).write(output_file, encoding="utf-8", xml_declaration=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill PT route edges and prune invalid routes/vehicles")
    ap.add_argument("--net", required=True, help="SUMO net.xml")
    ap.add_argument("--routes", required=True, help="Input PT routes .rou.xml (with stops_sequence)")
    ap.add_argument("--additional", action="append", default=[], help="Mapped stops .add.xml (repeatable)")
    ap.add_argument("--output", required=True, help="Output .rou.xml")
    ap.add_argument(
        "--method",
        choices=["sumolib", "duarouter"],
        default="sumolib",
        help="How to compute route edges (default: sumolib; duarouter is faster on large networks).",
    )
    ap.add_argument("--vclass", default="bus", help="Vehicle class for duarouter routing (default: bus)")
    ap.add_argument("--duarouter-bin", default="duarouter", help="Path to duarouter binary (default: duarouter)")
    ap.add_argument("--routing-threads", type=int, help="duarouter --routing-threads value")
    args = ap.parse_args()

    add_files = [Path(p) for p in (args.additional or [])]
    if not add_files:
        raise SystemExit("ERROR: pass at least one --additional file containing busStop->edge mapping")

    stats = fill_edges_and_prune(
        net_file=Path(args.net),
        routes_file=Path(args.routes),
        additional_files=add_files,
        output_file=Path(args.output),
        method=str(args.method),
        vclass=str(args.vclass),
        duarouter_bin=str(args.duarouter_bin),
        routing_threads=args.routing_threads,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
