#!/usr/bin/env python3
"""
Validate explicit_chains.json (a list of plans) against SUMO PT stop/route data.

This performs *data-level* validation (no network walk connectivity checks):
  - referenced busStops exist in provided additional (.add.xml) files
  - ride legs have at least one vehicle of the given line that stops at from->to in order
  - (optional) estimates catchability via <stop until="..."> given a time cursor

Intended workflow:
  1) python3 tools/od_to_explicit_chains.py ... --output-dir OUT
  2) python3 tools/explicit_chains_validate.py --chains OUT/explicit_chains.json --routes OUT/pt_0.rou.xml ... --additional ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Ref:
    kind: str
    value: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_ref(obj: Dict[str, Any]) -> Ref:
    if "edge" in obj:
        return Ref("edge", str(obj["edge"]))
    if "busStop" in obj:
        return Ref("busStop", str(obj["busStop"]))
    raise ValueError(f"Ref must contain 'edge' or 'busStop', got keys={sorted(obj.keys())}")


def _load_line_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    mapping = _load_json(path)
    if not isinstance(mapping, dict):
        raise ValueError("line-map must be a JSON object of {inputLine: sumoLine}")
    return {str(k): str(v) for k, v in mapping.items()}


def _map_lines(lines: Iterable[str], line_map: Dict[str, str], strip_prefixes: Tuple[str, ...]) -> List[str]:
    mapped: List[str] = []
    for raw in lines:
        raw = str(raw)
        if raw in line_map:
            mapped.append(line_map[raw])
            continue
        for prefix in strip_prefixes:
            if raw.startswith(prefix):
                mapped.append(raw[len(prefix) :])
                break
        else:
            mapped.append(raw)
    # unique while preserving order
    seen = set()
    out: List[str] = []
    for line in mapped:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _iter_bus_stop_ids(additional_xml: Path) -> Iterator[str]:
    for _, elem in ET.iterparse(additional_xml, events=("end",)):
        if elem.tag == "busStop":
            stop_id = elem.get("id")
            if stop_id:
                yield stop_id
        elem.clear()


def _collect_bus_stop_ids(additional_files: Sequence[Path]) -> Set[str]:
    ids: Set[str] = set()
    for f in additional_files:
        for stop_id in _iter_bus_stop_ids(f):
            ids.add(stop_id)
    return ids


def _iter_vehicle_candidates(routes_xml: Path, target_lines: Set[str]) -> Iterator[ET.Element]:
    for _, elem in ET.iterparse(routes_xml, events=("end",)):
        if elem.tag == "vehicle":
            line = elem.get("line") or ""
            if line in target_lines:
                yield elem
            elem.clear()


def _parse_until_seconds(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def _best_vehicle_arrival(
    routes_files: Sequence[Path],
    target_lines: Sequence[str],
    from_stop: str,
    to_stop: str,
    earliest_time: int,
) -> Optional[Dict[str, Any]]:
    target_set = set(target_lines)
    best: Optional[Dict[str, Any]] = None

    for routes_file in routes_files:
        for vehicle in _iter_vehicle_candidates(routes_file, target_set):
            veh_id = vehicle.get("id") or ""
            veh_line = vehicle.get("line") or ""
            depart_raw = vehicle.get("depart") or "0"
            depart = _parse_until_seconds(depart_raw) or 0

            from_until: Optional[int] = None
            to_until: Optional[int] = None
            seen_from = False

            for stop in vehicle.findall("stop"):
                bs = stop.get("busStop")
                until_raw = stop.get("until") or ""
                until = _parse_until_seconds(until_raw)
                if until is None or bs is None:
                    continue

                if bs == from_stop and from_until is None:
                    from_until = until
                    seen_from = True
                elif bs == to_stop and seen_from and to_until is None:
                    to_until = until
                    break

            if from_until is None or to_until is None:
                continue
            if earliest_time > from_until:
                continue

            arrival = to_until
            if best is None or arrival < best["arrivalTime"]:
                best = {
                    "routesFile": str(routes_file),
                    "vehicleId": veh_id,
                    "line": veh_line,
                    "vehicleDepart": depart,
                    "fromStop": from_stop,
                    "toStop": to_stop,
                    "boardLatest": from_until,
                    "arrivalTime": arrival,
                }
    return best


def _validate_one_plan(
    plan: Dict[str, Any],
    known_stops: Optional[Set[str]],
    routes_files: Sequence[Path],
    line_map: Dict[str, str],
    strip_prefixes: Tuple[str, ...],
) -> Dict[str, Any]:
    person_id = str(plan.get("personId", "")).strip()
    depart = int(plan.get("depart", 0))
    legs = plan.get("legs")
    if not person_id or not isinstance(legs, list) or not legs:
        return {"personId": person_id or None, "depart": depart, "ok": False, "errors": ["invalid plan shape"]}

    errors: List[str] = []
    stop_errors: List[str] = []
    ride_errors: List[str] = []
    ride_reports: List[Dict[str, Any]] = []

    # leg connectivity + stop existence
    prev_to: Optional[Ref] = None
    referenced_stops: Set[str] = set()
    referenced_lines: Set[str] = set()

    for idx, leg in enumerate(legs):
        leg_type = leg.get("type")
        if leg_type not in {"walk", "ride"}:
            errors.append(f"Leg #{idx} unknown type {leg_type!r}")
            continue

        from_ref = _parse_ref(leg["from"])
        to_ref = _parse_ref(leg["to"])

        if prev_to is not None and from_ref != prev_to:
            errors.append(f"Leg #{idx} starts at {from_ref} but previous ended at {prev_to}")
        prev_to = to_ref

        if from_ref.kind == "busStop":
            referenced_stops.add(from_ref.value)
        if to_ref.kind == "busStop":
            referenced_stops.add(to_ref.value)

        if leg_type == "ride":
            mapped_lines = _map_lines(leg.get("lines", []), line_map=line_map, strip_prefixes=strip_prefixes)
            referenced_lines.update(mapped_lines)

    if known_stops is not None:
        for s in sorted(referenced_stops):
            if s not in known_stops:
                stop_errors.append(f"Unknown busStop id: {s}")

    # timetable feasibility (ride legs only)
    time_cursor = depart
    for idx, leg in enumerate(legs):
        if leg.get("type") == "walk":
            duration = int(leg.get("duration_s", 0) or 0)
            time_cursor += max(0, duration)
            continue

        if leg.get("type") != "ride":
            continue

        from_ref = _parse_ref(leg["from"])
        to_ref = _parse_ref(leg["to"])
        if from_ref.kind != "busStop" or to_ref.kind != "busStop":
            ride_errors.append(f"Leg #{idx} ride must be busStop->busStop for timetable validation")
            continue

        mapped_lines = _map_lines(leg.get("lines", []), line_map=line_map, strip_prefixes=strip_prefixes)
        if not mapped_lines:
            ride_errors.append(f"Leg #{idx} ride has empty lines[]")
            continue

        if not routes_files:
            ride_errors.append(f"Leg #{idx} requires --routes to validate timetable (lines={mapped_lines})")
            continue

        best = _best_vehicle_arrival(
            routes_files=routes_files,
            target_lines=mapped_lines,
            from_stop=from_ref.value,
            to_stop=to_ref.value,
            earliest_time=time_cursor,
        )
        if best is None:
            ride_errors.append(
                f"Leg #{idx} no feasible PT vehicle found for lines={mapped_lines} from={from_ref.value} to={to_ref.value} at t>={time_cursor}"
            )
            continue
        ride_reports.append({"legIndex": idx, "earliestTimeAtFrom": time_cursor, **best})
        time_cursor = int(best["arrivalTime"])

    ok = not errors and not stop_errors and not ride_errors
    return {
        "personId": person_id,
        "depart": depart,
        "ok": ok,
        "errors": errors,
        "stopErrors": stop_errors,
        "rideErrors": ride_errors,
        "rides": ride_reports,
        "endTimeEstimate": time_cursor,
        "referencedLinesMapped": sorted(referenced_lines),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate explicit_chains.json against PT stop/routes data")
    ap.add_argument("--chains", required=True, help="explicit_chains.json path (list of plans)")
    ap.add_argument("--additional", action="append", default=[], help="SUMO additional files containing <busStop> (repeatable)")
    ap.add_argument("--routes", action="append", default=[], help="SUMO routes files containing PT vehicles (repeatable)")
    ap.add_argument("--line-map", help="Optional JSON mapping {inputLine: sumoLine}")
    ap.add_argument(
        "--strip-line-prefix",
        action="append",
        default=["subway:", "bus:"],
        help="Strip this prefix from a line label if no mapping exists (repeatable). Default: subway:, bus:",
    )
    ap.add_argument("--report", help="Optional output JSON report path")
    args = ap.parse_args()

    plans = _load_json(Path(args.chains))
    if not isinstance(plans, list):
        raise SystemExit("ERROR: --chains must be a JSON list (explicit_chains.json)")

    additional_files = [Path(p) for p in (args.additional or [])]
    routes_files = [Path(p) for p in (args.routes or [])]
    line_map = _load_line_map(Path(args.line_map) if args.line_map else None)
    strip_prefixes = tuple(str(p) for p in args.strip_line_prefix)

    known_stops = _collect_bus_stop_ids(additional_files) if additional_files else None

    per_person: List[Dict[str, Any]] = []
    ok_count = 0
    for plan in plans:
        if not isinstance(plan, dict):
            per_person.append({"ok": False, "errors": ["plan must be an object"]})
            continue
        rep = _validate_one_plan(
            plan=plan,
            known_stops=known_stops,
            routes_files=routes_files,
            line_map=line_map,
            strip_prefixes=strip_prefixes,
        )
        per_person.append(rep)
        if rep.get("ok"):
            ok_count += 1

    summary = {
        "chains": str(Path(args.chains)),
        "persons": len(per_person),
        "ok": ok_count,
        "failed": len(per_person) - ok_count,
        "results": per_person,
    }
    if args.report:
        _dump_json(Path(args.report), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok_count == len(per_person) else 1


if __name__ == "__main__":
    raise SystemExit(main())

