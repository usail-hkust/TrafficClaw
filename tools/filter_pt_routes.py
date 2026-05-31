#!/usr/bin/env python3
"""
Filter a SUMO routes file to a small subset.

Keeps:
  - <vType> definitions
  - <route> definitions that are referenced by kept vehicles
  - <vehicle> entries whose:
      - @line is in a whitelist (optional)
      - @depart is within [begin, end]

This helps building small experiments from huge PT files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
import xml.etree.ElementTree as ET


def _parse_time_seconds(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _clone(elem: ET.Element) -> ET.Element:
    copied = ET.Element(elem.tag, dict(elem.attrib))
    copied.text = elem.text
    copied.tail = elem.tail
    for child in list(elem):
        copied.append(_clone(child))
    return copied


def filter_routes_file(
    input_file: Path,
    output_file: Path,
    allowed_lines: Optional[Set[str]],
    begin: int,
    end: int,
) -> Dict[str, int]:
    vtypes: List[ET.Element] = []
    routes_by_id: Dict[str, ET.Element] = {}
    kept_vehicles: List[ET.Element] = []
    needed_route_ids: Set[str] = set()

    # Important: ElementTree iterparse fires "end" events bottom-up. If we clear nested
    # <stop> elements before cloning their parent <vehicle>, we will lose @busStop/@until.
    for _, elem in ET.iterparse(str(input_file), events=("end",)):
        if elem.tag == "vType":
            vtypes.append(_clone(elem))
            elem.clear()
            continue

        if elem.tag == "route":
            rid = elem.get("id")
            if rid:
                routes_by_id[rid] = _clone(elem)
            elem.clear()
            continue

        if elem.tag == "vehicle":
            line = (elem.get("line") or "").strip()
            depart = _parse_time_seconds(elem.get("depart"))
            line_ok = True if allowed_lines is None else line in allowed_lines
            if line_ok and depart is not None and begin <= depart <= end:
                kept_vehicles.append(_clone(elem))
                route_id = elem.get("route")
                if route_id:
                    needed_route_ids.add(route_id)
            elem.clear()
            continue

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    lines_desc = "*" if allowed_lines is None else str(sorted(allowed_lines))
    root.append(
        ET.Comment(
            f"Filtered from {input_file.name} lines={lines_desc} depart=[{begin},{end}]"
        )
    )

    for vt in vtypes:
        root.append(vt)

    missing = 0
    for rid in sorted(needed_route_ids):
        route = routes_by_id.get(rid)
        if route is None:
            missing += 1
            continue
        root.append(route)

    for veh in kept_vehicles:
        root.append(veh)

    ET.indent(root, space="  ", level=0)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_file, encoding="utf-8", xml_declaration=True)

    return {
        "vTypes": len(vtypes),
        "routes": len(needed_route_ids) - missing,
        "vehicles": len(kept_vehicles),
        "missingRoutes": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter SUMO routes file by line and departure time")
    parser.add_argument("--input", required=True, help="Input .rou.xml")
    parser.add_argument("--output", required=True, help="Output .rou.xml")
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Keep vehicles for all line names (only filter by departure time window)",
    )
    parser.add_argument("--lines", nargs="+", help="Allowed vehicle line names (omit when using --all-lines)")
    parser.add_argument("--begin", type=int, required=True, help="Keep vehicles with depart >= begin")
    parser.add_argument("--end", type=int, required=True, help="Keep vehicles with depart <= end")
    args = parser.parse_args()

    if not args.all_lines and not args.lines:
        raise SystemExit("ERROR: either pass --all-lines or specify --lines ...")

    stats = filter_routes_file(
        input_file=Path(args.input),
        output_file=Path(args.output),
        allowed_lines=None if args.all_lines else set(args.lines or []),
        begin=args.begin,
        end=args.end,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
