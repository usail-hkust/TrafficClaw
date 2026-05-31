#!/usr/bin/env python3
"""
Planning-only pipeline (no SUMO simulation):

  OD.json -> explicit_chains.json + persons.rou.xml (via TraCI intermodal routing)
        -> validate explicit_chains.json against PT stop/routes data

This is the "plan + validate" entrypoint:
  - It runs tools/od_to_explicit_chains.py
  - Then runs tools/explicit_chains_validate.py on the generated chains
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_filtered_pt_files(out_dir: Path) -> List[str]:
    files = sorted(out_dir.glob("pt_*.rou.xml"))
    return [str(p) for p in files]


def main() -> int:
    ap = argparse.ArgumentParser(description="OD.json -> explicit chains -> validate (planning-only, no simulation)")
    ap.add_argument("--od", required=True, help="OD JSON file")
    ap.add_argument("--output-dir", required=True, help="Output directory")

    ap.add_argument("--scenario", default="nyc", help="Scenario name from --scenario-config (default: nyc)")
    ap.add_argument(
        "--scenario-config",
        default=str(Path(__file__).resolve().parent / "pt_scenarios.json"),
        help="Scenario config JSON path",
    )

    ap.add_argument("--net", help="SUMO net file (overrides scenario)")
    ap.add_argument("--routes", action="append", default=[], help="PT routes .rou.xml (repeatable; overrides scenario if given)")
    ap.add_argument("--additional", action="append", default=[], help="PT stops additional .add.xml (repeatable; overrides scenario if given)")

    ap.add_argument("--sumo", default="sumo", help="SUMO binary (sumo or sumo-gui)")
    ap.add_argument(
        "--sumo-arg",
        action="append",
        default=[],
        help="Extra argument to pass to SUMO in the planning step (repeatable), e.g. --sumo-arg --ignore-route-errors",
    )
    ap.add_argument("--modes", default="public", help='Intermodal modes string (default: "public")')
    ap.add_argument("--walk-limit", type=float, action="append", default=[1000.0, 2000.0, 5000.0])
    ap.add_argument("--max-transfers", type=int, action="append", default=[2, 3, 4])
    ap.add_argument("--pt-window-before", type=int, default=3600)
    ap.add_argument("--pt-window-after", type=int, default=7200)
    ap.add_argument("--no-filter-pt", action="store_true")
    ap.add_argument("--walk-factor", type=float, default=-1.0)
    ap.add_argument("--pType", default="")

    # validation options
    ap.add_argument("--no-validate", action="store_true", help="Skip validation step")
    ap.add_argument("--line-map", help="Optional JSON mapping {inputLine: sumoLine} for validation")
    ap.add_argument(
        "--strip-line-prefix",
        action="append",
        default=["subway:", "bus:"],
        help="Strip this prefix from a line label for validation (repeatable). Default: subway:, bus:",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(Path(args.scenario_config))
    if args.scenario not in cfg:
        raise SystemExit(f"ERROR: unknown scenario {args.scenario!r} in {args.scenario_config}")
    scenario_net = str(cfg[args.scenario].get("net") or "")
    scenario_routes = [str(x) for x in (cfg[args.scenario].get("routes") or [])]
    scenario_additional = [str(x) for x in (cfg[args.scenario].get("additional") or [])]

    net_arg = args.net or scenario_net
    routes_arg = args.routes or scenario_routes
    additional_arg = args.additional or scenario_additional

    if not net_arg or not routes_arg:
        raise SystemExit("ERROR: missing net/routes (provide scenario or overrides)")

    # 1) planning
    planner_cmd: List[str] = [
        "python3",
        str(Path(__file__).resolve().parent / "od_to_explicit_chains.py"),
        "--od",
        args.od,
        "--output-dir",
        str(out_dir),
        "--net",
        net_arg,
    ]
    for rf in routes_arg:
        planner_cmd += ["--routes", rf]
    for af in additional_arg:
        planner_cmd += ["--additional", af]
    planner_cmd += ["--modes", args.modes]
    for w in args.walk_limit:
        planner_cmd += ["--walk-limit", str(w)]
    for t in args.max_transfers:
        planner_cmd += ["--max-transfers", str(t)]
    planner_cmd += ["--pt-window-before", str(args.pt_window_before), "--pt-window-after", str(args.pt_window_after)]
    if args.no_filter_pt:
        planner_cmd += ["--no-filter-pt"]
    planner_cmd += ["--sumo", args.sumo, "--walk-factor", str(args.walk_factor), "--pType", args.pType]
    for sa in args.sumo_arg:
        # allow passing SUMO flags like --ignore-route-errors (which start with '-')
        planner_cmd += [f"--sumo-arg={sa}"]
    subprocess.run(planner_cmd, check=True)

    if args.no_validate:
        return 0

    # 2) validation (prefer filtered pt_*.rou.xml files)
    chains_path = out_dir / "explicit_chains.json"
    filtered_pt = _find_filtered_pt_files(out_dir)
    routes_for_validation = filtered_pt if filtered_pt else routes_arg

    validator_cmd: List[str] = [
        "python3",
        str(Path(__file__).resolve().parent / "explicit_chains_validate.py"),
        "--chains",
        str(chains_path),
    ]
    for af in additional_arg:
        validator_cmd += ["--additional", af]
    for rf in routes_for_validation:
        validator_cmd += ["--routes", rf]
    if args.line_map:
        validator_cmd += ["--line-map", args.line_map]
    for p in args.strip_line_prefix:
        validator_cmd += ["--strip-line-prefix", p]
    validator_cmd += ["--report", str(out_dir / "validation_report.json")]

    subprocess.run(validator_cmd, check=True)
    print(json.dumps({"outputDir": str(out_dir), "chains": str(chains_path), "validation": str(out_dir / "validation_report.json")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
