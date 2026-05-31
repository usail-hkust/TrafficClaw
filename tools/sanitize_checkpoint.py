#!/usr/bin/env python3
"""
Sanitize SUMO checkpoint snapshots by removing invalid <person> blocks.

This is useful when SUMO fails to load a checkpoint due to incomplete person state,
such as "Unknown lane '' when loading walk for person ...".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


PERSON_START_RE = re.compile(r"<person\b")
PERSON_END_RE = re.compile(r"</person>")
PERSON_ID_RE = re.compile(r'\bid="([^"]+)"')
PERSON_STATE_RE = re.compile(r'\bstate="([^"]*)"')


def _extract_attr(pattern: re.Pattern[str], line: str) -> Optional[str]:
    match = pattern.search(line)
    return match.group(1) if match else None


def _should_drop_person(
    line: str,
    min_state_tokens: int,
    drop_empty_lane_edge: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    state = _extract_attr(PERSON_STATE_RE, line)
    if state is not None:
        tokens = [token for token in state.split() if token]
        if len(tokens) < min_state_tokens:
            reasons.append(f"short state tokens={len(tokens)}")
    if drop_empty_lane_edge and (
        ' lane=""' in line or ' edge=""' in line or ' edges=""' in line
    ):
        reasons.append("empty lane/edge attribute")
    return (len(reasons) > 0, reasons)


def sanitize_checkpoint(
    input_path: Path,
    output_path: Path,
    min_state_tokens: int = 3,
    drop_empty_lane_edge: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    max_report: int = 20,
) -> Tuple[int, List[Tuple[str, str]]]:
    removed: List[Tuple[str, str]] = []
    total_removed = 0

    in_person = False
    person_lines: List[str] = []
    person_id: Optional[str] = None
    drop_person = False
    reasons: List[str] = []

    output_file = None
    if not dry_run:
        output_file = output_path.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8", errors="ignore") as input_file:
        for line in input_file:
            if not in_person:
                if PERSON_START_RE.search(line):
                    in_person = True
                    person_lines = [line]
                    person_id = _extract_attr(PERSON_ID_RE, line) or "unknown"
                    drop_person, reasons = _should_drop_person(
                        line, min_state_tokens, drop_empty_lane_edge
                    )

                    person_ended = PERSON_END_RE.search(line) or line.rstrip().endswith("/>")
                    if person_ended:
                        if drop_person:
                            total_removed += 1
                            removed.append((person_id, "; ".join(reasons)))
                        else:
                            if output_file:
                                output_file.write(line)
                        in_person = False
                        person_lines = []
                        person_id = None
                        drop_person = False
                        reasons = []
                    continue

                if output_file:
                    output_file.write(line)
                continue

            person_lines.append(line)
            if not drop_person and drop_empty_lane_edge and (
                ' lane=""' in line or ' edge=""' in line or ' edges=""' in line
            ):
                drop_person = True
                reasons.append("empty lane/edge attribute")

            if PERSON_END_RE.search(line):
                if drop_person:
                    total_removed += 1
                    removed.append((person_id or "unknown", "; ".join(reasons)))
                else:
                    if output_file:
                        output_file.write("".join(person_lines))
                in_person = False
                person_lines = []
                person_id = None
                drop_person = False
                reasons = []

    if in_person:
        if drop_person:
            total_removed += 1
            removed.append((person_id or "unknown", "; ".join(reasons)))
        else:
            if output_file:
                output_file.write("".join(person_lines))

    if output_file:
        output_file.close()

    if verbose:
        if removed:
            print(f"Removed {total_removed} person blocks from {input_path.name}")
            for person_id, reason in removed[:max_report]:
                print(f"  - {person_id}: {reason}")
            if len(removed) > max_report:
                print(f"  ... {len(removed) - max_report} more removed")
        else:
            print("No invalid person blocks found.")

    return total_removed, removed


def sanitize_checkpoint_in_place(
    path: Path,
    min_state_tokens: int = 3,
    drop_empty_lane_edge: bool = True,
    verbose: bool = True,
) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp_file:
        temp_path = Path(tmp_file.name)

    removed_count = 0
    try:
        removed_count, _ = sanitize_checkpoint(
            input_path=path,
            output_path=temp_path,
            min_state_tokens=min_state_tokens,
            drop_empty_lane_edge=drop_empty_lane_edge,
            dry_run=False,
            verbose=verbose,
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return removed_count


def _default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".xml"):
        base = input_path.name[:-4]
        return input_path.with_name(f"{base}.clean.xml")
    return input_path.with_name(f"{input_path.name}.clean")


def _copy_metadata(input_path: Path, output_path: Path) -> None:
    metadata_path = Path(os.path.splitext(str(input_path))[0] + "_metadata.json")
    if not metadata_path.exists():
        return
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["checkpoint_file"] = os.path.basename(output_path)
        out_metadata_path = Path(os.path.splitext(str(output_path))[0] + "_metadata.json")
        with out_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Warning: failed to copy metadata: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove invalid <person> blocks from a SUMO checkpoint snapshot."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to checkpoint .xml file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (default: <input>.clean.xml)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file in place",
    )
    parser.add_argument(
        "--min-state-tokens",
        type=int,
        default=3,
        help="Minimum token count required in person state (default: 3)",
    )
    parser.add_argument(
        "--keep-empty-lane-edge",
        action="store_true",
        help="Do not drop persons containing empty lane/edge attributes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not write output",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Maximum number of removed person IDs to print (default: 20)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    drop_empty_lane_edge = not args.keep_empty_lane_edge

    if args.in_place and not args.dry_run:
        removed_count = sanitize_checkpoint_in_place(
            path=input_path,
            min_state_tokens=args.min_state_tokens,
            drop_empty_lane_edge=drop_empty_lane_edge,
        )
        print(f"Sanitized in place: {input_path} (removed {removed_count})")
        return 0

    removed_count, _ = sanitize_checkpoint(
        input_path=input_path,
        output_path=output_path,
        min_state_tokens=args.min_state_tokens,
        drop_empty_lane_edge=drop_empty_lane_edge,
        dry_run=args.dry_run,
        verbose=True,
        max_report=args.max_report,
    )

    if not args.dry_run:
        _copy_metadata(input_path, output_path)
        print(f"Sanitized checkpoint written to: {output_path} (removed {removed_count})")
    else:
        print("Dry-run complete; no output written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
