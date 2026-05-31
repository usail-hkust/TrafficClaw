import os
import sys
from pathlib import Path
from typing import List


def resolve_config_path(config_arg: str, workspace_root: Path) -> Path:
    cleaned = os.path.expandvars(os.path.expanduser(str(config_arg).strip()))
    candidate = Path(cleaned)

    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        for base in (Path.cwd(), workspace_root):
            resolved = base / candidate
            if resolved not in candidates:
                candidates.append(resolved)

        root_candidate = Path("/") / candidate
        if root_candidate not in candidates:
            candidates.append(root_candidate)

    for path in candidates:
        if path.exists():
            return path.resolve()

    basename = candidate.name
    if not basename:
        return candidates[-1].resolve() if candidates else candidate.resolve()

    search_roots: List[Path] = []
    for env_var in ("DEEP_CITY_SCENARIOS_ROOT", "ZONE_SCENARIOS_ROOT"):
        env_val = os.environ.get(env_var)
        if env_val:
            search_roots.append(Path(env_val))

    search_roots.extend([
        workspace_root / "sumo_config",
        workspace_root / "sumo_config_highway",
        workspace_root / "Data",
    ])

    default_zone_root = Path("/data/zhouyuping/Zone/zone_scenarios")
    if default_zone_root.exists():
        search_roots.append(default_zone_root)

    matches: List[Path] = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(basename))

    matches = [match for match in matches if match.is_file()]
    if len(matches) == 1:
        last_candidate = candidates[-1] if candidates else candidate
        print(f"Config file not found at {last_candidate}; using {matches[0]}")
        return matches[0].resolve()
    if len(matches) > 1:
        print(f"Error: Config file not found: {candidate}")
        print("Found multiple matching config files:")
        for match in matches:
            print(f"  - {match}")
        sys.exit(1)

    return candidates[-1].resolve() if candidates else candidate.resolve()
