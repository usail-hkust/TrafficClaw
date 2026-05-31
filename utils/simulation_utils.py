"""
Simulation utilities for SUMO environment setup and multi-modal transportation control.
Handles environment creation, signal timing control, traffic state collection, and more.
Supports: traffic signals, subway scheduling, bus scheduling, taxi scheduling, ramp metering, and highway speed limits.
"""

import os
import json
import traceback
import copy
import inspect
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import networkx as nx
from environment.sumo_env import SUMOEnv
from tqdm import tqdm
from utils.traffic_state_collector import TrafficStateCollector
from utils.checkpoint_logger import CheckpointLogger, extract_configs_only, effective_config

# Optional checkpoint sanitizer (removes invalid person states that break load-state)
try:
    from tools.sanitize_checkpoint import sanitize_checkpoint_in_place
except Exception:
    sanitize_checkpoint_in_place = None

# Add examples path to import SUMOEnv
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent

def _get_workspace_root() -> Path:
    """Get the workspace root directory."""
    return workspace_root


def _get_control_config_dir() -> Path:
    """Get the control_config directory path."""
    config_dir = _get_workspace_root() / "control_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _sanitize_checkpoint_snapshot(path: str, min_state_tokens: int = 3) -> None:
    """Sanitize checkpoint in-place to avoid load-state failures."""
    if sanitize_checkpoint_in_place is None:
        print("Warning: checkpoint sanitizer unavailable; skipping sanitize step.")
        return
    try:
        removed = sanitize_checkpoint_in_place(
            Path(path),
            min_state_tokens=min_state_tokens,
            drop_empty_lane_edge=True,
            verbose=False,
        )
        if removed:
            print(f"Sanitized checkpoint: removed {removed} invalid person blocks ({path})")
    except Exception as exc:
        print(f"Warning: Failed to sanitize checkpoint {path}: {exc}")


def load_signal_timing_config() -> Dict[str, Any]:
    """
    Load traffic signal timing configuration from signal_timing.json.
    If file doesn't exist, returns empty dict.

    Returns:
        Dictionary with intersection_id as keys and phase timing configs as values
        Format: {"intersection_id": {"phase_name": duration, ...}, ...}
    """
    config_file = _get_control_config_dir() / "signal_timing.json"

    if not config_file.exists():
        return {}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        if not isinstance(raw_config, dict):
            print("Warning: Signal timing config is not a dictionary; ignoring.")
            return {}
        # Normalize legacy format with {"timing": {...}} blocks
        normalized_config = {}
        for inter_id, timing in raw_config.items():
            if isinstance(timing, dict) and "timing" in timing and isinstance(timing["timing"], dict):
                normalized_config[inter_id] = timing["timing"]
            else:
                normalized_config[inter_id] = timing
        return normalized_config
    except Exception as exc:
        print(f"Warning: Failed to load signal timing config: {exc}")
        return {}


def save_signal_timing_config(config: Dict[str, Any]) -> bool:
    """
    Save traffic signal timing configuration to signal_timing.json.

    Args:
        config: Dictionary with intersection timing configurations

    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = _get_control_config_dir() / "signal_timing.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error saving signal timing config: {exc}")
        return False


def load_subway_scheduling_config() -> Dict[str, Any]:
    """
    Load subway scheduling configuration from subway_scheduling.json.
    If file doesn't exist, returns empty dict.

    Returns:
        Dictionary with route_id as keys and scheduling config as values
        Format: {"route_id": {"headway": 300, "schedule": [{"station_id": str, "dwell_time": int}]}, ...}
    """
    config_file = _get_control_config_dir() / "subway_scheduling.json"

    if not config_file.exists():
        return {}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: Failed to load subway scheduling config: {exc}")
        return {}


def save_subway_scheduling_config(config: Dict[str, Any]) -> bool:
    """
    Save subway scheduling configuration to subway_scheduling.json.

    Args:
        config: Dictionary with route scheduling configurations

    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = _get_control_config_dir() / "subway_scheduling.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error saving subway scheduling config: {exc}")
        return False


def load_bus_scheduling_config() -> Dict[str, Any]:
    """
    Load bus scheduling configuration from bus_scheduling.json.
    If file doesn't exist, returns empty dict.

    Returns:
        Dictionary with route_id as keys and scheduling config as values
        Format: {"route_id": {"headway": 180, "schedule": [{"station_id": str, "dwell_time": int}]}, ...}
    """
    config_file = _get_control_config_dir() / "bus_scheduling.json"

    if not config_file.exists():
        return {}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: Failed to load bus scheduling config: {exc}")
        return {}


def save_bus_scheduling_config(config: Dict[str, Any]) -> bool:
    """
    Save bus scheduling configuration to bus_scheduling.json.

    Args:
        config: Dictionary with route scheduling configurations

    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = _get_control_config_dir() / "bus_scheduling.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error saving bus scheduling config: {exc}")
        return False


def create_sumo_env(
    config_path: str,
    work_directory: Optional[str] = None,
    use_gui: bool = False,
    seed: Optional[int] = None,
    control_modules: Optional[List[str]] = None,
    run_counts: Optional[int] = None,
    config_dir_name: Optional[str] = None,
    load_state_path: Optional[str] = None,
    begin_time: Optional[float] = None,
    taxi_dispatch_algorithm: Optional[str] = None,
    taxi_idle_algorithm: Optional[str] = None,
    time_to_teleport: Optional[float] = None,
    vehicle_subscription_mode: Optional[str] = None,
    waiting_passenger_interval: Optional[float] = None,
    use_unique_work_dir: bool = False,
    taz_file_path: Optional[str] = None,
    use_simulated_taxi_system: bool = False,
) -> Tuple[SUMOEnv, Dict[str, Any]]:
    """
    Create and initialize a SUMO environment.

    Args:
        config_path: Path to SUMO config file (.sumocfg)
        work_directory: Working directory for SUMO files (default: temp directory)
        use_gui: Whether to use sumo-gui
        seed: Random seed for simulation
        control_modules: List of control module names to enable.
                        Available: 'signal_timing', 'subway_scheduling', 'bus_scheduling'
                        If None, defaults to ['signal_timing']
        run_counts: Number of simulation steps (default: 3600 if None).
                   Typically set to match checkpoint_interval for consistency.
        load_state_path: Optional path to checkpoint snapshot file to load.
                        If provided, will load checkpoint directly instead of starting fresh.
                        This avoids an extra SUMO restart.
        begin_time: Optional simulation clock (seconds) passed to SUMO ``--begin`` when **not**
                   using ``load_state_path``. Jumps the scenario timeline without restoring vehicles;
                   if None, simulation starts at time 0. When ``load_state_path`` is set, begin time
                   is taken from checkpoint metadata instead.
        taxi_dispatch_algorithm: Optional SUMO taxi dispatch algorithm (default: "traci").
        taxi_idle_algorithm: Optional SUMO taxi idle algorithm (default: "randomCircling").
        time_to_teleport: Optional SUMO time-to-teleport override. If None, read from sumocfg.
        vehicle_subscription_mode: Optional vehicle subscription mode ("all" or "departed").
        waiting_passenger_interval: Optional interval (seconds) for passenger waiting updates.
        use_unique_work_dir: If True, create a unique subdirectory for this simulation instance
                            to avoid log file conflicts when running multiple simulations.
        taz_file_path: Optional path to TAZ (Traffic Analysis Zone) file (districts.taz.xml).
                      If provided, zone infrastructure will be initialized for zone-based
                      infrastructure organization and queries.
        use_simulated_taxi_system: If True, bypass SUMO's native taxi device entirely.
                                  Removes taxi files from sumocfg, converts taxi vType to passenger,
                                  and skips taxi command-line options.

    Returns:
        Tuple of (SUMOEnv instance, environment configuration dict)
    """
    config_path_obj = Path(config_path).resolve()
    
    if not config_path_obj.exists():
        raise FileNotFoundError(f"SUMO config file not found: {config_path}")
    
    # Parse config file to get network and route files
    import xml.etree.ElementTree as ET
    tree = ET.parse(config_path_obj)
    root = tree.getroot()
    
    net_file = None
    route_files = []
    
    for input_elem in root.findall("input"):
        net_elem = input_elem.find("net-file")
        if net_elem is not None:
            net_file = net_elem.get("value")
        
        route_elem = input_elem.find("route-files")
        if route_elem is not None:
            route_files_str = route_elem.get("value", "")
            route_files = [f.strip() for f in route_files_str.split(",") if f.strip()]

    time_to_teleport_from_cfg = None
    for elem in root.iter("time-to-teleport"):
        val = elem.get("value")
        if val is None and elem.text:
            val = elem.text.strip()
        if val is None or val == "":
            continue
        try:
            time_to_teleport_from_cfg = float(val)
        except Exception:
            continue
        break
    
    if not net_file:
        raise ValueError("Network file not found in SUMO config")
    
    # Determine paths
    config_dir = config_path_obj.parent

    # Set work_directory to records directory instead of temp
    # If use_unique_work_dir is True, create a unique subdirectory to avoid log file conflicts
    if work_directory is None:
        workspace_root = _get_workspace_root()
        if use_unique_work_dir:
            # Create unique subdirectory for this simulation instance
            unique_id = str(uuid.uuid4())[:8]
            work_directory = str(workspace_root / "records" / "simulation_work" / f"policy_{unique_id}")
        else:
            work_directory = str(workspace_root / "records" / "simulation_work")
    
    work_dir_path = Path(work_directory)
    work_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Environment configuration
    # Note: net_file and route_files are relative to config_dir (PATH_TO_DATA)
    # Use run_counts parameter if provided, otherwise default to 3600
    if run_counts is None:
        run_counts = 3600
    
    dic_traffic_env_conf = {
        "ROADNET_FILE": net_file,  # Relative path from config_dir
        "TRAFFIC_FILE": route_files[0] if route_files else "routes.rou.xml",
        "SUMOCFG_FILE": config_path_obj.name,
        "INTERVAL": 1.0,
        "RUN_COUNTS": run_counts,
        "YELLOW_TIME": 5,
        "MIN_ACTION_TIME": 15,
        "NUM_INTERSECTIONS": 0,  # Will be set after env initialization
        "LIST_STATE_FEATURE": ["cur_phase", "time_this_phase", "lane_num_vehicle"],
        "DIC_REWARD_INFO": {"pressure": -0.25, "queue_length": -0.25},
        "TOP_K_ADJACENCY": 5,
    }

    # Simulated taxi system flag — bypasses SUMO's native taxi device entirely
    dic_traffic_env_conf["USE_SIMULATED_TAXI_SYSTEM"] = use_simulated_taxi_system

    # Taxi behavior knobs (used by SUMOEnv.reset to build sumo cmd)
    dic_traffic_env_conf["TAXI_DISPATCH_ALGORITHM"] = taxi_dispatch_algorithm or "traci"
    # Use randomCircling by default - "stop" can cause SUMO crashes when taxis have no destinations
    dic_traffic_env_conf["TAXI_IDLE_ALGORITHM"] = taxi_idle_algorithm or "randomCircling"
    if time_to_teleport is None:
        time_to_teleport = time_to_teleport_from_cfg
    dic_traffic_env_conf["TIME_TO_TELEPORT"] = 300 if time_to_teleport is None else float(time_to_teleport)
    if vehicle_subscription_mode:
        dic_traffic_env_conf["VEHICLE_SUBSCRIPTION_MODE"] = vehicle_subscription_mode
    else:
        dic_traffic_env_conf["VEHICLE_SUBSCRIPTION_MODE"] = "departed"
    if waiting_passenger_interval is not None:
        dic_traffic_env_conf["WAITING_PASSENGER_INTERVAL"] = float(waiting_passenger_interval)

    # TAZ file path for zone infrastructure
    # Auto-discover TAZ file if not provided
    if taz_file_path is None:
        candidate_taz_files = [
            config_dir / "districts.taz.xml",
            config_dir / "districts.aggregated.taz.xml",
            config_dir / "districts.filtered.taz.xml",
        ]
        for candidate in candidate_taz_files:
            if candidate.exists():
                taz_file_path = str(candidate)
                print(f"Auto-discovered TAZ file: {taz_file_path}")
                break

    if taz_file_path:
        dic_traffic_env_conf["TAZ_FILE_PATH"] = taz_file_path

    dic_path = {
        "PATH_TO_DATA": str(config_dir),  # This is where source files are located
        "PATH_TO_WORK_DIRECTORY": str(work_dir_path),  # This is where SUMO will run
    }
    
    # Create environment
    env = SUMOEnv(
        path_to_log=str(work_dir_path / "logs"),
        path_to_work_directory=str(work_dir_path),
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        control_modules=control_modules,
        config_dir_name=config_dir_name,  # Pass config_dir_name for control module initialization
        seed=seed  # Pass seed to ensure reproducibility from initialization
    )
    
    # Reset environment to initialize intersections
    # If load_state_path is provided, load checkpoint directly (avoids extra restart)
    if load_state_path:
        env.reset(use_gui=use_gui, seed=seed, load_state_path=load_state_path)
    elif begin_time is not None and float(begin_time) > 1e-6:
        env.reset(
            use_gui=use_gui,
            seed=seed,
            simulation_begin_time=float(begin_time),
        )
    else:
        env.reset(use_gui=use_gui, seed=seed)
    
    return env, dic_traffic_env_conf


def build_road_network_graphs(env: SUMOEnv) -> Dict[str, Any]:
    """
    DEPRECATED: Use env.get_road_network_graphs() instead.
    
    Build lane-interaction graph and intersection graph from SUMO environment.
    This is a backward-compatibility wrapper that calls the method on the env object.
    
    Args:
        env: SUMOEnv instance with initialized intersections
        
    Returns:
        Dictionary containing:
            - lane_inter_graph: DiGraph connecting lane_groups to intersections
            - intersection_graph: MultiDiGraph connecting intersections
            - lane_dict: Dictionary mapping lane_id to lane metadata
            - lane_graph: MultiDiGraph connecting individual lanes
            - lane_group_graph: MultiDiGraph connecting lane groups
            - road_graph: DiGraph connecting roads (edges) based on network topology
            - road_dict: Dictionary mapping road_id to road attributes (type, priority, numLanes, speed, from, to)
    """
    return env.get_road_network_graphs()


def _is_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def _jsonify(obj: Any) -> Any:
    """Convert nested structures into JSON-friendly types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    try:
        # Handle numpy scalars if present
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    # Fallback to string representation
    return str(obj)


def _build_checkpoint_extra_metadata(
    control_states: Optional[Dict[str, Dict[str, Any]]],
    control_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    control_modules: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract extra metadata for checkpoint files (e.g., control state/config snapshots)."""
    extra: Dict[str, Any] = {}

    if control_states:
        filtered_states = {
            key: value for key, value in control_states.items()
            if key not in {"_previous_checkpoint_path", "_skip_t_minus_1"}
        }
        if filtered_states:
            extra["control_states"] = _jsonify(filtered_states)
            taxi_state = filtered_states.get("taxi_scheduling")
            if taxi_state:
                extra["taxi_state"] = _jsonify(taxi_state)

    if control_configs:
        # Extract only configs (remove module instances) for JSON serialization
        configs_only = extract_configs_only(control_configs)
        extra["control_configs"] = _jsonify(configs_only)

    if control_modules:
        extra["control_modules"] = _jsonify(list(control_modules))

    return extra or None


# Import identifier generation functions from id_utils to avoid circular imports
from utils.id_utils import generate_file_prefix as _generate_file_prefix, generate_simulation_identifiers


def run_controlled_simulation(
    env: SUMOEnv,
    duration: float,
    step_seconds: int = 30,
    min_step_seconds: float = 1.0,
    traffic_state_collector: Optional[TrafficStateCollector] = None,
    checkpoint_interval: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    control_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    control_states: Optional[Dict[str, Dict[str, Any]]] = None,
    save_checkpoint: bool = True,
    is_first_simulation: bool = False,
    config_name: Optional[str] = None,
    llm_name: Optional[str] = None,
    simulation_id: Optional[str] = None,
    use_accelerated_stepping: bool = False,
    control_refresh_fn: Optional[Callable[[Any, float, Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    Run simulation with optional multi-modal transportation control.
    
    Control logic is applied only if control_configs is provided (i.e., after LLM planning).
    The first checkpoint interval runs without control to collect baseline data.
    
    Args:
        env: SUMOEnv instance
        duration: Total simulation duration in seconds (or remaining duration if resuming from checkpoint)
        step_seconds: Step size for simulation (default: 30)
        min_step_seconds: Minimum step size in seconds to reduce control loop frequency (default: 1)
        traffic_state_collector: Optional TrafficStateCollector instance for automatic traffic state collection
        checkpoint_interval: Optional checkpoint interval in seconds. If set, simulation will stop
                             at this interval and save a snapshot. Returns early with checkpoint info.
        checkpoint_dir: Directory to save checkpoint snapshots (default: records/checkpoints/)
        checkpoint_prefix: Prefix for checkpoint filenames (default: auto-generated from config_name, llm_name, control_modules)
        control_configs: Optional dictionary of control configurations by module
                        Format: {"signal_timing": {...}, "subway_scheduling": {...}, ...}
                        If None, simulation runs without control (baseline)
        control_states: Optional dictionary of control states by module (maintained across steps)
                       If None and control_configs is provided, will initialize states
                       Can also contain "_previous_checkpoint_path" key for t-1 state management
        save_checkpoint: Whether to save checkpoint snapshots and collect traffic states (default: True)
                        Set to False for policy testing to avoid saving checkpoints/states
        is_first_simulation: Whether this is the first simulation (default: False)
                            If True, always initializes new environment, never loads from checkpoint
                            This ensures the first simulation always starts fresh
        config_name: Config directory name (e.g., "jinan") for file naming
        llm_name: LLM model name (e.g., "deepseek-v3.2") for file naming
        simulation_id: Unique simulation identifier for checkpoint filename (e.g., "sim_20240101_120000")
        use_accelerated_stepping: If True, use EventScheduler-based sub-stepping for faster simulation
                                 with time-sensitive modules (signal_timing, ramp_metering).
                                 Events are pre-computed and simulation steps directly to event times.
                                 If False, use traditional per-second stepping (default: False for backward compatibility).
        control_refresh_fn: If set, called at the start of each control step with
            (env, current_time, control_configs, control_states). May mutate control_configs in place
            (e.g. update ramp_metering OPEN/CLOSE from feedback).
        
    Returns:
        Dictionary with simulation results. If checkpoint_interval is set and reached, includes:
        - checkpoint_reached: True
        - checkpoint_path: Path to saved snapshot
        - elapsed_time: Time elapsed in this checkpoint
        - remaining_duration: Remaining duration to simulate
        - traffic_state_stats: Statistics from traffic_state_collector if used
        - control_states: Updated control states (if control was applied)
        If the simulation aborts early, may include:
        - abort_checkpoint_path: Path to checkpoint saved at abort time (if saved)
        - abort_checkpoint_time: Simulation time when abort occurred
    """
    from control_modules import apply_control_logic

    # Determine if we should apply control logic
    # Always apply control if control_configs is provided OR if signal_timing/ramp_metering/taxi_scheduling is enabled in env
    apply_control = (control_configs is not None and len(control_configs) > 0) or \
                    ('signal_timing' in env.enabled_controls if hasattr(env, 'enabled_controls') else False) or \
                    ('ramp_metering' in env.enabled_controls if hasattr(env, 'enabled_controls') else False) or \
                    ('taxi_scheduling' in env.enabled_controls if hasattr(env, 'enabled_controls') else False)
    
    if apply_control:
        if control_configs is None:
            control_configs = {}
        print(f"Running simulation with control: {list(control_configs.keys())}")
        # Initialize control states if not provided
        if control_states is None:
            control_states = {}
            print(f"Initializing control states for modules: {list(control_configs.keys())}")

        # Restore taxi scheduling state from checkpoint if available
        # This ensures initial_fleet_size and other critical state is preserved across checkpoint loads
        if "taxi_scheduling" in (control_configs or {}):
            from control_modules import get_control_module
            taxi_module = get_control_module("taxi_scheduling")
            if taxi_module and hasattr(taxi_module, "restore_control_state_from_checkpoint"):
                checkpoint_state = None
                # Try to get checkpoint state from various sources
                if hasattr(env, "checkpoint_taxi_state"):
                    checkpoint_state = env.checkpoint_taxi_state
                elif hasattr(env, "checkpoint_extra_metadata"):
                    extra = env.checkpoint_extra_metadata or {}
                    checkpoint_state = extra.get("taxi_state") or extra.get("control_states", {}).get("taxi_scheduling")

                if checkpoint_state:
                    print(f"[TaxiScheduling] Restoring control state from checkpoint...")
                    control_states["taxi_scheduling"] = taxi_module.restore_control_state_from_checkpoint(
                        env=env,
                        control_state=control_states.get("taxi_scheduling"),
                        checkpoint_state=checkpoint_state
                    )

        # Initialize metrics for each enabled control module
        from control_modules import get_control_module
        module_metrics = {}
        for module_name in control_configs.keys():
            module = get_control_module(module_name)
            if module and hasattr(module, 'initialize_metrics'):
                module_metrics[module_name] = module.initialize_metrics()
                print(f"Initialized metrics for {module_name}")
    else:
        print("Running simulation without control (baseline)")
        module_metrics = {}
    # No phase initialization needed - control logic will be applied in the loop if needed
    
    start_time = env.get_current_time()
    target_end_time = start_time + duration
    
    # Mark first simulation
    if is_first_simulation:
        print(f"[First Simulation] Initializing new environment (not loading from checkpoint)")
    
    # Setup checkpoint directory if checkpoint_interval is set
    checkpoint_path = None
    checkpoint_path_t_minus_1 = None
    
    if checkpoint_interval is not None:
        if checkpoint_dir is None:
            workspace_root = _get_workspace_root()
            checkpoint_dir = str(workspace_root / "records" / "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if checkpoint_prefix is None:
            # Generate prefix from config_name, llm_name, and control_modules
            control_modules_list = list(control_configs.keys()) if control_configs else None
            file_prefix = _generate_file_prefix(
                config_name=config_name,
                llm_name=llm_name,
                control_modules=control_modules_list
            )
            checkpoint_prefix = f"{file_prefix}_checkpoint"
        else:
            # If checkpoint_prefix is provided, still add file prefix if config_name/llm_name/control_modules are available
            control_modules_list = list(control_configs.keys()) if control_configs else None
            file_prefix = _generate_file_prefix(
                config_name=config_name,
                llm_name=llm_name,
                control_modules=control_modules_list
            )
            if file_prefix != "default":
                checkpoint_prefix = f"{file_prefix}_{checkpoint_prefix}"
        
        # Add simulation_id to checkpoint prefix if provided
        if simulation_id:
            checkpoint_prefix = f"{checkpoint_prefix}_{simulation_id}"
        
        checkpoint_time = int(start_time + checkpoint_interval)
        checkpoint_filename = f"{checkpoint_prefix}_{checkpoint_time}.xml"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        # Path for t-1 state
        checkpoint_filename_t_minus_1 = f"{checkpoint_prefix}_{checkpoint_time}_t_minus_1.xml"
        checkpoint_path_t_minus_1 = os.path.join(checkpoint_dir, checkpoint_filename_t_minus_1)
        
        # If checkpoint is set and save_checkpoint is True, target end time should be checkpoint interval
        # Otherwise, run for full duration
        if save_checkpoint:
            target_end_time = start_time + checkpoint_interval
        else:
            target_end_time = start_time + duration
        
        # Save t-1 state: copy from previous checkpoint's t state if exists, otherwise save initial state
        # This should be done at the START of the checkpoint interval
        # Only save if save_checkpoint is True
        if save_checkpoint:
            skip_t_minus_1 = False
            if control_states and isinstance(control_states, dict):
                skip_t_minus_1 = bool(control_states.pop("_skip_t_minus_1", False))
            if skip_t_minus_1 and Path(checkpoint_path_t_minus_1).exists():
                print(f"Skipping t-1 checkpoint save (resume). Using existing: {checkpoint_path_t_minus_1}")
                # Keep existing t-1 checkpoint from the interval start
            else:
                try:
                    # Import modules needed for checkpoint operations
                    import shutil
                    import time

                    # Get previous_checkpoint_path from control_states if available
                    previous_checkpoint_path = None
                    if control_states and isinstance(control_states, dict):
                        previous_checkpoint_path = control_states.get("_previous_checkpoint_path")

                    if previous_checkpoint_path and Path(previous_checkpoint_path).exists():
                        # Copy previous checkpoint's t state as current t-1 state
                        # IMPORTANT: Also copy the metadata file (_metadata.json) which contains checkpoint_time
                        shutil.copy2(previous_checkpoint_path, checkpoint_path_t_minus_1)

                        # Copy metadata file if it exists (use _metadata.json format to match snapshot() method)
                        previous_metadata_path = os.path.splitext(previous_checkpoint_path)[0] + "_metadata.json"
                        t_minus_1_metadata_path = os.path.splitext(checkpoint_path_t_minus_1)[0] + "_metadata.json"
                        if Path(previous_metadata_path).exists():
                            shutil.copy2(previous_metadata_path, t_minus_1_metadata_path)
                            print(f"Copied metadata file: {t_minus_1_metadata_path}")
                        else:
                            # If metadata doesn't exist, create one from the current simulation time
                            # This ensures begin_time is available when loading the checkpoint
                            # Use same format as snapshot() method: checkpoint_time, checkpoint_file, timestamp
                            metadata = {
                                "checkpoint_time": float(start_time),
                                "checkpoint_file": os.path.basename(checkpoint_path_t_minus_1),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            try:
                                with open(t_minus_1_metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                print(
                                    f"Created metadata file for t-1 checkpoint: {t_minus_1_metadata_path} "
                                    f"(checkpoint_time: {start_time:.2f}s)"
                                )
                            except Exception as e:
                                print(f"Warning: Failed to create metadata file: {e}")

                        # Verify copy succeeded
                        time.sleep(0.1)  # Brief wait for file system sync
                        if Path(checkpoint_path_t_minus_1).exists():
                            print(f"Saved t-1 checkpoint (from previous t state) to: {checkpoint_path_t_minus_1}")
                            _sanitize_checkpoint_snapshot(checkpoint_path_t_minus_1)
                        else:
                            print(
                                f"ERROR: Failed to copy checkpoint file. Target file does not exist: "
                                f"{checkpoint_path_t_minus_1}"
                            )
                            raise FileNotFoundError(f"Checkpoint file copy failed: {checkpoint_path_t_minus_1}")
                    else:
                        # First checkpoint: save initial state as t-1
                        # Note: snapshot() automatically saves metadata (including checkpoint_time) for proper checkpoint resumption
                        extra_metadata = _build_checkpoint_extra_metadata(
                            control_states,
                            control_configs,
                            getattr(env, "control_modules", None),
                        )
                        saved_t_minus_1 = env.snapshot(
                            path=checkpoint_path_t_minus_1,
                            extra_metadata=extra_metadata
                        )
                        if saved_t_minus_1:
                            # Verify file actually exists after saving
                            time.sleep(0.1)  # Brief wait for file system sync
                            if Path(checkpoint_path_t_minus_1).exists():
                                print(
                                    f"Saved initial t-1 checkpoint at t={start_time:.0f}s to: {saved_t_minus_1}"
                                )
                                _sanitize_checkpoint_snapshot(checkpoint_path_t_minus_1)
                            else:
                                print(
                                    f"ERROR: env.snapshot() returned path but file does not exist: "
                                    f"{checkpoint_path_t_minus_1}"
                                )
                                print(
                                    "  This may indicate a SUMO TraCI error. Check SUMO logs for details."
                                )
                                raise FileNotFoundError(
                                    f"Checkpoint file was not created: {checkpoint_path_t_minus_1}"
                                )
                        else:
                            print(
                                "ERROR: env.snapshot() returned None. Simulation may not be running or SUMO error occurred."
                            )
                            raise RuntimeError(
                                "Failed to save t-1 checkpoint: env.snapshot() returned None"
                            )
                except Exception as e:
                    print(f"ERROR: Failed to save t-1 checkpoint: {e}")
                    traceback.print_exc()
                    raise  # Re-raise to stop execution if checkpoint save fails
        else:
            print(f"Skipping t-1 checkpoint save (save_checkpoint=False)")
    else:
        # No checkpoint, run for full duration (target_end_time already set)
        pass
    
    step_count = 0
    done = False
    
    # Track checkpoint start time for scheduled control actions
    checkpoint_start_time = start_time
    
    # Track travel times for checkpoint-specific calculation
    # Get initial arrived vehicle travel times to calculate checkpoint-specific avg travel time
    checkpoint_start_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
    checkpoint_start_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
    
    # Normalize step bounds
    if min_step_seconds is None:
        min_step_seconds = 1.0
    min_step_seconds = max(1.0, float(min_step_seconds))
    if min_step_seconds > step_seconds:
        min_step_seconds = float(step_seconds)

    def _traci_ready() -> bool:
        """Return True if TraCI connection appears healthy and usable."""
        if hasattr(env, "is_traci_healthy"):
            return env.is_traci_healthy()
        return getattr(env, "traci_conn", None) is not None

    # ========================================================================
    # ACCELERATED STEPPING MODE
    # ========================================================================
    # When use_accelerated_stepping=True, we use EventScheduler to pre-compute
    # all control events within a macro-step window, then execute sub-steps
    # only at event times. This significantly reduces the number of iterations
    # for time-sensitive modules (signal_timing, ramp_metering).
    #
    # Performance improvement: 3-6x faster for large networks with many intersections
    # ========================================================================

    if use_accelerated_stepping:
        from control_modules.shared.event_scheduler import EventScheduler
        from control_modules import get_control_module

        event_scheduler = EventScheduler()

        # Define action applicators for event-based control
        def apply_signal_action(inter_id: str, action_idx: int):
            """Apply signal phase change action."""
            if inter_id in env.intersection_dict:
                env.intersection_dict[inter_id].set_signal(action_idx, action_pattern="set")

        def apply_ramp_action(ramp_id: str, is_open: bool):
            """Apply ramp state change action."""
            if ramp_id in env.ramp_dict:
                env.ramp_dict[ramp_id].set_ramp_state(is_open)

        action_applicators = {
            "signal_timing": apply_signal_action,
            "ramp_metering": apply_ramp_action,
        }

        # Accelerated simulation loop
        with tqdm(total=target_end_time, desc="Simulation Progress (Accelerated)", unit="s", initial=env.get_current_time()) as pbar:
            while env.get_current_time() < target_end_time and not done:
                current_time = env.get_current_time()

                if control_refresh_fn is not None and control_configs is not None:
                    try:
                        control_refresh_fn(env, current_time, control_configs, control_states)
                    except Exception as e:
                        print(f"Warning: control_refresh_fn failed: {e}")
                        traceback.print_exc()

                # Check TraCI health
                if hasattr(env, 'is_traci_healthy') and not env.is_traci_healthy():
                    print("Warning: TraCI connection is unhealthy. Terminating simulation gracefully.")
                    done = True
                    break

                # Calculate macro step window
                remaining_to_target = target_end_time - current_time
                macro_step = min(step_seconds, remaining_to_target)
                step_end_time = current_time + macro_step

                # === Phase 1: Pre-schedule time-sensitive events ===
                event_scheduler.clear()

                # Get configs for time-sensitive modules
                signal_timing_config = None
                ramp_metering_config = None
                taxi_scheduling_config = None

                if hasattr(env, 'enabled_controls'):
                    if 'signal_timing' in env.enabled_controls:
                        signal_timing_module_info = env.enabled_controls['signal_timing']
                        raw = control_configs.get('signal_timing', {}) if control_configs and 'signal_timing' in control_configs else signal_timing_module_info.get('config', {})
                        signal_timing_config = effective_config(raw)

                    if 'ramp_metering' in env.enabled_controls:
                        ramp_metering_module_info = env.enabled_controls['ramp_metering']
                        raw = control_configs.get('ramp_metering', {}) if control_configs and 'ramp_metering' in control_configs else ramp_metering_module_info.get('config', {})
                        ramp_metering_config = effective_config(raw)

                    if 'taxi_scheduling' in env.enabled_controls:
                        taxi_scheduling_module_info = env.enabled_controls['taxi_scheduling']
                        raw = control_configs.get('taxi_scheduling', {}) if control_configs and 'taxi_scheduling' in control_configs else taxi_scheduling_module_info.get('config', {})
                        taxi_scheduling_config = effective_config(raw)

                # Schedule signal_timing events
                if signal_timing_config:
                    signal_module = get_control_module('signal_timing')
                    if signal_module and hasattr(signal_module, 'schedule_events'):
                        signal_events, updated_state = signal_module.schedule_events(
                            env=env,
                            config=signal_timing_config,
                            start_time=current_time,
                            end_time=step_end_time,
                            control_state=control_states.get('signal_timing') if control_states else None
                        )
                        event_scheduler.add_events(signal_events)
                        if control_states is None:
                            control_states = {}
                        control_states['signal_timing'] = updated_state

                # Schedule ramp_metering events
                if ramp_metering_config:
                    ramp_module = get_control_module('ramp_metering')
                    if ramp_module and hasattr(ramp_module, 'schedule_events'):
                        ramp_events, updated_state = ramp_module.schedule_events(
                            env=env,
                            config=ramp_metering_config,
                            start_time=current_time,
                            end_time=step_end_time,
                            control_state=control_states.get('ramp_metering') if control_states else None
                        )
                        event_scheduler.add_events(ramp_events)
                        if control_states is None:
                            control_states = {}
                        control_states['ramp_metering'] = updated_state

                # === Phase 2: Handle non-time-sensitive modules ===
                modules_to_control = {}
                if control_configs:
                    modules_to_control.update(control_configs)
                if signal_timing_config and 'signal_timing' not in modules_to_control:
                    modules_to_control['signal_timing'] = signal_timing_config
                if ramp_metering_config and 'ramp_metering' not in modules_to_control:
                    modules_to_control['ramp_metering'] = ramp_metering_config
                if taxi_scheduling_config and 'taxi_scheduling' not in modules_to_control:
                    modules_to_control['taxi_scheduling'] = taxi_scheduling_config

                module_results = {}
                for module_name in ['highway_speed_limit', 'bus_scheduling', 'subway_scheduling', 'taxi_scheduling']:
                    if module_name in modules_to_control:
                        try:
                            control_result = apply_control_logic(
                                module_name=module_name,
                                env=env,
                                config=effective_config(modules_to_control[module_name]),
                                current_time=current_time,
                                control_state=control_states.get(module_name) if control_states else None,
                                checkpoint_start_time=checkpoint_start_time
                            )
                            module_results[module_name] = control_result

                            # Apply non-time-sensitive actions immediately
                            if module_name == 'highway_speed_limit' and 'actions' in control_result:
                                for highway_id, speed_limit in control_result['actions'].items():
                                    if highway_id in env.highway_dict:
                                        env.highway_dict[highway_id].set_segment_speed_limit(speed_limit, unit="mph")

                            if 'control_state' in control_result:
                                if control_states is None:
                                    control_states = {}
                                control_states[module_name] = control_result['control_state']
                        except Exception as e:
                            print(f"Warning: Control logic for {module_name} failed: {e}")
                            traceback.print_exc()

                # === Phase 3: Execute sub-stepping loop ===
                unique_times = event_scheduler.get_unique_times(current_time, step_end_time)
                # Always step to end of macro step if no events or last event is before end
                if not unique_times or unique_times[-1] < step_end_time:
                    unique_times.append(step_end_time)

                # Collect traffic state before stepping
                if traffic_state_collector is not None and save_checkpoint:
                    if not _traci_ready():
                        print("Warning: TraCI connection lost before traffic state collection. Terminating simulation gracefully.")
                        done = True
                        break
                    traffic_state_collector.collect(current_time)
                    if not _traci_ready():
                        print("Warning: TraCI connection lost during traffic state collection. Terminating simulation gracefully.")
                        done = True
                        break

                substep_start_time = current_time
                for target_time in unique_times:
                    if target_time > env.get_current_time():
                        if not _traci_ready():
                            print("Warning: TraCI connection lost before simulationStep. Terminating simulation gracefully.")
                            done = True
                            break
                        traci_conn = getattr(env, "traci_conn", None)
                        if traci_conn is None:
                            print("Warning: TraCI connection missing before simulationStep. Terminating simulation gracefully.")
                            done = True
                            break
                        try:
                            # Use traci simulationStep for precise time control
                            traci_conn.simulationStep(target_time)
                        except Exception as e:
                            print(f"TraCI error during substep to {target_time}: {e}")
                            if hasattr(env, "mark_traci_unhealthy"):
                                env.mark_traci_unhealthy()
                            done = True
                            break
                        if not _traci_ready():
                            print("Warning: TraCI connection lost after simulationStep. Terminating simulation gracefully.")
                            done = True
                            break

                    # Execute events at this time
                    pending_events = event_scheduler.pop_events_until(target_time)
                    for event in pending_events:
                        if event.module in action_applicators:
                            action_applicators[event.module](event.entity_id, event.action)

                if done:
                    break

                # === Phase 4: Update system states ===
                try:
                    env._update_system_states()
                    env._update_waiting_vehicles(macro_step)
                    env._update_waiting_passengers()
                except RuntimeError as e:
                    if env._simulation_running:
                        env.close()
                    done = True
                    break

                # === Phase 5: Update control states and metrics ===
                current_time_after = env.get_current_time()

                # Update signal_timing control state
                if signal_timing_config and 'signal_timing' in modules_to_control:
                    signal_module = get_control_module('signal_timing')
                    if signal_module and hasattr(signal_module, 'update_control_state'):
                        cycle_dict = {inter_id: sum(cfg.values()) for inter_id, cfg in signal_timing_config.items()}
                        selected_intersections = []  # Events were already executed
                        if control_states is None:
                            control_states = {}
                        control_states['signal_timing'] = signal_module.update_control_state(
                            control_states.get('signal_timing'),
                            cycle_dict,
                            macro_step,
                            selected_intersections,
                            current_time_after
                        )

                # Update ramp_metering control state
                if ramp_metering_config and 'ramp_metering' in modules_to_control:
                    ramp_module = get_control_module('ramp_metering')
                    if ramp_module and hasattr(ramp_module, 'update_control_state'):
                        if control_states is None:
                            control_states = {}
                        control_states['ramp_metering'] = ramp_module.update_control_state(
                            control_states.get('ramp_metering'),
                            step_duration=macro_step,
                            env=env,
                            current_time=current_time_after,
                            applied_actions={},
                            config=ramp_metering_config
                        )

                # Update metrics for control modules
                if apply_control and module_metrics:
                    from control_modules import get_control_module as get_mod
                    for module_name in (control_configs or {}).keys():
                        if module_name in module_metrics:
                            module = get_mod(module_name)
                            if module and hasattr(module, 'update_metrics'):
                                module.update_metrics(
                                    metrics=module_metrics[module_name],
                                    env=env,
                                    reward=None,
                                    step_duration=macro_step
                                )

                step_count += 1
                pbar.update(macro_step)

                # Check if checkpoint interval reached
                if checkpoint_interval is not None:
                    elapsed_time = current_time_after - start_time
                    if elapsed_time >= checkpoint_interval:
                        # Save checkpoint snapshot (t state - current simulation end state)
                        print(f"Checkpoint reached at t={current_time_after:.0f}s (elapsed: {elapsed_time:.0f}s)")
                        if save_checkpoint:
                            try:
                                # Save current state as t state (checkpoint interval end state)
                                extra_metadata = _build_checkpoint_extra_metadata(
                                    control_states,
                                    control_configs,
                                    getattr(env, "control_modules", None),
                                )
                                saved_path = env.snapshot(
                                    path=checkpoint_path,
                                    extra_metadata=extra_metadata
                                )
                                if saved_path:
                                    if Path(checkpoint_path).exists():
                                        print(f"Checkpoint saved (t state) to: {saved_path}")
                                        _sanitize_checkpoint_snapshot(checkpoint_path)
                                        if Path(checkpoint_path_t_minus_1).exists():
                                            print(f"Checkpoint t-1 state available at: {checkpoint_path_t_minus_1}")
                                        else:
                                            print(f"WARNING: Checkpoint t-1 state NOT found at: {checkpoint_path_t_minus_1}")
                                            print(f"  Policy simulation will use t state checkpoint instead.")
                                    else:
                                        print(f"ERROR: Checkpoint file was not created despite snapshot() returning path: {checkpoint_path}")
                                        raise FileNotFoundError(f"Checkpoint file was not created: {checkpoint_path}")

                                    remaining_duration = duration - elapsed_time

                                    # Store current checkpoint path in control_states for next iteration's t-1
                                    if control_states is None:
                                        control_states = {}
                                    control_states["_previous_checkpoint_path"] = saved_path

                                    # Calculate metrics for enabled control modules at checkpoint
                                    accel_module_results = {}
                                    if apply_control and module_metrics:
                                        from control_modules import get_control_module as get_mod_ckpt
                                        for mod_name in control_configs.keys():
                                            if mod_name in module_metrics:
                                                mod = get_mod_ckpt(mod_name)
                                                if mod and hasattr(mod, 'calculate_final_results'):
                                                    calc_sig = inspect.signature(mod.calculate_final_results)
                                                    if "control_state" in calc_sig.parameters:
                                                        accel_module_results[mod_name] = mod.calculate_final_results(
                                                            metrics=module_metrics[mod_name],
                                                            env=env,
                                                            control_state=(control_states or {}).get(mod_name)
                                                        )
                                                    else:
                                                        accel_module_results[mod_name] = mod.calculate_final_results(
                                                            metrics=module_metrics[mod_name],
                                                            env=env
                                                        )

                                    # Calculate checkpoint-specific average travel time
                                    checkpoint_end_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
                                    checkpoint_end_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
                                    checkpoint_arrived_tt_sum = checkpoint_end_arrived_tt_sum - checkpoint_start_arrived_tt_sum
                                    checkpoint_arrived_count = checkpoint_end_arrived_count - checkpoint_start_arrived_count
                                    checkpoint_avg_travel_time = (checkpoint_arrived_tt_sum / checkpoint_arrived_count) if checkpoint_arrived_count > 0 else 0.0

                                    # Calculate checkpoint-specific average waiting time
                                    checkpoint_avg_waiting_time = env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0
                                    checkpoint_waiting_count = env.get_waiting_vehicle_count() if hasattr(env, 'get_waiting_vehicle_count') else 0

                                    # Safely get statistics
                                    try:
                                        total_departed = env.traci_conn.simulation.getDepartedNumber() if env.traci_conn else 0
                                        total_arrived = env.traci_conn.simulation.getArrivedNumber() if env.traci_conn else 0
                                    except (AttributeError, Exception):
                                        total_departed = 0
                                        total_arrived = 0

                                    accel_checkpoint_results = {
                                        "checkpoint_reached": True,
                                        "checkpoint_path": saved_path,
                                        "checkpoint_path_t_minus_1": checkpoint_path_t_minus_1 if Path(checkpoint_path_t_minus_1).exists() else None,
                                        "elapsed_time": elapsed_time,
                                        "remaining_duration": max(0, remaining_duration),
                                        "current_time": current_time_after,
                                        "step_count": step_count,
                                        "total_departed": total_departed,
                                        "total_arrived": total_arrived,
                                        "avg_travel_time": env.get_average_travel_time(),
                                        "checkpoint_avg_travel_time": checkpoint_avg_travel_time,
                                        "checkpoint_arrived_count": checkpoint_arrived_count,
                                        "avg_waiting_time": env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0,
                                        "checkpoint_avg_waiting_time": checkpoint_avg_waiting_time,
                                        "checkpoint_waiting_count": checkpoint_waiting_count,
                                        "control_states": control_states if apply_control else None
                                    }

                                    if accel_module_results:
                                        accel_checkpoint_results["module_metrics"] = accel_module_results

                                    # Save accumulated traffic state snapshots at checkpoint
                                    if traffic_state_collector is not None:
                                        traffic_state_collector.save_checkpoint_snapshots(checkpoint_time=current_time_after)
                                        accel_checkpoint_results["traffic_state_stats"] = traffic_state_collector.get_stats()

                                    return accel_checkpoint_results
                                else:
                                    print(f"ERROR: Failed to save checkpoint snapshot. env.snapshot() returned None.")
                                    raise RuntimeError(f"Failed to save checkpoint snapshot: {checkpoint_path}")
                            except Exception as e:
                                print(f"Error saving checkpoint: {e}")
                                traceback.print_exc()
                                # Checkpoint save failed, return early with checkpoint_reached=False
                                return {
                                    "success": False,
                                    "checkpoint_reached": False,
                                    "error": f"Failed to save checkpoint: {str(e)}",
                                    "current_time": current_time,
                                    "final_time": current_time,
                                }
                        else:
                            # Checkpoint interval reached but save_checkpoint=False, continue running
                            print(f"Checkpoint interval reached but save_checkpoint=False, continuing simulation...")
                            target_end_time = start_time + duration

        # End of accelerated stepping mode
        # Fall through to final statistics collection below

    # ========================================================================
    # TRADITIONAL STEPPING MODE (default - backward compatible)
    # ========================================================================
    # This is the original stepping logic that runs when use_accelerated_stepping=False
    # or after accelerated stepping completes for final statistics

    if not use_accelerated_stepping:
        # Initialize progress bar
        with tqdm(total=target_end_time, desc="Simulation Progress", unit="s", initial=env.get_current_time()) as pbar:
            # Main simulation loop
            while env.get_current_time() < target_end_time and not done:
                current_time = env.get_current_time()

                if control_refresh_fn is not None and control_configs is not None:
                    try:
                        control_refresh_fn(env, current_time, control_configs, control_states)
                    except Exception as e:
                        print(f"Warning: control_refresh_fn failed: {e}")
                        traceback.print_exc()

                # Check TraCI health status - terminate gracefully if connection is corrupted
                if hasattr(env, 'is_traci_healthy') and not env.is_traci_healthy():
                    print("Warning: TraCI connection is unhealthy. Terminating simulation gracefully.")
                    done = True
                    break
            
                # Apply control logic if configured
                # Step 1: Call apply_control for all modules to get their min_remaining and actions
                action_dict = {}
                min_remaining = float('inf')
                selected_modules = ['signal_timing']  # Modules with minimum min_remaining (can be multiple)
                module_results = {}  # Store results from all modules
            
                # Always apply signal_timing control if it's enabled in env (even if not in control_modules)
                # This ensures signal lights are always controlled, but metrics won't be collected for LLM agent
                signal_timing_config = None
                if 'signal_timing' in env.enabled_controls:
                    signal_timing_module_info = env.enabled_controls['signal_timing']
                    raw = control_configs.get('signal_timing', {}) if control_configs and 'signal_timing' in control_configs else signal_timing_module_info.get('config', {})
                    signal_timing_config = effective_config(raw)

                # Always apply ramp_metering control if it's enabled in env (even if not in control_configs)
                # This ensures ramp metering is always controlled, but metrics won't be collected for LLM agent
                ramp_metering_config = None
                if 'ramp_metering' in env.enabled_controls:
                    ramp_metering_module_info = env.enabled_controls['ramp_metering']
                    raw = control_configs.get('ramp_metering', {}) if control_configs and 'ramp_metering' in control_configs else ramp_metering_module_info.get('config', {})
                    ramp_metering_config = effective_config(raw)

                # Always apply taxi_scheduling control if it's enabled in env (even if not in control_configs)
                # This ensures taxi scheduling cleanup runs to prevent stale reservations (SUMO crash guard)
                taxi_scheduling_config = None
                if 'taxi_scheduling' in env.enabled_controls:
                    taxi_scheduling_module_info = env.enabled_controls['taxi_scheduling']
                    raw = control_configs.get('taxi_scheduling', {}) if control_configs and 'taxi_scheduling' in control_configs else taxi_scheduling_module_info.get('config', {})
                    taxi_scheduling_config = effective_config(raw)
            
                # Determine which modules to apply control for
                # Always include signal_timing, ramp_metering, and taxi_scheduling if enabled in env, plus modules in control_configs
                modules_to_control = {}
                if control_configs:
                    modules_to_control.update(control_configs)
                # Add signal_timing if it's enabled in env but not in control_configs
                if signal_timing_config and 'signal_timing' not in modules_to_control:
                    modules_to_control['signal_timing'] = signal_timing_config
                # Add ramp_metering if it's enabled in env but not in control_configs
                if ramp_metering_config and 'ramp_metering' not in modules_to_control:
                    modules_to_control['ramp_metering'] = ramp_metering_config
                # Add taxi_scheduling if it's enabled in env but not in control_configs
                if taxi_scheduling_config and 'taxi_scheduling' not in modules_to_control:
                    modules_to_control['taxi_scheduling'] = taxi_scheduling_config
            
                if modules_to_control:
                    # First pass: Get min_remaining from all modules (including signal_timing if enabled)
                    for module_name, module_config in modules_to_control.items():
                        try:
                            control_result = apply_control_logic(
                                module_name=module_name,
                                env=env,
                                config=effective_config(module_config),
                                current_time=current_time,
                                control_state=control_states.get(module_name),
                                checkpoint_start_time=checkpoint_start_time
                            )
                        
                            # Store result for later use
                            module_results[module_name] = control_result
                            module_min_remaining = control_result.get(
                                "min_remaining",
                                control_result.get("next_action_time", step_seconds)
                            )
                            min_remaining = min(min_remaining, module_min_remaining)
                        
                        except Exception as e:
                            print(f"Warning: Control logic for {module_name} failed: {e}")
                            traceback.print_exc()
                
                    for selected_module in module_results.keys():
                        if selected_module in module_results:
                            control_result = module_results[selected_module]
                        
                            # Map module_name to control_type for env.step()
                            if selected_module == "signal_timing":
                                control_type = "signal_timing"
                                # Add switch actions to action_dict (for signal timing)
                                if "actions" in control_result:
                                    if control_type not in action_dict:
                                        action_dict[control_type] = {}
                                    action_dict[control_type].update(control_result["actions"])
                        
                            elif selected_module == "subway_scheduling":
                                control_type = "subway_scheduling"
                                # Add dispatch actions to action_dict
                                if "dispatch_actions" in control_result:
                                    if control_type not in action_dict:
                                        action_dict[control_type] = {}
                                    action_dict[control_type]["dispatch_actions"] = control_result["dispatch_actions"]
                            
                                # Update minimum remaining time (similar to signal_timing)
                                if "next_dispatch_time" in control_result:
                                    min_remaining = min(min_remaining, control_result["next_dispatch_time"])
                        
                            elif selected_module == "highway_speed_limit":
                                control_type = "highway_speed_limit"
                                # Add speed limit actions to action_dict
                                if "actions" in control_result:
                                    if control_type not in action_dict:
                                        action_dict[control_type] = {}
                                    # actions format: {highway_id: speed_limit_mph}
                                    # Config format is already {highway_id: speed_limit_mph}
                                    action_dict[control_type].update(control_result["actions"])
                        
                            elif selected_module == "ramp_metering":
                                control_type = "ramp_metering"
                                # Add ramp metering actions to action_dict
                                if "actions" in control_result:
                                    if control_type not in action_dict:
                                        action_dict[control_type] = {}
                                    # actions format: {ramp_id: is_open (bool)}
                                    action_dict[control_type].update(control_result["actions"])

                            elif selected_module == "taxi_scheduling":
                                # Taxi scheduling applies actions directly via TraCI inside the module.
                                # Keep module_results for metrics/state updates, no env.step actions needed.
                                pass
                        
                            elif selected_module == "bus_scheduling":
                                control_type = "bus_scheduling"
                                # Add dispatch actions to action_dict
                                if "dispatch_actions" in control_result:
                                    if control_type not in action_dict:
                                        action_dict[control_type] = {}
                                    action_dict[control_type]["dispatch_actions"] = control_result["dispatch_actions"]
                            
                                # Update minimum remaining time (similar to subway_scheduling)
                                if "next_dispatch_time" in control_result:
                                    min_remaining = min(min_remaining, control_result["next_dispatch_time"])
                
                # Step 3: Update control states for all modules (store results, will be updated after step)
                # Only update control states for modules that are in control_configs (for LLM agent tracking)
                # But signal_timing and ramp_metering states are still updated if they're enabled in env
                for module_name, control_result in module_results.items():
                    # Always update signal_timing state if it's enabled in env
                    if module_name == 'signal_timing' and 'signal_timing' in env.enabled_controls:
                        if control_states is None:
                            control_states = {}
                        control_states[module_name] = control_result["control_state"]
                    # Always update ramp_metering state if it's enabled in env
                    elif module_name == 'ramp_metering' and 'ramp_metering' in env.enabled_controls:
                        if control_states is None:
                            control_states = {}
                        control_states[module_name] = control_result["control_state"]
                    # Update other modules only if they're in control_configs (for LLM agent)
                    elif control_configs and module_name in control_configs:
                        control_states[module_name] = control_result["control_state"]
            
                remaining_cap = min(step_seconds, target_end_time - current_time)
                min_remaining = min(min_remaining, remaining_cap)
                step_floor = min(min_step_seconds, remaining_cap)
                min_remaining = max(step_floor, min_remaining)
            
                # Collect traffic states BEFORE step (if collector is provided and save_checkpoint is True)
                # This ensures we collect state at the current time before advancing simulation
                if traffic_state_collector is not None and save_checkpoint:
                    if not _traci_ready():
                        print("Warning: TraCI connection lost before traffic state collection. Terminating simulation gracefully.")
                        done = True
                        break
                    traffic_state_collector.collect(current_time)
                    if not _traci_ready():
                        print("Warning: TraCI connection lost during traffic state collection. Terminating simulation gracefully.")
                        done = True
                        break

                # Step 4: Execute step with action_dict format: {control_type: {id: action, ...}}
                # Always pass action_dict if we have any actions (including signal_timing)
                # If no control is applied, pass empty dict
                has_actions = len(action_dict) > 0
                if not _traci_ready():
                    print("Warning: TraCI connection lost before simulation step. Terminating simulation gracefully.")
                    done = True
                    break
                next_state, reward, done, info = env.step(action_dict if has_actions else {}, min_remaining)
                if not _traci_ready():
                    print("Warning: TraCI connection lost after simulation step. Terminating simulation gracefully.")
                    done = True
                    break
            
                # Step 5: Update control states after step (based on actual step duration)
                # All modules need to update their states because time has passed
                # Skip if simulation is done (SUMO connection may be closed)
                if apply_control and not done:
                    # Also update signal_timing, ramp_metering, and taxi_scheduling states if they're enabled in env (even if not in control_configs)
                    current_time_after_step = env.get_current_time()
                    if control_states is None:
                        control_states = {}
                    modules_to_update = {}
                    if control_configs:
                        modules_to_update.update(control_configs)
                    # Add signal_timing if it's enabled in env but not in control_configs
                    if signal_timing_config and 'signal_timing' not in modules_to_update:
                        modules_to_update['signal_timing'] = signal_timing_config
                    # Add ramp_metering if it's enabled in env but not in control_configs
                    if ramp_metering_config and 'ramp_metering' not in modules_to_update:
                        modules_to_update['ramp_metering'] = ramp_metering_config
                    # Add taxi_scheduling if it's enabled in env but not in control_configs
                    if taxi_scheduling_config and 'taxi_scheduling' not in modules_to_update:
                        modules_to_update['taxi_scheduling'] = taxi_scheduling_config
                
                    if modules_to_update:
                        for module_name, module_config in modules_to_update.items():
                            if module_name == "signal_timing":
                                # Update signal_timing control state after step
                                # Always update if signal_timing is enabled in env (for internal tracking)
                                # But only track in control_states if it's in control_configs (for LLM agent)
                                # Extract cycle_dict from config
                                cycle_dict = {inter_id: sum(cfg.values()) for inter_id, cfg in signal_timing_config.items()}
                                # Get selected intersections from the result
                                selected_intersections = []
                                if module_name in module_results:
                                    selected_intersections = module_results.get(module_name, {}).get("selected_intersections", [])
                            
                                # Get module instance and update state
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    # Always update state (for internal tracking)
                                    if control_states is None:
                                        control_states = {}
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name),
                                        cycle_dict,
                                        min_remaining,
                                        selected_intersections,
                                        current_time_after_step
                                    )
                                
                                    # Update enabled_controls state for consistency with env.enabled_controls
                                    if 'signal_timing' in env.enabled_controls:
                                        env.enabled_controls['signal_timing']['state'] = control_states[module_name]
                        
                            elif module_name == "subway_scheduling":
                                # Update subway scheduling control state after step
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name, {}),
                                        step_duration=min_remaining
                                    )
                        
                            elif module_name == "highway_speed_limit":
                                # Update highway speed limit control state after step
                                # Get applied actions from action_dict (actions that were actually executed)
                                applied_actions = {}
                                if "highway_speed_limit" in action_dict:
                                    applied_actions = action_dict["highway_speed_limit"].copy()
                            
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name, {}),
                                        step_duration=min_remaining,
                                        env=env,
                                        current_time=current_time_after_step,
                                        applied_actions=applied_actions
                                    )

                                    # Update enabled_controls state for consistency with env.enabled_controls
                                    if 'highway_speed_limit' in env.enabled_controls:
                                        env.enabled_controls['highway_speed_limit']['state'] = control_states[module_name]
                        
                            elif module_name == "ramp_metering":
                                # Update ramp metering control state after step
                                # Get applied actions from action_dict (actions that were actually executed)
                                applied_actions = {}
                                if "ramp_metering" in action_dict:
                                    applied_actions = action_dict["ramp_metering"].copy()
                            
                                # Get module instance and update state
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name),
                                        step_duration=min_remaining,
                                        env=env,
                                        current_time=current_time_after_step,
                                        applied_actions=applied_actions,
                                        config=module_config
                                    )
                                
                                    # Update enabled_controls state for consistency with env.enabled_controls
                                    if 'ramp_metering' in env.enabled_controls:
                                        env.enabled_controls['ramp_metering']['state'] = control_states[module_name]

                            elif module_name == "taxi_scheduling":
                                # Update taxi scheduling control state after step
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name, {}),
                                        step_duration=min_remaining
                                    )
                        
                            elif module_name == "bus_scheduling":
                                # Update bus scheduling control state after step
                                from control_modules import get_control_module
                                module = get_control_module(module_name)
                                if module and hasattr(module, 'update_control_state'):
                                    control_states[module_name] = module.update_control_state(
                                        control_states.get(module_name, {}),
                                        step_duration=min_remaining
                                    )
            
                # Update metrics for enabled control modules
                if apply_control and module_metrics:
                    from control_modules import get_control_module
                    for module_name in control_configs.keys():
                        if module_name in module_metrics:
                            module = get_control_module(module_name)
                            if module and hasattr(module, 'update_metrics'):
                                module.update_metrics(
                                    metrics=module_metrics[module_name],
                                    env=env,
                                    reward=reward if isinstance(reward, list) else [reward] if reward else None,
                                    step_duration=min_remaining
                                )
            
                step_count += 1
                current_time = env.get_current_time()
                
                # Update progress bar with step duration (incremental update)
                pbar.update(min_remaining)
            
                # Check if checkpoint interval reached
                if checkpoint_interval is not None:
                    elapsed_time = current_time - start_time
                    if elapsed_time >= checkpoint_interval:
                        # Save checkpoint snapshot (t state - current simulation end state)
                        print(f"Checkpoint reached at t={current_time:.0f}s (elapsed: {elapsed_time:.0f}s)")
                        if save_checkpoint:
                            try:
                                # Save current state as t state (checkpoint interval end state)
                                # Note: snapshot() automatically saves metadata (including checkpoint_time) for proper checkpoint resumption
                                extra_metadata = _build_checkpoint_extra_metadata(
                                    control_states,
                                    control_configs,
                                    getattr(env, "control_modules", None),
                                )
                                saved_path = env.snapshot(
                                    path=checkpoint_path,
                                    extra_metadata=extra_metadata
                                )
                                if saved_path:
                                    # snapshot() already verifies file exists, but double-check
                                    if Path(checkpoint_path).exists():
                                        print(f"Checkpoint saved (t state) to: {saved_path}")
                                        _sanitize_checkpoint_snapshot(checkpoint_path)
                                        if Path(checkpoint_path_t_minus_1).exists():
                                            print(f"Checkpoint t-1 state available at: {checkpoint_path_t_minus_1}")
                                        else:
                                            print(f"WARNING: Checkpoint t-1 state NOT found at: {checkpoint_path_t_minus_1}")
                                            print(f"  Policy simulation will use t state checkpoint instead.")
                                    else:
                                        print(f"ERROR: Checkpoint file was not created despite snapshot() returning path: {checkpoint_path}")
                                        raise FileNotFoundError(f"Checkpoint file was not created: {checkpoint_path}")
                                
                                    remaining_duration = duration - elapsed_time
                                
                                    # Store current checkpoint path in control_states for next iteration's t-1
                                    if control_states is None:
                                        control_states = {}
                                    control_states["_previous_checkpoint_path"] = saved_path
                                
                                    # Calculate metrics for enabled control modules at checkpoint
                                    module_results = {}
                                    if apply_control and module_metrics:
                                        from control_modules import get_control_module
                                        for module_name in control_configs.keys():
                                            if module_name in module_metrics:
                                                module = get_control_module(module_name)
                                                if module and hasattr(module, 'calculate_final_results'):
                                                    calc_sig = inspect.signature(module.calculate_final_results)
                                                    if "control_state" in calc_sig.parameters:
                                                        module_results[module_name] = module.calculate_final_results(
                                                            metrics=module_metrics[module_name],
                                                            env=env,
                                                            control_state=(control_states or {}).get(module_name)
                                                        )
                                                    else:
                                                        module_results[module_name] = module.calculate_final_results(
                                                            metrics=module_metrics[module_name],
                                                            env=env
                                                        )
                                
                                    # Calculate checkpoint-specific average travel time
                                    # Only count vehicles that arrived during this checkpoint interval
                                    checkpoint_end_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
                                    checkpoint_end_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
                                    checkpoint_arrived_tt_sum = checkpoint_end_arrived_tt_sum - checkpoint_start_arrived_tt_sum
                                    checkpoint_arrived_count = checkpoint_end_arrived_count - checkpoint_start_arrived_count
                                    checkpoint_avg_travel_time = (checkpoint_arrived_tt_sum / checkpoint_arrived_count) if checkpoint_arrived_count > 0 else 0.0
                                
                                    # Calculate checkpoint-specific average waiting time
                                    # Get current average waiting time at checkpoint end (represents waiting time state during this checkpoint)
                                    checkpoint_avg_waiting_time = env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0
                                    checkpoint_waiting_count = env.get_waiting_vehicle_count() if hasattr(env, 'get_waiting_vehicle_count') else 0
                                
                                    # Reset checkpoint start state for next checkpoint
                                    checkpoint_start_arrived_tt_sum = checkpoint_end_arrived_tt_sum
                                    checkpoint_start_arrived_count = checkpoint_end_arrived_count
                                
                                    # Return early with checkpoint info
                                    # Safely get statistics (traci_conn may be None if simulation ended)
                                    try:
                                        total_departed = env.traci_conn.simulation.getDepartedNumber() if env.traci_conn else 0
                                        total_arrived = env.traci_conn.simulation.getArrivedNumber() if env.traci_conn else 0
                                    except (AttributeError, Exception):
                                        total_departed = 0
                                        total_arrived = 0
                                
                                    checkpoint_results = {
                                        "checkpoint_reached": True,
                                        "checkpoint_path": saved_path,  # t state
                                        "checkpoint_path_t_minus_1": checkpoint_path_t_minus_1 if Path(checkpoint_path_t_minus_1).exists() else None,  # t-1 state
                                        "elapsed_time": elapsed_time,
                                        "remaining_duration": max(0, remaining_duration),
                                        "current_time": current_time,
                                        "step_count": step_count,
                                        "total_departed": total_departed,
                                        "total_arrived": total_arrived,
                                        "avg_travel_time": env.get_average_travel_time(),  # Global average (all vehicles from start)
                                        "checkpoint_avg_travel_time": checkpoint_avg_travel_time,  # Checkpoint-specific average
                                        "checkpoint_arrived_count": checkpoint_arrived_count,  # Number of vehicles arrived in this checkpoint
                                        "avg_waiting_time": env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0,  # Global average (all waiting vehicles)
                                        "checkpoint_avg_waiting_time": checkpoint_avg_waiting_time,  # Checkpoint-specific average waiting time
                                        "checkpoint_waiting_count": checkpoint_waiting_count,  # Number of waiting vehicles at checkpoint
                                        "control_states": control_states if apply_control else None
                                    }
                                
                                    # Add module metrics if available
                                    if module_results:
                                        checkpoint_results["module_metrics"] = module_results
                                
                                    # Save accumulated traffic state snapshots at checkpoint
                                    if traffic_state_collector is not None:
                                        traffic_state_collector.save_checkpoint_snapshots(checkpoint_time=current_time)
                                        checkpoint_results["traffic_state_stats"] = traffic_state_collector.get_stats()
                                
                                    return checkpoint_results
                                else:
                                    print(f"ERROR: Failed to save checkpoint snapshot. env.snapshot() returned None.")
                                    raise RuntimeError(f"Failed to save checkpoint snapshot: {checkpoint_path}")
                            except Exception as e:
                                print(f"Error saving checkpoint: {e}")
                                traceback.print_exc()
                                # Checkpoint save failed, return early with checkpoint_reached=False
                                return {
                                    "success": False,
                                    "checkpoint_reached": False,
                                    "error": f"Failed to save checkpoint: {str(e)}",
                                    "current_time": current_time,
                                    "final_time": current_time,
                                }
                        else:
                            # Checkpoint interval reached but save_checkpoint=False, continue running
                            print(f"Checkpoint interval reached but save_checkpoint=False, continuing simulation...")
                        # Reset target_end_time to continue until duration
                        target_end_time = start_time + duration
    
    # Collect final statistics
    final_time = env.get_current_time()
    ended_early = False
    abort_reason = None
    if target_end_time is not None and final_time + 1e-6 < target_end_time:
        ended_early = True
        if hasattr(env, "is_traci_healthy") and not env.is_traci_healthy():
            abort_reason = "traci_unhealthy"
        elif getattr(env, "traci_conn", None) is None:
            abort_reason = "traci_disconnected"
        elif not getattr(env, "_simulation_running", True):
            abort_reason = "sumo_stopped"
        else:
            abort_reason = "ended_early"
    
    # Safely get statistics (traci_conn may be None if simulation ended)
    try:
        total_departed = env.traci_conn.simulation.getDepartedNumber() if env.traci_conn else 0
        total_arrived = env.traci_conn.simulation.getArrivedNumber() if env.traci_conn else 0
    except (AttributeError, Exception):
        total_departed = 0
        total_arrived = 0
    
    avg_travel_time = env.get_average_travel_time()  # Global average (all vehicles from start)
    avg_waiting_time = env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0  # Global average (all waiting vehicles)
    
    # Calculate checkpoint-specific average travel time (if checkpoint tracking was enabled)
    checkpoint_avg_travel_time = None
    checkpoint_arrived_count = 0
    checkpoint_avg_waiting_time = None
    checkpoint_waiting_count = 0
    if checkpoint_interval is not None:
        checkpoint_end_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
        checkpoint_end_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
        checkpoint_arrived_tt_sum = checkpoint_end_arrived_tt_sum - checkpoint_start_arrived_tt_sum
        checkpoint_arrived_count = checkpoint_end_arrived_count - checkpoint_start_arrived_count
        checkpoint_avg_travel_time = (checkpoint_arrived_tt_sum / checkpoint_arrived_count) if checkpoint_arrived_count > 0 else 0.0
        
        # Calculate checkpoint-specific average waiting time
        checkpoint_avg_waiting_time = env.get_average_waiting_time() if hasattr(env, 'get_average_waiting_time') else 0.0
        checkpoint_waiting_count = env.get_waiting_vehicle_count() if hasattr(env, 'get_waiting_vehicle_count') else 0

    abort_checkpoint_path = None
    abort_checkpoint_time = None
    abort_checkpoint_saved = False
    if ended_early and save_checkpoint and checkpoint_interval is not None:
        abort_checkpoint_time = final_time
        try:
            abort_checkpoint_filename = f"{checkpoint_prefix}_abort_{int(abort_checkpoint_time)}.xml"
            abort_checkpoint_path = os.path.join(checkpoint_dir, abort_checkpoint_filename)
            extra_metadata = _build_checkpoint_extra_metadata(
                control_states,
                control_configs,
                getattr(env, "control_modules", None),
            )
            saved_abort_path = env.snapshot(
                path=abort_checkpoint_path,
                extra_metadata=extra_metadata
            )
            if saved_abort_path and Path(abort_checkpoint_path).exists():
                _sanitize_checkpoint_snapshot(abort_checkpoint_path)
                abort_checkpoint_path = saved_abort_path
                abort_checkpoint_saved = True
                print(
                    f"Saved abort checkpoint at t={abort_checkpoint_time:.0f}s to: {abort_checkpoint_path}"
                )
            else:
                print("Warning: Failed to save abort checkpoint snapshot (env.snapshot returned None).")
                abort_checkpoint_path = None
        except Exception as e:
            print(f"Warning: Failed to save abort checkpoint at t={abort_checkpoint_time:.0f}s: {e}")
            abort_checkpoint_path = None
    
    results = {
        "checkpoint_reached": False,
        "final_time": final_time,
        "start_time": start_time,
        "duration": final_time - start_time,
        "step_count": step_count,
        "total_departed": total_departed,
        "total_arrived": total_arrived,
        "avg_travel_time": avg_travel_time,  # Global average (all vehicles from start)
        "checkpoint_avg_travel_time": checkpoint_avg_travel_time,  # Checkpoint-specific average (if applicable)
        "checkpoint_arrived_count": checkpoint_arrived_count,  # Number of vehicles arrived in this checkpoint (if applicable)
        "avg_waiting_time": avg_waiting_time,  # Global average (all waiting vehicles)
        "checkpoint_avg_waiting_time": checkpoint_avg_waiting_time,  # Checkpoint-specific average waiting time (if applicable)
        "checkpoint_waiting_count": checkpoint_waiting_count,  # Number of waiting vehicles (if applicable)
        "control_states": control_states if apply_control else None,
        "checkpoint_path_t_minus_1": checkpoint_path_t_minus_1 if checkpoint_interval is not None else None,
        "checkpoint_target_end_time": target_end_time,
        "aborted": ended_early,
        "abort_reason": abort_reason,
        "abort_checkpoint_path": abort_checkpoint_path,
        "abort_checkpoint_time": abort_checkpoint_time,
        "abort_checkpoint_saved": abort_checkpoint_saved
    }
    
    # Calculate final metrics for enabled control modules
    if apply_control and module_metrics:
        from control_modules import get_control_module
        module_results = {}
        for module_name in control_configs.keys():
            if module_name in module_metrics:
                module = get_control_module(module_name)
                if module and hasattr(module, 'calculate_final_results'):
                    calc_sig = inspect.signature(module.calculate_final_results)
                    if "control_state" in calc_sig.parameters:
                        module_results[module_name] = module.calculate_final_results(
                            metrics=module_metrics[module_name],
                            env=env,
                            control_state=(control_states or {}).get(module_name)
                        )
                    else:
                        module_results[module_name] = module.calculate_final_results(
                            metrics=module_metrics[module_name],
                            env=env
                        )
        if module_results:
            results["module_metrics"] = module_results
    
    # Add traffic state collector stats if used
    if traffic_state_collector is not None:
        results["traffic_state_stats"] = traffic_state_collector.get_stats()
    
    return results


def run_policy_simulation(
    checkpoint_path: str,
    control_configs: Dict[str, Dict[str, Any]],
    duration: float = 300.0,
    use_gui: bool = False,
    config_path: Optional[str] = None,
    checkpoint_interval: Optional[float] = None,
    run_duration: Optional[float] = None,
    seed: Optional[int] = None,
    initial_control_states: Optional[Dict[str, Dict[str, Any]]] = None,
    taxi_dispatch_algorithm: Optional[str] = None,
    taxi_idle_algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a simulation from a checkpoint to test control module policies.
    This is used by LLM agent to evaluate control configurations.

    Args:
        checkpoint_path: Path to checkpoint snapshot file (t state)
        control_configs: Dictionary of control configurations by module name
                       Format: {"signal_timing": {...}, "subway_scheduling": {...}, ...}
        duration: Simulation duration in seconds (default: 300 = 5 minutes)
        use_gui: Whether to use sumo-gui
        config_path: Path to SUMO config file (should match original simulation)
        checkpoint_interval: Checkpoint interval in seconds (used to find t-1 state)
        run_duration: Remaining simulation duration in seconds (used for run_counts)
        seed: Random seed for simulation (default: None). Should use the same seed as main simulation for consistency.
        initial_control_states: Optional control state snapshot to seed the policy run (e.g., taxi scheduling state).
        taxi_dispatch_algorithm: Optional SUMO taxi dispatch algorithm (default: "traci").
        taxi_idle_algorithm: Optional SUMO taxi idle algorithm (default: "randomCircling").

    Returns:
        Dictionary containing:
            - success: Whether simulation completed successfully
            - total_departed: Number of departed vehicles
            - total_arrived: Number of arrived vehicles
            - avg_travel_time: Average travel time in seconds
            - duration: Actual simulation duration
            - module_metrics: Metrics for each control module (if available)
            - error: Error message (if failed)
    """
    # Handle None checkpoint_path - this happens when master is first created
    if checkpoint_path is None:
        return {
            "success": False,
            "error": "No checkpoint available yet. Checkpoint will be created after simulation runs."
        }

    try:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            return {
                "success": False,
                "error": f"Checkpoint file not found: {checkpoint_path}"
            }
        
        # Load environment from checkpoint
        # We need to create a temporary SUMOEnv and load from checkpoint
        # First, we need to find the original config file
        # The checkpoint should be in records/checkpoints/, and we can infer config from there
        
        workspace_root = _get_workspace_root()
        
        # Use provided config_path or try to infer from checkpoint path
        if config_path is None:
            # Try to infer config path from checkpoint path
            # Checkpoint path format: .../checkpoints/checkpoint_<time>.xml
            # Try common config locations
            default_configs = [
                workspace_root / "sumo_config" / "nyc" / "nyc_pt_all.sumocfg",
                workspace_root / "sumo_config" / "jinan" / "jinan.sumocfg",
            ]
            config_path = None
            for cfg in default_configs:
                if cfg.exists():
                    config_path = str(cfg)
                    break
            
            if config_path is None:
                return {
                    "success": False,
                    "error": f"SUMO config file not found. Please provide config_path parameter."
                }
        else:
            config_path = str(config_path)
        
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            return {
                "success": False,
                "error": f"SUMO config file not found: {config_path}"
            }
        
        # Determine t-1 checkpoint path
        checkpoint_path_t_minus_1 = None
        if checkpoint_interval is not None:
            # Extract checkpoint time from checkpoint_path
            # Format: checkpoint_<time>.xml or checkpoint_<count>_<time>.xml
            checkpoint_filename = Path(checkpoint_path).name
            checkpoint_dir = Path(checkpoint_path).parent
            
            try:
                # Try to extract time from filename
                # Format could be: checkpoint_<time>.xml or checkpoint_<count>_<time>.xml
                # For t-1, the filename should be: checkpoint_<time>_t_minus_1.xml
                # So we construct it from the t checkpoint filename
                if "_t_minus_1.xml" in checkpoint_filename:
                    # Already a t-1 checkpoint, use it directly
                    checkpoint_path_t_minus_1 = checkpoint_path
                else:
                    # Construct t-1 checkpoint path from t checkpoint path
                    # Replace .xml with _t_minus_1.xml
                    checkpoint_filename_t_minus_1 = checkpoint_filename.replace(".xml", "_t_minus_1.xml")
                    checkpoint_path_t_minus_1 = str(checkpoint_dir / checkpoint_filename_t_minus_1)
                
            except Exception as e:
                print(f"Warning: Could not construct t-1 checkpoint path: {e}")
                print(f"Will try to load t state checkpoint instead")
        
        # Load from t-1 checkpoint (required, no fallback to t checkpoint)
        if checkpoint_path_t_minus_1 is None:
            return {
                "success": False,
                "error": f"t-1 checkpoint path not determined. checkpoint_interval may be missing."
            }
        
        if not Path(checkpoint_path_t_minus_1).exists():
            return {
                "success": False,
                "error": f"t-1 checkpoint file not found: {checkpoint_path_t_minus_1}. "
                        f"Checkpoint may not have been saved correctly during simulation."
            }
        
        # Read metadata to get begin_time (use _metadata.json format to match snapshot() method)
        begin_time = None
        metadata_path = os.path.splitext(checkpoint_path_t_minus_1)[0] + "_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Try checkpoint_time first (format used by snapshot()), fallback to sim_time for backward compatibility
                    begin_time = metadata.get("checkpoint_time") or metadata.get("sim_time")
                    if begin_time is not None:
                        print(f"Read checkpoint time from metadata: {begin_time:.2f}s")
            except Exception as e:
                print(f"Warning: Failed to read metadata from {metadata_path}: {e}")
        
        # Create environment and load checkpoint directly (avoids extra restart)
        print(f"Creating environment with config: {config_path}")
        print(f"Loading t-1 checkpoint directly during initialization: {checkpoint_path_t_minus_1}")
        
        # Determine control_modules from control_configs
        # Always include signal_timing and ramp_metering to ensure they are enabled
        # This matches the behavior in run_*_control.py where signal_timing and ramp_metering are always enabled
        control_modules = list(control_configs.keys()) if control_configs else []
        if 'signal_timing' not in control_modules:
            control_modules.append('signal_timing')
        if 'ramp_metering' not in control_modules:
            control_modules.append('ramp_metering')
        
        # Extract config_dir_name from config_path for control module initialization
        # Reuse config_path_obj that was already defined above
        config_dir_name = config_path_obj.parent.name  # e.g., "jinan"
        
        # Use remaining_duration as run_counts if available, otherwise fallback to checkpoint_interval
        if run_duration is not None:
            run_counts_value = int(run_duration)
        else:
            run_counts_value = 3600
        
        print(f"Enabling control modules: {control_modules}")

        # Store work_dir for cleanup after simulation
        policy_work_dir = None
        try:
            # Pass load_state_path to create_sumo_env to load checkpoint directly
            # This avoids creating environment and then resetting again
            # IMPORTANT: Always pass control_modules including signal_timing and ramp_metering to ensure they are enabled
            # Use use_unique_work_dir=True to avoid log file conflicts with main simulation
            env, env_config = create_sumo_env(
                config_path=config_path,
                use_gui=use_gui,
                run_counts=run_counts_value,  # Set RUN_COUNTS to match remaining_duration
                load_state_path=str(checkpoint_path_t_minus_1),  # Load checkpoint directly
                begin_time=begin_time,  # Pass begin_time for proper checkpoint resumption
                seed=seed,
                control_modules=control_modules,  # Pass control_modules to enable signal_timing and other modules
                config_dir_name=config_dir_name,  # Pass config_dir_name for control module initialization
                taxi_dispatch_algorithm=taxi_dispatch_algorithm,
                taxi_idle_algorithm=taxi_idle_algorithm,
                use_unique_work_dir=True,  # Use unique work directory to avoid log file conflicts
            )
            # Store work directory path for cleanup
            policy_work_dir = env.path_to_work_directory if hasattr(env, 'path_to_work_directory') else None
            print(f"Checkpoint loaded successfully. Current simulation time: {env.get_current_time():.0f}s")
            print(f"Enabled control modules: {control_modules}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create environment and load checkpoint from {checkpoint_path_t_minus_1}: {str(e)}"
            }
        
        # Track travel times for policy simulation
        # Get initial arrived vehicle travel times to calculate policy-specific avg travel time
        policy_start_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
        policy_start_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
        # Track highway-only travel times for policy simulation
        policy_start_highway_tt_sum = env._highway_arrived_tt_sum if hasattr(env, '_highway_arrived_tt_sum') else 0.0
        policy_start_highway_count = env._highway_arrived_count if hasattr(env, '_highway_arrived_count') else 0
        
        # Run simulation with the test policy
        module_names = list(control_configs.keys())
        print(f"Testing control configurations for {duration:.0f}s...")
        print(f"  Control modules: {module_names}")
        for module_name, config_entry in control_configs.items():
            config = effective_config(config_entry)
            if isinstance(config, dict):
                print(f"    - {module_name}: {len(config)} entries")

        control_states_for_test = copy.deepcopy(initial_control_states) if initial_control_states else None

        # Restore taxi scheduling state from checkpoint if available
        # This ensures initial_fleet_size is preserved across checkpoint loads in policy simulation
        if "taxi_scheduling" in (control_configs or {}):
            from control_modules import get_control_module
            taxi_module = get_control_module("taxi_scheduling")
            if taxi_module and hasattr(taxi_module, "restore_control_state_from_checkpoint"):
                checkpoint_state = None
                # Try to get checkpoint state from various sources
                if hasattr(env, "checkpoint_taxi_state"):
                    checkpoint_state = env.checkpoint_taxi_state
                elif hasattr(env, "checkpoint_extra_metadata"):
                    extra = env.checkpoint_extra_metadata or {}
                    checkpoint_state = extra.get("taxi_state") or extra.get("control_states", {}).get("taxi_scheduling")

                if checkpoint_state:
                    print(f"[TaxiScheduling] Restoring control state from checkpoint in policy simulation...")
                    if control_states_for_test is None:
                        control_states_for_test = {}
                    control_states_for_test["taxi_scheduling"] = taxi_module.restore_control_state_from_checkpoint(
                        env=env,
                        control_state=control_states_for_test.get("taxi_scheduling"),
                        checkpoint_state=checkpoint_state
                    )

        results = run_controlled_simulation(
            env=env,
            duration=duration,  # Use duration parameter for policy test
            step_seconds=30,
            traffic_state_collector=None,  # Don't collect traffic states during policy test
            checkpoint_interval=None,  # No checkpointing during policy test
            control_configs=control_configs,  # Apply control configurations
            control_states=control_states_for_test,  # Seed with provided control state if available
            save_checkpoint=False  # Don't save checkpoint or collect traffic states during policy test
        )
        
        # Calculate policy-specific average travel time
        # Only count vehicles that arrived during this policy simulation
        policy_end_arrived_tt_sum = env._arrived_tt_sum if hasattr(env, '_arrived_tt_sum') else 0.0
        policy_end_arrived_count = env._arrived_count if hasattr(env, '_arrived_count') else 0
        policy_arrived_tt_sum = policy_end_arrived_tt_sum - policy_start_arrived_tt_sum
        policy_arrived_count = policy_end_arrived_count - policy_start_arrived_count
        policy_avg_travel_time = (policy_arrived_tt_sum / policy_arrived_count) if policy_arrived_count > 0 else 0.0

        # Calculate policy-specific highway average travel time
        policy_end_highway_tt_sum = env._highway_arrived_tt_sum if hasattr(env, '_highway_arrived_tt_sum') else 0.0
        policy_end_highway_count = env._highway_arrived_count if hasattr(env, '_highway_arrived_count') else 0
        policy_highway_tt_sum = policy_end_highway_tt_sum - policy_start_highway_tt_sum
        policy_highway_arrived_count = policy_end_highway_count - policy_start_highway_count
        policy_highway_avg_travel_time = (policy_highway_tt_sum / policy_highway_arrived_count) if policy_highway_arrived_count > 0 else 0.0
        
        # Close environment
        env.close()

        # Clean up unique work directory to avoid disk space accumulation
        if policy_work_dir and "policy_" in policy_work_dir:
            try:
                import shutil
                shutil.rmtree(policy_work_dir, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"Warning: Failed to clean up policy work directory {policy_work_dir}: {cleanup_err}")
        
        # Return results
        return_dict = {
            "success": True,
            "total_departed": results.get("total_departed", 0),
            "total_arrived": results.get("total_arrived", 0),
            "avg_travel_time": results.get("avg_travel_time", 0),  # Global average (all vehicles from checkpoint start)
            "policy_avg_travel_time": policy_avg_travel_time,  # Policy-specific average (only vehicles arrived during this policy test)
            "policy_arrived_count": policy_arrived_count,  # Number of vehicles arrived during this policy test
            "policy_highway_avg_travel_time": policy_highway_avg_travel_time,  # Highway-only policy avg travel time
            "policy_highway_arrived_count": policy_highway_arrived_count,  # Highway-only arrived vehicles during policy test
            "duration": results.get("duration", 0),
            "error": None
        }
        
        # Add module metrics if available
        if "module_metrics" in results:
            return_dict["module_metrics"] = results["module_metrics"]
        
        return return_dict
        
    except Exception as e:
        error_msg = f"Policy simulation failed: {str(e)}"
        print(f"Error: {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "total_departed": 0,
            "total_arrived": 0,
            "avg_travel_time": 0,
            "duration": 0,
            "error": error_msg
        }
