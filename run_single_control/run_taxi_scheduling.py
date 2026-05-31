"""
LLM-controlled taxi scheduling simulation runner with checkpoint-based control.

This script implements taxi dispatch and repositioning control using LLM agents:
1. Simulation runs for checkpoint_interval seconds
2. LLM agent analyzes taxi fleet state and pending reservations
3. LLM generates dispatch and repositioning decisions
4. Decisions are applied to the simulation
5. Repeats until simulation_duration is reached

Usage:
    # Activate conda environment and run simulation
    conda activate deepcity
    python run_taxi_scheduling.py \
        --config /data/zhouyuping/Zone/zone_scenarios/Manhattan/Manhattan.sumocfg \
        --duration 3600 \
        --checkpoint-interval 300 \
        --step-seconds 10 \
        --llm-model "siliconflow/deepseek-ai/DeepSeek-V3.2" \
        --control-modules taxi_scheduling
"""

import sys
import copy
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
from datetime import datetime

# Add project root to path BEFORE importing utils modules
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent  # Go up from run_single_control/ to project root
sys.path.insert(0, str(workspace_root))

# Try to import wandb, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from utils.llm_agent import LLMAgent
from utils.simulation_utils import (
    create_sumo_env,
    run_controlled_simulation,
    run_policy_simulation,
    TrafficStateCollector
)
from utils.id_utils import generate_simulation_identifiers
from utils.traffic_state_collector import init_traffic_states_file
from utils.checkpoint_logger import CheckpointLogger, extract_configs_only, effective_config
from utils.path_utils import resolve_config_path
from utils.prompt_utils import append_user_query


def _load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint metadata JSON for a given checkpoint path."""
    metadata_path = os.path.splitext(checkpoint_path)[0] + "_metadata.json"
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: Failed to read checkpoint metadata: {metadata_path}: {exc}")
        return {}


def _restore_control_state_from_metadata(
    env,
    control_configs: Optional[Dict[str, Dict[str, Any]]],
    control_states: Optional[Dict[str, Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[Dict[str, Dict[str, Any]]], Optional[Dict[str, Dict[str, Any]]]]:
    """Restore control configs/states from checkpoint metadata and resync taxi state."""
    extra = {}
    if metadata:
        extra = metadata.get("extra", {}) or {}
    if not extra and hasattr(env, "checkpoint_extra_metadata"):
        extra = env.checkpoint_extra_metadata or {}

    checkpoint_configs = extra.get("control_configs")
    checkpoint_states = extra.get("control_states")

    if checkpoint_configs:
        control_configs = copy.deepcopy(checkpoint_configs)

    if checkpoint_states:
        control_states = copy.deepcopy(checkpoint_states)

    if control_states and "_previous_checkpoint_path" in control_states:
        control_states.pop("_previous_checkpoint_path", None)

    if "taxi_scheduling" in (control_configs or {}) or "taxi_scheduling" in getattr(env, "control_modules", []):
        from control_modules import get_control_module

        module = get_control_module("taxi_scheduling")
        if module and hasattr(module, "restore_control_state_from_checkpoint"):
            if control_states is None:
                control_states = {}
            current_state = control_states.get("taxi_scheduling")
            checkpoint_state = None
            if checkpoint_states:
                checkpoint_state = checkpoint_states.get("taxi_scheduling")
            control_states["taxi_scheduling"] = module.restore_control_state_from_checkpoint(
                env=env,
                control_state=current_state,
                checkpoint_state=checkpoint_state,
            )

    return control_configs, control_states


def log_metrics_to_wandb(
    wandb_run,
    checkpoint_number: int,
    elapsed_time: float,
    module_metrics: Dict[str, Dict[str, Any]],
    control_modules: Optional[List[str]] = None,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """
    Log module performance metrics to wandb.
    
    Args:
        wandb_run: wandb run object
        checkpoint_number: Checkpoint number
        elapsed_time: Elapsed simulation time in seconds
        module_metrics: Dictionary of module metrics {module_name: {metric_name: value}}
        control_modules: List of control module names
        additional_metrics: Additional metrics to log (e.g., total_departed, total_arrived)
    """
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    
    # Prepare metrics dictionary
    metrics_dict = {
        "checkpoint": checkpoint_number,
        "elapsed_time": elapsed_time,
        "elapsed_time_hours": elapsed_time / 3600.0
    }
    
    # Add additional metrics if provided
    if additional_metrics:
        metrics_dict.update(additional_metrics)
    
    # Add module-specific metrics
    if control_modules and module_metrics:
        for module_name in control_modules:
            if module_name not in module_metrics:
                continue
            
            metrics = module_metrics[module_name]
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    # Use module name as prefix for clarity
                    wandb_key = f"{module_name}/{metric_name}"
                    metrics_dict[wandb_key] = float(metric_value)
    
    # Log to wandb
    wandb_run.log(metrics_dict)


def calculate_average_module_metrics(
    checkpoints: List[Dict[str, Any]],
    control_modules: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average module metrics across all checkpoints.
    
    Args:
        checkpoints: List of checkpoint dictionaries, each containing module_metrics
        control_modules: List of control module names to calculate averages for
        
    Returns:
        Dictionary mapping module names to their average metrics
        Format: {module_name: {metric_name: average_value, ...}}
    """
    # Initialize metric accumulators for each module
    module_metric_sums = {}
    module_metric_counts = {}
    
    for checkpoint in checkpoints:
        module_metrics = checkpoint.get("module_metrics", {})
        
        for module_name in control_modules:
            if module_name not in module_metrics:
                continue
            
            # Initialize accumulators for this module if not exists
            if module_name not in module_metric_sums:
                module_metric_sums[module_name] = {}
                module_metric_counts[module_name] = {}
            
            # Accumulate metrics for this module
            for metric_name, metric_value in module_metrics[module_name].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in module_metric_sums[module_name]:
                        module_metric_sums[module_name][metric_name] = 0.0
                        module_metric_counts[module_name][metric_name] = 0
                    
                    module_metric_sums[module_name][metric_name] += float(metric_value)
                    module_metric_counts[module_name][metric_name] += 1
    
    # Calculate averages
    average_metrics = {}
    for module_name in control_modules:
        if module_name not in module_metric_sums:
            continue
        
        average_metrics[module_name] = {}
        for metric_name in module_metric_sums[module_name].keys():
            count = module_metric_counts[module_name][metric_name]
            if count > 0:
                average_metrics[module_name][metric_name] = (
                    module_metric_sums[module_name][metric_name] / count
                )
    
    return average_metrics


def format_simulation_time(seconds: float) -> Dict[str, Any]:
    """
    Format simulation time in seconds to human-readable time format.
    """
    total_hours = seconds / 3600.0
    day_hour = total_hours % 24
    hours = int(day_hour)
    minutes = int((day_hour - hours) * 60)
    day_number = int(total_hours / 24) + 1

    time_string = f"{hours:02d}:{minutes:02d}"

    # Determine time period (adjusted for taxi demand patterns)
    if 7 <= hours < 9:
        time_period = "Morning Rush"
        period_description = "High taxi demand for commute trips (7:00-9:00)"
    elif 9 <= hours < 12:
        time_period = "Late Morning"
        period_description = "Moderate demand, business trips (9:00-12:00)"
    elif 12 <= hours < 14:
        time_period = "Lunch"
        period_description = "Moderate demand, lunch trips (12:00-14:00)"
    elif 14 <= hours < 17:
        time_period = "Afternoon"
        period_description = "Moderate demand, business trips (14:00-17:00)"
    elif 17 <= hours < 20:
        time_period = "Evening Rush"
        period_description = "High taxi demand for commute and leisure (17:00-20:00)"
    elif 20 <= hours < 24:
        time_period = "Night"
        period_description = "Moderate demand, entertainment and leisure (20:00-24:00)"
    else:  # 0 <= hours < 7
        time_period = "Late Night"
        period_description = "Low demand, airport and late-night trips (00:00-7:00)"

    return {
        "hours": hours,
        "minutes": minutes,
        "time_string": time_string,
        "time_period": time_period,
        "period_description": period_description,
        "day_of_simulation": day_number,
        "total_hours": total_hours
    }


def run_taxi_scheduling_simulation(
    config_path: str,
    simulation_duration: float,
    checkpoint_interval: float,
    step_seconds: int = 10,
    use_gui: bool = False,
    seed: Optional[int] = None,
    time_to_teleport: Optional[float] = None,
    llm_model: str = "siliconflow/deepseek-ai/DeepSeek-V3.2",
    max_agent_turns: int = 10,
    control_modules: Optional[List[str]] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    traffic_state_interval: float = 60,
    max_reflection_turns: int = 5,
    max_interval_retries: int = 2,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    verbose: bool = True,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run taxi scheduling simulation with LLM-controlled dispatch and repositioning.
    
    Args:
        config_path: Path to SUMO config file
        simulation_duration: Total simulation duration in seconds
        checkpoint_interval: Interval between checkpoints in seconds
        step_seconds: Step size for simulation (default: 10)
        use_gui: Whether to use sumo-gui (default: False)
        seed: Random seed for simulation (default: None)
        time_to_teleport: SUMO time-to-teleport override (default: None uses sumocfg)
        llm_model: LLM model name for agent
        max_agent_turns: Maximum dialogue turns for LLM agent
        control_modules: List of control modules to enable
        max_interval_retries: Max retries per checkpoint interval on SUMO/TraCI failure
        
    Returns:
        Dictionary with final simulation results
    """
    # Create checkpoint directory
    checkpoint_dir = workspace_root / "records" / "taxi_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize simulation state
    env = None
    checkpoint_count = 0
    total_elapsed_time = 0.0
    accumulated_results = {
        "total_steps": 0,
        "total_departed": 0,
        "total_arrived": 0,
        "checkpoints": []
    }

    # Initialize traffic states recording
    TRAFFIC_STATE_INTERVAL = traffic_state_interval  # Interval for collecting traffic state data
    
    # Generate simulation_id and file_prefix using unified function
    config_path_obj = Path(config_path)
    config_dir_name = config_path_obj.parent.name
    simulation_id, file_prefix = generate_simulation_identifiers(
        config_name=config_dir_name,
        llm_name=llm_model,
        control_modules=control_modules
    )
    traffic_states_filepath = None

    # Initialize checkpoint logger
    checkpoint_logger = CheckpointLogger(
        simulation_id=simulation_id,
        llm_model_name=llm_model
    )
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        # Prepare control modules list for group name
        if control_modules is None:
            control_modules = ['taxi_scheduling']  # Default
        
        # Create group name: LLM模型名称-所有模块名称-sumo_config name
        # Sanitize LLM model name (remove slashes and special characters)
        llm_model_safe = llm_model.replace("/", "_").replace("\\", "_").replace(":", "_")
        modules_str = "_".join(sorted(control_modules)) if control_modules else "baseline"
        wandb_group = f"{llm_model_safe}-{modules_str}-{config_dir_name}"

        # Create run name: 实验开始的时间
        experiment_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        wandb_config = {
            "simulation_id": simulation_id,
            "config_path": config_path,
            "config_name": config_dir_name,
            "simulation_duration": simulation_duration,
            "checkpoint_interval": checkpoint_interval,
            "step_seconds": step_seconds,
            "seed": seed,
            "llm_model": llm_model,
            "max_agent_turns": max_agent_turns,
            "control_modules": control_modules if control_modules else [],
            "use_gui": use_gui,
        }
        
        wandb_run = wandb.init(
            project=wandb_project or "sumo_taxi_scheduling",
            name=experiment_start_time,
            group=wandb_group,
            config=wandb_config,
            reinit=True
        )
        print(f"Initialized wandb run: {wandb_run.name} (group: {wandb_group})")

    try:
        # Set random seeds for reproducibility
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            print(f"Set Python random seed: {seed}")
            print(f"Set NumPy random seed: {seed}")

        # Default to taxi_scheduling module
        if control_modules is None:
            control_modules = ['taxi_scheduling']

        print("=" * 80)
        print("Starting LLM-controlled Taxi Scheduling Simulation")
        print(f"  Total duration: {simulation_duration}s ({simulation_duration/3600:.1f} hours)")
        print(f"  Checkpoint interval: {checkpoint_interval}s ({checkpoint_interval/60:.0f} minutes)")
        print(f"  Step size: {step_seconds}s")
        print(f"  LLM model: {llm_model}")
        print(f"  Control modules: {control_modules}")
        print("=" * 80)

        # Extract config directory name
        config_path_obj = Path(config_path)
        config_dir_name = config_path_obj.parent.name

        print(f"\n[Checkpoint {checkpoint_count}] Creating new environment...")
        # Build control modules list, avoiding duplicates while preserving order
        enabled_modules = control_modules if control_modules else []
        always_enabled = ['signal_timing', 'ramp_metering']   # Modules that are always enabled
        
        env, env_config = create_sumo_env(
            config_path=config_path,
            use_gui=use_gui,
            seed=seed,
            control_modules=list(set(enabled_modules + always_enabled)),  # Merge user-selected and always-enabled modules
            run_counts=int(simulation_duration),
            config_dir_name=config_dir_name,
            time_to_teleport=time_to_teleport,
            use_simulated_taxi_system=True,
        )

        # Initialize traffic states file
        if traffic_states_filepath is None:
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=simulation_id,
                config_name=config_dir_name,
                llm_name=llm_model,
                control_modules=control_modules
            )
            print(f"Initialized traffic states file: {traffic_states_filepath.name}")

        # Get graphs for context
        graphs = {}
        lane_dict = None
        lane_inter_graph = None
        try:
            graphs = env.get_road_network_graphs()
            lane_dict = graphs.get("lane_dict", {})
            lane_inter_graph = graphs.get("lane_inter_graph")
            print(f"Retrieved graphs: {list(graphs.keys())}")
        except Exception as e:
            print(f"Warning: Failed to get graphs: {e}")

        # Create traffic state collector
        traffic_state_collector = TrafficStateCollector(
            env=env,
            traffic_states_filepath=traffic_states_filepath,
            interval=TRAFFIC_STATE_INTERVAL,
            lane_dict=lane_dict,
            lane_inter_graph=lane_inter_graph,
            simulation_id=simulation_id
        )
        print(f"Created traffic state collector (interval: {TRAFFIC_STATE_INTERVAL}s)")

        # Initialize LLM agent
        agent = LLMAgent(
            model_name=llm_model,
            temperature=temperature,
            max_turns=max_agent_turns,
            available_control_modules=control_modules,
            config_name=config_dir_name,
            max_reflection_turns=max_reflection_turns,
            base_url=base_url,
        )
        print(f"Initialized LLM agent (model: {llm_model}, max_turns: {max_agent_turns})")

        # Initialize control_configs only for modules that LLM agent will optimize
        # Note: env.enabled_controls already initializes configs for all enabled modules (including always_enabled)
        # We only need to extract configs for modules that LLM agent is responsible for optimizing
        control_configs = None
        control_states = None

        if control_modules and env.enabled_controls:
            print(f"\nInitializing control configs for LLM agent optimization modules: {control_modules}")
            control_configs = {}

            # Only initialize configs for modules that LLM agent will optimize
            # These configs will be passed to LLM agent for optimization
            for module_name in control_modules:
                # Map module_name to control_type (they are usually the same)
                control_type = module_name
                
                if control_type not in env.enabled_controls:
                    print(f"  - Warning: {module_name} not found in env.enabled_controls, skipping...")
                    continue
                
                module_info = env.enabled_controls[control_type]
                module = module_info.get('module')
                
                if module is None:
                    print(f"  - Warning: {module_name} module not available, skipping...")
                    continue

                # Get config from env.enabled_controls (already initialized by sumo_env.py)
                # We don't need to call get_default_config() again or modify env.enabled_controls
                print(f"  - Initializing {module_name} config for LLM agent...")
                existing_config = module_info.get('config', {})
                
                if existing_config:
                    # Use deep copy to avoid modifying env.enabled_controls
                    # Store both module and config for LLM agent manager
                    control_configs[module_name] = {
                        'module': module,
                        'config': copy.deepcopy(existing_config)
                    }
                    print(f"    ✓ Loaded config for {module_name}: {len(existing_config)} entries")
                else:
                    print(f"    ⚠ Warning: {module_name} config is empty in env.enabled_controls")

            # Removed legacy fallback to ensure consistency with run_sumo_simulation.py
            # Both scripts now use get_default_config() only, not file-based configs

            if control_configs:
                print(f"  Control modules initialized: {list(control_configs.keys())}")
            else:
                print(f"  No valid control configs found, running baseline simulation")
                control_configs = None
        else:
            print(f"\nNo control modules enabled, running baseline simulation")

        # Keep an immutable baseline copy of default configs for per-checkpoint comparisons
        default_control_configs = copy.deepcopy(control_configs) if control_configs else {}

        # Run simulation loop
        remaining_duration = simulation_duration
        previous_checkpoint_path = None
        is_first_simulation = True
        max_interval_retries = max(0, int(max_interval_retries))

        while remaining_duration > 0:
            checkpoint_count += 1
            current_checkpoint_duration = min(checkpoint_interval, remaining_duration)

            print(f"\n[Checkpoint {checkpoint_count}] Running simulation for {current_checkpoint_duration:.0f}s...")
            print(f"  Remaining duration: {remaining_duration:.0f}s")
            remaining_duration -= current_checkpoint_duration

            # Prepare control_states
            if control_states is None:
                control_states = {}
            if previous_checkpoint_path:
                control_states["_previous_checkpoint_path"] = previous_checkpoint_path

            # Snapshot control_states at the START of this checkpoint window (used for policy evaluation from t-1 state)
            checkpoint_start_control_states = copy.deepcopy(control_states)
            if "_previous_checkpoint_path" in checkpoint_start_control_states:
                del checkpoint_start_control_states["_previous_checkpoint_path"]

            interval_elapsed = 0.0
            interval_target_end = None
            retry_attempts = 0
            while True:
                if interval_target_end is None:
                    interval_target_end = env.get_current_time() + current_checkpoint_duration
                remaining_in_interval = max(0.0, interval_target_end - env.get_current_time())

                # Run simulation (possibly resuming mid-interval)
                results = run_controlled_simulation(
                    env=env,
                    duration=remaining_in_interval,
                    step_seconds=step_seconds,
                    traffic_state_collector=traffic_state_collector,
                    checkpoint_interval=remaining_in_interval,
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_prefix=f"taxi_checkpoint_{checkpoint_count}",
                    control_configs=control_configs,
                    control_states=control_states,
                    is_first_simulation=is_first_simulation,
                    config_name=config_dir_name,
                    llm_name=llm_model,
                    simulation_id=simulation_id,
                )

                if results.get("aborted") and not results.get("checkpoint_reached", False):
                    abort_reason = results.get("abort_reason", "unknown")
                    if retry_attempts < max_interval_retries:
                        retry_attempts += 1

                        attempt_elapsed = results.get("duration")
                        if attempt_elapsed is None:
                            start_t = results.get("start_time")
                            end_t = results.get("abort_checkpoint_time", results.get("final_time"))
                            if start_t is not None and end_t is not None:
                                attempt_elapsed = end_t - start_t
                        if attempt_elapsed is None:
                            attempt_elapsed = 0.0
                        attempt_elapsed = max(0.0, float(attempt_elapsed))

                        abort_checkpoint_path = results.get("abort_checkpoint_path")
                        resume_from_abort = False
                        if abort_checkpoint_path and Path(abort_checkpoint_path).exists():
                            resume_from_abort = True
                            retry_path = abort_checkpoint_path
                            interval_elapsed = min(
                                current_checkpoint_duration,
                                interval_elapsed + attempt_elapsed
                            )
                            abort_time = results.get("abort_checkpoint_time")
                            if abort_time is not None:
                                print(
                                    f"\n[Checkpoint {checkpoint_count}] Interval aborted early ({abort_reason}). "
                                    f"Retrying from abort checkpoint t={abort_time:.0f}s "
                                    f"({retry_attempts}/{max_interval_retries})..."
                                )
                            else:
                                print(
                                    f"\n[Checkpoint {checkpoint_count}] Interval aborted early ({abort_reason}). "
                                    f"Retrying from abort checkpoint ({retry_attempts}/{max_interval_retries})..."
                                )
                        else:
                            retry_path = results.get("checkpoint_path_t_minus_1")
                            interval_elapsed = 0.0
                            interval_target_end = None
                            print(
                                f"\n[Checkpoint {checkpoint_count}] Interval aborted early ({abort_reason}). "
                                f"Retrying from interval start ({retry_attempts}/{max_interval_retries})..."
                            )

                        if not retry_path or not Path(retry_path).exists():
                            raise RuntimeError(
                                f"Retry failed: checkpoint not found for checkpoint {checkpoint_count}"
                            )

                        loaded_ok = False
                        if hasattr(env, "load_state_inplace"):
                            loaded_ok = env.load_state_inplace(retry_path)
                        else:
                            env.reset(use_gui=use_gui, seed=seed, load_state_path=retry_path)
                            loaded_ok = True
                        if not loaded_ok:
                            raise RuntimeError(
                                f"Retry failed: could not load checkpoint {retry_path}"
                            )

                        metadata = _load_checkpoint_metadata(retry_path)
                        control_configs, control_states = _restore_control_state_from_metadata(
                            env,
                            control_configs,
                            control_states,
                            metadata=metadata
                        )

                        if resume_from_abort:
                            if control_states is None:
                                control_states = {}
                            control_states["_skip_t_minus_1"] = True

                        if traffic_state_collector is not None:
                            traffic_state_collector.pending_snapshots = []
                            checkpoint_start_time = env.get_current_time()
                            traffic_state_collector.last_collection_time = (
                                checkpoint_start_time - TRAFFIC_STATE_INTERVAL
                            )

                        is_first_simulation = False
                        continue

                    raise RuntimeError(
                        f"Checkpoint {checkpoint_count} aborted early ({abort_reason}) after "
                        f"{max_interval_retries} retries."
                    )

                break

            # Update control states
            if results.get("control_states"):
                control_states = results["control_states"]
                if "_previous_checkpoint_path" in control_states:
                    del control_states["_previous_checkpoint_path"]
                if "_skip_t_minus_1" in control_states:
                    del control_states["_skip_t_minus_1"]

            # Process checkpoint
            if results.get("checkpoint_reached", False):
                checkpoint_path = results.get("checkpoint_path")
                checkpoint_path_t_minus_1 = results.get("checkpoint_path_t_minus_1")
                elapsed_time = results.get("elapsed_time", 0)
                elapsed_time = interval_elapsed + elapsed_time

                if elapsed_time != current_checkpoint_duration:
                    remaining_duration += (current_checkpoint_duration - elapsed_time)
                    if remaining_duration < 0:
                        remaining_duration = 0

                if checkpoint_path:
                    previous_checkpoint_path = checkpoint_path

                print(f"\n[Checkpoint {checkpoint_count}] Checkpoint reached!")
                print(f"  Elapsed time: {elapsed_time:.0f}s")
                print(f"  Remaining duration: {remaining_duration:.0f}s")
                if checkpoint_path:
                    print(f"  Checkpoint saved: {checkpoint_path}")

                # Get taxi fleet state for context
                taxi_module = None
                if 'taxi_scheduling' in env.enabled_controls:
                    taxi_module = env.enabled_controls['taxi_scheduling'].get('module')

                fleet_state = {}
                pending_reservations = {}
                taz_stats = {}
                if taxi_module:
                    fleet_state = taxi_module.get_taxi_fleet_state(env, control_state=(control_states or {}).get("taxi_scheduling"))
                    # Include person pickup positions so the LLM can rank taxis by distance
                    pending_reservations = taxi_module.get_pending_reservations(env, include_person_positions=True)
                    if (
                        isinstance(pending_reservations, dict)
                        and "reservations" in pending_reservations
                        and control_states is not None
                        and "taxi_scheduling" in control_states
                    ):
                        control_states["taxi_scheduling"]["_cached_checkpoint_pending"] = pending_reservations
                    if (
                        isinstance(pending_reservations, dict)
                        and "reservations" in pending_reservations
                        and isinstance(getattr(env, "enabled_controls", None), dict)
                        and "taxi_scheduling" in env.enabled_controls
                    ):
                        state_ref = env.enabled_controls["taxi_scheduling"].setdefault("state", {})
                        if isinstance(state_ref, dict):
                            state_ref["_cached_checkpoint_pending"] = pending_reservations

                    # Update TAZ stats using collected data
                    taxi_module.update_taz_stats(env, env.get_current_time())
                    taz_stats = taxi_module.get_taz_stats(env)
                    if control_states and "taxi_scheduling" in control_states:
                        control_states["taxi_scheduling"].pop("_cached_checkpoint_pending", None)
                    if (
                        isinstance(getattr(env, "enabled_controls", None), dict)
                        and "taxi_scheduling" in env.enabled_controls
                        and isinstance(env.enabled_controls["taxi_scheduling"].get("state"), dict)
                    ):
                        env.enabled_controls["taxi_scheduling"]["state"].pop("_cached_checkpoint_pending", None)

                    print(f"\n  Taxi Fleet Status:")
                    print(f"    Fleet size: {fleet_state.get('fleet_size', 0)}")
                    print(f"    Idle: {fleet_state.get('idle_count', 0)}")
                    print(f"    Pickup: {fleet_state.get('pickup_count', 0)}")
                    print(f"    Occupied: {fleet_state.get('occupied_count', 0)}")
                    print(f"    Pending reservations: {pending_reservations.get('pending_count', 0)}")
                    print(f"    Utilization: {fleet_state.get('utilization_rate', 0):.1%}")

                # Accumulate results
                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)

                module_metrics = results.get("module_metrics", {})
                checkpoint_info = {
                    "checkpoint_number": checkpoint_count,
                    "checkpoint_path": checkpoint_path,
                    "elapsed_time": elapsed_time,
                    "remaining_duration": remaining_duration,
                    "step_count": results.get("step_count", 0),
                    "fleet_state": fleet_state,
                    "pending_reservations": pending_reservations.get("pending_count", 0)
                }
                if module_metrics:
                    checkpoint_info["module_metrics"] = module_metrics
                accumulated_results["checkpoints"].append(checkpoint_info)

                total_elapsed_time += elapsed_time

                # Log metrics to wandb at checkpoint
                if wandb_run:
                    additional_metrics = {
                        "step_count": results.get("step_count", 0),
                        "avg_travel_time": results.get("avg_travel_time", 0),
                        "total_departed": results.get("total_departed", 0),
                        "total_arrived": results.get("total_arrived", 0),
                        "remaining_duration": remaining_duration
                    }
                    # Add taxi-specific metrics
                    if module_metrics and "taxi_scheduling" in module_metrics:
                        taxi_m = module_metrics["taxi_scheduling"]
                        if taxi_m.get("avg_wait_time") is not None:
                            additional_metrics["avg_wait_time"] = taxi_m["avg_wait_time"]
                        if taxi_m.get("passenger_pickups") is not None:
                            additional_metrics["passenger_pickups"] = taxi_m["passenger_pickups"]
                        if taxi_m.get("passenger_dropoffs") is not None:
                            additional_metrics["passenger_dropoffs"] = taxi_m["passenger_dropoffs"]
                    
                    log_metrics_to_wandb(
                        wandb_run=wandb_run,
                        checkpoint_number=checkpoint_count,
                        elapsed_time=elapsed_time,
                        module_metrics=module_metrics,
                        control_modules=control_modules,
                        additional_metrics=additional_metrics
                    )

                # Sync control_state metrics to module_metrics for reporting
                # (Taxi module tracks execution counts in control_state, not standard metrics)
                if control_states and "taxi_scheduling" in control_states and "taxi_scheduling" in module_metrics:
                    taxi_state = control_states["taxi_scheduling"]
                    taxi_metrics = module_metrics["taxi_scheduling"]
                    
                    # Update counts from control state
                    taxi_metrics["total_dispatches"] = taxi_state.get("total_dispatches", 0)
                    
                    # Also update completed dispatches count if available
                    if "completed_dispatches" in taxi_state:
                        taxi_metrics["successful_dispatches"] = len(taxi_state["completed_dispatches"])

                    # Sync passenger metrics to accumulated results
                    accumulated_results["passenger_pickups"] = taxi_metrics.get("passenger_pickups", 0)
                    accumulated_results["passenger_dropoffs"] = taxi_metrics.get("passenger_dropoffs", 0)

                # Display module metrics at checkpoint (same style as run_traffic_signal_control.py)
                module_metrics_to_show = results.get("module_metrics", {})
                if control_states and "taxi_scheduling" in control_states and "taxi_scheduling" in module_metrics_to_show:
                    taxi_state = control_states["taxi_scheduling"]
                    taxi_metrics_show = module_metrics_to_show["taxi_scheduling"]
                    taxi_metrics_show["total_dispatches"] = taxi_state.get("total_dispatches", 0)
                    if "completed_dispatches" in taxi_state:
                        taxi_metrics_show["successful_dispatches"] = len(taxi_state["completed_dispatches"])
                if module_metrics_to_show:
                    print(f"\n  Control Module Performance Metrics:")
                    for module_name, metrics in module_metrics_to_show.items():
                        print(f"    {module_name.upper().replace('_', ' ')}:")
                        for metric_name, metric_value in sorted(metrics.items()):
                            if isinstance(metric_value, float):
                                if metric_name == "throughput":
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                                else:
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                            elif isinstance(metric_value, int):
                                print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")
                            else:
                                print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")
                print(f"\n  Travel Time Metrics:")
                print(f"    - Global avg travel time (all vehicles): {results.get('avg_travel_time', 0):.2f}s")
                if module_metrics_to_show and "taxi_scheduling" in module_metrics_to_show:
                    taxi_m = module_metrics_to_show["taxi_scheduling"]
                    if taxi_m.get("avg_wait_time") is not None:
                        print(f"\n  Taxi Waiting Time Metrics:")
                        print(f"    - Avg passenger wait time: {taxi_m['avg_wait_time']:.2f}s")
                    if taxi_m.get("passenger_pickups") is not None or taxi_m.get("passenger_dropoffs") is not None:
                        print(f"\n  Taxi Passenger Metrics:")
                        print(f"    - Passenger pickups (this checkpoint): {taxi_m.get('passenger_pickups', 0)}")
                        print(f"    - Passenger dropoffs (this checkpoint): {taxi_m.get('passenger_dropoffs', 0)}")

                # Log checkpoint
                # Extract only configs (remove module instances) before passing to logger
                control_configs_for_log = extract_configs_only(control_configs) if control_configs else None
                checkpoint_logger.log_checkpoint(
                    checkpoint_number=checkpoint_count,
                    checkpoint_path=checkpoint_path,
                    checkpoint_path_t_minus_1=results.get("checkpoint_path_t_minus_1"),
                    elapsed_time=elapsed_time,
                    remaining_duration=remaining_duration,
                    step_count=results.get("step_count", 0),
                    avg_travel_time=results.get("avg_travel_time", 0),
                    module_metrics=module_metrics,
                    control_configs=control_configs_for_log
                )

                print(f"\n{'='*80}")
                print(f"[Checkpoint {checkpoint_count}] LLM Agent Taxi Optimization")
                print(f"{'='*80}")

                # Baseline configs for policy evaluation (default + greedy fallback + randomCircling idle)
                baseline_control_configs = copy.deepcopy(default_control_configs) if default_control_configs else {}
                # Ensure baseline starts without one-off decisions
                if "taxi_scheduling" in baseline_control_configs and isinstance(baseline_control_configs["taxi_scheduling"], dict):
                    baseline_control_configs["taxi_scheduling"].pop("dispatch_decisions", None)
                    baseline_control_configs["taxi_scheduling"].pop("reposition_decisions", None)

                # Keep env's module configs aligned with what the agent will treat as the base config
                if getattr(env, "enabled_controls", None):
                    for module_name, cfg in baseline_control_configs.items():
                        if module_name in env.enabled_controls:
                            env.enabled_controls[module_name]["config"] = cfg

                # Prepare context for LLM agent
                agent_context = {
                    "lane_graph": graphs.get("lane_graph"),
                    "lane_inter_graph": graphs.get("lane_inter_graph"),
                    "lane_dict": graphs.get("lane_dict"),
                    "road_graph": graphs.get("road_graph"),
                    "road_dict": graphs.get("road_dict"),
                    "taxi_fleet_state": fleet_state,
                    "pending_reservations": pending_reservations,
                    "taz_stats": taz_stats,
                    # Zone infrastructure
                    "zone_dict": env.zone_dict if hasattr(env, 'zone_dict') else {},
                    "zone_graph": env.zone_graph if hasattr(env, 'zone_graph') else None,
                    "current_taxi_config": effective_config((baseline_control_configs or {}).get('taxi_scheduling', {})),
                    "current_configs": baseline_control_configs,
                    "traffic_states_filepath": str(traffic_states_filepath),
                    "simulation_id": simulation_id,
                    "checkpoint_path": checkpoint_path,
                    "config_path": config_path,
                    "checkpoint_interval": checkpoint_interval,
                    "test_duration": checkpoint_interval,
                    "initial_control_states": checkpoint_start_control_states,
                    "taxi_dispatch_algorithm": env_config.get("TAXI_DISPATCH_ALGORITHM"),
                    "taxi_idle_algorithm": env_config.get("TAXI_IDLE_ALGORITHM"),
                    "remaining_duration": remaining_duration,
                    "use_gui": use_gui,
                    "config_name": config_dir_name,
                    "llm_name": llm_model,
                    "control_modules": control_modules,
                    "run_duration": simulation_duration
                }

                # Create initial prompt
                current_time = env.get_current_time()
                time_info = format_simulation_time(current_time)
                next_time_info = format_simulation_time(current_time + checkpoint_interval)

                # Format wait time for display (handle None)
                avg_wait_display = "N/A"
                if module_metrics and "taxi_scheduling" in module_metrics:
                    taxi_m = module_metrics["taxi_scheduling"]
                    if taxi_m.get("avg_wait_time") is not None:
                        avg_wait_display = f"{taxi_m['avg_wait_time']:.1f}s"
                    
                    # Also include passenger stats in prompt
                    pass_pickups = taxi_m.get("passenger_pickups", 0)
                    pass_dropoffs = taxi_m.get("passenger_dropoffs", 0)
                else:
                    pass_pickups = 0
                    pass_dropoffs = 0

                initial_prompt = f"""You are optimizing taxi dispatch and repositioning for an urban taxi fleet.

Current Status:
- Current simulation time: {current_time:.0f} seconds ({time_info['time_string']})
- Time period: {time_info['time_period']} - {time_info['period_description']}
- Checkpoint: {checkpoint_count}

Taxi Fleet Status:
- Fleet size: {fleet_state.get('fleet_size', 0)} taxis
- Idle taxis: {fleet_state.get('idle_count', 0)}
- Pickup taxis: {fleet_state.get('pickup_count', 0)}
- Occupied taxis: {fleet_state.get('occupied_count', 0)}
- Pending reservations: {pending_reservations.get('pending_count', 0)} waiting passengers
- Fleet utilization: {fleet_state.get('utilization_rate', 0):.1%}

Performance Metrics (Cumulative):
- Passenger Pickups: {pass_pickups}
- Passenger Dropoffs: {pass_dropoffs}
- Average Wait Time: {avg_wait_display} (Target: Minimize)

Optimization Time Window:
- From: {time_info['time_string']} ({time_info['time_period']})
- To: {next_time_info['time_string']} ({next_time_info['time_period']})
- Duration: {checkpoint_interval:.0f} seconds

Begin your analysis."""
                initial_prompt = append_user_query(initial_prompt, user_query)

                # Baseline policy simulation (default config with greedy fallback + randomCircling)
                baseline_simulation_result = None
                if checkpoint_path_t_minus_1 and baseline_control_configs:
                    try:
                        print(f"\n[Checkpoint {checkpoint_count}] Running baseline policy simulation for {checkpoint_interval:.0f}s...")
                        print(f"  Using checkpoint: {checkpoint_path_t_minus_1} (t-1 state, start of interval)")
                        baseline_raw = run_policy_simulation(
                            checkpoint_path=checkpoint_path_t_minus_1,
                            control_configs=baseline_control_configs,
                            duration=checkpoint_interval,
                            use_gui=use_gui,
                            config_path=config_path,
                            checkpoint_interval=checkpoint_interval,
                            run_duration=simulation_duration,
                            initial_control_states=checkpoint_start_control_states,
                            taxi_dispatch_algorithm=env_config.get("TAXI_DISPATCH_ALGORITHM"),
                            taxi_idle_algorithm=env_config.get("TAXI_IDLE_ALGORITHM"),
                        )

                        if baseline_raw.get("success"):
                            baseline_simulation_result = {
                                "success": True,
                                "stats": {
                                    "total_departed": baseline_raw.get("total_departed", 0),
                                    "total_arrived": baseline_raw.get("total_arrived", 0),
                                    "avg_travel_time": baseline_raw.get("avg_travel_time", 0),
                                    "duration": baseline_raw.get("duration", 0),
                                },
                                "module_metrics": baseline_raw.get("module_metrics", {}),
                                "error": None,
                                "control_configs": copy.deepcopy(baseline_control_configs),
                            }
                        else:
                            baseline_simulation_result = {
                                "success": False,
                                "stats": {},
                                "module_metrics": {},
                                "error": baseline_raw.get("error", "Baseline simulation failed"),
                                "control_configs": copy.deepcopy(baseline_control_configs),
                            }
                    except Exception as e:
                        baseline_simulation_result = {
                            "success": False,
                            "stats": {},
                            "module_metrics": {},
                            "error": f"Baseline simulation error: {e}",
                            "control_configs": copy.deepcopy(baseline_control_configs),
                        }

                # Display baseline simulation metrics if available
                if baseline_simulation_result and baseline_simulation_result.get("success"):
                    baseline_module_metrics = baseline_simulation_result.get("module_metrics", {})
                    if baseline_module_metrics:
                        print(f"\n  Baseline Simulation Module Performance Metrics:")
                        for module_name, metrics in baseline_module_metrics.items():
                            print(f"    {module_name.upper().replace('_', ' ')}:")
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, float):
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                                else:
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")

                # Build cleaned initial_best_result for agent (JSON-serializable, no module instances)
                # Use baseline simulation result as initial best for taxi
                initial_best_result_for_agent = None
                if baseline_simulation_result and baseline_simulation_result.get("success"):
                    initial_best_result_for_agent = baseline_simulation_result.copy()
                    if isinstance(initial_best_result_for_agent, dict) and "control_configs" in initial_best_result_for_agent:
                        initial_best_result_for_agent["control_configs"] = extract_configs_only(initial_best_result_for_agent["control_configs"]) or {}

                if baseline_simulation_result:
                    # Extract configs from simulation result before passing to logger
                    baseline_result_for_log = baseline_simulation_result.copy() if isinstance(baseline_simulation_result, dict) else baseline_simulation_result
                    if isinstance(baseline_result_for_log, dict) and "control_configs" in baseline_result_for_log:
                        baseline_result_for_log["control_configs"] = extract_configs_only(baseline_result_for_log["control_configs"]) or {}
                    checkpoint_logger.add_policy_simulation_result(
                        checkpoint_number=checkpoint_count,
                        simulation_result=baseline_result_for_log
                    )

                optimization_result = agent.run_optimization(
                    initial_prompt=initial_prompt,
                    context=agent_context,
                    env=env,
                    verbose=verbose,
                    # Start each checkpoint session from the default baseline and compare policies against it.
                    initial_best_result=initial_best_result_for_agent,
                    initial_control_configs=copy.deepcopy(baseline_control_configs) if baseline_control_configs else None,
                    checkpoint_logger=checkpoint_logger,
                    checkpoint_number=checkpoint_count,
                )

                # Update checkpoint log with LLM data
                checkpoint_logger.update_checkpoint_llm_messages(
                    checkpoint_number=checkpoint_count,
                    messages=agent.get_messages()
                )
                # Extract configs from optimization_result before passing to logger
                optimization_result_for_log = optimization_result.copy() if optimization_result else {}
                if "final_control_configs" in optimization_result_for_log:
                    optimization_result_for_log["final_control_configs"] = extract_configs_only(optimization_result_for_log["final_control_configs"]) or {}
                checkpoint_logger.update_checkpoint_optimization_result(
                    checkpoint_number=checkpoint_count,
                    optimization_result=optimization_result_for_log
                )

                # Save log
                log_file = checkpoint_logger.save_log()
                conversation_file = checkpoint_logger.save_all_conversations()
                if conversation_file:
                    print(f"\n  Conversations saved to: {conversation_file}")

                # Update control configs if optimization succeeded
                if optimization_result.get("success"):
                    final_control_configs = optimization_result.get("final_control_configs", {})
                    if final_control_configs:
                        print(f"\nLLM Agent completed optimization:")
                        print(f"  - Turns used: {optimization_result['turn_count']}/{max_agent_turns}")
                        
                        control_configs = final_control_configs.copy()
                        # Extract only configs (remove module instances) before passing to logger
                        control_configs_for_log = extract_configs_only(control_configs) if control_configs else {}
                        checkpoint_logger.update_checkpoint_control_configs(
                            checkpoint_number=checkpoint_count,
                            control_configs=control_configs_for_log
                        )
                        print(f"  - Control modules updated: {list(control_configs.keys())}")

                print(f"{'='*80}\n")

                # Collect traffic state
                current_time = env.get_current_time()
                traffic_state_collector.collect_if_needed(current_time)

                # Reset metrics for next checkpoint interval (no SUMO restart)
                # New logic: Continue simulation without loading checkpoint
                if remaining_duration > 0:
                    print(f"\n[Checkpoint {checkpoint_count}] Resetting metrics for next checkpoint interval (no SUMO restart)...")
                    try:
                        # Reset only metrics (travel time, waiting time, etc.) without restarting SUMO
                        env.reset_metrics()
                        print(f"  Metrics reset successfully")
                        print(f"  Continuing simulation from time: {env.get_current_time():.0f}s")

                        # Reset traffic state collector time after metrics reset
                        # The simulation continues, so we keep track relative to current checkpoint
                        checkpoint_start_time = env.get_current_time()
                        traffic_state_collector.last_collection_time = checkpoint_start_time - TRAFFIC_STATE_INTERVAL

                        # Note: Taxi control state is NOT reset, as it tracks the ongoing state of taxis
                        # The control state already exists and will continue to the next checkpoint

                        # Mark that first simulation is done (after first checkpoint)
                        if is_first_simulation:
                            is_first_simulation = False
                    except Exception as e:
                        print(f"  Error resetting metrics: {e}")
                        traceback.print_exc()
                        break
                elif is_first_simulation:
                    print(f"\n[Checkpoint {checkpoint_count}] First simulation completed.")
                    is_first_simulation = False  # Mark that first simulation is done
            else:
                # Simulation completed normally
                print(f"\n[Checkpoint {checkpoint_count}] Simulation completed normally")
                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)
                total_elapsed_time += results.get("duration", 0)

                final_time = env.get_current_time()
                traffic_state_collector.collect_if_needed(final_time)
                break

            if remaining_duration < step_seconds:
                print(f"\nRemaining duration ({remaining_duration:.0f}s) < step size, ending.")
                break

        # Final summary
        print("\n" + "=" * 80)
        print("Taxi Scheduling Simulation Completed!")
        print(f"  Total checkpoints: {checkpoint_count}")
        print(f"  Total elapsed time: {total_elapsed_time:.0f}s")
        print(f"  Total steps: {accumulated_results['total_steps']}")
        print(f"  Total Taxi Pickups: {accumulated_results.get('passenger_pickups', 'N/A')}")
        print(f"  Total Taxi Dropoffs: {accumulated_results.get('passenger_dropoffs', 'N/A')}")
        print(f"  Total Departed (All Vehicles): {accumulated_results['total_departed']}")
        print(f"  Total Arrived (All Vehicles): {accumulated_results['total_arrived']}")
        print("=" * 80)

        # Calculate and display average module metrics across all checkpoints
        if control_modules and accumulated_results.get("checkpoints"):
            print("\n" + "=" * 80)
            print("Average Control Module Performance Metrics (Across All Checkpoints)")
            print("=" * 80)

            average_module_metrics = calculate_average_module_metrics(
                checkpoints=accumulated_results["checkpoints"],
                control_modules=control_modules
            )

            if average_module_metrics:
                for module_name in control_modules:
                    if module_name not in average_module_metrics:
                        print(f"\n{module_name.upper().replace('_', ' ')}: No metrics available")
                        continue

                    print(f"\n{module_name.upper().replace('_', ' ')}:")
                    metrics = average_module_metrics[module_name]

                    # Sort metrics for consistent display
                    sorted_metrics = sorted(metrics.items())
                    for metric_name, metric_value in sorted_metrics:
                        if isinstance(metric_value, float):
                            # Display throughput as integer (vehicle count), others as float
                            if metric_name == "throughput":
                                print(f"  - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                            else:
                                print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
                        else:
                            print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value}")

                # Store average metrics in result
                accumulated_results["average_module_metrics"] = average_module_metrics
            else:
                print("\nNo module metrics available for averaging.")

            print("=" * 80)

        if env:
            final_avg_travel_time = env.get_average_travel_time()
            accumulated_results["final_avg_travel_time"] = final_avg_travel_time
            accumulated_results["final_time"] = env.get_current_time()

        result = {
            "status": "success",
            "checkpoint_count": checkpoint_count,
            "total_elapsed_time": total_elapsed_time,
            "simulation_id": simulation_id,
            "results": accumulated_results
        }

        stats = traffic_state_collector.get_stats()
        result["traffic_state_file"] = stats["filepath"]
        result["traffic_state_snapshots"] = stats["snapshot_count"]
        
        # Finish wandb run
        if wandb_run:
            # Log final summary metrics
            final_metrics = {
                "final/checkpoint_count": checkpoint_count,
                "final/total_elapsed_time": total_elapsed_time,
                "final/total_steps": accumulated_results["total_steps"],
                "final/total_departed": accumulated_results["total_departed"],
                "final/total_arrived": accumulated_results["total_arrived"]
            }
            if accumulated_results.get("final_avg_travel_time"):
                final_metrics["final/avg_travel_time"] = accumulated_results["final_avg_travel_time"]
            if accumulated_results.get("passenger_pickups"):
                final_metrics["final/passenger_pickups"] = accumulated_results["passenger_pickups"]
            if accumulated_results.get("passenger_dropoffs"):
                final_metrics["final/passenger_dropoffs"] = accumulated_results["passenger_dropoffs"]
            
            # Add average module metrics
            if accumulated_results.get("average_module_metrics"):
                for module_name, metrics in accumulated_results["average_module_metrics"].items():
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            final_metrics[f"final/{module_name}/{metric_name}"] = float(metric_value)
            
            wandb_run.log(final_metrics)
            wandb_run.finish()
            print("Logged final metrics to wandb")

        return result

    except Exception as e:
        error_msg = f"Taxi scheduling simulation failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        traceback.print_exc()

        # Calculate and display average module metrics if available
        if 'control_modules' in locals() and control_modules and accumulated_results.get("checkpoints"):
            print("\n" + "=" * 80)
            print("Average Control Module Performance Metrics (Up to Error)")
            print("=" * 80)

            average_module_metrics = calculate_average_module_metrics(
                checkpoints=accumulated_results["checkpoints"],
                control_modules=control_modules
            )

            if average_module_metrics:
                for module_name in control_modules:
                    if module_name not in average_module_metrics:
                        print(f"\n{module_name.upper().replace('_', ' ')}: No metrics available")
                        continue

                    print(f"\n{module_name.upper().replace('_', ' ')}:")
                    metrics = average_module_metrics[module_name]

                    # Sort metrics for consistent display
                    sorted_metrics = sorted(metrics.items())
                    for metric_name, metric_value in sorted_metrics:
                        if isinstance(metric_value, float):
                            # Display throughput as integer (vehicle count), others as float
                            if metric_name == "throughput":
                                print(f"  - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                            else:
                                print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
                        else:
                            print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value}")

            print("=" * 80)
            accumulated_results["average_module_metrics"] = average_module_metrics

        result = {
            "status": "error",
            "error": error_msg,
            "checkpoint_count": checkpoint_count,
            "results": accumulated_results
        }
        
        if 'traffic_state_collector' in locals():
            stats = traffic_state_collector.get_stats()
            result["traffic_state_file"] = stats["filepath"]
            result["traffic_state_snapshots"] = stats["snapshot_count"]
        
        return result

    finally:
        if env:
            try:
                env.close()
                print("Environment closed.")
            except:
                pass


def main():
    """Main entry point for taxi scheduling simulation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SUMO taxi scheduling simulation with LLM-controlled dispatch and repositioning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data/zhouyuping/Zone/zone_scenarios/Manhattan/Manhattan.sumocfg",
        help="Path to SUMO config file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600,
        help="Total simulation duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=1800,
        help="Checkpoint interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=10,
        help="Simulation step size in seconds (default: 10)"
    )
    parser.add_argument(
        "--time-to-teleport",
        type=float,
        default=None,
        help="SUMO time-to-teleport override (default: read from sumocfg or 300)"
    )
    parser.add_argument(
        "--gui",
        default=False,
        action="store_true",
        help="Use sumo-gui instead of sumo"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for simulation"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="siliconflow/deepseek-ai/DeepSeek-V3.2",
        help="LLM model name for agent (default: siliconflow/deepseek-ai/DeepSeek-V3.2)"
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=10,
        help="Maximum dialogue turns for LLM agent (default: 10)"
    )
    parser.add_argument(
        "--control-modules",
        type=str,
        nargs='+',
        default=['taxi_scheduling'],
        choices=['taxi_scheduling', 'signal_timing'],
        help="Control modules to enable (default: taxi_scheduling)"
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default='ChatCity',
        help="Wandb project name (default: ChatCity)"
    )
    parser.add_argument(
        "--traffic-state-interval",
        type=float,
        default=60,
        help="Interval for collecting traffic state data in seconds (default: 60)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM agent sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--max-reflection-turns",
        type=int,
        default=5,
        help="Maximum number of reflection turns (default: 5)"
    )
    parser.add_argument(
        "--max-interval-retries",
        type=int,
        default=2,
        help="Max retries per checkpoint interval on SUMO/TraCI failure (default: 2)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for API (if None, will use default from provider)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional user instruction appended to the default LLM optimization prompt"
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = resolve_config_path(args.config, workspace_root)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Config file: {config_path}")
    
    # Run simulation
    results = run_taxi_scheduling_simulation(
        config_path=str(config_path),
        simulation_duration=args.duration,
        checkpoint_interval=args.checkpoint_interval,
        step_seconds=args.step_seconds,
        use_gui=args.gui,
        seed=args.seed,
        time_to_teleport=args.time_to_teleport,
        llm_model=args.llm_model,
        max_agent_turns=args.max_agent_turns,
        control_modules=args.control_modules,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        traffic_state_interval=args.traffic_state_interval,
        max_reflection_turns=args.max_reflection_turns,
        max_interval_retries=args.max_interval_retries,
        temperature=args.temperature,
        base_url=getattr(args, 'base_url', None),
        user_query=args.query,
    )

    if results["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
