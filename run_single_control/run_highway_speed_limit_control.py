"""
Agent-controlled simulation runner with checkpoint-based control for highway speed limits.

This script implements a checkpoint-based simulation control system where:
1. Simulation runs for checkpoint_interval seconds
2. Saves a snapshot checkpoint
3. LLM agent performs optimization
4. Loads checkpoint and continues simulation
5. Repeats until simulation_duration is reached

Currently supports:
- highway_speed_limit: Highway speed limit control

Future support (TODO):
- subway_scheduling: Subway scheduling control
- bus_scheduling: Bus scheduling control
"""

import os
import sys
import copy
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
from datetime import datetime

# Add project root to path BEFORE importing utils modules
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent  # Go up from run_single_control/ to project root
sys.path.insert(0, str(workspace_root))

from utils.llm_agent import LLMAgent
from utils.path_utils import resolve_config_path
from utils.prompt_utils import append_user_query

# Try to import wandb, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from utils.simulation_utils import (
    create_sumo_env,
    run_controlled_simulation,
    run_policy_simulation,
    TrafficStateCollector
)
from utils.id_utils import generate_simulation_identifiers
from utils.checkpoint_logger import CheckpointLogger, extract_configs_only, effective_config
from utils.traffic_state_collector import init_traffic_states_file


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
    
    Args:
        seconds: Simulation time in seconds
        
    Returns:
        Dictionary with formatted time information:
        - hours: Hour of day (0-23)
        - minutes: Minute of hour (0-59)
        - time_string: Formatted time string (HH:MM)
        - time_period: Time period description (e.g., "Morning Rush", "Evening Rush", "Night")
        - day_of_simulation: Day number (assuming 24-hour days)
    """
    total_hours = seconds / 3600.0
    day_hour = total_hours % 24  # Hour within a 24-hour day
    hours = int(day_hour)
    minutes = int((day_hour - hours) * 60)
    day_number = int(total_hours / 24) + 1  # Day 1, 2, 3, etc.
    
    # Format time string
    time_string = f"{hours:02d}:{minutes:02d}"
    
    # Determine time period
    if 6 <= hours < 11:
        time_period = "Morning Rush Hour"
        period_description = "Peak morning traffic period (6:00-11:00). Prioritize inbound traffic to city center."
    elif 11 <= hours < 14:
        time_period = "Midday"
        period_description = "Midday period (11:00-14:00). Moderate traffic levels."
    elif 14 <= hours < 16:
        time_period = "Afternoon"
        period_description = "Afternoon period (14:00-16:00). Moderate traffic levels."
    elif 16 <= hours < 21:
        time_period = "Evening Rush Hour"
        period_description = "Peak evening traffic period (16:00-21:00). Prioritize outbound traffic from city center."
    elif 21 <= hours < 24:
        time_period = "Evening"
        period_description = "Evening period (21:00-24:00). Low to moderate traffic levels."
    else:  # 0 <= hours < 6
        time_period = "Night"
        period_description = "Night period (00:00-06:00). Low traffic levels. Higher speed limits may be optimal."
    
    return {
        "hours": hours,
        "minutes": minutes,
        "time_string": time_string,
        "time_period": time_period,
        "period_description": period_description,
        "day_of_simulation": day_number,
        "total_hours": total_hours
    }


def run_checkpoint_based_simulation(
    config_path: str,
    simulation_duration: float,
    checkpoint_interval: float,
    step_seconds: int = 30,
    use_gui: bool = False,
    seed: Optional[int] = None,
    llm_model: str = "openai/gpt-4o-mini",
    max_agent_turns: int = 10,
    control_modules: Optional[List[str]] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    traffic_state_interval: float = 300,
    max_reflection_turns: int = 5,
    max_interval_retries: int = 2,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    verbose: bool = True,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run simulation with checkpoint-based control for LLM agent optimization of highway speed limits.
    
    Currently supports highway_speed_limit only. Other control types (subway_scheduling, 
    bus_scheduling) are planned for future implementation.
    
    Args:
        config_path: Path to SUMO config file
        simulation_duration: Total simulation duration in seconds
        checkpoint_interval: Interval between checkpoints in seconds
        step_seconds: Step size for simulation (default: 30)
        use_gui: Whether to use sumo-gui (default: False)
        seed: Random seed for simulation (default: None)
        llm_model: LLM model name for agent (default: "openai/gpt-4o-mini")
        max_agent_turns: Maximum dialogue turns for LLM agent (default: 10)
        traffic_state_interval: Interval for collecting traffic state data in seconds (default: 300)
        max_interval_retries: Max retries per checkpoint interval on SUMO/TraCI failure
        
    Returns:
        Dictionary with final simulation results
    """
    # Create checkpoint directory
    checkpoint_dir = workspace_root / "records" / "checkpoints"
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
        llm_model_name=llm_model  # Include LLM model name in log filename
    )
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        config_path_obj = Path(config_path)
        config_dir_name = config_path_obj.parent.name
        
        # Prepare control modules list for group name
        if control_modules is None:
            control_modules = ['highway_speed_limit']  # Default
        
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
            project=wandb_project or "sumo_highway_speed_limit_control",
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

            # CRITICAL: Set PYTHONHASHSEED to make hash-based collections (set, dict) deterministic
            os.environ['PYTHONHASHSEED'] = '0'

            random.seed(seed)
            np.random.seed(seed)
            print(f"Set Python random seed: {seed}")
            print(f"Set NumPy random seed: {seed}")
            print(f"Set PYTHONHASHSEED: 0 (for deterministic set/dict iteration)")

        # First checkpoint: initialize new environment
        print("=" * 80)
        print(f"Starting checkpoint-based simulation for Highway Speed Limit Control")
        print(f"  Total duration: {simulation_duration}s")
        print(f"  Checkpoint interval: {checkpoint_interval}s")
        print(f"  Expected checkpoints: {int(simulation_duration / checkpoint_interval)}")
        print("=" * 80)
        
        # Create initial SUMO environment with user-selected control modules
        if control_modules is None:
            control_modules = ['highway_speed_limit']  # Default to highway speed limit control only
        
        print(f"\n[Checkpoint {checkpoint_count}] Creating new environment...")
        print(f"  Enabled control modules: {control_modules}")
        # Extract config directory name from config_path (e.g., "jinan" from "sumo_config/jinan/jinan.sumocfg")
        config_path_obj = Path(config_path)
        config_dir_name = config_path_obj.parent.name  # e.g., "jinan"
        
        # Build control modules list, avoiding duplicates while preserving order
        enabled_modules = control_modules if control_modules else []
        always_enabled = ['signal_timing', 'ramp_metering']   # Modules that are always enabled
        
        env, env_config = create_sumo_env(
            config_path=config_path,
            use_gui=use_gui,
            seed=seed,
            control_modules=list(set(enabled_modules + always_enabled)),
            run_counts=int(simulation_duration),  # Set RUN_COUNTS to match checkpoint_interval
            config_dir_name=config_dir_name  # Pass config_dir_name for control module initialization
        )
        
        # Initialize traffic states file (only once, at the start)
        if traffic_states_filepath is None:
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=simulation_id,
                config_name=config_dir_name,
                llm_name=llm_model,
                control_modules=control_modules
            )
            print(f"Initialized traffic states file: {traffic_states_filepath.name}")
        
        # Build road network graphs for connectivity analysis and lane metadata
        # Get road network graphs (already built during env.reset())
        print("Getting road network graphs...")
        graphs = {}
        lane_dict = None
        lane_inter_graph = None
        try:
            graphs = env.get_road_network_graphs()
            lane_dict = graphs.get("lane_dict", {})
            lane_inter_graph = graphs.get("lane_inter_graph")
            print(f"Retrieved graphs: {list(graphs.keys())}")
            print(f"  Lane dict entries: {len(lane_dict)}")
        except Exception as e:
            print(f"Warning: Failed to get graphs: {e}")
            traceback.print_exc()
        
        # Create traffic state collector with lane metadata
        traffic_state_collector = TrafficStateCollector(
            env=env,
            traffic_states_filepath=traffic_states_filepath,
            interval=TRAFFIC_STATE_INTERVAL,
            lane_dict=lane_dict,
            lane_inter_graph=lane_inter_graph,
            simulation_id=simulation_id
        )
        print(f"Created traffic state collector (interval: {TRAFFIC_STATE_INTERVAL}s)")
        
        # Initialize LLM agent (reused across all checkpoints)
        agent = LLMAgent(
            model_name=llm_model,
            temperature=temperature,
            max_turns=max_agent_turns,
            available_control_modules=control_modules,
            config_name=config_dir_name,  # Pass config_name for file naming
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

            # Map control types to module names
            # control_type (in env.enabled_controls) -> module_name (for control_configs)
            control_type_to_module_name = {
                'signal_timing': 'signal_timing',
                'highway_speed_limit': 'highway_speed_limit',
                'ramp_metering': 'ramp_metering',
                'subway_scheduling': 'subway_scheduling',
                'bus_scheduling': 'bus_scheduling'
            }

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
            
            if control_configs:
                print(f"  Control modules initialized: {list(control_configs.keys())}")
            else:
                print(f"  No valid control configs found, running baseline simulation")
                control_configs = None
        else:
            print(f"\nNo control modules enabled, running baseline simulation")
        
        # Get highway speed limit config for context
        highway_speed_limit_config = env.enabled_controls.get('highway_speed_limit', {}).get('config', {}) if hasattr(env, 'enabled_controls') else {}
        
        # Run simulation loop with checkpoints
        remaining_duration = simulation_duration
        previous_checkpoint_path = None  # Track previous checkpoint path for t-1 state
        is_first_simulation = True  # Flag to mark first simulation (always initialize new env, never load checkpoint)
        max_interval_retries = max(0, int(max_interval_retries))

        while remaining_duration > 0:
            checkpoint_count += 1
            current_checkpoint_duration = min(checkpoint_interval, remaining_duration)
            
            print(f"\n[Checkpoint {checkpoint_count}] Running simulation for {current_checkpoint_duration:.0f}s...")
            print(f"  Remaining duration: {remaining_duration:.0f}s")
            if control_configs:
                print(f"  Control modules: {list(control_configs.keys())}")
            else:
                print(f"  Control: None (baseline)")
            remaining_duration -= current_checkpoint_duration
            
            # Run simulation until checkpoint
            # Pass previous_checkpoint_path via control_states if available
            if control_states is None:
                control_states = {}
            if previous_checkpoint_path:
                control_states["_previous_checkpoint_path"] = previous_checkpoint_path
            
            retry_attempts = 0
            while True:
                results = run_controlled_simulation(
                    env=env,
                    duration=current_checkpoint_duration,
                    step_seconds=step_seconds,
                    traffic_state_collector=traffic_state_collector,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_prefix=f"checkpoint_{checkpoint_count}",
                    control_configs=control_configs,  # Apply control if configured
                    control_states=control_states,  # Maintain control state across steps
                    is_first_simulation=is_first_simulation,  # Mark first simulation
                    config_name=config_dir_name,  # Pass config_name for file naming
                    llm_name=llm_model,  # Pass llm_name for file naming
                    simulation_id=simulation_id  # Pass simulation_id for checkpoint filename
                )

                if results.get("aborted") and not results.get("checkpoint_reached", False):
                    abort_reason = results.get("abort_reason", "unknown")
                    if retry_attempts < max_interval_retries:
                        retry_attempts += 1
                        print(
                            f"\n[Checkpoint {checkpoint_count}] Interval aborted early ({abort_reason}). "
                            f"Retrying from interval start ({retry_attempts}/{max_interval_retries})..."
                        )
                        retry_path = results.get("checkpoint_path_t_minus_1")
                        if not retry_path or not Path(retry_path).exists():
                            raise RuntimeError(
                                f"Retry failed: t-1 checkpoint not found for checkpoint {checkpoint_count}"
                            )

                        loaded_ok = False
                        if hasattr(env, "load_state_inplace"):
                            loaded_ok = env.load_state_inplace(retry_path)
                        else:
                            env.reset(use_gui=use_gui, seed=seed, load_state_path=retry_path)
                            loaded_ok = True
                        if not loaded_ok:
                            raise RuntimeError(
                                f"Retry failed: could not load t-1 checkpoint {retry_path}"
                            )

                        metadata = _load_checkpoint_metadata(retry_path)
                        control_configs, control_states = _restore_control_state_from_metadata(
                            env,
                            control_configs,
                            control_states,
                            metadata=metadata
                        )

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
            
            # Update control states if control was applied
            if results.get("control_states"):
                control_states = results["control_states"]
                # Remove internal metadata from control_states before next iteration
                if "_previous_checkpoint_path" in control_states:
                    del control_states["_previous_checkpoint_path"]
            
            # Check if checkpoint was reached
            if results.get("checkpoint_reached", False):
                checkpoint_path = results.get("checkpoint_path")  # t state
                checkpoint_path_t_minus_1 = results.get("checkpoint_path_t_minus_1")  # t-1 state
                elapsed_time = results.get("elapsed_time", 0)
                
                # Update previous_checkpoint_path for next checkpoint's t-1 state
                if checkpoint_path:
                    previous_checkpoint_path = checkpoint_path
                
                print(f"\n[Checkpoint {checkpoint_count}] Checkpoint reached!")
                print(f"  Elapsed time: {elapsed_time:.0f}s")
                print(f"  Remaining duration: {remaining_duration:.0f}s")
                print(f"  Checkpoint saved: {checkpoint_path}")
                
                # Save accumulated traffic state snapshots at checkpoint (if not already saved in run_controlled_simulation)
                if traffic_state_collector is not None:
                    current_time = results.get("current_time", elapsed_time)
                    traffic_state_collector.save_checkpoint_snapshots(checkpoint_time=current_time)
                
                # Accumulate results
                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)
                
                # Display module metrics if available
                # Filter module_metrics to only include control_modules (exclude signal_timing if not in control_modules)
                module_metrics = results.get("module_metrics", {})
                if module_metrics and control_modules:
                    # Only include modules that are in control_modules
                    filtered_module_metrics = {
                        name: metrics for name, metrics in module_metrics.items()
                        if name in control_modules
                    }
                    module_metrics = filtered_module_metrics
                
                if module_metrics:
                    print(f"\n  Control Module Performance Metrics:")
                    for module_name, metrics in module_metrics.items():
                        print(f"    {module_name.upper().replace('_', ' ')}:")
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, float):
                                # Display throughput as integer (vehicle count), others as float
                                if metric_name == "throughput":
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                                else:
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                            else:
                                print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")
                
                checkpoint_info = {
                    "checkpoint_number": checkpoint_count,
                    "checkpoint_path": checkpoint_path,
                    "elapsed_time": elapsed_time,
                    "remaining_duration": remaining_duration,
                    "step_count": results.get("step_count", 0),
                    "avg_travel_time": results.get("avg_travel_time", 0)
                }
                
                # Add module metrics to checkpoint info (already filtered)
                if module_metrics:
                    checkpoint_info["module_metrics"] = module_metrics
                
                accumulated_results["checkpoints"].append(checkpoint_info)
                
                total_elapsed_time += elapsed_time

                # Log metrics to wandb at checkpoint
                if wandb_run:
                    log_metrics_to_wandb(
                        wandb_run=wandb_run,
                        checkpoint_number=checkpoint_count,
                        elapsed_time=elapsed_time,
                        module_metrics=module_metrics,
                        control_modules=control_modules,
                        additional_metrics={
                            "step_count": results.get("step_count", 0),
                            "avg_travel_time": results.get("avg_travel_time", 0),
                            "total_departed": results.get("total_departed", 0),
                            "total_arrived": results.get("total_arrived", 0),
                            "remaining_duration": remaining_duration
                        }
                    )

                # Log checkpoint before LLM optimization (will update with LLM data after)
                # Note: control_configs will be updated after LLM optimization if new configs are generated
                # Extract only configs (remove module instances) before passing to logger
                control_configs_for_log = extract_configs_only(control_configs) if control_configs else None
                checkpoint_logger.log_checkpoint(
                    checkpoint_number=checkpoint_count,
                    checkpoint_path=checkpoint_path,
                    checkpoint_path_t_minus_1=checkpoint_path_t_minus_1,
                    elapsed_time=elapsed_time,
                    remaining_duration=remaining_duration,
                    step_count=results.get("step_count", 0),
                    avg_travel_time=results.get("avg_travel_time", 0),
                    module_metrics=module_metrics,
                    control_configs=control_configs_for_log  # Current configs before optimization (configs only)
                )
                
                # LLM Agent optimization process
                print(f"\n{'='*80}")
                print(f"[Checkpoint {checkpoint_count}] LLM Agent Optimization")
                print(f"{'='*80}")
                
                # Prepare context for LLM agent
                # Use control_configs (actual configs being used) instead of loading from file
                # This ensures sandbox gets the initialized configs, not empty dicts from non-existent files
                agent_context = {
                    # Layer 3: Module-specific graphs (intersection-related only)
                    "lane_inter_graph": graphs.get("lane_inter_graph"),
                    "intersection_graph": graphs.get("intersection_graph"),
                    "lane_dict": graphs.get("lane_dict"),  # Only intersection-related lanes

                    # Highway-specific data
                    "highway_graph": env.highway_subgraph if hasattr(env, 'highway_subgraph') else None,
                    "highway_segment_dict": env.highway_info_dict if hasattr(env, 'highway_info_dict') else {},
                    "highway_segment_graph": graphs.get("highway_segment_graph") or (env.get_highway_segment_graph() if hasattr(env, 'get_highway_segment_graph') else None),

                    # Layer 1: Foundation Layer - Complete network graphs
                    "network_graphs": env.network_graphs if hasattr(env, 'network_graphs') else {},
                    "network_dicts": env.network_dicts if hasattr(env, 'network_dicts') else {},
                    "full_lane_graph": env.network_graphs.get("lane_graph") if hasattr(env, 'network_graphs') else None,
                    "full_road_graph": env.network_graphs.get("road_graph") if hasattr(env, 'network_graphs') else None,
                    "full_lane_dict": env.network_dicts.get("lane_dict") if hasattr(env, 'network_dicts') else {},
                    "full_road_dict": env.network_dicts.get("road_dict") if hasattr(env, 'network_dicts') else {},

                    # Layer 2: Zone infrastructure
                    "zone_dict": env.zone_dict if hasattr(env, 'zone_dict') else {},
                    "zone_graph": env.zone_graph if hasattr(env, 'zone_graph') else None,

                    # Simulation context
                    "traffic_states_filepath": str(traffic_states_filepath),
                    "simulation_id": simulation_id,
                    "current_configs": control_configs,
                    "checkpoint_path": checkpoint_path,
                    "config_path": config_path,
                    "checkpoint_interval": checkpoint_interval,
                    "test_duration": checkpoint_interval,
                    "remaining_duration": remaining_duration,
                    "use_gui": use_gui,
                    "config_name": config_dir_name,
                    "llm_name": llm_model,
                    "control_modules": control_modules,
                    "run_duration": simulation_duration,
                    "seed": seed
                }
                
                # Convert checkpoint simulation results to format compatible with agent's best result tracking
                # This will be used as the initial best result for this checkpoint optimization
                # Filter module_metrics to only include control_modules (already filtered above)
                # Use extract_configs_only so initial_best_result is JSON-serializable (no module instances)
                checkpoint_simulation_result = {
                    "success": True,
                    "stats": {
                        "total_departed": results.get("total_departed", 0),
                        "total_arrived": results.get("total_arrived", 0),
                        "avg_travel_time": results.get("avg_travel_time", 0),
                        "duration": elapsed_time
                    },
                    "module_metrics": module_metrics.copy() if module_metrics else {},
                    "control_configs": extract_configs_only(control_configs) if control_configs else {}
                }
                
                print(f"  Initial best result (from checkpoint simulation):")
                print(f"    - Average travel time: {results.get('avg_travel_time', 0):.2f}s")
                print(f"    - Total arrived: {results.get('total_arrived', 0)}")
                if module_metrics and 'highway_speed_limit' in module_metrics:
                    hw_metrics = module_metrics['highway_speed_limit']
                    print(f"    - Highway avg travel time: {hw_metrics.get('avg_travel_time', 'N/A'):.2f}s" if isinstance(hw_metrics.get('avg_travel_time'), (int, float)) else f"    - Highway avg travel time: N/A")
                    # Display throughput (now represents vehicle count, not veh/s)
                    throughput = hw_metrics.get('throughput')
                    if throughput is not None and isinstance(throughput, (int, float)):
                        print(f"    - Highway throughput: {int(throughput)} vehicles")
                    else:
                        print(f"    - Highway throughput: N/A")

                # Run baseline simulation using current control_configs
                baseline_simulation_result = None
                if checkpoint_path_t_minus_1 and control_configs:
                    try:
                        print(f"\n[Checkpoint {checkpoint_count}] Running baseline policy simulation for {checkpoint_interval:.0f}s...")
                        print(f"  Using checkpoint: {checkpoint_path_t_minus_1} (t-1 state, start of interval)")
                        baseline_raw = run_policy_simulation(
                            checkpoint_path=checkpoint_path_t_minus_1,
                            control_configs=copy.deepcopy(control_configs),
                            duration=checkpoint_interval,
                            use_gui=use_gui,
                            config_path=config_path,
                            checkpoint_interval=checkpoint_interval,
                            run_duration=simulation_duration,
                            seed=seed,
                        )

                        if baseline_raw.get("success"):
                            # Include policy_highway_avg_travel_time and policy_avg_travel_time in stats
                            # so _is_better_simulation_result uses the same metric for baseline vs control comparison
                            baseline_stats = {
                                "total_departed": baseline_raw.get("total_departed", 0),
                                "total_arrived": baseline_raw.get("total_arrived", 0),
                                "avg_travel_time": baseline_raw.get("avg_travel_time", 0),
                                "duration": baseline_raw.get("duration", 0)
                            }
                            if baseline_raw.get("policy_highway_avg_travel_time") is not None:
                                baseline_stats["policy_highway_avg_travel_time"] = baseline_raw["policy_highway_avg_travel_time"]
                            if baseline_raw.get("policy_avg_travel_time") is not None:
                                baseline_stats["policy_avg_travel_time"] = baseline_raw["policy_avg_travel_time"]
                            baseline_simulation_result = {
                                "success": True,
                                "stats": baseline_stats,
                                "module_metrics": baseline_raw.get("module_metrics", {}),
                                "error": None,
                                "control_configs": copy.deepcopy(control_configs),
                            }
                        else:
                            baseline_simulation_result = {
                                "success": False,
                                "stats": {},
                                "module_metrics": {},
                                "error": baseline_raw.get("error", "Baseline simulation failed"),
                                "control_configs": copy.deepcopy(control_configs),
                            }
                    except Exception as e:
                        baseline_simulation_result = {
                            "success": False,
                            "stats": {},
                            "module_metrics": {},
                            "error": f"Baseline simulation error: {e}",
                            "control_configs": copy.deepcopy(control_configs),
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
                # Use baseline simulation result as initial best if available, otherwise use checkpoint simulation result
                initial_best_result_for_agent = None
                if baseline_simulation_result and baseline_simulation_result.get("success"):
                    initial_best_result_for_agent = baseline_simulation_result.copy()
                    if isinstance(initial_best_result_for_agent, dict) and "control_configs" in initial_best_result_for_agent:
                        initial_best_result_for_agent["control_configs"] = extract_configs_only(initial_best_result_for_agent["control_configs"]) or {}
                else:
                    # Fallback to checkpoint simulation result if baseline failed
                    initial_best_result_for_agent = checkpoint_simulation_result

                if baseline_simulation_result:
                    # Extract configs from simulation result before passing to logger
                    baseline_result_for_log = baseline_simulation_result.copy() if isinstance(baseline_simulation_result, dict) else baseline_simulation_result
                    if isinstance(baseline_result_for_log, dict) and "control_configs" in baseline_result_for_log:
                        baseline_result_for_log["control_configs"] = extract_configs_only(baseline_result_for_log["control_configs"]) or {}
                    checkpoint_logger.add_policy_simulation_result(
                        checkpoint_number=checkpoint_count,
                        simulation_result=baseline_result_for_log
                    )
                
                # Create initial prompt for LLM agent
                current_time = env.get_current_time()
                total_duration = simulation_duration
                elapsed_time = current_time
                remaining_time = total_duration - elapsed_time
                
                # Format simulation time to human-readable format
                time_info = format_simulation_time(current_time)
                
                # Calculate next checkpoint time
                next_checkpoint_time = current_time + checkpoint_interval
                next_checkpoint_time_info = format_simulation_time(next_checkpoint_time)
                
                # Get highway segment count
                num_highway_segments = len(highway_speed_limit_config) if highway_speed_limit_config else 0
                
                initial_prompt = f"""You are optimizing highway speed limits for a city transportation network.

Current Status:
- Current simulation time: {current_time:.0f} seconds ({time_info['time_string']})
- Total simulation duration: {total_duration:.0f} seconds ({total_duration/3600:.2f} hours)
- Elapsed time: {elapsed_time:.0f} seconds ({elapsed_time/3600:.2f} hours)
- Remaining time: {remaining_time:.0f} seconds ({remaining_time/3600:.2f} hours)
- Progress: {elapsed_time/total_duration*100:.1f}%
- Checkpoint: {checkpoint_count}
- Number of highway segments: {num_highway_segments}

**Optimization Time Window:**
- **Optimization Start**: {time_info['time_string']} (Day {time_info['day_of_simulation']}, {time_info['time_period']})
- **Optimization End**: {next_checkpoint_time_info['time_string']} (Day {next_checkpoint_time_info['day_of_simulation']}, {next_checkpoint_time_info['time_period']})
- **Duration**: {checkpoint_interval:.0f} seconds ({checkpoint_interval/3600:.2f} hours)
- **Your optimized speed limits will be applied from {time_info['time_string']} to {next_checkpoint_time_info['time_string']}**

For configuration format, constraints, optimization strategies, and task description, follow the highway_speed_limit module's domain knowledge (provided via GET_CONTROL_API).

Available time range for analysis: 0 to {current_time:.0f} seconds (00:00 to {time_info['time_string']}). You can ONLY use the traffic snapshot data within this range.

Begin your analysis."""
                initial_prompt = append_user_query(initial_prompt, user_query)
                
                optimization_result = agent.run_optimization(
                    initial_prompt=initial_prompt,
                    context=agent_context,
                    env=env,  # Pass env to access enabled control modules
                    verbose=verbose,
                    initial_best_result=initial_best_result_for_agent,  # Use baseline simulation result as initial best
                    initial_control_configs=copy.deepcopy(control_configs) if control_configs else None,  # Pass current control configs
                    checkpoint_logger=checkpoint_logger,
                    checkpoint_number=checkpoint_count,
                )
                
                # Extract policy simulation results from LLM agent history
                policy_simulation_results = []
                if optimization_result.get("history"):
                    for action_type, action_result in optimization_result.get("history", []):
                        if action_type == "SIMULATION" and isinstance(action_result, dict):
                            policy_simulation_results.append(action_result)
                
                # Update checkpoint log with LLM agent data
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
                # Add all policy simulation results to checkpoint log
                for sim_result in policy_simulation_results:
                    # Extract configs from simulation result before passing to logger
                    sim_result_for_log = sim_result.copy() if isinstance(sim_result, dict) else sim_result
                    if isinstance(sim_result_for_log, dict) and "control_configs" in sim_result_for_log:
                        sim_result_for_log["control_configs"] = extract_configs_only(sim_result_for_log["control_configs"]) or {}
                    checkpoint_logger.add_policy_simulation_result(
                        checkpoint_number=checkpoint_count,
                        simulation_result=sim_result_for_log
                    )
                
                # Save all checkpoint conversations to JSON file after each checkpoint optimization
                log_file = checkpoint_logger.save_log()
                conversation_file = checkpoint_logger.save_all_conversations()
                if conversation_file:
                    print(f"\n  All checkpoint conversations saved to: {conversation_file}")
                
                # Update control configs if optimization succeeded
                if optimization_result.get("success"):
                    # Get all control configs from optimization result
                    final_control_configs = optimization_result.get("final_control_configs", {})
                    
                    # Backward compatibility: if final_control_configs not available, use final_signal_config
                    if not final_control_configs and optimization_result.get("final_highway_speed_limit_config"):
                        final_control_configs = {"highway_speed_limit": optimization_result["final_highway_speed_limit_config"]}
                    
                    if final_control_configs:
                        print(f"\nLLM Agent completed optimization:")
                        print(f"  - Turns used: {optimization_result['turn_count']}/{max_agent_turns}")
                        print(f"  - Control modules updated: {list(final_control_configs.keys())}")
                        
                        # Update each control module's config
                        for module_name, module_config in final_control_configs.items():
                            if module_name == "highway_speed_limit":
                                # Update highway speed limit config
                                from control_modules import get_control_module
                                highway_module = get_control_module("highway_speed_limit")
                                if highway_module:
                                    highway_speed_limit_config = module_config  # Update local variable
                        
                        # Prepare control_configs for next checkpoint
                        # Use all optimized configs from LLM
                        control_configs = final_control_configs.copy()
                        
                        # Update checkpoint log with new control configs
                        # Extract only configs (remove module instances) before passing to logger
                        control_configs_for_log = extract_configs_only(control_configs) if control_configs else {}
                        checkpoint_logger.update_checkpoint_control_configs(
                            checkpoint_number=checkpoint_count,
                            control_configs=control_configs_for_log
                        )
                        
                        # Reset control states to initialize with new configs
                        control_states = None
                        print(f"  - Control modules will be applied in next checkpoint: {list(control_configs.keys())}")
                    else:
                        print(f"\nLLM Agent optimization did not produce new configuration.")
                        print(f"  - Continuing with current configuration...")
                else:
                    print(f"\nLLM Agent optimization did not succeed.")
                    print(f"  - Reason: {optimization_result.get('error', 'Unknown')}")
                    print(f"  - Continuing with current configuration...")
                
                print(f"{'='*80}\n")
                
                # Collect final traffic state before checkpoint if needed
                current_time = env.get_current_time()
                traffic_state_collector.collect(current_time)

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
                # Simulation completed normally (no checkpoint reached)
                print(f"\n[Checkpoint {checkpoint_count}] Simulation completed normally")
                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)
                total_elapsed_time += results.get("duration", 0)
                
                # Collect final traffic state
                final_time = env.get_current_time()
                traffic_state_collector.collect(final_time)
                break
            
            # Safety check: if remaining_duration is very small, break
            if remaining_duration < step_seconds:
                print(f"\nRemaining duration ({remaining_duration:.0f}s) is less than step size, ending simulation.")
                break
        
        # Final summary
        print("\n" + "=" * 80)
        print("Simulation completed!")
        print(f"  Total checkpoints: {checkpoint_count}")
        print(f"  Total elapsed time: {total_elapsed_time:.0f}s")
        print(f"  Total steps: {accumulated_results['total_steps']}")
        print(f"  Total departed: {accumulated_results['total_departed']}")
        print(f"  Total arrived: {accumulated_results['total_arrived']}")
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
        
        # Get final statistics
        if env:
            # Get checkpoint-scoped travel time (current checkpoint interval only)
            final_avg_travel_time = env.get_average_travel_time()
            # Get global travel time (accumulated across all checkpoints)
            global_avg_travel_time = env.get_global_average_travel_time()
            global_highway_avg_travel_time = env.get_global_highway_average_travel_time()
            global_arrived_count = env.get_global_arrived_count()
            global_highway_arrived_count = env.get_global_highway_arrived_count()
            
            accumulated_results["final_avg_travel_time"] = final_avg_travel_time
            accumulated_results["global_avg_travel_time"] = global_avg_travel_time
            accumulated_results["global_highway_avg_travel_time"] = global_highway_avg_travel_time
            accumulated_results["global_arrived_count"] = global_arrived_count
            accumulated_results["global_highway_arrived_count"] = global_highway_arrived_count
            accumulated_results["final_time"] = env.get_current_time()
            
            # Print global travel time statistics
            print(f"\n  Global Travel Time Statistics (across all checkpoints):")
            print(f"    Global Average Travel Time: {global_avg_travel_time:.2f}s")
            print(f"    Global Arrived Vehicle Count: {global_arrived_count}")
            if global_highway_arrived_count > 0:
                print(f"    Global Highway Average Travel Time: {global_highway_avg_travel_time:.2f}s")
                print(f"    Global Highway Arrived Vehicle Count: {global_highway_arrived_count}")
            print(f"  Checkpoint-scoped Travel Time (current checkpoint only):")
            print(f"    Final Average Travel Time: {final_avg_travel_time:.2f}s")
        
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
            if accumulated_results.get("average_module_metrics"):
                final_metrics = {
                    "final/checkpoint_count": checkpoint_count,
                    "final/total_elapsed_time": total_elapsed_time,
                    "final/total_steps": accumulated_results["total_steps"],
                    "final/total_departed": accumulated_results["total_departed"],
                    "final/total_arrived": accumulated_results["total_arrived"]
                }
                if accumulated_results.get("final_avg_travel_time"):
                    final_metrics["final/avg_travel_time"] = accumulated_results["final_avg_travel_time"]
                if accumulated_results.get("global_avg_travel_time"):
                    final_metrics["final/global_avg_travel_time"] = accumulated_results["global_avg_travel_time"]
                if accumulated_results.get("global_highway_avg_travel_time"):
                    final_metrics["final/global_highway_avg_travel_time"] = accumulated_results["global_highway_avg_travel_time"]
                if accumulated_results.get("global_arrived_count"):
                    final_metrics["final/global_arrived_count"] = accumulated_results["global_arrived_count"]
                if accumulated_results.get("global_highway_arrived_count"):
                    final_metrics["final/global_highway_arrived_count"] = accumulated_results["global_highway_arrived_count"]
                
                # Add average module metrics
                for module_name, metrics in accumulated_results["average_module_metrics"].items():
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            final_metrics[f"final/{module_name}/{metric_name}"] = float(metric_value)
                
                wandb_run.log(final_metrics)
            
            wandb_run.finish()
            print("Logged final metrics to wandb")

        return result
        
    except Exception as e:
        error_msg = f"Checkpoint-based simulation failed: {str(e)}"
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
        
        # Save checkpoint log even on error
        try:
            if 'checkpoint_logger' in locals():
                log_filepath = checkpoint_logger.save_log()
                print(f"\nCheckpoint log saved to: {log_filepath}")
        except Exception as log_error:
            print(f"Warning: Failed to save checkpoint log: {log_error}")
        
        result = {
            "status": "error",
            "message": str(e),
            "checkpoint_count": checkpoint_count,
            "total_elapsed_time": total_elapsed_time,
            "simulation_id": simulation_id,
            "results": accumulated_results
        }
        
        if 'traffic_state_collector' in locals():
            stats = traffic_state_collector.get_stats()
            result["traffic_state_file"] = stats["filepath"]
            result["traffic_state_snapshots"] = stats["snapshot_count"]
        
        return result
    finally:
        if env:
            env.close()


def main():
    """Main entry point for checkpoint-based simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with checkpoint-based LLM agent control for highway speed limits"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sumo_config_highway/Manhattan/Manhattan.sumocfg",
        help="Path to SUMO config file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600*24,
        help="Total simulation duration in seconds (default: 86400 = 24 hours)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=1800,
        help="Checkpoint interval in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=30,
        help="Simulation step size in seconds (default: 30)"
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
        default=15,
        help="Maximum dialogue turns for LLM agent (default: 15)"
    )
    parser.add_argument(
        "--control-modules",
        type=str,
        nargs='+',
        default=['highway_speed_limit'],
        choices=['highway_speed_limit', 'signal_timing', 'subway_scheduling', 'bus_scheduling'],
        help="Control modules to enable. Available: highway_speed_limit, signal_timing, subway_scheduling (TODO), bus_scheduling (TODO). "
             "Can specify multiple modules, e.g., --control-modules highway_speed_limit signal_timing. "
             "(default: highway_speed_limit)"
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
        help="Wandb project name (default: sumo_highway_speed_limit_control)"
    )
    parser.add_argument(
        "--traffic-state-interval",
        type=float,
        default=60,
        help="Interval for collecting traffic state data in seconds (default: 300)"
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
        "--temperature",
        type=float,
        default=0.3,
        help="LLM agent sampling temperature (default: 0.3)"
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
    
    # Run checkpoint-based simulation
    results = run_checkpoint_based_simulation(
        config_path=str(config_path),
        simulation_duration=args.duration,
        checkpoint_interval=args.checkpoint_interval,
        step_seconds=args.step_seconds,
        use_gui=args.gui,
        seed=args.seed,
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
