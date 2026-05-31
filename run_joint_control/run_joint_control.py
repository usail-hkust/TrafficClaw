"""
Joint agent-controlled simulation runner for LLM-driven joint optimization.

This script enables joint optimization for multiple control modules using
a simplified workflow aligned with single-module control scripts.

Features:
- Joint optimization for configured modules at each checkpoint
- Inline prompt construction with module dependency context
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
workspace_root = current_file.parent.parent
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
from utils.traffic_state_collector import init_traffic_states_file
from utils.checkpoint_logger import CheckpointLogger, extract_configs_only, effective_config
from control_modules.shared.decision_context import DecisionContextManager


DEFAULT_CONTROL_MODULES = ["signal_timing", "bus_scheduling"]
DEFAULT_ALWAYS_ENABLED = ["signal_timing", "ramp_metering"]

KNOWN_CONTROL_MODULES = [
    "signal_timing",
    "bus_scheduling",
    "subway_scheduling",
    "highway_speed_limit",
    "ramp_metering",
    "taxi_scheduling",
]

MODULE_DEPENDENCIES = {
    "signal_timing": {"affects": ["bus_scheduling", "taxi_scheduling"], "affected_by": []},
    "highway_speed_limit": {"affects": ["ramp_metering"], "affected_by": []},
    "ramp_metering": {"affects": [], "affected_by": ["highway_speed_limit"]},
    "bus_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
    "taxi_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
}

MODULE_SHORT_NAMES = {
    "signal_timing": "sig",
    "bus_scheduling": "bus",
    "subway_scheduling": "sub",
    "highway_speed_limit": "hwy",
    "ramp_metering": "ramp",
    "taxi_scheduling": "taxi",
}


def get_modules_short_name(control_modules: Optional[List[str]]) -> str:
    """Compact wandb group segment: sorted module abbreviations joined by underscores."""
    if not control_modules:
        return "baseline"
    short_names = [MODULE_SHORT_NAMES.get(m, m[:4]) for m in sorted(control_modules)]
    return "_".join(short_names)


def _merge_unique_modules(*module_lists: Optional[List[str]]) -> List[str]:
    """Merge module name lists, preserving order and removing duplicates."""
    merged = []
    seen = set()
    for modules in module_lists:
        if not modules:
            continue
        for name in modules:
            if name not in seen:
                merged.append(name)
                seen.add(name)
    return merged




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

    # Restore taxi state with live SUMO sync if available.
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


def format_dependencies(control_modules: List[str]) -> str:
    """Format module dependencies for prompt display."""
    if not control_modules:
        return "- None"

    lines = []
    enabled_set = set(control_modules)
    for module in control_modules:
        deps = MODULE_DEPENDENCIES.get(module, {})
        affects = [m for m in deps.get("affects", []) if m in enabled_set]
        affected_by = [m for m in deps.get("affected_by", []) if m in enabled_set]

        if not affects and not affected_by:
            lines.append(f"- {module}: independent")
            continue

        parts = []
        if affects:
            parts.append(f"affects {', '.join(affects)}")
        if affected_by:
            parts.append(f"affected by {', '.join(affected_by)}")

        line = f"- {module}: " + "; ".join(parts)
        if affects and not affected_by:
            line += " (optimize first)"
        lines.append(line)

    return "\n".join(lines)


def format_metrics(module_metrics: Optional[Dict[str, Dict[str, Any]]],
                   control_modules: List[str]) -> str:
    """Format module performance metrics for prompt display."""
    if not module_metrics or not control_modules:
        return "- No metrics available"

    lines = []
    for module in control_modules:
        metrics = module_metrics.get(module)
        if not metrics:
            lines.append(f"- {module}: no metrics available")
            continue
        formatted = []
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                formatted.append(f"{metric_name}={metric_value:.2f}")
            else:
                formatted.append(f"{metric_name}={metric_value}")
        lines.append(f"- {module}: {', '.join(formatted)}")

    return "\n".join(lines) if lines else "- No metrics available"


def format_simulation_time(seconds: float) -> Dict[str, Any]:
    """Format simulation time in seconds to human-readable time format."""
    total_hours = seconds / 3600.0
    day_hour = total_hours % 24
    hours = int(day_hour)
    minutes = int((day_hour - hours) * 60)
    day_number = int(total_hours / 24) + 1

    time_string = f"{hours:02d}:{minutes:02d}"

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
    else:
        time_period = "Night"
        period_description = "Night period (00:00-06:00). Low traffic levels. Shorter signal cycles may be optimal."

    return {
        "hours": hours,
        "minutes": minutes,
        "time_string": time_string,
        "time_period": time_period,
        "period_description": period_description,
        "day_of_simulation": day_number,
        "total_hours": total_hours
    }


def log_metrics_to_wandb(
    wandb_run,
    checkpoint_number: int,
    elapsed_time: float,
    module_metrics: Dict[str, Dict[str, Any]],
    control_modules: Optional[List[str]] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None
):
    """
    Log module performance metrics to wandb (aligned with run_single_control scripts).

    Joint control additionally merges optional baseline_metrics from policy simulation.
    """
    if not WANDB_AVAILABLE or wandb_run is None:
        return

    metrics_dict = {
        "checkpoint": checkpoint_number,
        "elapsed_time": elapsed_time,
        "elapsed_time_hours": elapsed_time / 3600.0
    }

    if additional_metrics:
        metrics_dict.update(additional_metrics)
    if baseline_metrics:
        metrics_dict.update(baseline_metrics)

    if control_modules and module_metrics:
        for module_name in control_modules:
            if module_name not in module_metrics:
                continue

            metrics = module_metrics[module_name]
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    wandb_key = f"{module_name}/{metric_name}"
                    metrics_dict[wandb_key] = float(metric_value)

    wandb_run.log(metrics_dict)


def build_baseline_metrics(
    baseline_result: Dict[str, Any],
    control_modules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Build baseline metrics dict for wandb logging."""
    metrics: Dict[str, Any] = {}

    stats_keys = [
        "total_departed",
        "total_arrived",
        "avg_travel_time",
        "policy_avg_travel_time",
        "policy_arrived_count",
        "policy_highway_avg_travel_time",
        "policy_highway_arrived_count",
        "checkpoint_avg_travel_time",
        "checkpoint_avg_waiting_time",
        "checkpoint_waiting_count",
        "duration"
    ]
    for key in stats_keys:
        if key in baseline_result and baseline_result.get(key) is not None:
            value = baseline_result.get(key)
            if isinstance(value, (int, float)):
                metrics[f"baseline/{key}"] = float(value)

    module_metrics = baseline_result.get("module_metrics", {})
    if module_metrics and control_modules:
        for module_name in control_modules:
            if module_name not in module_metrics:
                continue
            for metric_name, metric_value in module_metrics[module_name].items():
                if isinstance(metric_value, (int, float)):
                    metrics[f"baseline/{module_name}/{metric_name}"] = float(metric_value)

    return metrics


def calculate_average_module_metrics(
    checkpoints: List[Dict[str, Any]],
    control_modules: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate average module metrics across all checkpoints."""
    module_metric_sums = {}
    module_metric_counts = {}

    for checkpoint in checkpoints:
        module_metrics = checkpoint.get("module_metrics", {})

        for module_name in control_modules:
            if module_name not in module_metrics:
                continue

            if module_name not in module_metric_sums:
                module_metric_sums[module_name] = {}
                module_metric_counts[module_name] = {}

            for metric_name, metric_value in module_metrics[module_name].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in module_metric_sums[module_name]:
                        module_metric_sums[module_name][metric_name] = 0.0
                        module_metric_counts[module_name][metric_name] = 0

                    module_metric_sums[module_name][metric_name] += float(metric_value)
                    module_metric_counts[module_name][metric_name] += 1

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


def extract_coordination_insights(
    decision_context_manager: DecisionContextManager,
    accumulated_results: Dict
) -> List[str]:
    """
    Extract cross-module coordination insights from decision context.

    Analyzes decision history to identify effective coordination patterns
    and generates reusable insights for future experiments.

    Args:
        decision_context_manager: The decision context manager with recorded decisions
        accumulated_results: Accumulated simulation results

    Returns:
        List of insight strings for memory persistence
    """
    insights = []
    decisions = decision_context_manager.decisions

    # Analyze Signal → Bus coordination pattern
    if "signal_timing" in decisions and "bus_scheduling" in decisions:
        signal_focuses = [d.get('optimization_focus', '') for d in decisions.get("signal_timing", [])]
        if any('northbound' in f.lower() for f in signal_focuses if f):
            insights.append(
                "Cross-module insight: When signal timing prioritizes northbound flow, "
                "bus routes on northbound corridors benefit from reduced delays."
            )
        if any('southbound' in f.lower() for f in signal_focuses if f):
            insights.append(
                "Cross-module insight: When signal timing prioritizes southbound flow, "
                "bus routes on southbound corridors benefit from reduced delays."
            )

    # Analyze Highway → Ramp coordination pattern
    if "highway_speed_limit" in decisions and "ramp_metering" in decisions:
        vsl_decisions = decisions.get("highway_speed_limit", [])
        if vsl_decisions:
            last_focus = vsl_decisions[-1].get('optimization_focus', '')
            if 'congestion' in last_focus.lower():
                insights.append(
                    "Cross-module insight: During highway congestion, coordinating "
                    "restrictive ramp metering with lower VSL improves throughput."
                )

    # Analyze Transit → Taxi coordination pattern
    transit_modules = ["bus_scheduling", "subway_scheduling"]
    if "taxi_scheduling" in decisions and any(m in decisions for m in transit_modules):
        insights.append(
            "Cross-module insight: Position taxis near transit service gaps "
            "and high-demand stations for improved coverage."
        )

    # Analyze Signal → Subway coordination pattern
    if "signal_timing" in decisions and "subway_scheduling" in decisions:
        signal_decisions = decisions.get("signal_timing", [])
        if signal_decisions:
            affected_count = sum(
                len(d.get('affected_entities', [])) for d in signal_decisions
            )
            if affected_count > 5:
                insights.append(
                    "Cross-module insight: When multiple intersections are optimized, "
                    "coordinate subway feeder bus schedules for improved station access."
                )

    # Analyze Bus → Subway coordination pattern
    if "bus_scheduling" in decisions and "subway_scheduling" in decisions:
        insights.append(
            "Cross-module insight: Coordinating bus and subway schedules at "
            "transfer stations reduces passenger waiting time."
        )

    return insights


def run_joint_control_simulation(
    config_path: str,
    simulation_duration: float,
    checkpoint_interval: float,
    step_seconds: int = 30,
    min_step_seconds: float = 1.0,
    use_gui: bool = False,
    seed: Optional[int] = None,
    llm_model: str = "openai/gpt-4o-mini",
    max_agent_turns: int = 10,
    control_modules: Optional[List[str]] = None,
    always_enabled: Optional[List[str]] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    traffic_state_interval: float = 300,
    traffic_lane_sample_rate: float = 1.0,
    traffic_lane_sample_size: int = 0,
    waiting_passenger_interval: Optional[float] = None,
    vehicle_subscription_mode: str = "departed",
    resume_checkpoint: Optional[str] = None,
    max_interval_retries: int = 2,
    max_reflection_turns: int = 5,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run simulation with checkpoint-based control for joint LLM optimization.

    Args:
        config_path: Path to SUMO config file
        simulation_duration: Total simulation duration in seconds
        checkpoint_interval: Interval between checkpoints in seconds
        step_seconds: Simulation step size in seconds
        min_step_seconds: Minimum step size for control loops
        use_gui: Whether to use SUMO GUI
        seed: Random seed for simulation
        llm_model: LLM model name for the agent
        max_agent_turns: Maximum dialogue turns for LLM agent
        control_modules: List of control modules to optimize
        always_enabled: Modules always enabled but not necessarily optimized
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        traffic_state_interval: Interval for traffic state collection
        traffic_lane_sample_rate: Sample rate for lane traffic collection
        traffic_lane_sample_size: Fixed sample size for lanes
        waiting_passenger_interval: Interval for passenger waiting updates
        vehicle_subscription_mode: Vehicle subscription mode
        resume_checkpoint: Optional path to checkpoint XML for resuming simulation
        max_interval_retries: Max retries per checkpoint interval on SUMO/TraCI failure
        max_reflection_turns: Maximum number of reflection turns (default: 5)
        temperature: LLM sampling temperature (default: 0.3)
        base_url: Optional API base URL for the LLM provider

    Returns:
        Dictionary containing simulation results
    """
    checkpoint_dir = workspace_root / "records" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = None
    checkpoint_count = 0
    total_elapsed_time = 0.0
    accumulated_results = {
        "total_steps": 0,
        "total_departed": 0,
        "total_arrived": 0,
        "checkpoints": []
    }

    global_travel_times = []
    global_waiting_times = []

    control_modules = control_modules or DEFAULT_CONTROL_MODULES
    always_enabled = always_enabled if always_enabled is not None else DEFAULT_ALWAYS_ENABLED

    resume_metadata: Dict[str, Any] = {}
    resume_checkpoint_path: Optional[str] = None
    if resume_checkpoint:
        resume_checkpoint_path = str(Path(resume_checkpoint))
        if not Path(resume_checkpoint_path).exists():
            return {
                "status": "error",
                "message": f"Resume checkpoint not found: {resume_checkpoint_path}"
            }
        resume_metadata = _load_checkpoint_metadata(resume_checkpoint_path)
        resume_extra = resume_metadata.get("extra", {}) or {}

        resume_control_configs = resume_extra.get("control_configs") or {}
        resume_control_modules = resume_extra.get("control_modules") or []

        if resume_control_configs:
            resolved_modules = [m for m in resume_control_configs.keys() if m in KNOWN_CONTROL_MODULES]
            if resolved_modules:
                control_modules = resolved_modules

        if resume_control_modules:
            resolved_enabled = [m for m in resume_control_modules if m in KNOWN_CONTROL_MODULES]
            if resolved_enabled:
                always_enabled = [m for m in resolved_enabled if m not in control_modules]

    TRAFFIC_STATE_INTERVAL = traffic_state_interval

    config_path_obj = Path(config_path)
    config_dir_name = config_path_obj.parent.name
    simulation_id, file_prefix = generate_simulation_identifiers(
        config_name=config_dir_name,
        llm_name=llm_model,
        control_modules=control_modules
    )
    traffic_states_filepath = None

    checkpoint_logger = CheckpointLogger(
        simulation_id=simulation_id,
        llm_model_name=llm_model
    )

    # Initialize decision context manager for cross-module coordination
    decision_context_manager = DecisionContextManager()

    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        llm_model_safe = llm_model.replace("/", "_").replace("\\", "_").replace(":", "_")
        modules_short = get_modules_short_name(control_modules)
        wandb_group = f"{llm_model_safe}-{modules_short}-{config_dir_name}"
        experiment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            "control_modules_short": modules_short,
            "always_enabled": always_enabled if always_enabled else [],
            "use_gui": use_gui,
            "max_reflection_turns": max_reflection_turns,
        }

        wandb_run = wandb.init(
            project=wandb_project or "sumo_joint_control",
            name=experiment_start_time,
            group=wandb_group,
            config=wandb_config,
            reinit=True
        )
        print(f"Initialized wandb run: {wandb_run.name} (group: {wandb_group})")

    try:
        # Set random seeds for reproducibility
        # This ensures all Python random operations (random.choice, random.shuffle, etc.)
        # and NumPy random operations are deterministic
        if seed is not None:
            import random
            import numpy as np

            # CRITICAL: Set PYTHONHASHSEED to make hash-based collections (set, dict) deterministic
            # This must be set BEFORE Python starts, but we set it here for subprocess/child processes
            os.environ['PYTHONHASHSEED'] = '0'

            random.seed(seed)
            np.random.seed(seed)
            print(f"Set Python random seed: {seed}")
            print(f"Set NumPy random seed: {seed}")
            print(f"Set PYTHONHASHSEED: 0 (for deterministic set/dict iteration)")

        print("=" * 80)
        print("Starting joint checkpoint-based simulation")
        print(f"  Total duration: {simulation_duration}s")
        print(f"  Checkpoint interval: {checkpoint_interval}s")
        print(f"  Expected checkpoints: {int(simulation_duration / checkpoint_interval)}")
        print(f"  Step seconds: {step_seconds} (min: {min_step_seconds})")
        print(f"  Vehicle subscription mode: {vehicle_subscription_mode}")
        if waiting_passenger_interval is not None:
            print(f"  Waiting passenger interval: {waiting_passenger_interval}s")
        print(f"  Control modules: {control_modules}")
        print(f"  Always enabled: {always_enabled}")
        print("=" * 80)

        enabled_modules = _merge_unique_modules(control_modules, always_enabled)

        env, env_config = create_sumo_env(
            config_path=config_path,
            use_gui=use_gui,
            seed=seed,
            control_modules=enabled_modules,
            run_counts=int(simulation_duration),
            config_dir_name=config_dir_name,
            vehicle_subscription_mode=vehicle_subscription_mode,
            waiting_passenger_interval=waiting_passenger_interval,
            load_state_path=resume_checkpoint_path,
            use_simulated_taxi_system="taxi_scheduling" in enabled_modules,
        )

        if traffic_states_filepath is None:
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=simulation_id,
                config_name=config_dir_name,
                llm_name=llm_model,
                control_modules=control_modules
            )
            print(f"Initialized traffic states file: {traffic_states_filepath.name}")

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

        lane_ids = None
        if lane_dict:
            sample_rate = max(0.0, min(1.0, float(traffic_lane_sample_rate)))
            sample_size = max(0, int(traffic_lane_sample_size))
            if sample_size > 0 or sample_rate < 1.0:
                import random

                lane_ids = list(lane_dict.keys())
                rng = random.Random(seed if seed is not None else 0)
                rng.shuffle(lane_ids)
                if sample_size > 0:
                    lane_ids = lane_ids[:min(sample_size, len(lane_ids))]
                else:
                    if sample_rate <= 0.0:
                        lane_ids = []
                    else:
                        target = max(1, int(len(lane_ids) * sample_rate))
                        lane_ids = lane_ids[:target]
                print(f"  Traffic state lane sampling: {len(lane_ids)}/{len(lane_dict)} lanes")

        traffic_state_collector = TrafficStateCollector(
            env=env,
            traffic_states_filepath=traffic_states_filepath,
            interval=TRAFFIC_STATE_INTERVAL,
            lane_dict=lane_dict,
            lane_inter_graph=lane_inter_graph,
            simulation_id=simulation_id,
            lane_ids=lane_ids
        )
        print(f"Created traffic state collector (interval: {TRAFFIC_STATE_INTERVAL}s)")

        agent = LLMAgent(
            model_name=llm_model,
            temperature=temperature,
            max_turns=max_agent_turns,
            available_control_modules=control_modules,
            config_name=config_dir_name,
            max_memory_items=10,
            decision_context_manager=decision_context_manager,
            is_joint_control=True,
            max_reflection_turns=max_reflection_turns,
            base_url=base_url,
        )
        print(f"Initialized LLM agent (model: {llm_model}, max_turns: {max_agent_turns})")

        control_configs = None
        control_states = None

        if control_modules and env.enabled_controls:
            print(f"\nInitializing control configs for optimization modules: {control_modules}")
            control_configs = {}

            for module_name in control_modules:
                if module_name not in env.enabled_controls:
                    print(f"  - Warning: {module_name} not found in env.enabled_controls, skipping...")
                    continue

                module_info = env.enabled_controls[module_name]
                module = module_info.get("module")
                if module is None:
                    print(f"  - Warning: {module_name} module not available, skipping...")
                    continue

                existing_config = module_info.get("config", {})
                if existing_config:
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
                print("  No valid control configs found, running baseline simulation")
                control_configs = None
        else:
            print("\nNo control modules enabled, running baseline simulation")

        if resume_checkpoint_path:
            control_configs, control_states = _restore_control_state_from_metadata(
                env,
                control_configs,
                control_states,
                metadata=resume_metadata
            )
        # Baseline policy simulation (per checkpoint) uses the same configs as the interval just
        # simulated — replay from t-1 with copy.deepcopy(control_configs), aligned with
        # run_traffic_signal_control.py.

        remaining_duration = simulation_duration
        previous_checkpoint_path = None
        is_first_simulation = True

        if resume_checkpoint_path:
            resume_time = env.get_current_time()
            checkpoint_time = resume_metadata.get("checkpoint_time") or resume_metadata.get("sim_time")
            if checkpoint_time is not None:
                try:
                    resume_time = float(checkpoint_time)
                except Exception:
                    pass
            if resume_time:
                remaining_duration = max(0.0, simulation_duration - float(resume_time))
                checkpoint_count = int(float(resume_time) // checkpoint_interval)
                previous_checkpoint_path = resume_checkpoint_path
                total_elapsed_time = float(resume_time)
                is_first_simulation = False
                decision_context_manager.set_checkpoint(checkpoint_count)
                traffic_state_collector.last_collection_time = resume_time - TRAFFIC_STATE_INTERVAL
                print(
                    f"Resuming from checkpoint at t={resume_time:.0f}s "
                    f"(remaining {remaining_duration:.0f}s, next checkpoint {checkpoint_count + 1})"
                )

        max_interval_retries = max(0, int(max_interval_retries))

        while remaining_duration > 0:
            checkpoint_count += 1
            current_checkpoint_duration = min(checkpoint_interval, remaining_duration)

            print(f"\n[Checkpoint {checkpoint_count}] Running simulation for {current_checkpoint_duration:.0f}s...")
            print(f"  Remaining duration: {remaining_duration:.0f}s")
            if control_configs:
                print(f"  Control modules: {list(control_configs.keys())}")
            else:
                print("  Control: None (baseline)")
            remaining_duration -= current_checkpoint_duration

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
                    min_step_seconds=min_step_seconds,
                    traffic_state_collector=traffic_state_collector,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_prefix=f"checkpoint_{checkpoint_count}",
                    control_configs=control_configs,
                    control_states=control_states,
                    is_first_simulation=is_first_simulation,
                    config_name=config_dir_name,
                    llm_name=llm_model,
                    simulation_id=simulation_id,
                    use_accelerated_stepping=False  # Match run_single_control scripts (traditional stepping)
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

            if results.get("control_states"):
                control_states = results["control_states"]
                if "_previous_checkpoint_path" in control_states:
                    del control_states["_previous_checkpoint_path"]

            # Snapshot control_states at the START of this checkpoint window (used for policy evaluation from t-1 state)
            checkpoint_start_control_states = copy.deepcopy(control_states)
            if "_previous_checkpoint_path" in checkpoint_start_control_states:
                del checkpoint_start_control_states["_previous_checkpoint_path"]

            if results.get("checkpoint_reached", False):
                checkpoint_path = results.get("checkpoint_path")
                checkpoint_path_t_minus_1 = results.get("checkpoint_path_t_minus_1")
                elapsed_time = results.get("elapsed_time", 0)
                current_time = results.get("current_time", 0)

                if checkpoint_path:
                    previous_checkpoint_path = checkpoint_path

                print(f"\n[Checkpoint {checkpoint_count}] Checkpoint reached!")
                print(f"  Elapsed time: {elapsed_time:.0f}s")
                print(f"  Remaining duration: {remaining_duration:.0f}s")
                print(f"  Checkpoint saved: {checkpoint_path}")

                if traffic_state_collector is not None:
                    current_time = results.get("current_time", elapsed_time)
                    traffic_state_collector.save_checkpoint_snapshots(checkpoint_time=current_time)

                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)

                module_metrics = results.get("module_metrics", {})
                # Keep metrics for every enabled module that appears in results (matches wandb in
                # run_taxi_scheduling.py: taxi must not be dropped when only in always_enabled).
                if module_metrics and enabled_modules:
                    module_metrics = {
                        name: metrics
                        for name, metrics in module_metrics.items()
                        if name in enabled_modules
                    }

                checkpoint_avg_travel_time = results.get("checkpoint_avg_travel_time")
                checkpoint_arrived_count = results.get("checkpoint_arrived_count", 0)
                if checkpoint_avg_travel_time is not None and checkpoint_arrived_count > 0:
                    global_travel_times.append({
                        "checkpoint": checkpoint_count,
                        "avg_travel_time": checkpoint_avg_travel_time,
                        "arrived_count": checkpoint_arrived_count
                    })

                checkpoint_avg_waiting_time = results.get("checkpoint_avg_waiting_time")
                checkpoint_waiting_count = results.get("checkpoint_waiting_count", 0)
                if checkpoint_avg_waiting_time is not None:
                    global_waiting_times.append({
                        "checkpoint": checkpoint_count,
                        "avg_waiting_time": checkpoint_avg_waiting_time,
                        "waiting_count": checkpoint_waiting_count
                    })

                checkpoint_info = {
                    "checkpoint_number": checkpoint_count,
                    "checkpoint_path": checkpoint_path,
                    "elapsed_time": elapsed_time,
                    "remaining_duration": remaining_duration,
                    "step_count": results.get("step_count", 0),
                    "avg_travel_time": results.get("avg_travel_time", 0),
                    "checkpoint_avg_travel_time": checkpoint_avg_travel_time,
                    "checkpoint_arrived_count": checkpoint_arrived_count,
                    "checkpoint_avg_waiting_time": checkpoint_avg_waiting_time,
                    "checkpoint_waiting_count": checkpoint_waiting_count
                }

                if module_metrics:
                    checkpoint_info["module_metrics"] = module_metrics

                accumulated_results["checkpoints"].append(checkpoint_info)
                total_elapsed_time += elapsed_time

                additional_metrics = {
                    "step_count": results.get("step_count", 0),
                    "avg_travel_time": results.get("avg_travel_time", 0),
                    "total_departed": results.get("total_departed", 0),
                    "total_arrived": results.get("total_arrived", 0),
                    "remaining_duration": remaining_duration,
                }
                if checkpoint_avg_travel_time is not None:
                    additional_metrics["checkpoint_avg_travel_time"] = checkpoint_avg_travel_time
                    additional_metrics["checkpoint_arrived_count"] = checkpoint_arrived_count
                if checkpoint_avg_waiting_time is not None:
                    additional_metrics["checkpoint_avg_waiting_time"] = checkpoint_avg_waiting_time
                    additional_metrics["checkpoint_waiting_count"] = checkpoint_waiting_count
                if results.get("avg_waiting_time") is not None:
                    additional_metrics["avg_waiting_time"] = results.get("avg_waiting_time")

                wandb_module_names = list(module_metrics.keys()) if module_metrics else []

                baseline_metrics = None
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
                            initial_control_states=checkpoint_start_control_states if "taxi_scheduling" in enabled_modules else None,
                            taxi_dispatch_algorithm=env_config.get("TAXI_DISPATCH_ALGORITHM") if "taxi_scheduling" in enabled_modules else None,
                            taxi_idle_algorithm=env_config.get("TAXI_IDLE_ALGORITHM") if "taxi_scheduling" in enabled_modules else None,
                        )

                        if baseline_raw.get("success"):
                            baseline_simulation_result = {
                                "success": True,
                                "stats": {
                                    "total_departed": baseline_raw.get("total_departed", 0),
                                    "total_arrived": baseline_raw.get("total_arrived", 0),
                                    "avg_travel_time": baseline_raw.get("avg_travel_time", 0),
                                    "policy_avg_travel_time": baseline_raw.get("policy_avg_travel_time"),
                                    "policy_arrived_count": baseline_raw.get("policy_arrived_count", 0),
                                    "policy_highway_avg_travel_time": baseline_raw.get("policy_highway_avg_travel_time"),
                                    "policy_highway_arrived_count": baseline_raw.get("policy_highway_arrived_count", 0),
                                    "duration": baseline_raw.get("duration", 0)
                                },
                                "module_metrics": baseline_raw.get("module_metrics", {}),
                                "error": None,
                                "control_configs": copy.deepcopy(control_configs),
                            }
                            baseline_metrics = build_baseline_metrics(baseline_raw, enabled_modules)
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
                    if baseline_module_metrics and enabled_modules:
                        baseline_module_metrics = {
                            name: metrics
                            for name, metrics in baseline_module_metrics.items()
                            if name in enabled_modules
                        }

                    if baseline_module_metrics:
                        print("\n  Baseline Simulation Module Performance Metrics:")
                        for module_name, metrics in baseline_module_metrics.items():
                            print(f"    {module_name.upper().replace('_', ' ')}:")
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, float):
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                                else:
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")

                if baseline_simulation_result:
                    # Extract configs from simulation result before passing to logger
                    baseline_result_for_log = baseline_simulation_result.copy() if isinstance(baseline_simulation_result, dict) else baseline_simulation_result
                    if isinstance(baseline_result_for_log, dict) and "control_configs" in baseline_result_for_log:
                        baseline_result_for_log["control_configs"] = extract_configs_only(baseline_result_for_log["control_configs"]) or {}
                    checkpoint_logger.add_policy_simulation_result(
                        checkpoint_number=checkpoint_count,
                        simulation_result=baseline_result_for_log
                    )

                # Wandb: align with run_taxi_scheduling.py — log before taxi control_state sync;
                # cumulative taxi counters (pickups, income, …) come from calculate_final_results.
                # Sync control_state metrics to module_metrics BEFORE wandb logging (same order as run_taxi_scheduling.py)
                if control_states and "taxi_scheduling" in control_states and "taxi_scheduling" in module_metrics:
                    taxi_state = control_states["taxi_scheduling"]
                    taxi_metrics = module_metrics["taxi_scheduling"]
                    taxi_metrics["total_dispatches"] = taxi_state.get("total_dispatches", 0)
                    if "completed_dispatches" in taxi_state:
                        taxi_metrics["successful_dispatches"] = len(taxi_state["completed_dispatches"])
                    accumulated_results["passenger_pickups"] = taxi_metrics.get("passenger_pickups", 0)
                    accumulated_results["passenger_dropoffs"] = taxi_metrics.get("passenger_dropoffs", 0)

                if wandb_run:
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
                        control_modules=wandb_module_names,
                        additional_metrics=additional_metrics,
                        baseline_metrics=baseline_metrics
                    )

                if module_metrics:
                    print("\n  Control Module Performance Metrics:")
                    for module_name, metrics in module_metrics.items():
                        print(f"    {module_name.upper().replace('_', ' ')}:")
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, float):
                                print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                            else:
                                print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")

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
                    control_configs=control_configs_for_log
                )

                print(f"\n{'=' * 80}")
                print(f"[Checkpoint {checkpoint_count}] LLM Agent Optimization")
                print(f"{'=' * 80}")

                agent_context = {
                    "lane_graph": graphs.get("lane_graph"),
                    "lane_inter_graph": graphs.get("lane_inter_graph"),
                    "intersection_graph": graphs.get("intersection_graph"),
                    "lane_dict": graphs.get("lane_dict"),
                    "highway_graph": env.highway_subgraph if hasattr(env, "highway_subgraph") else None,
                    "highway_segment_dict": env.highway_info_dict if hasattr(env, "highway_info_dict") else {},
                    "highway_segment_graph": graphs.get("highway_segment_graph") or (
                        env.get_highway_segment_graph() if hasattr(env, "get_highway_segment_graph") else None
                    ),
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
                    "cross_module_dependencies": MODULE_DEPENDENCIES,
                    "module_metrics": module_metrics,
                    "run_duration": simulation_duration,
                    "seed": seed
                }

                checkpoint_avg_tt = (
                    checkpoint_avg_travel_time
                    if checkpoint_avg_travel_time is not None
                    else results.get("avg_travel_time", 0)
                )
                # Use extract_configs_only so initial_best_result is JSON-serializable (no module instances)
                checkpoint_simulation_result = {
                    "success": True,
                    "stats": {
                        "total_departed": results.get("total_departed", 0),
                        "total_arrived": checkpoint_arrived_count,
                        "avg_travel_time": checkpoint_avg_tt,
                        "duration": elapsed_time
                    },
                    "module_metrics": module_metrics.copy() if module_metrics else {},
                    "control_configs": extract_configs_only(control_configs) if control_configs else {}
                }

                current_time = env.get_current_time()
                time_info = format_simulation_time(current_time)
                next_checkpoint_time = current_time + checkpoint_interval
                next_time_info = format_simulation_time(next_checkpoint_time)
                module_list = ", ".join(control_modules) if control_modules else "None"
                decision_context_manager.set_checkpoint(checkpoint_count)

                initial_prompt = f"""You are optimizing multiple traffic control modules jointly.

Current Status:
- Current simulation time: {current_time:.0f} seconds ({time_info['time_string']})
- Time period: {time_info['time_period']}
- Checkpoint: {checkpoint_count}

Enabled Modules: {module_list}

Module Dependencies:
{format_dependencies(control_modules)}

Performance Metrics:
{format_metrics(module_metrics, control_modules)}

Optimization Window:
- Start: {time_info['time_string']} (Day {time_info['day_of_simulation']}, {time_info['time_period']})
- End: {next_time_info['time_string']} (Day {next_time_info['day_of_simulation']}, {next_time_info['time_period']})
- Duration: {checkpoint_interval:.0f} seconds ({checkpoint_interval/3600:.2f} hours)
- Your optimized controls apply for this window.

Your Tasks:
1. Use GET_CONTROL_API to query module-specific APIs
2. Analyze traffic data using DATA_ANALYSIS
3. Create optimized policies using POLICY_PLANNING
4. Complete with FINISH action

Turn Efficiency:
- Use GET_CONTROL_API only when needed; you can query multiple modules in one action (Module: a, b).
- POLICY_PLANNING must include "Control Modules: ..." and define a `config` dict (no return).

Begin your analysis."""
                initial_prompt = append_user_query(initial_prompt, user_query)

                agent.available_control_modules = control_modules
                if hasattr(agent, 'manager') and agent.manager is not None:
                    agent.manager.available_control_modules = control_modules
                    agent.manager.module_metrics = module_metrics

                # Build cleaned initial_best_result for agent (JSON-serializable, no module instances)
                # Use baseline simulation result for all modules as initial best
                initial_best_result_for_agent = None
                if baseline_simulation_result and baseline_simulation_result.get("success"):
                    initial_best_result_for_agent = baseline_simulation_result.copy()
                    if isinstance(initial_best_result_for_agent, dict) and "control_configs" in initial_best_result_for_agent:
                        initial_best_result_for_agent["control_configs"] = extract_configs_only(initial_best_result_for_agent["control_configs"]) or {}
                else:
                    # Fallback to checkpoint simulation result if baseline failed
                    initial_best_result_for_agent = checkpoint_simulation_result
                    print(f"\n[Checkpoint {checkpoint_count}] Baseline simulation failed or unavailable, using control simulation result as initial best")

                optimization_result = agent.run_optimization(
                    initial_prompt=initial_prompt,
                    context=agent_context,
                    env=env,
                    verbose=True,
                    # Use baseline simulation result for all modules as initial best
                    initial_best_result=initial_best_result_for_agent,
                    initial_control_configs=copy.deepcopy(control_configs) if control_configs else None,
                    checkpoint_logger=checkpoint_logger,
                    checkpoint_number=checkpoint_count,
                )

                policy_simulation_results = []
                if optimization_result.get("history"):
                    for action_type, action_result in optimization_result.get("history", []):
                        if action_type == "SIMULATION" and isinstance(action_result, dict):
                            policy_simulation_results.append(action_result)

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
                for sim_result in policy_simulation_results:
                    # Extract configs from simulation result before passing to logger
                    sim_result_for_log = sim_result.copy() if isinstance(sim_result, dict) else sim_result
                    if isinstance(sim_result_for_log, dict) and "control_configs" in sim_result_for_log:
                        sim_result_for_log["control_configs"] = extract_configs_only(sim_result_for_log["control_configs"]) or {}
                    checkpoint_logger.add_policy_simulation_result(
                        checkpoint_number=checkpoint_count,
                        simulation_result=sim_result_for_log
                    )

                log_file = checkpoint_logger.save_log()
                conversation_file = checkpoint_logger.save_all_conversations()
                if conversation_file:
                    print(f"\n  All checkpoint conversations saved to: {conversation_file}")

                if optimization_result.get("success"):
                    final_control_configs = optimization_result.get("final_control_configs", {})

                    if not final_control_configs and optimization_result.get("final_signal_config"):
                        final_control_configs = {"signal_timing": optimization_result["final_signal_config"]}

                    if final_control_configs:
                        print("\nLLM Agent completed optimization:")
                        print(f"  - Turns used: {optimization_result['turn_count']}/{max_agent_turns}")
                        print(f"  - Control modules updated: {list(final_control_configs.keys())}")

                        # Merge with existing configs (preserve modules not updated this round)
                        if control_configs:
                            for module_name, config in final_control_configs.items():
                                control_configs[module_name] = config
                        else:
                            control_configs = final_control_configs.copy()

                        # Extract only configs (remove module instances) before passing to logger
                        control_configs_for_log = extract_configs_only(control_configs) if control_configs else {}
                        checkpoint_logger.update_checkpoint_control_configs(
                            checkpoint_number=checkpoint_count,
                            control_configs=control_configs_for_log
                        )
                        # Align with run_single_control (e.g. run_traffic_signal_control.py): reset
                        # module control state when configs change so per-checkpoint metrics are
                        # interval-scoped, not carried across LLM rounds. Taxi scheduling is the
                        # exception: preserve its control_state so income/pickups stay cumulative.
                        if "taxi_scheduling" in enabled_modules:
                            taxi_state = None
                            if control_states and isinstance(
                                control_states.get("taxi_scheduling"), dict
                            ):
                                taxi_state = copy.deepcopy(control_states["taxi_scheduling"])
                            control_states = (
                                {"taxi_scheduling": taxi_state}
                                if taxi_state is not None
                                else None
                            )
                        else:
                            control_states = None
                        print(f"  - Control modules will be applied in next checkpoint: {list(control_configs.keys())}")

                    else:
                        print("\nLLM Agent optimization did not produce new configuration.")
                        print("  - Continuing with current configuration...")
                else:
                    print("\nLLM Agent optimization did not succeed.")
                    print(f"  - Reason: {optimization_result.get('error', 'Unknown')}")
                    print("  - Continuing with current configuration...")

                try:
                    coordination_insights = extract_coordination_insights(
                        decision_context_manager,
                        accumulated_results
                    )
                    if coordination_insights:
                        existing_memory = set(agent.get_memory())
                        new_items = [item for item in coordination_insights if item not in existing_memory]
                        if new_items:
                            agent.add_memory(new_items)
                            print(f"  - Added {len(new_items)} coordination insight(s) to memory")
                except Exception as mem_error:
                    print(f"Warning: Failed to update memory: {mem_error}")

                print(f"{'=' * 80}\n")

                current_time = env.get_current_time()
                if "taxi_scheduling" in (control_modules or []):
                    traffic_state_collector.collect_if_needed(current_time)
                else:
                    traffic_state_collector.collect(current_time)

                if remaining_duration > 0:
                    print(f"\n[Checkpoint {checkpoint_count}] Resetting metrics for next checkpoint interval (no SUMO restart)...")
                    try:
                        env.reset_metrics()
                        print(f"  Metrics reset successfully")
                        print(f"  Continuing simulation from time: {env.get_current_time():.0f}s")

                        checkpoint_start_time = env.get_current_time()
                        traffic_state_collector.last_collection_time = checkpoint_start_time - TRAFFIC_STATE_INTERVAL

                        if is_first_simulation:
                            is_first_simulation = False
                    except Exception as e:
                        print(f"  Error resetting metrics: {e}")
                        traceback.print_exc()
                        break
                elif is_first_simulation:
                    print(f"\n[Checkpoint {checkpoint_count}] First simulation completed.")
                    is_first_simulation = False
            else:
                print(f"\n[Checkpoint {checkpoint_count}] Simulation completed normally")
                accumulated_results["total_steps"] += results.get("step_count", 0)
                accumulated_results["total_departed"] = results.get("total_departed", 0)
                accumulated_results["total_arrived"] = results.get("total_arrived", 0)
                total_elapsed_time += results.get("duration", 0)

                final_time = env.get_current_time()
                if "taxi_scheduling" in (control_modules or []):
                    traffic_state_collector.collect_if_needed(final_time)
                else:
                    traffic_state_collector.collect(final_time)
                break

            if remaining_duration < step_seconds:
                print(f"\nRemaining duration ({remaining_duration:.0f}s) is less than step size, ending simulation.")
                break

        print("\n" + "=" * 80)
        print("Simulation completed!")
        print(f"  Total checkpoints: {checkpoint_count}")
        print(f"  Total elapsed time: {total_elapsed_time:.0f}s")
        print(f"  Total steps: {accumulated_results['total_steps']}")
        print(f"  Total departed: {accumulated_results['total_departed']}")
        print(f"  Total arrived: {accumulated_results['total_arrived']}")
        print("=" * 80)

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

                    sorted_metrics = sorted(metrics.items())
                    for metric_name, metric_value in sorted_metrics:
                        if isinstance(metric_value, float):
                            if metric_name == "throughput":
                                print(f"  - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                            else:
                                print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
                        else:
                            print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value}")

                accumulated_results["average_module_metrics"] = average_module_metrics
            else:
                print("\nNo module metrics available for averaging.")

            print("=" * 80)

        if env:
            # Get checkpoint-scoped travel time (current checkpoint interval only)
            final_avg_travel_time = env.get_average_travel_time()
            # Get global travel time (accumulated across all checkpoints)
            global_avg_travel_time = env.get_global_average_travel_time()
            global_highway_avg_travel_time = env.get_global_highway_average_travel_time()
            global_arrived_count = env.get_global_arrived_count()
            global_highway_arrived_count = env.get_global_highway_arrived_count()
            final_avg_waiting_time = (
                env.get_average_waiting_time() if hasattr(env, "get_average_waiting_time") else 0.0
            )

            accumulated_results["final_avg_travel_time"] = final_avg_travel_time
            accumulated_results["global_avg_travel_time"] = global_avg_travel_time
            accumulated_results["global_highway_avg_travel_time"] = global_highway_avg_travel_time
            accumulated_results["global_arrived_count"] = global_arrived_count
            accumulated_results["global_highway_arrived_count"] = global_highway_arrived_count
            accumulated_results["final_avg_waiting_time"] = final_avg_waiting_time
            accumulated_results["final_time"] = env.get_current_time()
            accumulated_results["global_travel_times"] = global_travel_times
            accumulated_results["global_waiting_times"] = global_waiting_times

            # Print global travel time statistics
            print(f"\n  Global Travel Time Statistics (across all checkpoints):")
            print(f"    Global Average Travel Time: {global_avg_travel_time:.2f}s")
            print(f"    Global Arrived Vehicle Count: {global_arrived_count}")
            if global_highway_arrived_count > 0:
                print(f"    Global Highway Average Travel Time: {global_highway_avg_travel_time:.2f}s")
                print(f"    Global Highway Arrived Vehicle Count: {global_highway_arrived_count}")
            print(f"  Checkpoint-scoped Travel Time (current checkpoint only):")
            print(f"    Final Average Travel Time: {final_avg_travel_time:.2f}s")

            if global_waiting_times:
                total_waiting_time_sum = sum(
                    wt["avg_waiting_time"] * wt["waiting_count"] for wt in global_waiting_times
                )
                total_waiting_count = sum(wt["waiting_count"] for wt in global_waiting_times)
                overall_avg_waiting_time = (
                    total_waiting_time_sum / total_waiting_count if total_waiting_count > 0 else 0.0
                )
                accumulated_results["overall_avg_waiting_time"] = overall_avg_waiting_time
                print(f"\n  Overall Average Waiting Time (across all checkpoints): {overall_avg_waiting_time:.2f}s")
                print(f"  Final Average Waiting Time (current waiting vehicles): {final_avg_waiting_time:.2f}s")

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

        if wandb_run:
            if accumulated_results.get("average_module_metrics"):
                final_metrics = {
                    "final/checkpoint_count": checkpoint_count,
                    "final/total_elapsed_time": total_elapsed_time,
                    "final/total_steps": accumulated_results["total_steps"],
                    "final/total_departed": accumulated_results["total_departed"],
                    "final/total_arrived": accumulated_results["total_arrived"],
                }
                if accumulated_results.get("final_avg_travel_time") is not None:
                    final_metrics["final/avg_travel_time"] = accumulated_results["final_avg_travel_time"]
                if accumulated_results.get("global_avg_travel_time") is not None:
                    final_metrics["final/global_avg_travel_time"] = accumulated_results["global_avg_travel_time"]
                if accumulated_results.get("global_highway_avg_travel_time") is not None:
                    final_metrics["final/global_highway_avg_travel_time"] = accumulated_results[
                        "global_highway_avg_travel_time"
                    ]
                if accumulated_results.get("global_arrived_count") is not None:
                    final_metrics["final/global_arrived_count"] = accumulated_results["global_arrived_count"]
                if accumulated_results.get("global_highway_arrived_count") is not None:
                    final_metrics["final/global_highway_arrived_count"] = accumulated_results[
                        "global_highway_arrived_count"
                    ]
                if accumulated_results.get("final_avg_waiting_time") is not None:
                    final_metrics["final/avg_waiting_time"] = accumulated_results["final_avg_waiting_time"]
                if accumulated_results.get("overall_avg_waiting_time") is not None:
                    final_metrics["final/overall_avg_waiting_time"] = accumulated_results[
                        "overall_avg_waiting_time"
                    ]
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

        if "control_modules" in locals() and control_modules and accumulated_results.get("checkpoints"):
            print("\n" + "=" * 80)
            print("Average Control Module Performance Metrics (Up to Error)")
            print("=" * 80)
            average_module_metrics = calculate_average_module_metrics(
                checkpoints=accumulated_results["checkpoints"],
                control_modules=control_modules,
            )
            if average_module_metrics:
                for module_name in control_modules:
                    if module_name not in average_module_metrics:
                        print(f"\n{module_name.upper().replace('_', ' ')}: No metrics available")
                        continue
                    print(f"\n{module_name.upper().replace('_', ' ')}:")
                    metrics = average_module_metrics[module_name]
                    sorted_metrics = sorted(metrics.items())
                    for metric_name, metric_value in sorted_metrics:
                        if isinstance(metric_value, float):
                            if metric_name == "throughput":
                                print(f"  - {metric_name.replace('_', ' ').title()}: {int(metric_value)}")
                            else:
                                print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
                        else:
                            print(f"  - {metric_name.replace('_', ' ').title()}: {metric_value}")
            print("=" * 80)
            accumulated_results["average_module_metrics"] = average_module_metrics

        try:
            if "checkpoint_logger" in locals():
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

        stats = traffic_state_collector.get_stats() if "traffic_state_collector" in locals() else {}
        if stats:
            result["traffic_state_file"] = stats["filepath"]
            result["traffic_state_snapshots"] = stats["snapshot_count"]

        return result
    finally:
        if env:
            env.close()


def main():
    """Main entry point for joint checkpoint-based simulation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with checkpoint-based LLM agent control (joint modules)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sumo_config/jinan/jinan.sumocfg",
        help="Path to SUMO config file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600 * 24,
        help="Total simulation duration in seconds (default: 86400)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=3600,
        help="Checkpoint interval in seconds (default: 3600)"
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=30,
        help="Simulation step size in seconds (default: 30)"
    )
    parser.add_argument(
        "--min-step-seconds",
        type=float,
        default=1.0,
        help="Minimum step size in seconds to reduce control loop frequency (default: 1)"
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
        nargs="+",
        default=DEFAULT_CONTROL_MODULES,
        choices=KNOWN_CONTROL_MODULES,
        help="Control modules to optimize. Default: signal_timing bus_scheduling."
    )
    parser.add_argument(
        "--always-enabled",
        type=str,
        nargs="+",
        default=DEFAULT_ALWAYS_ENABLED,
        choices=KNOWN_CONTROL_MODULES,
        help="Modules enabled in the environment but not necessarily optimized by LLM. Default: signal_timing ramp_metering."
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
        default="sumo_joint_control",
        help="Wandb project name"
    )
    parser.add_argument(
        "--traffic-state-interval",
        type=float,
        default=300,
        help="Interval for collecting traffic state data in seconds (default: 300)"
    )
    parser.add_argument(
        "--traffic-lane-sample-rate",
        type=float,
        default=1.0,
        help="Sample ratio for traffic-state lane collection (0-1, default: 1.0)"
    )
    parser.add_argument(
        "--traffic-lane-sample-size",
        type=int,
        default=0,
        help="Sample size for traffic-state lane collection (0 disables, default: 0)"
    )
    parser.add_argument(
        "--waiting-passenger-interval",
        type=float,
        default=None,
        help="Interval in seconds for updating waiting passengers (default: every step)"
    )
    parser.add_argument(
        "--vehicle-subscription-mode",
        type=str,
        default="departed",
        choices=["all", "departed"],
        help="Vehicle subscription mode: all=subscribe every step, departed=subscribe new vehicles only"
    )
    parser.add_argument(
        "--max-interval-retries",
        type=int,
        default=2,
        help="Max retries per checkpoint interval on SUMO/TraCI failure (default: 2)"
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint XML to resume from (optional)"
    )
    parser.add_argument(
        "--max-reflection-turns",
        type=int,
        default=5,
        help="Maximum number of reflection turns (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
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

    config_path = resolve_config_path(args.config, workspace_root)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    results = run_joint_control_simulation(
        config_path=str(config_path),
        simulation_duration=args.duration,
        checkpoint_interval=args.checkpoint_interval,
        step_seconds=args.step_seconds,
        min_step_seconds=args.min_step_seconds,
        use_gui=args.gui,
        seed=args.seed,
        llm_model=args.llm_model,
        max_agent_turns=args.max_agent_turns,
        control_modules=args.control_modules,
        always_enabled=args.always_enabled,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        traffic_state_interval=args.traffic_state_interval,
        traffic_lane_sample_rate=args.traffic_lane_sample_rate,
        traffic_lane_sample_size=args.traffic_lane_sample_size,
        waiting_passenger_interval=args.waiting_passenger_interval,
        vehicle_subscription_mode=args.vehicle_subscription_mode,
        resume_checkpoint=args.resume_checkpoint,
        max_interval_retries=args.max_interval_retries,
        max_reflection_turns=args.max_reflection_turns,
        temperature=args.temperature,
        base_url=getattr(args, "base_url", None),
        user_query=args.query,
    )

    if results["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
