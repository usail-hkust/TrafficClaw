"""Unified method entry point for TrafficClaw LLM control simulations.

This module intentionally delegates to the existing run_single_control and
run_joint_control runners so their checkpoint, baseline, logging, and agent
flows stay 1:1 with the original scripts.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.path_utils import resolve_config_path


AVAILABLE_CONTROL_MODULES = [
    "signal_timing",
    "highway_speed_limit",
    "ramp_metering",
    "bus_scheduling",
    "subway_scheduling",
    "taxi_scheduling",
]


@dataclass(frozen=True)
class RunnerDefaults:
    config: str
    duration: float
    checkpoint_interval: float
    step_seconds: int
    llm_model: str
    max_agent_turns: int
    traffic_state_interval: float
    max_reflection_turns: int
    temperature: float
    wandb_project: str = "ChatCity"
    use_wandb: bool = True


DEFAULTS: Dict[str, RunnerDefaults] = {
    "signal_timing": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=1800,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=60,
        max_reflection_turns=5,
        temperature=0.0,
    ),
    "highway_speed_limit": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=1800,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=60,
        max_reflection_turns=5,
        temperature=0.0,
    ),
    "ramp_metering": RunnerDefaults(
        config="Data/sumo_config_highway/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=1800,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=60,
        max_reflection_turns=3,
        temperature=0.0,
    ),
    "bus_scheduling": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=1800,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=300,
        max_reflection_turns=5,
        temperature=0.0,
    ),
    "subway_scheduling": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=1800,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=300,
        max_reflection_turns=5,
        temperature=0.0,
    ),
    "taxi_scheduling": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600,
        checkpoint_interval=1800,
        step_seconds=10,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=60,
        max_reflection_turns=5,
        temperature=0.0,
    ),
    "joint": RunnerDefaults(
        config="Data/sumo_config/Manhattan/Manhattan.sumocfg",
        duration=3600 * 24,
        checkpoint_interval=3600,
        step_seconds=30,
        llm_model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        max_agent_turns=10,
        traffic_state_interval=300,
        max_reflection_turns=5,
        temperature=0.0,
        wandb_project="sumo_joint_control",
    ),
}


@dataclass
class SimulationRunOptions:
    control_modules: List[str]
    user_query: Optional[str] = None
    config: Optional[str] = None
    duration: Optional[float] = None
    checkpoint_interval: Optional[float] = None
    step_seconds: Optional[int] = None
    use_gui: bool = False
    seed: Optional[int] = 2026
    llm_model: Optional[str] = None
    max_agent_turns: Optional[int] = None
    use_wandb: Optional[bool] = None
    wandb_project: Optional[str] = None
    traffic_state_interval: Optional[float] = None
    max_reflection_turns: Optional[int] = None
    max_interval_retries: int = 2
    temperature: Optional[float] = None
    base_url: Optional[str] = None
    verbose: bool = True
    always_enabled: Optional[List[str]] = None
    min_step_seconds: float = 1.0
    traffic_lane_sample_rate: float = 1.0
    traffic_lane_sample_size: int = 0
    waiting_passenger_interval: Optional[float] = None
    vehicle_subscription_mode: str = "departed"
    resume_checkpoint: Optional[str] = None
    time_to_teleport: Optional[float] = None


class TrafficClawSimulationEntrypoint:
    """Callable facade for launching the original simulation entry flows."""

    def run(self, options: SimulationRunOptions) -> Dict[str, Any]:
        return run_simulation_entrypoint(options)


def default_profile_for_modules(control_modules: List[str]) -> RunnerDefaults:
    """Return the exact script-default profile for the selected entry point."""
    modules = _normalize_modules(control_modules)
    if len(modules) == 1:
        return DEFAULTS[modules[0]]
    return DEFAULTS["joint"]


def fill_defaults(options: SimulationRunOptions) -> SimulationRunOptions:
    """Fill missing options from the same defaults used by the original scripts."""
    defaults = default_profile_for_modules(options.control_modules)
    return replace(
        options,
        config=options.config or defaults.config,
        duration=options.duration if options.duration is not None else defaults.duration,
        checkpoint_interval=(
            options.checkpoint_interval
            if options.checkpoint_interval is not None
            else defaults.checkpoint_interval
        ),
        step_seconds=options.step_seconds if options.step_seconds is not None else defaults.step_seconds,
        llm_model=options.llm_model or defaults.llm_model,
        max_agent_turns=(
            options.max_agent_turns
            if options.max_agent_turns is not None
            else defaults.max_agent_turns
        ),
        use_wandb=options.use_wandb if options.use_wandb is not None else defaults.use_wandb,
        wandb_project=options.wandb_project or defaults.wandb_project,
        traffic_state_interval=(
            options.traffic_state_interval
            if options.traffic_state_interval is not None
            else defaults.traffic_state_interval
        ),
        max_reflection_turns=(
            options.max_reflection_turns
            if options.max_reflection_turns is not None
            else defaults.max_reflection_turns
        ),
        temperature=options.temperature if options.temperature is not None else defaults.temperature,
    )


def run_simulation_entrypoint(options: SimulationRunOptions) -> Dict[str, Any]:
    """Run the selected original simulation flow as a callable method."""
    options = fill_defaults(options)
    modules = _normalize_modules(options.control_modules)
    config_path = _resolve_existing_config(options.config)

    if len(modules) == 1:
        return _run_single_module(module_name=modules[0], options=options, config_path=config_path)

    from run_joint_control.run_joint_control import run_joint_control_simulation

    return run_joint_control_simulation(
        config_path=str(config_path),
        simulation_duration=float(options.duration),
        checkpoint_interval=float(options.checkpoint_interval),
        step_seconds=int(options.step_seconds),
        min_step_seconds=options.min_step_seconds,
        use_gui=options.use_gui,
        seed=options.seed,
        llm_model=str(options.llm_model),
        max_agent_turns=int(options.max_agent_turns),
        control_modules=modules,
        always_enabled=options.always_enabled,
        use_wandb=bool(options.use_wandb),
        wandb_project=options.wandb_project,
        traffic_state_interval=float(options.traffic_state_interval),
        traffic_lane_sample_rate=options.traffic_lane_sample_rate,
        traffic_lane_sample_size=options.traffic_lane_sample_size,
        waiting_passenger_interval=options.waiting_passenger_interval,
        vehicle_subscription_mode=options.vehicle_subscription_mode,
        resume_checkpoint=options.resume_checkpoint,
        max_interval_retries=options.max_interval_retries,
        max_reflection_turns=int(options.max_reflection_turns),
        temperature=float(options.temperature),
        base_url=options.base_url,
        user_query=options.user_query,
    )


def _run_single_module(
    module_name: str,
    options: SimulationRunOptions,
    config_path: Path,
) -> Dict[str, Any]:
    common_kwargs = {
        "config_path": str(config_path),
        "simulation_duration": float(options.duration),
        "checkpoint_interval": float(options.checkpoint_interval),
        "step_seconds": int(options.step_seconds),
        "use_gui": options.use_gui,
        "seed": options.seed,
        "llm_model": str(options.llm_model),
        "max_agent_turns": int(options.max_agent_turns),
        "control_modules": [module_name],
        "use_wandb": bool(options.use_wandb),
        "wandb_project": options.wandb_project,
        "traffic_state_interval": float(options.traffic_state_interval),
        "max_reflection_turns": int(options.max_reflection_turns),
        "max_interval_retries": options.max_interval_retries,
        "temperature": float(options.temperature),
        "base_url": options.base_url,
        "verbose": options.verbose,
        "user_query": options.user_query,
    }

    if module_name == "signal_timing":
        from run_single_control.run_traffic_signal_control import run_checkpoint_based_simulation

        return run_checkpoint_based_simulation(**common_kwargs)
    if module_name == "highway_speed_limit":
        from run_single_control.run_highway_speed_limit_control import run_checkpoint_based_simulation

        return run_checkpoint_based_simulation(**common_kwargs)
    if module_name == "ramp_metering":
        from run_single_control.run_ramp_metering_control import run_checkpoint_based_simulation

        return run_checkpoint_based_simulation(**common_kwargs)
    if module_name == "bus_scheduling":
        from run_single_control.run_bus_scheduling_control import run_checkpoint_based_simulation

        return run_checkpoint_based_simulation(**common_kwargs)
    if module_name == "subway_scheduling":
        from run_single_control.run_subway_scheduling_control import run_checkpoint_based_simulation

        return run_checkpoint_based_simulation(**common_kwargs)
    if module_name == "taxi_scheduling":
        from run_single_control.run_taxi_scheduling import run_taxi_scheduling_simulation

        return run_taxi_scheduling_simulation(
            **common_kwargs,
            time_to_teleport=options.time_to_teleport,
        )

    raise ValueError(f"Unsupported control module: {module_name}")


def _normalize_modules(control_modules: List[str]) -> List[str]:
    modules: List[str] = []
    seen = set()
    for module in control_modules:
        normalized = module.strip()
        if not normalized:
            continue
        if normalized not in AVAILABLE_CONTROL_MODULES:
            raise ValueError(f"Unknown control module: {normalized}")
        if normalized not in seen:
            modules.append(normalized)
            seen.add(normalized)
    if not modules:
        raise ValueError("At least one control module must be selected")
    return modules


def _resolve_existing_config(config: Optional[str]) -> Path:
    if not config:
        raise ValueError("SUMO config path is required")

    workspace_root = Path(__file__).resolve().parent
    config_path = resolve_config_path(config, workspace_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path
