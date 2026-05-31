"""
Registry for control modules and control logic functions.
"""

from typing import Dict, Any, Optional, List, Callable
from .base import ControlModule
from .signal_timing import TrafficSignalModule
from .subway_scheduling import SubwaySchedulingModule
from .bus_scheduling import BusSchedulingModule
from .highway_speed_limit import HighwaySpeedLimitModule
from .ramp_metering import RampMeteringModule
from .taxi_scheduling import TaxiSchedulingModule


# Registry of all available control modules
CONTROL_MODULES: Dict[str, type] = {
    "signal_timing": TrafficSignalModule,
    "subway_scheduling": SubwaySchedulingModule,
    "bus_scheduling": BusSchedulingModule,
    "highway_speed_limit": HighwaySpeedLimitModule,
    "ramp_metering": RampMeteringModule,
    "taxi_scheduling": TaxiSchedulingModule,
}

# Registry of control logic functions
# Maps module_name to control logic function
CONTROL_LOGIC_REGISTRY: Dict[str, Callable] = {}


def get_control_module(module_name: str, config_dir_name: Optional[str] = None) -> Optional[ControlModule]:
    """
    Get a control module instance by name.
    
    Args:
        module_name: Name of the control module
        config_dir_name: Optional directory name from sumo config path (e.g., "jinan").
                        If provided, config path will be control_config/{config_dir_name}/
                        If None, uses default location: control_config/
                        Note: Config files are no longer automatically saved/loaded. Each experiment initializes config fresh.
        
    Returns:
        ControlModule instance or None if not found
    """
    module_class = CONTROL_MODULES.get(module_name)
    if module_class is None:
        print(f"Error: Unknown control module '{module_name}'")
        print(f"Available modules: {list(CONTROL_MODULES.keys())}")
        return None
    
    # Get config filename based on module name
    if module_name == 'signal_timing':
        config_filename = 'signal_timing.json'
    elif module_name == 'highway_speed_limit':
        config_filename = 'highway_speed_limit.json'
    elif module_name == 'subway_scheduling':
        config_filename = 'subway_scheduling.json'
    elif module_name == 'bus_scheduling':
        config_filename = 'bus_scheduling.json'
    elif module_name == 'ramp_metering':
        config_filename = 'ramp_metering.json'
    elif module_name == 'taxi_scheduling':
        config_filename = 'taxi_scheduling.json'
    else:
        config_filename = f'{module_name}.json'
    
    return module_class(config_dir_name=config_dir_name)


def list_available_modules() -> List[str]:
    """
    List all available control modules.
    
    Returns:
        List of module names
    """
    return list(CONTROL_MODULES.keys())


def register_control_logic(module_name: str):
    """
    Decorator to register control logic for a specific module.
    
    Usage:
        @register_control_logic("signal_timing")
        def apply_signal_timing_control(env, config, **kwargs):
            # Control logic here
            pass
    
    Args:
        module_name: Name of the control module
    """
    def decorator(func: Callable) -> Callable:
        CONTROL_LOGIC_REGISTRY[module_name] = func
        return func
    return decorator


def get_control_logic(module_name: str) -> Optional[Callable]:
    """
    Get the control logic function for a specific module.
    
    Args:
        module_name: Name of the control module
        
    Returns:
        Control logic function or None if not found
    """
    return CONTROL_LOGIC_REGISTRY.get(module_name)


def list_registered_control_logic() -> List[str]:
    """
    List all registered control logic modules.
    
    Returns:
        List of module names with registered control logic
    """
    return list(CONTROL_LOGIC_REGISTRY.keys())


def apply_control_logic(
    module_name: str,
    env: Any,
    config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Apply control logic for a specific module.
    
    Args:
        module_name: Name of the control module
        env: SUMOEnv instance
        config: Configuration for the module
        **kwargs: Additional arguments for the control logic
        
    Returns:
        Dictionary with control state information
    """
    control_func = get_control_logic(module_name)
    if control_func is None:
        raise ValueError(f"No control logic registered for module '{module_name}'")
    
    return control_func(env, config, **kwargs)


# ============================================================================
# Signal Timing Control Logic Registration
# ============================================================================

@register_control_logic("signal_timing")
def apply_signal_timing_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call TrafficSignalModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    module = TrafficSignalModule()
    return module.apply_control(env, config, current_time, control_state, **kwargs)


# ============================================================================
# Highway Speed Limit Control Logic Registration
# ============================================================================

@register_control_logic("highway_speed_limit")
def apply_highway_speed_limit_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call HighwaySpeedLimitModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    module = HighwaySpeedLimitModule()
    return module.apply_control(env, config, current_time, control_state, **kwargs)


# ============================================================================
# Subway Scheduling Control Logic Registration
# ============================================================================

@register_control_logic("subway_scheduling")
def apply_subway_scheduling_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call SubwaySchedulingModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    module = SubwaySchedulingModule()
    return module.apply_control(env, config, current_time, control_state, **kwargs)


# ============================================================================
# Bus Scheduling Control Logic Registration
# ============================================================================

@register_control_logic("bus_scheduling")
def apply_bus_scheduling_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call BusSchedulingModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    module = BusSchedulingModule()
    return module.apply_control(env, config, current_time, control_state, **kwargs)


# ============================================================================
# Ramp Metering Control Logic Registration
# ============================================================================

@register_control_logic("ramp_metering")
def apply_ramp_metering_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call RampMeteringModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    module = RampMeteringModule()
    return module.apply_control(env, config, current_time, control_state, **kwargs)


# ============================================================================
# Taxi Scheduling Control Logic Registration
# ============================================================================

@register_control_logic("taxi_scheduling")
def apply_taxi_scheduling_control(
    env: Any,
    config: Dict[str, Any],
    current_time: float,
    control_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to call TaxiSchedulingModule.apply_control().
    This allows the decorator-based registration to work with class methods.
    """
    # Prefer the module instance already initialized inside the env (keeps per-env caches like TAZ mappings).
    module = None
    try:
        enabled_controls = getattr(env, "enabled_controls", None)
        if isinstance(enabled_controls, dict):
            module_info = enabled_controls.get("taxi_scheduling") or {}
            module = module_info.get("module")
    except Exception:
        module = None

    if module is None:
        # Fallback: create a standalone module instance.
        # Use env.config_dir_name when available so config paths/TAZ inference remain consistent.
        config_dir_name = getattr(env, "config_dir_name", None)
        module = TaxiSchedulingModule(config_dir_name=config_dir_name)
    return module.apply_control(env, config, current_time, control_state, **kwargs)
