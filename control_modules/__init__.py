"""
Modular control system for different transportation facilities.
Supports traffic signals, subway scheduling, bus scheduling, etc.

Each control module can register its own control logic that will be applied
during simulation. Control logic is only applied after LLM agent planning.
"""

from .base import ControlModule, _get_workspace_root, _get_control_config_dir
from .signal_timing import TrafficSignalModule
from .subway_scheduling import SubwaySchedulingModule
from .bus_scheduling import BusSchedulingModule
from .highway_speed_limit import HighwaySpeedLimitModule
from .ramp_metering import RampMeteringModule
from .taxi_scheduling import TaxiSchedulingModule
from .registry import (
    CONTROL_MODULES,
    CONTROL_LOGIC_REGISTRY,
    get_control_module,
    list_available_modules,
    register_control_logic,
    get_control_logic,
    list_registered_control_logic,
    apply_control_logic,
)

__all__ = [
    # Base classes
    "ControlModule",
    "_get_workspace_root",
    "_get_control_config_dir",
    # Control modules
    "TrafficSignalModule",
    "SubwaySchedulingModule",
    "BusSchedulingModule",
    "HighwaySpeedLimitModule",
    "RampMeteringModule",
    "TaxiSchedulingModule",
    # Registry
    "CONTROL_MODULES",
    "CONTROL_LOGIC_REGISTRY",
    "get_control_module",
    "list_available_modules",
    "register_control_logic",
    "get_control_logic",
    "list_registered_control_logic",
    "apply_control_logic",
]
