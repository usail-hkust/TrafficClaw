"""
Utility functions for generating simulation identifiers and file prefixes.

This module provides unified functions for generating simulation_id and file_prefix
to avoid circular import issues.
"""

import os
import random
import string
from typing import Optional, List, Tuple
from datetime import datetime


def generate_file_prefix(
    config_name: Optional[str] = None,
    llm_name: Optional[str] = None,
    control_modules: Optional[List[str]] = None
) -> str:
    """
    Generate a file prefix from config name, LLM name, and control modules.
    
    Args:
        config_name: Config directory name (e.g., "jinan")
        llm_name: LLM model name (e.g., "deepseek-v3.2")
        control_modules: List of control module names (e.g., ["signal_timing", "highway_speed_limit"])
    
    Returns:
        File prefix string (e.g., "jinan_deepseek-v3.2_signal_timing_highway_speed_limit")
    """
    parts = []
    
    if config_name:
        # Sanitize config_name (remove invalid filename characters)
        safe_config_name = config_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        parts.append(safe_config_name)
    
    if llm_name:
        # Sanitize llm_name (remove invalid filename characters)
        safe_llm_name = llm_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        parts.append(safe_llm_name)
    
    if control_modules:
        # Sort control_modules for consistent ordering
        sorted_modules = sorted(control_modules)
        # Sanitize each module name
        safe_modules = [m.replace("/", "_").replace("\\", "_").replace(":", "_") for m in sorted_modules]
        parts.extend(safe_modules)
    
    if not parts:
        return "default"
    
    return "_".join(parts)


def generate_simulation_identifiers(
    config_name: Optional[str] = None,
    llm_name: Optional[str] = None,
    control_modules: Optional[List[str]] = None,
    simulation_id: Optional[str] = None
) -> Tuple[str, str]:
    """
    Unified function to generate simulation_id and file_prefix.

    This function centralizes the generation of simulation identifiers to ensure consistency
    across all modules that need these values.

    Args:
        config_name: Config directory name (e.g., "jinan")
        llm_name: LLM model name (e.g., "deepseek-v3.2")
        control_modules: List of control module names (e.g., ["signal_timing", "highway_speed_limit"])
        simulation_id: Optional simulation ID. If None, generates a new one with:
                      - Microsecond precision timestamp
                      - Random 4-character suffix
                      - Process ID (PID)
                      Format: "sim_YYYYMMDD_HHMMSS_ffffff_xxxx_pPID"
                      This ensures uniqueness even when multiple scripts run concurrently.

    Returns:
        Tuple of (simulation_id, file_prefix):
        - simulation_id: Unique simulation identifier (e.g., "sim_20240101_120000_123456_ab12_p12345")
        - file_prefix: File prefix string (e.g., "jinan_deepseek-v3.2_signal_timing_highway_speed_limit")
    """
    # Generate file_prefix
    file_prefix = generate_file_prefix(
        config_name=config_name,
        llm_name=llm_name,
        control_modules=control_modules
    )
    
    # Generate simulation_id if not provided
    if simulation_id is None:
        # Use microsecond precision timestamp + random suffix + PID to ensure uniqueness
        # across concurrent runs started at the same time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # %f = microseconds (6 digits)
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        pid = os.getpid()
        simulation_id = f"sim_{timestamp}_{random_suffix}_p{pid}"

    return simulation_id, file_prefix
