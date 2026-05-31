"""
Base classes and utilities for control modules.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


def _get_workspace_root() -> Path:
    """Get the workspace root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def _get_control_config_dir() -> Path:
    """Get the control_config directory path."""
    workspace_root = _get_workspace_root()
    config_dir = workspace_root / "control_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


class ControlModule(ABC):
    """
    Abstract base class for transportation control modules.
    Each control module manages configuration for a specific facility type.
    """
    
    def __init__(self, module_name: str, config_filename: str, config_dir_name: Optional[str] = None):
        """
        Initialize control module.
        
        Args:
            module_name: Name of the control module (e.g., 'signal_timing', 'subway_scheduling')
            config_filename: Name of the configuration file (e.g., 'signal_timing.json')
            config_dir_name: Optional directory name from sumo config path (e.g., "jinan").
                           If provided, config path will be control_config/{config_dir_name}/{config_filename}
                           If None, uses default location: control_config/{config_filename}
                           Note: Config files are no longer automatically saved/loaded. Each experiment initializes config fresh.
        """
        self.module_name = module_name
        self.config_filename = config_filename
        if config_dir_name:
            # Use subdirectory based on sumo config directory name
            config_dir = _get_control_config_dir() / config_dir_name
            config_dir.mkdir(parents=True, exist_ok=True)
            self.config_path = config_dir / config_filename
        else:
            # Default location for backward compatibility
            self.config_path = _get_control_config_dir() / config_filename
        
    @abstractmethod
    def get_default_config(self, env: Any) -> Dict[str, Any]:
        """
        Get default configuration for this control module.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any], reference_config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for this control module.
        
        Args:
            config: Configuration dictionary to validate
            reference_config: Optional reference configuration to check completeness.
                            If provided, validates that config contains all required elements from reference_config.
            
        Returns:
            Tuple of (is_valid, error_message):
            - is_valid: True if configuration is valid, False otherwise
            - error_message: Error message string if invalid, None if valid
        """
        pass

    def load_config(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load configuration from disk.

        Args:
            env: Optional environment used to generate default config if load fails

        Returns:
            Configuration dictionary (empty if missing or invalid)
        """
        if not self.config_path.exists():
            if env is not None:
                return self.get_default_config(env=env)
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            print(f"Warning: {self.module_name} config is not a dictionary ({self.config_path})")
            return {}
        except Exception as exc:
            print(f"Warning: Failed to load {self.module_name} config ({self.config_path}): {exc}")
            if env is not None:
                return self.get_default_config(env=env)
            return {}

    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to disk.

        Args:
            config: Configuration dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as exc:
            print(f"Error saving {self.module_name} config ({self.config_path}): {exc}")
            return False
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize metrics dictionary for tracking performance.
        
        Returns:
            Dictionary with initialized metric structures
        """
        return {
            'total_reward': 0.0,
            'queue_length_episode': [],
            'waiting_time_episode': [],
            'global_waiting_times': []
        }
    
    def update_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        reward: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update training metrics with current step data.
        
        Args:
            metrics: Metrics dictionary to update
            env: SUMOEnv instance
            reward: Optional list of rewards for this step
            **kwargs: Additional arguments (e.g., step_duration) - subclasses may use these
        """
        # Default implementation - subclasses should override
        if reward:
            metrics['total_reward'] += sum(reward)
    
    def calculate_final_results(
        self,
        metrics: Dict[str, Any],
        env: Any
    ) -> Dict[str, float]:
        """
        Calculate final training results and metrics.
        
        Args:
            metrics: Metrics dictionary with collected data
            env: SUMOEnv instance
            
        Returns:
            Dictionary with final metric values
        """
        # Default implementation - subclasses should override
        return {
            "reward": float(metrics.get('total_reward', 0.0))
        }
