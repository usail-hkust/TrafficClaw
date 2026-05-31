# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 DeepCity Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing , software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepCity Interaction for verl framework.

This module provides the interaction layer between verl's agent loop and
DeepCity's transportation optimization environment. It uses LLMAgentManager
to manage dialogue state and action execution.
"""

import asyncio
import logging
import os
import sys
import time
import random
import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Add DeepCity project root to path for imports
DEEPCITY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if DEEPCITY_ROOT not in sys.path:
    sys.path.insert(0, DEEPCITY_ROOT)

from .base import BaseInteraction

# Import DeepCity modules
try:
    from utils.llm_agent_manager import LLMAgentManager
    from verl.actors.deepcity_worker import DeepCityWorker
    DEEPCITY_AVAILABLE = True
except ImportError:
    DEEPCITY_AVAILABLE = False
    LLMAgentManager = None
    DeepCityWorker = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepCityInteraction(BaseInteraction):
    """
    DeepCity transportation optimization interaction.
    
    Uses LLMAgentManager to manage dialogue state and action execution:
    - LLM outputs: ACTION: PLAN / DATA_ANALYSIS / POLICY_PLANNING / FINISH
    - LLMAgentManager parses actions, executes code/simulation, returns feedback
    - Reward = cumulative reward (sum of rewards across all turns)
      - Each turn: reward = module_count_reward + metric_improvement_reward
      - module_count_reward = num_improved_modules / total_modules
      - metric_improvement_reward = sum of metric improvements for improved modules
      - Metric improvements calculated based on key metrics per module:
        * signal_timing: avg_travel_time↓, avg_queue_len↓
        * bus_scheduling: avg_passenger_waiting_time↓, total_fuel_consumption_g↓
        * highway_speed_limit: avg_travel_time↓
        * taxi_scheduling: avg_wait_time↓, total_income↑
    
    Methods:
        start_interaction: Initialize LLMAgentManager and generate dialogue template
        get_messages: Return complete messages including system prompt
        generate_response: Process LLM response using manager.step()
        finalize_interaction: Clean up simulation resources
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self._lock = threading.Lock()  #   Added: global state lock, protects concurrent access to _instance_dict
        self.max_turns = config.get("max_turns", 10)
        self._worker_actors = {}  #   Store Worker Actor: instance_id -> DeepCityWorker Actor
        
        #   Semaphore: limit concurrent SUMO simulation count (None means no limit)
        max_concurrent_simulations = config.get("max_concurrent_simulations", None)
        self._simulation_semaphore = asyncio.Semaphore(max_concurrent_simulations) if max_concurrent_simulations else None
        
        #   Key metrics config for each module (used for reward calculation)
        self.module_key_metrics = {
            "signal_timing": [
                {"name": "avg_travel_time", "direction": "lower"},  # Lower is better
                {"name": "avg_queue_len", "direction": "lower"}
            ],
            "bus_scheduling": [
                {"name": "avg_passenger_waiting_time", "direction": "lower"},
                {"name": "total_fuel_consumption_g", "direction": "lower"}  # Energy consumption
            ],
            "highway_speed_limit": [
                {"name": "avg_travel_time", "direction": "lower"},
                {"name": "throughput", "direction": "higher"}
            ],
            "taxi_scheduling": [
                {"name": "avg_wait_time", "direction": "lower"},
                {"name": "total_income", "direction": "higher"},  # Higher is better
                {"name": "avg_income_per_taxi", "direction": "higher"}
            ]
        }
        
        #   Environment selection mode: fixed (manual list) or random (random combination)
        environment_mode = config.get("environment_mode", "fixed")
        if environment_mode == "random":
            self.environments = self._generate_random_environments(config)
        else:
            self.environments = config["environments"]
        
        #   Val independent environment config (optional)
        # If val_environments is configured, val uses independent environment pool
        # If not configured, val reuses train's environments
        self.val_environments = config.get("val_environments", None)
        self.val_simulation_duration = config.get("val_simulation_duration", None)  # Val-specific simulation duration
        if self.val_environments:
            logger.info(f"Val environments configured: {len(self.val_environments)} environments (independent from train)")
            if self.val_simulation_duration:
                logger.info(f"Val simulation duration: {self.val_simulation_duration}s ({self.val_simulation_duration/3600:.1f}h)")
        else:
            logger.info("Val environments not configured, will reuse train environments")
        
        #   No longer set control_modules here
        # It will be obtained from Master's context at start_interaction()
        # This way each environment only uses its corresponding control modules
        self.control_modules = None  # Will be set in start_interaction
        
        # Multi-Master architecture (see verl_architecture.md)
        # num_masters Train Masters run in parallel, each triggers worker sampling when reaching checkpoint
        # num_val_masters Val Masters run in parallel (can share Master Actor with Train)
        self.num_masters = config.get("num_masters", 2)
        self.num_val_masters = config.get("num_val_masters", self.num_masters)  # Default same as train
        self._total_masters = max(self.num_masters, self.num_val_masters)  # Actual total Masters to create
        self.master_name_prefix = config.get("master_name_prefix", "DeepCityMaster")
        self._masters = {}  # Cache master connections {master_id: actor}
        logger.info(f"Master config: num_masters(train)={self.num_masters}, num_val_masters={self.num_val_masters}, total={self._total_masters}")
        
        #   Reward weight configuration
        reward_weights = config.get("reward_weights", {})
        self.w_metric = reward_weights.get("metric_improvement", 1.0)
        self.w_module_count = reward_weights.get("module_count", 1.0)
        self.w_judge = reward_weights.get("judge", 1.0)
        logger.info(f"Reward weights: metric_improvement={self.w_metric}, module_count={self.w_module_count}, judge={self.w_judge}")
        
        # Check DeepCity availability
        if not DEEPCITY_AVAILABLE:
            logger.warning("DeepCity modules not available. Some features may not work.")
    
    def _generate_random_environments(self, config: dict) -> list:
        """
        Enumerate all possible environment combinations from random_environment_pool config.
        Combination = each map × all non-empty subsets of modules (constrained by min/max_modules).
        
        e.g. 2 maps × 4 modules → 2 × (C(4,1)+C(4,2)+C(4,3)+C(4,4)) = 2 × 15 = 30 environments.
        Master will rotate from this full pool using "fewest runs priority + random" strategy.
        
        If control_modules includes highway_speed_limit, use sumo_config_highway road network.
        """
        from itertools import combinations
        
        pool_config = config.get("random_environment_pool", {})
        sumo_paths = pool_config.get("sumo_config_paths", [])
        sumo_highway_paths = pool_config.get("sumo_config_highway_paths", [])
        all_modules = pool_config.get("control_modules", [])
        min_modules = pool_config.get("min_modules", 1)
        max_modules = pool_config.get("max_modules", len(all_modules))
        
        if not sumo_paths:
            raise ValueError("random_environment_pool.sumo_config_paths is empty")
        if not all_modules:
            raise ValueError("random_environment_pool.control_modules is empty")
        if sumo_highway_paths and len(sumo_highway_paths) != len(sumo_paths):
            raise ValueError(
                "sumo_config_highway_paths must have same length as sumo_config_paths "
                f"(got {len(sumo_highway_paths)} vs {len(sumo_paths)})"
            )
        
        max_modules = min(max_modules, len(all_modules))
        min_modules = max(1, min(min_modules, max_modules))
        
        environments = []
        # First decide module combinations, use highway map only when highway_speed_limit is present
        for n in range(min_modules, max_modules + 1):
            for module_combo in combinations(all_modules, n):
                use_highway = "highway_speed_limit" in module_combo and sumo_highway_paths
                paths_to_use = sumo_highway_paths if use_highway else sumo_paths
                for path in paths_to_use:
                    environments.append({
                        "sumo_config_path": path,
                        "control_modules": list(module_combo),
                        "episodes": 1,
                    })
        
        random.shuffle(environments)
        
        logger.info(
            f"Random environment mode: enumerated {len(environments)} environments "
            f"({len(sumo_paths)} maps × all subsets of {len(all_modules)} modules, "
            f"size {min_modules}~{max_modules})"
        )
        for i, env in enumerate(environments):
            logger.info(f"  env[{i}]: {env['sumo_config_path'].split('/')[-2]} × {env['control_modules']}")
        
        return environments
    
    def _get_master_id(self, instance_id: str) -> int:
        """
        Assign to corresponding Master based on instance_id.
        Uses the number in sample_id for deterministic assignment, ensuring all rollouts of the same sample go to the same Master.
        
        Supported instance_id formats:
        - "sample_0_w123456" → extract "sample_0" → number 0 → Master_0
        - "sample_1_w789012" → extract "sample_1" → number 1 → Master_1
        - "batch_0_w123456" → extract "batch_0" → number 0 → Master_0 (compatible with old format)
        
        Assignment rules:
        - sample_0 / batch_0 → 0 % num_masters → Master_0
        - sample_1 / batch_1 → 1 % num_masters → Master_1
        - sample_2 / batch_2 → 2 % num_masters → Master_0 (if num_masters=2)
        """
        # Extract prefix (remove _w suffix)
        if "_w" in instance_id:
            prefix = instance_id.split("_w")[0]
        else:
            prefix = instance_id
        
        # Extract numeric part for deterministic assignment
        # batch_0 → 0, batch_1 → 1, batch_2 → 2, ...
        if '_' in prefix:
            try:
                num = int(prefix.split('_')[-1])
                return num % self.num_masters
            except ValueError:
                pass
        
        # Fallback to hash (for non-standard instance_id formats)
        import hashlib
        hash_value = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
        return hash_value % self.num_masters
    
    def _get_val_master_id(self, instance_id: str) -> int:
        """
        Val-specific routing: assign to val Master based on instance_id.
        Same logic as _get_master_id, but uses num_val_masters for modulo.
        """
        if "_w" in instance_id:
            prefix = instance_id.split("_w")[0]
        else:
            prefix = instance_id
        
        if '_' in prefix:
            try:
                num = int(prefix.split('_')[-1])
                return num % self.num_val_masters
            except ValueError:
                pass
        
        import hashlib
        hash_value = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
        return hash_value % self.num_val_masters
    
    def _build_master_config(self) -> Dict[str, Any]:
        """
        Build Master config parameter dict.
        Separate config extraction logic to keep main flow clean.
        """
        if not self.environments:
            raise ValueError("No environments configured. Please specify 'environments' in interaction config.")
        
        config = {
            "environments": self.environments,
            "checkpoint_interval": self.config.get("checkpoint_interval", 300),
            "seed": self.config.get("seed", 42),
            "seed_increment": self.config.get("seed_increment", 1000),
            "use_gui": self.config.get("use_gui", False),
            "simulation_duration": self.config.get("simulation_duration"),
            "auto_restart": self.config.get("auto_restart", True),
            "max_episodes": self.config.get("max_episodes"),
            #   HTTP mode support
            "use_http": self.config.get("use_http", False),
            "sumo_server_url": self.config.get("sumo_server_url"),
            "http_timeout": self.config.get("http_timeout", 300.0),
            #   Batch parallel environment allocation
            "num_masters": self._total_masters,  # Pass actual total Master count (including val-only)
            "num_val_masters": self.num_val_masters,
            "num_train_masters": self.num_masters,  # Actual train master count (val-only Masters skip train env)
            #   Val independent environment config (None means Master reuses train environments)
            "val_environments": self.val_environments,
            "val_simulation_duration": self.val_simulation_duration,
        }
        
        return config
    
    async def _get_or_create_master(self, instance_id: str = None, is_validate: bool = False):
        """
        Get or create DeepCityMaster for given instance.
        Optimizations: exponential backoff retry, config unpacking, cache health check.
        
          Changed to async method, uses await instead of ray.get() to avoid blocking event loop.
          Supports val independent routing: is_validate=True uses num_val_masters routing.
        """
        if instance_id is None:
            master_id = 0
        elif is_validate:
            master_id = self._get_val_master_id(instance_id)
        else:
            master_id = self._get_master_id(instance_id)
        master_name = f"{self.master_name_prefix}_{master_id}"

        # 1. Check if local cache is valid (Fast Path)
        if master_id in self._masters:
            master = self._masters[master_id]
            # Optional: if worried about Master crash, can ping here, or let upper logic handle ActorDeadError
            return master

        # 2. Prepare config parameters (separate lengthy config extraction logic to keep main flow clean)
        master_config = self._build_master_config()

        # 3. Concurrency-safe Get-or-Create loop
        import ray
        max_retries = 10
        base_delay = 1.0  # Initial wait 1 second
        
        for attempt in range(max_retries):
            try:
                # Attempt A: Get existing Master
                master = ray.get_actor(master_name)
                logger.info(f"[Worker] Connected to existing {master_name}")
                
                #     [Critical fix added]: Check zombie state and revive
                try:
                    #   Use await instead of ray.get() to avoid blocking event loop
                    status = await master.get_status.remote()
                    if not status.get("running"):
                        logger.warning(f"[Worker] Master {master_name} is alive but NOT running (Zombie). Restarting...")
                        
                        # Force restart Master's simulation environment
                        start_result = await master.start.remote()
                        if not start_result.get("success"):
                            raise RuntimeError(f"Failed to restart zombie master: {start_result.get('error')}")
                        logger.info(f"[Worker] Successfully restarted zombie {master_name}")
                    else:
                        logger.info(f"[Worker] Master {master_name} is healthy and running.")
                        
                except Exception as e:
                    # If even getting status fails (e.g. Actor crashed), may need to rebuild
                    logger.error(f"[Worker] Existing master seems broken: {e}")
                    raise ValueError("Master broken")  # Raise ValueError to trigger rebuild flow below
                
                self._masters[master_id] = master
                return master

            except ValueError:
                # Master doesn't exist or is damaged, enter creation flow
                pass

            # Attempt B: Create new Master (only when unable to get existing one)
            try:
                logger.info(f"[Worker] Attempting to create {master_name} (attempt {attempt + 1})")
                
                # Lazy import to avoid circular dependency
                from verl.actors.deepcity_master import DeepCityMaster

                # Use options to ensure naming uniqueness
                master = DeepCityMaster.options(
                    name=master_name,
                    num_cpus=1,
                    lifetime="detached",  # Suggestion: set detached to prevent Master destruction after Driver exits
                    max_concurrency=100  # Suggestion: set concurrency limit based on load
                ).remote(
                    master_id=master_id,
                    **master_config  # Use dict unpacking to pass parameters
                )

                #   Use await instead of ray.get() to avoid blocking event loop
                # Note: if other workers get_actor at this moment, they may get a master that is still starting
                result = await master.start.remote()
                
                if not result.get("success"):
                    # If start logic fails, destroy actor and raise exception
                    ray.kill(master)
                    raise RuntimeError(f"Failed to start {master_name}: {result.get('error')}")

                self._masters[master_id] = master
                logger.info(f"[Worker] Successfully created and started {master_name}")
                return master

            except ValueError:
                # Race condition: another Worker created it just as we were about to
                # No action needed, directly enter next loop, ray.get_actor at loop head will succeed
                logger.debug(f"{master_name} created by another worker just now.")
            
            except Exception as e:
                # Other unknown errors (e.g. insufficient resources), raise directly
                logger.error(f"Unexpected error creating master: {e}")
                raise e

            # 4. Exponential backoff strategy (Exponential Backoff + Jitter)
            # Avoid all Workers retrying simultaneously, reduce Ray GCS pressure
            if attempt < max_retries - 1:
                sleep_time = base_delay * (1.5 ** attempt) + random.uniform(0, 1)
                logger.debug(f"Waiting {sleep_time:.2f}s before retry...")
                await asyncio.sleep(sleep_time)  #   Use asyncio.sleep instead of time.sleep

        raise RuntimeError(f"Failed to get or create {master_name} after {max_retries} attempts")
    
    @staticmethod
    def _format_simulation_time(seconds: float) -> Dict[str, Any]:
        """Format simulation time in seconds to human-readable time format.
        
        Matches the original project's format_simulation_time() function.
        """
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
            time_period = "Night/Early Morning"
            period_description = "Night/early morning period (0:00-6:00). Low traffic levels."
        
        return {
            "hours": hours,
            "minutes": minutes,
            "time_string": time_string,
            "time_period": time_period,
            "period_description": period_description,
            "day_of_simulation": day_number,
        }

    @staticmethod
    def _get_module_item_counts(current_configs: Dict[str, Any], control_modules: List[str]) -> Dict[str, str]:
        """Get human-readable item counts for each control module."""
        counts = {}
        for module_name in control_modules:
            if module_name in current_configs:
                config_data = current_configs[module_name]
                config = config_data.get('config', config_data) if isinstance(config_data, dict) else {}
                num_items = len(config) if isinstance(config, dict) else 0
                label_map = {
                    "signal_timing": f"{num_items} intersections",
                    "highway_speed_limit": f"{num_items} highway segments",
                    "bus_scheduling": f"{num_items} bus routes",
                    "taxi_scheduling": f"{num_items} taxi config entries",
                    "subway_scheduling": f"{num_items} subway routes",
                    "ramp_metering": f"{num_items} ramps",
                }
                counts[module_name] = label_map.get(module_name, f"{num_items} items")
            else:
                counts[module_name] = "N/A"
        return counts

    def _generate_initial_prompt(
        self,
        current_time: float,
        checkpoint_interval: float,
        current_control_modules: List[str],
        master_context: Dict[str, Any],
        current_configs: Dict[str, Any],
    ) -> str:
        """Generate initial prompt aligned with the original project's run_*_control.py scripts."""
        time_info = self._format_simulation_time(current_time)
        next_checkpoint_time = current_time + checkpoint_interval
        next_time_info = self._format_simulation_time(next_checkpoint_time)
        
        total_duration = master_context.get("run_duration", 86400)
        elapsed_time = current_time
        remaining_time = max(0, total_duration - current_time)
        progress = (elapsed_time / total_duration * 100) if total_duration > 0 else 0
        checkpoint_count = int(current_time / checkpoint_interval) if checkpoint_interval > 0 else 0
        
        # Module item counts
        item_counts = self._get_module_item_counts(current_configs, current_control_modules)
        item_lines = "\n".join(f"- {name}: {count}" for name, count in item_counts.items())
        
        # Module-specific or generic opening
        if len(current_control_modules) == 1:
            module = current_control_modules[0]
            module_titles = {
                "signal_timing": "traffic signal timing",
                "highway_speed_limit": "highway speed limits",
                "bus_scheduling": "bus scheduling",
                "taxi_scheduling": "taxi dispatch and repositioning",
                "subway_scheduling": "subway scheduling",
                "ramp_metering": "ramp metering control",
            }
            title = module_titles.get(module, module.replace('_', ' '))
            opening = f"You are optimizing {title} for a city transportation network."
        else:
            module_list = ", ".join(m.replace('_', ' ') for m in current_control_modules)
            opening = f"You are jointly optimizing multiple transportation systems ({module_list}) for a city network."
        
        prompt = f"""{opening}

Current Status:
- **Current City Time**: {time_info['time_string']} (Day {time_info['day_of_simulation']} of simulation)
- **Time Period**: {time_info['time_period']} - {time_info['period_description']}
- Current simulation time: {current_time:.0f} seconds ({time_info['time_string']})
- Total simulation duration: {total_duration:.0f} seconds ({total_duration/3600:.2f} hours)
- Elapsed time: {elapsed_time:.0f} seconds ({elapsed_time/3600:.2f} hours)
- Remaining time: {remaining_time:.0f} seconds ({remaining_time/3600:.2f} hours)
- Progress: {progress:.1f}%
- Checkpoint: {checkpoint_count}
- Control modules: {', '.join(current_control_modules)}
{item_lines}

**Optimization Time Window (NO SNAPSHOT DATA AVAILABLE. This period is in the FUTURE.):**
- **Optimization Start**: {time_info['time_string']} (Day {time_info['day_of_simulation']}, {time_info['time_period']})
- **Optimization End**: {next_time_info['time_string']} (Day {next_time_info['day_of_simulation']}, {next_time_info['time_period']})
- **Duration**: {checkpoint_interval:.0f} seconds ({checkpoint_interval/3600:.2f} hours)
- **Your optimized policies will be applied from {time_info['time_string']} to {next_time_info['time_string']}**

For configuration format, constraints, optimization strategies, and task description, follow each module's domain knowledge (provided via GET_CONTROL_API).

Available time range for analysis: 0 to {current_time:.0f} seconds (00:00 to {time_info['time_string']}). You can ONLY use the traffic snapshot data within this range.

Begin your analysis."""
        return prompt

    def _calculate_metric_improvement_reward(
        self,
        module_name: str,
        current_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a single module based on metric improvement magnitude.
        
        Args:
            module_name: Module name
            current_metrics: Current turn's metrics
            best_metrics: Historical best metrics
            
        Returns:
            Average improvement ratio (0.0 - 1.0+)
        """
        if module_name not in self.module_key_metrics:
            return 0.0
        
        key_metrics = self.module_key_metrics[module_name]
        improvements = []
        
        for metric_config in key_metrics:
            metric_name = metric_config["name"]
            direction = metric_config["direction"]
            
            current_value = current_metrics.get(metric_name)
            best_value = best_metrics.get(metric_name) if best_metrics else None
            
            # Skip missing metrics
            if current_value is None:
                continue
            
            #   Take absolute value for metrics that may be negative (e.g. fuel consumption)
            # SUMO's getFuelConsumption() may return negative values
            if metric_name in ["total_fuel_consumption_g", "fuel_consumption"]:
                current_value = abs(current_value)
                if best_value is not None:
                    best_value = abs(best_value)
            
            # If no historical best value, use current value as baseline (improvement is 0)
            if best_value is None or best_value == 0:
                continue
            
            # Calculate improvement ratio
            if direction == "lower":
                # Lower is better: improvement = (best - current) / best
                # e.g.: best=100, current=80 → improvement = 0.2 (20% improvement)
                improvement = (best_value - current_value) / abs(best_value)
            else:  # direction == "higher"
                # Higher is better: improvement = (current - best) / best
                # e.g.: best=100, current=120 → improvement = 0.2 (20% improvement)
                improvement = (current_value - best_value) / abs(best_value)
            
            # Clamp improvement ratio to reasonable range (avoid outliers)
            improvement = max(-1.0, min(1.0, improvement))
            improvements.append(improvement)
        
        # Return average of all key metric improvement ratios
        if improvements:
            return sum(improvements) / len(improvements)
        return 0.0
    
    async def _get_default_configs(self, master, control_modules: list = None, is_validate: bool = False) -> Dict[str, Any]:
        """
        Get default control configs from Master's env.enabled_controls.
        
        Args:
            master: Ray actor handle for DeepCityMaster
            control_modules: List of control module names to extract configs for
            is_validate: If True, get configs from val_env instead of train env
            
        Returns:
            Dictionary of default configs by module name
        """
        import copy
        
        try:
            #   Use await instead of ray.get(), pass is_validate to get correct environment's controls
            enabled_controls = await master.get_enabled_controls.remote(is_validate=is_validate)
            
            if not enabled_controls:
                logger.warning("No enabled_controls found in Master")
                return {}
            
            # Use provided control_modules or all enabled controls
            modules_to_extract = control_modules or list(enabled_controls.keys())
            
            # Extract configs for modules that LLM agent will optimize
            #   Preserve complete {module, config} structure so LLMAgentManager can call module.validate_config()
            current_configs = {}
            for module_name in modules_to_extract:
                if module_name in enabled_controls:
                    module_info = enabled_controls[module_name]
                    module = module_info.get('module')
                    config = module_info.get('config', {})
                    if config:
                        # Deep copy config to avoid modifying env.enabled_controls
                        # Preserve module object (class can be pickled)
                        current_configs[module_name] = {
                            'module': module,
                            'config': copy.deepcopy(config)
                        }
                        logger.info(f"Loaded default config for {module_name}: {len(config)} entries (with module object)")
                    else:
                        logger.warning(f"Config for {module_name} is empty in enabled_controls")
                else:
                    logger.warning(f"Module {module_name} not found in enabled_controls")
            
            return current_configs
            
        except Exception as e:
            logger.error(f"Failed to get default configs from Master: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _step_master_and_get_context(self, instance_id: str = None) -> dict:
        """
        Trigger Master to run to next checkpoint and get context.
        
        This is the core of the main loop:
        1. Trigger Master to run simulation (checkpoint_interval seconds)
        2. Master automatically saves checkpoint (t and t-1)
        3. Master automatically collects traffic_states
        4. Return context for Agent to use
        
        Multi-Master architecture: connect to corresponding Master based on instance_id to get checkpoint.
        """
        master = self._get_or_create_master(instance_id)
        if not master:
            raise RuntimeError("Failed to get or create master")
        
        try:
            import ray
            
            #   Core: trigger Master to run to next checkpoint
            checkpoint_interval = self.config.get("checkpoint_interval", 1800)
            step_result = ray.get(master.step.remote(checkpoint_interval))
            
            if not step_result.get("success"):
                raise RuntimeError(f"Master step failed: {step_result.get('error')}")
            
            logger.info(f"Master stepped to time {step_result.get('current_time')}s")
            
            #   Get context (includes checkpoint path, traffic_states, etc.)
            ctx = ray.get(master.get_context.remote())
            
            # Add master_id info
            if instance_id:
                ctx["master_id"] = self._get_master_id(instance_id)
            
            # Add checkpoint_interval for Agent to use
            ctx["checkpoint_interval"] = checkpoint_interval
            ctx["test_duration"] = checkpoint_interval
            
            return ctx
            
        except Exception as e:
            logger.error(f"Failed to step master and get context: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _ensure_imports(self):
        """Check DeepCity modules are available."""
        if not DEEPCITY_AVAILABLE:
            raise ImportError("DeepCity modules not available. Please check your PYTHONPATH.")

    async def start_interaction(
        self, 
        instance_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        is_validate: bool = False,
        **kwargs
    ) -> str:
        """
        Initialize a new interaction instance.
        
          Use LLMAgentManager.reset() to generate complete dialogue template (including system prompt).
        
        Args:
            instance_id: Unique identifier for this interaction
            checkpoint_path: Path to SUMO checkpoint (optional)
            is_validate: Whether it's validation mode (True uses val_env, doesn't affect train timeline)
            
        Returns:
            instance_id: The assigned instance ID 
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Get or create Master (async, val uses independent routing)
        master = await self._get_or_create_master(instance_id, is_validate=is_validate)
        master_id = (self._get_val_master_id(instance_id) if is_validate else self._get_master_id(instance_id)) if instance_id else 0
        
        #     [After fix] Atomic call: regardless of which Worker I am, only call this one interface
        # Master internally decides whether to Step or return data directly
        # This solves Check-Then-Act race condition, avoiding multiple Workers triggering Step simultaneously
        #   Use await instead of ray.get() to avoid blocking event loop
        try:
            master_context = await master.prepare_batch_and_get_context.remote(instance_id, is_validate=is_validate)
            logger.info(f"Worker {instance_id} prepared batch on Master {master_id}")
        except Exception as e:
            logger.error(f"Failed to prepare batch on Master: {e}")
            raise
        
        #   Safety net: from here worker is registered on Master, if subsequent steps fail,
        # must unregister to prevent checkpoint_in_use being True forever (causing subsequent val unable to step)
        try:
            return await self._complete_start_interaction(
                instance_id, master, master_id, master_context,
                checkpoint_path, is_validate, **kwargs
            )
        except Exception as e:
            #   Critical: when subsequent steps fail, must unregister worker to avoid checkpoint_in_use leak
            logger.error(f"start_interaction failed after prepare_batch, unregistering worker {instance_id}: {e}")
            try:
                await master.unregister_worker.remote(instance_id, is_validate=is_validate)
            except Exception as unreg_err:
                logger.warning(f"Failed to unregister worker {instance_id} during cleanup: {unreg_err}")
            raise
    
    async def _complete_start_interaction(
        self,
        instance_id: str,
        master,
        master_id: int,
        master_context: dict,
        checkpoint_path: Optional[str],
        is_validate: bool,
        **kwargs
    ) -> str:
        """Subsequent steps of start_interaction (from after prepare_batch_and_get_context).
        
        Split out for unified try-except handling of unregister_worker in outer layer.
        """
        # Use provided checkpoint_path or get from master
        effective_checkpoint = checkpoint_path or master_context.get("checkpoint_path")
        
        #   Get current environment's control_modules from Master context
        current_control_modules = master_context.get("control_modules", [])
        if not current_control_modules:
            # Fallback: if Master didn't return, use default values from config
            current_control_modules = self.control_modules or [
                "bus_scheduling",
                "signal_timing",
                "taxi_scheduling",
                "highway_speed_limit",
            ]
        
        logger.info(
            f"Instance {instance_id}: Using control modules from Master: {current_control_modules} "  
            f"(env_index={master_context.get('current_env_index')}, "
            f"episodes={master_context.get('current_env_episodes')})"
        )
        
        #   Create DeepCityWorker Actor (separate process, avoid GIL contention)
        worker_actor = None
        if DeepCityWorker is not None:
            try:
                import ray
                worker_actor = DeepCityWorker.remote(
                    worker_id=instance_id,
                    master_id=master_id,
                    available_control_modules=current_control_modules,  #   Use current environment's modules
                    max_turns=self.max_turns,
                    max_reflection_turns=self.config.get("max_reflection_turns", 5),
                    max_memory_items=self.config.get("max_memory_items", 10),
                    #   HTTP mode config
                    use_http=self.config.get("use_http", False),
                    sumo_server_url=self.config.get("sumo_server_url"),
                    http_timeout=self.config.get("http_timeout", 300.0),
                )
                logger.info(f"Created DeepCityWorker Actor for instance {instance_id} (http_mode: {self.config.get('use_http', False)})")
            except Exception as e:
                logger.error(f"Failed to create DeepCityWorker Actor: {e}")
                raise
        else:
            raise ImportError("DeepCityWorker not available")
        
        # Prepare context (for manager.reset() and subsequent operations)
        # Consistent with agent_context in original project's run_highway_speed_limit_control.py
        current_configs = await self._get_default_configs(master, current_control_modules, is_validate=is_validate)
        print(f"[DeepCityInteraction] current_configs for {instance_id}: {list(current_configs.keys())}")
        for module_name, module_info in current_configs.items():
            config = module_info.get('config', {})
            print(f"[DeepCityInteraction]   {module_name}: {len(config)} entries")
        
        context = {
            **master_context,  # From Master: graphs, traffic_states_filepath, etc.
            
            # === Control Config ===
            "current_configs": current_configs,  # Get default config from Master's env.enabled_controls
            "available_control_modules": current_control_modules,  #   Use current environment's modules
            
            # === Checkpoint & Config ===
            "checkpoint_path": effective_checkpoint,
            "config_path": master_context.get("config_path"),  # SUMO config file path (for policy simulation, from Master)
            "checkpoint_interval": master_context.get("checkpoint_interval", 1800),
            "test_duration": master_context.get("checkpoint_interval", 1800),  # test_duration must equal checkpoint_interval
            
            # === Simulation Parameters ===
            # IMPORTANT: simulation_id must match Master's traffic_states file naming
            # Master uses: f"master_{master_id}" (no instance_id)
            # So Worker must use the same to find the traffic_states file
            "simulation_id": f"master_{master_context.get('master_id', 0)}",  # Simulation ID (for data filtering)
            "use_gui": self.config.get("use_gui", False),  # Whether to use SUMO GUI
            "run_duration": master_context.get("run_duration", self.config.get("simulation_duration", 86400)),  # Total simulation duration (val uses val_simulation_duration)
            "seed": master_context.get("seed", 42),  # Random seed
            
            # === File Naming Parameters ===
            "control_modules": current_control_modules,  #   Use current environment's modules (for generating file names)
        }
        
        #   Use manager.reset() to generate complete dialogue template (including system prompt)
        # Note: completely ignore initial_messages in kwargs (dataset samples)
        # DeepCity's optimization is entirely driven by system prompt, dataset is only to satisfy verl DataLoader requirements
        
        # Dynamically generate initial_prompt (aligned with original project's run_*_control.py format)
        current_time = master_context.get("current_time", 0)
        checkpoint_interval = master_context.get("checkpoint_interval", 1800)
        
        initial_prompt = self._generate_initial_prompt(
            current_time=current_time,
            checkpoint_interval=checkpoint_interval,
            current_control_modules=current_control_modules,
            master_context=master_context,
            current_configs=current_configs,
        )
        
        #   Construct checkpoint_simulation_result from Master context as baseline
        # Prefer baseline_module_metrics (obtained by Master from checkpoint via run_policy_simulation)
        # This way baseline and policy simulation have identical Python-level starting point, metrics are comparable
        checkpoint_module_metrics = master_context.get("baseline_module_metrics") or master_context.get("module_metrics", {})
        if checkpoint_module_metrics:
            checkpoint_simulation_result = {
                "success": True,
                "stats": {},
                "module_metrics": checkpoint_module_metrics,
            }
            logger.info(f"Constructed baseline from Master checkpoint: {list(checkpoint_module_metrics.keys())}")
        else:
            checkpoint_simulation_result = None
            logger.warning(f"No module_metrics in master_context, baseline will be None")
        
        #   Get current control config from Master context (for initial_control_configs)
        from utils.checkpoint_logger import extract_configs_only
        initial_control_configs = extract_configs_only(current_configs) if current_configs else None
        
        #   Use Worker Actor's reset method
        messages = await worker_actor.reset.remote(
            initial_prompt=initial_prompt,
            context=context,
            initial_best_result=checkpoint_simulation_result,
            initial_control_configs=initial_control_configs,
            memory=None,
        )
        
        #   Lock to protect state writes
        with self._lock:
            self._worker_actors[instance_id] = worker_actor  # Store Worker Actor
            self._instance_dict[instance_id] = {
                # === Worker Actor ===
                "worker_actor": worker_actor,     # DeepCityWorker Actor (separate process)
                "messages": messages,             # Complete dialogue template (including system prompt)
                
                # === Turn Management ===
                "turn": 0,                        # Current turn
                "max_turns": self.max_turns,      # Max turns
                
                # === Reward Calculation ===
                "cumulative_reward": 0.0,         # Cumulative reward (sum of metric improvements) - deprecated, kept for compatibility
                "turn_rewards": {},               # Per-turn reward dict {turn_count: reward} (at FINISH, take reward for best_turn)
                "total_modules": len(current_control_modules),  # Total module count
                "improved_modules_set": set(),    # Track all modules that have been improved (for final module count reward calculation)
                "control_modules": current_control_modules,  # Control module list
                
                # === New reward function fields ===
                "improvement_count": 0,           # Number of times improvement occurred (incremented when improved_modules is non-empty)
                "cumulative_module_count_reward": 0.0,  # Cumulative module improvement reward (accumulate module ratio each improvement)
                
                # === Best result tracking (synced from FINISH action_result) ===
                "best_simulation_result": None,   # Best simulation result (synced from manager.best_simulation_result)
                "best_control_configs": None,     # Best control config (synced from manager.best_control_configs)
                "best_simulation_turn": None,     # Turn where best result occurred (synced from manager.best_simulation_turn)
                "initial_best_simulation_result": checkpoint_simulation_result,  #   Initial baseline = Master checkpoint metrics
                
                # === Code Execution ===
                "last_code_result": None,         # Last code execution result (synced from manager.last_code_result, used for DEBUG action)
                
                # === Context ===
                "context": context,               # Context dict (for manager.step())
                "is_validate": is_validate,       #   Whether it's validation mode
            }
        
        logger.info(f"Started DeepCity interaction: {instance_id} with {len(messages)} initial messages")
        return instance_id
    
    def get_messages(self, instance_id: str) -> list[dict[str, Any]]:
        """Return complete messages for current instance (including system prompt)"""
        with self._lock:  #   Lock for reading
            if instance_id not in self._instance_dict:
                raise ValueError(f"Instance {instance_id} not found")
            # Return from cached messages (updated after each step)
            return self._instance_dict[instance_id]["messages"]
    
    def get_instance_state_summary(self, instance_id: str) -> dict:
        """Return key state summary for instance (used to supplement log info and reward calculation for non-normal FINISH)"""
        with self._lock:
            if instance_id not in self._instance_dict:
                return {}
            state = self._instance_dict[instance_id]
            initial_baseline = state.get("initial_best_simulation_result")
            best_sim_res = state.get("best_simulation_result")
            control_modules = state.get("control_modules", [])
            improved_modules_set = state.get("improved_modules_set", set())
            total_modules = state.get("total_modules", 0)
            
            #   Re-calculate metric_improvement_reward (consistent with FINISH path logic: best vs baseline)
            baseline_module_metrics = initial_baseline.get("module_metrics", {}) if initial_baseline else {}
            best_module_metrics = best_sim_res.get("module_metrics", {}) if best_sim_res else {}
            
            module_rewards = {}
            for module_name in control_modules:
                best_metrics = best_module_metrics.get(module_name, {})
                base_metrics = baseline_module_metrics.get(module_name, {})
                module_reward = self._calculate_metric_improvement_reward(
                    module_name, best_metrics, base_metrics
                )
                module_rewards[module_name] = module_reward
            
            metric_improvement_reward = sum(module_rewards.values()) / len(module_rewards) if module_rewards else 0.0
            
            #   New reward function components
            improvement_count = state.get("improvement_count", 0)
            cumulative_module_count_reward = state.get("cumulative_module_count_reward", 0.0)
            
            # Legacy module count reward (for comparison)
            module_count_reward_old = float(len(improved_modules_set)) / float(total_modules) if total_modules > 0 else 0.0
            
            return {
                "total_modules": total_modules,
                "control_modules": control_modules,
                "improved_modules_count": len(improved_modules_set),
                "best_simulation_turn": state.get("best_simulation_turn"),
                "turn_count": state.get("turn", 0),
                "baseline_metrics": initial_baseline.get("module_metrics") if initial_baseline else None,
                "best_metrics": best_sim_res.get("module_metrics") if best_sim_res else None,
                "metric_improvement_reward": metric_improvement_reward,
                "module_rewards": module_rewards if module_rewards else None,
                "module_count_reward": module_count_reward_old,  # Legacy (for comparison)
                "improvement_count": improvement_count,  # New
                "cumulative_module_count_reward": cumulative_module_count_reward,  # New
            }

    async def generate_response(
        self, 
        instance_id: str, 
        llm_response: str,
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """
        Process LLM output, execute action, return environment feedback.
        
          Use LLMAgentManager.step() to process LLM response.
          Thread-safe: only do state read/write inside lock, do expensive computation outside lock.
        
        Args:
            instance_id: The interaction instance ID
            llm_response: LLM response text (assistant message content)
            
        Returns:
            tuple of (should_terminate, env_feedback, reward, extra_info)
        """
        self._ensure_imports()
        
        # 1. Lock and read state (Read)
        with self._lock:
            if instance_id not in self._instance_dict:
                raise ValueError(f"Instance {instance_id} not found")
            state = self._instance_dict[instance_id]
            
            # Extract needed objects, avoid holding lock for long
            worker_actor = state["worker_actor"]
            context = state["context"]
        
        # 2. Execute expensive operation (Process) -     Use Worker Actor in separate process
        # Worker Actor runs in separate process, avoiding GIL contention
        # Multiple Workers can truly parallelize policy simulation
        #   Use semaphore to control concurrency (if configured)
        if self._simulation_semaphore:
            async with self._simulation_semaphore:
                updated_messages, action_result = await worker_actor.step.remote(
                    llm_response=llm_response,
                    context=context,
                    verbose=True
                )
        else:
            # No concurrency limit
            updated_messages, action_result = await worker_actor.step.remote(
                llm_response=llm_response,
                context=context,
                verbose=True
            )
        
        # 3. Lock and update state (Write)
        with self._lock:
            # Re-check instance exists (prevent mid-deletion)
            if instance_id not in self._instance_dict:
                raise RuntimeError(f"Instance {instance_id} was removed during step")
            state = self._instance_dict[instance_id]
            
            #   Update state (from action_result)
            state["messages"] = updated_messages
            state["turn"] = action_result["turn_count"]
            
            #   Sync Worker Actor state to local cache
            state["best_simulation_result"] = action_result.get("best_simulation_result")
            state["best_simulation_turn"] = action_result.get("best_simulation_turn")
            state["best_control_configs"] = action_result.get("best_control_configs")
            state["memory"] = action_result.get("memory", [])
            state["last_code_result"] = action_result.get("last_code_result")
            
            #   Update current_configs in context
            current_control_configs = action_result.get("current_control_configs")
            if current_control_configs:
                state["context"]["current_configs"] = current_control_configs
            
            #   Calculate per-turn reward and accumulate (only metric improvement magnitude, module count reward at FINISH)
            action_type = action_result.get("action_type", "")
            action_data = action_result.get("action_result", {})
            total_modules = state["total_modules"]
            
            if action_type == "POLICY_PLANNING" or action_type == "DEBUG":
                improved_modules = action_data.get("improved_modules", [])
                
                # 1️⃣ Add improved modules to set (track which modules have been improved)
                if improved_modules:
                    state["improved_modules_set"].update(improved_modules)
                
                # 2️⃣ Calculate this turn's metric improvement reward relative to baseline
                sim_result = action_data.get("simulation_result", {})
                module_rewards = {}
                turn_metric_reward = 0.0
                
                #   Diagnostic log: track whether policy simulation succeeded
                is_val = state.get("is_validate", False)
                if is_val or not sim_result.get("success"):
                    logger.warning(
                        f"[DIAG] {action_type} instance={instance_id} is_validate={is_val} "
                        f"sim_success={sim_result.get('success')} sim_error={sim_result.get('error', 'N/A')} "
                        f"config_path={state['context'].get('config_path', 'N/A')} "
                        f"seed={state['context'].get('seed', 'N/A')} "
                        f"turn={state['turn']}"
                    )
                
                if sim_result.get("success") and total_modules > 0:
                    # Get current turn and initial baseline's module_metrics
                    current_module_metrics = sim_result.get("module_metrics", {})
                    #   Use initial baseline result to calculate reward (total improvement relative to baseline)
                    initial_baseline = action_data.get("initial_best_simulation_result")
                    baseline_module_metrics = initial_baseline.get("module_metrics", {}) if initial_baseline else {}
                    # Fallback: if no initial baseline, use best result before update (compatible with old version)
                    if not baseline_module_metrics:
                        best_sim_result = action_data.get("best_result_before_update")
                        baseline_module_metrics = best_sim_result.get("module_metrics", {}) if best_sim_result else {}
                    
                    # Calculate metric improvement ratio for all modules relative to baseline
                    control_modules = state.get("control_modules", [])
                    for module_name in control_modules:
                        current_metrics = current_module_metrics.get(module_name, {})
                        baseline_metrics = baseline_module_metrics.get(module_name, {})
                        
                        # Calculate this module's metric improvement ratio relative to initial baseline
                        module_reward = self._calculate_metric_improvement_reward(
                            module_name, current_metrics, baseline_metrics
                        )
                        module_rewards[module_name] = module_reward
                    
                    # This turn's metric improvement reward (average across all modules)
                    turn_metric_reward = sum(module_rewards.values()) / len(module_rewards) if module_rewards else 0.0
                    
                    # Save to dict (using turn_count as key)
                    state["turn_rewards"][state["turn"]] = turn_metric_reward
                    
                    # Compatibility: still update cumulative_reward (but no longer used for final calculation)
                    state["cumulative_reward"] += turn_metric_reward
                    
                    # Detailed logging
                    reward_details = ", ".join([f"{m}: {r:.3f}" for m, r in module_rewards.items()]) if module_rewards else "N/A"
                    logger.info(f"{action_type} metric rewards vs baseline: [{reward_details}], turn_metric_reward: {turn_metric_reward:.3f}")
                
                # 3️⃣ If this turn has module improvements, accumulate reward
                if improved_modules and total_modules > 0:
                    #   Increment count whenever any module improves
                    state["improvement_count"] += 1
                    
                    # Accumulate this turn's module improvement ratio
                    module_count_this_turn = len(improved_modules) / total_modules
                    state["cumulative_module_count_reward"] += module_count_this_turn
                    
                    logger.info(
                        f"{action_type} Turn {state['turn']}: has improved modules! "
                        f"improved_modules: {len(improved_modules)}/{total_modules}, "
                        f"module_count_this_turn: {module_count_this_turn:.3f}, "
                        f"improvement_count: {state['improvement_count']}, "
                        f"cumulative_module_count_reward: {state['cumulative_module_count_reward']:.3f}"
                    )
            
            elif action_type == "FINISH":
                state["best_simulation_result"] = action_data.get("best_simulation_result")
                state["best_simulation_turn"] = action_data.get("best_simulation_turn")
                state["best_control_configs"] = action_data.get("final_control_configs", {})
            
            # Prepare data needed for return value (read inside lock)
            cumulative_reward = state["cumulative_reward"]  # Kept for compatibility
            turn_rewards = state["turn_rewards"]  # Per-turn reward list
            improved_modules_set = state["improved_modules_set"]
            total_modules = state["total_modules"]
            final_configs = state["best_control_configs"]
            best_sim_res = state["best_simulation_result"]
            best_sim_turn = state["best_simulation_turn"]
            current_turn = state["turn"]
        
        # 4. Post-process - outside lock
        env_feedback = ""
        if updated_messages and updated_messages[-1].get("role") == "user":
            env_feedback = updated_messages[-1]["content"]
        
        should_terminate = action_result["finished"]
        
        reward = 0.0
        extra_info = {}
        
        if should_terminate:
            #   New reward function: (metric_improvement + judge_score) × improvement_count + cumulative_module_count_reward
            
            # 1Calculate metric improvement reward: best_simulation_result vs initial baseline
            initial_baseline = state.get("initial_best_simulation_result")
            baseline_module_metrics = initial_baseline.get("module_metrics", {}) if initial_baseline else {}
            best_module_metrics = best_sim_res.get("module_metrics", {}) if best_sim_res else {}
            control_modules = state.get("control_modules", [])
            
            module_rewards = {}
            for module_name in control_modules:
                best_metrics = best_module_metrics.get(module_name, {})
                base_metrics = baseline_module_metrics.get(module_name, {})
                module_reward = self._calculate_metric_improvement_reward(
                    module_name, best_metrics, base_metrics
                )
                module_rewards[module_name] = module_reward
            
            metric_improvement_reward = sum(module_rewards.values()) / len(module_rewards) if module_rewards else 0.0
            
            # 2 Get new reward function components
            improvement_count = state.get("improvement_count", 0)
            cumulative_module_count_reward = state.get("cumulative_module_count_reward", 0.0)
            
            # 3 Calculate reward without judge_score (judge_score is added by agent_loop after calling Judge LLM)
            # agent_loop will recalculate with full formula: (metric_improvement + judge_score) × improvement_count + cumulative_module_count_reward
            reward = metric_improvement_reward * improvement_count + cumulative_module_count_reward
            
            # Legacy module count reward (for comparison)
            module_count_reward_old = float(len(improved_modules_set)) / float(total_modules) if total_modules > 0 else 0.0
            
            reward_details = ", ".join([f"{m}: {r:.3f}" for m, r in module_rewards.items()]) if module_rewards else "N/A"
            logger.info(
                f"FINISH: improved {len(improved_modules_set)}/{total_modules} unique module(s): {sorted(improved_modules_set)}, "
                f"best_turn: {best_sim_turn}, policy_turns: {len(turn_rewards)}, "
                f"improvement_count: {improvement_count}, "
                f"metric_improvement_reward (best vs baseline): {metric_improvement_reward:.3f} [{reward_details}], "
                f"cumulative_module_count_reward: {cumulative_module_count_reward:.3f}, "
                f"reward_without_judge: {metric_improvement_reward:.3f} × {improvement_count} + {cumulative_module_count_reward:.3f} = {reward:.3f} "
                f"(agent_loop will add judge_score and recompute final reward)"
            )
            
            extra_info = {
                "success": True,
                "cumulative_reward": reward,  # Final reward from new reward function (without judge_score, overridden after agent_loop adds it)
                "improvement_count": improvement_count,  # Number of improvements
                "cumulative_module_count_reward": cumulative_module_count_reward,  # Cumulative module improvement reward
                "metric_improvement_reward": metric_improvement_reward,  # Metric improvement reward (best vs baseline)
                "module_count_reward": module_count_reward_old,  # Legacy module count reward (for comparison)
                "module_rewards": module_rewards,  # Per-module metric improvement reward details
                "turn_rewards": turn_rewards,  # Per-turn reward dict {turn: reward} (kept for analysis)
                "improved_modules_count": len(improved_modules_set),
                "total_modules": total_modules,
                "control_modules": control_modules,  #   Control module list (for Judge LLM use)
                "final_control_configs": final_configs or {},
                "best_simulation_result": best_sim_res,
                "best_simulation_turn": best_sim_turn,
                "turn_count": current_turn,
                "initial_best_simulation_result": initial_baseline,  #   Baseline metrics
            }
        else:
            # Calculate current best turn_reward (for intermediate turn monitoring)
            current_best_reward = max(turn_rewards.values()) if turn_rewards else 0.0
            extra_info = {
                "cumulative_reward": cumulative_reward,  # Kept for compatibility
                "metric_improvement_reward_best": current_best_reward,  # Best turn's reward so far
                "turn_rewards": turn_rewards,  # Per-turn reward dict {turn: reward}
                "improved_modules_count": len(improved_modules_set),
                "total_modules": total_modules
            }
        
        return should_terminate, env_feedback, reward, extra_info

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Clean up interaction resources and report best policy to Master.
        
          Report Agent's best policy to Master.
          Master collects 8 workers' best policies, randomly selects 1 to apply, all 8 used for GRPO training.
          Thread-safe: use atomic pop to remove state.
        """
        # 1. Lock and remove state (Atomic Pop)
        state = None
        worker_actor = None
        with self._lock:
            if instance_id in self._instance_dict:
                state = self._instance_dict.pop(instance_id)  #   Use pop to atomically remove
                worker_actor = self._worker_actors.pop(instance_id, None)  #   Remove Worker Actor reference
        
        if state and worker_actor:
            #   Read is_validate flag
            is_validate = state.get("is_validate", False)
            mode = "val" if is_validate else "train"
            
            # 2. Get final state from Worker Actor
            try:
                worker_state = await worker_actor.get_state.remote()
                best_configs = worker_state.get("best_control_configs")
                best_turn = worker_state.get("best_simulation_turn")
                memory = worker_state.get("memory", [])
            except Exception as e:
                logger.error(f"Failed to get worker state: {e}")
                best_configs = None
                best_turn = None
                memory = []
            
            #   Explicitly destroy Worker Actor to avoid process accumulation
            try:
                import ray
                ray.kill(worker_actor)
                logger.debug(f"Worker Actor {instance_id} killed")
            except Exception as e:
                logger.warning(f"Failed to kill Worker Actor {instance_id}: {e}")
            
            # Report best policy and memory to Master
            if best_configs:
                master = await self._get_or_create_master(instance_id, is_validate=is_validate)
                master_id = self._get_val_master_id(instance_id) if is_validate else self._get_master_id(instance_id)
                if master:
                    try:
                        await master.report_best_policy.remote(best_configs, memory, instance_id, is_validate=is_validate)
                        logger.info(f"[{mode}] Reported best policy and memory ({len(memory)} items) to Master_{master_id}")
                        
                        is_last = await master.unregister_worker.remote(instance_id, is_validate=is_validate)
                        if is_last:
                            logger.info(f"[{mode}] Last worker completed for Master_{master_id}, applying best policies...")
                            await master.apply_best_policies.remote(is_validate=is_validate)
                            logger.info(f"[{mode}] Best policies applied, checkpoint flag cleared for next batch")
                    except Exception as e:
                        logger.warning(f"[{mode}] Failed to report best policy or unregister worker: {e}")
            else:
                master = await self._get_or_create_master(instance_id, is_validate=is_validate)
                master_id = self._get_val_master_id(instance_id) if is_validate else self._get_master_id(instance_id)
                if master:
                    try:
                        master_context = await master.get_context.remote(is_validate=is_validate)
                        current_control_modules = master_context.get("control_modules", [])
                        
                        default_configs = await self._get_default_configs(master, current_control_modules, is_validate=is_validate)
                        if default_configs:
                            configs_only = {name: info['config'] for name, info in default_configs.items()}
                            await master.report_best_policy.remote(configs_only, memory, instance_id, is_validate=is_validate)
                            logger.info(f"[{mode}] Reported default config and memory ({len(memory)} items) to Master_{master_id}")
                        
                        is_last = await master.unregister_worker.remote(instance_id, is_validate=is_validate)
                        if is_last:
                            logger.info(f"[{mode}] Last worker completed for Master_{master_id}, applying best policies...")
                            await master.apply_best_policies.remote(is_validate=is_validate)
                            logger.info(f"[{mode}] Best policies applied, checkpoint flag cleared for next batch")
                    except Exception as e:
                        logger.warning(f"[{mode}] Failed to report default config or unregister worker: {e}")
            
            # Log statistics
            logger.info(f"Finalizing DeepCity interaction: {instance_id}, best_turn: {best_turn}")
