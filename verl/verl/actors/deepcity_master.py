# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 DeepCity Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepCityMaster Ray Actor for managing the main simulation timeline.

This actor runs the main SUMO simulation, periodically saves checkpoints,
and provides checkpoint paths to DeepCityInteraction instances for policy evaluation.
"""

import os
import sys
import ray
import time
import copy
import logging
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add DeepCity project root to path
DEEPCITY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if DEEPCITY_ROOT not in sys.path:
    sys.path.insert(0, DEEPCITY_ROOT)

logger = logging.getLogger(__name__)


@ray.remote
class DeepCityMaster:
    """
    Ray Actor that manages the main SUMO simulation timeline.
    
    Responsibilities:
    1. Run the main simulation (SUMOEnv)
    2. Periodically save checkpoints
    3. Provide checkpoint paths to workers
    4. Aggregate best policies from workers
    5. Apply best policies to main simulation
    """
    
    def __init__(
        self,
        master_id: int,
        environments: List[Dict[str, Any]],
        checkpoint_interval: int = 1800,
        checkpoint_dir: Optional[str] = None,
        use_gui: bool = False,
        seed: Optional[int] = None,
        seed_increment: int = 1000,
        simulation_duration: Optional[int] = None,
        auto_restart: bool = True,
        max_episodes: Optional[int] = None,
        use_http: bool = False,
        sumo_server_url: Optional[str] = None,
        http_timeout: float = 300.0,
        num_masters: int = 4,
        num_val_masters: int = None,
        num_train_masters: int = None,
        val_environments: List[Dict[str, Any]] = None,
        val_simulation_duration: int = None,
    ):
        """
        Initialize DeepCityMaster.
        
        Multi-Master Architecture: When batch=4, start DeepCityMaster_0, DeepCityMaster_1, DeepCityMaster_2, DeepCityMaster_3
        Each Master maintains an independent simulation timeline.
        
          Batch Parallel Environment Allocation Mechanism:
        - environments: List of environment configs, each containing {sumo_config_path, control_modules}
        - In each episode, 4 Masters use 4 different environments simultaneously
        - After restart, select the next batch of 4 environments, cycle through the list
        - Example (14 environments, 4 Masters):
            Round 1: Master_0→env[0], Master_1→env[1], Master_2→env[2], Master_3→env[3]
            Round 2: Master_0→env[4], Master_1→env[5], Master_2→env[6], Master_3→env[7]
            Round 3: Master_0→env[8], Master_1→env[9], Master_2→env[10], Master_3→env[11]
            Round 4: Master_0→env[12], Master_1→env[13], Master_2→env[0], Master_3→env[1]
        
          Universal Seed Generation Mechanism:
        - seed = base_seed + master_id + (episode_count * seed_increment)
        - Environment index calculation: batch parallel allocation (see example above)
        - seed always increments to ensure data diversity
        
        Example (base_seed=42, seed_increment=1000, Master_0):
            Episode 0: seed=42
            Episode 1: seed=1042
            Episode 2: seed=2042
            Episode 3: seed=3042
        
        Args:
            master_id: Unique identifier for Master (0, 1, 2, ...)
            environments: Environment config list, format:
                [
                    {"sumo_config_path": "...", "control_modules": [...]},
                    {"sumo_config_path": "...", "control_modules": [...]},
                    ...
                ]
            checkpoint_interval: Simulation time interval between checkpoints (seconds)
            checkpoint_dir: Directory to save checkpoints
            use_gui: Whether to use SUMO GUI
            seed: Base random seed (will add master_id to distinguish different Masters)
            seed_increment: Seed increment on each restart (default 1000)
            simulation_duration: Simulation duration (seconds), auto-restart when reached (e.g., 86400s = 24 hours)
            auto_restart: Whether to auto-restart (switches to next batch of environments on restart)
            max_episodes: Maximum episode count (restart limit), None means unlimited restarts
            use_http: Whether to use HTTP mode to call remote SUMO server
            sumo_server_url: SUMO server URL (required in HTTP mode)
            http_timeout: HTTP request timeout (seconds)
            num_masters: Total number of Masters (for batch environment allocation, default 4)
            num_val_masters: Total number of Val Masters (for val environment allocation, default same as num_masters)
            val_environments: Val-specific environment config list (None to reuse train environments)
            val_simulation_duration: Val simulation duration (seconds), resets from 0:00 after completion (None uses train's simulation_duration)
        """
        self.master_id = master_id
        self.num_masters = num_masters  # Total number of Masters (for batch environment allocation)
        self.num_val_masters = num_val_masters if num_val_masters is not None else num_masters
        self.num_train_masters = num_train_masters if num_train_masters is not None else num_masters
        self._is_val_only = (master_id >= self.num_train_masters)  # val-only Master does not start train env
        self.environments = environments  # Train environment config list
        self.val_environments = val_environments  # Val-specific environment config list (None to reuse train)
        self.val_simulation_duration = val_simulation_duration  # Val-specific simulation duration (None uses train's)
        self.checkpoint_interval = checkpoint_interval
        self.use_gui = use_gui
        
        #   HTTP mode support
        self.use_http = use_http
        self.sumo_server_url = sumo_server_url
        self.http_timeout = http_timeout
        self.http_client = None
        self.http_master_id = None  # Master ID on HTTP server
        
        if self.use_http:
            if not self.sumo_server_url:
                raise ValueError("sumo_server_url is required when use_http=True")
            # Delayed HTTP client initialization (created in start())
            logger.info(f"Master_{master_id}: HTTP mode enabled, server={sumo_server_url}")
        
        #   Universal seed generation mechanism
        self.base_seed = seed or 0
        self.seed_increment = seed_increment  # Seed increment on each restart
        self.episode_count = 0  # Current episode count (starts from 0)
        
        # Initial seed = base_seed + master_id
        self.seed = self.base_seed + master_id
        #   val fixed seed: calculated once at initialization, never changes afterwards
        # Ensures validation sees the same traffic scenarios across different training stages for fair policy comparison
        self.val_seed = self.base_seed + master_id + 10000
        
        self.simulation_duration = simulation_duration  # Simulation duration (e.g., 86400s)
        self.auto_restart = auto_restart  # Whether to auto-restart
        self.max_episodes = max_episodes  # Maximum episode count
        
        #   Environment run count tracking (for environment selection on abnormal restart)
        self.env_run_counts = {i: 0 for i in range(len(self.environments))}  # Successful run count for each environment
        
        #   Initial environment selection: random selection (all environments have 0 run counts)
        import random
        self.current_env_index = random.randint(0, len(self.environments) - 1)
        logger.info(f"Master_{master_id}: Initial environment randomly selected: [{self.current_env_index}]")
        
        # Extract config_path and control_modules from current environment config
        current_env = self.environments[self.current_env_index]
        self.config_path = current_env["sumo_config_path"]
        self.control_modules = current_env.get("control_modules", [])
        self.current_env_episodes = current_env.get("episodes", 1)  # Number of times to run this environment
        # Set up checkpoint directory (each Master has its own subdirectory)
        # val-only Master does not create train directory, val directory is lazily created by _ensure_val_env_ready
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir) / f"master_{master_id}"
        else:
            self.checkpoint_dir = Path(DEEPCITY_ROOT) / "checkpoints" / f"master_{master_id}"
        if not self._is_val_only:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.env = None
        self.current_time = 0
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_policies: Dict[str, List[Dict]] = {}  # module -> list of configs from workers
        self.running = False
        self._starting = False  #   New: indicates initialization in progress, prevents race conditions
        # [Modified] Use RLock (Re-entrant Lock), allows same thread to acquire lock multiple times
        # This prevents deadlock when prepare_batch_and_get_context calls step, and step acquires lock again internally
        self._lock = threading.RLock()
        
        # Batch-level checkpoint management
        self.checkpoint_in_use = False  # Whether checkpoint for current batch has been generated
        self.active_worker_ids = set()  # Set of worker IDs currently using checkpoint
        
        #   Maintain _previous_checkpoint_path (consistent with run_joint_control.py)
        # Used to pass to run_controlled_simulation, ensuring t-1 checkpoint is correctly saved
        self._previous_checkpoint_path = None      # train environment
        self._val_previous_checkpoint_path = None   # val environment
        
        #   val-specific state (isolated from train, each with independent timeline)
        self.val_env = None
        self.val_current_time = 0
        self.val_checkpoints: List[Dict[str, Any]] = []
        self.val_checkpoint_in_use = False
        self.val_active_worker_ids = set()
        self.val_best_policies: Dict[str, List[Dict]] = {}
        self.val_running = False
        self.val_traffic_states_filepath: Optional[str] = None
        self.val_traffic_state_collector = None
        self.val_checkpoint_dir = None
        self.val_episode_count = 0
        self._val_config_path = None          # config_path currently used by val
        self._val_control_modules = None      # control_modules currently used by val
        self._val_current_env_index = 0       # current environment index for val
        # val environment run count tracking
        val_env_pool = self.val_environments if self.val_environments else self.environments
        self._val_env_run_counts = {i: 0 for i in range(len(val_env_pool))}
        
        # Traffic states collection (for DATA_ANALYSIS)
        self.traffic_states_filepath: Optional[str] = None
        self.traffic_state_collector = None
        
        # Context for code execution
        self.context: Dict[str, Any] = {
            "master_id": master_id,
            "config_path": self.config_path,  #   Use self.config_path (extracted from current environment)
            "checkpoint_interval": checkpoint_interval,
            "seed": self.seed,
            "memory": [],  #   Initialize memory as empty list, accumulates within 86400s
        }
        
        logger.info(
            f"DeepCityMaster_{master_id} initialized\n"
            f"  - Environment: [{self.current_env_index}/{len(self.environments)}] {self.config_path}\n"
            f"  - Modules: {self.control_modules}\n"
            f"  - Episodes: {self.current_env_episodes}\n"
            f"  - Seed: {self.seed}"
        )
    
    def _get_current_env_index(self, episode_count: int, num_masters: int = 4) -> int:
        """
        Batch parallel environment allocation: Each Master uses a different environment in the same round.
        
          Batch parallel mechanism (assuming num_masters=4, 14 environments):
            Round 1 (episode=0):
                Master_0 → env[0]
                Master_1 → env[1]
                Master_2 → env[2]
                Master_3 → env[3]
            
            Round 2 (episode=1):
                Master_0 → env[4]
                Master_1 → env[5]
                Master_2 → env[6]
                Master_3 → env[7]
            
            Round 3 (episode=2):
                Master_0 → env[8]
                Master_1 → env[9]
                Master_2 → env[10]
                Master_3 → env[11]
            
            Round 4 (episode=3):
                Master_0 → env[12]
                Master_1 → env[13]
                Master_2 → env[0]  # Cycle
                Master_3 → env[1]
        
        Args:
            episode_count: Current episode count (restart count)
            num_masters: Total number of Masters (default 4)
            
        Returns:
            Environment index that should be used currently
        """
        total_envs = len(self.environments)
        
        # Calculate global environment index: episode_count * num_masters + master_id
        # Then modulo total environment count to implement cycling
        global_env_index = (episode_count * num_masters + self.master_id) % total_envs
        
        return global_env_index
    
    def start(self) -> Dict[str, Any]:
        """
        Start the main simulation.
          Fix: Add idempotency check to prevent multiple Workers from triggering start simultaneously causing multiple initializations.
          Fix: Protect the entire method with a lock to prevent race conditions causing duplicate creation of traffic_states files.
          Fix: Use _starting flag to prevent race conditions, ensuring running=True is only set after env initialization completes.
        
        Returns:
            Initial status dict
        """
        #   Protect state check with lock
        with self._lock:
            # === Idempotency check 1: Already running ===
            if self.running and (self.env is not None or self._is_val_only):
                logger.warning(f"DeepCityMaster_{self.master_id} is ALREADY running. Ignoring duplicate start request.")
                return {
                    "success": True,
                    "current_time": self.current_time,
                    "checkpoint_count": len(self.checkpoints),
                    "traffic_states_filepath": self.traffic_states_filepath,
                    "note": "already_running"
                }
            
            # === Idempotency check 2: Currently initializing (other Worker is executing start) ===
            if self._starting:
                logger.warning(f"DeepCityMaster_{self.master_id} is STARTING. Waiting for completion...")
                # Wait for initialization to complete (simple spin wait)
                import time
                max_wait = 120  # Wait up to 120 seconds
                waited = 0
                while self._starting and waited < max_wait:
                    self._lock.release()  # Release lock to let other Worker complete
                    time.sleep(1)
                    waited += 1
                    self._lock.acquire()  # Reacquire lock to check status
                
                # Check if initialization succeeded
                if self.running and (self.env is not None or self._is_val_only):
                    return {
                        "success": True,
                        "current_time": self.current_time,
                        "checkpoint_count": len(self.checkpoints),
                        "traffic_states_filepath": self.traffic_states_filepath,
                        "note": "waited_for_start"
                    }
                else:
                    # Initialization failed, retry initialization
                    logger.warning(f"DeepCityMaster_{self.master_id} start failed, retrying...")
            
            #   Mark as initializing
            self._starting = True
        
        # The following initialization operations are executed outside the lock (time-consuming operations)
        try:
            import sys
            sys.path.insert(0, str(DEEPCITY_ROOT))
            
            #   Val-only Master (master_id >= num_train_masters): Skip train SUMO environment creation
            # train will never route to these Masters, no need to start train env, save resources
            # val env is lazily loaded by _ensure_val_env_ready()
            if self._is_val_only:
                logger.info(f"Master_{self.master_id}: Val-only master (id={self.master_id} >= num_train_masters={self.num_train_masters}), "
                           f"skipping train env creation. Val env will be lazily initialized.")
                self.env = None
                self.traffic_state_collector = None
                self.traffic_states_filepath = None
                self.current_time = 0
                
                with self._lock:
                    self.running = True
                    self._starting = False
                
                return {
                    "success": True,
                    "current_time": 0,
                    "checkpoint_count": 0,
                    "traffic_states_filepath": None,
                    "note": "val_only_master"
                }
            
            #   HTTP mode: Create remote Master via HTTP client
            if self.use_http:
                from verl.utils.deepcity_http_client import DeepCityHttpClient
                
                self.http_client = DeepCityHttpClient(
                    base_url=self.sumo_server_url,
                    timeout=self.http_timeout
                )
                
                # Create remote Master
                self.http_master_id = f"master_{self.master_id}"
                logger.info(f"Master_{self.master_id}: Creating HTTP Master on server {self.sumo_server_url}")
                
                #   Pass complete environments list to allow server-side environment switching on restart
                logger.info(f"Master_{self.master_id}: Using environment {self.current_env_index}/{len(self.environments)}: "
                           f"{self.environments[self.current_env_index].get('sumo_config_path')}")
                
                result = self.http_client.create_session(
                    session_id=self.http_master_id,
                    environments=self.environments,  #   Pass complete list
                    current_env_index=self.current_env_index,  #   Specify current environment index in use
                    checkpoint_interval=self.checkpoint_interval,
                    run_duration=self.simulation_duration or 86400,
                    seed=self.seed,
                    use_gui=self.use_gui,
                )
                
                logger.info(f"Master_{self.master_id}: HTTP Master creation request sent, result: {result}")
                
                #   Get server-side context (contains road network graph structure)
                # Add retry logic because session initialization may take time
                import time
                max_retries = 30  # Increased to 30
                retry_delay = 5  # Increased to 5 seconds
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Master_{self.master_id}: Attempting to get context (attempt {attempt + 1}/{max_retries})")
                        server_context = self.http_client.get_session(session_id=self.http_master_id)
                        
                        # Check if context contains necessary fields (ensure initialization complete)
                        if server_context and "control_modules" in server_context:
                            self.context.update(server_context)
                            logger.info(f"Master_{self.master_id}: Retrieved context from server, keys: {list(server_context.keys())}")
                            logger.info(f"Master_{self.master_id}: control_modules: {server_context.get('control_modules')}")
                            break
                        else:
                            # Context incomplete, continue retrying
                            if attempt < max_retries - 1:
                                logger.warning(f"Master_{self.master_id}: Context incomplete (attempt {attempt + 1}), retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                            else:
                                logger.error(f"Master_{self.master_id}: Context still incomplete after {max_retries} attempts")
                                raise RuntimeError("Failed to get complete context from server")
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Master_{self.master_id}: Failed to get context (attempt {attempt + 1}): {e}, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Master_{self.master_id}: Failed to get context after {max_retries} attempts: {e}")
                            raise
                
                self.env = None  # No local env needed in HTTP mode
            
            #   Local mode: Create local SUMO environment
            else:
                from environment.sumo_env import SUMOEnv
                from utils.traffic_state_collector import TrafficStateCollector
                from utils.simulation_utils import create_sumo_env
                
                # Use create_sumo_env to properly initialize the environment
                # Enable simulated taxi system when taxi_scheduling module is active
                # (matches run_taxi_scheduling.py and run_taxi_scheduling_baseline.py)
                _use_sim_taxi = 'taxi_scheduling' in (self.control_modules or [])
                self.env, _ = create_sumo_env(
                    config_path=self.config_path,
                    use_gui=self.use_gui,
                    seed=self.seed,
                    control_modules=self.control_modules,
                    use_simulated_taxi_system=_use_sim_taxi,
                )
            # self.running = True  #   Already set in lock, no need to set again
            self.current_time = 0
            
            #   HTTP mode: TrafficStateCollector is managed by server side
            if self.use_http:
                self.traffic_state_collector = None
                # Get traffic_states_filepath from HTTP server
                try:
                    session_info = self.http_client.get_session(self.http_master_id)
                    self.traffic_states_filepath = session_info.get("traffic_states_filepath")
                    logger.info(f"Master_{self.master_id}: HTTP mode, traffic_states_filepath from server: {self.traffic_states_filepath}")
                except Exception as e:
                    logger.warning(f"Master_{self.master_id}: Failed to get traffic_states_filepath from server: {e}")
                    self.traffic_states_filepath = None
            
            #   Local mode: Initialize TrafficStateCollector
            else:
                # Initialize TrafficStateCollector for DATA_ANALYSIS
                # Use init_traffic_states_file to generate file with proper file_prefix
                # This allows Worker's read_subway_states() to find the file via file_prefix matching
                from utils.traffic_state_collector import init_traffic_states_file
                
                # Extract config_name from config_path
                config_name = None
                if self.config_path:
                    config_name = Path(self.config_path).parent.name
                
                # Generate traffic_states file with proper naming convention
                # IMPORTANT: simulation_id should NOT include instance_id, only master_id
                # This way all Workers under the same Master can find the same file
                traffic_states_filepath = init_traffic_states_file(
                    simulation_id=f"master_{self.master_id}",  # Only master_id, no instance_id
                    config_name=config_name,
                    llm_name=None,  # Master doesn't use LLM
                    control_modules=self.control_modules
                )
                
                # Get road network graphs from env
                graphs = self.env.get_road_network_graphs()
                
                self.traffic_state_collector = TrafficStateCollector(
                    env=self.env,
                    traffic_states_filepath=traffic_states_filepath,
                    interval=60.0,  # Collect state every 60 seconds
                    lane_dict=graphs.get("lane_dict"),
                    lane_inter_graph=graphs.get("lane_inter_graph"),
                    simulation_id=f"master_{self.master_id}",  # Same as above
                )
                self.traffic_states_filepath = str(traffic_states_filepath)
            
            #   Initialization successful, set running=True, clear _starting flag
            with self._lock:
                self.running = True
                self._starting = False
            
            # Note: Initial checkpoint will be saved by first run_controlled_simulation() call
            
            logger.info(f"DeepCityMaster started successfully with traffic_states: {self.traffic_states_filepath}")
            return {
                "success": True,
                "current_time": self.current_time,
                "checkpoint_count": len(self.checkpoints),
                "traffic_states_filepath": self.traffic_states_filepath,
            }
            
        except Exception as e:
            logger.error(f"Failed to start DeepCityMaster: {e}")
            #   Initialization failed, reset all states, allow retry
            with self._lock:
                self.running = False
                self._starting = False
                self.env = None
            return {"success": False, "error": str(e)}
    
    def _ensure_val_env_ready(self):
        """
        Lazy load val_env: Create independent SUMO simulation environment on first val request.
        val_env is completely isolated from train env, each maintains independent timeline.
        
          Support independent val environment configuration:
        - If val_environments is configured, Master_i is fixed to val_environments[i % len]
        - If not configured, reuse train's current environment configuration (config_path + control_modules)
        """
        if self.val_running and self.val_env is not None:
            return  # Already ready
        
        logger.info(f"Master_{self.master_id}: Initializing val_env (lazy load)...")
        
        from utils.simulation_utils import create_sumo_env
        from utils.traffic_state_collector import TrafficStateCollector, init_traffic_states_file
        
        #   Determine environment configuration for val
        if self.val_environments:
            #   Fixed allocation: Master_i binds to val_environments[i % len(val_environments)]
            # This ensures each validation runs the same fixed scenario for easy performance comparison
            val_env_index = self.master_id % len(self.val_environments)
            
            val_env_config = self.val_environments[val_env_index]
            val_config_path = val_env_config["sumo_config_path"]
            val_control_modules = val_env_config.get("control_modules", [])
            self._val_current_env_index = val_env_index
            self._val_env_run_counts[val_env_index] = self._val_env_run_counts.get(val_env_index, 0) + 1
            
            logger.info(
                f"Master_{self.master_id}: Val fixed environment [{val_env_index}/{len(self.val_environments)}]: "
                f"{val_config_path}, modules={val_control_modules}"
            )
        else:
            # Reuse train's current environment configuration
            val_config_path = self.config_path
            val_control_modules = self.control_modules
            logger.info(f"Master_{self.master_id}: Val reusing train environment: {val_config_path}")
        
        # Store environment information currently used by val (for context and restart)
        self._val_config_path = val_config_path
        self._val_control_modules = val_control_modules
        
        _use_sim_taxi = 'taxi_scheduling' in (val_control_modules or [])
        self.val_env, _ = create_sumo_env(
            config_path=val_config_path,
            use_gui=False,  # val doesn't need GUI
            seed=self.val_seed,  #   Use fixed val_seed (calculated at initialization, never changes)
            control_modules=val_control_modules,
            use_simulated_taxi_system=_use_sim_taxi,
        )
        
        self.val_current_time = 0
        
        # Val-specific checkpoint directory (same level as train's master_{id}, named master_{id}_val)
        self.val_checkpoint_dir = self.checkpoint_dir.parent / f"master_{self.master_id}_val"
        self.val_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Val-specific TrafficStateCollector
        config_name = Path(val_config_path).parent.name if val_config_path else None
        val_traffic_states_filepath = init_traffic_states_file(
            simulation_id=f"master_{self.master_id}_val",
            config_name=config_name,
            llm_name=None,
            control_modules=val_control_modules,
        )
        
        graphs = self.val_env.get_road_network_graphs()
        self.val_traffic_state_collector = TrafficStateCollector(
            env=self.val_env,
            traffic_states_filepath=val_traffic_states_filepath,
            interval=60.0,
            lane_dict=graphs.get("lane_dict"),
            lane_inter_graph=graphs.get("lane_inter_graph"),
            simulation_id=f"master_{self.master_id}_val",
        )
        self.val_traffic_states_filepath = str(val_traffic_states_filepath)
        
        self.val_running = True
        logger.info(f"Master_{self.master_id}: val_env initialized successfully (config={val_config_path}, modules={val_control_modules})")
    
    def is_checkpoint_ready(self) -> bool:
        """Check if checkpoint for current batch has been generated."""
        with self._lock:
            return self.checkpoint_in_use
    
    def register_worker(self, worker_id: str) -> None:
        """
        Register a worker to start using checkpoint.
        The first worker will trigger step(), subsequent workers reuse checkpoint.
        
        Args:
            worker_id: Worker's unique identifier (instance_id)
        """
        with self._lock:
            if worker_id in self.active_worker_ids:
                logger.warning(f"Master_{self.master_id}: Worker {worker_id} already registered, skipping")
                return
            
            self.active_worker_ids.add(worker_id)
            logger.debug(f"Master_{self.master_id}: Worker {worker_id} registered, active_workers={len(self.active_worker_ids)}")
    
    def unregister_worker(self, worker_id: str, is_validate: bool = False) -> bool:
        """
        Unregister a worker (finished using checkpoint).
        
        Args:
            worker_id: Worker's unique identifier (instance_id)
            is_validate: Whether it's a validation mode worker
        
        Returns:
            bool: Return True if it's the last worker
        """
        mode = "val" if is_validate else "train"
        with self._lock:
            if is_validate:
                active_ids = self.val_active_worker_ids
            else:
                active_ids = self.active_worker_ids
            
            if worker_id not in active_ids:
                logger.warning(f"Master_{self.master_id}[{mode}]: Worker {worker_id} not registered, skipping")
                return False
            
            active_ids.discard(worker_id)
            logger.debug(f"Master_{self.master_id}[{mode}]: Worker {worker_id} unregistered, active_workers={len(active_ids)}")
            
            if len(active_ids) == 0:
                if is_validate:
                    self.val_checkpoint_in_use = False
                else:
                    self.checkpoint_in_use = False
                logger.info(f"Master_{self.master_id}[{mode}]: All workers completed, checkpoint flag cleared")
                return True
            return False
    
    def mark_checkpoint_ready(self) -> None:
        """Mark checkpoint as ready, subsequent workers can reuse."""
        with self._lock:
            self.checkpoint_in_use = True
            logger.info(f"Master_{self.master_id}: Checkpoint marked as ready for reuse")
    
    def prepare_batch_and_get_context(self, worker_id: str, is_validate: bool = False) -> Dict[str, Any]:
        """
        [Atomic operation] Prepare environment context for Worker.
        If it's the first Worker of the batch -> Trigger Step, generate Checkpoint, mark Ready.
        If it's a subsequent Worker -> Return existing Context directly.
        
        This method solves the Check-Then-Act race condition in start_interaction.
        Use RLock to ensure thread safety, Worker B will queue outside waiting for Worker A to complete step.
        
        Args:
            worker_id: Worker's unique identifier (instance_id)
            is_validate: Whether it's a validation request (True uses val_env, doesn't affect train timeline)
            
        Returns:
            Dict[str, Any]: Environment context, including checkpoint_path, traffic_states, etc.
        """
        mode = "val" if is_validate else "train"
        
        # val mode: Ensure val_env is initialized
        if is_validate:
            self._ensure_val_env_ready()
        
        # Select corresponding state variables
        if is_validate:
            active_worker_ids = self.val_active_worker_ids
            checkpoint_in_use_attr = "val_checkpoint_in_use"
            current_time_attr = "val_current_time"
        else:
            active_worker_ids = self.active_worker_ids
            checkpoint_in_use_attr = "checkpoint_in_use"
            current_time_attr = "current_time"
        
        with self._lock:
            # 1. Register Worker
            if worker_id not in active_worker_ids:
                active_worker_ids.add(worker_id)
                logger.debug(f"Master_{self.master_id}[{mode}]: Worker {worker_id} registered")

            # 2. Check if current batch is already ready
            if not getattr(self, checkpoint_in_use_attr):
                logger.info(f"Master_{self.master_id}[{mode}]: New batch started by {worker_id}, stepping simulation...")
                
                step_result = self.step(self.checkpoint_interval, is_validate=is_validate)
                
                if not step_result.get("success"):
                    raise RuntimeError(f"Master[{mode}] step failed: {step_result.get('error')}")
                
                #   Fix: If abnormal restart occurred, active_worker_ids was cleared, need to re-register current Worker
                if step_result.get("abnormal_restart"):
                    if worker_id not in active_worker_ids:
                        active_worker_ids.add(worker_id)
                        logger.info(f"Master_{self.master_id}[{mode}]: Re-registered Worker {worker_id} after abnormal restart")
                
                current_time = getattr(self, current_time_attr)
                logger.info(f"Master_{self.master_id}[{mode}]: Batch ready at time {current_time}s")
                
                # Mark checkpoint as ready
                setattr(self, checkpoint_in_use_attr, True)
            else:
                current_time = getattr(self, current_time_attr)
                logger.info(f"Master_{self.master_id}[{mode}]: Worker {worker_id} joining existing batch at {current_time}s")

            # 3. Return Context (context is definitely up-to-date at this point)
            return self.get_context(is_validate=is_validate)
    
    def step(self, duration: int = None, _retry_count: int = 0, is_validate: bool = False) -> Dict[str, Any]:
        """
        Advance the simulation by calling run_controlled_simulation.
        
        Support train/val dual mode: is_validate=True operates on val_env, doesn't affect train timeline.
        
        Args:
            duration: Time to advance (uses checkpoint_interval if None)
            _retry_count: Internal parameter, used to track retry count for abnormal restart, prevent infinite recursion
            is_validate: Whether to operate on val simulation (True uses val_* states)
            
        Returns:
            Status dict with checkpoint info if created
        """
        MAX_RETRY = 3
        mode = "val" if is_validate else "train"
        
        # ===== Select corresponding state variables (local aliases) =====
        if is_validate:
            env = self.val_env
            current_time = self.val_current_time
            running = self.val_running
            checkpoint_dir = self.val_checkpoint_dir
            traffic_state_collector = self.val_traffic_state_collector
            episode_count = self.val_episode_count
        else:
            env = self.env
            current_time = self.current_time
            running = self.running
            checkpoint_dir = self.checkpoint_dir
            traffic_state_collector = self.traffic_state_collector
            episode_count = self.episode_count
        
        if not running:
            return {"success": False, "error": f"{mode} simulation not running"}
        
        if not self.use_http and env is None:
            return {"success": False, "error": f"{mode} simulation not running"}
        
        duration = duration or self.checkpoint_interval
        
        try:
            #   HTTP mode: Only train supported
            if self.use_http and not is_validate:
                return self._step_http(duration)
            elif self.use_http and is_validate:
                return {"success": False, "error": "HTTP mode not supported for validation"}
            
            from utils.simulation_utils import run_controlled_simulation
            
            # Check if simulation should end or restart
            # val uses independent simulation_duration (if val_simulation_duration is configured)
            effective_duration = (self.val_simulation_duration or self.simulation_duration) if is_validate else self.simulation_duration
            if effective_duration and current_time >= effective_duration:
                if is_validate:
                    # val restart: Reset val_env to restart from 0:00 (same environment configuration)
                    self._restart_val_simulation()
                    env = self.val_env
                    current_time = self.val_current_time
                else:
                    # train restart: Original logic
                    if self.max_episodes is not None and self.episode_count >= self.max_episodes:
                        logger.info(f"Reached max episodes ({self.max_episodes}), stopping simulation")
                        self.running = False
                        return {
                            "success": True,
                            "current_time": self.current_time,
                            "simulation_ended": True,
                            "episode_count": self.episode_count,
                            "reason": "max_episodes_reached"
                        }
                    
                    if self.auto_restart:
                        logger.info(f"Simulation reached {self.simulation_duration}s, restarting with new seed...")
                        self._restart_simulation()
                        env = self.env
                        current_time = self.current_time
                    else:
                        logger.info(f"Simulation ended at {self.simulation_duration}s")
                        self.running = False
                        return {
                            "success": True,
                            "current_time": self.current_time,
                            "simulation_ended": True,
                            "episode_count": self.episode_count,
                        }
            
            # Get control configuration (from env.enabled_controls)
            control_configs = {}
            control_states = {}
            if hasattr(env, 'enabled_controls') and env.enabled_controls:
                for module_name, module_info in env.enabled_controls.items():
                    control_configs[module_name] = module_info.get('config', {})
                    control_states[module_name] = module_info.get('state')
            
            #   Pass _previous_checkpoint_path (consistent with run_joint_control.py)
            # Used by run_controlled_simulation to correctly save t-1 checkpoint
            prev_cp = self._val_previous_checkpoint_path if is_validate else self._previous_checkpoint_path
            if prev_cp:
                control_states["_previous_checkpoint_path"] = prev_cp
            
            # Call run_controlled_simulation
            cp_prefix = f"master_{self.master_id}_val" if is_validate else f"master_{self.master_id}"
            sim_id = f"master_{self.master_id}_val_ep{episode_count}" if is_validate else f"master_{self.master_id}_ep{episode_count}"
            
            results = run_controlled_simulation(
                env=env,
                duration=duration,
                step_seconds=30,
                min_step_seconds=1.0,
                traffic_state_collector=traffic_state_collector,
                checkpoint_interval=self.checkpoint_interval,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_prefix=cp_prefix,
                control_configs=control_configs if control_configs else None,
                control_states=control_states if control_states else None,
                save_checkpoint=True,
                is_first_simulation=(current_time == 0),
                config_name=self.context.get("config_name"),
                llm_name=None,
                simulation_id=sim_id,
            )
            
            # ===== Write back current time =====
            new_time = int(env.get_current_time())
            if is_validate:
                self.val_current_time = new_time
            else:
                self.current_time = new_time
            current_time = new_time
            
            # Update control states to env.enabled_controls
            if results.get("control_states"):
                for module_name, state in results["control_states"].items():
                    if module_name.startswith("_"):  # Skip special keys
                        continue
                    if module_name in env.enabled_controls:
                        env.enabled_controls[module_name]['state'] = state
            
            # Handle checkpoint
            checkpoint_info = None
            if results.get("checkpoint_reached"):
                checkpoint_info = {
                    "id": f"checkpoint_{current_time}",
                    "checkpoint_path": results.get("checkpoint_path"),
                    "checkpoint_path_t_minus_1": results.get("checkpoint_path_t_minus_1"),
                    "time": current_time,
                    "module_metrics": results.get("module_metrics", {}),
                }
                
                #   Update _previous_checkpoint_path (consistent with run_joint_control.py)
                # Next step() will pass this path to run_controlled_simulation for saving t-1
                cp_path = results.get("checkpoint_path")
                if cp_path:
                    if is_validate:
                        self._val_previous_checkpoint_path = cp_path
                    else:
                        self._previous_checkpoint_path = cp_path
                
                if not is_validate:
                    #   train only: Increase environment run count
                    self.env_run_counts[self.current_env_index] += 1
                    logger.info(
                        f"Master_{self.master_id}[{mode}]: Successfully reached checkpoint at {current_time}s. "
                        f"Env[{self.current_env_index}] run_count: {self.env_run_counts[self.current_env_index]}"
                    )
                    
                    #   train only: Run baseline simulation
                    try:
                        from utils.simulation_utils import run_policy_simulation
                        import copy
                        baseline_control_configs = {}
                        if hasattr(env, 'enabled_controls') and env.enabled_controls:
                            for mod_name, mod_info in env.enabled_controls.items():
                                baseline_control_configs[mod_name] = copy.deepcopy(mod_info.get('config', {}))
                        
                        # Ensure baseline starts without one-off decisions (similar to run_joint_control.py)
                        if "taxi_scheduling" in baseline_control_configs:
                            taxi_config = baseline_control_configs["taxi_scheduling"]
                            if isinstance(taxi_config, dict):
                                taxi_config.pop("dispatch_decisions", None)
                                taxi_config.pop("reposition_decisions", None)
                        
                        # Use checkpoint_path_t_minus_1 (t-1 state, start of interval) for fair comparison
                        # This matches run_joint_control.py logic
                        baseline_checkpoint = checkpoint_info.get("checkpoint_path_t_minus_1") or checkpoint_info["checkpoint_path"]
                        
                        baseline_result = run_policy_simulation(
                            checkpoint_path=baseline_checkpoint,
                            control_configs=baseline_control_configs,
                            duration=self.checkpoint_interval,
                            use_gui=False,
                            config_path=self.config_path,
                            checkpoint_interval=self.checkpoint_interval,
                            seed=self.seed,
                        )
                        
                        if baseline_result.get("success"):
                            checkpoint_info["baseline_module_metrics"] = baseline_result.get("module_metrics", {})
                            logger.info(f"Master_{self.master_id}: Baseline simulation completed, metrics: {list(checkpoint_info['baseline_module_metrics'].keys())}")
                        else:
                            logger.warning(f"Master_{self.master_id}: Baseline simulation failed: {baseline_result.get('error')}, falling back to main simulation metrics")
                            checkpoint_info["baseline_module_metrics"] = checkpoint_info.get("module_metrics", {})
                    except Exception as baseline_err:
                        logger.warning(f"Master_{self.master_id}: Baseline simulation error: {baseline_err}, falling back to main simulation metrics")
                        checkpoint_info["baseline_module_metrics"] = checkpoint_info.get("module_metrics", {})
                else:
                    #   val environment also needs to run baseline simulation
                    #   Use val-specific environment configuration (not train's self.config_path)
                    val_cfg_path = self._val_config_path or self.config_path
                    logger.info(f"Master_{self.master_id}[{mode}]: Reached checkpoint at {current_time}s")
                    try:
                        from utils.simulation_utils import run_policy_simulation
                        import copy
                        baseline_control_configs = {}
                        if hasattr(env, 'enabled_controls') and env.enabled_controls:
                            for mod_name, mod_info in env.enabled_controls.items():
                                baseline_control_configs[mod_name] = copy.deepcopy(mod_info.get('config', {}))
                        
                        # Ensure baseline starts without one-off decisions
                        if "taxi_scheduling" in baseline_control_configs:
                            taxi_config = baseline_control_configs["taxi_scheduling"]
                            if isinstance(taxi_config, dict):
                                taxi_config.pop("dispatch_decisions", None)
                                taxi_config.pop("reposition_decisions", None)
                        
                        # Use checkpoint_path_t_minus_1 for fair comparison
                        baseline_checkpoint = checkpoint_info.get("checkpoint_path_t_minus_1") or checkpoint_info["checkpoint_path"]
                        
                        baseline_result = run_policy_simulation(
                            checkpoint_path=baseline_checkpoint,
                            control_configs=baseline_control_configs,
                            duration=self.checkpoint_interval,
                            use_gui=False,
                            config_path=val_cfg_path,
                            checkpoint_interval=self.checkpoint_interval,
                            seed=self.val_seed,  #   Use fixed val_seed (consistent with val main simulation)
                        )
                        
                        if baseline_result.get("success"):
                            checkpoint_info["baseline_module_metrics"] = baseline_result.get("module_metrics", {})
                            logger.info(f"Master_{self.master_id}[{mode}]: Baseline simulation completed, metrics: {list(checkpoint_info['baseline_module_metrics'].keys())}")
                        else:
                            logger.warning(f"Master_{self.master_id}[{mode}]: Baseline simulation failed: {baseline_result.get('error')}, falling back to main simulation metrics")
                            checkpoint_info["baseline_module_metrics"] = checkpoint_info.get("module_metrics", {})
                    except Exception as baseline_err:
                        logger.warning(f"Master_{self.master_id}[{mode}]: Baseline simulation error: {baseline_err}, falling back to main simulation metrics")
                        checkpoint_info["baseline_module_metrics"] = checkpoint_info.get("module_metrics", {})
                
                # ===== Write back checkpoints =====
                with self._lock:
                    if is_validate:
                        self.val_checkpoints = [checkpoint_info]
                    else:
                        self.checkpoints = [checkpoint_info]
                
                # Reset metrics for next checkpoint interval
                logger.info(f"Master_{self.master_id}[{mode}]: Resetting metrics, continuing from {current_time}s")
                env.reset_metrics()
            else:
                # ❌ Checkpoint not reached, trigger abnormal restart
                logger.warning(
                    f"Master_{self.master_id}[{mode}]: Simulation ended at {current_time}s "
                    f"without reaching checkpoint (expected {self.checkpoint_interval}s). Triggering restart..."
                )
                if self.auto_restart:
                    if _retry_count >= MAX_RETRY:
                        logger.error(f"Master_{self.master_id}[{mode}]: Max retry count ({MAX_RETRY}) reached, giving up")
                        return {"success": False, "error": f"Max retry reached after {_retry_count} attempts. Checkpoint not reached."}
                    
                    if is_validate:
                        self._restart_val_simulation(is_abnormal=True)
                    else:
                        self._handle_abnormal_termination()
                    
                    logger.info(f"Master_{self.master_id}[{mode}]: Retrying after restart (retry {_retry_count + 1}/{MAX_RETRY})...")
                    new_step_result = self.step(duration, _retry_count=_retry_count + 1, is_validate=is_validate)
                    
                    new_step_result["abnormal_restart"] = True
                    new_step_result["retry_count"] = _retry_count + 1
                    new_step_result["previous_error"] = "Checkpoint not reached"
                    return new_step_result
                else:
                    return {"success": False, "error": "Checkpoint not reached"}
            
            return {
                "success": True,
                "current_time": current_time,
                "checkpoint": checkpoint_info,
                "episode_count": episode_count,
                "module_metrics": results.get("module_metrics", {}),
            }
            
        except Exception as e:
            logger.error(f"Master_{self.master_id}[{mode}]: Error during step: {e}")
            import traceback
            traceback.print_exc()
            
            if self.auto_restart:
                if _retry_count >= MAX_RETRY:
                    logger.error(f"Master_{self.master_id}[{mode}]: Max retry count ({MAX_RETRY}) reached, giving up")
                    return {"success": False, "error": f"Max retry reached after {_retry_count} attempts. Last error: {e}"}
                
                logger.warning(f"Master_{self.master_id}[{mode}]: Triggering restart due to exception (retry {_retry_count + 1}/{MAX_RETRY})...")
                try:
                    if is_validate:
                        self._restart_val_simulation(is_abnormal=True)
                    else:
                        self._handle_abnormal_termination()
                    
                    new_step_result = self.step(duration, _retry_count=_retry_count + 1, is_validate=is_validate)
                    new_step_result["abnormal_restart"] = True
                    new_step_result["retry_count"] = _retry_count + 1
                    new_step_result["previous_error"] = str(e)
                    return new_step_result
                    
                except Exception as restart_error:
                    logger.error(f"Master_{self.master_id}[{mode}]: Restart failed: {restart_error}")
                    return {"success": False, "error": f"Step error: {e}, Restart error: {restart_error}"}
            else:
                return {"success": False, "error": str(e)}
    
    def _step_http(self, duration: int) -> Dict[str, Any]:
        """
        HTTP mode step implementation: Call remote SUMO server via HTTP client.
        
        Args:
            duration: Time to advance (seconds)
            
        Returns:
            Status dict with checkpoint info if created
        """
        try:
            logger.info(f"Master_{self.master_id}: Calling HTTP step_master, duration={duration}s")
            
            # Call HTTP server's run_simulation API
            result = self.http_client.run_simulation(
                session_id=self.http_master_id,
                duration=duration,
                checkpoint_interval=self.checkpoint_interval
            )
            
            # Update local state
            self.current_time = result.get("current_time", self.current_time)
            checkpoint_path = result.get("checkpoint_path")
            
            logger.info(f"Master_{self.master_id}: HTTP step completed, time={self.current_time}s")
            
            # Construct checkpoint information (consistent with local mode)
            checkpoint_info = None
            if result.get("checkpoint_reached"):
                checkpoint_info = {
                    "id": f"checkpoint_{self.current_time}",
                    "checkpoint_path": checkpoint_path,  # Managed by server side in HTTP mode
                    "checkpoint_path_t_minus_1": None,
                    "time": self.current_time,
                    "module_metrics": result.get("baseline_result", {}),
                }
                
                with self._lock:
                    self.checkpoints = [checkpoint_info]
                
                #   Successfully reached checkpoint, increase current environment run count
                self.env_run_counts[self.current_env_index] += 1
                logger.info(
                    f"Master_{self.master_id}: Successfully reached checkpoint at {self.current_time}s (HTTP mode). "
                    f"Env[{self.current_env_index}] run_count: {self.env_run_counts[self.current_env_index]}"
                )
            else:
                # ❌ Checkpoint not reached, trigger abnormal restart
                logger.warning(
                    f"Master_{self.master_id}: Simulation ended at {self.current_time}s "
                    f"without reaching checkpoint (expected {self.checkpoint_interval}s). Triggering abnormal restart (HTTP mode)..."
                )
                if self.auto_restart:
                    self._handle_abnormal_termination_http()
                    return {
                        "success": True,
                        "abnormal_restart": True,
                        "current_time": 0,
                        "episode_count": self.episode_count,
                    }
                else:
                    return {"success": False, "error": "Checkpoint not reached"}
            
            # Check if simulation ended (managed by Ray Master)
            if self.simulation_duration and self.current_time >= self.simulation_duration:
                logger.info(f"Master_{self.master_id}: Simulation duration reached ({self.current_time}s >= {self.simulation_duration}s)")
                if self.auto_restart:
                    logger.info(f"Master_{self.master_id}: Auto-restarting simulation via HTTP...")
                    self._restart_http()
                else:
                    self.running = False
            
            return {
                "success": True,
                "current_time": self.current_time,
                "checkpoint": checkpoint_info,
                "episode_count": self.episode_count,
                "module_metrics": result.get("baseline_result", {}),
            }
            
        except Exception as e:
            logger.error(f"Master_{self.master_id}: HTTP step failed: {e}")
            import traceback
            traceback.print_exc()
            
            #   Exception occurred, trigger abnormal restart
            if self.auto_restart:
                logger.warning(f"Master_{self.master_id}: Triggering abnormal restart (HTTP mode) due to exception...")
                try:
                    self._handle_abnormal_termination_http()
                    return {
                        "success": True,
                        "abnormal_restart": True,
                        "current_time": 0,
                        "episode_count": self.episode_count,
                        "error": str(e)
                    }
                except Exception as restart_error:
                    logger.error(f"Master_{self.master_id}: HTTP abnormal restart failed: {restart_error}")
                    return {"success": False, "error": f"Step error: {e}, Restart error: {restart_error}"}
            else:
                return {"success": False, "error": str(e)}
    
    def _handle_abnormal_termination_http(self):
        """
        HTTP mode abnormal restart: Select new environment and restart via HTTP API
        
        Trigger conditions:
        - HTTP request failed
        - Remote SUMO server exception
        """
        logger.warning(
            f"Master_{self.master_id}: Abnormal termination detected (HTTP mode). Selecting new environment..."
        )
        
        # Select new environment (excluding current environment)
        self.current_env_index = self._select_next_env(exclude_current=True)
        current_env = self.environments[self.current_env_index]
        self.config_path = current_env["sumo_config_path"]
        self.control_modules = current_env.get("control_modules", [])
        
        # Call HTTP restart (internally updates episode_count and seed)
        self._restart_http_internal(is_abnormal=True)
    
    def _restart_http(self):
        """
        HTTP mode normal restart: Restart remote SUMO session via HTTP API
        
        - Select environment with fewest runs (not excluding current environment)
        - Update episode_count and seed
        - Call internal restart method
        """
        # Select next environment (not excluding current environment)
        self.current_env_index = self._select_next_env(exclude_current=False)
        current_env = self.environments[self.current_env_index]
        self.config_path = current_env["sumo_config_path"]
        self.control_modules = current_env.get("control_modules", [])
        self.current_env_episodes = current_env.get("episodes", 1)
        
        # Call internal restart method (normal restart)
        self._restart_http_internal(is_abnormal=False)
    
    def _restart_http_internal(self, is_abnormal=False):
        """
        HTTP mode internal restart method, supports normal and abnormal restart
        
        Args:
            is_abnormal: Whether it's an abnormal restart
                        - False: Normal restart (reached simulation_duration)
                        - True: Abnormal restart (HTTP request failed or server exception)
        
        Note:
            Whether normal or abnormal restart, episode_count is incremented and seed is updated,
            to avoid using the same seed on abnormal restart causing repeated exceptions.
        """
        try:
            # Whether normal or abnormal restart, update episode count
            self.episode_count += 1
            
            # Update seed (ensure each restart uses a different seed)
            self.seed = self.base_seed + self.master_id + (self.episode_count * self.seed_increment)
            
            current_env = self.environments[self.current_env_index]
            
            logger.info(
                f"Master_{self.master_id} {'[ABNORMAL]' if is_abnormal else '[NORMAL]'} Restart Episode {self.episode_count} (HTTP mode):\n"
                f"  - Environment: [{self.current_env_index}/{len(self.environments)}] {self.config_path}\n"
                f"  - Modules: {self.control_modules}\n"
                f"  - Seed: {self.seed}\n"
                f"  - Env run_counts: {dict(sorted(self.env_run_counts.items()))}"
            )
            
            # Call HTTP API to restart remote session
            result = self.http_client.restart_session(
                session_id=self.http_master_id,
                env_index=self.current_env_index,
                seed=self.seed,
                episode_count=self.episode_count
            )
            
            if not result.get("success"):
                raise RuntimeError(f"HTTP restart failed: {result.get('error', 'Unknown error')}")
            
            # Reset local state
            self.current_time = 0
            with self._lock:
                self.checkpoints = []
                self.checkpoint_in_use = False
                self.active_worker_ids = set()
            
            #   Clear memory, new 86400s cycle starts from scratch
            self.context["memory"] = []
            logger.info("Memory cleared for new episode")
            
            #   Get new traffic_states_filepath from HTTP server
            try:
                session_info = self.http_client.get_session(self.http_master_id)
                self.traffic_states_filepath = session_info.get("traffic_states_filepath")
                logger.info(f"Master_{self.master_id}: Updated traffic_states_filepath from server: {self.traffic_states_filepath}")
            except Exception as e:
                logger.warning(f"Master_{self.master_id}: Failed to get traffic_states_filepath from server: {e}")
            
            logger.info(f"Master_{self.master_id} Episode {self.episode_count}: HTTP restart completed successfully")
            
        except Exception as e:
            logger.error(f"Master_{self.master_id}: HTTP restart failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _select_next_env(self, exclude_current=False):
        """
        Unified environment selection method: Fewest runs priority + random selection
        
        Args:
            exclude_current: Whether to exclude current environment (True for abnormal restart)
        
        Returns:
            selected_env_index: Selected environment index
        """
        import random
        
        # 1. Determine available environment list
        if exclude_current:
            available_envs = [i for i in range(len(self.environments)) if i != self.current_env_index]
        else:
            available_envs = list(range(len(self.environments)))
        
        if not available_envs:
            logger.error(f"Master_{self.master_id}: No available environments!")
            raise RuntimeError("No available environments for selection")
        
        # 2. Find environments with fewest runs
        min_count = min(self.env_run_counts[i] for i in available_envs)
        candidates = [i for i in available_envs if self.env_run_counts[i] == min_count]
        
        # 3. Randomly select one
        selected_env_index = random.choice(candidates)
        
        # 4. Log selection info
        logger.info(
            f"Master_{self.master_id}: Environment selection:\n"
            f"  - Current env: [{self.current_env_index}] (run_count={self.env_run_counts[self.current_env_index]})\n"
            f"  - Selected env: [{selected_env_index}] (run_count={self.env_run_counts[selected_env_index]})\n"
            f"  - Candidates: {candidates} (all with run_count={min_count})\n"
            f"  - Exclude current: {exclude_current}\n"
            f"  - All run_counts: {dict(sorted(self.env_run_counts.items()))}"
        )
        
        return selected_env_index
    
    def _handle_abnormal_termination(self):
        """
        Handle abnormal termination: Select new environment and restart
        
        Trigger conditions:
        - Simulation interrupted before reaching checkpoint_interval
        - step() returned success=False
        """
        logger.warning(
            f"Master_{self.master_id}: Abnormal termination detected at {self.current_time}s "
            f"(expected {self.checkpoint_interval}s). Selecting new environment..."
        )
        
        # Select new environment (excluding current environment)
        self.current_env_index = self._select_next_env(exclude_current=True)
        current_env = self.environments[self.current_env_index]
        self.config_path = current_env["sumo_config_path"]
        self.control_modules = current_env.get("control_modules", [])
        
        # Call restart (does not increase episode_count, since this is an abnormal restart)
        self._restart_simulation_internal(is_abnormal=True)
    
    def _restart_simulation(self):
        """
        Normal restart: Restart after reaching simulation_duration
        
        - Select environment with fewest runs (can be current environment)
        - Update episode_count
        - Call internal restart method
        """
        # Select next environment (not excluding current environment)
        self.current_env_index = self._select_next_env(exclude_current=False)
        current_env = self.environments[self.current_env_index]
        self.config_path = current_env["sumo_config_path"]
        self.control_modules = current_env.get("control_modules", [])
        self.current_env_episodes = current_env.get("episodes", 1)
        
        # Call internal restart method (normal restart, episode_count +1)
        self._restart_simulation_internal(is_abnormal=False)
    
    def _select_next_val_env(self, exclude_current=False):
        """
        Val-specific environment selection: Select from val_environments (or train environments).
        Fewest runs priority + random selection.
        """
        import random
        
        env_pool = self.val_environments if self.val_environments else self.environments
        run_counts = self._val_env_run_counts
        current_idx = self._val_current_env_index
        
        if exclude_current:
            available = [i for i in range(len(env_pool)) if i != current_idx]
        else:
            available = list(range(len(env_pool)))
        
        if not available:
            raise RuntimeError("No available val environments for selection")
        
        min_count = min(run_counts.get(i, 0) for i in available)
        candidates = [i for i in available if run_counts.get(i, 0) == min_count]
        selected = random.choice(candidates)
        
        logger.info(
            f"Master_{self.master_id}[val]: Environment selection: "
            f"current=[{current_idx}], selected=[{selected}] (run_count={run_counts.get(selected, 0)}), "
            f"candidates={candidates}, exclude_current={exclude_current}"
        )
        return selected
    
    def _restart_val_simulation(self, is_abnormal=False):
        """
        Restart val simulation environment.
        
          Val restart strategy differs from Train:
        - Normal restart (duration reached): Keep same environment, only reset time to 0:00 (for fixed scenario evaluation)
        - Abnormal restart: Switch to other environment (excluding current environment)
        
          Does not modify train's self.config_path / self.control_modules / self.current_env_index
        
        Args:
            is_abnormal: Whether it's an abnormal restart
                        - False: Normal restart (keep same environment, reset timeline)
                        - True: Abnormal restart (switch to other environment)
        """
        try:
            self.val_episode_count += 1
            
            if is_abnormal:
                # Abnormal restart: Switch to other environment
                val_env_index = self._select_next_val_env(exclude_current=True)
                env_pool = self.val_environments if self.val_environments else self.environments
                val_env_config = env_pool[val_env_index]
                val_config_path = val_env_config["sumo_config_path"]
                val_control_modules = val_env_config.get("control_modules", [])
                
                self._val_current_env_index = val_env_index
                self._val_config_path = val_config_path
                self._val_control_modules = val_control_modules
            else:
                # Normal restart: Keep same environment, only reset timeline
                val_config_path = self._val_config_path
                val_control_modules = self._val_control_modules
                val_env_index = self._val_current_env_index
            
            # Update val environment run count
            self._val_env_run_counts[val_env_index] = self._val_env_run_counts.get(val_env_index, 0) + 1
            
            #   Always use fixed val_seed (calculated at initialization, never changes)
            # Ensure validation at different training stages always sees the same traffic scenarios for fair policy comparison
            # Also use fixed seed on abnormal restart, environment switch itself provides variation
            val_seed = self.val_seed
            
            _val_pool = self.val_environments if self.val_environments else self.environments
            logger.info(
                f"Master_{self.master_id}[val] {'[ABNORMAL]' if is_abnormal else '[NORMAL]'} Restart Episode {self.val_episode_count}:\n"
                f"  - Environment: [{val_env_index}/{len(_val_pool)}] {val_config_path}\n"
                f"  - Modules: {val_control_modules}\n"
                f"  - Seed: {val_seed} (fixed val_seed)"
            )
            
            if self.val_env:
                self.val_env.close()
            
            from utils.simulation_utils import create_sumo_env
            _use_sim_taxi = 'taxi_scheduling' in (val_control_modules or [])
            self.val_env, _ = create_sumo_env(
                config_path=val_config_path,
                use_gui=False,
                seed=val_seed,
                control_modules=val_control_modules,
                use_simulated_taxi_system=_use_sim_taxi,
            )
            
            self.val_current_time = 0
            self._val_previous_checkpoint_path = None  #   Reset val t-1 checkpoint path
            with self._lock:
                self.val_checkpoints = []
                self.val_checkpoint_in_use = False
                self.val_active_worker_ids = set()
            
            # Clean up old checkpoint files
            if self.val_checkpoint_dir and self.val_checkpoint_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.val_checkpoint_dir)
                    self.val_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Master_{self.master_id}[val]: Failed to clean checkpoint dir: {e}")
            
            # Recreate TrafficStateCollector
            from utils.traffic_state_collector import TrafficStateCollector, init_traffic_states_file
            config_name = Path(val_config_path).parent.name if val_config_path else None
            val_traffic_states_filepath = init_traffic_states_file(
                simulation_id=f"master_{self.master_id}_val",
                config_name=config_name,
                llm_name=None,
                control_modules=val_control_modules,
            )
            graphs = self.val_env.get_road_network_graphs()
            self.val_traffic_state_collector = TrafficStateCollector(
                env=self.val_env,
                traffic_states_filepath=val_traffic_states_filepath,
                interval=60.0,
                lane_dict=graphs.get("lane_dict"),
                lane_inter_graph=graphs.get("lane_inter_graph"),
                simulation_id=f"master_{self.master_id}_val",
            )
            self.val_traffic_states_filepath = str(val_traffic_states_filepath)
            
            logger.info(f"Master_{self.master_id}[val]: val_env restarted successfully (config={val_config_path}, modules={val_control_modules})")
            
        except Exception as e:
            logger.error(f"Master_{self.master_id}[val]: Failed to restart val_env: {e}")
            raise
    
    def _restart_simulation_internal(self, is_abnormal=False):
        """
        Internal restart method, supports normal and abnormal restart
        
        Args:
            is_abnormal: Whether it's an abnormal restart
                        - False: Normal restart (reached simulation_duration)
                        - True: Abnormal restart (did not reach checkpoint_interval)
        
        Note:
            Whether normal or abnormal restart, episode_count is incremented and seed is updated,
            to avoid using the same seed on abnormal restart causing repeated exceptions.
        """
        try:
            # Whether normal or abnormal restart, update episode count
            self.episode_count += 1
            
            # Update seed (ensure each restart uses a different seed)
            self.seed = self.base_seed + self.master_id + (self.episode_count * self.seed_increment)
            
            current_env = self.environments[self.current_env_index]
            
            logger.info(
                f"Master_{self.master_id} {'[ABNORMAL]' if is_abnormal else '[NORMAL]'} Restart Episode {self.episode_count}:\n"
                f"  - Environment: [{self.current_env_index}/{len(self.environments)}] {self.config_path}\n"
                f"  - Modules: {self.control_modules}\n"
                f"  - Seed: {self.seed}\n"
                f"  - Env run_counts: {dict(sorted(self.env_run_counts.items()))}"
            )
            
            # Reset environment with new seed and environment config
            if self.env:
                self.env.close()
            
            from utils.simulation_utils import create_sumo_env
            _use_sim_taxi = 'taxi_scheduling' in (self.control_modules or [])
            self.env, _ = create_sumo_env(
                config_path=self.config_path,
                use_gui=self.use_gui,
                seed=self.seed,
                control_modules=self.control_modules,
                use_simulated_taxi_system=_use_sim_taxi,
            )
            
            # Reset time and checkpoints
            self.current_time = 0
            self._previous_checkpoint_path = None  #   Reset t-1 checkpoint path
            with self._lock:
                self.checkpoints = []
                self.checkpoint_in_use = False
                self.active_worker_ids = set()
            
            #   Clear memory, new 86400s cycle starts from scratch
            self.context["memory"] = []
            logger.info("Memory cleared for new episode")
            
            # Delete old checkpoint files
            if self.checkpoint_dir and self.checkpoint_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.checkpoint_dir)
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Deleted old checkpoint directory: {self.checkpoint_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint directory: {e}")
            
            # Delete traffic_states file and recreate collector
            if self.traffic_state_collector and hasattr(self.traffic_state_collector, 'traffic_states_filepath'):
                traffic_states_file = self.traffic_state_collector.traffic_states_filepath
                if traffic_states_file and Path(traffic_states_file).exists():
                    try:
                        Path(traffic_states_file).unlink()
                        logger.info(f"Deleted traffic_states file: {traffic_states_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete traffic_states file: {e}")
            
            # Recreate TrafficStateCollector with fresh state (equivalent to reset)
            from utils.traffic_state_collector import init_traffic_states_file
            
            # Extract config_name from config_path
            config_name = None
            if self.config_path:
                config_name = Path(self.config_path).parent.name
            
            # Generate traffic_states file with proper naming convention
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=f"master_{self.master_id}",  # Only master_id, no instance_id
                config_name=config_name,
                llm_name=None,  # Master doesn't use LLM
                control_modules=self.control_modules
            )
            
            graphs = self.env.get_road_network_graphs()
            
            from utils.simulation_utils import TrafficStateCollector
            self.traffic_state_collector = TrafficStateCollector(
                env=self.env,
                traffic_states_filepath=traffic_states_filepath,
                interval=60.0,
                lane_dict=graphs.get("lane_dict"),
                lane_inter_graph=graphs.get("lane_inter_graph"),
                simulation_id=f"master_{self.master_id}",
            )
            self.traffic_states_filepath = str(traffic_states_filepath)
            
            # Note: Initial checkpoint will be saved by first run_controlled_simulation() call
            
            logger.info(f"Episode {self.episode_count}: Simulation restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart simulation: {e}")
            raise
    
    
    def get_latest_checkpoint(self, is_validate: bool = False) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint info."""
        with self._lock:
            checkpoints = self.val_checkpoints if is_validate else self.checkpoints
            if checkpoints:
                return checkpoints[-1].copy()
        return None
    
    def get_checkpoint_by_time(self, time: int) -> Optional[Dict[str, Any]]:
        """Get checkpoint closest to given time."""
        with self._lock:
            for cp in reversed(self.checkpoints):
                if cp["time"] <= time:
                    return cp.copy()
        return None
    
    def get_context(self, is_validate: bool = False) -> Dict[str, Any]:
        """
        Get context for code execution in workers.
        
        Corresponds to the context used by LLMAgent in run_bus_scheduling_control.py.
        
        Returns:
            Context dict containing:
            
            # === Checkpoint related (from records/checkpoints/) ===
            - checkpoint_path: str - SUMO simulation state snapshot file path
            - checkpoint_time: float - Simulation time corresponding to checkpoint (seconds)
            
            # === Road network graph structure (from simulation_utils.build_road_network_graphs()) ===
            - lane_graph: NetworkX MultiDiGraph - Lane topology graph, connecting lanes
            - lane_inter_graph: NetworkX DiGraph - Lane group to intersection connection graph
            - intersection_graph: NetworkX MultiDiGraph - Intersection connection graph
            - lane_group_graph: NetworkX MultiDiGraph - Lane group connection graph
            - road_graph: NetworkX DiGraph - Road (edge) topology graph
            
            # === Metadata dictionaries ===
            - lane_dict: Dict - Lane metadata {lane_id: {length, speed, ...}}
            - road_dict: Dict - Road attributes {road_id: {type, priority, numLanes, speed, from, to}}
            
            # === Traffic state data (from records/traffic_states/) ===
            - traffic_states_filepath: str - Traffic state time-series data file path
            
            # === Other configuration ===
            - control_modules: List[str] - List of enabled control modules
            - simulation_id: str - Simulation session ID
        """
        checkpoint = self.get_latest_checkpoint(is_validate=is_validate)
        ctx = self.context.copy()
        
        # Select corresponding env
        env = self.val_env if is_validate else self.env
        
        # === Checkpoint information ===
        if checkpoint:
            ctx["checkpoint_path"] = checkpoint.get("checkpoint_path")  # t state
            ctx["checkpoint_path_t_minus_1"] = checkpoint.get("checkpoint_path_t_minus_1")  # t-1 state
            ctx["checkpoint_time"] = checkpoint["time"]
        
        # === Road network graph structure (corresponds to simulation_utils.build_road_network_graphs()) ===
        if env:
            # Use env.get_road_network_graphs() to get complete graph structure
            if hasattr(env, 'get_road_network_graphs'):
                try:
                    graphs = env.get_road_network_graphs()
                    # Layer 3: Module-specific graphs (intersection-related only)
                    ctx["lane_graph"] = graphs.get("lane_graph")
                    ctx["lane_inter_graph"] = graphs.get("lane_inter_graph")
                    ctx["intersection_graph"] = graphs.get("intersection_graph")
                    ctx["lane_group_graph"] = graphs.get("lane_group_graph")
                    ctx["road_graph"] = graphs.get("road_graph")
                    ctx["lane_dict"] = graphs.get("lane_dict")
                    ctx["road_dict"] = graphs.get("road_dict")
                    ctx["highway_segment_graph"] = graphs.get("highway_segment_graph")
                except Exception as e:
                    logger.warning(f"Failed to get road network graphs: {e}")
            else:
                # Fallback: Get directly from env attributes
                if hasattr(env, 'lane_graph'):
                    ctx["lane_graph"] = env.lane_graph
                if hasattr(env, 'intersection_graph'):
                    ctx["intersection_graph"] = env.intersection_graph
            
            # Highway-specific data
            if hasattr(env, 'highway_dict'):
                ctx["highway_ids"] = list(env.highway_dict.keys())
            if hasattr(env, 'highway_subgraph'):
                ctx["highway_graph"] = env.highway_subgraph
            if hasattr(env, 'highway_info_dict'):
                ctx["highway_segment_dict"] = env.highway_info_dict
            
            # Layer 1: Foundation Layer - Complete network graphs
            if hasattr(env, 'network_graphs'):
                ctx["network_graphs"] = env.network_graphs
                ctx["full_lane_graph"] = env.network_graphs.get("lane_graph")
                ctx["full_road_graph"] = env.network_graphs.get("road_graph")
            if hasattr(env, 'network_dicts'):
                ctx["network_dicts"] = env.network_dicts
                ctx["full_lane_dict"] = env.network_dicts.get("lane_dict")
                ctx["full_road_dict"] = env.network_dicts.get("road_dict")
            
            # Layer 2: Zone infrastructure
            if hasattr(env, 'zone_dict'):
                ctx["zone_dict"] = env.zone_dict
            if hasattr(env, 'zone_graph'):
                ctx["zone_graph"] = env.zone_graph
            
            # Transit-specific data (for bus_scheduling & subway_scheduling)
            if hasattr(env, 'transit_graph'):
                ctx["transit_graph"] = env.transit_graph
            if hasattr(env, 'bus_route_info'):
                ctx["bus_route_info"] = env.bus_route_info
            
            # Ramp-specific data (for ramp_metering)
            if hasattr(env, 'ramp_lane_graph'):
                ctx["ramp_lane_graph"] = env.ramp_lane_graph
            
            # Taxi-specific data (for taxi_scheduling)
            if hasattr(env, 'get_taxi_fleet_state'):
                try:
                    ctx["taxi_fleet_state"] = env.get_taxi_fleet_state()
                except Exception as e:
                    logger.warning(f"Failed to get taxi fleet state: {e}")
            if hasattr(env, 'get_pending_reservations'):
                try:
                    ctx["pending_reservations"] = env.get_pending_reservations()
                except Exception as e:
                    logger.warning(f"Failed to get pending reservations: {e}")
            if hasattr(env, 'get_taz_stats'):
                try:
                    ctx["taz_stats"] = env.get_taz_stats()
                except Exception as e:
                    logger.warning(f"Failed to get TAZ stats: {e}")
            if hasattr(env, 'enabled_controls') and 'taxi_scheduling' in env.enabled_controls:
                ctx["current_taxi_config"] = env.enabled_controls['taxi_scheduling'].get('config', {})
            if hasattr(env, 'config'):
                env_config = env.config
                ctx["taxi_dispatch_algorithm"] = env_config.get("TAXI_DISPATCH_ALGORITHM")
                ctx["taxi_idle_algorithm"] = env_config.get("TAXI_IDLE_ALGORITHM")
        
        # === Traffic state data path ===
        tsf = self.val_traffic_states_filepath if is_validate else self.traffic_states_filepath
        if tsf:
            ctx["traffic_states_filepath"] = str(tsf)
        
        #   Add current environment's control_modules (important!)
        # Worker needs to know which control modules the current environment uses to correctly initialize LLMAgentManager
        # val mode uses val-specific environment information (if independently configured)
        if is_validate and self._val_control_modules is not None:
            ctx["control_modules"] = self._val_control_modules
            ctx["current_env_index"] = self._val_current_env_index
            ctx["current_env_episodes"] = 1  # val doesn't need episodes concept
        else:
            ctx["control_modules"] = self.control_modules
            ctx["current_env_index"] = self.current_env_index
            ctx["current_env_episodes"] = self.current_env_episodes
        
        #   Add current_configs (current control configuration)
        # Extract current control configuration from env.enabled_controls, corresponds to run_joint_control.py
        #   Preserve complete {module, config} structure so LLMAgentManager can call module.validate_config()
        if env and hasattr(env, 'enabled_controls') and env.enabled_controls:
            control_configs = {}
            for module_name, module_info in env.enabled_controls.items():
                module = module_info.get('module')
                config = module_info.get('config', {})
                if module and config:
                    # Preserve module object (class can be pickled)
                    control_configs[module_name] = {
                        'module': module,
                        'config': copy.deepcopy(config)
                    }
            ctx["current_configs"] = control_configs
        else:
            ctx["current_configs"] = {}
        
        #   Add other fields corresponding to run_joint_control.py
        ctx["checkpoint_interval"] = self.checkpoint_interval
        ctx["test_duration"] = self.checkpoint_interval  # Same as checkpoint_interval
        ctx["use_gui"] = self.use_gui
        ctx["enabled_modules"] = ctx["control_modules"]  # Same as control_modules (for val, already set to _val_control_modules above)
        #   val mode uses fixed val_seed (calculated at initialization, never changes)
        # Ensure Worker policy simulation and baseline use the same random seed, and it doesn't change with train episodes
        ctx["seed"] = self.val_seed if is_validate else self.seed
        
        #   val mode uses val-specific config_path (may be a different map)
        # Ensure Worker policy simulation loads the same map configuration as val_env/baseline
        effective_config_path = (self._val_config_path or self.config_path) if is_validate else self.config_path
        if effective_config_path:
            from pathlib import Path
            ctx["config_path"] = effective_config_path
            config_name = Path(effective_config_path).parent.name
            ctx["config_name"] = config_name
        
        # Add simulation_duration (if any)
        effective_sim_duration = (self.val_simulation_duration or self.simulation_duration) if is_validate else self.simulation_duration
        if effective_sim_duration:
            ctx["run_duration"] = effective_sim_duration
        
        #   Add module_metrics (from latest checkpoint)
        # Corresponds to the logic of getting results.get("module_metrics") in run_joint_control.py
        if checkpoint:
            ctx["module_metrics"] = checkpoint.get("module_metrics", {})
            #   Prefer baseline simulation metrics as initial_best_result
            # Baseline loads from checkpoint, Python state initializes from zero, consistent with policy simulation starting point
            ctx["baseline_module_metrics"] = checkpoint.get("baseline_module_metrics", checkpoint.get("module_metrics", {}))
        else:
            ctx["module_metrics"] = {}
            ctx["baseline_module_metrics"] = {}
        
        #   Add cross_module_dependencies (inter-module dependency relationships)
        # Corresponds to MODULE_DEPENDENCIES in run_joint_control.py
        ctx["cross_module_dependencies"] = {
            "signal_timing": {"affects": ["bus_scheduling", "taxi_scheduling"], "affected_by": []},
            "highway_speed_limit": {"affects": ["ramp_metering"], "affected_by": []},
            "ramp_metering": {"affects": [], "affected_by": ["highway_speed_limit"]},
            "bus_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
            "taxi_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
            "subway_scheduling": {"affects": [], "affected_by": []},
        }
            
        return ctx
    
    def get_enabled_controls(self, is_validate: bool = False) -> Dict[str, Any]:
        """
        Get enabled_controls from environment.
        
        Args:
            is_validate: If True, return val_env's enabled_controls
        
        Returns:
            Dictionary of enabled control modules with their configs
        """
        #   HTTP mode: Get from context (returned by server)
        if self.use_http:
            return self.context.get('enabled_controls', {})
        
        #   Local mode: Select corresponding env based on is_validate
        env = self.val_env if is_validate else self.env
        if env and hasattr(env, 'enabled_controls'):
            return env.enabled_controls
        return {}
    
    def report_best_policy(
        self, 
        policy: Dict[str, Dict],
        memory: List[str],
        worker_id: str,
        is_validate: bool = False,
    ) -> Dict[str, Any]:
        """
        Workers report their sampled policy and memory for GRPO exploration.
        
          New design: Collect (policy, memory) pairs, randomly select one pair to apply in apply_best_policies.
        Memory accumulates within 86400s, cleared on restart.
        
        Args:
            policy: Complete policy dict {module_name: config}
            memory: Memory list from this worker
            worker_id: Worker identifier
            is_validate: Whether it's validation mode
            
        Returns:
            Acknowledgment dict
        """
        mode = "val" if is_validate else "train"
        with self._lock:
            policies_dict = self.val_best_policies if is_validate else self.best_policies
            policies_dict[worker_id] = {
                "policy": policy,
                "memory": memory,
                "updated_at": time.time(),
            }
            
            num_policies = len(policies_dict)
            logger.info(f"[{mode}] Received policy and memory ({len(memory)} items) from {worker_id} (total: {num_policies} workers)")
            return {"success": True, "updated": True}
    
    def apply_best_policies(self, is_validate: bool = False) -> Dict[str, Any]:
        """
        Apply best policies to simulation.
        
          Support train/val dual mode: is_validate=True applies to val_env.
        """
        import random
        mode = "val" if is_validate else "train"
        
        with self._lock:
            policies_dict = self.val_best_policies if is_validate else self.best_policies
            env = self.val_env if is_validate else self.env
            
            if not policies_dict:
                logger.warning(f"[{mode}] No policies to apply")
                return {"success": False, "applied": []}
            
            #   Randomly select a (policy, memory) pair
            worker_ids = list(policies_dict.keys())
            selected_worker = random.choice(worker_ids)
            selected_pair = policies_dict[selected_worker]
            
            selected_policy = selected_pair["policy"]
            selected_memory = selected_pair["memory"]
            
            logger.info(f"[{mode}] Randomly selected policy from {selected_worker} (out of {len(worker_ids)} workers)")
            
            # Apply policy to corresponding env
            applied = []
            for module_name, config in selected_policy.items():
                try:
                    if env and hasattr(env, 'enabled_controls'):
                        if module_name in env.enabled_controls:
                            env.enabled_controls[module_name]['config'] = config
                            applied.append(module_name)
                            logger.info(f"[{mode}] Applied policy for {module_name}")
                except Exception as e:
                    logger.error(f"[{mode}] Failed to apply policy for {module_name}: {e}")
            
            #   Only train saves memory to context (val doesn't accumulate memory)
            if not is_validate:
                self.context["memory"] = selected_memory.copy()
                logger.info(f"Saved memory ({len(selected_memory)} items) to context for next batch")
            
            # Clear corresponding best_policies
            if is_validate:
                self.val_best_policies = {}
            else:
                self.best_policies = {}
            
            return {"success": True, "applied": applied, "memory_items": len(selected_memory)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current master status."""
        return {
            "running": self.running,
            "current_time": self.current_time,
            "checkpoint_count": len(self.checkpoints),
            "best_policies": list(self.best_policies.keys()),
        }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the simulation."""
        self.running = False
        if self.env:
            try:
                self.env.close()
            except:
                pass
            self.env = None
        if self.val_env:
            try:
                self.val_env.close()
            except:
                pass
            self.val_env = None
        
        logger.info("DeepCityMaster stopped")
        return {"success": True}



