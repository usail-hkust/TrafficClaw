
"""
DeepCity Interaction using HTTP client for dual-server architecture.

This module provides the interaction layer between verl's agent loop and
DeepCity's transportation optimization environment via HTTP API.

Architecture:
- Server A (SUMO Server): Runs SUMO + LLMAgentManager + FastAPI REST Server
- Server B (Agent Loop Server): Runs verl PPO + LLM inference, uses this interaction

The interaction communicates with the SUMO server via HTTP to:
- Initialize workers (reset)
- Process LLM responses (step)
- Finalize workers and collect best policies 
"""

import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

# Import HTTP client
try:
    from verl.utils.deepcity_http_client import DeepCityHttpClient, DeepCityHttpClientError
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False
    DeepCityHttpClient = None
    DeepCityHttpClientError = Exception

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepCityInteractionHttp(BaseInteraction):
    """
    DeepCity transportation optimization interaction using HTTP.  

    This class provides the same interface as DeepCityInteraction but uses
    HTTP to communicate with a remote SUMO server instead of Ray actors.

    Methods:
        start_interaction: Initialize worker via HTTP and get initial messages
        get_messages: Return complete messages from the server
        generate_response: Process LLM response via HTTP
        finalize_interaction: Finalize worker and report best policy
    """

    def __init__(self, config: dict):
        """
        Initialize HTTP-based DeepCity interaction.

        Args:
            config: Configuration dictionary containing:
                - sumo_server_url: URL of the SUMO server (required)
                - num_masters: Number of master sessions (default: 2)
                - master_name_prefix: Prefix for master IDs (default: "master")
                - max_turns: Maximum dialogue turns (default: 10)
                - timeout: HTTP request timeout in seconds (default: 300)
                - Other SUMO/environment configuration
        """
        super().__init__(config)

        if not HTTP_CLIENT_AVAILABLE:
            raise ImportError(
                "DeepCityHttpClient not available. "
                "Please check verl/utils/deepcity_http_client.py"
            )

        # HTTP client configuration
        self.sumo_server_url = config.get("sumo_server_url", "http://localhost:8011")
        self.timeout = config.get("timeout", 300.0)

        # Create HTTP client
        self.http_client = DeepCityHttpClient(
            base_url=self.sumo_server_url,
            timeout=self.timeout
        )

        # Multi-master configuration
        self.num_masters = config.get("num_masters", 2)
        self.master_name_prefix = config.get("master_name_prefix", "master")
        self._masters = {}  # 缓存已创建的 Master {master_id: {"created_at": timestamp, "control_modules": [...]}}（参考 Ray 版本）

        # Instance tracking: instance_id -> {master_id, worker_id, messages, ...}
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # Max turns for dialogue
        self.max_turns = config.get("max_turns", 10)

        # Environment configuration (for master creation)
        # Support both direct config_path and environments list
        self.config_path = config.get("config_path", "")
        environment_mode = config.get("environment_mode", "fixed")
        if environment_mode == "random":
            self.environments = self._generate_random_environments(config)
        else:
            self.environments = config.get("environments", [])

        # If config_path not set directly, try to get from environments
        # Support both "config_path" and "sumo_config_path" keys for compatibility
        if not self.config_path and self.environments:
            env_config = self.environments[0]
            self.config_path = env_config.get("config_path", "") or env_config.get("sumo_config_path", "")

        # Validate config_path
        if not self.config_path:
            logger.warning("No config_path specified in DeepCityInteractionHttp config!")
        elif not self.config_path.endswith(".sumocfg"):
            logger.warning(f"config_path should be a .sumocfg file, got: {self.config_path}")

        self.checkpoint_interval = config.get("checkpoint_interval", 1800)
        self.seed = config.get("seed", 42)
        self.seed_increment = config.get("seed_increment", 1000)  # 每次 restart 的 seed 增量
        self.use_gui = config.get("use_gui", False)
        self.simulation_duration = config.get("simulation_duration", 7200)
        self.auto_restart = config.get("auto_restart", True)  # 是否自动重启
        self.max_episodes = config.get("max_episodes")  # 最大 episode 数（None 表示无限）

        # Control modules - support both top-level and nested in environments
        self.control_modules = config.get("control_modules", [])
        if not self.control_modules and self.environments:
            env_config = self.environments[0]
            self.control_modules = env_config.get("control_modules", [])

        #   各模块的关键指标配置（用于计算奖励）
        self.module_key_metrics = {
            "signal_timing": [
                {"name": "avg_travel_time", "direction": "lower"},
                {"name": "avg_waiting_time", "direction": "lower"}
            ],
            "bus_scheduling": [
                {"name": "avg_passenger_waiting_time", "direction": "lower"},
                {"name": "total_fuel_consumption_g", "direction": "lower"}
            ],
            "highway_speed_limit": [
                {"name": "avg_travel_time", "direction": "lower"},
                {"name": "throughput", "direction": "higher"}
            ],
            "taxi_scheduling": [
                {"name": "avg_wait_time", "direction": "lower"},
                {"name": "total_income", "direction": "higher"},
                {"name": "avg_income_per_taxi", "direction": "higher"}
            ]
        }

        logger.info(
            f"DeepCityInteractionHttp initialized: "
            f"server={self.sumo_server_url}, num_masters={self.num_masters}, "
            f"control_modules={self.control_modules}"
        )

    def _generate_random_environments(self, config: dict) -> list:
        """
        从 random_environment_pool 配置中枚举所有可能的环境组合。
        组合 = 每个地图 × 模块的所有非空子集（受 min/max_modules 约束）。
        
        例如 2 maps × 4 modules → 2 × (C(4,1)+C(4,2)+C(4,3)+C(4,4)) = 2 × 15 = 30 个环境。
        Master 会从这个完整池中按"运行次数最少优先 + 随机"策略轮选。
        
        若 control_modules 包含 highway_speed_limit，则选用 sumo_config_highway 路网。
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
        # 先决定模块组合，有 highway_speed_limit 才用 highway 地图
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
    
    def _get_master_id(self, instance_id: str, control_modules: List[str] = None) -> str:
        """
        Get master ID for an instance using numeric index (与 Ray 版本一致).

        根据 instance_id 分配到对应的 Master。
        使用 instance_id 中的数字进行确定性分配，确保同一 sample 的所有 rollout 分配到同一 Master。
        
        支持的 instance_id 格式：
        - "sample_0_w123456" → 提取 "sample_0" → 数字 0 → Master 0
        - "sample_1_w789012" → 提取 "sample_1" → 数字 1 → Master 1
        - "batch_0_w123456" → 提取 "batch_0" → 数字 0 → Master 0（兼容旧格式）
        
        分配规则：
        - sample_0 / batch_0 → 0 % num_masters → Master 0
        - sample_1 / batch_1 → 1 % num_masters → Master 1
        - sample_2 / batch_2 → 2 % num_masters → Master 0 (如果 num_masters=2)

        Args:
            instance_id: The instance ID
            control_modules: List of control modules (unused, for compatibility)

        Returns:
            Master ID string (e.g., "0", "1", "2", "3")
        """
        # 提取前缀（去掉 _w 后缀）
        if "_w" in instance_id:
            prefix = instance_id.split("_w")[0]
        else:
            prefix = instance_id
        
        # 提取数字部分进行确定性分配
        # batch_0 → 0, batch_1 → 1, batch_2 → 2, ...
        if '_' in prefix:
            try:
                num = int(prefix.split('_')[-1])
                master_index = num % max(1, self.num_masters)
                return f"{master_index}"
            except ValueError:
                pass
        
        # 回退到 hash（用于非标准格式的 instance_id）
        import hashlib
        hash_value = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
        master_index = hash_value % max(1, self.num_masters)
        return f"{master_index}"

    async def _ensure_master_exists(self, master_id: str, control_modules: List[str] = None) -> bool:
        """
        Ensure master session exists on the server (异步版本，支持并发).
        
        参考 Ray 版本的重试机制，处理并发创建时的竞态条件。

        Args:
            master_id: The master ID to check/create
            control_modules: Control modules for this master (used when creating)

        Returns True if master exists or was created successfully.
        """
        import random
        
        #   1. 检查本地缓存（Fast Path，参考 Ray 版本）
        if master_id in self._masters:
            logger.debug(f"Master {master_id} found in local cache")
            return True
        
        # Use provided control_modules or fall back to default
        modules_to_use = control_modules if control_modules else self.control_modules
        
        #   2. 重试机制（参考 Ray 版本）
        max_retries = 10
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # 尝试 A: 获取已存在的 Master
                await asyncio.to_thread(self.http_client.get_session, master_id)
                logger.info(f"Connected to existing master {master_id} (attempt {attempt + 1})")
                #   缓存到本地（存储元信息）
                import time
                self._masters[master_id] = {
                    "created_at": time.time(),
                    "control_modules": modules_to_use
                }
                return True
            except DeepCityHttpClientError as e:
                if "404" not in str(e):
                    logger.error(f"Error checking master {master_id} (attempt {attempt + 1}): {e}")
                    return False
                else:
                    logger.debug(f"Master {master_id} not found (404), will try to create...")
            
            # 尝试 B: 创建新的 Master
            try:
                logger.info(f"Master {master_id} not found, creating (attempt {attempt + 1}/{max_retries})...")
                
                #   支持多环境配置（与 Ray 版本一致）
                if self.environments:
                    # 使用 environments 列表
                    logger.info(f"Creating master {master_id} with {len(self.environments)} environment(s)")
                    environments = self.environments
                    config_path = None  # 使用 environments 时不需要单独的 config_path
                else:
                    # 兼容旧版本：单环境配置
                    config_path = self.config_path
                    if not config_path:
                        config_path = "/data/zhouyuping/Zone/zone_scenarios/Manhattan/Manhattan.sumocfg"
                        logger.warning(f"No config_path in config, using hardcoded: {config_path}")
                    environments = None

                #   计算 Master 的初始 seed（与服务端逻辑一致）
                # seed = base_seed + master_id（服务端会进一步处理 episode_count）
                master_seed = self.seed + int(master_id)
                
                await asyncio.to_thread(
                    self.http_client.create_session,
                    session_id=master_id,
                    environments=environments,  #   传递多环境列表
                    config_path=config_path,  # 兼容旧版本
                    checkpoint_interval=self.checkpoint_interval,
                    run_duration=self.simulation_duration,
                    seed=master_seed,  #   每个 Master 使用不同的 seed
                    use_gui=self.use_gui,
                    available_control_modules=modules_to_use,
                    max_turns=self.max_turns
                )
                logger.info(f"Successfully created master {master_id}")
                #   缓存到本地（存储元信息）
                import time
                self._masters[master_id] = {
                    "created_at": time.time(),
                    "control_modules": modules_to_use
                }
                return True
                
            except DeepCityHttpClientError as create_error:
                # 竞态条件：被其他 worker 抢先创建了
                if "already exists" in str(create_error).lower() or "400" in str(create_error):
                    logger.warning(f"Master {master_id} creation conflict (attempt {attempt + 1}/{max_retries}): {create_error}")
                    logger.info(f"Master {master_id} was created by another worker (race condition), will retry GET...")
                    # 不直接返回，继续重试循环，下次将成功获取
                else:
                    logger.error(f"Failed to create master {master_id} with non-retryable error: {create_error}")
                    return False
            
            # 指数退避 + 随机抖动（参考 Ray 版本）
            if attempt < max_retries - 1:
                sleep_time = base_delay * (1.5 ** attempt) + random.uniform(0, 1)
                logger.debug(f"Waiting {sleep_time:.2f}s before retry...")
                await asyncio.sleep(sleep_time)
        
        logger.error(f"Failed to ensure master {master_id} exists after {max_retries} attempts")
        return False

    def _calculate_metric_improvement_reward(
        self,
        module_name: str,
        current_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any]
    ) -> float:
        """
        计算单个模块基于指标改进幅度的奖励。
        
        Args:
            module_name: 模块名称
            current_metrics: 当前轮的指标
            best_metrics: 历史最佳指标
            
        Returns:
            改进比例的平均值 (0.0 - 1.0+)
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
            
            # 跳过缺失的指标
            if current_value is None:
                continue
            
            #   对可能为负数的指标取绝对值（如 fuel consumption）
            if metric_name in ["total_fuel_consumption_g", "fuel_consumption"]:
                current_value = abs(current_value)
                if best_value is not None:
                    best_value = abs(best_value)
            
            # 如果没有历史最佳值，当前值作为基准（改进为0）
            if best_value is None or best_value == 0:
                continue
            
            # 计算改进比例
            if direction == "lower":
                # 越小越好：improvement = (best - current) / best
                improvement = (best_value - current_value) / abs(best_value)
            else:  # direction == "higher"
                # 越大越好：improvement = (current - best) / best
                improvement = (current_value - best_value) / abs(best_value)
            
            # 限制改进比例在合理范围内（避免异常值）
            improvement = max(-1.0, min(1.0, improvement))
            improvements.append(improvement)
        
        # 返回所有关键指标改进比例的平均值
        if improvements:
            return sum(improvements) / len(improvements)
        return 0.0

    async def _step_master_if_needed(self, master_id: str) -> Dict[str, Any]:
        """
        Step master to run simulation and create checkpoint if needed (异步版本，支持并发).

        This is critical for the dual-server architecture:
        1. Check current checkpoint_index from server
        2. Compare with expected checkpoint_index for this worker
        3. If behind, call step_master to run simulation
        4. Return checkpoint info

        The server handles concurrency - multiple workers calling this
        will not cause duplicate simulations due to server-side locking.
        """
        try:
            # Get current master status (异步执行，避免阻塞)
            master_status = await asyncio.to_thread(self.http_client.get_session, master_id)
            current_checkpoint_index = master_status.get("checkpoint_index", 0)

            # For the first batch, we need checkpoint_index >= 1
            # (checkpoint_index 0 means no simulation has run yet)
            expected_checkpoint_index = 1  # TODO: track this per-batch if needed

            if current_checkpoint_index >= expected_checkpoint_index:
                logger.info(
                    f"Master {master_id} already at checkpoint {current_checkpoint_index} "
                    f"(expected >= {expected_checkpoint_index})"
                )
                return {
                    "checkpoint_index": current_checkpoint_index,
                    "checkpoint_ready": True,
                    "simulation_finished": master_status.get("simulation_finished", False)
                }

            # Need to step master to create checkpoint
            logger.info(
                f"Stepping master {master_id} from checkpoint {current_checkpoint_index} "
                f"to {expected_checkpoint_index}..."
            )
            # 异步执行 run_simulation，避免阻塞
            result = await asyncio.to_thread(
                self.http_client.run_simulation,
                session_id=master_id,
                duration=self.checkpoint_interval,
                checkpoint_interval=self.checkpoint_interval
            )

            # Handle race condition: another worker may have already stepped
            if result.get("already_ready"):
                logger.info(f"Master {master_id} was stepped by another worker (race condition)")

            logger.info(
                f"Master {master_id} stepped: checkpoint_index={result.get('checkpoint_index')}, "
                f"checkpoint_time={result.get('checkpoint_time')}"
            )
            return result

        except DeepCityHttpClientError as e:
            logger.error(f"Failed to step master {master_id}: {e}")
            raise

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        control_modules: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Initialize a new interaction instance via HTTP.

        Args:
            instance_id: Unique identifier for this interaction
            checkpoint_path: Path to SUMO checkpoint (optional, managed by server)
            control_modules: List of control modules for this instance (optional)
                            If not provided, uses default from config

        Returns:
            instance_id: The assigned instance ID
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Determine control_modules for this instance
        # Priority: explicit parameter > extract from instance_id > default from config
        instance_control_modules = control_modules

        if not instance_control_modules:
            # Try to extract from instance_id pattern: batch_{index}_w{worker_id}
            # Use batch_index to look up environment config
            if instance_id.startswith("batch_"):
                try:
                    parts = instance_id.split("_")
                    batch_index = int(parts[1])
                    if self.environments and batch_index < len(self.environments):
                        env_config = self.environments[batch_index]
                        instance_control_modules = env_config.get("control_modules", [])
                        logger.info(f"Instance {instance_id}: using control_modules from environments[{batch_index}]: {instance_control_modules}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to extract batch_index from instance_id {instance_id}: {e}")

        if not instance_control_modules:
            instance_control_modules = self.control_modules
            logger.info(f"Instance {instance_id}: using default control_modules: {instance_control_modules}")

        # Determine master and worker IDs based on control_modules
        master_id = self._get_master_id(instance_id, instance_control_modules)
        worker_id = f"worker_{instance_id}"

        logger.info(f"[HTTP] Instance {instance_id} -> Master {master_id} (control_modules={instance_control_modules})")
        logger.info(f"[HTTP] Parsed instance_id: {instance_id} -> master_id: {master_id} (num_masters={self.num_masters})")

        # Ensure master exists with correct control_modules (异步调用)
        if not await self._ensure_master_exists(master_id, instance_control_modules):
            raise RuntimeError(f"Failed to ensure master {master_id} exists")

        # CRITICAL: Step master to run simulation and create checkpoint
        # This must happen before workers can do policy planning
        # The first worker to call this will trigger the simulation,
        # subsequent workers will get the existing checkpoint
        try:
            step_result = await self._step_master_if_needed(master_id)
            logger.info(f"Master {master_id} checkpoint ready: index={step_result.get('checkpoint_index')}")
        except Exception as e:
            logger.error(f"Failed to step master {master_id}: {e}")
            raise RuntimeError(f"Failed to step master: {e}")

        # Generate initial prompt with instance-specific control_modules
        initial_prompt = self._generate_initial_prompt(instance_control_modules)

        try:
            # Reset worker on the server
            result = self.http_client.reset_worker(
                session_id=master_id,
                worker_id=worker_id,
                initial_prompt=initial_prompt,
                memory=kwargs.get("memory"),
                initial_best_result=kwargs.get("initial_best_result"),
                initial_control_configs=kwargs.get("initial_control_configs")
            )

            messages = result.get("messages", [])

            # Store instance state
            self._instance_dict[instance_id] = {
                "master_id": master_id,
                "worker_id": worker_id,
                "messages": messages,
                "turn": 0,
                "cumulative_reward": 0.0,  #   累计奖励（每轮改进比例的总和）
                "total_modules": len(instance_control_modules),  #   总模块数量
                "best_simulation_result": None,
                "best_control_configs": None,
                "best_simulation_turn": None,
                "control_modules": instance_control_modules,  # Store for reference
            }

            logger.info(
                f"Started HTTP interaction: {instance_id} "
                f"(master={master_id}, worker={worker_id}, {len(messages)} messages)"
            )
            return instance_id

        except DeepCityHttpClientError as e:
            logger.error(f"Failed to start interaction {instance_id}: {e}")
            raise RuntimeError(f"Failed to start interaction: {e}")

    def _generate_initial_prompt(self, control_modules: List[str] = None) -> str:
        """Generate initial prompt for the optimization task.

        Args:
            control_modules: List of control modules for this instance.
                            If not provided, uses default from config.
        """
        modules = control_modules if control_modules else self.control_modules
        control_modules_str = ", ".join(modules) if modules else "all available"

        return f"""You are optimizing transportation control for a city network.

Current Status:
- Checkpoint interval: {self.checkpoint_interval} seconds ({self.checkpoint_interval/3600:.2f} hours)
- Control modules: {control_modules_str}

**Optimization Time Window:**
- Your optimized policies will be applied for the next {self.checkpoint_interval} seconds
- Consider traffic patterns and time-of-day effects when optimizing

Your Task:
1. Analyze traffic data to identify optimization opportunities
2. Optimize control policies using POLICY_PLANNING action
3. Note: Simulation is automatically executed after POLICY_PLANNING - you will receive results automatically
4. Complete with FINISH action when satisfied

Begin your analysis."""

    def get_messages(self, instance_id: str) -> List[Dict[str, Any]]:
        """
        Return current messages for the instance.

        Args:
            instance_id: The interaction instance ID

        Returns:
            List of messages (system + conversation history)
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")

        state = self._instance_dict[instance_id]

        # Optionally refresh from server
        try:
            result = self.http_client.get_worker_messages(
                session_id=state["master_id"],
                worker_id=state["worker_id"]
            )
            state["messages"] = result.get("messages", state["messages"])
        except DeepCityHttpClientError as e:
            logger.warning(f"Failed to refresh messages for {instance_id}: {e}")

        return state["messages"]

    async def generate_response(
        self,
        instance_id: str,
        llm_response: str,
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        Process LLM output via HTTP, return environment feedback.

        Args:
            instance_id: The interaction instance ID
            llm_response: LLM response text (assistant message content)

        Returns:
            tuple of (should_terminate, env_feedback, reward, extra_info)
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")

        state = self._instance_dict[instance_id]

        try:
            # Call step_worker on the server
            result = self.http_client.step_worker(
                session_id=state["master_id"],
                worker_id=state["worker_id"],
                llm_response=llm_response,
                verbose=kwargs.get("verbose", False)
            )

            # Update local state
            state["messages"] = result.get("messages", [])
            state["turn"] = result.get("turn_count", state["turn"] + 1)
            
            #   每轮同步所有状态（从服务端返回的数据）
            state["best_simulation_result"] = result.get("best_simulation_result")
            state["best_simulation_turn"] = result.get("best_simulation_turn")
            state["best_control_configs"] = result.get("best_control_configs")
            # state["last_code_result"] = result.get("last_code_result")  # 如果需要的话
            # state["memory"] = result.get("memory", [])  # 如果需要的话

            action_result = result.get("action_result", {})
            action_type = action_result.get("action_type", "")
            action_data = action_result.get("action_result", {})
            total_modules = state["total_modules"]

            #   计算每轮奖励并累加（结合模块数量比例和指标改进幅度）
            if action_type in ["POLICY_PLANNING", "DEBUG"]:
                improved_modules = action_data.get("improved_modules", [])
                
                if improved_modules and total_modules > 0:
                    # 1️⃣ 基于改进模块数量的比例奖励
                    num_improved = len(improved_modules)
                    module_count_reward = float(num_improved) / float(total_modules)
                    
                    # 2️⃣ 基于指标改进幅度的奖励
                    sim_result = action_data.get("simulation_result", {})
                    metric_improvement_reward = 0.0
                    module_rewards = {}
                    
                    if sim_result and sim_result.get("success"):
                        # 获取当前轮和历史最佳的 module_metrics
                        current_module_metrics = sim_result.get("module_metrics", {})
                        best_sim_result = state.get("best_simulation_result")
                        best_module_metrics = best_sim_result.get("module_metrics", {}) if best_sim_result else {}
                        
                        # 计算每个改进模块的指标改进比例
                        for module_name in improved_modules:
                            current_metrics = current_module_metrics.get(module_name, {})
                            best_metrics = best_module_metrics.get(module_name, {})
                            
                            # 计算该模块的指标改进比例
                            module_reward = self._calculate_metric_improvement_reward(
                                module_name, current_metrics, best_metrics
                            )
                            module_rewards[module_name] = module_reward
                            metric_improvement_reward += module_reward
                    
                    # 3️⃣ 组合奖励：模块数量比例 + 指标改进幅度
                    turn_reward = module_count_reward + metric_improvement_reward
                    
                    # 累加到总奖励
                    state["cumulative_reward"] += turn_reward
                    
                    # 详细日志
                    reward_details = ", ".join([f"{m}: {r:.3f}" for m, r in module_rewards.items()]) if module_rewards else "N/A"
                    logger.info(
                        f"{action_type} improved {num_improved}/{total_modules} module(s): {', '.join(improved_modules)}, "
                        f"module_count_reward: {module_count_reward:.3f}, metric_improvement_reward: {metric_improvement_reward:.3f}, "
                        f"module_rewards: [{reward_details}], turn_reward: {turn_reward:.3f}, cumulative_reward: {state['cumulative_reward']:.3f}"
                    )

            #   不再需要在 FINISH 时单独同步状态，因为每轮都已经从服务端同步了

            # Extract environment feedback (last user message)
            env_feedback = ""
            messages = state["messages"]
            if messages and messages[-1].get("role") == "user":
                env_feedback = messages[-1]["content"]

            # Determine termination
            should_terminate = result.get("finished", False)

            # Calculate reward
            reward = 0.0
            extra_info = {}

            if should_terminate:
                #   奖励 = 累计奖励（每轮改进比例的总和）
                reward = state["cumulative_reward"]
                extra_info = {
                    "success": True,
                    "cumulative_reward": state["cumulative_reward"],
                    "total_modules": total_modules,
                    "final_control_configs": state["best_control_configs"] or {},
                    "best_simulation_result": state["best_simulation_result"],
                    "best_simulation_turn": state["best_simulation_turn"],
                    "turn_count": state["turn"]
                }
            else:
                extra_info = {
                    "cumulative_reward": state["cumulative_reward"],
                    "total_modules": total_modules
                }

            return should_terminate, env_feedback, reward, extra_info

        except DeepCityHttpClientError as e:
            logger.error(f"Failed to process response for {instance_id}: {e}")
            # Return error state
            return True, f"Error: {e}", 0.0, {"error": str(e)}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Clean up interaction resources and report best policy to server.

        Args:
            instance_id: The interaction instance ID
        """
        if instance_id not in self._instance_dict:
            logger.warning(f"Instance {instance_id} not found for finalization")
            return

        state = self._instance_dict[instance_id]

        try:
            # Finalize worker on the server
            result = self.http_client.finalize_worker(
                session_id=state["master_id"],
                worker_id=state["worker_id"]
            )

            logger.info(
                f"Finalized HTTP interaction: {instance_id}, "
                f"cumulative_reward: {state['cumulative_reward']:.3f}, "
                f"best_turn: {result.get('best_simulation_turn')}"
            )

        except DeepCityHttpClientError as e:
            logger.warning(f"Failed to finalize interaction {instance_id}: {e}")

        finally:
            # Clean up local state
            del self._instance_dict[instance_id]

    def close(self):
        """Close HTTP client and clean up resources."""
        if self.http_client:
            self.http_client.close()
            logger.info("DeepCityInteractionHttp closed")
