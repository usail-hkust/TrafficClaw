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
DeepCity Interaction for verl framework.

This module provides the interaction layer between verl's agent loop and
DeepCity's transportation optimization environment. It reuses the existing
logic from utils/llm_agent.py to parse LLM actions and execute simulations.
"""

import logging
import os
import sys
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
    from utils.llm_agent import LLMAgent  # 保留用于兼容性
    DEEPCITY_AVAILABLE = True
except ImportError:
    DEEPCITY_AVAILABLE = False
    LLMAgentManager = None
    LLMAgent = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepCityInteraction(BaseInteraction):
    """
    DeepCity transportation optimization interaction.
    
    Replicates the dialogue interaction from utils/llm_agent.py:
    - LLM outputs: ACTION: PLAN / DATA_ANALYSIS / POLICY_PLANNING / FINISH
    - Environment parses actions, executes code/simulation, returns feedback
    - Reward = improvement_count (number of successful policy improvements)
    
    Methods:
        start_interaction: Initialize simulation state for a trajectory
        generate_response: Parse LLM action, execute, return environment feedback
        finalize_interaction: Clean up simulation resources
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.max_turns = config.get("max_turns", 10)
        self.control_modules = config.get("control_modules", [
            "bus_scheduling",
            "signal_timing",
            "subway_scheduling",
            "highway_speed_limit",
        ])
        self.config_name = config.get("config_name", None)
        
        # 多 Master 架构（参考 verl_architecture.md）
        # 2 个 Master 并行运行，每个 Master 到达 checkpoint 时触发 8 个 worker 采样
        self.num_masters = config.get("num_masters", 2)
        self.master_name_prefix = config.get("master_name_prefix", "DeepCityMaster")
        self._masters = {}  # 缓存 master 连接 {master_id: actor}
        
        # Check DeepCity availability
        if not DEEPCITY_AVAILABLE:
            logger.warning("DeepCity modules not available. Some features may not work.")
    
    def _get_master_id(self, instance_id: str) -> int:
        """
        根据 instance_id 分配到对应的 Master。
        使用 hash 确保同一 instance 始终连接同一 Master。
        
        对于带 worker 后缀的 instance_id（如 "sample_0_w123456"），
        提取前缀（"sample_0"）进行 hash，确保同一样本的多个 worker 连接到同一个 Master。
        """
        # 提取前缀（去掉 _w 后缀）
        if "_w" in instance_id:
            prefix = instance_id.split("_w")[0]
        else:
            prefix = instance_id
        
        return hash(prefix) % self.num_masters
    
    def _get_or_create_master(self, instance_id: str = None):
        """
        Get or create DeepCityMaster for given instance.
        
        多 Master 架构：
        - DeepCityMaster_0, DeepCityMaster_1 各自维护独立时间轴
        - Worker 根据 hash(instance_id) % num_masters 分配到对应 Master
        - 同一 Master 的 8 个 worker 共享同一个 checkpoint
        
        如果 Master 不存在，自动创建并启动。
        """
        if instance_id is None:
            master_id = 0
        else:
            master_id = self._get_master_id(instance_id)
        
        if master_id not in self._masters:
            import ray
            master_name = f"{self.master_name_prefix}_{master_id}"
            
            try:
                # 尝试连接已存在的 Master
                self._masters[master_id] = ray.get_actor(master_name)
                logger.info(f"Connected to existing {master_name}")
            except ValueError:
                # Master 不存在，自动创建
                logger.info(f"Creating new {master_name}...")
                
                from verl.actors.deepcity_master import DeepCityMaster
                
                # 从配置获取参数
                sumo_config = self.config.get("sumo_config_path")
                checkpoint_interval = self.config.get("checkpoint_interval", 300)
                seed = self.config.get("seed", 42)
                use_gui = self.config.get("use_gui", False)
                simulation_duration = self.config.get("simulation_duration")  # 仿真时长（如 86400s）
                auto_restart = self.config.get("auto_restart", True)  # 是否自动重启
                max_episodes = self.config.get("max_episodes")  # 最大 episode 数
                
                if not sumo_config:
                    raise ValueError("sumo_config_path not found in interaction config")
                
                # 创建 Master
                master = DeepCityMaster.options(
                    name=master_name,
                    num_cpus=1
                ).remote(
                    master_id=master_id,
                    config_path=sumo_config,
                    checkpoint_interval=checkpoint_interval,
                    use_gui=use_gui,
                    seed=seed,
                    simulation_duration=simulation_duration,
                    auto_restart=auto_restart,
                    max_episodes=max_episodes,
                    control_modules=self.control_modules
                )
                
                # 启动仿真
                result = ray.get(master.start.remote())
                if not result.get("success"):
                    raise RuntimeError(f"Failed to start {master_name}: {result.get('error')}")
                
                self._masters[master_id] = master
                logger.info(f"Created and started {master_name}")
        
        return self._masters.get(master_id)
    
    def _get_default_configs(self, master) -> Dict[str, Any]:
        """
        Get default control configs from Master's env.enabled_controls.
        
        Args:
            master: Ray actor handle for DeepCityMaster
            
        Returns:
            Dictionary of default configs by module name
        """
        import ray
        import copy
        
        try:
            # Get enabled_controls from Master's env
            enabled_controls = ray.get(master.get_enabled_controls.remote())
            
            if not enabled_controls:
                logger.warning("No enabled_controls found in Master")
                return {}
            
            # Extract configs for modules that LLM agent will optimize
            current_configs = {}
            for module_name in self.control_modules:
                if module_name in enabled_controls:
                    module_info = enabled_controls[module_name]
                    config = module_info.get('config', {})
                    if config:
                        # Deep copy to avoid modifying env.enabled_controls
                        current_configs[module_name] = copy.deepcopy(config)
                        logger.info(f"Loaded default config for {module_name}: {len(config)} entries")
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
        触发 Master 运行到下一个 checkpoint，并获取 context。
        
        这是主循环的核心：
        1. 触发 Master 运行仿真（checkpoint_interval 秒）
        2. Master 自动保存 checkpoint（t 和 t-1）
        3. Master 自动收集 traffic_states
        4. 返回 context 供 Agent 使用
        
        多 Master 架构：根据 instance_id 连接对应的 Master 获取 checkpoint。
        
        返回的 context 包含（与 run_bus_scheduling_control.py 对应）：
        
        # === Checkpoint 相关 (来自 records/checkpoints/) ===
        - checkpoint_path: str - SUMO 仿真状态快照文件路径（t-1 state）
        - checkpoint_time: float - checkpoint 对应的仿真时间（秒）
        
        # === 路网图结构 (来自 simulation_utils.build_road_network_graphs()) ===
        - lane_graph: NetworkX MultiDiGraph - 车道拓扑图
        - lane_inter_graph: NetworkX DiGraph - 车道组到交叉口的连接图
        - intersection_graph: NetworkX MultiDiGraph - 交叉口之间的连接图
        - lane_group_graph: NetworkX MultiDiGraph - 车道组之间的连接图
        - road_graph: NetworkX DiGraph - 道路拓扑图
        
        # === 元数据字典 ===
        - lane_dict: Dict - 车道元数据 {lane_id: {length, speed, ...}}
        - road_dict: Dict - 道路属性 {road_id: {type, priority, numLanes, ...}}
        
        # === 交通状态数据 (来自 records/traffic_states/) ===
        - traffic_states_filepath: str - 交通状态时序数据文件路径
        
        """
        master = self._get_or_create_master(instance_id)
        if not master:
            raise RuntimeError("Failed to get or create master")
        
        try:
            import ray
            
            #   核心：触发 Master 运行到下一个 checkpoint
            # 这相当于原项目的 run_controlled_simulation()
            checkpoint_interval = self.config.get("checkpoint_interval", 300)
            step_result = ray.get(master.step.remote(checkpoint_interval))
            
            if not step_result.get("success"):
                raise RuntimeError(f"Master step failed: {step_result.get('error')}")
            
            logger.info(f"Master stepped to time {step_result.get('current_time')}s")
            
            #   获取 context（包含 checkpoint 路径、traffic_states 等）
            ctx = ray.get(master.get_context.remote())
            
            # 添加 master_id 信息
            if instance_id:
                ctx["master_id"] = self._get_master_id(instance_id)
            
            # 添加 checkpoint_interval 供 Agent 使用
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
        **kwargs
    ) -> str:
        """
        Initialize a new interaction instance.
        
          主循环核心：每次调用都会触发 Master 运行到下一个 checkpoint。
        
        多 Master 架构：
        - Worker 根据 instance_id 分配到对应 Master
        - 每次调用触发 Master.step(checkpoint_interval)
        - Master 运行仿真、保存 checkpoint、收集 traffic_states
        - 返回 context 供 Agent 做 10 轮对话
        
        这相当于原项目的：
        ```python
        while remaining_duration > 0:
            results = run_controlled_simulation(env, duration=checkpoint_interval)
            checkpoint_path = results["checkpoint_path"]
            agent.run_optimization(context)
        ```
        
        Args:
            instance_id: Unique identifier for this interaction
            checkpoint_path: Path to SUMO checkpoint (optional, 优先于 Master 提供的)
            
        Returns:
            instance_id: The assigned instance ID
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # 获取或创建 Master
        master = self._get_or_create_master(instance_id)
        master_id = self._get_master_id(instance_id) if instance_id else 0
        
        import ray
        
        #   批次级 checkpoint 复用机制
        # 检查当前批次的 checkpoint 是否已生成
        checkpoint_ready = ray.get(master.is_checkpoint_ready.remote())
        
        if checkpoint_ready:
            # 复用当前批次的 checkpoint（后续 worker）
            latest_checkpoint = ray.get(master.get_latest_checkpoint.remote())
            checkpoint_time = latest_checkpoint.get('time') if latest_checkpoint else 0
            logger.info(f"Worker reusing checkpoint at time {checkpoint_time}s (Master {master_id})")
            master_context = ray.get(master.get_context.remote())
        else:
            # 第一个 worker：触发 Master 运行到下一个 checkpoint
            logger.info(f"First worker triggering Master {master_id} to step...")
            master_context = self._step_master_and_get_context(instance_id)
            # 标记 checkpoint 已就绪，后续 worker 可以复用
            ray.get(master.mark_checkpoint_ready.remote())
        
        # 注册 worker（用于追踪何时所有 worker 完成）
        ray.get(master.register_worker.remote())
        
        # Use provided checkpoint_path or get from master
        effective_checkpoint = checkpoint_path or master_context.get("checkpoint_path")
        
        # Create LLMAgent instance to reuse its methods (same as run_bus_scheduling_control.py)
        # Use mock.patch to skip LLM initialization without modifying llm_agent.py
        llm_agent = None
        if LLMAgent is not None:
            try:
                from unittest.mock import patch, MagicMock
                # Patch LLM class to avoid loading transformers/torchvision
                with patch('utils.llm_agent.LLM', MagicMock()):
                    llm_agent = LLMAgent(
                        model_name="placeholder",  # verl 中不用它推理，只复用其方法
                        temperature=0.7,
                        max_turns=self.max_turns,
                        available_control_modules=self.control_modules,
                    )
            except Exception as e:
                logger.warning(f"Failed to create LLMAgent: {e}. Using fallback methods.")
        
        # Initialize state (参考 llm_agent.py LLMAgent 的状态)
        self._instance_dict[instance_id] = {
            # === 轮次管理 ===
            "turn": 0,                    # 当前轮次 (对应 LLMAgent.turn_count)
            "max_turns": self.max_turns,  # 最大轮次
            
            # === 奖励计算 ===
            "improvement_count": 0,       # 改进次数 (reward)
            "all_policies": [],           # 收集所有 POLICY_PLANNING 生成的 policy（用于随机采样上报）
            
            # === 最佳结果追踪 (对应 LLMAgent 的 best_simulation_result 等) ===
            "best_simulation_result": None,   # 最佳仿真结果
            "best_control_configs": None,     # 最佳控制配置
            "best_simulation_turn": None,     # 最佳结果所在轮次
            
            # === 代码执行 ===
            "last_code_result": None,     # 上次代码执行结果 (用于 DEBUG 动作)
            
            # === 工具类 ===
            "llm_agent": llm_agent,       # LLMAgent 实例，复用 _is_better_simulation_result / _format_simulation_result
            
            # === 上下文 (用于 execute_code) ===
            "context": {
                # 与原项目 run_highway_speed_limit_control.py 的 agent_context 保持一致
                **master_context,                           # 来自 Master 的 graphs, traffic_states_filepath 等
                
                # === 控制配置 ===
                "current_configs": self._get_default_configs(master),  # 从 Master 的 env.enabled_controls 获取默认配置
                "available_control_modules": self.control_modules,
                
                # === Checkpoint 和配置 ===
                "checkpoint_path": effective_checkpoint,
                "config_path": self.config.get("sumo_config_path"),  # SUMO 配置文件路径 (用于 policy simulation)
                "checkpoint_interval": master_context.get("checkpoint_interval", 3600),
                "test_duration": master_context.get("test_duration", 3600),
                "remaining_duration": None,                 # 剩余仿真时长 (verl 中由 Master 管理，设为 None)
                
                # === 仿真参数 ===
                "simulation_id": f"master_{master_context.get('master_id', 0)}_{instance_id[:8]}",  # 仿真 ID (用于过滤数据)
                "use_gui": self.config.get("use_gui", False),  # 是否使用 SUMO GUI
                "run_duration": self.config.get("simulation_duration", 86400),  # 总仿真时长
                "seed": master_context.get("seed", 42),     # 随机种子
                
                # === 文件命名参数 ===
                "config_name": None,                        # 配置名称 (verl 中不使用)
                "llm_name": None,                           # LLM 模型名称 (verl 中不使用)
                "control_modules": self.control_modules,    # 控制模块列表
            },
        }
        
        logger.info(f"Started DeepCity interaction: {instance_id}")
        return instance_id

    async def generate_response(
        self, 
        instance_id: str, 
        messages: list[dict[str, Any]], 
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """
        Process LLM output, execute action, return environment feedback.
        
        Replicates the logic from utils/llm_agent.py run_optimization_loop().
        
        Args:
            instance_id: The interaction instance ID
            messages: Conversation history, last message is LLM output
            
        Returns:
            tuple of (should_stop, response, reward, extra_info)
        """
        
        self._ensure_imports()
        
        state = self._instance_dict[instance_id]
        state["turn"] += 1
        
        # Get LLM output (last assistant message)
        llm_output = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                llm_output = messages[i].get("content", "")
                break
        
        if not llm_output:
            return False, "No assistant message found.", 0.0, {}
        
        # 1. Parse action (reuse original function)
        action_type, action_content = parse_llm_action(llm_output)
        logger.debug(f"Parsed action: {action_type}")
        
        # Add turn count reminder (与原项目 llm_agent.py 保持一致)
        turn_reminder = ""
        if state["turn"] > 1:
            remaining_turns = state["max_turns"] - state["turn"]
            turn_reminder = f"\n[Reminder: You are on turn {state['turn']}/{state['max_turns']}. {remaining_turns} turns remaining. Please work efficiently and use FINISH when optimization is complete.]"
        
        # 2. Execute action based on type
        response = ""
        should_stop = False
        reward = 0.0
        
        if action_type == "PLAN":
            response = self._handle_plan(state, action_content)
            
        elif action_type == "GET_CONTROL_API":
            response = self._handle_get_control_api(state, action_content)
            
        elif action_type == "DATA_ANALYSIS":
            response = self._handle_data_analysis(state, action_content)
            
        elif action_type == "POLICY_PLANNING":
            response = self._handle_policy_planning(state, action_content)
            
        elif action_type == "DEBUG":
            response = self._handle_debug(state, action_content)
            
        elif action_type == "FINISH":
            # Get all optimized control_configs from agent's internal state
            # Return the BEST configs found during optimization, not the current (last) configs
            # This ensures we use the best-performing policy for the next checkpoint
            final_control_configs = state["best_control_configs"].copy() if state["best_control_configs"] else {}
            
            # Fallback: if no best_control_configs, use current configs
            if not final_control_configs:
                final_control_configs = state["context"].get("current_configs", {}).copy()
            
            # 简短确认消息（原项目不返回响应给 LLM，verl 需要返回以结束对话）
            reward = float(state["improvement_count"])
            response = f"Optimization completed. Final reward: {reward}"
            
            # Prepare extra_info with detailed results (与原项目的返回字典一致)
            extra_info = {
                "success": True,
                "final_control_configs": final_control_configs,
                "turn_count": state["turn"],
                "final_message": action_content,
                "best_simulation_result": state["best_simulation_result"],
                "best_simulation_turn": state["best_simulation_turn"],
                "improvement_count": state["improvement_count"],
            }
            
            should_stop = True
            return should_stop, response, reward, extra_info
            
        else:
            response = f"Unknown action: {action_type}. Available actions: PLAN, GET_CONTROL_API, DATA_ANALYSIS, POLICY_PLANNING, DEBUG, FINISH."
        
        # Add turn reminder (only if turn > 1, 与原项目 llm_agent.py 保持一致)
        if turn_reminder:
            response += turn_reminder
        
        # Prepare extra_info for all actions (agent_loop needs improvement_count)
        extra_info = {
            "improvement_count": state["improvement_count"],
        }
        
        # Check max turns
        if state["turn"] >= state["max_turns"]:
            reward = float(state["improvement_count"])
            response += f"\n\nMax turns reached. Final reward: {reward}"
            should_stop = True
            # Add final results to extra_info
            extra_info.update({
                "success": False,  # Max turns reached without FINISH
                "final_control_configs": state["best_control_configs"] or state["context"].get("current_configs", {}),
                "turn_count": state["turn"],
                "best_simulation_result": state["best_simulation_result"],
                "best_simulation_turn": state["best_simulation_turn"],
            })
            return should_stop, response, reward, extra_info
        
        return should_stop, response, reward, extra_info

    def _handle_plan(self, state: dict, content: str) -> str:
        """
        Handle PLAN action.
        与原项目 llm_agent.py 的处理逻辑完全一致。
        """
        # 提取计划内容（移除 ACTION: PLAN 行，与原项目一致）
        plan_content = content
        if "ACTION: PLAN" in content.upper() or "ACTION:PLAN" in content.upper():
            # Try to extract content after ACTION: PLAN
            lines = content.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if "ACTION: PLAN" in line.upper() or "ACTION:PLAN" in line.upper():
                    start_idx = i + 1
                    break
            plan_content = '\n'.join(lines[start_idx:]).strip()
        
        feedback = "Plan received and acknowledged.\n\n"
        feedback += plan_content
        feedback += "\n\nYou can now proceed with the planned actions (DATA_ANALYSIS, POLICY_PLANNING, etc.)."
        return feedback

    def _handle_get_control_api(self, state: dict, content: str) -> str:
        """
        Handle GET_CONTROL_API action - return API documentation.
        与原项目 llm_agent.py 的处理逻辑一致，直接调用 LLMAgent 的方法。
        """
        llm_agent = state.get("llm_agent")
        
        # 提取模块名（直接调用原项目的方法）
        module_name = llm_agent._extract_module_name(content) if llm_agent else None
        
        # 调用原项目的 _load_control_specs 和 _format_module_api
        if LLMAgent is not None:
            all_specs = LLMAgent._load_control_specs()
            
            if module_name and module_name in all_specs:
                spec = all_specs[module_name]
                api_info = LLMAgent._format_module_api(module_name, spec)
                
                feedback = f"API information for '{module_name}' control module:\n\n"
                feedback += api_info
                feedback += "\nYou can now use this information to write code for DATA_ANALYSIS or POLICY_PLANNING actions."
                return feedback
            else:
                available_modules = list(all_specs.keys())
                if module_name is None:
                    return f"Module name not specified. Please use format:\nACTION: GET_CONTROL_API\nModule: <module_name>\n\nAvailable modules: {available_modules}"
                else:
                    return f"Unknown module '{module_name}'. Available modules: {available_modules}"
        else:
            return "LLMAgent not available. Cannot load control specs."

    def _handle_data_analysis(self, state: dict, content: str) -> str:
        """
        Handle DATA_ANALYSIS action - execute analysis code.
        与原项目 llm_agent.py 的处理逻辑一致。
        """
        # parse_llm_action 对 DATA_ANALYSIS 已提取代码，content 就是纯代码
        # 直接使用 content（与原项目一致）
        result = execute_code(content, state["context"], verbose=False)
        state["last_code_result"] = result  # Store for potential DEBUG
        
        # 直接调用 LLMAgent._format_code_result（与原项目 _handle_data_analysis_result 一致）
        llm_agent = state.get("llm_agent")
        return llm_agent._format_code_result(result, check_control_config=False)

    def _handle_policy_planning(self, state: dict, content: str) -> str:
        """
        Handle POLICY_PLANNING action - execute code and run simulation.
        与原项目 llm_agent.py 的处理逻辑一致。
        """
        try:
            llm_agent = state.get("llm_agent")
            
            # 提取指定的控制模块（直接调用原项目的方法）
            specified_modules = llm_agent._extract_control_modules(content) if llm_agent else []
            if not specified_modules:
                available = state["context"].get("available_control_modules", [])
                return f"POLICY_PLANNING action must specify 'Control Modules: <module1>, <module2>' line.\nAvailable modules: {available}"
            
            # 提取代码块
            code = extract_code_block(content) if content else None
            if not code:
                return "POLICY_PLANNING action must contain code block. Please provide code."
            
            # Execute code to generate config using DeepCity's code_sandbox
            result = execute_code(code, state["context"], verbose=False)
            state["last_code_result"] = result  # Store for DEBUG
            
            # 调用适配后的 _handle_policy_planning_result 进行配置验证和合并
            env = state["context"].get("env")  # SUMOEnv 实例（如果有）
            planning_success, planning_feedback = self._handle_policy_planning_result(
                llm_agent=llm_agent,
                result=result,
                specified_modules=specified_modules,
                env=env,
                context=state["context"],
                verbose=False
            )
            
            if not planning_success:
                # 配置验证失败，返回反馈信息
                return planning_feedback
            
            # _handle_policy_planning_result 已更新 context["current_configs"]
            control_configs = state["context"].get("current_configs", {})
            
            if control_configs is None:
                control_configs = {}
            
            if control_configs:
                # Pass available_control_modules to context for filtering module_metrics
                state["context"]["available_control_modules"] = llm_agent.available_control_modules if llm_agent else []
                # Run simulation using DeepCity's execute_simulation_action
                sim_result = execute_simulation_action(
                    control_configs=control_configs,
                    context=state["context"],
                    verbose=False
                )
            else:
                # 与原项目 llm_agent.py 保持一致：发送警告但不停止执行
                warning_msg = "Warning: POLICY_PLANNING succeeded but no control_configs available for simulation. Please check your policy code."
                return warning_msg
            
            # 收集这个 policy（用于随机采样上报给 Master）
            state["all_policies"].append({
                "config": control_configs.copy(),
                "result": sim_result,
                "turn": state["turn"],
            })
            
            # Save best_simulation_result BEFORE comparison and update (for accurate comparison in feedback)
            import copy
            best_result_before_update = None
            if state["best_simulation_result"] is not None:
                best_result_before_update = copy.deepcopy(state["best_simulation_result"])
            
            # Update best simulation result and configs per module if any module is better
            improved_modules = []
            module_comparison = {}
            if sim_result.get("success"):
                module_comparison = llm_agent._is_better_simulation_result(sim_result, state["best_simulation_result"]) if llm_agent else {}
                
                # Initialize best_simulation_result if it doesn't exist
                if state["best_simulation_result"] is None:
                    state["best_simulation_result"] = {
                        "success": True,
                        "stats": {},
                        "module_metrics": {}
                    }
                
                # Update best configs and metrics for each module that improved
                for module_name, is_better in module_comparison.items():
                    if is_better and module_name != "general":
                        # Save best config for this module
                        if module_name in control_configs:
                            if state["best_control_configs"] is None:
                                state["best_control_configs"] = {}
                            state["best_control_configs"][module_name] = control_configs[module_name].copy()
                            improved_modules.append(module_name)
                        
                        # Update best metrics for this module
                        current_module_metrics = sim_result.get("module_metrics", {}).get(module_name)
                        if current_module_metrics:
                            if "module_metrics" not in state["best_simulation_result"]:
                                state["best_simulation_result"]["module_metrics"] = {}
                            state["best_simulation_result"]["module_metrics"][module_name] = current_module_metrics.copy()
                
                # Update general stats if general is better
                if module_comparison.get("general", False):
                    if "stats" not in state["best_simulation_result"]:
                        state["best_simulation_result"]["stats"] = {}
                    state["best_simulation_result"]["stats"].update(sim_result.get("stats", {}).copy())
                
                # Update best_simulation_turn if any module improved
                if improved_modules or module_comparison.get("general", False):
                    state["best_simulation_turn"] = state["turn"]
                    state["improvement_count"] += 1  # 记录改进次数 (用于 reward)
                else:
                    # Current result is worse for all modules - rollback to best configs
                    if state["best_control_configs"]:
                        # Rollback: restore best configs
                        llm_agent.current_control_configs = state["best_control_configs"].copy()
                        # Update context with best configs
                        state["context"]["current_configs"] = state["best_control_configs"].copy()
            
            # Provide feedback to LLM with comparison to best result (use best_result_before_update for accurate comparison)
            feedback = llm_agent._format_simulation_result(sim_result, compare_to_best=True, best_result=best_result_before_update)
            feedback = "[Auto-executed after POLICY_PLANNING]\n\n" + feedback
            
            # Add notification if any module achieved new best result
            if sim_result.get("success"):
                # Use the comparison result from before update
                improved_modules = [name for name, is_better in module_comparison.items() if is_better and name != "general"]
                general_better = module_comparison.get("general", False)
                
                if improved_modules or general_better:
                    feedback += f"\n\n🎉 **New Best Result Achieved**: "
                    if improved_modules:
                        feedback += f"Modules {', '.join(improved_modules)} achieved better performance"
                        if general_better:
                            feedback += " and general metrics improved"
                        feedback += f" in turn {state['turn']}!"
                    elif general_better:
                        feedback += f"General metrics improved in turn {state['turn']}!"
                    feedback += f"\nYou can continue optimizing based on this round's results to further improve performance."
            
            # Add rollback notification if rollback occurred
            if sim_result.get("success"):
                any_better = any(module_comparison.values())
                if not any_better and state["best_control_configs"]:
                    feedback += f"\n\n⚠️ **Policy Rollback**: The current policy performed worse than the best policy (from turn {state['best_simulation_turn']})."
                    feedback += f"\nThe system has automatically rolled back to the best-performing configuration."
                    feedback += f"\nYour current working configuration is now the best one from turn {state['best_simulation_turn']}."
                    feedback += f"\nYou can continue refining from this best configuration."
            
            return feedback
            
        except Exception as e:
            return f"Error processing POLICY_PLANNING: {str(e)}"

    def _handle_debug(self, state: dict, content: str) -> str:
        """
        Handle DEBUG action - re-execute fixed code.
        与原项目 llm_agent.py 的处理逻辑一致。
        """
        # 检查是否有上次代码执行结果
        if state.get("last_code_result") is None:
            return "No previous code execution found. Use ACTION: DATA_ANALYSIS or ACTION: POLICY_PLANNING action first."
        
        llm_agent = state.get("llm_agent")
        
        # 提取 debug 类型（直接调用原项目的方法）
        debug_type = llm_agent._extract_debug_type(content) if llm_agent else None
        
        if not debug_type:
            return "DEBUG action must specify 'Debug Type: DATA_ANALYSIS' or 'Debug Type: POLICY_PLANNING'."
        
        # 提取代码块
        code = extract_code_block(content) if content else None
        if not code:
            return "DEBUG action must contain code block. Please provide fixed code."
        
        # Execute debug code
        result = execute_code(code, state["context"], verbose=False)
        state["last_code_result"] = result  # Update last result
        
        # Handle based on debug type using handler methods
        if debug_type == "DATA_ANALYSIS":
            # 直接调用 LLMAgent._format_code_result（与原项目 _handle_data_analysis_result 一致）
            return llm_agent._format_code_result(result, check_control_config=False)
        
        elif debug_type == "POLICY_PLANNING":
            # Extract specified control modules
            specified_modules = llm_agent._extract_control_modules(content) if llm_agent else []
            
            if not specified_modules:
                return "DEBUG with POLICY_PLANNING must specify 'Control Modules: <module1>, <module2>' line."
            
            # 调用适配后的 _handle_policy_planning_result 进行配置验证和合并
            env = state["context"].get("env")  # SUMOEnv 实例（如果有）
            planning_success, planning_feedback = self._handle_policy_planning_result(
                llm_agent=llm_agent,
                result=result,
                specified_modules=specified_modules,
                env=env,
                context=state["context"],
                verbose=False
            )
            
            if not planning_success:
                # 配置验证失败，返回反馈信息
                return planning_feedback
            
            # Automatically execute simulation after successful DEBUG POLICY_PLANNING (same as normal POLICY_PLANNING)
            control_configs = state["context"].get("current_configs", {})
            
            if control_configs is None:
                control_configs = {}
            
            if control_configs:
                # Pass available_control_modules to context for filtering module_metrics
                state["context"]["available_control_modules"] = llm_agent.available_control_modules if llm_agent else []
                # Execute simulation
                sim_result = execute_simulation_action(
                    control_configs=control_configs,
                    context=state["context"],
                    verbose=False
                )
                
                # 收集这个 policy（用于随机采样上报给 Master）
                state["all_policies"].append({
                    "config": control_configs.copy(),
                    "result": sim_result,
                    "turn": state["turn"],
                })
                
                # Save best_simulation_result BEFORE comparison and update (for accurate comparison in feedback)
                import copy
                best_result_before_update_debug = None
                if state["best_simulation_result"] is not None:
                    best_result_before_update_debug = copy.deepcopy(state["best_simulation_result"])
                
                # Update best simulation result and configs per module if any module is better
                improved_modules = []
                module_comparison = {}
                if sim_result.get("success"):
                    module_comparison = llm_agent._is_better_simulation_result(sim_result, state["best_simulation_result"]) if llm_agent else {}
                    
                    # Initialize best_simulation_result if it doesn't exist
                    if state["best_simulation_result"] is None:
                        state["best_simulation_result"] = {
                            "success": True,
                            "stats": {},
                            "module_metrics": {}
                        }
                    
                    # Update best configs and metrics for each module that improved
                    for module_name, is_better in module_comparison.items():
                        if is_better and module_name != "general":
                            # Save best config for this module
                            if module_name in control_configs:
                                if state["best_control_configs"] is None:
                                    state["best_control_configs"] = {}
                                state["best_control_configs"][module_name] = control_configs[module_name].copy()
                                improved_modules.append(module_name)
                            
                            # Update best metrics for this module
                            current_module_metrics = sim_result.get("module_metrics", {}).get(module_name)
                            if current_module_metrics:
                                if "module_metrics" not in state["best_simulation_result"]:
                                    state["best_simulation_result"]["module_metrics"] = {}
                                state["best_simulation_result"]["module_metrics"][module_name] = current_module_metrics.copy()
                    
                    # Update general stats if general is better
                    if module_comparison.get("general", False):
                        if "stats" not in state["best_simulation_result"]:
                            state["best_simulation_result"]["stats"] = {}
                        state["best_simulation_result"]["stats"].update(sim_result.get("stats", {}).copy())
                    
                    # Update best_simulation_turn if any module improved
                    if improved_modules or module_comparison.get("general", False):
                        state["best_simulation_turn"] = state["turn"]
                    else:
                        # Current result is worse for all modules - rollback to best configs
                        if state["best_control_configs"]:
                            # Rollback: restore best configs
                            if llm_agent:
                                llm_agent.current_control_configs = state["best_control_configs"].copy()
                            # Update context with best configs
                            state["context"]["current_configs"] = state["best_control_configs"].copy()
                
                # Provide feedback to LLM with comparison to best result (use best_result_before_update_debug for accurate comparison)
                feedback = llm_agent._format_simulation_result(sim_result, compare_to_best=True, best_result=best_result_before_update_debug)
                feedback = "[Auto-executed after DEBUG POLICY_PLANNING]\n\n" + feedback
                
                # Add notification if any module achieved new best result
                if sim_result.get("success"):
                    # Use the comparison result from before update
                    improved_modules = [name for name, is_better in module_comparison.items() if is_better and name != "general"]
                    general_better = module_comparison.get("general", False)
                    
                    if improved_modules or general_better:
                        feedback += f"\n\n🎉 **New Best Result Achieved**: "
                        if improved_modules:
                            feedback += f"Modules {', '.join(improved_modules)} achieved better performance"
                            if general_better:
                                feedback += " and general metrics improved"
                            feedback += f" in turn {state['turn']}!"
                        elif general_better:
                            feedback += f"General metrics improved in turn {state['turn']}!"
                        feedback += f"\nYou can continue optimizing based on this round's results to further improve performance."
                
                # Add rollback notification if rollback occurred
                if sim_result.get("success"):
                    any_better = any(module_comparison.values())
                    if not any_better and state["best_control_configs"]:
                        feedback += f"\n\n⚠️ **Policy Rollback**: The current policy performed worse than the best policy (from turn {state['best_simulation_turn']})."
                        feedback += f"\nThe system has automatically rolled back to the best-performing configuration."
                        feedback += f"\nYour current working configuration is now the best one from turn {state['best_simulation_turn']}."
                        feedback += f"\nYou can continue refining from this best configuration."
                
                return feedback
            else:
                # 与原项目 llm_agent.py 保持一致：发送警告但不停止执行
                warning_msg = "Warning: DEBUG POLICY_PLANNING succeeded but no control_configs available for simulation. Please check your policy code."
                return warning_msg
        
        else:
            return f"Unknown debug type: {debug_type}"

    def _is_better_result(self, state: dict, current: dict, prev: Optional[dict]) -> bool:
        """
        Compare simulation results using LLMAgent's method.
        
        Args:
            state: Instance state containing llm_agent
            current: Current simulation result
            prev: Previous simulation result
        """
        llm_agent = state.get("llm_agent")
        if llm_agent is not None:
            # 使用原项目的比较逻辑
            comparison = llm_agent._is_better_simulation_result(current, prev)
            # 如果任何模块有改进，返回 True
            return any(comparison.values()) if comparison else False
        
        # Fallback: simple comparison
        if prev is None:
            return True
        return current.get("reward", 0) > prev.get("reward", 0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up interaction resources and report sampled policy to Master."""
        if instance_id in self._instance_dict:
            state = self._instance_dict[instance_id]
            
            # 随机采样一个 policy 上报给 Master（GRPO 探索策略）
            if state["all_policies"]:
                import random
                sampled = random.choice(state["all_policies"])
                sampled_config = sampled["config"]
                sampled_result = sampled["result"]
                
                # 连接到对应的 Master（多 Master 架构）
                master = self._get_master(instance_id)
                master_id = self._get_master_id(instance_id)
                if master:
                    try:
                        import ray
                        for module_name, config in sampled_config.items():
                            metrics = sampled_result.get("module_metrics", {}).get(module_name, {})
                            ray.get(master.report_best_policy.remote(module_name, config, metrics))
                            logger.info(f"Reported sampled policy for {module_name} to Master_{master_id} (turn {sampled['turn']})")
                        
                        #   注销 worker，最后一个 worker 完成时自动清除 checkpoint flag
                        is_last = ray.get(master.unregister_worker.remote())
                        if is_last:
                            logger.info(f"Last worker completed for Master_{master_id}, checkpoint flag cleared for next batch")
                    except Exception as e:
                        logger.warning(f"Failed to report sampled policy or unregister worker: {e}")
            else:
                # 没有 policy 也要注销 worker
                master = self._get_or_create_master(instance_id)
                if master:
                    try:
                        import ray
                        is_last = ray.get(master.unregister_worker.remote())
                        if is_last:
                            master_id = self._get_master_id(instance_id)
                            logger.info(f"Last worker completed for Master_{master_id}, checkpoint flag cleared for next batch")
                    except Exception as e:
                        logger.warning(f"Failed to unregister worker: {e}")
            
            logger.info(f"Finalizing DeepCity interaction: {instance_id}, improvements: {state['improvement_count']}, policies: {len(state['all_policies'])}")
            del self._instance_dict[instance_id]




    def _handle_policy_planning_result(
        self,
        llm_agent,
        result: Dict[str, Any],
        specified_modules: List[str],
        env: Optional[Any],
        context: Dict[str, Any],
        verbose: bool = True
    ) -> Tuple[bool, str]:
        """
        Handle POLICY_PLANNING action result.
        Validates, merges, and updates control module configurations.
        适配自原项目 llm_agent.py 的 _handle_policy_planning_result。
        
        Args:
            llm_agent: LLMAgent instance (for _format_code_result)
            result: Code execution result dictionary
            specified_modules: List of module names specified by LLM
            env: Optional SUMOEnv instance to access enabled control modules
            context: Context dictionary to update with current_configs
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (success: bool, feedback: str)
        """
        # Check if code returned configurations
        if result.get("success") and result.get("return_value"):
            returned_value = result["return_value"]
            
            # Check if it's a multi-module configuration dict
            if isinstance(returned_value, dict):
                # If a single module was specified and the LLM returned a flat config dict,
                # wrap it under the module name to avoid "not specified in Control Modules" warnings.
                if len(specified_modules) == 1:
                    module_name = specified_modules[0]
                    returned_keys = set(returned_value.keys())
                    available_modules = set(llm_agent.available_control_modules or [])
                    # 如果模块名不在返回键中，且返回键中没有任何已知模块名，则需要包装
                    if module_name not in returned_keys and not (returned_keys & available_modules):
                        returned_value = {module_name: returned_value}
                        result["return_value"] = returned_value
                        logger.debug(f"Wrapped single-module config under '{module_name}'")

                updated_modules = []
                validation_errors = []
                
                # Get current control configs from environment if available
                current_control_configs = {}
                if env and hasattr(env, 'enabled_controls'):
                    for module_name, module_info in env.enabled_controls.items():
                        module = module_info.get('module')
                        config = module_info.get('config', {})
                        if module and config and module_name in (llm_agent.available_control_modules or []):
                            # module_name in enabled_controls is already the correct name (e.g., 'signal_timing')
                            current_control_configs[module_name] = {
                                'module': module,
                                'config': config
                            }
                
                # Only handle modules that were specified by LLM
                for module_name, policy_config in returned_value.items():
                    # Check if this module was specified by LLM
                    if module_name not in specified_modules:
                        if verbose:
                            print(f"Warning: {module_name} was returned but not specified in Control Modules, skipping...")
                        continue
                    
                    if module_name not in current_control_configs:
                        # Module not enabled in environment, skip or warn
                        if verbose:
                            print(f"Warning: {module_name} is not enabled in the environment, skipping...")
                        continue
                    
                    module_info = current_control_configs[module_name]
                    module = module_info['module']
                    current_config = module_info['config'].copy()  # Copy to avoid modifying original
                    
                    if module is None:
                        validation_errors.append(f"{module_name}: Module not available")
                        continue
                    
                    # Get reference config for completeness check
                    reference_config = None
                    if module_name == 'signal_timing' and 'signal_timing' in current_control_configs:
                        reference_config = current_control_configs['signal_timing'].get('config', {})
                    elif module_name == 'highway_speed_limit' and 'highway_speed_limit' in current_control_configs:
                        reference_config = current_control_configs['highway_speed_limit'].get('config', {})
                    elif module_name == 'ramp_metering' and 'ramp_metering' in current_control_configs:
                        reference_config = current_control_configs['ramp_metering'].get('config', {})
                    elif module_name == 'bus_scheduling' and 'bus_scheduling' in current_control_configs:
                        reference_config = current_control_configs['bus_scheduling'].get('config', {})
                    elif module_name == 'subway_scheduling' and 'subway_scheduling' in current_control_configs:
                        reference_config = current_control_configs['subway_scheduling'].get('config', {})
                    
                    # Validate policy config using module's validate_config
                    is_valid, error_msg = module.validate_config(policy_config, reference_config=reference_config)
                    if not is_valid:
                        error_detail = f": {error_msg}" if error_msg else ""
                        validation_errors.append(f"{module_name}: Invalid configuration format{error_detail}")
                        if verbose:
                            print(f"Warning: Invalid {module_name} configuration format{error_detail}")
                        continue
                    
                    # Update current config with policy config (LLM policy may be partial)
                    current_config.update(policy_config)
                    
                    # Store in agent's internal control_configs (LLM optimized configs)
                    llm_agent.current_control_configs[module_name] = current_config
                    
                    # Update context with current_configs for simulation
                    if "current_configs" not in context:
                        context["current_configs"] = {}
                    context["current_configs"][module_name] = current_config
                    
                    updated_modules.append(f"{module_name} ({len(policy_config)} entries updated)")
                
                if updated_modules:
                    logger.info(f"Configuration updated: {', '.join(updated_modules)}")
                    feedback = llm_agent._format_code_result(result, check_control_config=True) if llm_agent else str(result)
                    feedback += f"\n\nConfiguration updated for: {', '.join(updated_modules)}"
                    if validation_errors:
                        feedback += f"\n\nValidation errors: {'; '.join(validation_errors)}"
                    feedback += "\n[Note: Simulation will be automatically executed to test this configuration.]"
                    return True, feedback
                else:
                    # Code executed but didn't return valid config
                    feedback = llm_agent._format_code_result(result, check_control_config=True) if llm_agent else str(result)
                    feedback += "\n\nWARNING: POLICY_PLANNING action should return config dict with module names as keys."
                    feedback += "\nExpected format: {\"signal_timing\": {...}} or {\"subway_scheduling\": {...}}"
                    if validation_errors:
                        feedback += f"\n\nValidation errors: {'; '.join(validation_errors)}"
                    return False, feedback
            else:
                # Code executed but didn't return dict
                feedback = llm_agent._format_code_result(result, check_control_config=True) if llm_agent else str(result)
                feedback += "\n\nWARNING: POLICY_PLANNING action should return a configuration dictionary."
                return False, feedback
        else:
            # Code execution failed
            feedback = llm_agent._format_code_result(result) if llm_agent else str(result)
            return False, feedback
