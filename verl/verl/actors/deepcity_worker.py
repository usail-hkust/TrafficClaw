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
DeepCityWorker Ray Actor for parallel policy simulation rollouts.

Each worker runs in an independent process to avoid GIL contention.
"""

import os
import sys
import ray
import logging
from typing import Any, Dict, List, Optional, Tuple

# Add DeepCity project root to path
DEEPCITY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if DEEPCITY_ROOT not in sys.path:
    sys.path.insert(0, DEEPCITY_ROOT)

from utils.llm_agent_manager import LLMAgentManager

logger = logging.getLogger(__name__)


@ray.remote
class DeepCityWorker:
    """
    Ray Actor for running policy simulation rollouts in parallel.
    
    Each worker is an independent process that:
    1. Maintains its own LLMAgentManager instance
    2. Executes manager.step() which includes policy simulation
    3. Runs independently without GIL contention
    
    This allows multiple workers to run policy simulations in parallel,
    each creating its own SUMO process.
    """
    
    def __init__(
        self,
        worker_id: str,
        master_id: int,
        available_control_modules: List[str],
        max_turns: int = 10,
        max_reflection_turns: int = 5,
        max_memory_items: int = 10,
        use_http: bool = False,
        sumo_server_url: Optional[str] = None,
        http_timeout: float = 300.0,
    ):
        """
        Initialize DeepCityWorker.
        
        Args:
            worker_id: Unique identifier for this worker
            master_id: ID of the master this worker belongs to
            available_control_modules: List of available control modules
            max_turns: Maximum number of optimization turns
            max_reflection_turns: Maximum number of reflection turns
            max_memory_items: Maximum number of memory items
            use_http: Whether to use HTTP mode for policy simulation
            sumo_server_url: SUMO server URL (required if use_http=True)
            http_timeout: HTTP request timeout (seconds)
        """
        self.worker_id = worker_id
        self.master_id = master_id
        self.use_http = use_http
        self.sumo_server_url = sumo_server_url
        self.http_timeout = http_timeout
        
        #   HTTP mode initialization
        self.http_client = None
        self.http_master_id = None
        self.http_worker_id = None
        
        if self.use_http:
            if not self.sumo_server_url:
                raise ValueError("sumo_server_url is required when use_http=True")
            
            # Initialize HTTP client
            from verl.utils.deepcity_http_client import DeepCityHttpClient
            self.http_client = DeepCityHttpClient(
                base_url=self.sumo_server_url,
                timeout=self.http_timeout
            )
            self.http_master_id = f"master_{master_id}"
            self.http_worker_id = worker_id
            logger.info(f"DeepCityWorker {worker_id}: HTTP mode enabled, server={sumo_server_url}")
        
        #   Auto-detect whether to enable joint control (enabled when multiple modules)
        is_joint_control = len(available_control_modules) > 1 if available_control_modules else False
        
        # Create LLMAgentManager for this worker
        self.manager = LLMAgentManager(
            available_control_modules=available_control_modules,
            max_turns=max_turns,
            max_reflection_turns=max_reflection_turns,
            max_memory_items=max_memory_items,
            is_joint_control=is_joint_control
        )
        
        logger.info(f"DeepCityWorker {worker_id} initialized for Master {master_id} "
                   f"(modules: {available_control_modules}, joint_control: {is_joint_control}, http_mode: {use_http})")
    
    def reset(
        self,
        initial_prompt: str,
        context: Dict[str, Any],
        initial_best_result: Optional[Dict[str, Any]] = None,
        initial_control_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        memory: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reset the worker's LLMAgentManager with initial prompt and context.
        
        Args:
            initial_prompt: Initial system prompt
            context: Environment context (checkpoint_path, traffic_states, etc.)
            initial_best_result: Optional initial best result
            initial_control_configs: Optional initial control configs
            memory: Optional memory items
            
        Returns:
            List of initial messages
        """
        #   HTTP mode: Reset worker via HTTP client
        if self.use_http:
            try:
                result = self.http_client.reset_worker(
                    session_id=self.http_master_id,
                    worker_id=self.http_worker_id,
                    initial_prompt=initial_prompt,
                    memory=memory,
                    initial_best_result=initial_best_result,
                    initial_control_configs=initial_control_configs
                )
                messages = result.get("messages", [])
                logger.debug(f"Worker {self.worker_id} reset via HTTP with {len(messages)} messages")
                return messages
            except Exception as e:
                logger.error(f"Worker {self.worker_id} HTTP reset failed: {e}")
                raise
        
        #   Local mode: Use LLMAgentManager directly
        # Fresh episode: clear memory unless caller passes a list (LLMAgentManager.reset keeps memory otherwise).
        if memory is not None:
            self.manager.set_memory(memory)
        else:
            self.manager.set_memory([])

        # Reset manager
        messages = self.manager.reset(
            initial_prompt=initial_prompt,
            context=context,
            verbose=False,
            initial_best_result=initial_best_result,
            initial_control_configs=initial_control_configs
        )
        
        logger.debug(f"Worker {self.worker_id} reset with {len(messages)} messages")
        return messages
    
    def step(
        self,
        llm_response: str,
        context: Dict[str, Any],
        verbose: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process LLM response and execute action (including policy simulation).
        
        This method runs in an independent process, so policy simulations
        from multiple workers can run in parallel without GIL contention.
        
        Args:
            llm_response: LLM response text
            context: Environment context snapshot
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (updated_messages, action_result)
            action_result contains additional state information for synchronization
        """
        #   HTTP mode: Step worker via HTTP client
        if self.use_http:
            try:
                result = self.http_client.step_worker(
                    session_id=self.http_master_id,
                    worker_id=self.http_worker_id,
                    llm_response=llm_response,
                    verbose=verbose
                )
                messages = result.get("messages", [])
                action_result = result.get("action_result", {})
                logger.debug(f"Worker {self.worker_id} step via HTTP, turn={action_result.get('turn_count')}")
                return messages, action_result
            except Exception as e:
                logger.error(f"Worker {self.worker_id} HTTP step failed: {e}")
                raise
        
        #   Local mode: Use LLMAgentManager directly
        # Execute manager.step() which may include policy simulation
        messages, action_result = self.manager.step(
            llm_response=llm_response,
            context=context,
            env=None,  # Policy simulation creates its own env
            verbose=verbose
        )
        
        #   Add additional state information to action_result for DeepCityInteraction synchronization
        action_result["best_simulation_result"] = self.manager.best_simulation_result
        action_result["best_simulation_turn"] = self.manager.best_simulation_turn
        action_result["best_control_configs"] = self.manager.best_control_configs
        action_result["current_control_configs"] = self.manager.current_control_configs
        action_result["memory"] = self.manager.memory.copy() if self.manager.memory else []
        action_result["last_code_result"] = self.manager.last_code_result
        
        return messages, action_result
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current worker state.
        
        Returns:
            Dictionary containing worker state
        """
        return {
            "worker_id": self.worker_id,
            "turn_count": self.manager.turn_count,
            "is_reflection_phase": self.manager.is_reflection_phase,
            "reflection_turn": self.manager.reflection_turn,
            "best_simulation_result": self.manager.best_simulation_result,
            "best_simulation_turn": self.manager.best_simulation_turn,
            "best_control_configs": self.manager.best_control_configs,
            "memory": self.manager.memory.copy() if self.manager.memory else [],
            "last_code_result": self.manager.last_code_result,
        }
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get current messages.
        
        Returns:
            List of messages
        """
        return self.manager.messages.copy()
    
    def start_reflection(
        self,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Start reflection phase.
        
        Args:
            history: List of (action_type, action_result) dicts
            
        Returns:
            List of messages for reflection
        """
        # Convert dict format to tuple format for get_reflection_message
        history_tuples = [(h.get("action_type", ""), h.get("action_result", {})) for h in history]
        return self.manager.get_reflection_message(history_tuples)
    
    def update_memory_from_reflection(
        self,
        reflection_response: str,
        verbose: bool = False,
    ) -> List[str]:
        """
        Update memory from reflection response.
        
        Args:
            reflection_response: LLM's reflection response
            verbose: Whether to print detailed logs
            
        Returns:
            Updated memory list
        """
        return self.manager.update_memory_from_reflection(
            reflection_response=reflection_response,
            verbose=verbose
        )
