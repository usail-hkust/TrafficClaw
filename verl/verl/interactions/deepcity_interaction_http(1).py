
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

import logging
import os
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
        self.sumo_server_url = config.get("sumo_server_url", "http://localhost:8000")
        self.timeout = config.get("timeout", 300.0)

        # Create HTTP client
        self.http_client = DeepCityHttpClient(
            base_url=self.sumo_server_url,
            timeout=self.timeout
        )

        # Multi-master configuration
        self.num_masters = config.get("num_masters", 2)
        self.master_name_prefix = config.get("master_name_prefix", "master")

        # Instance tracking: instance_id -> {master_id, worker_id, messages, ...}
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # Max turns for dialogue
        self.max_turns = config.get("max_turns", 10)

        # Environment configuration (for master creation)
        # Support both direct config_path and environments list
        self.config_path = config.get("config_path", "")
        self.environments = config.get("environments", [])

        # If config_path not set directly, try to get from environments
        if not self.config_path and self.environments:
            env_config = self.environments[0]
            self.config_path = env_config.get("config_path", "")

        # Validate config_path
        if not self.config_path:
            logger.warning("No config_path specified in DeepCityInteractionHttp config!")
        elif not self.config_path.endswith(".sumocfg"):
            logger.warning(f"config_path should be a .sumocfg file, got: {self.config_path}")

        self.checkpoint_interval = config.get("checkpoint_interval", 1800)
        self.seed = config.get("seed", 42)
        self.use_gui = config.get("use_gui", False)
        self.simulation_duration = config.get("simulation_duration", 7200)

        # Control modules
        self.control_modules = config.get("control_modules", [])

        logger.info(
            f"DeepCityInteractionHttp initialized: "
            f"server={self.sumo_server_url}, num_masters={self.num_masters}"
        )

    def _get_master_id(self, instance_id: str) -> str:
        """
        Get master ID for an instance based on batch index.

        Uses the same logic as the Ray-based interaction for consistency.
        """
        # Extract prefix (remove _w suffix if present)
        if "_w" in instance_id:
            prefix = instance_id.split("_w")[0]
        else:
            prefix = instance_id

        # Extract number for deterministic assignment
        if '_' in prefix:
            try:
                num = int(prefix.split('_')[-1])
                master_index = num % self.num_masters
                return f"{self.master_name_prefix}_{master_index}"
            except ValueError:
                pass

        # Fallback to hash
        import hashlib
        hash_value = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
        master_index = hash_value % self.num_masters
        return f"{self.master_name_prefix}_{master_index}"

    def _ensure_master_exists(self, master_id: str) -> bool:
        """
        Ensure master session exists on the server.

        Returns True if master exists or was created successfully.
        """
        try:
            # Check if master exists
            self.http_client.get_master(master_id)
            return True
        except DeepCityHttpClientError as e:
            if "404" in str(e):
                # Master doesn't exist, try to create it
                logger.info(f"Master {master_id} not found, creating...")
                try:
                    # Use config_path (hardcoded for now)
                    config_path = self.config_path
                    if not config_path:
                        # Fallback to hardcoded path
                        config_path = "/data/zhouyuping/Zone/zone_scenarios/Upper_Manhattan/Upper_Manhattan.sumocfg"
                        logger.warning(f"No config_path in config, using hardcoded: {config_path}")

                    self.http_client.create_master(
                        master_id=master_id,
                        config_path=config_path,
                        checkpoint_interval=self.checkpoint_interval,
                        run_duration=self.simulation_duration,
                        seed=self.seed,
                        use_gui=self.use_gui,
                        available_control_modules=self.control_modules,
                        max_turns=self.max_turns
                    )
                    logger.info(f"Created master {master_id}")
                    return True
                except DeepCityHttpClientError as create_error:
                    # Check if error is "already exists" - this is actually success (race condition)
                    if "already exists" in str(create_error).lower():
                        logger.info(f"Master {master_id} was created by another worker (race condition), continuing...")
                        return True
                    logger.error(f"Failed to create master {master_id}: {create_error}")
                    return False
            else:
                logger.error(f"Error checking master {master_id}: {e}")
                return False

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Initialize a new interaction instance via HTTP.

        Args:
            instance_id: Unique identifier for this interaction
            checkpoint_path: Path to SUMO checkpoint (optional, managed by server)

        Returns:
            instance_id: The assigned instance ID
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Determine master and worker IDs
        master_id = self._get_master_id(instance_id)
        worker_id = f"worker_{instance_id}"

        # Ensure master exists
        if not self._ensure_master_exists(master_id):
            raise RuntimeError(f"Failed to ensure master {master_id} exists")

        # Generate initial prompt
        initial_prompt = self._generate_initial_prompt()

        try:
            # Reset worker on the server
            result = self.http_client.reset_worker(
                master_id=master_id,
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
                "improvement_count": 0,
                "best_simulation_result": None,
                "best_control_configs": None,
                "best_simulation_turn": None,
            }

            logger.info(
                f"Started HTTP interaction: {instance_id} "
                f"(master={master_id}, worker={worker_id}, {len(messages)} messages)"
            )
            return instance_id

        except DeepCityHttpClientError as e:
            logger.error(f"Failed to start interaction {instance_id}: {e}")
            raise RuntimeError(f"Failed to start interaction: {e}")

    def _generate_initial_prompt(self) -> str:
        """Generate initial prompt for the optimization task."""
        control_modules_str = ", ".join(self.control_modules) if self.control_modules else "all available"

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
                master_id=state["master_id"],
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
                master_id=state["master_id"],
                worker_id=state["worker_id"],
                llm_response=llm_response,
                verbose=kwargs.get("verbose", False)
            )

            # Update local state
            state["messages"] = result.get("messages", [])
            state["turn"] = result.get("turn_count", state["turn"] + 1)

            action_result = result.get("action_result", {})
            action_type = action_result.get("action_type", "")
            action_data = action_result.get("action_result", {})

            # Track improvement count
            if action_type in ["POLICY_PLANNING", "DEBUG"]:
                improved_modules = action_data.get("improved_modules", [])
                if improved_modules:
                    state["improvement_count"] += len(improved_modules)
                    logger.info(
                        f"{action_type} improved {len(improved_modules)} module(s): "
                        f"{', '.join(improved_modules)}, "
                        f"total: {state['improvement_count']}"
                    )

            elif action_type == "FINISH":
                # Sync final state
                state["best_simulation_result"] = action_data.get("best_simulation_result")
                state["best_simulation_turn"] = action_data.get("best_simulation_turn")
                state["best_control_configs"] = action_data.get("final_control_configs", {})

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
                reward = float(state["improvement_count"])
                extra_info = {
                    "success": True,
                    "improvement_count": state["improvement_count"],
                    "final_control_configs": state["best_control_configs"] or {},
                    "best_simulation_result": state["best_simulation_result"],
                    "best_simulation_turn": state["best_simulation_turn"],
                    "turn_count": state["turn"]
                }
            else:
                extra_info = {
                    "improvement_count": state["improvement_count"]
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
                master_id=state["master_id"],
                worker_id=state["worker_id"]
            )

            logger.info(
                f"Finalized HTTP interaction: {instance_id}, "
                f"improvements: {state['improvement_count']}, "
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
