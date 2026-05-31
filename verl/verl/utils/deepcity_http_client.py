
"""
HTTP Client for communicating with SUMO Server.

This module provides a client for the dual-server architecture:
- Server A (SUMO Server): Runs SUMO + LLMAgentManager + FastAPI
- Server B (Agent Loop): Runs verl PPO + LLM inference, uses this client

The client handles all HTTP communication with the SUMO server, including:
- Session management (create, get, delete)
- Simulation execution (run_simulation)
- Worker operations (reset, step, finalize)
- Reflection operations (start, update_memory, get_memory)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepCityHttpClientError(Exception):
    """Base exception for DeepCityHttpClient errors."""
    pass


class DeepCityHttpClient:
    """
    HTTP client for communicating with SUMO Server.

    Provides methods for:
    - Session management (create, get, delete)
    - Simulation execution (run_simulation)
    - Worker operations (reset, step, finalize)
    - Reflection operations

    Features:
    - Connection pooling via requests.Session
    - Automatic retries with exponential backoff
    - Configurable timeouts
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8011",
        timeout: float = 300.0,  # 5 minutes default (simulations can be slow)
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5
    ):
        """
        Initialize HTTP client.

        Args:
            base_url: Base URL of the SUMO server (e.g., "http://sumo-server:8000")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"DeepCityHttpClient initialized with base_url={base_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the server.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (e.g., "/sessions")
            json_data: JSON data to send (for POST requests)
            timeout: Request timeout (uses default if not specified)

        Returns:
            Response JSON as dictionary

        Raises:
            DeepCityHttpClientError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout

        try:
            if method == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method == "POST":
                response = self.session.post(url, json=json_data, timeout=timeout)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for HTTP errors
            if response.status_code >= 400:
                error_detail = response.json().get("detail", response.text) if response.text else "Unknown error"
                raise DeepCityHttpClientError(
                    f"HTTP {response.status_code} error for {method} {endpoint}: {error_detail}"
                )

            return response.json()

        except requests.exceptions.Timeout:
            raise DeepCityHttpClientError(f"Request timeout for {method} {endpoint}")
        except requests.exceptions.ConnectionError as e:
            raise DeepCityHttpClientError(f"Connection error for {method} {endpoint}: {e}")
        except requests.exceptions.RequestException as e:
            raise DeepCityHttpClientError(f"Request error for {method} {endpoint}: {e}")

    # ==================== Session Operations ====================

    def create_session(
        self,
        session_id: Optional[str] = None,
        environments: Optional[List[Dict[str, Any]]] = None,
        current_env_index: int = 0,
        config_path: str = "",
        checkpoint_interval: int = 1800,
        run_duration: int = 7200,
        seed: int = 42,
        use_gui: bool = False,
        available_control_modules: Optional[List[str]] = None,
        max_turns: int = 10,
        max_reflection_turns: int = 5,
        max_memory_items: int = 10
    ) -> Dict[str, Any]:
        """
        创建新的仿真会话（SUMO 环境）。

        Args:
            session_id: Optional session ID (auto-generated if not provided)
            environments: List of environment configurations (supports multi-env cycling)
            current_env_index: Index of the environment to use initially (for batch parallel allocation)
            config_path: Path to SUMO config file (for backward compatibility)
            checkpoint_interval: Interval between checkpoints in seconds
            run_duration: Total simulation duration in seconds
            seed: Random seed
            use_gui: Whether to use SUMO GUI
            available_control_modules: List of control modules to enable
            max_turns: Maximum dialogue turns for LLMAgentManager
            max_reflection_turns: Maximum reflection turns
            max_memory_items: Maximum memory items

        Returns:
            Dictionary with session_id and configuration
        """
        data = {
            "checkpoint_interval": checkpoint_interval,
            "run_duration": run_duration,
            "seed": seed,
            "use_gui": use_gui,
            "max_turns": max_turns,
            "max_reflection_turns": max_reflection_turns,
            "max_memory_items": max_memory_items
        }
        
        #   支持多环境配置（与 Ray 版本一致）
        if environments:
            data["environments"] = environments
            data["current_env_index"] = current_env_index  #   指定初始环境索引
        else:
            # 兼容旧版本：单环境配置
            data["config_path"] = config_path
            data["available_control_modules"] = available_control_modules or []
        
        if session_id:
            data["session_id"] = session_id  # 使用新的 session_id 字段名

        return self._request("POST", "/sessions", json_data=data)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取仿真会话状态。

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session status
        """
        return self._request("GET", f"/sessions/{session_id}")

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        删除仿真会话并关闭 SUMOEnv。

        Args:
            session_id: Session ID

        Returns:
            Dictionary confirming deletion
        """
        return self._request("DELETE", f"/sessions/{session_id}")

    def run_simulation(
        self,
        session_id: str,
        duration: int,
        checkpoint_interval: int
    ) -> Dict[str, Any]:
        """
        运行 SUMO 仿真（无状态服务）。

        Args:
            session_id: Session ID
            duration: 仿真时长（秒）
            checkpoint_interval: Checkpoint 间隔（秒）

        Returns:
            Dictionary with checkpoint_path, baseline_result, current_time
        """
        return self._request(
            "POST",
            f"/sessions/{session_id}/run_simulation?duration={duration}&checkpoint_interval={checkpoint_interval}"
        )

    def restart_session(
        self,
        session_id: str,
        env_index: int,
        seed: int,
        episode_count: int
    ) -> Dict[str, Any]:
        """
        重启仿真会话，切换到新的环境配置。

        Args:
            session_id: Session ID
            env_index: 新环境在 environments 列表中的索引
            seed: 新的随机种子
            episode_count: 当前 episode 计数

        Returns:
            Dictionary with session status after restart
        """
        data = {
            "env_index": env_index,
            "seed": seed,
            "episode_count": episode_count
        }
        return self._request(
            "POST",
            f"/sessions/{session_id}/restart",
            json_data=data
        )

    # ==================== Worker Operations ====================

    def reset_worker(
        self,
        session_id: str,
        worker_id: str,
        initial_prompt: str,
        memory: Optional[List[str]] = None,
        initial_best_result: Optional[Dict[str, Any]] = None,
        initial_control_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        初始化/重置 Worker 的 LLMAgentManager。

        Args:
            session_id: Session ID
            worker_id: Worker ID
            initial_prompt: Initial prompt for the optimization task
            memory: Optional list of memory items from previous sessions
            initial_best_result: Optional initial best simulation result
            initial_control_configs: Optional initial control configurations

        Returns:
            Dictionary with messages and worker_id
        """
        data = {
            "initial_prompt": initial_prompt,
            "memory": memory,
            "initial_best_result": initial_best_result,
            "initial_control_configs": initial_control_configs
        }
        return self._request(
            "POST",
            f"/sessions/{session_id}/workers/{worker_id}/reset",
            json_data=data
        )

    def step_worker(
        self,
        session_id: str,
        worker_id: str,
        llm_response: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        处理 Worker 的 LLM 响应并执行策略仿真。

        Args:
            session_id: Session ID
            worker_id: Worker ID
            llm_response: LLM response text
            verbose: Whether to print detailed logs on server

        Returns:
            Dictionary with messages, action_result, finished flag, and turn_count
        """
        data = {
            "llm_response": llm_response,
            "verbose": verbose
        }
        return self._request(
            "POST",
            f"/sessions/{session_id}/workers/{worker_id}/step",
            json_data=data
        )

    def get_worker_messages(
        self,
        session_id: str,
        worker_id: str
    ) -> Dict[str, Any]:
        """
        获取 Worker 的当前 messages。

        Args:
            session_id: Session ID
            worker_id: Worker ID

        Returns:
            Dictionary with messages and turn_count
        """
        return self._request(
            "GET",
            f"/sessions/{session_id}/workers/{worker_id}/messages"
        )

    def get_worker_state(
        self,
        session_id: str,
        worker_id: str
    ) -> Dict[str, Any]:
        """
        获取 Worker 的状态，包括最佳结果。

        Args:
            session_id: Session ID
            worker_id: Worker ID

        Returns:
            Dictionary with turn_count, reflection state, and best results
        """
        return self._request(
            "GET",
            f"/sessions/{session_id}/workers/{worker_id}/state"
        )

    def finalize_worker(
        self,
        session_id: str,
        worker_id: str
    ) -> Dict[str, Any]:
        """
        Finalize a worker and return its best policy.

        Args:
            session_id: Session ID
            worker_id: Worker ID

        Returns:
            Dictionary with best_control_configs and best_simulation_result
        """
        return self._request(
            "POST",
            f"/sessions/{session_id}/workers/{worker_id}/finalize",
            json_data={}
        )

    # ==================== Reflection Operations ====================

    def start_reflection(
        self,
        session_id: str,
        worker_id: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        启动 Worker 的 reflection 阶段。

        Args:
            session_id: Session ID
            worker_id: Worker ID
            history: List of (action_type, action_result) as dicts

        Returns:
            Dictionary with updated messages
        """
        data = {"history": history}
        return self._request(
            "POST",
            f"/sessions/{session_id}/workers/{worker_id}/reflection/start",
            json_data=data
        )

    def update_memory(
        self,
        session_id: str,
        worker_id: str,
        reflection_response: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        从 reflection 响应更新 Worker 的 memory。

        Args:
            session_id: Session ID
            worker_id: Worker ID
            reflection_response: LLM's reflection response (JSON array)
            verbose: Whether to print detailed logs

        Returns:
            Dictionary with updated memory list
        """
        data = {
            "reflection_response": reflection_response,
            "verbose": verbose
        }
        return self._request(
            "POST",
            f"/sessions/{session_id}/workers/{worker_id}/reflection/update_memory",
            json_data=data
        )

    def get_memory(
        self,
        session_id: str,
        worker_id: str
    ) -> Dict[str, Any]:
        """
        获取 Worker 的当前 memory。

        Args:
            session_id: Session ID
            worker_id: Worker ID

        Returns:
            Dictionary with memory list
        """
        return self._request(
            "GET",
            f"/sessions/{session_id}/workers/{worker_id}/memory"
        )

    # ==================== Health Check ====================

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            Dictionary with status and sessions_count
        """
        return self._request("GET", "/health")

    def wait_for_server(
        self,
        timeout: float = 60.0,
        interval: float = 1.0
    ) -> bool:
        """
        Wait for server to become available.

        Args:
            timeout: Maximum time to wait in seconds
            interval: Time between checks in seconds

        Returns:
            True if server is available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.health_check()
                logger.info(f"SUMO server is available at {self.base_url}")
                return True
            except DeepCityHttpClientError:
                logger.debug(f"Waiting for SUMO server at {self.base_url}...")
                time.sleep(interval)

        logger.warning(f"Timeout waiting for SUMO server at {self.base_url}")
        return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
