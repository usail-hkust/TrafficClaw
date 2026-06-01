"""
LLM Agent Master - Handles interaction between LLM and other modules.
Separates LLM interaction logic from the main LLMAgent class.
"""

from calendar import c
import sys
import json
import traceback
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from utils.code_sandbox import execute_code
from control_modules.shared.decision_context import DecisionContextManager

def parse_llm_action(response: str) -> Tuple[str, str]:
    """
    Parse LLM response to extract action type and content.
    
    Args:
        response: LLM response text
        
    Returns:
        Tuple of (action_type, action_content)
        action_type: One of "PLAN", "DATA_ANALYSIS", "POLICY_PLANNING", "DEBUG", "GET_CONTROL_API", "FINISH", or "UNKNOWN"
        action_content: The content associated with the action
    """
    response_upper = response.upper()
    first_nonempty_line = ""
    for line in response.splitlines():
        if line.strip():
            first_nonempty_line = line.strip()
            break

    # Check for SELECT_MODULES action (for joint control module selection)
    if "ACTION: SELECT_MODULES" in response_upper or "ACTION:SELECT_MODULES" in response_upper:
        return ("SELECT_MODULES", response)

    # Check for FINISH action
    if "ACTION: FINISH" in response_upper or "ACTION:FINISH" in response_upper:
        return ("FINISH", response)
    
    # Check for REFLECTION_FINISH action (used only in reflection phase)
    if "ACTION: REFLECTION_FINISH" in response_upper or "ACTION:REFLECTION_FINISH" in response_upper:
        return ("REFLECTION_FINISH", response)
    
    # Check for GET_CONTROL_API action
    if "ACTION: GET_CONTROL_API" in response_upper or "ACTION:GET_CONTROL_API" in response_upper or "ACTION: GET_CONTROL_API" in response_upper.replace(" ", ""):
        return ("GET_CONTROL_API", response)
    # Fallback: allow missing ACTION prefix if response starts with GET_CONTROL_API
    if first_nonempty_line.upper().startswith("GET_CONTROL_API") or first_nonempty_line.upper().startswith("GET CONTROL API"):
        return ("GET_CONTROL_API", response)
    
    # Check for PLAN action
    if "ACTION: PLAN" in response_upper or "ACTION:PLAN" in response_upper:
        return ("PLAN", response)
    
    # Check for DEBUG action
    if "ACTION: DEBUG" in response_upper or "ACTION:DEBUG" in response_upper:
        return ("DEBUG", response)
    
    # Check for DATA_ANALYSIS action (support both underscore and space formats for backward compatibility)
    if ("ACTION: DATA_ANALYSIS" in response_upper or "ACTION:DATA_ANALYSIS" in response_upper or "ACTION: DATA ANALYSIS" in response_upper):
        # Extract code block
        code = extract_code_block(response)
        if code:
            if code.strip():
                return ("DATA_ANALYSIS", code)
            else:
                return ("UNKNOWN", "DATA_ANALYSIS action detected but extracted code is empty after cleaning")
        else:
            return ("UNKNOWN", "DATA_ANALYSIS action detected but no code block found")
    
    # Check for POLICY_PLANNING action (support both underscore and space formats for backward compatibility)
    if ("ACTION: POLICY_PLANNING" in response_upper or "ACTION:POLICY_PLANNING" in response_upper or "ACTION: POLICY PLANNING" in response_upper):
        # Extract code block
        code = extract_code_block(response)
        if code:
            if code.strip():
                # Include the full response (not just code) so that "Control Modules:" line is preserved
                # The code block will be extracted again in execute_code if needed
                return ("POLICY_PLANNING", response)
            else:
                return ("UNKNOWN", "POLICY_PLANNING action detected but extracted code is empty after cleaning")
        else:
            return ("UNKNOWN", "POLICY_PLANNING action detected but no code block found")
    
    # Default: treat as unknown
    return ("UNKNOWN", response)


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract Python code block from markdown-formatted text.
    
    Args:
        text: Text containing code block
        
    Returns:
        Extracted code string, or None if no code block found
    """
    # Look for ```python ... ``` or ``` ... ```
    lines = text.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                break
            else:
                # Start of code block
                in_code_block = True
                continue
        
        if in_code_block:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None


def execute_simulation_action(
    control_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute a simulation to test control module configurations.
    
    Args:
        control_configs: Dictionary of control configurations by module name
                       Format: {"signal_timing": {...}, "subway_scheduling": {...}, ...}
        context: Context dictionary containing checkpoint_path and other info
        verbose: Whether to print execution details
        
    Returns:
        Dictionary containing:
            - success: Whether simulation succeeded
            - stats: Simulation statistics (departed, arrived, avg_travel_time, etc.)
            - module_metrics: Metrics for each control module (if available)
            - error: Error message (if failed)
    """
    import copy
    
    # Save a deep copy of control_configs at the start to preserve LLM agent-specified modules
    # This ensures we only return the configs that the LLM agent explicitly planned,
    # even if run_policy_simulation or run_controlled_simulation modify the original dict
    # Extract only configs (remove module instances) for return value
    from utils.checkpoint_logger import extract_configs_only
    original_control_configs = extract_configs_only(control_configs) if control_configs else {}
    
    if not original_control_configs:
        return {
            "success": False,
            "stats": {},
            "error": "No control configuration available. Use POLICY_PLANNING action to create one first."
        }
    
    if context is None:
        context = {}
    
    if verbose:
        module_names = list(original_control_configs.keys())
        print(f"Running policy simulation with control modules: {module_names}")
        for module_name, config_entry in original_control_configs.items():
            # Handle both bundle format {'module': ..., 'config': ...} and plain config dict
            config = config_entry.get("config", config_entry) if isinstance(config_entry, dict) and "config" in config_entry else config_entry
            if isinstance(config, dict):
                print(f"  - {module_name}: {len(config)} entries")
    
    try:
        from utils.simulation_utils import run_policy_simulation
        
        checkpoint_path = context.get("checkpoint_path")
        config_path = context.get("config_path")  # Get config_path from context
        checkpoint_interval = context.get("checkpoint_interval")  # Get checkpoint_interval from context
        test_duration = context.get("test_duration", 300)  # Default 5 minutes test
        run_duration = context.get("run_duration")  # Get run_duration for run_counts
        use_gui = context.get("use_gui", False)  # Get use_gui from context, default to False
        seed = context.get("seed")  # Get seed from context to use same seed as main simulation
        initial_control_states = context.get("initial_control_states")  # Optional: stateful control module data
        taxi_dispatch_algorithm = context.get("taxi_dispatch_algorithm")
        taxi_idle_algorithm = context.get("taxi_idle_algorithm")
        
        result = run_policy_simulation(
            checkpoint_path=checkpoint_path,
            control_configs=control_configs,  # Pass original control_configs (may be modified internally)
            duration=test_duration,
            use_gui=use_gui,  # Use use_gui from context to match main simulation settings
            config_path=config_path,  # Pass config_path to use same config as original simulation
            checkpoint_interval=checkpoint_interval,  # Pass checkpoint_interval to load t-1 state
            run_duration=run_duration,  # Pass run_duration for run_counts
            seed=seed,  # Pass seed to use same seed as main simulation
            initial_control_states=initial_control_states,
            taxi_dispatch_algorithm=taxi_dispatch_algorithm,
            taxi_idle_algorithm=taxi_idle_algorithm,
        )
        
        if result.get("success"):
            # Get policy-specific travel time (preferred for comparison)
            policy_avg_travel_time = result.get("policy_avg_travel_time")
            policy_arrived_count = result.get("policy_arrived_count", 0)
            policy_highway_avg_travel_time = result.get("policy_highway_avg_travel_time")
            policy_highway_arrived_count = result.get("policy_highway_arrived_count", 0)
            avg_travel_time = result.get("avg_travel_time", 0)
            
            if verbose:
                print(f"Simulation completed successfully.")
                print(f"  Departed: {result.get('total_departed', 0)}")
                print(f"  Arrived: {result.get('total_arrived', 0)}")
                if policy_avg_travel_time is not None:
                    print(f"  Policy avg travel time ({policy_arrived_count} vehicles): {policy_avg_travel_time:.2f}s")
                    print(f"  Global avg travel time: {avg_travel_time:.2f}s")
                else:
                    print(f"  Avg travel time: {avg_travel_time:.2f}s")
                if policy_highway_avg_travel_time is not None:
                    print(f"  Policy highway avg travel time ({policy_highway_arrived_count} vehicles): {policy_highway_avg_travel_time:.2f}s")
            
                # Display module metrics if available
                # Filter module_metrics to only include available_control_modules
                module_metrics = result.get("module_metrics", {})
                if module_metrics:
                    available_control_modules = context.get("available_control_modules") if context else None
                    filtered_module_metrics = {}
                    if available_control_modules is not None:
                        for module_name in module_metrics.keys():
                            if module_name in available_control_modules:
                                filtered_module_metrics[module_name] = module_metrics[module_name]
                    else:
                        # If available_control_modules is None, include all modules
                        filtered_module_metrics = module_metrics
                    
                    if filtered_module_metrics:
                        print(f"\n  Control Module Metrics:")
                        for module_name, metrics in filtered_module_metrics.items():
                            print(f"    {module_name.upper().replace('_', ' ')}:")
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, float):
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                                else:
                                    print(f"      - {metric_name.replace('_', ' ').title()}: {metric_value}")
            
            # Use original_control_configs saved at function start to ensure we only return
            # the configs that the LLM agent explicitly planned, even if control_configs was modified
            return_dict = {
                "success": True,
                "stats": {
                    "total_departed": result.get("total_departed", 0),
                    "total_arrived": policy_arrived_count if policy_arrived_count > 0 else result.get("total_arrived", 0),  # Use policy count if available
                    "avg_travel_time": avg_travel_time,  # Global average
                    "policy_avg_travel_time": policy_avg_travel_time,  # Policy-specific average (preferred for comparison)
                    "policy_arrived_count": policy_arrived_count,  # Vehicles arrived during policy test
                    "policy_highway_avg_travel_time": policy_highway_avg_travel_time,  # Highway-only policy avg travel time
                    "policy_highway_arrived_count": policy_highway_arrived_count,  # Highway-only arrived vehicles during policy test
                    "duration": result.get("duration", 0)
                },
                "error": None,
                "control_configs": original_control_configs  # Include only LLM agent-specified control configs (saved at function start)
            }
            
            # Add module metrics if available
            # Filter module_metrics to only include available_control_modules
            if "module_metrics" in result:
                available_control_modules = context.get("available_control_modules") if context else None
                module_metrics = result["module_metrics"]
                if available_control_modules is not None:
                    filtered_module_metrics = {
                        name: metrics for name, metrics in module_metrics.items()
                        if name in available_control_modules
                    }
                    return_dict["module_metrics"] = filtered_module_metrics
                else:
                    # If available_control_modules is None, include all modules
                    return_dict["module_metrics"] = module_metrics
            
            return return_dict
        else:
            error = result.get("error", "Unknown error")
            if verbose:
                print(f"Simulation failed: {error}")
            
            # Use original_control_configs saved at function start
            return {
                "success": False,
                "stats": {},
                "error": error,
                "control_configs": original_control_configs  # Include only LLM agent-specified control configs (saved at function start)
            }
            
    except Exception as e:
        error_msg = f"Simulation execution error: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
            traceback.print_exc()
        
        # Use original_control_configs saved at function start
        return {
            "success": False,
            "stats": {},
            "error": error_msg,
            "control_configs": original_control_configs  # Include only LLM agent-specified control configs (saved at function start)
        }


class LLMAgentManager:
    """
    Master class for handling LLM interaction with other modules.
    Separates message management and action execution from LLM inference.

    Methods:
        - reset(): Initialize messages based on experiment parameters
        - step(): Process LLM response, execute actions, and update messages
    """

    # Cross-module dependencies for joint control
    MODULE_DEPENDENCIES = {
        "signal_timing": {"affects": ["bus_scheduling", "taxi_scheduling"], "affected_by": []},
        "highway_speed_limit": {"affects": ["ramp_metering"], "affected_by": []},
        "ramp_metering": {"affects": [], "affected_by": ["highway_speed_limit"]},
        "bus_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
        "taxi_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
        "subway_scheduling": {"affects": [], "affected_by": []},
    }

    def __init__(
        self,
        available_control_modules: Optional[List[str]] = None,
        max_turns: int = 10,
        config_name: Optional[str] = None,
        max_memory_items: int = 10,
        max_reflection_turns: int = 5,
        decision_context_manager: Optional[Any] = None,
        is_joint_control: bool = False,
        module_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize LLM Agent Manager.

        Args:
            available_control_modules: List of available control module names
            max_turns: Maximum number of dialogue turns
            config_name: Config directory name for file naming
            max_memory_items: Maximum number of memory items to keep (default: 10)
            max_reflection_turns: Maximum number of reflection turns (default: 5)
            decision_context_manager: Optional DecisionContextManager for cross-module coordination
            is_joint_control: Whether this is a joint control session (affects system message format)
            module_metrics: Current module performance metrics (for joint control mode)
        """
        self.available_control_modules = available_control_modules
        self.max_turns = max_turns
        self.config_name = config_name
        self.max_memory_items = max_memory_items
        self.max_reflection_turns = max_reflection_turns
        self.decision_context_manager = decision_context_manager
        self.is_joint_control = is_joint_control
        self.module_metrics = module_metrics or {}

        # Internal state
        self.messages = []
        self.turn_count = 0
        self.current_control_configs = {}
        self.last_code_result = None

        # Track best simulation result
        self.best_simulation_result: Optional[Dict[str, Any]] = None
        self.best_simulation_turn: Optional[int] = None
        self.best_control_configs: Dict[str, Dict[str, Any]] = {}

        # Memory for cross-experiment learning
        self.memory: List[str] = []
        
        # Reflection phase state
        self.is_reflection_phase: bool = False
        self.reflection_turn: int = 0

        # Load control specs for API queries
        self._control_specs = self._load_control_specs()
    
    def reset(
        self,
        initial_prompt: str,
        context: Dict[str, Any],
        verbose: bool = True,
        initial_best_result: Optional[Dict[str, Any]] = None,
        initial_control_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reset agent state and initialize messages for a new optimization session.
        
        Args:
            initial_prompt: Initial user prompt describing the optimization task
            context: Context dictionary containing:
                - lane_graph: NetworkX graph
                - lane_inter_graph: NetworkX graph
                - intersection_graph: NetworkX graph
                - lane_dict: Dictionary
                - traffic_states_filepath: Path to traffic states file
                - checkpoint_path: Path to current checkpoint
                - current_configs: Dictionary containing all current module configurations
                - session_id: Session ID for cache directory (optional)
                - cache_dir: Cache directory path (optional)
                - memory: Optional explicit memory list; if omitted, existing self.memory is kept (not cleared)
            verbose: Whether to print detailed logs
            initial_best_result: Initial best simulation result to set (e.g., from checkpoint)
            initial_control_configs: Initial control configs to set as best configs (e.g., from checkpoint)
            
        Returns:
            List of initialized messages (system + initial user message)
        """
        # Reset internal state
        self.messages = []
        self.turn_count = 0
        # Initialize current_control_configs from context["current_configs"] if available
        # Use deepcopy to avoid modifying the original context data
        self.current_control_configs = copy.deepcopy(context.get("current_configs", {})) if context else {}
        self.last_code_result = None
        # Reset reflection phase state
        self.is_reflection_phase = False
        self.reflection_turn = 0
        # Store initial baseline result for later comparison in reflection
        self.initial_best_simulation_result = copy.deepcopy(initial_best_result) if initial_best_result else None
        self.best_simulation_result = copy.deepcopy(initial_best_result) if initial_best_result else None
        self.best_simulation_turn = 0 if initial_best_result else None
        self.best_control_configs = initial_control_configs.copy() if initial_control_configs else {}
        
        # Memory persists across reset() (e.g. multi-checkpoint runs). Reflection / add_memory
        # updates carry forward. Override only when context explicitly contains "memory".
        if context is not None and "memory" in context:
            mem = context["memory"]
            self.memory = list(mem) if mem else []
        
        # Set up cache directory for this optimization session
        if "session_id" not in context:
            import time
            context["session_id"] = f"session_{int(time.time() * 1000)}"
        
        # Add system message
        self.add_system_message()
        
        # Add initial user message
        self.add_user_message(initial_prompt)
        
        if verbose:
            print(f"\n{'='*80}")
            print("LLM AGENT MANAGER INITIALIZED")
            print(f"{'='*80}")
            print(f"Initial Prompt: {initial_prompt}\n")
        
        return self.messages.copy()
    
    def step(
        self,
        llm_response: str,
        context: Dict[str, Any],
        env: Optional[Any] = None,
        verbose: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process LLM response, execute actions, and update messages.
        
        Args:
            llm_response: LLM response text
            context: Context dictionary (same as in reset)
            env: Optional SUMOEnv instance to access enabled control modules
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (updated_messages, action_result)
            - updated_messages: Updated list of messages after processing
            - action_result: Dictionary containing:
                - action_type: Type of action executed
                - action_result: Result of action execution
                - finished: Whether FINISH action was executed
                - turn_count: Current turn count
        """
        # Update turn count based on phase
        if self.is_reflection_phase:
            self.reflection_turn += 1
        else:
            self.turn_count += 1
        
        if verbose:
            if self.is_reflection_phase:
                print(f"\n--- Reflection Turn {self.reflection_turn}/{self.max_reflection_turns} ---")
            else:
                print(f"\n--- Turn {self.turn_count}/{self.max_turns} ---")
            print(f"LLM Response:\n{llm_response}\n")
        
        # Add assistant message
        self.add_assistant_message(llm_response)
        
        # Add turn count reminder if not first turn
        if self.is_reflection_phase:
            remaining_turns = self.max_reflection_turns - self.reflection_turn
            if remaining_turns == 1:
                # Next turn will be the last one - warn LLM
                turn_reminder = f"\n[CRITICAL: This turn ({self.reflection_turn + 1}/{self.max_reflection_turns}) is your LAST reflection turn. NOW, you MUST execute ACTION: REFLECTION_FINISH and return the memory as a JSON array in that final response. Format: ACTION: REFLECTION_FINISH\n```json\n[\"INSIGHT 1\", \"INSIGHT 2\", ...]\n```]"
                self.add_user_message(turn_reminder)
            else:
                turn_reminder = f"\n[Reminder: You are on reflection turn {self.reflection_turn + 1}/{self.max_reflection_turns}. {remaining_turns} reflection turns remaining. Please work efficiently and use REFLECTION_FINISH when reflection is complete.]"
                self.add_user_message(turn_reminder)
        else:
            turn_reminder = f"\n[Reminder: You are on turn {self.turn_count + 1}/{self.max_turns}. {self.max_turns - self.turn_count} turns remaining. Please work efficiently and use FINISH when optimization is complete.]"
            self.add_user_message(turn_reminder)
        
        # Parse action from response
        action_type, action_content = parse_llm_action(llm_response)
        
        if verbose:
            print(f"Parsed Action: {action_type}")
        
        action_result = {
            "action_type": action_type,
            "action_result": None,
            "finished": False,
            "turn_count": self.turn_count,
            "memory": self.memory
        }
        
        # Execute action based on type
        if action_type == "PLAN":
            action_result["action_result"] = self._handle_plan_action(action_content, verbose=verbose)
        
        elif action_type == "DATA_ANALYSIS":
            result = execute_code(action_content, context, verbose=verbose)
            action_result["action_result"] = result
            self.last_code_result = result
            self._handle_data_analysis_result(result, verbose=verbose)
        
        elif action_type == "POLICY_PLANNING":
            action_result["action_result"] = self._handle_policy_planning_action(
                action_content, context, env, verbose=verbose
            )
        
        elif action_type == "DEBUG":
            action_result["action_result"] = self._handle_debug_action(
                action_content, context, env, verbose=verbose
            )
        
        elif action_type == "GET_CONTROL_API":
            action_result["action_result"] = self._handle_get_control_api_action(
                action_content,
                context=context,
                verbose=verbose
            )
        
        elif action_type == "FINISH":
            action_result["finished"] = True
            # Extract only configs (remove module instances) before returning
            from utils.checkpoint_logger import extract_configs_only
            raw_final_configs = self.best_control_configs.copy() if self.best_control_configs else self.current_control_configs.copy()
            final_configs_clean = extract_configs_only(raw_final_configs) if raw_final_configs else {}
            # Clean best_simulation_result so it is JSON-serializable (no module instances in control_configs)
            best_result_for_action = None
            if self.best_simulation_result:
                best_result_for_action = self.best_simulation_result.copy()
                if isinstance(best_result_for_action, dict) and "control_configs" in best_result_for_action:
                    best_result_for_action["control_configs"] = extract_configs_only(best_result_for_action["control_configs"]) or {}
            action_result["action_result"] = {
                "message": "Optimization completed",
                "final_control_configs": final_configs_clean,
                "best_simulation_result": best_result_for_action,
                "best_simulation_turn": self.best_simulation_turn
            }
        
        elif action_type == "REFLECTION_FINISH":
            # Used only in reflection phase to indicate the last reflection response      
            # (which should contain the updated memory list as JSON).
            # We do NOT mark finished=True here to avoid affecting the main optimization loop.
            # Reset reflection phase state
            self.is_reflection_phase = False
            self.reflection_turn = 0
            action_result["action_result"] = {
                "message": "Reflection finished"
            }
        
        else:
            # Unknown action
            error_msg = f"Unknown action type: {action_type}. Please use `ACTION: PLAN`, `ACTION: DATA_ANALYSIS`, `ACTION: POLICY_PLANNING`, `ACTION: DEBUG`, `ACTION: GET_CONTROL_API`, `ACTION: SELECT_MODULES`, or `ACTION: FINISH`."
            error_msg += "\nNote: Simulation is automatically executed after POLICY_PLANNING - you don't need to call it manually."
            self.add_user_message(error_msg)
            action_result["action_result"] = {"error": error_msg}
        
        return self.messages.copy(), action_result
    
    def add_system_message(self):
        """Add system message to conversation."""
        system_message = self._format_system_message()
        self.messages.append({
            "role": "system",
            "content": system_message
        })
    
    def add_user_message(self, content: str):
        """Add user message to conversation."""
        if self.messages and self.messages[-1].get("role") == "user":
            last_content = self.messages[-1].get("content", "")
            # If last message is the last reflection reminder and content is debug info, skip adding
            is_last_reflection_reminder = (
                "CRITICAL" in last_content
                and "LAST reflection turn" in last_content
                and "REFLECTION_FINISH" in last_content
            )
            debug_keywords = ["Code execution failed", "Please fix the code", "Please fix", "fix the code"]
            is_debug_info = any(kw in content for kw in debug_keywords)
            if is_last_reflection_reminder and is_debug_info:
                return
            # Merge with previous user message
            self.messages[-1]["content"] = self.messages[-1]["content"] + "\n\n" + content
        else:
            self.messages.append({
                "role": "user",
                "content": content
            })
    
    def add_assistant_message(self, content: str):
        """Add assistant message to conversation."""
        if self.messages and self.messages[-1].get("role") == "assistant":
            # Merge with previous assistant message
            self.messages[-1]["content"] = self.messages[-1]["content"] + "\n\n" + content
        else:
            self.messages.append({
                "role": "assistant",
                "content": content
            })
    
    def _handle_plan_action(self, action_content: str, verbose: bool = True) -> Dict[str, Any]:
        """Handle PLAN action."""
        if verbose:
            print("PLAN action received - acknowledging plan")
        
        # Extract plan content (remove ACTION: PLAN line if present)
        plan_content = action_content
        if "ACTION: PLAN" in action_content.upper() or "ACTION:PLAN" in action_content.upper():
            lines = action_content.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if "ACTION: PLAN" in line.upper() or "ACTION:PLAN" in line.upper():
                    start_idx = i + 1
                    break
            plan_content = '\n'.join(lines[start_idx:]).strip()
        
        feedback = "Plan received and acknowledged.\n\n"
        feedback += plan_content
        feedback += "\n\nYou can now proceed with the planned actions (DATA_ANALYSIS, POLICY_PLANNING, etc.)."
        
        self.add_user_message(feedback)
        return {"content": plan_content}
    
    def _handle_data_analysis_result(self, result: Dict[str, Any], verbose: bool = True) -> None:
        """Handle DATA_ANALYSIS action result."""
        feedback = self._format_code_result(result, check_control_config=False)
        self.add_user_message(feedback)
    
    def _handle_policy_planning_action(
        self,
        action_content: str,
        context: Dict[str, Any],
        env: Optional[Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Handle POLICY_PLANNING action."""
        # Extract specified control modules from action content
        specified_modules = self._extract_control_modules(action_content)
        
        if not specified_modules:
            error_msg = "POLICY_PLANNING action must specify 'Control Modules: <module1>, <module2>' line."
            error_msg += "\nAvailable modules: 'signal_timing', 'highway_speed_limit', 'ramp_metering', 'subway_scheduling', 'bus_scheduling'"
            self.add_user_message(error_msg)
            if verbose:
                print(f"Error: {error_msg}")
            return {"error": error_msg}
        
        if verbose:
            print(f"POLICY_PLANNING specified modules: {specified_modules}")
        
        # Extract code block from action_content for execution
        code = extract_code_block(action_content) if action_content else None
        if not code:
            error_msg = "POLICY_PLANNING action must contain code block. Please provide code."
            self.add_user_message(error_msg)
            if verbose:
                print(f"Error: {error_msg}")
            return {"error": error_msg}
        
        result = execute_code(code, context, verbose=verbose)
        self.last_code_result = result
        
        # Handle result
        planning_success = self._handle_policy_planning_result(
            result=result,
            specified_modules=specified_modules,
            env=env,
            context=context,
            verbose=verbose
        )
        
        # Automatically execute simulation after successful POLICY_PLANNING
        if planning_success:
            if verbose:
                print(f"\n[Auto-executing simulation after POLICY_PLANNING...]")
            
            # Get control_configs from context (set by POLICY_PLANNING)
            control_configs = context.get("control_configs")
            
            if control_configs is None:
                control_configs = {}
            
            if control_configs:
                # Pass available_control_modules to context for filtering module_metrics
                context["available_control_modules"] = self.available_control_modules
                
                # Execute simulation
                sim_result = execute_simulation_action(
                    control_configs=control_configs,
                    context=context,
                    verbose=verbose
                )
                
                # Save best_simulation_result BEFORE comparison and update
                best_result_before_update = None
                if self.best_simulation_result is not None:
                    best_result_before_update = copy.deepcopy(self.best_simulation_result)
                
                # Update best simulation result and configs per module if any module is better 
                if sim_result.get("success"):
                    module_comparison = self._is_better_simulation_result(sim_result, self.best_simulation_result)
                    
                    # Initialize best_simulation_result if it doesn't exist
                    if self.best_simulation_result is None:
                        self.best_simulation_result = {
                            "success": True,
                            "stats": {},
                            "module_metrics": {}
                        }
                    
                    # Update best configs and metrics for each module that improved
                    improved_modules = []
                    for module_name, is_better in module_comparison.items():
                        if is_better:
                            # Save best config for this module (store bundle so rollback keeps module reference)
                            if module_name in control_configs:
                                if self.best_control_configs is None:
                                    self.best_control_configs = {}
                                entry = self.current_control_configs.get(module_name)
                                if isinstance(entry, dict) and "module" in entry and "config" in entry:
                                    self.best_control_configs[module_name] = {
                                        "module": entry["module"],
                                        "config": control_configs[module_name].copy(),
                                    }
                                else:
                                    self.best_control_configs[module_name] = control_configs[module_name].copy()
                                improved_modules.append(module_name)
                            
                            # Update best metrics for this module
                            current_module_metrics = sim_result.get("module_metrics", {}).get(module_name)
                            if current_module_metrics:
                                if "module_metrics" not in self.best_simulation_result:
                                    self.best_simulation_result["module_metrics"] = {}
                                self.best_simulation_result["module_metrics"][module_name] = current_module_metrics.copy()
                    
                    # Update best_simulation_result["control_configs"] when any module improved
                    if improved_modules:
                        from utils.checkpoint_logger import extract_configs_only
                        if "control_configs" not in self.best_simulation_result:
                            self.best_simulation_result["control_configs"] = {}
                        # Update control_configs in best_simulation_result
                        best_configs_extracted = extract_configs_only(self.best_control_configs) or {}
                        self.best_simulation_result["control_configs"].update(best_configs_extracted)
                    
                    # Update best_simulation_turn if any module improved
                    if improved_modules:
                        self.best_simulation_turn = self.turn_count
                        if verbose:
                            print(f"✓ New best result achieved for modules: {', '.join(improved_modules)} at turn {self.turn_count}")
                    else:
                        # Current result is worse for all modules - rollback to best configs
                        if self.best_control_configs:
                            if verbose:
                                print(f"⚠ Current policy performs worse than best. Rolling back to best configs from turn {self.best_simulation_turn}...")
                            
                            # Rollback: restore best configs
                            self.current_control_configs = copy.deepcopy(self.best_control_configs)
                            
                            best_taxi_entry = self.best_control_configs.get("taxi_scheduling")
                            if best_taxi_entry is not None:
                                cfg = best_taxi_entry.get("config", best_taxi_entry) if isinstance(best_taxi_entry, dict) and "config" in best_taxi_entry else best_taxi_entry
                                context["current_taxi_config"] = cfg.copy() if isinstance(cfg, dict) else {}
                            
                            # Update context with best configs (bundle format; simulation_utils uses _effective_config)
                            context["control_configs"] = self.best_control_configs.copy()
                            
                            if verbose:
                                print(f"✓ Rolled back to best configs from turn {self.best_simulation_turn}")
                
                # Provide feedback to LLM with comparison to best result
                feedback = self._format_simulation_result(sim_result, compare_to_best=True, best_result=best_result_before_update)
                feedback = "[Auto-executed after POLICY_PLANNING]\n\n" + feedback
                
                # Add notification if any module achieved new best result
                if sim_result.get("success"):
                    improved_modules = [name for name, is_better in module_comparison.items() if is_better]
                    
                    if improved_modules:
                        feedback += f"\n\n🎉 **New Best Result Achieved**: "
                        feedback += f"Modules {', '.join(improved_modules)} achieved better performance"
                        feedback += f" in turn {self.turn_count}!"
                        feedback += f"\nYou can continue optimizing based on this round's results to further improve performance."
                
                # Add rollback notification if rollback occurred
                if sim_result.get("success"):
                    any_better = any(module_comparison.values())
                    if not any_better and self.best_control_configs:
                        feedback += f"\n\n⚠️ **Policy Rollback**: The current policy performed worse than the best policy (from turn {self.best_simulation_turn})."
                        feedback += f"\nThe system has automatically rolled back to the best-performing configuration."
                        feedback += f"\nYour current working configuration is now the best one from turn {self.best_simulation_turn}."
                        feedback += f"\nYou can continue refining from this best configuration."
                
                self.add_user_message(feedback)
                
                # Build return payload including improved modules info if any
                payload: Dict[str, Any] = {
                    "simulation_result": sim_result,
                    "planning_result": result,
                    "best_result_before_update": best_result_before_update,  # ✅ 更新前的最佳结果（用于计算指标改进奖励）
                    "initial_best_simulation_result": self.initial_best_simulation_result,  # ✅ 初始 baseline（用于计算相对于 baseline 的总改进）
                }
                if sim_result.get("success"):
                    payload["improved_modules"] = improved_modules
                    payload["general_better"] = False
                
                return payload
            else:
                warning_msg = "Warning: POLICY_PLANNING succeeded but no control_configs available for simulation. Please check your policy code."
                if verbose:
                    print(warning_msg)
                self.add_user_message(warning_msg)
                return {"error": warning_msg, "planning_result": result} 
        else:
            return {"planning_result": result}
    
    def _handle_debug_action(
        self,
        action_content: str,
        context: Dict[str, Any],
        env: Optional[Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Handle DEBUG action."""
        if self.last_code_result is None:
            error_msg = "No previous code execution found. Use `ACTION: DATA_ANALYSIS` or `ACTION: POLICY_PLANNING` action first."
            self.add_user_message(error_msg)
            if verbose:
                print(f"Error: {error_msg}")
            return {"error": error_msg}
        
        # Extract debug type from action content
        debug_type = self._extract_debug_type(action_content)
        
        if not debug_type:
            error_msg = "DEBUG action must specify 'Debug Type: DATA_ANALYSIS' or 'Debug Type: POLICY_PLANNING'."
            self.add_user_message(error_msg)
            if verbose:
                print(f"Error: {error_msg}")
            return {"error": error_msg}
        
        # Extract code from DEBUG action
        code = extract_code_block(action_content) if action_content else None
        if not code:
            error_msg = "DEBUG action must contain code block. Please provide fixed code."
            self.add_user_message(error_msg)
            if verbose:
                print(f"Error: {error_msg}")
            return {"error": error_msg}
        
        # Execute debug code
        result = execute_code(code, context, verbose=verbose)
        self.last_code_result = result
        
        # Handle based on debug type
        if debug_type == "DATA_ANALYSIS":
            self._handle_data_analysis_result(result, verbose=verbose)
            return {"debug_result": result}
        
        elif debug_type == "POLICY_PLANNING":
            # Extract specified control modules
            specified_modules = self._extract_control_modules(action_content)
            
            if not specified_modules:
                error_msg = "DEBUG with POLICY_PLANNING must specify 'Control Modules: <module1>, <module2>' line."
                self.add_user_message(error_msg)
                if verbose:
                    print(f"Error: {error_msg}")
                return {"error": error_msg}
            
            # Handle as POLICY_PLANNING
            planning_success = self._handle_policy_planning_result(
                result=result,
                specified_modules=specified_modules,
                env=env,
                context=context,
                verbose=verbose
            )
            
            # Automatically execute simulation after successful POLICY_PLANNING (same as normal POLICY_PLANNING)
            if planning_success:
                control_configs = context.get("control_configs")
                
                if control_configs is None:
                    control_configs = {}
                
                if control_configs:
                    context["available_control_modules"] = self.available_control_modules
                    
                    sim_result = execute_simulation_action(
                        control_configs=control_configs,
                        context=context,
                        verbose=verbose
                    )
                    
                    # Similar update logic as POLICY_PLANNING
                    best_result_before_update_debug = None
                    if self.best_simulation_result is not None:
                        best_result_before_update_debug = copy.deepcopy(self.best_simulation_result)
                    
                    if sim_result.get("success"):
                        module_comparison = self._is_better_simulation_result(sim_result, self.best_simulation_result)
                        
                        if self.best_simulation_result is None:
                            self.best_simulation_result = {
                                "success": True,
                                "stats": {},
                                "module_metrics": {}
                            }
                        
                        improved_modules = []
                        for module_name, is_better in module_comparison.items():
                            if is_better:
                                if module_name in control_configs:
                                    if self.best_control_configs is None:
                                        self.best_control_configs = {}
                                    entry = self.current_control_configs.get(module_name)
                                    if isinstance(entry, dict) and "module" in entry and "config" in entry:
                                        self.best_control_configs[module_name] = {
                                            "module": entry["module"],
                                            "config": control_configs[module_name].copy(),
                                        }
                                    else:
                                        self.best_control_configs[module_name] = control_configs[module_name].copy()
                                    improved_modules.append(module_name)
                                
                                current_module_metrics = sim_result.get("module_metrics", {}).get(module_name)
                                if current_module_metrics:
                                    if "module_metrics" not in self.best_simulation_result:
                                        self.best_simulation_result["module_metrics"] = {}
                                    self.best_simulation_result["module_metrics"][module_name] = current_module_metrics.copy()
                        
                        # Update best_simulation_result["control_configs"] when any module improved
                        if improved_modules:
                            from utils.checkpoint_logger import extract_configs_only
                            if "control_configs" not in self.best_simulation_result:
                                self.best_simulation_result["control_configs"] = {}
                            # Update control_configs in best_simulation_result
                            best_configs_extracted = extract_configs_only(self.best_control_configs) or {}
                            self.best_simulation_result["control_configs"].update(best_configs_extracted)
                        
                        if improved_modules:
                            self.best_simulation_turn = self.turn_count
                            if verbose:
                                print(f"✓ New best result achieved for modules: {', '.join(improved_modules)} at turn {self.turn_count}")
                        else:
                            if self.best_control_configs:
                                if verbose:
                                    print(f"⚠ Current policy performs worse than best. Rolling back to best configs from turn {self.best_simulation_turn}...")
                                
                                self.current_control_configs = copy.deepcopy(self.best_control_configs)
                                
                                best_taxi_entry = self.best_control_configs.get("taxi_scheduling")
                                if best_taxi_entry is not None:
                                    cfg = best_taxi_entry.get("config", best_taxi_entry) if isinstance(best_taxi_entry, dict) and "config" in best_taxi_entry else best_taxi_entry
                                    context["current_taxi_config"] = cfg.copy() if isinstance(cfg, dict) else {}
                                
                                context["control_configs"] = self.best_control_configs.copy()
                                
                                if verbose:
                                    print(f"✓ Rolled back to best configs from turn {self.best_simulation_turn}")
                    
                    feedback = self._format_simulation_result(sim_result, compare_to_best=True, best_result=best_result_before_update_debug)
                    feedback = "[Auto-executed after DEBUG POLICY_PLANNING]\n\n" + feedback
                    
                    if sim_result.get("success"):
                        improved_modules = [name for name, is_better in module_comparison.items() if is_better]
                        
                        if improved_modules:
                            feedback += f"\n\n🎉 **New Best Result Achieved**: "
                            feedback += f"Modules {', '.join(improved_modules)} achieved better performance"
                            feedback += f" in turn {self.turn_count}!"
                            feedback += f"\nYou can continue optimizing based on this round's results to further improve performance."
                    
                    if sim_result.get("success"):
                        any_better = any(module_comparison.values())
                        if not any_better and self.best_control_configs:
                            feedback += f"\n\n⚠️ **Policy Rollback**: The current policy performed worse than the best policy (from turn {self.best_simulation_turn})."
                            feedback += f"\nThe system has automatically rolled back to the best-performing configuration."
                            feedback += f"\nYour current working configuration is now the best one from turn {self.best_simulation_turn}."
                            feedback += f"\nYou can continue refining from this best configuration."
                    
                    self.add_user_message(feedback)
                    
                    # Build return payload including improved modules info if any
                    payload: Dict[str, Any] = {
                        "simulation_result": sim_result,
                        "debug_result": result,
                        "best_result_before_update": best_result_before_update_debug,  # ✅ 更新前的最佳结果（用于计算指标改进奖励）
                        "initial_best_simulation_result": self.initial_best_simulation_result,  # ✅ 初始 baseline（用于计算相对于 baseline 的总改进）
                    }
                    if sim_result.get("success"):
                        payload["improved_modules"] = improved_modules
                        payload["general_better"] = False
                    
                    return payload
                else:
                    warning_msg = "Warning: DEBUG POLICY_PLANNING succeeded but no control_configs available for simulation. Please check your policy code."
                    if verbose:
                        print(warning_msg)
                    self.add_user_message(warning_msg)
                    return {"error": warning_msg, "debug_result": result}
            else:
                return {"debug_result": result}

        return {"debug_result": result}

    def _handle_get_control_api_action(
        self,
        action_content: str,
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Handle GET_CONTROL_API action.

        Enhanced for joint control mode to include:
        - Module validation against enabled modules
        - Current performance metrics
        - Domain knowledge
        - Current configuration summary
        """
        module_names = self._extract_module_names(action_content)

        # Show only enabled modules in joint control mode
        if self.is_joint_control and self.available_control_modules:
            available_modules = self.available_control_modules
        else:
            available_modules = list(self._control_specs.keys())

        if not module_names:
            error_msg = (
                "Module name not specified. Please use format:\n"
                "ACTION: GET_CONTROL_API\n"
                "Module: <module_name1>, <module_name2>\n\n"
                f"Available modules: {available_modules}"
            )
            if verbose:
                print(f"Error: {error_msg}")
            self.add_user_message(error_msg)
            return {"error": error_msg}

        valid_modules = []
        invalid_modules = []
        unknown_modules = []
        for module_name in module_names:
            if module_name not in self._control_specs:
                unknown_modules.append(module_name)
                continue
            if self.is_joint_control and self.available_control_modules and module_name not in self.available_control_modules:
                invalid_modules.append(module_name)
                continue
            valid_modules.append(module_name)

        if not valid_modules:
            error_msg = (
                f"No valid modules requested. Available modules: {available_modules}\n"
                "Please query only enabled modules."
            )
            if unknown_modules:
                error_msg += f"\nUnknown modules: {', '.join(unknown_modules)}"
            if invalid_modules:
                error_msg += f"\nNot enabled: {', '.join(invalid_modules)}"
            if verbose:
                print(f"Error: {error_msg}")
            self.add_user_message(error_msg)
            return {"error": error_msg}

        feedback_parts = []
        api_info_by_module: Dict[str, str] = {}

        # Add current configuration summary if available
        current_configs = {}
        if context:
            if isinstance(context.get("current_configs"), dict):
                current_configs = context["current_configs"]
            elif isinstance(context.get("control_configs"), dict):
                current_configs = context["control_configs"]
        if not current_configs and self.current_control_configs:
            current_configs = self.current_control_configs

        for module_name in valid_modules:
            spec = self._control_specs[module_name]
            api_info = self._format_module_api(module_name, spec)
            api_info_by_module[module_name] = api_info

            feedback_parts.append(f"# {module_name.replace('_', ' ').title()} Control Module API\n")

            # Add current performance metrics (for joint control mode)
            if self.is_joint_control and self.module_metrics and module_name in self.module_metrics:
                feedback_parts.append("## Current Performance Metrics")
                metrics = self.module_metrics[module_name]
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        feedback_parts.append(f"- {metric_name.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        feedback_parts.append(f"- {metric_name.replace('_', ' ').title()}: {value}")
                feedback_parts.append("")

            feedback_parts.append("## Current Configuration Summary")
            if module_name in current_configs:
                current_config = current_configs.get(module_name)
                if isinstance(current_config, dict):
                    config_keys = list(current_config.keys())
                    feedback_parts.append(f"- Entries: {len(config_keys)}")
                    if config_keys:
                        preview_keys = ", ".join(config_keys[:5])
                        feedback_parts.append(f"- Example keys: {preview_keys}")
                else:
                    feedback_parts.append(f"- Type: {type(current_config).__name__}")
            else:
                feedback_parts.append("- No current configuration available for this module.")
            feedback_parts.append("")

            if spec.get("domain_knowledge"):
                feedback_parts.append("## Domain Knowledge")
                feedback_parts.append(spec.get("domain_knowledge"))
                feedback_parts.append("")

            feedback_parts.append("## API Documentation")
            feedback_parts.append(api_info)
            feedback_parts.append("")

        feedback_parts.append(
            "You can now use this information to write code for DATA_ANALYSIS or POLICY_PLANNING actions."
        )
        feedback = "\n".join(feedback_parts).strip()

        if verbose:
            if len(valid_modules) == 1:
                print(f"API query for module: {valid_modules[0]}")
            else:
                print(f"API query for modules: {', '.join(valid_modules)}")

        self.add_user_message(feedback)
        payload: Dict[str, Any] = {"modules": valid_modules, "api_info": api_info_by_module}
        if len(valid_modules) == 1:
            payload["module"] = valid_modules[0]
        if unknown_modules:
            payload["unknown_modules"] = unknown_modules
        if invalid_modules:
            payload["invalid_modules"] = invalid_modules
        return payload

    def _handle_select_modules_action(
        self,
        action_content: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Handle SELECT_MODULES action for joint control module selection.

        Parses the LLM response to extract selected modules and reasoning.
        Used in joint control optimization to let the LLM choose which
        modules to optimize at each checkpoint.

        Args:
            action_content: The full LLM response containing module selection
            verbose: Whether to print detailed logs

        Returns:
            Dictionary containing:
            - selected_modules: List of selected module names
            - reasoning: LLM's reasoning for the selection
            - error: Error message if parsing failed
        """
        selected_modules = []
        reasoning = ""

        lines = action_content.split('\n')
        for line in lines:
            line_lower = line.lower().strip()

            # Parse selected modules line
            if 'selected modules:' in line_lower or 'selected module:' in line_lower:
                if 'selected modules:' in line_lower:
                    modules_str = line.split(':', 1)[1].strip()
                else:
                    modules_str = line.split(':', 1)[1].strip()

                # Handle various formats: comma-separated, bracketed, quoted
                modules_str = modules_str.replace('[', '').replace(']', '')
                modules_str = modules_str.replace('"', '').replace("'", '')
                selected_modules = [m.strip() for m in modules_str.split(',') if m.strip()]

            # Parse reasoning line
            if 'reasoning:' in line_lower:
                reasoning = line.split(':', 1)[1].strip()

        # Validate selected modules against available modules
        if self.available_control_modules:
            valid_modules = []
            invalid_modules = []
            for module in selected_modules:
                if module in self.available_control_modules:
                    valid_modules.append(module)
                else:
                    invalid_modules.append(module)

            if invalid_modules:
                if verbose:
                    print(f"Warning: Invalid modules ignored: {invalid_modules}")

            selected_modules = valid_modules

        if not selected_modules:
            error_msg = (
                "SELECT_MODULES action must include 'Selected Modules: module1, module2' line.\n"
                f"Available modules: {self.available_control_modules or 'Not specified'}"
            )
            if verbose:
                print(f"Error: {error_msg}")
            self.add_user_message(error_msg)
            return {"error": error_msg}

        if verbose:
            print(f"Selected modules: {selected_modules}")
            if reasoning:
                print(f"Reasoning: {reasoning}")

        # Provide feedback
        feedback = f"Module selection received:\n"
        feedback += f"- Selected modules: {', '.join(selected_modules)}\n"
        if reasoning:
            feedback += f"- Reasoning: {reasoning}\n"
        feedback += "\nYou can now proceed with DATA_ANALYSIS or POLICY_PLANNING for these modules."

        self.add_user_message(feedback)

        return {
            "selected_modules": selected_modules,
            "reasoning": reasoning
        }

    def _normalize_transit_policy_config(
        self,
        module_name: str,
        policy_config: Dict[str, Any],
        reference_config: Optional[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Normalize bus/subway scheduling configs into timetable format.

        Accepts partial updates like {"route_id": {"headway": 180}} and
        merges them with the reference timetable/schedule.
        """
        if module_name not in ("bus_scheduling", "subway_scheduling"):
            return policy_config
        if not isinstance(policy_config, dict):
            return policy_config
        if not isinstance(reference_config, dict) or not reference_config:
            return policy_config

        route_ids = set(reference_config.keys())
        policy_keys = set(policy_config.keys())

        def _apply_route_updates(base_route: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            base_route = copy.deepcopy(base_route) if isinstance(base_route, dict) else {}
            if not isinstance(updates, dict):
                return base_route

            if "timetable" in updates:
                updated_timetable = updates.get("timetable")
                if isinstance(updated_timetable, list):
                    base_timetable = base_route.get("timetable", [])
                    if not isinstance(base_timetable, list):
                        base_timetable = []
                    merged_timetable = copy.deepcopy(base_timetable)
                    for idx, segment in enumerate(updated_timetable):
                        if not isinstance(segment, dict):
                            continue
                        if idx < len(merged_timetable) and isinstance(merged_timetable[idx], dict):
                            merged = merged_timetable[idx]
                            if "time_range" in segment:
                                merged["time_range"] = segment["time_range"]
                            if "headway" in segment:
                                merged["headway"] = segment["headway"]
                            if "schedule" in segment:
                                merged["schedule"] = segment["schedule"]
                            if "dwell_time" in segment and "schedule" in merged:
                                for stop in merged.get("schedule", []):
                                    if isinstance(stop, dict):
                                        stop["dwell_time"] = segment["dwell_time"]
                            merged_timetable[idx] = merged
                        else:
                            merged_timetable.append(segment)
                    base_route["timetable"] = merged_timetable
                else:
                    base_route["timetable"] = updated_timetable
            else:
                headway = updates.get("headway")
                schedule = updates.get("schedule")
                dwell_time = updates.get("dwell_time")

                if isinstance(base_route.get("timetable"), list):
                    for segment in base_route["timetable"]:
                        if not isinstance(segment, dict):
                            continue
                        if headway is not None:
                            segment["headway"] = headway
                        if schedule is not None:
                            segment["schedule"] = schedule
                        if dwell_time is not None and isinstance(segment.get("schedule"), list):
                            for stop in segment["schedule"]:
                                if isinstance(stop, dict):
                                    stop["dwell_time"] = dwell_time
                else:
                    # Fallback: build a single-segment timetable
                    base_route["timetable"] = [{
                        "time_range": [0, 3600],
                        "headway": headway if headway is not None else 300,
                        "schedule": schedule if schedule is not None else base_route.get("schedule", []),
                    }]

            return base_route

        # Global updates (no route ids specified) -> apply to all routes
        if not (policy_keys & route_ids):
            normalized = {}
            for route_id in route_ids:
                normalized[route_id] = _apply_route_updates(reference_config.get(route_id, {}), policy_config)
            if verbose:
                print(f"[Normalize] Applied global {module_name} updates to {len(normalized)} routes")
            return normalized

        normalized = {}
        for route_id, updates in policy_config.items():
            if route_id in route_ids:
                normalized[route_id] = _apply_route_updates(reference_config.get(route_id, {}), updates)
            else:
                # Keep unknown routes as-is
                normalized[route_id] = updates
        return normalized

    def _handle_policy_planning_result(
        self,
        result: Dict[str, Any],
        specified_modules: List[str],
        env: Optional[Any],
        context: Dict[str, Any],
        verbose: bool = True
    ) -> bool:
        """
        Handle POLICY_PLANNING action result.
        Validates, merges, and updates control module configurations.
        
        Returns:
            True if policy planning succeeded and configs were updated, False otherwise
        """
        # Check if code returned configurations
        if result.get("success") and result.get("return_value"):
            returned_value = result["return_value"]
            
            # Check if it's a multi-module configuration dict
            if isinstance(returned_value, dict):
                # If a single module was specified and the LLM returned a flat config dict,
                # wrap it under the module name
                if len(specified_modules) == 1:
                    module_name = specified_modules[0]
                    returned_keys = set(returned_value.keys())
                    available_modules = set(self.available_control_modules or [])
                    if module_name not in returned_keys and not (returned_keys & available_modules):
                        returned_value = {module_name: returned_value}
                        result["return_value"] = returned_value
                        if verbose:
                            print(f"Info: Wrapped single-module config under '{module_name}'.")

                returned_keys = set(returned_value.keys())
                expected_modules = set(specified_modules)
                if self.available_control_modules:
                    expected_modules.update(self.available_control_modules)
                if not (returned_keys & expected_modules):
                    feedback = self._format_code_result(result, check_control_config=True)
                    feedback += (
                        "\nPOLICY_PLANNING must return a dict keyed by module names "
                        "(e.g., {'signal_timing': {...}})."
                    )
                    feedback += f"\nReturned keys: {sorted(returned_keys) if returned_keys else '[]'}"
                    feedback += f"\nExpected module keys (specified): {sorted(specified_modules)}"
                    self.add_user_message(feedback)
                    return False
                
                updated_modules = []
                validation_errors = []
                recorded_decisions = []  # Track decisions for context manager

                # ✅ 兼容式写法：优先使用 self.current_control_configs（VERL 分布式模式）
                # 如果为空，则尝试从 env.enabled_controls 获取（原始单进程模式）
                current_control_configs = self.current_control_configs
                
                # Fallback: 如果 self.current_control_configs 为空，尝试从 env 获取
                if not current_control_configs and env and hasattr(env, 'enabled_controls'):
                    for module_name, module_info in env.enabled_controls.items():
                        module = module_info.get('module')
                        config = module_info.get('config', {})
                        if module and config and module_name in (self.available_control_modules or []):
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
                        if verbose:
                            print(f"Warning: {module_name} is not enabled in the environment, skipping...")
                        continue
                    
                    module_info = current_control_configs[module_name]
                    # Accept both bundle format {'module': ..., 'config': ...} and config-only (e.g. after rollback)
                    if isinstance(module_info, dict) and "module" in module_info and "config" in module_info:
                        module = module_info["module"]
                        current_config = module_info["config"].copy()
                    else:
                        module = None
                        current_config = copy.deepcopy(module_info) if isinstance(module_info, dict) else {}
                        if env and hasattr(env, "enabled_controls") and module_name in env.enabled_controls:
                            module = env.enabled_controls[module_name].get("module")
                    old_config = copy.deepcopy(current_config)  # Capture old config for decision context

                    if module is None:
                        validation_errors.append(f"{module_name}: Module not available")
                        continue
                    
                    # Get reference config for completeness check
                    reference_config = None
                    if module_name in current_control_configs:
                        entry = current_control_configs[module_name]
                        reference_config = entry.get("config", entry) if isinstance(entry, dict) else {}

                    # Normalize transit configs (bus/subway) into timetable format
                    policy_config = self._normalize_transit_policy_config(
                        module_name=module_name,
                        policy_config=policy_config,
                        reference_config=reference_config,
                        verbose=verbose
                    )

                    # Validate policy config using module's validate_config
                    is_valid, error_msg = module.validate_config(policy_config, reference_config=reference_config)
                    if not is_valid:
                        error_detail = f": {error_msg}" if error_msg else ""
                        validation_errors.append(f"{module_name}: Invalid configuration format{error_detail}")
                        if verbose:
                            print(f"Warning: Invalid {module_name} configuration format{error_detail}")
                        continue
                    
                    # Update current config with policy config
                    current_config.update(policy_config)
                    
                    if module_name == "taxi_scheduling":
                        context["current_taxi_config"] = current_config
                    
                    # Store in agent's internal control_configs
                    # ✅ 保持完整的 {module, config} 结构
                    self.current_control_configs[module_name] = {
                        'module': module,
                        'config': current_config
                    }
                    
                    # Update context with control_configs for simulation
                    if "control_configs" not in context:
                        context["control_configs"] = {}
                    context["control_configs"][module_name] = current_config

                    # Keep env.enabled_controls in sync so metrics use the latest config
                    if env and hasattr(env, "enabled_controls") and module_name in env.enabled_controls:
                        env.enabled_controls[module_name]["config"] = copy.deepcopy(current_config)

                    updated_modules.append(f"{module_name} ({len(policy_config)} entries updated)")

                    # Capture decision context for cross-module coordination
                    decision_summary, decision_changes, decision_focus = self._extract_decision_context(
                        module_name, old_config, policy_config
                    )
                    if decision_summary and self.decision_context_manager is not None:
                        self.decision_context_manager.record_decision(
                            module=module_name,
                            summary=decision_summary,
                            changes=decision_changes,
                            optimization_focus=decision_focus
                        )
                        recorded_decisions.append(module_name)
                
                if updated_modules:
                    if verbose:
                        print(f"\n✓ Configuration updated: {', '.join(updated_modules)}")
                    feedback = self._format_code_result(result, check_control_config=True)
                    feedback += f"\n\nConfiguration updated for: {', '.join(updated_modules)}"
                    if validation_errors:
                        feedback += f"\n\nValidation errors: {'; '.join(validation_errors)}"
                    feedback += "\n[Note: Simulation will be automatically executed to test this configuration.]"
                    self.add_user_message(feedback)
                    return True
                else:
                    feedback = self._format_code_result(result, check_control_config=True)
                    if validation_errors:
                        feedback += f"\n\nValidation errors: {'; '.join(validation_errors)}"
                    feedback += "\nNo valid control configurations were updated. Please check your code."
                    self.add_user_message(feedback)
                    return False
            else:
                feedback = self._format_code_result(result, check_control_config=True)
                feedback += "\nPOLICY_PLANNING must return a dictionary of control configurations."
                self.add_user_message(feedback)
                return False
        else:
            feedback = self._format_code_result(result, check_control_config=True)
            feedback += "\nPOLICY_PLANNING code must return control configurations."
            self.add_user_message(feedback)
            return False

    def _extract_decision_context(
        self,
        module_name: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        """
        Extract decision context from config changes.

        Args:
            module_name: Name of the control module
            old_config: Previous configuration
            new_config: New configuration (policy changes)

        Returns:
            Tuple of (summary, changes, optimization_focus)
        """
        summary = self._extract_decision_summary(module_name, old_config, new_config)
        changes = self._compute_config_diff(module_name, old_config, new_config)
        focus = self._extract_optimization_focus()
        return summary, changes, focus

    def _extract_decision_summary(
        self,
        module_name: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary of what changed.

        Args:
            module_name: Name of the control module
            old_config: Previous configuration
            new_config: New configuration (policy changes)

        Returns:
            Summary string like "Increased phase A at 5 intersections"
        """
        if not new_config:
            return ""

        changed_entities = list(new_config.keys())
        num_changes = len(changed_entities)

        if module_name == "signal_timing":
            # Analyze signal timing changes
            increased_count = 0
            decreased_count = 0

            for entity_id, phases in new_config.items():
                if not isinstance(phases, dict):
                    continue
                old_phases = old_config.get(entity_id, {})
                if not isinstance(old_phases, dict):
                    continue
                for phase, duration in phases.items():
                    old_duration = old_phases.get(phase, duration)
                    if duration > old_duration:
                        increased_count += 1
                    elif duration < old_duration:
                        decreased_count += 1

            if increased_count > decreased_count:
                action = "Increased"
                phase_count = increased_count
            elif decreased_count > increased_count:
                action = "Decreased"
                phase_count = decreased_count
            else:
                action = "Adjusted"
                phase_count = increased_count + decreased_count

            sample_entities = changed_entities[:3]
            return f"{action} phase durations at {num_changes} intersections ({', '.join(sample_entities)})"

        elif module_name == "highway_speed_limit":
            # Analyze VSL changes
            speeds = []
            for entity_id, speed in new_config.items():
                if isinstance(speed, (int, float)):
                    speeds.append(speed)
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                sample_entities = changed_entities[:3]
                return f"Set speed limits on {num_changes} segments (avg {avg_speed:.1f} km/h, {', '.join(sample_entities)})"
            return f"Updated speed limits on {num_changes} segments"

        elif module_name == "ramp_metering":
            # Analyze ramp metering changes
            sample_entities = changed_entities[:3]
            return f"Adjusted metering at {num_changes} ramps ({', '.join(sample_entities)})"

        elif module_name == "bus_scheduling":
            # Analyze bus scheduling changes
            sample_entities = changed_entities[:3]
            return f"Updated schedules for {num_changes} bus routes ({', '.join(sample_entities)})"

        elif module_name == "subway_scheduling":
            sample_entities = changed_entities[:3]
            return f"Updated schedules for {num_changes} subway lines ({', '.join(sample_entities)})"

        elif module_name == "taxi_scheduling":
            dispatch_count = len(new_config.get("dispatch_decisions") or [])
            reposition_count = len(new_config.get("reposition_decisions") or [])
            return f"Updated taxi scheduling ({dispatch_count} dispatch, {reposition_count} reposition decisions)"

        return f"Updated {num_changes} entries for {module_name}"

    def _compute_config_diff(
        self,
        module_name: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Compute detailed diff between configs.

        Args:
            module_name: Name of the control module
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            List of change dicts with entity, action, details
        """
        changes = []

        if module_name == "signal_timing":
            for entity_id, phases in new_config.items():
                if not isinstance(phases, dict):
                    continue
                old_phases = old_config.get(entity_id, {})
                if not isinstance(old_phases, dict):
                    old_phases = {}

                for phase, duration in phases.items():
                    old_duration = old_phases.get(phase, duration)
                    if duration != old_duration:
                        delta = duration - old_duration
                        action = "increased" if delta > 0 else "decreased"
                        changes.append({
                            "entity": entity_id,
                            "action": action,
                            "phase": phase,
                            "old_value": old_duration,
                            "new_value": duration,
                            "delta": delta
                        })

        elif module_name == "highway_speed_limit":
            for entity_id, schedule in new_config.items():
                old_schedule = old_config.get(entity_id, schedule)
                # Handle list of schedule entries: [{'time': 0, 'speed_limit': 55}, ...]
                if isinstance(schedule, list) and isinstance(old_schedule, list):
                    # Extract the initial speed_limit from each schedule for comparison
                    new_speed = schedule[0].get('speed_limit', 0) if schedule else 0
                    old_speed = old_schedule[0].get('speed_limit', 0) if old_schedule else 0
                    if schedule != old_schedule:
                        if new_speed != old_speed:
                            action = "increased" if new_speed > old_speed else "decreased"
                        else:
                            action = "modified"  # Schedule changed but initial speed same
                        changes.append({
                            "entity": entity_id,
                            "action": action,
                            "old_value": old_schedule,
                            "new_value": schedule
                        })
                elif schedule != old_schedule:
                    # Fallback for simple numeric values (backward compatibility)
                    if isinstance(schedule, (int, float)) and isinstance(old_schedule, (int, float)):
                        action = "increased" if schedule > old_schedule else "decreased"
                    else:
                        action = "modified"
                    changes.append({
                        "entity": entity_id,
                        "action": action,
                        "old_value": old_schedule,
                        "new_value": schedule
                    })

        elif module_name == "ramp_metering":
            for entity_id, config in new_config.items():
                if entity_id not in old_config:
                    changes.append({
                        "entity": entity_id,
                        "action": "added",
                        "new_value": config
                    })
                elif config != old_config.get(entity_id):
                    changes.append({
                        "entity": entity_id,
                        "action": "modified",
                        "old_value": old_config.get(entity_id),
                        "new_value": config
                    })

        else:
            # Generic diff for other modules
            for entity_id, value in new_config.items():
                old_value = old_config.get(entity_id)
                if old_value != value:
                    changes.append({
                        "entity": entity_id,
                        "action": "modified" if old_value else "added",
                        "old_value": old_value,
                        "new_value": value
                    })

        return changes

    def _extract_optimization_focus(self) -> Optional[str]:
        """
        Extract optimization focus from LLM's reasoning in recent messages.

        Looks for keywords indicating focus areas like congestion, queue, direction, etc.

        Returns:
            Optimization focus string or None
        """
        # Look at recent assistant messages for reasoning
        keywords_map = {
            "northbound": "northbound_traffic",
            "southbound": "southbound_traffic",
            "eastbound": "eastbound_traffic",
            "westbound": "westbound_traffic",
            "congestion": "congestion_reduction",
            "queue": "queue_reduction",
            "rush hour": "rush_hour_optimization",
            "morning rush": "morning_rush",
            "evening rush": "evening_rush",
            "throughput": "throughput_improvement",
            "delay": "delay_reduction",
            "travel time": "travel_time_reduction",
            "waiting": "waiting_reduction",
            "bunching": "bus_bunching",
            "headway": "headway_optimization",
            "dwell": "dwell_time_optimization",
            "ramp": "ramp_flow_control",
            "speed": "speed_harmonization"
        }

        focus_candidates = []

        # Search recent messages for keywords
        for msg in reversed(self.messages[-6:]):  # Check last 6 messages
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "").lower()
            for keyword, focus in keywords_map.items():
                if keyword in content and focus not in focus_candidates:
                    focus_candidates.append(focus)

        if focus_candidates:
            return focus_candidates[0]  # Return most recent/relevant focus

        return None

    def _format_code_result(self, result: Dict[str, Any], check_control_config: bool = False) -> str:
        """
        Format code execution result as feedback message.

        Args:
            result: Code execution result dictionary
            check_control_config: Whether to check if return_value is a valid control config.
                                 Set to False for DATA_ANALYSIS actions to avoid validation errors.
        """
        if result.get("success"):
            output = result.get("output", "")
            return_value = result.get("return_value")
            
            feedback = "Code executed successfully.\n"
            if output:
                # Limit output to first 10000 characters to avoid overly long feedback
                original_output_len = len(output)
                if original_output_len > 10000:
                    output = output[:10000] + f"\n... (output truncated, {original_output_len - 10000} more characters)"
                feedback += f"\nOutput:\n{output}\n"
            if return_value is not None:
                # Only check for control configs if explicitly requested (e.g., in POLICY_PLANNING)
                if check_control_config and isinstance(return_value, dict):
                    try:
                        if self._is_valid_control_config(return_value):
                            module_names = list(return_value.keys())
                            feedback += f"\nControl configurations are valid and returned for modules: {', '.join(module_names)}"
                        else:
                            feedback += f"\nReturn value: {return_value}"
                    except Exception:
                        # If validation fails, just show the return value
                        feedback += f"\nReturn value: {return_value}"
                else:
                    feedback += f"\nReturn value: {return_value}"
            
            return feedback
        else:
            error = result.get("error", "Unknown error")
            return f"Code execution failed.\nError: {error}\n\nPlease fix the code and try again by using `ACTION: DEBUG`."
    
    def _format_simulation_result(self, result: Dict[str, Any], compare_to_best: bool = False, best_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Format simulation result as feedback message.

        Args:
            result: Simulation result dictionary
            compare_to_best: Whether to compare with best result
            best_result: Best result to compare against (if None, uses self.best_simulation_result)
        """
        if result.get("success"):
            stats = result.get("stats", {})
            feedback = "Simulation completed successfully.\n\n"
            feedback += "General Performance Metrics:\n"
            feedback += f"- Total departed vehicles: {stats.get('total_departed', 'N/A')}\n"
            # Prefer policy_avg_travel_time (checkpoint-specific) over avg_travel_time (global)
            travel_time = stats.get('policy_avg_travel_time') or stats.get('avg_travel_time', 'N/A')
            travel_time_label = "Policy avg travel time" if stats.get('policy_avg_travel_time') is not None else "Average travel time"
            feedback += f"- {travel_time_label}: {travel_time:.2f}s\n"
            if stats.get('policy_avg_travel_time') is not None and stats.get('avg_travel_time') is not None:
                feedback += f"- Global avg travel time: {stats.get('avg_travel_time', 'N/A'):.2f}s\n"
            feedback += f"- Simulation duration: {stats.get('duration', 'N/A'):.2f}s\n"
            
            # Add module-specific metrics if available
            # Only include modules that are in available_control_modules
            module_metrics = result.get("module_metrics", {})
            if module_metrics:
                # Filter module_metrics to only include available_control_modules
                filtered_module_metrics = {}
                if self.available_control_modules is not None:
                    for module_name in module_metrics.keys():
                        if module_name in self.available_control_modules:
                            filtered_module_metrics[module_name] = module_metrics[module_name]
                else:
                    # If available_control_modules is None, include all modules
                    filtered_module_metrics = module_metrics
                
                if filtered_module_metrics:
                    feedback += "\nControl Module Performance Metrics:\n"
                    for module_name, metrics in filtered_module_metrics.items():
                        feedback += f"\n{module_name.upper().replace('_', ' ')}:\n"
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, float):
                                feedback += f"  - {metric_name.replace('_', ' ').title()}: {metric_value:.2f}\n"
                            else:
                                feedback += f"  - {metric_name.replace('_', ' ').title()}: {metric_value}\n"
            
            # Compare with best result if requested and best exists
            best_result_to_compare = best_result if best_result is not None else self.best_simulation_result
            if compare_to_best and best_result_to_compare is not None:
                feedback += self._format_comparison_with_best(result, best_result=best_result_to_compare)
            
            return feedback
        else:
            error = result.get("error", "Unknown error")
            return f"Simulation failed.\nError: {error}"
    
    def _format_comparison_with_best(self, current_result: Dict[str, Any], best_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Format comparison between current result and best result.

        Args:
            current_result: Current simulation result
            best_result: Best result to compare against (if None, uses self.best_simulation_result)

        Returns:
            Formatted comparison string
        """
        best_result_to_compare = best_result if best_result is not None else self.best_simulation_result
        if best_result_to_compare is None or self.best_simulation_turn is None:
            return ""

        current_stats = current_result.get("stats", {})
        best_stats = best_result_to_compare.get("stats", {})

        # Prefer policy_avg_travel_time (checkpoint-specific) over avg_travel_time (global)
        # This ensures we compare policy simulations with checkpoint simulations using the same metric
        current_avg_travel_time = current_stats.get("policy_avg_travel_time") or current_stats.get("avg_travel_time", 0)
        best_avg_travel_time = best_stats.get("policy_avg_travel_time") or best_stats.get("avg_travel_time", 0)

        # Get queue length metrics from module metrics
        current_avg_queue_len = None
        best_avg_queue_len = None

        # Get highway speed limit metrics (most important: travel time and throughput)
        current_highway_travel_time = None
        best_highway_travel_time = None
        current_highway_throughput = None
        best_highway_throughput = None

        current_module_metrics = current_result.get("module_metrics", {})
        best_module_metrics = best_result_to_compare.get("module_metrics", {})

        # Filter module_metrics to only include available_control_modules
        if self.available_control_modules is not None:
            filtered_current_module_metrics = {
                name: metrics for name, metrics in current_module_metrics.items()
                if name in self.available_control_modules
            }
            filtered_best_module_metrics = {
                name: metrics for name, metrics in best_module_metrics.items()
                if name in self.available_control_modules
            }
        else:
            filtered_current_module_metrics = current_module_metrics
            filtered_best_module_metrics = best_module_metrics

        # Get policy_avg_travel_time from stats if available (preferred for comparison)
        current_stats_for_comparison = current_result.get("stats", {})
        best_stats_for_comparison = best_result_to_compare.get("stats", {}) if best_result_to_compare else {}

        comparison = "\n" + "=" * 60 + "\n"
        comparison += "Comparison with Best Result:\n"
        if self.best_simulation_turn == 0:
            comparison += "Best result: Checkpoint baseline (before optimization)\n\n"
        else:
            comparison += f"Best result achieved at turn {self.best_simulation_turn}\n\n"

        # # Compare average travel time (general metric)
        # travel_time_diff = current_avg_travel_time - best_avg_travel_time
        # if travel_time_diff < 0:
        #     comparison += f"✓ Average travel time: {current_avg_travel_time:.2f}s (IMPROVED by {abs(travel_time_diff):.2f}s from best: {best_avg_travel_time:.2f}s)\n"
        # elif travel_time_diff > 0:
        #     comparison += f"✗ Average travel time: {current_avg_travel_time:.2f}s (WORSE by {travel_time_diff:.2f}s from best: {best_avg_travel_time:.2f}s)\n"
        # else:
        #     comparison += f"= Average travel time: {current_avg_travel_time:.2f}s (SAME as best)\n"

        # Compare each module separately
        for module_name in filtered_current_module_metrics.keys():
            module_comparison = self._format_module_comparison(
                module_name,
                filtered_current_module_metrics.get(module_name, {}),
                filtered_best_module_metrics.get(module_name, {}),
                current_stats_for_comparison,
                best_stats_for_comparison
            )
            if module_comparison:
                comparison += "\n" + module_comparison

        comparison += "\n" + "=" * 60 + "\n"

        return comparison

    def _format_module_comparison(
        self,
        module_name: str,
        current_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any],
        current_stats: Dict[str, Any],
        best_stats: Dict[str, Any]
    ) -> str:
        """
        Format module-specific comparison between current and best metrics.

        Args:
            module_name: Name of the module
            current_metrics: Current module metrics
            best_metrics: Best module metrics
            current_stats: Current simulation stats
            best_stats: Best simulation stats

        Returns:
            Formatted comparison string for this module
        """
        if not current_metrics and not best_metrics:
            return ""

        comparison_lines = []
        comparison_lines.append(f"{module_name.upper().replace('_', ' ')} Module:")

        if module_name == "signal_timing":
            # Signal timing: compare travel time and waiting time (both lower is better)
            current_policy_tt = current_stats.get("policy_avg_travel_time")
            best_policy_tt = best_stats.get("policy_avg_travel_time")
            current_signal_tt = current_policy_tt if current_policy_tt is not None else current_metrics.get("avg_travel_time")
            best_signal_tt = best_policy_tt if best_policy_tt is not None else best_metrics.get("avg_travel_time")

            current_waiting_time = current_metrics.get("avg_waiting_time")
            best_waiting_time = best_metrics.get("avg_waiting_time")

            if current_signal_tt is not None and best_signal_tt is not None:
                tt_diff = current_signal_tt - best_signal_tt
                if tt_diff < 0:
                    comparison_lines.append(f"  ✓ Travel time: {current_signal_tt:.2f}s (IMPROVED by {abs(tt_diff):.2f}s from best: {best_signal_tt:.2f}s)")
                elif tt_diff > 0:
                    comparison_lines.append(f"  ✗ Travel time: {current_signal_tt:.2f}s (WORSE by {tt_diff:.2f}s from best: {best_signal_tt:.2f}s)")
                else:
                    comparison_lines.append(f"  = Travel time: {current_signal_tt:.2f}s (SAME as best)")

            if current_waiting_time is not None and best_waiting_time is not None:
                wait_diff = current_waiting_time - best_waiting_time
                if wait_diff < 0:
                    comparison_lines.append(f"  ✓ Waiting time: {current_waiting_time:.2f}s (IMPROVED by {abs(wait_diff):.2f}s from best: {best_waiting_time:.2f}s)")
                elif wait_diff > 0:
                    comparison_lines.append(f"  ✗ Waiting time: {current_waiting_time:.2f}s (WORSE by {wait_diff:.2f}s from best: {best_waiting_time:.2f}s)")
                else:
                    comparison_lines.append(f"  = Waiting time: {current_waiting_time:.2f}s (SAME as best)")

        elif module_name == "highway_speed_limit":
            # Highway speed limit: compare travel time and throughput
            current_highway_tt = current_metrics.get("avg_travel_time")
            best_highway_tt = best_metrics.get("avg_travel_time")
            current_throughput = current_metrics.get("throughput")
            best_throughput = best_metrics.get("throughput")

            if current_highway_tt is not None and best_highway_tt is not None:
                tt_diff = current_highway_tt - best_highway_tt
                if tt_diff < 0:
                    comparison_lines.append(f"  ✓ Travel time: {current_highway_tt:.2f}s (IMPROVED by {abs(tt_diff):.2f}s from best: {best_highway_tt:.2f}s)")
                elif tt_diff > 0:
                    comparison_lines.append(f"  ✗ Travel time: {current_highway_tt:.2f}s (WORSE by {tt_diff:.2f}s from best: {best_highway_tt:.2f}s)")
                else:
                    comparison_lines.append(f"  = Travel time: {current_highway_tt:.2f}s (SAME as best)")

            if current_throughput is not None and best_throughput is not None:
                throughput_diff = current_throughput - best_throughput
                if throughput_diff > 0:
                    comparison_lines.append(f"  ✓ Throughput: {int(current_throughput)} vehicles (IMPROVED by {int(throughput_diff)} from best: {int(best_throughput)})")
                elif throughput_diff < 0:
                    comparison_lines.append(f"  ✗ Throughput: {int(current_throughput)} vehicles (WORSE by {int(abs(throughput_diff))} from best: {int(best_throughput)})")
                else:
                    comparison_lines.append(f"  = Throughput: {int(current_throughput)} vehicles (SAME as best)")

        elif module_name == "ramp_metering":
            # Ramp metering: compare queue length and throughput (no travel time)
            current_queue_len = current_metrics.get("avg_queue_len")
            best_queue_len = best_metrics.get("avg_queue_len")
            current_throughput = current_metrics.get("throughput")
            best_throughput = best_metrics.get("throughput")

            if current_queue_len is not None and best_queue_len is not None:
                queue_diff = current_queue_len - best_queue_len
                if queue_diff < 0:
                    comparison_lines.append(f"  ✓ Queue length: {current_queue_len:.2f} (IMPROVED by {abs(queue_diff):.2f} from best: {best_queue_len:.2f})")
                elif queue_diff > 0:
                    comparison_lines.append(f"  ✗ Queue length: {current_queue_len:.2f} (WORSE by {queue_diff:.2f} from best: {best_queue_len:.2f})")
                else:
                    comparison_lines.append(f"  = Queue length: {current_queue_len:.2f} (SAME as best)")

            # Throughput comparison (higher is better)
            if current_throughput is not None and best_throughput is not None:
                throughput_diff = current_throughput - best_throughput
                if throughput_diff > 0:
                    comparison_lines.append(f"  ✓ Throughput: {current_throughput:.2f} (IMPROVED by {throughput_diff:.2f} from best: {best_throughput:.2f})")
                elif throughput_diff < 0:
                    comparison_lines.append(f"  ✗ Throughput: {current_throughput:.2f} (WORSE by {abs(throughput_diff):.2f} from best: {best_throughput:.2f})")
                else:
                    comparison_lines.append(f"  = Throughput: {current_throughput:.2f} (SAME as best)")

        elif module_name == "bus_scheduling":
            # Bus scheduling: prioritize waiting time/count; fuel savings when wait is comparable
            current_waiting_time = current_metrics.get("avg_passenger_waiting_time")
            best_waiting_time = best_metrics.get("avg_passenger_waiting_time")
            current_waiting_count = current_metrics.get("avg_passenger_waiting_count")
            best_waiting_count = best_metrics.get("avg_passenger_waiting_count")
            current_arrived_persons = current_metrics.get("total_arrived_persons")
            best_arrived_persons = best_metrics.get("total_arrived_persons")
            # Use total_fuel_consumption_g (actual metric name)
            current_fuel_g = current_metrics.get("total_fuel_consumption_g")
            best_fuel_g = best_metrics.get("total_fuel_consumption_g")

            # Fuel consumption (must be minimized)
            if current_fuel_g is not None and best_fuel_g is not None:
                fuel_diff = current_fuel_g - best_fuel_g
                if fuel_diff < 0:
                    comparison_lines.append(f"  ✓ Fuel consumption: {current_fuel_g:.2f}g (IMPROVED by {abs(fuel_diff):.2f}g from best: {best_fuel_g:.2f}g)")
                elif fuel_diff > 0:
                    comparison_lines.append(f"  ✗ Fuel consumption: {current_fuel_g:.2f}g (WORSE by {fuel_diff:.2f}g from best: {best_fuel_g:.2f}g)")
                else:
                    comparison_lines.append(f"  = Fuel consumption: {current_fuel_g:.2f}g (SAME as best)")

            # Service metrics (at least one must be improved): waiting time
            if current_waiting_time is not None and best_waiting_time is not None:
                wait_diff = current_waiting_time - best_waiting_time
                if wait_diff < 0:
                    comparison_lines.append(f"  ✓ Waiting time: {current_waiting_time:.2f}s (IMPROVED by {abs(wait_diff):.2f}s from best: {best_waiting_time:.2f}s)")
                elif wait_diff > 0:
                    comparison_lines.append(f"  ✗ Waiting time: {current_waiting_time:.2f}s (WORSE by {wait_diff:.2f}s from best: {best_waiting_time:.2f}s)")
                else:
                    comparison_lines.append(f"  = Waiting time: {current_waiting_time:.2f}s (SAME as best)")

            # Service metrics: waiting count
            if current_waiting_count is not None and best_waiting_count is not None:
                count_diff = current_waiting_count - best_waiting_count
                if count_diff < 0:
                    comparison_lines.append(f"  ✓ Waiting count: {current_waiting_count:.2f} (IMPROVED by {abs(count_diff):.2f} from best: {best_waiting_count:.2f})")
                elif count_diff > 0:
                    comparison_lines.append(f"  ✗ Waiting count: {current_waiting_count:.2f} (WORSE by {count_diff:.2f} from best: {best_waiting_count:.2f})")
                else:
                    comparison_lines.append(f"  = Waiting count: {current_waiting_count:.2f} (SAME as best)")

            # Service metrics: arrived persons (higher is better)
            if current_arrived_persons is not None and best_arrived_persons is not None:
                arrived_diff = current_arrived_persons - best_arrived_persons
                if arrived_diff > 0:
                    comparison_lines.append(f"  ✓ Arrived persons: {current_arrived_persons:.0f} (IMPROVED by {arrived_diff:.0f} from best: {best_arrived_persons:.0f})")
                elif arrived_diff < 0:
                    comparison_lines.append(f"  ✗ Arrived persons: {current_arrived_persons:.0f} (WORSE by {abs(arrived_diff):.0f} from best: {best_arrived_persons:.0f})")
                else:
                    comparison_lines.append(f"  = Arrived persons: {current_arrived_persons:.0f} (SAME as best)")

        elif module_name == "subway_scheduling":
            # Subway scheduling: prioritize waiting time/count; electricity savings when wait is comparable
            current_waiting_time = current_metrics.get("avg_passenger_waiting_time")
            best_waiting_time = best_metrics.get("avg_passenger_waiting_time")
            current_waiting_count = current_metrics.get("avg_passenger_waiting_count")
            best_waiting_count = best_metrics.get("avg_passenger_waiting_count")
            current_arrived_persons = current_metrics.get("total_arrived_persons")
            best_arrived_persons = best_metrics.get("total_arrived_persons")
            # Use total_electricity_consumption_kwh (actual metric name)
            current_electricity_kwh = current_metrics.get("total_electricity_consumption_kwh")
            best_electricity_kwh = best_metrics.get("total_electricity_consumption_kwh")

            # Electricity consumption (must be minimized)
            if current_electricity_kwh is not None and best_electricity_kwh is not None:
                elec_diff = current_electricity_kwh - best_electricity_kwh
                if elec_diff < 0:
                    comparison_lines.append(f"  ✓ Electricity: {current_electricity_kwh:.2f}kWh (IMPROVED by {abs(elec_diff):.2f}kWh from best: {best_electricity_kwh:.2f}kWh)")
                elif elec_diff > 0:
                    comparison_lines.append(f"  ✗ Electricity: {current_electricity_kwh:.2f}kWh (WORSE by {elec_diff:.2f}kWh from best: {best_electricity_kwh:.2f}kWh)")
                else:
                    comparison_lines.append(f"  = Electricity: {current_electricity_kwh:.2f}kWh (SAME as best)")

            # Service metrics (at least one must be improved): waiting time
            if current_waiting_time is not None and best_waiting_time is not None:
                wait_diff = current_waiting_time - best_waiting_time
                if wait_diff < 0:
                    comparison_lines.append(f"  ✓ Waiting time: {current_waiting_time:.2f}s (IMPROVED by {abs(wait_diff):.2f}s from best: {best_waiting_time:.2f}s)")
                elif wait_diff > 0:
                    comparison_lines.append(f"  ✗ Waiting time: {current_waiting_time:.2f}s (WORSE by {wait_diff:.2f}s from best: {best_waiting_time:.2f}s)")
                else:
                    comparison_lines.append(f"  = Waiting time: {current_waiting_time:.2f}s (SAME as best)")

            # Service metrics: waiting count
            if current_waiting_count is not None and best_waiting_count is not None:
                count_diff = current_waiting_count - best_waiting_count
                if count_diff < 0:
                    comparison_lines.append(f"  ✓ Waiting count: {current_waiting_count:.2f} (IMPROVED by {abs(count_diff):.2f} from best: {best_waiting_count:.2f})")
                elif count_diff > 0:
                    comparison_lines.append(f"  ✗ Waiting count: {current_waiting_count:.2f} (WORSE by {count_diff:.2f} from best: {best_waiting_count:.2f})")
                else:
                    comparison_lines.append(f"  = Waiting count: {current_waiting_count:.2f} (SAME as best)")

            # Service metrics: arrived persons (higher is better)
            if current_arrived_persons is not None and best_arrived_persons is not None:
                arrived_diff = current_arrived_persons - best_arrived_persons
                if arrived_diff > 0:
                    comparison_lines.append(f"  ✓ Arrived persons: {current_arrived_persons:.0f} (IMPROVED by {arrived_diff:.0f} from best: {best_arrived_persons:.0f})")
                elif arrived_diff < 0:
                    comparison_lines.append(f"  ✗ Arrived persons: {current_arrived_persons:.0f} (WORSE by {abs(arrived_diff):.0f} from best: {best_arrived_persons:.0f})")
                else:
                    comparison_lines.append(f"  = Arrived persons: {current_arrived_persons:.0f} (SAME as best)")

        elif module_name == "taxi_scheduling":
            # Taxi scheduling: prioritize total income and passenger dropoffs
            current_total_income = current_metrics.get("total_income")
            best_total_income = best_metrics.get("total_income")
            current_dropoffs = current_metrics.get("passenger_dropoffs")
            best_dropoffs = best_metrics.get("passenger_dropoffs")
            current_income_per_taxi = current_metrics.get("income_per_taxi")
            best_income_per_taxi = best_metrics.get("income_per_taxi")

            if current_total_income is not None and best_total_income is not None:
                income_diff = current_total_income - best_total_income
                if income_diff > 0:
                    comparison_lines.append(f"  ✓ Total income: {current_total_income:.2f} (IMPROVED by {income_diff:.2f} from best: {best_total_income:.2f})")
                elif income_diff < 0:
                    comparison_lines.append(f"  ✗ Total income: {current_total_income:.2f} (WORSE by {abs(income_diff):.2f} from best: {best_total_income:.2f})")
                else:
                    comparison_lines.append(f"  = Total income: {current_total_income:.2f} (SAME as best)")

            if current_dropoffs is not None and best_dropoffs is not None:
                dropoff_diff = current_dropoffs - best_dropoffs
                if dropoff_diff > 0:
                    comparison_lines.append(f"  ✓ Passenger dropoffs: {current_dropoffs} (IMPROVED by {dropoff_diff} from best: {best_dropoffs})")
                elif dropoff_diff < 0:
                    comparison_lines.append(f"  ✗ Passenger dropoffs: {current_dropoffs} (WORSE by {abs(dropoff_diff)} from best: {best_dropoffs})")
                else:
                    comparison_lines.append(f"  = Passenger dropoffs: {current_dropoffs} (SAME as best)")

            if current_income_per_taxi is not None and best_income_per_taxi is not None:
                income_per_taxi_diff = current_income_per_taxi - best_income_per_taxi
                if income_per_taxi_diff > 0:
                    comparison_lines.append(f"  ✓ Income per taxi: {current_income_per_taxi:.2f} (IMPROVED by {income_per_taxi_diff:.2f} from best: {best_income_per_taxi:.2f})")
                elif income_per_taxi_diff < 0:
                    comparison_lines.append(f"  ✗ Income per taxi: {current_income_per_taxi:.2f} (WORSE by {abs(income_per_taxi_diff):.2f} from best: {best_income_per_taxi:.2f})")
                else:
                    comparison_lines.append(f"  = Income per taxi: {current_income_per_taxi:.2f} (SAME as best)")
        if len(comparison_lines) == 1:
            # Only header line, no metrics to compare
            return ""

        return "\n".join(comparison_lines)

    def _is_better_simulation_result(
        self,
        current_result: Dict[str, Any],
        best_result: Optional[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Compare two simulation results and determine if current is better for each module.

        Args:
            current_result: Current simulation result
            best_result: Best simulation result so far (None if no best yet)

        Returns:
            Dictionary mapping module name to bool:
            - Key: module name (e.g., 'signal_timing', 'highway_speed_limit')
            - Value: True if current_result is better than best_result for this module, False otherwise
            Only includes modules that exist in current_result
        """
        result = {}

        if best_result is None:
            # If no best result, all modules in current result are better
            # But only include modules that are in available_control_modules
            current_module_metrics = current_result.get("module_metrics", {})
            if self.available_control_modules is not None:
                for module_name in current_module_metrics.keys():
                    if module_name in self.available_control_modules:
                        result[module_name] = True
            else:
                for module_name in current_module_metrics.keys():
                    result[module_name] = True
            return result

        if not current_result.get("success"):
            # If current result failed, no module is better
            return result

        if not best_result.get("success"):
            # If best result failed but current succeeded, all modules are better
            # But only include modules that are in available_control_modules
            current_module_metrics = current_result.get("module_metrics", {})
            if self.available_control_modules is not None:
                for module_name in current_module_metrics.keys():
                    if module_name in self.available_control_modules:
                        result[module_name] = True
            else:
                for module_name in current_module_metrics.keys():
                    result[module_name] = True
            return result

        current_module_metrics = current_result.get("module_metrics", {})
        best_module_metrics = best_result.get("module_metrics", {})

        # Filter module_metrics to only include available_control_modules
        if self.available_control_modules is not None:
            # Only compare modules that are in available_control_modules
            filtered_current_module_metrics = {
                name: metrics for name, metrics in current_module_metrics.items()
                if name in self.available_control_modules
            }
            filtered_best_module_metrics = {
                name: metrics for name, metrics in best_module_metrics.items()
                if name in self.available_control_modules
            }
        else:
            # If available_control_modules is None, include all modules
            filtered_current_module_metrics = current_module_metrics
            filtered_best_module_metrics = best_module_metrics

        # Get policy_avg_travel_time from stats if available (preferred for comparison)
        current_stats = current_result.get("stats", {})
        best_stats = best_result.get("stats", {}) if best_result else {}
        current_policy_tt = current_stats.get("policy_avg_travel_time")
        best_policy_tt = best_stats.get("policy_avg_travel_time")
        current_policy_highway_tt = current_stats.get("policy_highway_avg_travel_time")
        best_policy_highway_tt = best_stats.get("policy_highway_avg_travel_time")

        # Evaluate each module separately using their own metrics
        for module_name in filtered_current_module_metrics.keys():
            module_better = False

            if module_name == "signal_timing":
                # Signal timing: require BOTH travel time AND waiting time to decrease
                signal_metrics_current = filtered_current_module_metrics["signal_timing"]
                signal_metrics_best = filtered_best_module_metrics.get("signal_timing", {})

                # Use signal module metrics avg_travel_time directly for comparison
                # (policy_avg_travel_time is global across all vehicle types and is not representative
                # of signal-timing-controlled intersections alone)
                current_signal_travel_time = signal_metrics_current.get("avg_travel_time")
                best_signal_travel_time = signal_metrics_best.get("avg_travel_time")

                # Get waiting time metrics
                current_signal_waiting_time = signal_metrics_current.get("avg_waiting_time")
                best_signal_waiting_time = signal_metrics_best.get("avg_waiting_time")

                # Both travel time and waiting time must be lower to be considered better
                if current_signal_travel_time is not None and best_signal_travel_time is not None:
                    travel_time_better = current_signal_travel_time < best_signal_travel_time
                elif current_signal_travel_time is not None:
                    travel_time_better = True
                else:
                    travel_time_better = False

                if current_signal_waiting_time is not None and best_signal_waiting_time is not None:
                    waiting_time_better = current_signal_waiting_time < best_signal_waiting_time
                elif current_signal_waiting_time is not None:
                    waiting_time_better = True
                else:
                    waiting_time_better = False

                module_better = travel_time_better and waiting_time_better

            elif module_name == "highway_speed_limit":
                # Highway speed limit: compare travel time and throughput
                # Better if (1) travel_time lower AND throughput acceptable, OR (2) throughput higher AND travel_time acceptable
                highway_metrics_current = filtered_current_module_metrics["highway_speed_limit"]
                highway_metrics_best = filtered_best_module_metrics.get("highway_speed_limit", {})

                # Always use module_metrics avg_travel_time for consistent comparison
                # This ensures we compare the same metric type (all highway vehicles) rather than mixing
                # policy_highway_avg_travel_time (policy test vehicles only) with module metrics (all vehicles)
                current_highway_travel_time = highway_metrics_current.get("avg_travel_time")
                best_highway_travel_time = highway_metrics_best.get("avg_travel_time")
                current_highway_throughput = highway_metrics_current.get("throughput")
                best_highway_throughput = highway_metrics_best.get("throughput")

                # Travel time strictly lower (lower is better)
                travel_time_lower = False
                if current_highway_travel_time is not None and best_highway_travel_time is not None:
                    travel_time_lower = current_highway_travel_time < best_highway_travel_time
                elif current_highway_travel_time is not None and best_highway_travel_time is None:
                    travel_time_lower = True
                else:
                    travel_time_lower = False

                # Travel time lower or at most 1% higher than best (acceptable)
                if current_highway_travel_time is not None and best_highway_travel_time is not None:
                    travel_time_acceptable = current_highway_travel_time <= best_highway_travel_time * 1.05
                elif current_highway_travel_time is not None and best_highway_travel_time is None:
                    travel_time_acceptable = True
                else:
                    travel_time_acceptable = False

                # Throughput strictly higher (higher is better)
                throughput_higher = False
                if current_highway_throughput is not None and best_highway_throughput is not None:
                    throughput_higher = current_highway_throughput > best_highway_throughput
                elif current_highway_throughput is not None and best_highway_throughput is None:
                    throughput_higher = True
                else:
                    throughput_higher = False

                # Throughput higher or at least 99% of best (acceptable)
                throughput_acceptable = False
                if current_highway_throughput is not None and best_highway_throughput is not None:
                    throughput_acceptable = current_highway_throughput >= best_highway_throughput * 0.95
                elif current_highway_throughput is not None and best_highway_throughput is None:
                    throughput_acceptable = True
                else:
                    throughput_acceptable = False

                condition1 = travel_time_lower and throughput_acceptable
                condition2 = throughput_higher and travel_time_acceptable
                module_better = condition1 or condition2

            elif module_name == "ramp_metering":
                # Ramp metering: compare queue length and throughput
                # Better if (1) queue_len lower AND throughput acceptable, OR (2) throughput higher AND queue_len acceptable
                ramp_metrics_current = filtered_current_module_metrics["ramp_metering"]
                ramp_metrics_best = filtered_best_module_metrics.get("ramp_metering", {})

                current_ramp_queue_len = ramp_metrics_current.get("avg_queue_len")
                best_ramp_queue_len = ramp_metrics_best.get("avg_queue_len")
                current_ramp_throughput = ramp_metrics_current.get("throughput")
                best_ramp_throughput = ramp_metrics_best.get("throughput")

                # Queue length strictly lower (lower is better)
                queue_len_lower = False
                if current_ramp_queue_len is not None and best_ramp_queue_len is not None:
                    queue_len_lower = current_ramp_queue_len < best_ramp_queue_len
                elif current_ramp_queue_len is not None and best_ramp_queue_len is None:
                    queue_len_lower = True
                else:
                    queue_len_lower = False

                # Queue length lower or at most 1% higher than best (acceptable)
                queue_len_acceptable = False
                if current_ramp_queue_len is not None and best_ramp_queue_len is not None:
                    queue_len_acceptable = current_ramp_queue_len <= best_ramp_queue_len * 1.05
                elif current_ramp_queue_len is not None and best_ramp_queue_len is None:
                    queue_len_acceptable = True
                else:
                    queue_len_acceptable = False

                # Throughput strictly higher (higher is better)
                throughput_higher = False
                if current_ramp_throughput is not None and best_ramp_throughput is not None:
                    throughput_higher = current_ramp_throughput > best_ramp_throughput
                elif current_ramp_throughput is not None and best_ramp_throughput is None:
                    throughput_higher = True
                else:
                    throughput_higher = False

                # Throughput higher or at least 99% of best (acceptable)
                throughput_acceptable = False
                if current_ramp_throughput is not None and best_ramp_throughput is not None:
                    throughput_acceptable = current_ramp_throughput >= best_ramp_throughput * 0.95
                elif current_ramp_throughput is not None and best_ramp_throughput is None:
                    throughput_acceptable = True
                else:
                    throughput_acceptable = False

                condition1 = queue_len_lower and throughput_acceptable
                condition2 = throughput_higher and queue_len_acceptable
                module_better = condition1 or condition2

            elif module_name == "subway_scheduling":
                # Subway: PRIMARY=waiting time, SECONDARY=electricity. Accept if (1) waiting improves AND electricity ok,
                # OR (2) electricity improves AND waiting does NOT increase (strict, no tolerance)
                subway_metrics_current = current_module_metrics["subway_scheduling"]
                subway_metrics_best = best_module_metrics.get("subway_scheduling", {})

                current_waiting_time = subway_metrics_current.get("avg_passenger_waiting_time")
                best_waiting_time = subway_metrics_best.get("avg_passenger_waiting_time")
                current_electricity_kwh = subway_metrics_current.get("total_electricity_consumption_kwh")
                best_electricity_kwh = subway_metrics_best.get("total_electricity_consumption_kwh")
                current_electricity = current_electricity_kwh * 1000.0 if current_electricity_kwh is not None else None
                best_electricity = best_electricity_kwh * 1000.0 if best_electricity_kwh is not None else None

                # Waiting time strictly lower
                if current_waiting_time is not None and best_waiting_time is not None:
                    waiting_time_lower = current_waiting_time < best_waiting_time
                elif current_waiting_time is not None and best_waiting_time is None:
                    waiting_time_lower = True
                else:
                    waiting_time_lower = False

                # Waiting time must NOT increase when electricity improves (strict: no tolerance)
                if current_waiting_time is not None and best_waiting_time is not None:
                    waiting_time_acceptable = current_waiting_time <= best_waiting_time
                elif current_waiting_time is not None and best_waiting_time is None:
                    waiting_time_acceptable = True
                else:
                    waiting_time_acceptable = False

                # Electricity strictly lower
                if current_electricity is not None and best_electricity is not None:
                    electricity_lower = current_electricity < best_electricity
                elif current_electricity is not None and best_electricity is None:
                    electricity_lower = True
                else:
                    electricity_lower = False

                # Electricity lower or at most 25% higher than best
                if current_electricity is not None and best_electricity is not None:
                    electricity_acceptable = current_electricity <= best_electricity * 1.05
                elif current_electricity is not None and best_electricity is None:
                    electricity_acceptable = True
                else:
                    electricity_acceptable = False

                condition1 = waiting_time_lower and electricity_acceptable
                condition2 = electricity_lower and waiting_time_acceptable
                module_better = condition1 or condition2

            elif module_name == "bus_scheduling":
                bus_metrics_current = current_module_metrics["bus_scheduling"]
                bus_metrics_best = best_module_metrics.get("bus_scheduling", {})

                current_waiting_time = bus_metrics_current.get("avg_passenger_waiting_time")
                best_waiting_time = bus_metrics_best.get("avg_passenger_waiting_time")
                current_fuel_g = bus_metrics_current.get("total_fuel_consumption_g")
                best_fuel_g = bus_metrics_best.get("total_fuel_consumption_g")
                current_fuel = current_fuel_g * 1000.0 if current_fuel_g is not None else None
                best_fuel = best_fuel_g * 1000.0 if best_fuel_g is not None else None

                # Waiting time strictly lower
                if current_waiting_time is not None and best_waiting_time is not None:
                    waiting_time_lower = current_waiting_time < best_waiting_time
                elif current_waiting_time is not None and best_waiting_time is None:
                    waiting_time_lower = True
                else:
                    waiting_time_lower = False

                if current_waiting_time is not None and best_waiting_time is not None:
                    waiting_time_acceptable = current_waiting_time <= best_waiting_time * 1.01
                elif current_waiting_time is not None and best_waiting_time is None:
                    waiting_time_acceptable = True
                else:
                    waiting_time_acceptable = False

                # Fuel strictly lower
                if current_fuel is not None and best_fuel is not None:
                    fuel_lower = current_fuel < best_fuel
                elif current_fuel is not None and best_fuel is None:
                    fuel_lower = True
                else:
                    fuel_lower = False

                # Fuel lower or at most 25% higher than best
                if current_fuel is not None and best_fuel is not None:
                    fuel_acceptable = current_fuel <= best_fuel * 1.05
                elif current_fuel is not None and best_fuel is None:
                    fuel_acceptable = True
                else:
                    fuel_acceptable = False

                condition1 = waiting_time_lower and fuel_acceptable
                condition2 = fuel_lower and waiting_time_acceptable
                module_better = condition1 or condition2

            elif module_name == "taxi_scheduling":
                # Taxi: better if total income OR passenger dropoffs improves
                taxi_metrics_current = filtered_current_module_metrics["taxi_scheduling"]
                taxi_metrics_best = filtered_best_module_metrics.get("taxi_scheduling", {})
                
                current_income = taxi_metrics_current.get("total_income")
                best_income = taxi_metrics_best.get("total_income")
                current_pickups = taxi_metrics_current.get("passenger_pickups")
                best_pickups = taxi_metrics_best.get("passenger_pickups")

                income_improved = (
                    current_income is not None
                    and best_income is not None
                    and current_income > best_income
                )
                pickups_improved = (
                    current_pickups is not None
                    and best_pickups is not None
                    and current_pickups > best_pickups
                )
                module_better = income_improved or pickups_improved
            else:
                # Unknown module: try to use module's own metrics if available
                module_metrics_current = filtered_current_module_metrics.get(module_name, {})
                module_metrics_best = filtered_best_module_metrics.get(module_name, {})

                # Prefer policy_avg_travel_time from stats over module metrics avg_travel_time
                # This ensures we compare policy simulations with checkpoint simulations using the same metric
                current_module_travel_time = current_policy_tt if current_policy_tt is not None else module_metrics_current.get(
                    "avg_travel_time")
                best_module_travel_time = best_policy_tt if best_policy_tt is not None else module_metrics_best.get(
                    "avg_travel_time")

                if current_module_travel_time is not None and best_module_travel_time is not None:
                    module_better = current_module_travel_time < best_module_travel_time
                elif current_module_travel_time is not None:
                    # Current has data but best doesn't
                    module_better = True
                else:
                    # No module-specific metrics available
                    module_better = False

            result[module_name] = module_better

        return result
    
    def _is_valid_control_config(self, control_configs: Dict[str, Any]) -> bool:
        """Check if control module configurations are valid."""
        if not isinstance(control_configs, dict):
            return False
        
        from control_modules import get_control_module
        
        for module_name, config in control_configs.items():
            if not isinstance(config, dict):
                return False
            
            module = get_control_module(module_name)
            
            if module is None:
                print(f"Error: Control module '{module_name}' not found")
                return False
            
            try:
                is_valid, error_msg = module.validate_config(config)
                if not is_valid:
                    if error_msg:
                        print(f"Error validating {module_name} config: {error_msg}")
                    return False
            except Exception as e:
                print(f"Error validating {module_name} config: {e}")
                return False
        
        return True
    
    def _extract_control_modules(self, action_content: str) -> List[str]:
        """Extract control module names from POLICY_PLANNING or DEBUG action content."""
        modules = []
        lines = action_content.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if 'control modules:' in line_lower or 'control module:' in line_lower:
                if 'control modules:' in line_lower:
                    module_str = line_lower.split('control modules:', 1)[1].strip()
                else:
                    module_str = line_lower.split('control module:', 1)[1].strip()
                
                module_list = [m.strip() for m in module_str.split(',')]
                modules.extend(module_list)
                break
        
        return modules
    
    def _extract_debug_type(self, action_content: str) -> Optional[str]:
        """Extract debug type (DATA_ANALYSIS or POLICY_PLANNING) from DEBUG action content."""
        lines = action_content.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if 'debug type:' in line_lower:
                debug_type = line_lower.split('debug type:', 1)[1].strip().upper()
                if debug_type in ['DATA_ANALYSIS', 'POLICY_PLANNING']:
                    return debug_type
        
        return None
    
    def _extract_module_names(self, action_content: str) -> List[str]:
        """Extract module names from GET_CONTROL_API action content."""
        def _parse_module_list(raw: str) -> List[str]:
            cleaned = raw.replace('[', '').replace(']', '')
            cleaned = cleaned.replace('"', '').replace("'", '')
            parts = [part.strip() for part in cleaned.split(',') if part.strip()]
            if len(parts) == 1 and ' ' in parts[0]:
                parts = [part.strip() for part in parts[0].split() if part.strip()]
            return parts

        modules = []
        lines = action_content.split('\n')
        for line in lines:
            raw_line = line.strip()
            if not raw_line:
                continue
            line_lower = raw_line.lower()
            if line_lower.startswith('modules:') or line_lower.startswith('module:'):
                module_str = raw_line.split(':', 1)[1].strip()
                modules.extend(_parse_module_list(module_str))
                break
            if line_lower.startswith('action: get_control_api') or line_lower.startswith('action:get_control_api'):
                if 'get_control_api' in line_lower:
                    module_str = line_lower.split('get_control_api', 1)[1].strip(" :-")
                    if module_str:
                        modules.extend(_parse_module_list(module_str))
                        break

        if not modules:
            # Try to find module names in the content
            content_lower = action_content.lower()
            for module_name in self._control_specs.keys():
                if module_name.lower() in content_lower:
                    modules.append(module_name)

        # Preserve order and remove duplicates
        seen = set()
        ordered = []
        for module_name in modules:
            if module_name and module_name not in seen:
                ordered.append(module_name)
                seen.add(module_name)
        return ordered

    def _extract_module_name(self, action_content: str) -> Optional[str]:
        """Extract single module name from GET_CONTROL_API action content (legacy)."""
        module_names = self._extract_module_names(action_content)
        return module_names[0] if module_names else None
    
    @staticmethod
    def _load_control_specs() -> Dict[str, Dict[str, Any]]:
        """Load control module specifications from JSON files."""
        agent_tools_dir = workspace_root / "agent_tools"
        specs = {}
        
        module_files = [
            ("signal_timing", "signal_timing.json"),
            ("subway_scheduling", "subway_scheduling.json"),
            ("bus_scheduling", "bus_scheduling.json"),
            ("highway_speed_limit", "highway_speed_limit.json"),
            ("ramp_metering", "ramp_metering.json"),
            ("taxi_scheduling", "taxi_scheduling.json"),
        ]
        
        for module_name, filename in module_files:
            spec_path = agent_tools_dir / filename
            try:
                with open(spec_path, 'r', encoding='utf-8') as f:
                    specs[module_name] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {filename} not found at {spec_path}")
                specs[module_name] = {"data": [], "functions": []}
            except json.JSONDecodeError as e:
                print(f"Error parsing {filename}: {e}")
                specs[module_name] = {"data": [], "functions": []}
            
            # Load domain knowledge from control module class if available
            from control_modules.registry import CONTROL_MODULES
            if module_name in CONTROL_MODULES:
                module_class = CONTROL_MODULES[module_name]
                if hasattr(module_class, 'DOMAIN_KNOWLEDGE'):
                    specs[module_name]["domain_knowledge"] = module_class.DOMAIN_KNOWLEDGE
        
        return specs
    
    def _format_module_api(self, module_name: str, spec: Dict[str, Any]) -> str:
        """Format API information for a specific control module."""
        message_parts = [
            f"**{module_name.replace('_', ' ').title()} Control Module API:**",
            ""
        ]
        
        # Format data section
        if spec.get("data"):
            message_parts.append("**Available Data (Pre-defined Variables):**")
            message_parts.append("")
            message_parts.append("**IMPORTANT**: These are pre-defined variables (dict, NetworkX graph, etc.) that are already available in your code execution environment. Use them directly by name (e.g., `zone_dict`, `highway_segment_dict`). DO NOT call them as functions (e.g., `zone_dict()` is WRONG).")
            message_parts.append("")

            for data_item in spec.get("data", []):
                name = data_item.get("name", "")
                description = data_item.get("description", "")
                data_format = data_item.get("data_format", {})
                usage_example = data_item.get("usage_example", "")

                message_parts.append(f"- **`{name}`**: {description}")

                # Format data structure
                if isinstance(data_format, dict):
                    if "type" in data_format:
                        message_parts.append(f"  - Type: {data_format['type']}")
                    if "nodes" in data_format:
                        if isinstance(data_format["nodes"], dict):
                            for key, value in data_format["nodes"].items():
                                message_parts.append(f"  - {key.capitalize()}: {value}")
                        else:
                            message_parts.append(f"  - Nodes: {data_format['nodes']}")
                    if "edges" in data_format:
                        if isinstance(data_format["edges"], dict):
                            for key, value in data_format["edges"].items():
                                message_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
                        else:
                            message_parts.append(f"  - Edges: {data_format['edges']}")
                    if "edge_attributes" in data_format:
                        message_parts.append("  - Edge attributes:")
                        for attr, desc in data_format["edge_attributes"].items():
                            message_parts.append(f"    - `{attr}`: {desc}")
                    if "structure" in data_format:
                        message_parts.append(f"  - Structure: {data_format['structure']}")
                    if "example" in data_format:
                        message_parts.append("  - Example:")
                        example_str = json.dumps(data_format["example"], indent=6)
                        message_parts.append(f"    ```python\n{example_str}\n    ```")
                    if "relevant_fields" in data_format:
                        message_parts.append("  - Relevant fields:")
                        for field, desc in data_format["relevant_fields"].items():
                            message_parts.append(f"    - `{field}`: {desc}")

                if usage_example:
                    message_parts.append(f"  - Usage:")
                    message_parts.append(f"    {usage_example}")

                message_parts.append("")
        
        # Format functions section
        if spec.get("functions"):
            message_parts.append("**Available Functions:**")
            message_parts.append("")

            for func_item in spec.get("functions", []):
                name = func_item.get("name", "")
                description = func_item.get("description", "")
                input_format = func_item.get("input_format", {})
                output_format = func_item.get("output_format", {})
                usage_example = func_item.get("usage_example", "")

                message_parts.append(f"- **`{name}(...)`**: {description}")

                # Format input
                if input_format:
                    message_parts.append("  - Input parameters:")
                    for param, desc in input_format.items():
                        message_parts.append(f"    - `{param}`: {desc}")

                # Format output
                if output_format:
                    message_parts.append("  - Returns:")
                    if "type" in output_format:
                        message_parts.append(f"    - Type: {output_format['type']}")
                    if "structure" in output_format:
                        structure_str = json.dumps(output_format["structure"], indent=8)
                        message_parts.append(f"    - Structure:")
                        message_parts.append(f"      ```python\n{structure_str}\n      ```")

                if usage_example:
                    message_parts.append("  - Usage examples:")
                    message_parts.append(f"    {usage_example}")

                message_parts.append("")

        if not spec.get("data") and not spec.get("functions"):
            message_parts.append("No data or functions available for this module yet (TODO).")
            message_parts.append("")

        # Format domain knowledge section
        if spec.get("domain_knowledge"):
            message_parts.append("**Domain Knowledge:**")
            message_parts.append("")
            message_parts.append(spec.get("domain_knowledge"))
            message_parts.append("")

        return "\n".join(message_parts)

    def _format_global_knowledge_section(self) -> str:
        """
        Format global knowledge section for joint control system message.

        Returns:
            Formatted string containing enabled modules, dependencies, and optimization principles.
        """
        lines = []

        # Enabled modules
        lines.append("### Enabled Modules")
        if self.available_control_modules:
            for module in self.available_control_modules:
                lines.append(f"- {module}")
            lines.append("(Only these modules can be optimized)")
        else:
            lines.append("- None specified")
        lines.append("")

        # Cross-module dependencies (filtered to enabled modules only)
        lines.append("### Cross-Module Dependencies")
        if self.available_control_modules:
            for module in self.available_control_modules:
                deps = self.MODULE_DEPENDENCIES.get(module, {})
                affects = [m for m in deps.get("affects", []) if m in self.available_control_modules]
                affected_by = [m for m in deps.get("affected_by", []) if m in self.available_control_modules]

                if affects or affected_by:
                    dep_parts = []
                    if affects:
                        dep_parts.append(f"affects: {', '.join(affects)}")
                    if affected_by:
                        dep_parts.append(f"affected_by: {', '.join(affected_by)}")
                    lines.append(f"- {module}: {'; '.join(dep_parts)}")
                else:
                    lines.append(f"- {module}: independent (no dependencies with enabled modules)")
        lines.append("")

        # Shared optimization principles
        lines.append("### Shared Optimization Principles")
        lines.append("- Optimize upstream modules first (modules with no dependencies)")
        lines.append("- Downstream modules should consider upstream module decisions")
        lines.append("- Coordinate related modules for better overall performance")

        return "\n".join(lines)

    def _format_module_metrics_section(self) -> str:
        """
        Format current module performance metrics for the system message.

        Returns:
            Formatted string containing module metrics.
        """
        if not self.module_metrics:
            return "No metrics available yet."

        lines = []
        for module_name, metrics in self.module_metrics.items():
            if self.available_control_modules and module_name not in self.available_control_modules:
                continue
            lines.append(f"**{module_name.replace('_', ' ').title()}:**")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  - {metric_name.replace('_', ' ').title()}: {value:.2f}")
                else:
                    lines.append(f"  - {metric_name.replace('_', ' ').title()}: {value}")
        return "\n".join(lines) if lines else "No metrics available yet."

    def _format_joint_control_system_message(self) -> str:
        """
        Generate system message for joint control mode.

        This provides a simplified system message focused on joint optimization
        with global knowledge about enabled modules and their dependencies.

        Returns:
            Formatted system message string.
        """
        # Filter specs based on available control modules
        if self.available_control_modules is not None:
            available_specs = {name: spec for name, spec in self._control_specs.items() if name in self.available_control_modules}
        else:
            available_specs = self._control_specs
        available_module_list = ", ".join(available_specs.keys())

        message_parts = [
            "# Urban Transportation Joint Control Agent",
            "",
            "## Your Role",
            "You are an expert urban transportation control agent optimizing multiple systems jointly.",
            "Your goal is to coordinate multiple control modules to improve overall urban mobility.",
            "",
            "## Global Knowledge",
            "",
            self._format_global_knowledge_section(),
            "",
            "## Turn Limit",
            f"You have a maximum of {self.max_turns} dialogue turns.",
            "Each action counts as one turn. Plan efficiently and use FINISH when complete.",
            "",
            "## Available Actions",
            "1. **PLAN**: Think and plan your optimization strategy",
            "2. **GET_CONTROL_API**: Query module-specific APIs, data, and domain knowledge",
            f"   Available modules: {available_module_list}",
            "3. **DATA_ANALYSIS**: Analyze traffic data (use save_cache() to store results)",
            "4. **POLICY_PLANNING**: Design control configurations (simulation runs automatically)",
            "5. **DEBUG**: Fix code errors",
            "6. **FINISH**: Complete optimization",
            "",
            "## Important Notes",
            "- Use GET_CONTROL_API to query module APIs and domain knowledge before optimization",
            "- Only enabled modules can be optimized",
            "- Consider module dependencies when planning optimization order",
            "- Simulation is automatically executed after POLICY_PLANNING",
            "",
            "## Required Formats (Critical)",
            "- Always start with: ACTION: <ACTION_NAME>",
            "- GET_CONTROL_API format:",
            "  ACTION: GET_CONTROL_API",
            f"  Module: {available_module_list}  # comma-separated allowed",
            "- DATA_ANALYSIS format:",
            "  ACTION: DATA_ANALYSIS",
            "  ```python",
            "  # Use save_cache(...) to store analysis results",
            "  ```",
            "- POLICY_PLANNING format:",
            "  ACTION: POLICY_PLANNING",
            f"  Control Modules: {available_module_list}",
            "  ```python",
            "  # Define a variable named `config` and do NOT use return",
            "  config = {...}",
            "  ```",
            "- DEBUG (policy) format:",
            "  ACTION: DEBUG",
            "  Debug Type: POLICY_PLANNING",
            f"  Control Modules: {available_module_list}",
            "  ```python",
            "  # Fixed code",
            "  ```",
            "- One action per turn. Do not combine actions in a single response.",
        ]

        # Add memory section if memory exists
        if self.memory:
            message_parts.extend([
                "",
                "## Previous Optimization Experience (Memory)",
                ""
            ])
            for i, memory_item in enumerate(self.memory, 1):
                message_parts.append(f"{i}. {memory_item}")

        message_parts.extend([
            "",
            "Begin with PLAN to outline your optimization strategy, then use GET_CONTROL_API to query module APIs."
        ])

        return "\n".join(message_parts)

    def add_memory(self, memory_items: Any) -> List[str]:
        """
        Add memory items to the agent's memory.

        Args:
            memory_items: Single memory item or list of items to add

        Returns:
            Updated memory list
        """
        if not memory_items:
            return self.memory.copy()

        if isinstance(memory_items, str):
            items = [memory_items]
        elif isinstance(memory_items, (list, tuple)):
            items = list(memory_items)
        else:
            return self.memory.copy()

        validated_items = self._parse_and_validate_memory(items)
        for item in validated_items:
            if item not in self.memory:
                self.memory.append(item)

        # Keep most recent items within limit
        if len(self.memory) > self.max_memory_items:
            self.memory = self.memory[-self.max_memory_items:]

        return self.memory.copy()

    def _format_system_message(self) -> str:
        """Generate system message from control module specifications."""
        # Use joint control system message if in joint control mode
        if self.is_joint_control:
            return self._format_joint_control_system_message()

        # Filter specs based on available control modules
        if self.available_control_modules is not None:
            available_specs = {name: spec for name, spec in self._control_specs.items() if name in self.available_control_modules}
        else:
            available_specs = self._control_specs
        
        message_parts = [
            "You are an expert urban transportation control agent. Your goal is to analyze and optimize various transportation systems to improve overall urban mobility and efficiency.",
            "",
            "**CRITICAL: Code Length Limit**",
            "**ALL code you write (DATA_ANALYSIS, POLICY_PLANNING, DEBUG) MUST be <50lines.**",
            "**Simplify your algorithm as much as possible. Keep code simple, focused, and efficient.**",
            "",
            "**IMPORTANT: Turn Limit**",
            f"You have a maximum of {self.max_turns} dialogue turns to complete your optimization task.",
            f"Each action (PLAN, DATA_ANALYSIS, POLICY_PLANNING, DEBUG, GET_CONTROL_API) counts as one turn.",
            "Note: Simulation is automatically executed after POLICY_PLANNING and does not count as a separate turn.",
            "You MUST complete the optimization and use FINISH action within this limit.",
            f"Current turn: You are on turn 1/{self.max_turns}. Plan your actions efficiently to complete within the limit.",
            "",
            "You can control and optimize multiple transportation facilities:",
        ]
        
        # Add available facilities dynamically based on available_control_modules
        facility_descriptions = {
            "signal_timing": "- Traffic Signals: Optimize signal timing to reduce congestion",
            "highway_speed_limit": "- Highway Speed Limits: Optimize variable speed limits to alleviate highway congestion",
            "ramp_metering": "- Ramp Metering: Optimize ramp control (OPEN/CLOSE durations) to manage highway on-ramp flow",
            "taxi_scheduling": "- Taxi Scheduling: Optimize taxi dispatching and repositioning to maximize total income and passenger dropoffs",
            "subway_scheduling": "- Subway Scheduling: Optimize train schedules and headways to improve passenger experience and energy efficiency",
            "bus_scheduling": "- Bus Scheduling: Optimize bus routes and frequencies to improve passenger service and reduce fuel consumption"
        }
        
        for module_name in available_specs.keys():
            if module_name in facility_descriptions:
                message_parts.append(facility_descriptions[module_name])
        
        # Add interconnection notes
        if len(available_specs) > 1:
            message_parts.extend([
            "",
            "**Important**: Different transportation systems are interconnected. For example:",
            "- Better traffic signal control can improve bus travel times",
            "- Highway speed limit control can reduce congestion and improve overall network flow",
            "- Subway schedules can affect surface traffic patterns",
            "- You can access data from ALL transportation systems to make informed decisions",
            ])
        else:
            message_parts.extend([
            "",
                "You have access to tools and data for a single transportation system.",
            ])
        
        message_parts.extend([
            "",
            "**Note**: Detailed API documentation (data structures, functions, and domain knowledge) for each control module is available on-demand.",
            "Use the GET_CONTROL_API action to query specific module APIs when needed.",
            "Available modules: " + ", ".join([name.replace('_', ' ') for name in available_specs.keys()]) + ".",
            "",
            "**Cache Functions (Available in All Code Execution):**",
            "",
            "1. **save_cache(cache_dict: dict)** - Save values to cache",
            "   Format: save_cache({\"key\": {\"value\": data, \"description\": \"desc\"}, ...})",
            "   Example: save_cache({\"intersection_states\": {\"value\": intersection_states, \"description\": \"Intersection states\"}})",
            "",
            "2. **load_cache(key: str)** - Load cached value",
            "   Returns: The cached value DIRECTLY (not wrapped in dict). Use: data = load_cache(\"key\")",
            "   **CRITICAL: NEVER use load_cache(key)[\"value\"]** - save_cache uses {\"value\": x} but load_cache returns x directly!",
            "   Correct: signal_analysis = load_cache(\"signal_analysis\")  # Use directly, no [\"value\"]",
            "   Wrong: signal_analysis = load_cache(\"signal_analysis\")[\"value\"]  # KeyError!",
            "   If key missing: load_cache returns None. Check with: data = load_cache(\"key\"); if data is None: ...",
            "",
            "3. **list_cache() -> dict** - List all cached keys and descriptions",
            "   Example: cache_list = list_cache()",
            "",
            "4. **clear_cache(key: str = None)** - Clear cache (specific key or all)",
            "   Example: clear_cache(\"intersection_states\") or clear_cache()",
            "",
            "**Cache Guidelines:**",
            "- Cache persists across all actions in the same session",
            "- Use save_cache() in DATA_ANALYSIS, load_cache() in POLICY_PLANNING",
            "- Supports any Python object (NetworkX graphs, lists, dicts, etc.)",
            "",
            ""
        ])
        
        # Add Actions and Workflow sections
        message_parts.extend([
            "**Available Actions:**",
            "You can ONLY perform ONE of the following actions in ONE turn, like this: ACTION: <action_name>. The detailed format is as follows:",
            "",
            "1. **PLAN**: Think and plan your next steps before executing code",
            "   ACTION: PLAN",
            "   [Current Status, Next Steps, Expected Outcomes]",
            "",
            "2. **GET_CONTROL_API**: Query available data and functions for a specific control module",
            "   ACTION: GET_CONTROL_API",
            "   Module: signal_timing",
            "",
            "3. **DATA_ANALYSIS**: Write Python code to analyze traffic data and identify congestion patterns",
            "   ACTION: DATA_ANALYSIS",
            "   ```python",
            "   # Analyze traffic demands",
            "   # Use save_cache()/load_cache()/list_cache() (see Cache Functions section)",
            "   ```",
            "   ANALYSIS ONLY, no config. **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.** Use GET_CONTROL_API first.",
            "",
            "4. **POLICY_PLANNING**: Write Python code to design control configurations",
            "   ACTION: POLICY_PLANNING",
            f"   Control Modules: {', '.join(available_specs.keys())}",
            "   ```python",
            "   # Use GET_CONTROL_API APIs to query historical data",
        ])
        
        # Add configuration examples for available modules
        config_examples = {
            "signal_timing": [
                "   # Signal timing configuration",
                "   signal_config = current_signal_config.copy()  # Start from current config",
                "   # Load cache: use load_cache(\"key\") directly, NEVER load_cache(\"key\")[\"value\"]",
                "   signal_analysis = load_cache(\"signal_analysis\")  # Returns value directly",
                "   # Modify specific intersections based on historical data analysis",
                "   signal_config[\"intersection_id\"][\"NTST\"] += 3  # seconds (direct phase dict format)",
            ],
            "highway_speed_limit": [
                "   # Highway speed limit configuration",
                "   highway_config = current_highway_speed_limit_config.copy() if current_highway_speed_limit_config else {{}}",
                "   # IMPORTANT: Each segment's value is a LIST of scheduled changes: [{{'time': <TIME>, 'speed_limit': <MPH>}}, ...]",
                "   # Adjust speed limits based on traffic conditions",
                "   highway_config[\"highway_segment_0\"] = [{{'time': 0, 'speed_limit': 55}}]  # Reduce from 65 to 55 for congestion",
                "   # Schedule multiple changes: highway_config[\"highway_segment_1\"] = [{{'time': 0, 'speed_limit': 60}}, {{'time': 900, 'speed_limit': 55}}]",
            ],
            "ramp_metering": [
                "   # Ramp metering configuration",
                "   ramp_config = current_ramp_metering_config.copy() if current_ramp_metering_config else {{}}",
                "   # Adjust OPEN/CLOSE durations based on traffic conditions",
                "   ramp_config[\"ramp_1\"] = {{\"OPEN\": 300, \"CLOSE\": 300}}  # Balanced control: open 300s, closed 300s",
            ],
            "subway_scheduling": [
                "   # Subway scheduling configuration (timetable format required)",
                "   import copy",
                "   subway_config = copy.deepcopy(current_subway_schedule)  # CRITICAL: Use deepcopy for nested structure",
                "   # Example: adjust headway for a specific line and time segment",
                "   # subway_config['subway_1:0']['timetable'][0]['headway'] = 180",
                "   # Example: adjust dwell time at all stops for a line",
                "   # for seg in subway_config['subway_1:0']['timetable']:",
                "   #     for stop in seg['schedule']:",
                "   #         stop['dwell_time'] = 25",
            ],
            "bus_scheduling": [
                "   # Bus scheduling configuration (timetable format required)",
                "   import copy",
                "   bus_config = copy.deepcopy(current_bus_schedule)  # CRITICAL: Use deepcopy for nested structure",
                "   # Example: adjust headway for a specific route and time segment",
                "   # bus_config['bus_Bx19:0']['timetable'][0]['headway'] = 180",
                "   # Example: adjust dwell time at all stops for a route",
                "   # for seg in bus_config['bus_Bx19:0']['timetable']:",
                "   #     for stop in seg['schedule']:",
                "   #         stop['dwell_time'] = 20",
            ],
            "taxi_scheduling": [
                "   # Taxi scheduling configuration - dispatch and reposition decisions",
                "   taxi_config = current_taxi_config.copy() if current_taxi_config else {}",
                "   # Create dispatch decisions: assign taxis to pending reservations",
                "   taxi_config['dispatch_decisions'] = [{'taxi_id': 'taxi_fleet_0', 'reservation_ids': ['res_id']}]",
                "   # Create repositioning decisions: move idle taxis to high-demand areas",
                "   taxi_config['reposition_decisions'] = [{'taxi_id': 'taxi_fleet_1', 'target_edge': 'edge_id'}]",
            ]
        }
        
        config_dict_parts = []
        for module_name in available_specs.keys():
            if module_name in config_examples:
                message_parts.extend(config_examples[module_name])
                message_parts.append("   ")
                var_name_map = {
                    "signal_timing": "signal_config",
                    "highway_speed_limit": "highway_config",
                    "ramp_metering": "ramp_config",
                    "subway_scheduling": "subway_config",
                    "bus_scheduling": "bus_config",
                    "taxi_scheduling": "taxi_config"
                }
                var_name = var_name_map.get(module_name, f"{module_name.replace('_', '_')}_config")
                config_dict_parts.append(f'"{module_name}": {var_name}')
        
        message_parts.extend([
            "   # IMPORTANT: Define config variable (DO NOT use return statement or sys.exit())",
            f"   config = {{{', '.join(config_dict_parts)}}}",
            "   ```",
            "   **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.** Use historical data APIs.",
            "",
            "5. **DEBUG**: Fix code errors",
            "   ACTION: DEBUG",
            "   Debug Type: DATA_ANALYSIS | POLICY_PLANNING",
            f"   Control Modules: {', '.join(available_specs.keys())}  # if POLICY_PLANNING",
            "   ```python",
            "   # Fixed code",
            "   ```",
            "   **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.**",
            "",
            "6. **FINISH**: Complete optimization",
            "   ACTION: FINISH",
            "   [Brief summary]",
            "",
            "**Workflow & Important Rules:**",
            "",
            "**Step-by-Step Process:**",
            "1. PLAN (NECESSARY) - Think about your approach and break down the task",
            "2. GET_CONTROL_API (NECESSARY) - Query available data and functions for control modules",
            "3. DATA_ANALYSIS - Analyze historical data from ALL relevant transportation systems",
            "   - **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.** Focus on one control model at a time",
            "   - Use cache functions (save_cache/load_cache/list_cache) to store and retrieve analysis results",
            "   - Access traffic data and current configurations (see POLICY_PLANNING examples for config variable names)",
        ])
        
        message_parts.extend([
            "4. If analysis fails, use DEBUG to rewrite all codes, fix errors, and retry",
            "5. POLICY_PLANNING - Design improved control configurations",
            "   - **CRITICAL**: ALWAYS start from current_config.copy() (see examples above), DO NOT build from scratch",
            "   - Use load_cache(\"key\") to retrieve cached results - returns value DIRECTLY, NEVER use [\"value\"]",
            "   - Run DATA_ANALYSIS first to populate cache. Use list_cache() to see available keys before loading.",
            "   - If load_cache(\"key\") returns None, the key was not saved - run DATA_ANALYSIS or use fallback logic.",
            "   - Use historical data APIs (e.g., read_traffic_states) and combine with cached results",
            "   - **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.** Focus on one facility type or small group",
            "6. If planning fails, use DEBUG to rewrite all codes, fix errors, and retry",
            "7. Review simulation results and iterate: perform ≥2 POLICY_PLANNING actions",
            "   - First establishes baseline, subsequent refine based on results",
            "   - System auto-rolls back if new policy performs worse than best",
            "8. FINISH - Complete optimization when satisfied",
            "",
            "**Data Analysis Types:**",
            "- **Spatial Analysis**: Analyze the spatial distribution of traffic and the spatial correlation of traffic",
            "- **Temporal Analysis**: Analyze the temporal distribution of traffic and the temporal correlation of traffic",
            "- **Cross-Module Analysis**: Analyze the correlation (e.g. traffic flow, intervention effect) between different modules",
            "You can use these analysis types in DATA_ANALYSIS to analyze the network and identify the best optimization strategy.",
            "",
            "**Critical Rules:**",
            "- **VERY IMPORTANT**: Only ONE action per turn. DO NOT combine multiple actions.",
            "- ALWAYS separate DATA_ANALYSIS from POLICY_PLANNING.",
            "- In DATA_ANALYSIS: Use save_cache() instead of printing. DO NOT print more than 10 lines.",
            "- In POLICY_PLANNING: Start from current_config.copy(), define 'config' variable (DO NOT use return).",
            "- load_cache(\"key\") returns the value directly - NEVER append [\"value\"]. Example: x = load_cache(\"key\")",
            "- **CRITICAL: Keep code <50lines. Simplify your algorithm as much as possible.**",
            "- Control configs must respect engineering principles (min green time 15s, safe headways).",
        ])
        
        # Add module-specific notes
        if len(available_specs) > 1:
            message_parts.append("- Multiple transportation systems are interconnected - consider all when optimizing.")
        
        implemented_modules = [name for name in available_specs.keys() if name in [
            'signal_timing',
            'highway_speed_limit',
            'ramp_metering',
            'taxi_scheduling',
            'subway_scheduling',
            'bus_scheduling'
        ]]
        todo_modules = []
        
        if implemented_modules:
            module_list_impl = ', '.join([f'{name!r}' for name in implemented_modules])
            message_parts.append(f"- {module_list_impl} {'are' if len(implemented_modules) > 1 else 'is'} implemented.")
        
        # Add memory section if memory exists
        if self.memory:
            message_parts.extend([
                "",
                "**Previous Optimization Experience (Memory):**",
                "The following are lessons learned from previous optimization sessions:",
                ""
            ])
            for i, memory_item in enumerate(self.memory, 1):
                message_parts.append(f"{i}. {memory_item}")
            message_parts.extend([
                "",
                "Use these insights to guide your optimization strategy in this session."
            ])
        
        message_parts.extend([
            "",
            "Begin with PLAN, then GET_CONTROL_API to query available tools."
        ])
        
        return "\n".join(message_parts)
    
    def get_reflection_message(self, history: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a reflection prompt and add it to messages, then return updated messages list.
        
        This starts a dedicated reflection phase where the LLM can:
        - Perform up to max_reflection_turns turns of DATA_ANALYSIS actions using the sandbox and cache
        - Then summarize overall patterns of this network
        - Finally output an updated memory list as a JSON array in the last response
        
        Args:
            history: List of (action_type, action_result) tuples from the optimization session
            
        Returns:
            Updated messages list with reflection prompt added
        """
        prompt_parts = [
            "You have just completed an optimization session. Above is the complete conversation history",
            "including all your actions, code executions, simulation results, and feedback.",
            "",
            "**Current Memory** (from previous sessions):",
        ]
        
        if self.memory:
            for i, memory_item in enumerate(self.memory, 1):
                prompt_parts.append(f"{i}. {memory_item}")
        else:
            prompt_parts.append("(No previous memory)")
        
        # Add session summary for quick reference
        action_counts = {}
        for action_type, action_result in history:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        prompt_parts.extend([
            "",
            "**Session Summary** (for quick reference, but review full conversation above):",
            f"- Total turns: {self.turn_count}",
            f"- Actions taken: {', '.join([f'{k}({v})' for k, v in action_counts.items()])}",
        ])
        
        if self.best_simulation_result is not None and self.best_simulation_turn is not None:
            # Prefer reporting per-module best metrics if available
            module_metrics = self.best_simulation_result.get("module_metrics", {}) or {}
            reported_module_metrics = False

            if module_metrics:
                # Optionally filter to enabled/available modules
                if self.available_control_modules is not None:
                    filtered_module_metrics = {
                        name: metrics
                        for name, metrics in module_metrics.items()
                        if name in self.available_control_modules
                    }
                else:
                    filtered_module_metrics = module_metrics

                if filtered_module_metrics:
                    reported_module_metrics = True
                    prompt_parts.append("")
                    prompt_parts.append("**Best Module-level Results:**")
                    for module_name, metrics in filtered_module_metrics.items():
                        pretty_name = module_name.replace("_", " ").title()
                        metric_str_parts = []
                        for metric_name, metric_value in metrics.items():
                            display_name = metric_name.replace("_", " ").title()
                            if isinstance(metric_value, float):
                                metric_str_parts.append(f"{display_name}: {metric_value:.2f}")
                            else:
                                metric_str_parts.append(f"{display_name}: {metric_value}")
                        metric_str = ", ".join(metric_str_parts) if metric_str_parts else "N/A"

                        if self.best_simulation_turn == 0:
                            # Turn 0 means we are still at the initial baseline (no improvement yet)
                            prompt_parts.append(
                                f"- {pretty_name} best result is the same as the initial baseline (no improvement yet): {metric_str}"
                            )
                        else:
                            prompt_parts.append(
                                f"- {pretty_name} best at turn {self.best_simulation_turn}: {metric_str}"
                            )

            # Fallback to overall stats if no module metrics are available
            if not reported_module_metrics:
                best_stats = self.best_simulation_result.get("stats", {}) or {}
                avg_tt = best_stats.get("avg_travel_time", None)
                if isinstance(avg_tt, (int, float)):
                    prompt_parts.append(
                        f"- Best overall result at turn {self.best_simulation_turn}: avg_travel_time={avg_tt:.2f}s"
                    )
                else:
                    prompt_parts.append(
                        f"- Best overall result at turn {self.best_simulation_turn}: avg_travel_time=N/A"
                    )

            # Compare final best vs initial baseline to see how many modules improved
            if self.initial_best_simulation_result is not None:
                improvement_flags = self._is_better_simulation_result(
                    self.best_simulation_result,
                    self.initial_best_simulation_result
                )
                improved_modules_from_initial = [
                    name for name, is_better in improvement_flags.items()
                    if is_better
                ]
                if improved_modules_from_initial:
                    pretty_list = ", ".join(
                        name.replace("_", " ").title() for name in improved_modules_from_initial
                    )
                    prompt_parts.append(
                        f"- Compared to the initial baseline, {len(improved_modules_from_initial)} module(s) achieved better best results: {pretty_list}."
                    )
                else:
                    prompt_parts.append(
                        "- Compared to the initial baseline, no enabled module achieved a strictly better best result."
                    )
        
        prompt_parts.extend([
            "",
            "**Your Reflection Task (Multi-turn, using DATA_ANALYSIS):**",
            "1. You are now entering a short reflection phase.",
            f"2. You may perform up to **{self.max_reflection_turns} turns** of DATA_ANALYSIS actions to deeply analyze this network.",
            "   - In these DATA_ANALYSIS actions, you can:",
            "     - Re-use any cached data in the sandbox via save_cache()/load_cache()/list_cache()",
            "     - Load and analyze traffic states or other available data APIs",
            "     - Compute aggregate statistics and identify general patterns of this network",
            "     - Summarize reusable insights and patterns that can be used in future sessions",
            "     - Identify recurring failure patterns in your optimization history",
            "   - DO NOT perform POLICY_PLANNING or change control configs in this phase.",
            "3. After you finish your analyses (or you think you have enough information):",
            "   - In your **final reflection response**, you must use the action:",
            "     ACTION: REFLECTION_FINISH",
            "   - In that same final response, you must **ONLY** output an updated memory list.",
            "   - That final response should be a JSON array (list) of strings, wrapped in a ```json code block.",
            "",
            "**Final Output Format (Last Reflection Response with REFLECTION_FINISH):**",
            "ACTION: REFLECTION_FINISH",
            "```json",
            "[",
            "    \"INSIGHT 1\",",
            "    \"INSIGHT 2\",",
            "    \"INSIGHT 3\",",
            "    \"...\",",
            "    \"INSIGHT N\"",
            "]",
            "```",
            "",
            "**Important**:",
            f"- At the end of reflection, return ONLY a valid JSON array (maximum {self.max_memory_items} items total).",
            "- Include items you want to keep from current memory (you can update/refine them if needed).",
            "- Include new insights learned from this session and your additional DATA_ANALYSIS actions.",
            "- Exclude items you want to remove (simply don't include them in the array).",
            "- Each memory item should be a concise, actionable insight (one sentence).",
            "- Focus on robust patterns and general strategies that are likely to transfer to future sessions.",
            "- Do NOT include explanations or commentary outside the JSON array in the final response."
        ])
        
        # Add reflection prompt to messages
        reflection_prompt = "\n".join(prompt_parts)

        # Otherwise, append it normally
        self.add_user_message(reflection_prompt)
        
        # Initialize reflection phase state
        self.is_reflection_phase = True
        self.reflection_turn = 0
        
        # Return updated messages list
        return self.messages.copy()
    
    def update_memory(self, memory_items: List[str]) -> List[str]:
        """
        Update the agent's memory with new items.
        
        Args:
            memory_items: List of new memory items to set
            
        Returns:
            Updated memory list (after applying max_memory_items limit)
        """
        if len(memory_items) > self.max_memory_items:
            self.memory = memory_items[:self.max_memory_items]
        else:
            self.memory = memory_items.copy()
        
        return self.memory.copy()
    
    def get_memory(self) -> List[str]:
        """
        Get current memory items.
        
        Returns:
            List of current memory items
        """
        return self.memory.copy()
    
    def set_memory(self, memory_items: List[str]):
        """
        Set the agent's memory.
        
        Args:
            memory_items: List of memory items to set
        """
        self.memory = self._parse_and_validate_memory(memory_items)
    
    def _parse_and_validate_memory(self, memory_items: List[str]) -> List[str]:
        """
        Parse and validate memory items.
        
        Args:
            memory_items: List of memory items to validate
            
        Returns:
            Validated memory list
        """
        validated = []
        for item in memory_items:
            if isinstance(item, str) and item.strip():
                validated.append(item.strip())
        
        # Limit to max_memory_items
        if len(validated) > self.max_memory_items:
            validated = validated[:self.max_memory_items]
        
        return validated
    
    def update_memory_from_reflection(self, reflection_response: str, verbose: bool = True) -> List[str]:
        """
        Parse memory items from LLM reflection response and update internal memory.
        
        Args:
            reflection_response: LLM's reflection response text (should be a JSON array)
            verbose: Whether to print detailed logs
        
        Returns:
            Updated memory list
        """
        updated_memory = self._parse_memory_from_reflection(reflection_response)
        if updated_memory is None:
            if verbose:
                print("Warning: Failed to parse reflection memory; keeping existing memory.")
            return self.memory.copy()
        self.update_memory(updated_memory)
        
        if verbose:
            print(f"✓ Memory updated. Total memory items: {len(self.memory)}")
            for i, item in enumerate(self.memory, 1):
                print(f"  {i}. {item}")
        
        return self.memory.copy()
    
    def _parse_memory_from_reflection(self, reflection_response: str) -> Optional[List[str]]:
        """
        Parse memory items from LLM reflection response.
        Expects a JSON array of strings, optionally wrapped in ```json code block.
        
        Args:
            reflection_response: LLM's reflection response text (should be a JSON array)
            
        Returns:
            List of memory items (strings) if parsed, otherwise None
        """
        import json
        import re
        
        memory_items: List[str] = []
        
        try:
            # Try to parse as JSON first
            # Remove any markdown code blocks if present
            response_text = reflection_response.strip()
            
            # Remove "ACTION: REFLECTION_FINISH" line if present (case-insensitive)
            lines = response_text.split("\n")
            filtered_lines = []
            for line in lines:
                line_upper = line.strip().upper()
                # Skip lines that are ACTION: REFLECTION_FINISH or ACTION:REFLECTION_FINISH
                if not (line_upper.startswith("ACTION:") and "REFLECTION_FINISH" in line_upper):
                    filtered_lines.append(line)
            response_text = "\n".join(filtered_lines).strip()
            
            # Remove markdown code blocks (```json ... ``` or ``` ... ```)
            if response_text.startswith("```"):
                # Find the closing ```
                lines = response_text.split("\n")
                # Skip first line (```json or ```)
                start_idx = 1
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if i > 0 and line.strip().startswith("```"):
                        end_idx = i
                        break
                response_text = "\n".join(lines[start_idx:end_idx]).strip()
            
            # Parse JSON array
            parsed = json.loads(response_text)
            
            if isinstance(parsed, list):
                # Filter out empty strings and ensure all items are strings
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        memory_items.append(item.strip())
                    elif isinstance(item, (int, float)):
                        # Convert numbers to strings if needed
                        memory_items.append(str(item).strip())
            else:
                print(f"Warning: LLM reflection response is not a JSON array. Got type: {type(parsed)}")
                return None
        
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON array from the response
            print(f"Warning: Failed to parse JSON from reflection response: {e}")
            print(f"Response preview: {reflection_response[:200]}...")
            
            # Try to find JSON array in the response using regex
            json_match = re.search(r"\[[\s\S]*?\]", reflection_response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str) and item.strip():
                                memory_items.append(item.strip())
                        return memory_items
                    print(f"Warning: Extracted JSON is not a list. Got type: {type(parsed)}")
                    return None
                except json.JSONDecodeError:
                    print("Warning: Failed to parse extracted JSON array. Falling back to empty list.")
                    return None
            else:
                # Fallback: return empty list if we can't parse
                print("Warning: No JSON array found in reflection response. Returning empty list.")
                return None

        return memory_items

# ============================================================
# REST API Server for Remote LLMAgentManager Access
# ============================================================
#
# Architecture:
#   - Server A (SUMO Server): Runs SUMO + LLMAgentManager + FastAPI
#   - Server B (Agent Loop): Runs LLM inference and calls HTTP API
#
# Key Design:
#   - All checkpoint/graph loading happens on SUMO server (no transmission)
#   - Agent Loop only sends LLM responses and receives messages
#   - Each Master manages one SUMOEnv, multiple workers share checkpoint
# ============================================================

from dataclasses import dataclass, field
import asyncio
import uuid

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define dummy classes for type hints when FastAPI is not installed
    class BaseModel:
        pass


# --- Pydantic Request/Response Models ---

if FASTAPI_AVAILABLE:
    class MasterCreateRequest(BaseModel):
        """Request to create a new Master session."""
        session_id: Optional[str] = None  # Session ID (renamed from master_id)
        master_id: Optional[str] = None  # Backward compatibility alias
        config_path: Optional[str] = None  # Path to SUMO config file (optional if environments is provided)
        environments: Optional[List[Dict[str, Any]]] = None  # Multi-environment configuration
        current_env_index: int = 0  # Index of the environment to use initially (for batch parallel allocation)
        checkpoint_interval: int = 1800
        run_duration: int = 7200
        seed: int = 42
        use_gui: bool = False
        available_control_modules: Optional[List[str]] = None
        # LLMAgentManager settings
        max_turns: int = 10
        max_reflection_turns: int = 5
        max_memory_items: int = 10

    class MasterStepRequest(BaseModel):
        """Request to advance SUMO simulation to next checkpoint."""
        apply_best_policies: bool = True  # Whether to apply collected best policies

    class MasterStepResponse(BaseModel):
        """Response after advancing SUMO simulation."""
        checkpoint_index: int
        checkpoint_time: float
        simulation_finished: bool
        baseline_result: Optional[Dict[str, Any]] = None

    class WorkerResetRequest(BaseModel):
        """Request to reset/initialize a worker's LLMAgentManager."""
        initial_prompt: str
        memory: Optional[List[str]] = None
        initial_best_result: Optional[Dict[str, Any]] = None
        initial_control_configs: Optional[Dict[str, Dict[str, Any]]] = None

    class WorkerResetResponse(BaseModel):
        """Response after resetting a worker."""
        messages: List[Dict[str, Any]]
        worker_id: str

    class WorkerStepRequest(BaseModel):
        """Request to process LLM response in a worker."""
        llm_response: str
        verbose: bool = False

    class WorkerStepResponse(BaseModel):
        """Response after processing LLM response."""
        messages: List[Dict[str, Any]]
        action_result: Dict[str, Any]
        finished: bool
        turn_count: int

    class WorkerFinalizeRequest(BaseModel):
        """Request to finalize a worker and report best policy."""
        pass  # Best configs are extracted from the manager

    class WorkerStateResponse(BaseModel):
        """Response containing worker state."""
        turn_count: int
        is_reflection_phase: bool
        reflection_turn: int
        best_simulation_turn: Optional[int]
        best_simulation_result: Optional[Dict[str, Any]]
        best_control_configs: Optional[Dict[str, Dict[str, Any]]]

    class ReflectionRequest(BaseModel):
        """Request to start reflection phase."""
        history: List[Dict[str, Any]]  # List of (action_type, action_result) as dicts

    class ReflectionResponse(BaseModel):
        """Response after starting reflection."""
        messages: List[Dict[str, Any]]

    class UpdateMemoryRequest(BaseModel):
        """Request to update memory from reflection response."""
        reflection_response: str
        verbose: bool = False

    class UpdateMemoryResponse(BaseModel):
        """Response after updating memory."""
        memory: List[str]


# --- Master Session Data Structure ---

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if RAY_AVAILABLE:
    @ray.remote(max_concurrency=10)  # ✅ 允许多个 Worker 请求并发处理（每个 Worker 的策略仿真创建独立 SUMO 进程）
    class SimulationSessionActor:
        """
        ✅ Ray Actor 版本的 SimulationSession，每个 Session 在独立进程中运行。
        这样可以实现真正的多进程并行，避免 FastAPI 事件循环阻塞。
        
        注意：SUMOEnv 无法序列化，所以在 Actor 内部创建，而不是通过参数传递。
        """
        def __init__(self, session_id: str, env_config: Dict[str, Any],
                     max_turns: int = 15, max_reflection_turns: int = 5, max_memory_items: int = 10,
                     available_control_modules: List[str] = None,
                     environments: List[Dict[str, Any]] = None):
            self.session_id = session_id
            self.env_config = env_config
            self.max_turns = max_turns
            self.max_reflection_turns = max_reflection_turns
            self.max_memory_items = max_memory_items
            self.available_control_modules = available_control_modules or []
            self.environments = environments or []  # ✅ 保存环境列表，用于 restart 时切换环境
            self.workers = {}
            self.context = {}
            
            # ✅ 在 Actor 内部创建 SUMOEnv 和 TrafficStateCollector
            self.env = None
            self.traffic_state_collector = None
            self._initialize_env()
        
        def _initialize_env(self):
            """在 Actor 内部创建 SUMOEnv，避免序列化问题。"""
            try:
                from utils.simulation_utils import create_sumo_env
                from utils.traffic_state_collector import TrafficStateCollector
                from pathlib import Path
                
                print(f"🔄 Ray Actor {self.session_id}: Initializing SUMOEnv...")
                
                # 创建 SUMOEnv
                self.env, _ = create_sumo_env(
                    config_path=self.env_config['config_path'],
                    use_gui=self.env_config.get('use_gui', False),
                    seed=self.env_config.get('seed', 42),
                    control_modules=self.available_control_modules,
                    run_counts=self.env_config.get('checkpoint_interval', 1800),
                    use_unique_work_dir=True,
                )
                
                print(f"✅ Ray Actor {self.session_id}: SUMOEnv created")
                
                # 创建 TrafficStateCollector（使用与 Ray 模式相同的命名逻辑）
                from utils.traffic_state_collector import init_traffic_states_file
                
                config_dir_name = Path(self.env_config['config_path']).parent.name
                traffic_states_filepath = init_traffic_states_file(
                    simulation_id=self.session_id,
                    config_name=config_dir_name,
                    llm_name=None,
                    control_modules=self.available_control_modules
                )
                
                lane_dict = self.env.lane_dict if hasattr(self.env, 'lane_dict') else {}
                lane_inter_graph = self.env.lane_inter_graph if hasattr(self.env, 'lane_inter_graph') else None
                
                self.traffic_state_collector = TrafficStateCollector(
                    env=self.env,
                    traffic_states_filepath=str(traffic_states_filepath),
                    interval=30,
                    lane_dict=lane_dict,
                    lane_inter_graph=lane_inter_graph,
                    simulation_id=self.session_id
                )
                
                # ✅ 初始化 context，参照纯 Ray 模式的 Master.get_context() 
                from pathlib import Path as PathlibPath
                config_path = self.env_config['config_path']
                config_name = PathlibPath(config_path).parent.name if config_path else None
                
                self.context = {
                    # === 基本信息 ===
                    "traffic_states_filepath": traffic_states_filepath,
                    "control_modules": self.available_control_modules,
                    "enabled_modules": self.available_control_modules,  # 与 control_modules 相同
                    "simulation_id": self.session_id,
                    
                    # === 环境配置 ===
                    "config_path": config_path,
                    "config_name": config_name,
                    "current_env_index": 0,  # 初始为 0，restart 时更新
                    "current_env_episodes": 1,
                    
                    # === 仿真参数 ===
                    "checkpoint_interval": self.env_config.get('checkpoint_interval', 1800),
                    "test_duration": self.env_config.get('checkpoint_interval', 1800),
                    "seed": self.env_config.get('seed', 42),
                    "use_gui": self.env_config.get('use_gui', False),
                    
                    # === 模块依赖关系（与 Master.get_context() 保持一致）===
                    "cross_module_dependencies": {
                        "signal_timing": {"affects": ["bus_scheduling", "taxi_scheduling"], "affected_by": []},
                        "highway_speed_limit": {"affects": ["ramp_metering"], "affected_by": []},
                        "ramp_metering": {"affects": [], "affected_by": ["highway_speed_limit"]},
                        "bus_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
                        "taxi_scheduling": {"affects": [], "affected_by": ["signal_timing"]},
                        "subway_scheduling": {"affects": [], "affected_by": []},
                    },
                }
                
                # === 添加 Highway 相关数据 ===
                if hasattr(self.env, 'highway_dict'):
                    self.context["highway_ids"] = list(self.env.highway_dict.keys())
                if hasattr(self.env, 'highway_info_dict'):
                    self.context["highway_segment_dict"] = self.env.highway_info_dict
                
                # === 添加 Bus 相关数据 ===
                if hasattr(self.env, 'bus_route_info'):
                    self.context["bus_route_info"] = self.env.bus_route_info
                
                # === 添加 enabled_controls 和 current_configs（当前控制配置）===
                if hasattr(self.env, 'enabled_controls') and self.env.enabled_controls:
                    import copy
                    # 保存完整的 enabled_controls（包含 module 和 config）
                    self.context["enabled_controls"] = self.env.enabled_controls
                    
                    # 同时保存 current_configs（与 DeepCityMaster 保持一致）
                    control_configs = {}
                    for module_name, module_info in self.env.enabled_controls.items():
                        module = module_info.get('module')
                        config = module_info.get('config', {})
                        if module and config:
                            control_configs[module_name] = {
                                'module': module,
                                'config': copy.deepcopy(config)
                            }
                    self.context["current_configs"] = control_configs
                else:
                    self.context["enabled_controls"] = {}
                    self.context["current_configs"] = {}
                
                # 添加路网图结构信息（留在服务器端，不通过 HTTP 传输）
                if hasattr(self.env, 'get_road_network_graphs'):
                    try:
                        graphs = self.env.get_road_network_graphs()
                        self.context.update({
                            "lane_graph": graphs.get("lane_graph"),
                            "lane_inter_graph": graphs.get("lane_inter_graph"),
                            "intersection_graph": graphs.get("intersection_graph"),
                            "lane_group_graph": graphs.get("lane_group_graph"),
                            "road_graph": graphs.get("road_graph"),
                            "lane_dict": graphs.get("lane_dict"),
                            "road_dict": graphs.get("road_dict"),
                            "highway_segment_graph": graphs.get("highway_segment_graph"),
                            "ramp_lane_graph": graphs.get("ramp_lane_graph"),  # ✅ 添加 ramp_lane_graph
                        })
                    except Exception as e:
                        print(f"⚠️ Failed to get road network graphs: {e}")
                
                # ✅ 添加 highway_graph (即 highway_subgraph)
                if hasattr(self.env, 'highway_subgraph') and self.env.highway_subgraph is not None:
                    self.context["highway_graph"] = self.env.highway_subgraph
                
                # Layer 1: Foundation Layer - Complete network graphs
                if hasattr(self.env, 'network_graphs'):
                    self.context["network_graphs"] = self.env.network_graphs
                    self.context["full_lane_graph"] = self.env.network_graphs.get("lane_graph")
                    self.context["full_road_graph"] = self.env.network_graphs.get("road_graph")
                if hasattr(self.env, 'network_dicts'):
                    self.context["network_dicts"] = self.env.network_dicts
                    self.context["full_lane_dict"] = self.env.network_dicts.get("lane_dict")
                    self.context["full_road_dict"] = self.env.network_dicts.get("road_dict")
                
                # Layer 2: Zone infrastructure
                if hasattr(self.env, 'zone_dict'):
                    self.context["zone_dict"] = self.env.zone_dict
                if hasattr(self.env, 'zone_graph'):
                    self.context["zone_graph"] = self.env.zone_graph
                
                # Transit-specific data (for bus_scheduling & subway_scheduling)
                if hasattr(self.env, 'transit_graph'):
                    self.context["transit_graph"] = self.env.transit_graph
                if hasattr(self.env, 'bus_route_info'):
                    self.context["bus_route_info"] = self.env.bus_route_info
                
                # Ramp-specific data (for ramp_metering)
                if hasattr(self.env, 'ramp_lane_graph'):
                    self.context["ramp_lane_graph"] = self.env.ramp_lane_graph
                
                print(f"✅ Ray Actor {self.session_id}: SUMOEnv and TrafficStateCollector initialized")
                print(f"   Context keys: {list(self.context.keys())}")
                
            except Exception as e:
                import traceback
                error_msg = f"❌ Ray Actor {self.session_id}: Failed to initialize SUMOEnv: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # 重新抛出异常，让 Ray 知道初始化失败
                raise RuntimeError(error_msg) from e
            
        def run_simulation(self, duration: int, checkpoint_interval: int, 
                          control_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
            """
            在独立进程中运行 SUMO 仿真（同步调用，但不阻塞主进程）。
            """
            from utils.simulation_utils import run_controlled_simulation
            from pathlib import Path
            
            # 为每个 session 创建独立的 checkpoint 目录
            workspace_root = Path(__file__).parent.parent
            checkpoint_dir = str(workspace_root / "records" / "checkpoints" / self.session_id)
            
            # ✅ 从 env.enabled_controls 获取控制配置（与本地 Ray Master 保持一致）
            # 如果调用者传递了 control_configs，则优先使用调用者的配置
            if control_configs is None:
                control_configs = {}
                control_states = {}
                if hasattr(self.env, 'enabled_controls') and self.env.enabled_controls:
                    for module_name, module_info in self.env.enabled_controls.items():
                        control_configs[module_name] = module_info.get('config', {})
                        control_states[module_name] = module_info.get('state')
                    print(f"Session {self.session_id}: Using control modules from env: {list(control_configs.keys())}")
            else:
                control_states = None
            
            # ✅ 获取 config_name（与本地 Ray Master 保持一致）
            from pathlib import Path as PathlibPath
            config_name = PathlibPath(self.env_config.get('config_path', '')).parent.name if self.env_config.get('config_path') else None
            
            # ✅ 判断是否是第一次仿真（与本地 Ray Master 保持一致）
            current_time = self.env.get_current_time() if self.env else 0
            is_first_simulation = (current_time == 0)
            
            # ✅ 直接调用同步的 run_controlled_simulation
            # 因为这个方法运行在独立的 Ray Actor 进程中，不会阻塞其他 Session
            # ✅ 添加 try-except 与纯 Ray 模式的 _step_local 保持一致
            try:
                result = run_controlled_simulation(
                    env=self.env,
                    duration=duration,
                    step_seconds=30,  # ✅ 与本地 Ray Master 保持一致
                    min_step_seconds=1.0,  # ✅ 与本地 Ray Master 保持一致
                    save_checkpoint=True,
                    control_configs=control_configs if control_configs else None,
                    control_states=control_states,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=self.session_id,  # ✅ 只传递 session_id（例如 "master_2"）
                    is_first_simulation=is_first_simulation,  # ✅ 关键参数！
                    config_name=config_name,  # ✅ 与本地 Ray Master 保持一致
                    simulation_id=None,  # ✅ 不传递 simulation_id，避免重复
                    traffic_state_collector=self.traffic_state_collector,
                )
            except Exception as e:
                # ✅ 与纯 Ray 模式的 _step_local 保持一致：捕获异常并返回错误
                import traceback
                print(f"⚠️ Error during simulation step: {e}")
                traceback.print_exc()
                return {"success": False, "error": str(e)}
            
            # ✅ 更新控制状态到 env.enabled_controls（与本地 Ray Master 保持一致）
            if result.get("control_states"):
                for module_name, state in result["control_states"].items():
                    if hasattr(self.env, 'enabled_controls') and module_name in self.env.enabled_controls:
                        self.env.enabled_controls[module_name]['state'] = state
            
            # Update context
            checkpoint_path = result.get("checkpoint_path")
            if checkpoint_path:
                self.context['checkpoint_path'] = checkpoint_path
            
            baseline_result = result.get("stats") if result.get("success") else None
            current_time = result.get("final_time", 0)
            
            return {
                "success": result.get("success", False),
                "checkpoint_path": checkpoint_path,
                "current_time": current_time,
                "baseline_result": baseline_result,
                "checkpoint_reached": result.get("checkpoint_reached", False)
            }
        
        def get_context(self) -> Dict[str, Any]:
            """获取 Session 上下文。"""
            return self.context
        
        def restart(self, env_index: int, seed: int, episode_count: int) -> Dict[str, Any]:
            """
            重启仿真会话，切换到新的环境配置。
            
            Args:
                env_index: 新环境在 environments 列表中的索引
                seed: 新的随机种子
                episode_count: 当前 episode 计数
                
            Returns:
                Dictionary with session status after restart
            """
            from utils.simulation_utils import create_sumo_env
            from utils.traffic_state_collector import TrafficStateCollector
            from pathlib import Path
            import shutil
            
            print(f"🔄 Session {self.session_id}: Restarting with env_index={env_index}, seed={seed}, episode={episode_count}")
            
            # 关闭现有环境
            if self.env:
                try:
                    self.env.close()
                except Exception as e:
                    print(f"⚠️ Error closing env: {e}")
            
            # 获取新环境配置
            if hasattr(self, 'environments') and self.environments and env_index < len(self.environments):
                new_env = self.environments[env_index]
                config_path = new_env.get("sumo_config_path")
                control_modules = new_env.get("control_modules", [])
            else:
                # 回退到原始配置
                config_path = self.env_config.get('config_path')
                control_modules = self.available_control_modules
            
            print(f"  New config: {config_path}")
            print(f"  Control modules: {control_modules}")
            
            # 更新配置
            self.env_config['config_path'] = config_path
            self.env_config['seed'] = seed
            self.available_control_modules = control_modules
            
            # 创建新环境
            self.env, _ = create_sumo_env(
                config_path=config_path,
                use_gui=self.env_config.get('use_gui', False),
                seed=seed,
                control_modules=control_modules,
                run_counts=self.env_config.get('checkpoint_interval', 1800),
                use_unique_work_dir=True,
            )
            
            # 清理旧的 checkpoint 文件
            workspace_root = Path(__file__).parent.parent
            checkpoint_dir = workspace_root / "records" / "checkpoints" / self.session_id
            if checkpoint_dir.exists():
                try:
                    shutil.rmtree(checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  Deleted old checkpoint directory: {checkpoint_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to delete checkpoint directory: {e}")
            
            # 删除旧的 traffic_states 文件（如果存在）
            if self.traffic_state_collector and hasattr(self.traffic_state_collector, 'traffic_states_filepath'):
                old_traffic_states_file = self.traffic_state_collector.traffic_states_filepath
                if old_traffic_states_file and Path(old_traffic_states_file).exists():
                    try:
                        Path(old_traffic_states_file).unlink()
                        print(f"  Deleted old traffic_states file: {old_traffic_states_file}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete old traffic_states file: {e}")
            
            # 重新创建 TrafficStateCollector（使用与 Ray 模式相同的命名逻辑）
            from utils.traffic_state_collector import init_traffic_states_file
            
            config_dir_name = Path(config_path).parent.name
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=self.session_id,
                config_name=config_dir_name,
                llm_name=None,
                control_modules=control_modules
            )
            
            lane_dict = self.env.lane_dict if hasattr(self.env, 'lane_dict') else {}
            lane_inter_graph = self.env.lane_inter_graph if hasattr(self.env, 'lane_inter_graph') else None
            
            self.traffic_state_collector = TrafficStateCollector(
                env=self.env,
                traffic_states_filepath=traffic_states_filepath,
                interval=30,
                lane_dict=lane_dict,
                lane_inter_graph=lane_inter_graph,
                simulation_id=self.session_id
            )
            
            # 更新 context
            self.context = {
                "traffic_states_filepath": traffic_states_filepath,
                "control_modules": control_modules,
                "simulation_id": self.session_id,
                "config_path": config_path,
                "episode_count": episode_count,
            }
            
            # 添加路网图结构信息
            if hasattr(self.env, 'get_road_network_graphs'):
                try:
                    graphs = self.env.get_road_network_graphs()
                    self.context.update({
                        "lane_graph": graphs.get("lane_graph"),
                        "lane_inter_graph": graphs.get("lane_inter_graph"),
                        "intersection_graph": graphs.get("intersection_graph"),
                        "lane_group_graph": graphs.get("lane_group_graph"),
                        "road_graph": graphs.get("road_graph"),
                        "lane_dict": graphs.get("lane_dict"),
                        "road_dict": graphs.get("road_dict"),
                        "highway_segment_graph": graphs.get("highway_segment_graph"),
                        "ramp_lane_graph": graphs.get("ramp_lane_graph"),
                    })
                except Exception as e:
                    print(f"⚠️ Failed to get road network graphs: {e}")
            
            # Highway-specific data
            if hasattr(self.env, 'highway_dict'):
                self.context["highway_ids"] = list(self.env.highway_dict.keys())
            if hasattr(self.env, 'highway_subgraph'):
                self.context["highway_graph"] = self.env.highway_subgraph
            if hasattr(self.env, 'highway_info_dict'):
                self.context["highway_segment_dict"] = self.env.highway_info_dict
            
            # Layer 1: Foundation Layer - Complete network graphs
            if hasattr(self.env, 'network_graphs'):
                self.context["network_graphs"] = self.env.network_graphs
                self.context["full_lane_graph"] = self.env.network_graphs.get("lane_graph")
                self.context["full_road_graph"] = self.env.network_graphs.get("road_graph")
            if hasattr(self.env, 'network_dicts'):
                self.context["network_dicts"] = self.env.network_dicts
                self.context["full_lane_dict"] = self.env.network_dicts.get("lane_dict")
                self.context["full_road_dict"] = self.env.network_dicts.get("road_dict")
            
            # Layer 2: Zone infrastructure
            if hasattr(self.env, 'zone_dict'):
                self.context["zone_dict"] = self.env.zone_dict
            if hasattr(self.env, 'zone_graph'):
                self.context["zone_graph"] = self.env.zone_graph
            
            # Transit-specific data (for bus_scheduling & subway_scheduling)
            if hasattr(self.env, 'transit_graph'):
                self.context["transit_graph"] = self.env.transit_graph
            if hasattr(self.env, 'bus_route_info'):
                self.context["bus_route_info"] = self.env.bus_route_info
            
            # Ramp-specific data (for ramp_metering)
            if hasattr(self.env, 'ramp_lane_graph'):
                self.context["ramp_lane_graph"] = self.env.ramp_lane_graph
            
            # 清空 workers
            self.workers = {}
            
            print(f"✅ Session {self.session_id}: Restarted successfully")
            
            return {
                "success": True,
                "session_id": self.session_id,
                "config_path": config_path,
                "control_modules": control_modules,
                "seed": seed,
                "episode_count": episode_count,
            }
        
        # ==================== Worker Methods ====================
        
        def reset_worker(self, worker_id: str, initial_prompt: str, 
                        memory: List[str] = None,
                        initial_best_result: Dict = None,
                        initial_control_configs: Dict = None) -> Dict[str, Any]:
            """初始化/重置 Worker 的 LLMAgentManager。"""
            manager = LLMAgentManager(
                available_control_modules=self.available_control_modules,
                max_turns=self.max_turns,
                max_reflection_turns=self.max_reflection_turns,
                max_memory_items=self.max_memory_items
            )
            
            if memory:
                manager.set_memory(memory)
            
            messages = manager.reset(
                initial_prompt=initial_prompt,
                context=self.context,
                verbose=False,
                initial_best_result=initial_best_result,
                initial_control_configs=initial_control_configs
            )
            
            self.workers[worker_id] = manager
            return {"messages": messages, "worker_id": worker_id}
        
        def step_worker(self, worker_id: str, llm_response: str, verbose: bool = False) -> Dict[str, Any]:
            """处理 Worker 的 LLM 响应并执行策略仿真。"""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            
            manager = self.workers[worker_id]
            context_snapshot = self.context.copy()
            
            messages, action_result = manager.step(
                llm_response=llm_response,
                context=context_snapshot,
                env=None,
                verbose=verbose
            )
            
            return {
                "messages": messages,
                "action_result": action_result,
                "finished": action_result.get("finished", False),
                "turn_count": action_result.get("turn_count", manager.turn_count)
            }
        
        def get_worker_messages(self, worker_id: str) -> Dict[str, Any]:
            """获取 Worker 的当前 messages。"""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            return {"messages": manager.messages.copy(), "turn_count": manager.turn_count}
        
        def get_worker_state(self, worker_id: str) -> Dict[str, Any]:
            """获取 Worker 的状态。"""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            return {
                "turn_count": manager.turn_count,
                "is_reflection_phase": manager.is_reflection_phase,
                "reflection_turn": manager.reflection_turn,
                "best_simulation_turn": manager.best_simulation_turn,
                "best_simulation_result": manager.best_simulation_result,
                "best_control_configs": manager.best_control_configs
            }
        
        def finalize_worker(self, worker_id: str) -> Dict[str, Any]:
            """Finalize a worker and return its best policy."""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            return {
                "worker_id": worker_id,
                "best_control_configs": manager.best_control_configs.copy() if manager.best_control_configs else {},
                "best_simulation_result": manager.best_simulation_result.copy() if manager.best_simulation_result else None,
                "best_simulation_turn": manager.best_simulation_turn
            }
        
        def start_reflection(self, worker_id: str, history: List[Dict]) -> Dict[str, Any]:
            """Start reflection phase for a worker."""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            history_tuples = [(h.get("action_type", ""), h.get("action_result", {})) for h in history]
            messages = manager.get_reflection_message(history_tuples)
            return {"messages": messages}
        
        def update_memory(self, worker_id: str, llm_response: str) -> Dict[str, Any]:
            """从 reflection 响应更新 Worker 的 memory。"""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            manager.update_memory_from_reflection(llm_response)
            return {"memory": manager.get_memory()}
        
        def get_memory(self, worker_id: str) -> Dict[str, Any]:
            """获取 Worker 的 memory。"""
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not found")
            manager = self.workers[worker_id]
            return {"memory": manager.get_memory()}

# Fallback: 非 Ray 模式使用原始的 dataclass
@dataclass
class SimulationSession:
    """
    简化的仿真会话，只管理 SUMO 环境和 Worker。
    协调逻辑（checkpoint_index, episode_count 等）由 Ray Master 管理。
    """
    session_id: str
    env: Any  # SUMOEnv instance
    
    # LLMAgentManager settings (用于创建 Worker)
    max_turns: int = 15
    max_reflection_turns: int = 5
    max_memory_items: int = 10
    available_control_modules: List[str] = field(default_factory=list)

    # Traffic state collector for collecting historical data
    traffic_state_collector: Optional[Any] = None
    traffic_states_filepath: Optional[str] = None

    # Workers: worker_id -> LLMAgentManager
    workers: Dict[str, 'LLMAgentManager'] = field(default_factory=dict)

    # Async lock for thread safety
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # ✅ 保留必要的环境信息（用于返回给客户端）
    context: Dict[str, Any] = field(default_factory=dict)


# --- LLMAgentManager Server ---

class LLMAgentManagerServer:
    """
    REST API server for remote SUMO simulation service.
    
    提供无状态的仿真服务：
    - 管理 SUMO 环境实例
    - 执行主仿真和策略仿真
    - 协调逻辑由 Ray Master 管理
    """

    def __init__(self, use_ray: bool = True):
        """
        Args:
            use_ray: 是否使用 Ray Actor 管理 Session（实现多进程并行）
        """
        self.use_ray = use_ray and RAY_AVAILABLE
        self.sessions: Dict[str, Any] = {}  # Ray Actor 或 SimulationSession
        self._lock = asyncio.Lock()
        
        if self.use_ray:
            print("✅ SUMO Server: Using Ray Actor for multi-process parallelism")
            # 初始化 Ray（如果还没初始化）
            if not ray.is_initialized():
                # 启动独立的本地 Ray 集群，不连接到现有集群
                ray.init(
                    ignore_reinit_error=True,
                    address=None,  # 不连接到现有集群
                    _temp_dir="/tmp/ray_sumo_server"  # 使用独立的临时目录
                )
                print(f"✅ Ray initialized with local cluster")
        else:
            print("⚠️ SUMO Server: Using single-process mode (serial execution)")

    async def create_session(self, request: 'MasterCreateRequest') -> Dict[str, Any]:
        """创建新的仿真会话（SUMO 环境）。"""
        from pathlib import Path

        # ✅ 优先使用 session_id，如果没有则回退到 master_id（向后兼容）
        session_id = request.session_id or request.master_id or f"session_{uuid.uuid4().hex[:8]}"
        
        print(f"🔄 Creating session {session_id}...")

        # ✅ 如果 Session 已存在，直接返回（避免重复创建导致 Actor 被杀死）
        async with self._lock:
            if session_id in self.sessions:
                print(f"⚠️ Session {session_id} already exists, returning existing session")
                # 返回已存在的 session 信息
                if request.environments and len(request.environments) > 0:
                    first_env = request.environments[0]
                    config_path = first_env.get("sumo_config_path")
                    control_modules = first_env.get("control_modules", [])
                else:
                    config_path = request.config_path
                    control_modules = request.available_control_modules or []
                
                return {
                    "session_id": session_id,
                    "config_path": config_path,
                    "available_control_modules": control_modules,
                    "status": "existing"
                }

        # ✅ 解析环境配置（使用 current_env_index 支持批量并行分配）
        if request.environments and len(request.environments) > 0:
            # 使用 current_env_index 选择初始环境
            env_index = min(request.current_env_index, len(request.environments) - 1)
            current_env = request.environments[env_index]
            config_path = current_env.get("sumo_config_path")
            control_modules = current_env.get("control_modules", [])
            print(f"Session {session_id}: Using environment {env_index}/{len(request.environments)}")
        else:
            config_path = request.config_path
            control_modules = request.available_control_modules or []
            env_index = 0
        
        print(f"Session {session_id}: config={config_path}, modules={control_modules}")

        # ✅ Ray 模式：只创建 Actor，让 Actor 自己初始化 SUMOEnv
        if self.use_ray:
            env_config = {
                'config_path': config_path,
                'use_gui': request.use_gui,
                'seed': request.seed,
                'checkpoint_interval': request.checkpoint_interval,
            }
            
            try:
                session_actor = SimulationSessionActor.remote(
                    session_id=session_id,
                    env_config=env_config,
                    max_turns=request.max_turns,
                    max_reflection_turns=request.max_reflection_turns,
                    max_memory_items=request.max_memory_items,
                    available_control_modules=control_modules,
                    environments=request.environments or [],  # ✅ 传递环境列表，用于 restart 时切换环境
                )
                
                # ✅ 立即注册 session，不等待初始化完成（避免 HTTP 超时）
                async with self._lock:
                    self.sessions[session_id] = session_actor
                
                print(f"✅ Session {session_id} registered, initializing in background...")
                
                # ✅ 异步触发初始化（不等待结果）
                # Actor 会在后台初始化，客户端可以通过 get_session 轮询检查状态
                import ray
                import asyncio
                
                async def _initialize_actor():
                    try:
                        # 尝试获取 context 以触发初始化
                        await asyncio.to_thread(ray.get, session_actor.get_context.remote(), timeout=600)
                        print(f"✅ Session {session_id} initialized successfully")
                    except Exception as e:
                        # ⚠️ 初始化失败时不移除 session，只记录错误
                        # 客户端可以继续使用 session，Actor 会在首次调用时完成初始化
                        print(f"⚠️ Session {session_id} background initialization failed (will initialize on first use): {e}")
                        import traceback
                        traceback.print_exc()
                
                # 启动后台任务（不阻塞）
                asyncio.create_task(_initialize_actor())
                
                return {
                    "session_id": session_id,
                    "config_path": config_path,
                    "available_control_modules": control_modules,
                    "status": "initializing"
                }
                
            except Exception as e:
                import traceback
                error_msg = f"Failed to create session {session_id}: {str(e)}"
                print(f"❌ {error_msg}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=error_msg)
        
        # ✅ 非 Ray 模式：在主进程创建 SUMOEnv
        else:
            from utils.simulation_utils import create_sumo_env
            from utils.traffic_state_collector import TrafficStateCollector
            from utils.id_utils import generate_file_prefix
            
            # 创建 SUMOEnv
            try:
                env, _ = create_sumo_env(
                    config_path=config_path,
                    use_gui=request.use_gui,
                    seed=request.seed,
                    control_modules=control_modules,
                    run_counts=request.checkpoint_interval,
                    use_unique_work_dir=True,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to create SUMOEnv: {str(e)}")
            
            # 创建 TrafficStateCollector（使用与 Ray 模式相同的命名逻辑）
            from utils.traffic_state_collector import init_traffic_states_file
            
            config_dir_name = Path(config_path).parent.name
            traffic_states_filepath = init_traffic_states_file(
                simulation_id=session_id,
                config_name=config_dir_name,
                llm_name=None,
                control_modules=control_modules
            )
            
            traffic_state_collector = TrafficStateCollector(
                env=env,
                traffic_states_filepath=str(traffic_states_filepath),
                interval=30,
                lane_dict=env.lane_dict if hasattr(env, 'lane_dict') else {},
                lane_inter_graph=env.lane_inter_graph if hasattr(env, 'lane_inter_graph') else None,
                simulation_id=session_id
            )
            
            session = SimulationSession(
                session_id=session_id,
                env=env,
                max_turns=request.max_turns,
                max_reflection_turns=request.max_reflection_turns,
                max_memory_items=request.max_memory_items,
                available_control_modules=control_modules,
                traffic_state_collector=traffic_state_collector,
                traffic_states_filepath=traffic_states_filepath,
            )
            self._update_context(session)
            async with self._lock:
                self.sessions[session_id] = session
            
            print(f"✅ Session {session_id} created successfully (non-Ray mode)")
            
            return {
                "session_id": session_id,
                "config_path": config_path,
                "available_control_modules": control_modules,
                "traffic_states_filepath": traffic_states_filepath
            }

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取仿真会话的 context（参照纯 Ray 模式的 Master.get_context()）。
        
        注意：HTTP 模式下不返回复杂图结构（NetworkX 图等无法 JSON 序列化），
        这些数据留在服务器端，Worker 通过 HTTP API 调用服务器端方法进行推理。
        
        返回的字段与纯 Ray 模式的 Master.get_context() 保持一致（除图结构外）。
        """
        async with self._lock:
            if session_id not in self.sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            session = self.sessions[session_id]
        
        # Ray Actor 模式：获取完整 context，过滤掉图结构
        if self.use_ray:
            import ray
            import asyncio
            try:
                context = await asyncio.to_thread(ray.get, session.get_context.remote())
                
                # 参照 Master.get_context() 返回所有 Worker 需要的字段
                # 只排除无法 JSON 序列化的图结构和 module 对象
                
                # 处理 current_configs：只保留 config，排除 module 对象（无法序列化）
                current_configs = context.get("current_configs", {})
                serializable_configs = {}
                for mod_name, mod_info in current_configs.items():
                    if isinstance(mod_info, dict):
                        serializable_configs[mod_name] = {
                            'config': mod_info.get('config', {})
                            # 排除 'module' 键，因为 Python 类对象无法 JSON 序列化
                        }
                
                # 处理 enabled_controls：只保留 config 部分（与 current_configs 类似）
                enabled_controls = context.get("enabled_controls", {})
                serializable_enabled_controls = {}
                for mod_name, mod_info in enabled_controls.items():
                    if isinstance(mod_info, dict):
                        serializable_enabled_controls[mod_name] = {
                            'config': mod_info.get('config', {})
                            # 排除 'module' 和其他不可序列化的字段
                        }
                
                result = {
                    # === 基本信息 ===
                    "session_id": session_id,
                    "simulation_id": context.get("simulation_id"),
                    
                    # === Checkpoint 信息 ===
                    "checkpoint_path": context.get("checkpoint_path"),
                    "checkpoint_path_t_minus_1": context.get("checkpoint_path_t_minus_1"),
                    "checkpoint_time": context.get("checkpoint_time"),
                    "current_time": context.get("current_time", 0),
                    
                    # === 环境配置 ===
                    "config_path": context.get("config_path"),
                    "config_name": context.get("config_name"),
                    "control_modules": context.get("control_modules"),
                    "current_env_index": context.get("current_env_index", 0),
                    "current_env_episodes": context.get("current_env_episodes", 1),
                    "enabled_modules": context.get("enabled_modules"),
                    "enabled_controls": serializable_enabled_controls,  # 只包含 config，不含 module
                    "current_configs": serializable_configs,  # 只包含 config，不含 module
                    
                    # === 仿真参数 ===
                    "checkpoint_interval": context.get("checkpoint_interval", 1800),
                    "test_duration": context.get("test_duration", 1800),
                    "run_duration": context.get("run_duration"),
                    "seed": context.get("seed", 42),
                    "use_gui": context.get("use_gui", False),
                    
                    # === 交通数据路径 ===
                    "traffic_states_filepath": context.get("traffic_states_filepath"),
                    
                    # === 模块指标和依赖 ===
                    "module_metrics": context.get("module_metrics", {}),
                    "cross_module_dependencies": context.get("cross_module_dependencies", {}),
                    
                    # === Highway/Ramp 相关（简单数据） ===
                    "highway_ids": context.get("highway_ids", []),
                    "highway_segment_dict": context.get("highway_segment_dict"),
                    
                    # === Taxi 相关（简单数据） ===
                    "taxi_fleet_state": context.get("taxi_fleet_state"),
                    "pending_reservations": context.get("pending_reservations"),
                    "taz_stats": context.get("taz_stats"),
                    "current_taxi_config": context.get("current_taxi_config"),
                    "taxi_dispatch_algorithm": context.get("taxi_dispatch_algorithm"),
                    "taxi_idle_algorithm": context.get("taxi_idle_algorithm"),
                    
                    # === Bus 相关 ===
                    "bus_route_info": context.get("bus_route_info"),
                }
                
                # 过滤掉 None 值，减少传输量
                return {k: v for k, v in result.items() if v is not None}
                
            except Exception as e:
                print(f"⚠️ Failed to get context from Ray Actor: {e}")
                import traceback
                traceback.print_exc()
                return {"session_id": session_id, "error": str(e)}
        
        # 非 Ray 模式：返回 session 的 context（过滤图结构）
        else:
            ctx = session.context
            return {
                "session_id": session_id,
                "simulation_id": ctx.get("simulation_id"),
                "traffic_states_filepath": session.traffic_states_filepath,
                "control_modules": ctx.get("control_modules"),
                "config_path": ctx.get("config_path"),
                "checkpoint_path": ctx.get("checkpoint_path"),
                "checkpoint_interval": ctx.get("checkpoint_interval", 1800),
                "worker_count": len(session.workers),
                "worker_ids": list(session.workers.keys()),
            }

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除仿真会话并关闭 SUMOEnv。"""
        async with self._lock:
            if session_id not in self.sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            session = self.sessions.pop(session_id)

        # Ray 模式：终止 Actor
        if self.use_ray:
            import ray
            try:
                ray.kill(session)
                print(f"✅ Killed Ray Actor for {session_id}")
            except Exception as e:
                print(f"⚠️ Error killing Ray Actor for {session_id}: {e}")
        # 非 Ray 模式：关闭 SUMOEnv
        else:
            try:
                if session.env is not None:
                    session.env.close()
            except Exception as e:
                print(f"⚠️ Error closing SUMOEnv for {session_id}: {e}")

        return {"session_id": session_id, "deleted": True}

    async def run_simulation(
        self, 
        session_id: str, 
        duration: int,
        checkpoint_interval: int,
        control_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        运行 SUMO 仿真（无状态服务）。
        
        Args:
            session_id: 仿真会话 ID
            duration: 仿真时长（秒）
            checkpoint_interval: Checkpoint 间隔（秒）
            control_configs: 控制配置（可选）
        
        Returns:
            仿真结果，包含 checkpoint_path 和 baseline_result
        """
        session = self._get_session(session_id)

        # ✅ Ray Actor 模式：异步调用 Ray Actor 的 run_simulation 方法
        if self.use_ray:
            import ray
            import asyncio
            print(f"Running simulation on Ray Actor for {session_id}")
            # 使用 asyncio.to_thread 包裹 ray.get，避免阻塞事件循环
            # 这样 4 个 Session 的仿真可以并行执行
            object_ref = session.run_simulation.remote(
                duration=duration,
                checkpoint_interval=checkpoint_interval,
                control_configs=control_configs
            )
            result = await asyncio.to_thread(ray.get, object_ref)
            return result
        
        # ✅ 单进程模式：使用锁保护
        async with session.lock:
            try:
                from utils.simulation_utils import run_controlled_simulation

                # Get default control configs and states from environment's enabled_controls
                control_states = None
                if control_configs is None:
                    control_configs = {}
                    control_states = {}
                    if hasattr(session.env, 'enabled_controls') and session.env.enabled_controls:
                        for module_name, module_info in session.env.enabled_controls.items():
                            control_configs[module_name] = module_info.get('config', {})
                            control_states[module_name] = module_info.get('state')

                print(f"Running simulation with control modules: {list(control_configs.keys())}")

                # 为每个 session 创建独立的 checkpoint 目录
                from pathlib import Path
                workspace_root = Path(__file__).parent.parent
                checkpoint_dir = str(workspace_root / "records" / "checkpoints" / session_id)
                
                # ✅ 获取 config_name（与本地 Ray Master 保持一致）
                config_name = Path(session.context.get('config_path', '')).parent.name if session.context.get('config_path') else None
                
                # ✅ 判断是否是第一次仿真（与本地 Ray Master 保持一致）
                current_time = session.env.get_current_time() if session.env else 0
                is_first_simulation = (current_time == 0)

                result = run_controlled_simulation(
                    env=session.env,
                    duration=duration,
                    step_seconds=30,  # ✅ 与本地 Ray Master 保持一致
                    min_step_seconds=1.0,  # ✅ 与本地 Ray Master 保持一致
                    save_checkpoint=True,
                    control_configs=control_configs if control_configs else None,
                    control_states=control_states,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=session_id,  # ✅ 只传递 session_id（例如 "master_2"）
                    is_first_simulation=is_first_simulation,  # ✅ 关键参数！
                    config_name=config_name,  # ✅ 与本地 Ray Master 保持一致
                    simulation_id=None,  # ✅ 不传递 simulation_id，避免重复
                    traffic_state_collector=session.traffic_state_collector,
                )
                
                # ✅ 更新控制状态到 env.enabled_controls
                if result.get("control_states"):
                    for module_name, state in result["control_states"].items():
                        if hasattr(session.env, 'enabled_controls') and module_name in session.env.enabled_controls:
                            session.env.enabled_controls[module_name]['state'] = state

                # Update context with new checkpoint data
                self._update_context(session)

                # Extract checkpoint_path from simulation result
                checkpoint_path = result.get("checkpoint_path")
                if checkpoint_path:
                    session.context['checkpoint_path'] = checkpoint_path
                    print(f"Checkpoint saved: {checkpoint_path}")

                # Get baseline result
                baseline_result = result.get("stats") if result.get("success") else None
                current_time = result.get("final_time", 0)

                return {
                    "success": result.get("success", False),
                    "checkpoint_path": checkpoint_path,
                    "current_time": current_time,
                    "baseline_result": baseline_result,
                    "checkpoint_reached": result.get("checkpoint_reached", False)
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

    async def restart_session(
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
        session = self._get_session(session_id)
        
        # Ray 模式：调用 Actor 的 restart 方法
        if self.use_ray:
            import ray
            import asyncio
            result = await asyncio.to_thread(
                ray.get,
                session.restart.remote(
                    env_index=env_index,
                    seed=seed,
                    episode_count=episode_count
                )
            )
            return result
        
        # 非 Ray 模式：本地处理（需要实现 SimulationSession.restart）
        raise HTTPException(status_code=501, detail="Non-Ray mode restart not implemented")

    async def reset_worker(
        self,
        session_id: str,
        worker_id: str,
        request: 'WorkerResetRequest'
    ) -> Dict[str, Any]:
        """初始化/重置 Worker 的 LLMAgentManager。"""
        session = self._get_session(session_id)

        # Ray 模式：调用 Actor 的方法
        if self.use_ray:
            import ray
            import asyncio
            result = await asyncio.to_thread(
                ray.get,
                session.reset_worker.remote(
                    worker_id=worker_id,
                    initial_prompt=request.initial_prompt,
                    memory=request.memory,
                    initial_best_result=request.initial_best_result,
                    initial_control_configs=request.initial_control_configs
                )
            )
            return result

        # 非 Ray 模式：本地处理
        async with session.lock:
            manager = LLMAgentManager(
                available_control_modules=session.available_control_modules,
                max_turns=session.max_turns,
                max_reflection_turns=session.max_reflection_turns,
                max_memory_items=session.max_memory_items
            )

            if request.memory:
                manager.set_memory(request.memory)

            messages = manager.reset(
                initial_prompt=request.initial_prompt,
                context=session.context,
                verbose=False,
                initial_best_result=request.initial_best_result,
                initial_control_configs=request.initial_control_configs
            )

            session.workers[worker_id] = manager

        return {"messages": messages, "worker_id": worker_id}

    async def step_worker(
        self,
        session_id: str,
        worker_id: str,
        request: 'WorkerStepRequest'
    ) -> Dict[str, Any]:
        """处理 Worker 的 LLM 响应并执行策略仿真。"""
        session = self._get_session(session_id)

        # Ray 模式：调用 Actor 的方法
        if self.use_ray:
            import ray
            import asyncio
            result = await asyncio.to_thread(
                ray.get,
                session.step_worker.remote(
                    worker_id=worker_id,
                    llm_response=request.llm_response,
                    verbose=request.verbose
                )
            )
            return result

        # 非 Ray 模式：本地处理
        manager = self._get_worker(session, worker_id)
        async with session.lock:
            context_snapshot = session.context.copy()
            if 'current_configs' in context_snapshot:
                context_snapshot['current_configs'] = session.context['current_configs'].copy()

        messages, action_result = manager.step(
            llm_response=request.llm_response,
            context=context_snapshot,
            env=None,
            verbose=request.verbose
        )

        return {
            "messages": messages,
            "action_result": action_result,
            "finished": action_result.get("finished", False),
            "turn_count": action_result.get("turn_count", manager.turn_count)
        }

    async def get_worker_messages(self, session_id: str, worker_id: str) -> Dict[str, Any]:
        """获取 Worker 的当前 messages。"""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.get_worker_messages.remote(worker_id))
        
        manager = self._get_worker(session, worker_id)
        return {"messages": manager.messages.copy(), "turn_count": manager.turn_count}

    async def get_worker_state(self, session_id: str, worker_id: str) -> Dict[str, Any]:
        """获取 Worker 的状态，包括最佳结果。"""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.get_worker_state.remote(worker_id))
        
        manager = self._get_worker(session, worker_id)
        return {
            "turn_count": manager.turn_count,
            "is_reflection_phase": manager.is_reflection_phase,
            "reflection_turn": manager.reflection_turn,
            "best_simulation_turn": manager.best_simulation_turn,
            "best_simulation_result": manager.best_simulation_result,
            "best_control_configs": manager.best_control_configs
        }

    async def finalize_worker(
        self,
        session_id: str,
        worker_id: str,
        request: 'WorkerFinalizeRequest'
    ) -> Dict[str, Any]:
        """Finalize a worker and return its best policy."""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.finalize_worker.remote(worker_id))
        
        manager = self._get_worker(session, worker_id)
        best_configs = manager.best_control_configs.copy() if manager.best_control_configs else {}
        best_result = manager.best_simulation_result.copy() if manager.best_simulation_result else None
        return {
            "worker_id": worker_id,
            "best_control_configs": best_configs,
            "best_simulation_result": best_result,
            "best_simulation_turn": manager.best_simulation_turn
        }

    async def start_reflection(
        self,
        session_id: str,
        worker_id: str,
        request: 'ReflectionRequest'
    ) -> Dict[str, Any]:
        """Start reflection phase for a worker."""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.start_reflection.remote(worker_id, request.history))
        
        manager = self._get_worker(session, worker_id)
        history = [(h.get("action_type", ""), h.get("action_result", {})) for h in request.history]
        async with session.lock:
            messages = manager.get_reflection_message(history)
        return {"messages": messages}

    async def update_memory(
        self,
        session_id: str,
        worker_id: str,
        request: 'UpdateMemoryRequest'
    ) -> Dict[str, Any]:
        """从 reflection 响应更新 Worker 的 memory。"""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.update_memory.remote(worker_id, request.reflection_response))
        
        manager = self._get_worker(session, worker_id)
        async with session.lock:
            memory = manager.update_memory_from_reflection(request.reflection_response, verbose=request.verbose)
        return {"memory": memory}

    async def get_memory(self, session_id: str, worker_id: str) -> Dict[str, Any]:
        """获取 Worker 的当前 memory。"""
        session = self._get_session(session_id)
        
        if self.use_ray:
            import ray
            import asyncio
            return await asyncio.to_thread(ray.get, session.get_memory.remote(worker_id))
        
        manager = self._get_worker(session, worker_id)
        return {"memory": manager.get_memory()}

    def _get_session(self, session_id: str) -> SimulationSession:
        """获取仿真会话或抛出 404 错误。"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return self.sessions[session_id]

    def _get_worker(self, session: SimulationSession, worker_id: str) -> 'LLMAgentManager':
        """获取 Worker 的 LLMAgentManager 或抛出 404 错误。"""
        if worker_id not in session.workers:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
        return session.workers[worker_id]

    def _update_context(self, session: SimulationSession) -> None:
        """更新 context，包含环境数据（graphs, traffic_states 等）。"""
        env = session.env

        # Get graphs and data from environment
        if hasattr(env, 'lane_graph'):
            session.context['lane_graph'] = env.lane_graph
        if hasattr(env, 'lane_inter_graph'):
            session.context['lane_inter_graph'] = env.lane_inter_graph
        if hasattr(env, 'intersection_graph'):
            session.context['intersection_graph'] = env.intersection_graph
        if hasattr(env, 'lane_dict'):
            session.context['lane_dict'] = env.lane_dict

        # Add road graphs for highway/ramp control
        if hasattr(env, 'road_graph'):
            session.context['road_graph'] = env.road_graph
        if hasattr(env, 'road_dict'):
            session.context['road_dict'] = env.road_dict
        if hasattr(env, 'highway_graph'):
            session.context['highway_graph'] = env.highway_graph
        if hasattr(env, 'highway_segment_graph'):
            session.context['highway_segment_graph'] = env.highway_segment_graph
        if hasattr(env, 'highway_segment_dict'):
            session.context['highway_segment_dict'] = env.highway_segment_dict
        if hasattr(env, 'ramp_lane_graph'):
            session.context['ramp_lane_graph'] = env.ramp_lane_graph

        # Add zone infrastructure for taxi scheduling
        if hasattr(env, 'zone_dict'):
            session.context['zone_dict'] = env.zone_dict
        if hasattr(env, 'zone_graph'):
            session.context['zone_graph'] = env.zone_graph
        if hasattr(env, 'transit_graph'):
            session.context['transit_graph'] = env.transit_graph
        if hasattr(env, 'bus_route_info'):
            session.context['bus_route_info'] = env.bus_route_info

        # Get current control configs
        if hasattr(env, 'enabled_controls'):
            current_configs = {}
            for module_name, module_info in env.enabled_controls.items():
                config = module_info.get('config', {})
                if config:
                    current_configs[module_name] = config
            session.context['current_configs'] = current_configs

            # Add module-specific current configs for easier access
            if 'signal_timing' in current_configs:
                session.context['current_signal_config'] = current_configs['signal_timing']
            if 'bus_scheduling' in current_configs:
                session.context['current_bus_schedule'] = current_configs['bus_scheduling']
            if 'subway_scheduling' in current_configs:
                session.context['current_subway_schedule'] = current_configs['subway_scheduling']
            if 'taxi_scheduling' in current_configs:
                session.context['current_taxi_config'] = current_configs['taxi_scheduling']
            if 'highway_speed_limit' in current_configs:
                session.context['current_highway_config'] = current_configs['highway_speed_limit']
            if 'ramp_metering' in current_configs:
                session.context['current_ramp_config'] = current_configs['ramp_metering']

        # Note: checkpoint_path is NOT stored on env object
        # It's returned by run_controlled_simulation and set in run_simulation
        # Don't overwrite it here if it's already set

        # Add available control modules
        session.context['available_control_modules'] = session.available_control_modules
        session.context['control_modules'] = session.available_control_modules  # Alias
        
        # Add session ID for identification
        session.context['session_id'] = session.session_id

        # Add taxi-specific context if taxi_scheduling is enabled
        if 'taxi_scheduling' in session.available_control_modules and hasattr(env, 'enabled_controls'):
            taxi_info = env.enabled_controls.get('taxi_scheduling', {})
            taxi_module = taxi_info.get('module')
            if taxi_module:
                # Get taxi dispatch/idle algorithm settings
                if hasattr(env, 'taxi_dispatch_algorithm'):
                    session.context['taxi_dispatch_algorithm'] = env.taxi_dispatch_algorithm
                if hasattr(env, 'taxi_idle_algorithm'):
                    session.context['taxi_idle_algorithm'] = env.taxi_idle_algorithm

        # Add traffic states filepath for LLM to read historical data
        if session.traffic_states_filepath:
            session.context['traffic_states_filepath'] = session.traffic_states_filepath



# --- FastAPI Application Factory ---

def create_app() -> 'FastAPI':
    """Create FastAPI application with all routes."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="LLMAgentManager Server",
        description="REST API for remote LLMAgentManager access in distributed SUMO simulation",
        version="1.0.0"
    )

    # ✅ 默认启用 Ray Actor 模式，实现真正的多进程并行
    server = LLMAgentManagerServer(use_ray=True)

    # --- Session Routes ---

    @app.post("/sessions", response_model=None)
    async def create_session(request: MasterCreateRequest):
        """创建新的仿真会话（SUMO 环境）。"""
        return await server.create_session(request)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        """获取仿真会话状态。"""
        return await server.get_session(session_id)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """删除仿真会话并关闭 SUMOEnv。"""
        return await server.delete_session(session_id)

    @app.post("/sessions/{session_id}/run_simulation")
    async def run_simulation_route(session_id: str, duration: int, checkpoint_interval: int):
        """运行 SUMO 仿真（无状态服务）。"""
        return await server.run_simulation(
            session_id=session_id,
            duration=duration,
            checkpoint_interval=checkpoint_interval,
            control_configs=None
        )

    @app.post("/sessions/{session_id}/restart")
    async def restart_session_route(session_id: str, request: dict):
        """重启仿真会话，切换到新的环境配置。"""
        return await server.restart_session(
            session_id=session_id,
            env_index=request.get("env_index", 0),
            seed=request.get("seed", 42),
            episode_count=request.get("episode_count", 0)
        )

    # --- Worker Routes ---

    @app.post("/sessions/{session_id}/workers/{worker_id}/reset")
    async def reset_worker(session_id: str, worker_id: str, request: WorkerResetRequest):
        """初始化/重置 Worker 的 LLMAgentManager。"""
        return await server.reset_worker(session_id, worker_id, request)

    @app.post("/sessions/{session_id}/workers/{worker_id}/step")
    async def step_worker(session_id: str, worker_id: str, request: WorkerStepRequest):
        """处理 Worker 的 LLM 响应并执行策略仿真。"""
        return await server.step_worker(session_id, worker_id, request)

    @app.get("/sessions/{session_id}/workers/{worker_id}/messages")
    async def get_worker_messages(session_id: str, worker_id: str):
        """获取 Worker 的当前 messages。"""
        return await server.get_worker_messages(session_id, worker_id)

    @app.get("/sessions/{session_id}/workers/{worker_id}/state")
    async def get_worker_state(session_id: str, worker_id: str):
        """获取 Worker 的状态，包括最佳结果。"""
        return await server.get_worker_state(session_id, worker_id)

    @app.post("/sessions/{session_id}/workers/{worker_id}/finalize")
    async def finalize_worker(session_id: str, worker_id: str, request: WorkerFinalizeRequest):
        """Finalize a worker and return its best policy."""
        return await server.finalize_worker(session_id, worker_id, request)

    # --- Reflection Routes ---

    @app.post("/sessions/{session_id}/workers/{worker_id}/reflection/start")
    async def start_reflection(session_id: str, worker_id: str, request: ReflectionRequest):
        """启动 Worker 的 reflection 阶段。"""
        return await server.start_reflection(session_id, worker_id, request)

    @app.post("/sessions/{session_id}/workers/{worker_id}/reflection/update_memory")
    async def update_memory(session_id: str, worker_id: str, request: UpdateMemoryRequest):
        """从 reflection 响应更新 Worker 的 memory。"""
        return await server.update_memory(session_id, worker_id, request)

    @app.get("/sessions/{session_id}/workers/{worker_id}/memory")
    async def get_memory(session_id: str, worker_id: str):
        """获取 Worker 的当前 memory。"""
        return await server.get_memory(session_id, worker_id)

    # --- Health Check ---

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "sessions_count": len(server.sessions)}

    return app


# --- CLI Entry Point ---

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the LLMAgentManager server."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not installed. Install with: pip install fastapi uvicorn")
        return

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMAgentManager REST API Server")
    parser.add_argument("--serve", action="store_true", help="Start the REST API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
    else:
        print("Use --serve to start the REST API server")
        print("Example: python -m utils.llm_agent_manager --serve --host 0.0.0.0 --port 8000")
