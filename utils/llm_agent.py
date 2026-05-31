"""
LLM Agent V2 - Simplified version using LLMAgentManager.

This version is a simplified refactoring of the original LLMAgent class that delegates
most functionality to LLMAgentManager. The key improvements are:

1. **Separation of Concerns**: 
   - LLMAgentV2 focuses only on LLM interaction and optimization loop coordination
   - LLMAgentManager handles all message management, action execution, and result processing

2. **Simplified Code**:
   - Removed ~2000+ lines of action handling logic (now in LLMAgentManager)
   - Removed message formatting logic (now in LLMAgentManager)
   - Removed code execution logic (now in LLMAgentManager)
   - Removed simulation execution logic (now in LLMAgentManager)

3. **Maintained Compatibility**:
   - Same public API as original LLMAgent
   - Same return format from run_optimization()
   - Can be used as a drop-in replacement

Usage:
    from utils.llm_agent import LLMAgentV2
    
    agent = LLMAgentV2(
        model_name="gpt-4",
        max_turns=10,
        available_control_modules=["signal_timing", "highway_speed_limit"]
    )
    
    result = agent.run_optimization(
        initial_prompt="Optimize traffic signals...",
        context={...},
        env=env
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from utils.language_models import LLM
from utils.llm_agent_manager import LLMAgentManager
from utils.checkpoint_logger import extract_configs_only, effective_config

# Context length limits for message truncation (Qwen3-8B)
MAX_CONTEXT_TOKENS = 32000
RESERVED_RESPONSE_TOKENS = 8192
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - RESERVED_RESPONSE_TOKENS  # 24576

# Lazy-loaded Qwen3-8B tokenizer for token counting
_QWEN_TOKENIZER = None


def _get_qwen_tokenizer():
    """Lazy load Qwen3-8B tokenizer for token counting."""
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is None and HAS_TRANSFORMERS:
        try:
            _QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load Qwen3-8B tokenizer: {e}. Message truncation will be disabled.")
    return _QWEN_TOKENIZER


def _count_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens in messages using Qwen3-8B tokenizer (approximate for chat format)."""
    tokenizer = _get_qwen_tokenizer()
    if tokenizer is None:
        return 0
    try:
        # Use apply_chat_template for accurate token count with chat format
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        pass
    # Fallback: sum token counts per message
    try:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(tokenizer.encode(content, add_special_tokens=False))
        return total
    except Exception:
        return 0


def _truncate_messages_for_context(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Truncate messages when input exceeds MAX_INPUT_TOKENS (32k - 8192 for response).
    Preserves: first 3 (system, user, assistant), last user message.
    Removes middle user-assistant pairs to fit within budget.
    Each remaining message stays in user-assistant pair form.
    """
    if len(messages) <= 4:
        return messages
    tokenizer = _get_qwen_tokenizer()
    if tokenizer is None:
        return messages
    # Quick check: skip truncation if under limit
    if _count_tokens(messages) <= MAX_INPUT_TOKENS:
        return messages
    n = len(messages)
    # First 3: system, user, assistant (must keep)
    # Last: user (must keep)
    # Middle: indices 3..n-2, form pairs (user, assistant), (user, assistant), ...
    first_three = messages[:3]
    last_user = messages[-1]
    if last_user.get("role") != "user":
        return messages  # Unexpected format, don't truncate
    middle = messages[3 : n - 1]
    # Build pairs: (msg[i], msg[i+1]) for i in 0,2,4,...
    pairs = []
    i = 0
    while i + 1 < len(middle):
        if middle[i].get("role") == "user" and middle[i + 1].get("role") == "assistant":
            pairs.append([middle[i], middle[i + 1]])
            i += 2
        else:
            i += 1
    # Check total tokens
    def _tokens(msgs: List[Dict]) -> int:
        return _count_tokens(msgs)

    total = _tokens(first_three) + _tokens([last_user])
    if total > MAX_INPUT_TOKENS:
        return messages  # Can't fit even minimal, return as-is
    # Add pairs from most recent to oldest until we exceed budget
    kept_pairs = []
    for pair in reversed(pairs):
        candidate = first_three + kept_pairs + pair + [last_user]
        if _tokens(candidate) <= MAX_INPUT_TOKENS:
            kept_pairs = pair + kept_pairs
        else:
            break
    return first_three + kept_pairs + [last_user]


class LLMAgent:
    """
    Simplified LLM Agent that uses LLMAgentManager for most functionality.
    
    This agent focuses on:
    - LLM initialization and inference
    - Optimization loop coordination
    - Delegating action handling to LLMAgentManager
    
    All message management, action execution, and result processing is handled by LLMAgentManager.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_turns: int = 10,
        available_control_modules: Optional[List[str]] = None,
        config_name: Optional[str] = None,
        max_memory_items: int = 10,
        max_reflection_turns: int = 5,
        decision_context_manager: Optional[Any] = None,
        is_joint_control: bool = False,
        module_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM Agent.

        Args:
            model_name: Name of the language model to use
            temperature: Sampling temperature for LLM
            max_turns: Maximum number of dialogue turns
            available_control_modules: List of available control module names
            config_name: Config directory name for file naming
            max_memory_items: Maximum number of memory items to keep (default: 10)
            max_reflection_turns: Maximum number of reflection turns (default: 5)
            decision_context_manager: Optional DecisionContextManager for cross-module coordination
            is_joint_control: Whether to use joint control system message formatting
            module_metrics: Optional module performance metrics for joint control mode
            base_url: Base URL for API (if None, will use default from provider)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_turns = max_turns
        self.available_control_modules = available_control_modules
        self.config_name = config_name
        self.max_memory_items = max_memory_items
        self.max_reflection_turns = max_reflection_turns

        # Initialize LLM instance
        self.llm = LLM(
            model_path=model_name,
            temperature=temperature,
            max_tokens=8192,
            base_url=base_url
        )

        # Initialize LLMAgentManager to handle message management and action execution
        self.manager = LLMAgentManager(
            available_control_modules=available_control_modules,
            max_turns=max_turns,
            config_name=config_name,
            max_memory_items=max_memory_items,
            max_reflection_turns=max_reflection_turns,
            decision_context_manager=decision_context_manager,
            is_joint_control=is_joint_control,
            module_metrics=module_metrics
        )
    
    def run_optimization(
        self,
        initial_prompt: str,
        context: Dict[str, Any],
        env: Optional[Any] = None,
        verbose: bool = True,
        initial_best_result: Optional[Dict[str, Any]] = None,
        initial_control_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        checkpoint_logger: Optional[Any] = None,
        checkpoint_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the optimization process with multi-turn dialogue.
        
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
            env: Optional SUMOEnv instance to access enabled control modules
            verbose: Whether to print detailed logs
            initial_best_result: Initial best simulation result to set (e.g., from checkpoint)
            initial_control_configs: Initial control configs to set as best configs (e.g., from checkpoint)
            checkpoint_logger: Optional checkpoint logger (ignored by LLMAgent)
            checkpoint_number: Checkpoint index for checkpoint_logger (ignored by LLMAgent)
            
        Returns:
            Dictionary containing:
                - success: Whether optimization completed successfully
                - final_control_configs: All control module configs (if any)
                - turn_count: Number of dialogue turns
                - history: List of (action_type, action_result) tuples
                - best_simulation_result: Best simulation result (if any)
                - best_simulation_turn: Turn when best result was achieved
        """
        _ = checkpoint_logger
        _ = checkpoint_number
        # Reset manager and initialize messages. Memory persists across checkpoints.
        messages = self.manager.reset(
            initial_prompt=initial_prompt,
            context=context,
            verbose=verbose,
            initial_best_result=initial_best_result,
            initial_control_configs=initial_control_configs
        )
        
        history: List[Tuple[str, Any]] = []
        finished_with_finish = False
        
        if verbose:
            print(f"\n{'='*80}")
            print("LLM AGENT OPTIMIZATION SESSION")
            print(f"{'='*80}")
            print(f"Model: {self.model_name}")
            print(f"Max Turns: {self.max_turns}")
            print(f"Available Control Modules: {self.available_control_modules or 'All'}\n")
        
        # Optimization loop
        # Note: step() increments turn_count internally, so we check before calling it
        while self.manager.turn_count < self.max_turns:
            # Get LLM response (truncate messages if exceeding 32k to reserve 8192 for response)
            try:
                inference_messages = _truncate_messages_for_context(messages)
                llm_response = self.llm.inference_messages(inference_messages)
            except Exception as e:
                error_msg = f"ERROR: Failed to get LLM response: {str(e)}"
                if verbose:
                    print(f"Error: {error_msg}")
                    import traceback
                    traceback.print_exc()
                
                # Add error message to conversation
                self.manager.add_user_message(error_msg)
                history.append(("ERROR", {"error": error_msg}))
                break
            
            # Process LLM response using manager
            # step() will increment turn_count internally
            messages, action_result = self.manager.step(
                llm_response=llm_response,
                context=context,
                env=env,
                verbose=verbose
            )
            
            # Record action in history
            history.append((action_result["action_type"], action_result["action_result"]))
            
            # Check if finished
            if action_result["finished"]:
                finished_with_finish = True
                if verbose:
                    print("\n✓ Optimization completed with FINISH action")
                break
            
            # Check if we've reached max turns after step() incremented turn_count
            if self.manager.turn_count >= self.max_turns:
                if verbose:
                    print(f"\n⚠ Maximum turns ({self.max_turns}) reached")
                break
        
        # Reflection phase after optimization
        if history:
            try:
                if verbose:
                    if finished_with_finish:
                        print("\n[Reflection] Optimization completed with FINISH action, starting reflection...")
                    else:
                        print("\n[Reflection] Optimization ended, starting reflection...")
                self.reflect_and_update_memory(
                    history=history,
                    context=context,
                    env=env,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: reflection phase failed with error: {e}")
                    import traceback
                    traceback.print_exc()

        # Prepare return dictionary
        # Extract only configs (remove module instances) from best/current control_configs
        raw_final_configs = self.manager.best_control_configs.copy() if self.manager.best_control_configs else self.manager.current_control_configs.copy()
        final_configs = extract_configs_only(raw_final_configs) if raw_final_configs else {}
        
        # Clean history to remove module instances from action results
        cleaned_history = []
        for action_type, action_result in history:
            if isinstance(action_result, dict):
                cleaned_result = action_result.copy()
                # Clean final_control_configs if present
                if "final_control_configs" in cleaned_result:
                    cleaned_result["final_control_configs"] = extract_configs_only(cleaned_result["final_control_configs"]) or {}
                # Clean control_configs if present
                if "control_configs" in cleaned_result:
                    cleaned_result["control_configs"] = extract_configs_only(cleaned_result["control_configs"]) or {}
                # Clean best_simulation_result if present (e.g. in FINISH action_result)
                if "best_simulation_result" in cleaned_result and isinstance(cleaned_result["best_simulation_result"], dict):
                    br = cleaned_result["best_simulation_result"].copy()
                    if "control_configs" in br:
                        br["control_configs"] = extract_configs_only(br["control_configs"]) or {}
                    cleaned_result["best_simulation_result"] = br
                cleaned_history.append((action_type, cleaned_result))
            else:
                cleaned_history.append((action_type, action_result))
        
        # Clean best_simulation_result if it contains control_configs
        best_result = None
        if self.manager.best_simulation_result:
            best_result = self.manager.best_simulation_result.copy()
            if isinstance(best_result, dict) and "control_configs" in best_result:
                best_result["control_configs"] = extract_configs_only(best_result["control_configs"]) or {}
        
        result = {
            "success": True,
            "final_control_configs": final_configs,
            "turn_count": self.manager.turn_count,
            "history": cleaned_history,
            "best_simulation_result": best_result,
            "best_simulation_turn": self.manager.best_simulation_turn
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print("OPTIMIZATION SESSION COMPLETED")
            print(f"{'='*80}")
            print(f"Total Turns: {self.manager.turn_count}/{self.max_turns}")
            if self.manager.best_simulation_result:
                print(f"Best Result Achieved at Turn: {self.manager.best_simulation_turn}")
            print(f"Final Control Configs: {list(final_configs.keys())}")
        
        return result
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get current conversation messages."""
        return self.manager.messages.copy()
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get best simulation result."""
        return self.manager.best_simulation_result.copy() if self.manager.best_simulation_result else None
    
    def get_best_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get best control configurations."""
        return self.manager.best_control_configs.copy() if self.manager.best_control_configs else {}
    
    def reflect_and_update_memory(
        self,
        history: List[tuple],
        context: Dict[str, Any],
        env: Optional[Any] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Perform reflection on optimization session and update memory based on LLM feedback.
        
        This method should be called after run_optimization() to extract insights
        from the completed optimization session and update the agent's memory for future sessions.
        
        Args:
            history: List of (action_type, action_result) tuples from the optimization session
            verbose: Whether to print detailed logs
            
        Returns:
            List of updated memory items (the complete new memory list)
        """
        import traceback

        max_reflection_turns = self.manager.max_reflection_turns
        
        if verbose:
            print(f"\n[Reflection] Starting reflection phase with up to {max_reflection_turns} analysis turns...")
        
        # Add reflection prompt (manager will append it to messages)
        reflection_messages = self.manager.get_reflection_message(history)
        
        last_reflection_response: Optional[str] = None
        reflection_turn = 0
        reflection_finished = False

        try:
            while reflection_turn < max_reflection_turns:
                reflection_turn += 1
                if verbose:
                    print(f"\n[Reflection] Turn {reflection_turn}/{max_reflection_turns}")

                # Get LLM response for reflection / analysis (truncate if exceeding 32k)
                inference_messages = _truncate_messages_for_context(reflection_messages)
                last_reflection_response = self.llm.inference_messages(inference_messages)

                if verbose:
                    print(f"LLM Reflection Turn Response:\n{last_reflection_response}\n")

                # Process response using regular manager step so DATA_ANALYSIS code
                # can execute in the sandbox and use cache
                reflection_messages, action_result = self.manager.step(
                    llm_response=last_reflection_response,
                    context=context,
                    env=env,
                    verbose=verbose
                )

                # If LLM uses REFLECTION_FINISH during reflection, stop early
                if action_result["action_type"] == "REFLECTION_FINISH":
                    reflection_finished = True
                    if verbose:
                        print("[Reflection] REFLECTION_FINISH received, ending reflection phase.")
                    break

            # After up to max_reflection_turns (or FINISH), use the last LLM response
            # to update memory (it should now summarize overall patterns as JSON list)
            if reflection_finished and last_reflection_response is not None:
                updated_memory = self.manager.update_memory_from_reflection(
                    last_reflection_response,
                    verbose=verbose
                )
            else:
                if verbose and not reflection_finished:
                    print("[Reflection] REFLECTION_FINISH not received; keeping existing memory.")
                updated_memory = self.manager.get_memory()
            
            # Ensure reflection phase state is reset after reflection ends
            if self.manager.is_reflection_phase:
                self.manager.is_reflection_phase = False
                self.manager.reflection_turn = 0

            return updated_memory

        except Exception as e:
            print(f"Error during reflection phase: {e}")
            traceback.print_exc()
            # Ensure reflection phase state is reset even on error
            if self.manager.is_reflection_phase:
                self.manager.is_reflection_phase = False
                self.manager.reflection_turn = 0
            return self.manager.get_memory()
    
    def get_memory(self) -> List[str]:
        """
        Get current memory items.
        
        Returns:
            List of current memory items
        """
        return self.manager.get_memory()

    def add_memory(self, memory_items: Any) -> List[str]:
        """
        Add memory items to the agent.

        Args:
            memory_items: Single memory item or list of items to add

        Returns:
            Updated memory list
        """
        return self.manager.add_memory(memory_items)
    
    def set_memory(self, memory_items: List[str]):
        """
        Set the agent's memory for next optimization sessions.
        
        Args:
            memory_items: List of memory items to set
        """
        self.manager.set_memory(memory_items)
    
