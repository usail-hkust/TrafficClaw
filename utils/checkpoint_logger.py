"""
Checkpoint Logger for recording optimization and simulation processes.

Records checkpoint-level information including:
1. Control module metrics
2. LLM agent conversation messages
3. Policy simulation results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def effective_config(entry: Any) -> Any:
    """Extract the actual config dict from a control_configs entry if it's a bundle {'module': ..., 'config': ...}."""
    if isinstance(entry, dict) and "config" in entry:
        return entry["config"]
    return entry


def extract_configs_only(control_configs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract only config dictionaries from control_configs bundle format, removing module instances for JSON serialization."""
    if not control_configs:
        return None
    configs_only = {}
    for module_name, entry in control_configs.items():
        configs_only[module_name] = effective_config(entry)
    return configs_only

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


class CheckpointLogger:
    """
    Logger for recording checkpoint-based optimization and simulation processes.
    
    Records:
    - Control module performance metrics
    - LLM agent conversation messages
    - Policy simulation results
    """
    
    def __init__(
        self,
        simulation_id: str,
        output_dir: Optional[Path] = None,
        llm_model_name: Optional[str] = None
    ):
        """
        Initialize checkpoint logger.
        
        Args:
            simulation_id: Unique identifier for this simulation run
            output_dir: Directory to save log files. If None, uses records/checkpoint_logs/
            llm_model_name: Name of the LLM model used (for filename)
        """
        self.simulation_id = simulation_id
        self.llm_model_name = llm_model_name
        
        # Determine output directory
        if output_dir is None:
            output_dir = workspace_root / "records" / "checkpoint_logs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        # Generate fixed filenames (same file for all checkpoints)
        if self.llm_model_name:
            model_name_safe = self.llm_model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        else:
            model_name_safe = "unknown_model"
        
        # Fixed filename for checkpoint log (same file, overwritten each time)
        self.checkpoint_log_filename = f"checkpoint_log_{self.simulation_id}_{model_name_safe}.json"
        
        # Fixed filename for conversations (same file, overwritten each time)
        self.conversation_filename = f"all_checkpoints_conversation_{self.simulation_id}_{model_name_safe}.json"
        
        # Initialize checkpoint data storage
        self.checkpoints: List[Dict[str, Any]] = []
        self.simulation_metadata: Dict[str, Any] = {
            "simulation_id": simulation_id,
            "llm_model_name": llm_model_name,
            "start_time": datetime.now().isoformat(),
            "checkpoints": []
        }
    
    def log_checkpoint(
        self,
        checkpoint_number: int,
        checkpoint_path: Optional[str] = None,
        checkpoint_path_t_minus_1: Optional[str] = None,
        elapsed_time: float = 0.0,
        remaining_duration: float = 0.0,
        step_count: int = 0,
        avg_travel_time: float = 0.0,
        module_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        llm_agent_messages: Optional[List[Dict[str, str]]] = None,
        llm_optimization_result: Optional[Dict[str, Any]] = None,
        policy_simulation_results: Optional[List[Dict[str, Any]]] = None,
        control_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Log a checkpoint with all relevant information.
        
        Note: Each policy_simulation_result should contain "control_configs" field
        indicating which control configs were tested in that simulation.
        """
        """
        Log a checkpoint with all relevant information.
        
        Args:
            checkpoint_number: Checkpoint number (1-indexed)
            checkpoint_path: Path to checkpoint file (t state)
            checkpoint_path_t_minus_1: Path to t-1 checkpoint file
            elapsed_time: Elapsed simulation time in seconds
            remaining_duration: Remaining simulation duration in seconds
            step_count: Number of simulation steps
            avg_travel_time: Average travel time
            module_metrics: Dictionary of control module metrics
                          Format: {module_name: {metric_name: value, ...}}
            llm_agent_messages: List of LLM agent conversation messages
                              Format: [{"role": "system|user|assistant", "content": "..."}, ...]
            llm_optimization_result: Result from LLM agent optimization
                                   Format: {"success": bool, "turn_count": int, "final_control_configs": {...}, ...}
            policy_simulation_results: List of policy simulation results from SIMULATION actions
                                      Format: [{"success": bool, "stats": {...}, "module_metrics": {...}, "control_configs": {...}}, ...]
                                      Each result should contain "control_configs" indicating which configs were tested
            control_configs: Control configurations applied at this checkpoint
                           Format: {module_name: config_dict}
        """
        checkpoint_data = {
            "checkpoint_number": checkpoint_number,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "checkpoint_path_t_minus_1": checkpoint_path_t_minus_1,
            "simulation_time": {
                "elapsed_time": elapsed_time,
                "remaining_duration": remaining_duration,
                "step_count": step_count,
                "avg_travel_time": avg_travel_time
            },
            "control_module_metrics": module_metrics or {},
            "llm_agent": {
                "messages": llm_agent_messages or [],
                "optimization_result": llm_optimization_result or {}
            },
            "policy_simulations": policy_simulation_results or [],
            "control_configs": control_configs or {}
        }
        
        self.checkpoints.append(checkpoint_data)
        self.simulation_metadata["checkpoints"].append(checkpoint_data)
    
    def save_log(self, filename: Optional[str] = None) -> Path:
        """
        Save checkpoint log to JSON file.
        Uses the same filename for all checkpoints (overwrites previous saves).
        
        Args:
            filename: Custom filename. If None, uses fixed filename based on simulation_id and model_name.
            
        Returns:
            Path to saved log file
        """
        # Use fixed filename if not specified
        if filename is None:
            filename = self.checkpoint_log_filename
        
        filepath = self.output_dir / filename
        
        # Update end time
        self.simulation_metadata["end_time"] = datetime.now().isoformat()
        self.simulation_metadata["total_checkpoints"] = len(self.checkpoints)
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.simulation_metadata, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_all_conversations(self, filename: Optional[str] = None) -> Path:
        """
        Save all checkpoint conversations to a single JSON file.
        This is called after each checkpoint optimization completes.
        Uses the same filename for all checkpoints (overwrites previous saves).
        
        Args:
            filename: Custom filename. If None, uses fixed filename based on simulation_id and model_name.
            
        Returns:
            Path to saved conversation file
        """
        # Determine output directory (use llm_conversations directory)
        output_dir = workspace_root / "records" / "llm_conversations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed filename if not specified
        if filename is None:
            filename = self.conversation_filename
        
        filepath = output_dir / filename
        
        # Collect all conversations from all checkpoints
        all_conversations = []
        for checkpoint in self.checkpoints:
            checkpoint_number = checkpoint.get("checkpoint_number", 0)
            messages = checkpoint.get("llm_agent", {}).get("messages", [])
            optimization_result = checkpoint.get("llm_agent", {}).get("optimization_result", {})
            
            checkpoint_conversation = {
                "checkpoint_number": checkpoint_number,
                "timestamp": checkpoint.get("timestamp"),
                "messages": messages,
                "optimization_result": {
                    "success": optimization_result.get("success", False),
                    "turn_count": optimization_result.get("turn_count", 0),
                    "final_control_configs": optimization_result.get("final_control_configs", {}),
                    "error": optimization_result.get("error")
                }
            }
            all_conversations.append(checkpoint_conversation)
        
        # Prepare data to save
        conversation_data = {
            "metadata": {
                "simulation_id": self.simulation_id,
                "llm_model_name": self.llm_model_name,
                "total_checkpoints": len(all_conversations),
                "timestamp": datetime.now().isoformat(),
                "checkpoints_included": [cp["checkpoint_number"] for cp in all_conversations]
            },
            "checkpoints": all_conversations
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def add_policy_simulation_result(
        self,
        checkpoint_number: int,
        simulation_result: Dict[str, Any]
    ) -> None:
        """
        Add a policy simulation result to a checkpoint.
        
        Args:
            checkpoint_number: Checkpoint number to add result to
            simulation_result: Result from run_policy_simulation
                              Should contain "control_configs" field indicating
                              which control configs were tested in this simulation
        """
        # Ensure simulation_result contains control_configs if available
        # The control_configs should already be in simulation_result from execute_simulation_action
        # But we ensure it's properly formatted
        
        # Find the checkpoint and add simulation result
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_number"] == checkpoint_number:
                if "policy_simulations" not in checkpoint:
                    checkpoint["policy_simulations"] = []
                
                # Create a copy of simulation_result to avoid modifying original
                result_entry = simulation_result.copy()
                
                # Ensure control_configs is included (should already be there from execute_simulation_action)
                if "control_configs" not in result_entry:
                    result_entry["control_configs"] = {}
                    # Note: This shouldn't happen if execute_simulation_action is working correctly
                
                checkpoint["policy_simulations"].append(result_entry)
                
                # Also update in metadata
                for cp in self.simulation_metadata["checkpoints"]:
                    if cp["checkpoint_number"] == checkpoint_number:
                        if "policy_simulations" not in cp:
                            cp["policy_simulations"] = []
                        # Use the same cleaned result_entry
                        cp["policy_simulations"].append(result_entry)
                        break
                break
    
    def update_checkpoint_llm_messages(
        self,
        checkpoint_number: int,
        messages: List[Dict[str, str]]
    ) -> None:
        """
        Update LLM agent messages for a checkpoint.
        
        Args:
            checkpoint_number: Checkpoint number to update
            messages: List of LLM agent messages
        """
        # Find the checkpoint and update messages
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_number"] == checkpoint_number:
                checkpoint["llm_agent"]["messages"] = messages
                # Also update in metadata
                for cp in self.simulation_metadata["checkpoints"]:
                    if cp["checkpoint_number"] == checkpoint_number:
                        cp["llm_agent"]["messages"] = messages
                        break
                break
    
    def update_checkpoint_optimization_result(
        self,
        checkpoint_number: int,
        optimization_result: Dict[str, Any]
    ) -> None:
        """
        Update LLM optimization result for a checkpoint.
        
        Args:
            checkpoint_number: Checkpoint number to update
            optimization_result: Result from LLM agent optimization
        """
        # Find the checkpoint and update optimization result
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_number"] == checkpoint_number:
                checkpoint["llm_agent"]["optimization_result"] = optimization_result
                # Also update control_configs if final_control_configs is in optimization_result
                if "final_control_configs" in optimization_result:
                    checkpoint["control_configs"] = optimization_result["final_control_configs"]
                # Also update in metadata
                for cp in self.simulation_metadata["checkpoints"]:
                    if cp["checkpoint_number"] == checkpoint_number:
                        cp["llm_agent"]["optimization_result"] = optimization_result
                        if "final_control_configs" in optimization_result:
                            cp["control_configs"] = optimization_result["final_control_configs"]
                        break
                break
    
    def update_checkpoint_control_configs(
        self,
        checkpoint_number: int,
        control_configs: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Update control configurations for a checkpoint.
        
        Args:
            checkpoint_number: Checkpoint number to update
            control_configs: Control configurations dictionary (configs only, no module instances)
        """
        # Find the checkpoint and update control_configs
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_number"] == checkpoint_number:
                checkpoint["control_configs"] = control_configs
                # Also update in metadata
                for cp in self.simulation_metadata["checkpoints"]:
                    if cp["checkpoint_number"] == checkpoint_number:
                        cp["control_configs"] = control_configs
                        break
                break

