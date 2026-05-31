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
DeepCity Reward Manager for transportation optimization RL training.

New Reward Function (2026-03):
total_reward = (metric_improvement + best_judge_score) × improvement_count + cumulative_module_count_reward

Components:
1. metric_improvement_reward: Mean of metric improvements across all modules (best vs baseline)
   - Calculated only once at FINISH using best simulation result
   - Each module's improvement = mean of its key metrics' improvement rates
   - Key metrics per module:
     * signal_timing: avg_travel_time↓, avg_queue_len↓
     * bus_scheduling: avg_passenger_waiting_time↓, total_fuel_consumption_g↓
     * highway_speed_limit: avg_travel_time↓
     * subway_scheduling: avg_passenger_waiting_time↓, electricity_consumption↓
     * taxi_scheduling: avg_wait_time↓, total_income↑

2. judge_score: LLM evaluation score for the entire reasoning trajectory
   - Calculated at FINISH by evaluating the complete dialogue history
   - Range: [0.0, 1.0]

3. improvement_count: Number of times POLICY_PLANNING achieved improvement
   - Incremented when improved_modules is non-empty
   - Counts any improvement relative to previous best, not necessarily vs baseline

4. cumulative_module_count_reward: Sum of module improvement ratios across all improvement turns
   - Each improvement turn adds: num_improved_modules / total_modules
   - Example: Turn 3 improves 2/4 modules (+0.5), Turn 7 improves 4/4 modules (+1.0) → total = 1.5

Example:
- 4 modules, 2 improvements (Turn 3: 2/4 modules, Turn 7: 4/4 modules)
- Turn 3: metric=0.3, judge=0.8
- Turn 7: metric=0.5 (best), judge=0.9 (best)
- Final reward = (0.5 + 0.9) × 2 + (0.5 + 1.0) = 1.4 × 2 + 1.5 = 4.3
"""

import json
import os
import logging
from collections import defaultdict
from datetime import datetime

import torch

from verl import DataProto
from .registry import register

logger = logging.getLogger(__name__)


@register("deepcity")
class DeepCityRewardManager:
    """
    Reward manager for DeepCity transportation optimization.
    
    Unlike NaiveRewardManager which uses compute_score to evaluate text answers,
    DeepCityRewardManager directly uses cumulative_reward from the interaction's extra_info.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine: int = 1,
        compute_score=None,  # Not used, kept for interface compatibility
        reward_fn_key: str = "data_source",  # Not used, kept for interface compatibility
        **kwargs
    ) -> None:
        """
        Initialize DeepCityRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs (for debugging output).
            num_examine: Number of samples to print for debugging.
            compute_score: Not used, kept for interface compatibility.
            reward_fn_key: Not used, kept for interface compatibility.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.examined_count = 0
        self.batch_count = 0  # Global batch counter
        
        # Reward log file path (from environment variable or default)
        log_dir = os.environ.get("DEEPCITY_LOG_DIR", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.reward_log_path = os.path.join(log_dir, "reward_breakdown.jsonl")
        logger.info(f"DeepCityRewardManager: reward log will be written to {self.reward_log_path}")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute rewards for a batch of DeepCity rollouts.
        
        The reward is directly taken from extra_info["cumulative_reward"] 
        set by DeepCityInteraction during the 10-turn dialogue.
        
        Args:
            data: DataProto containing batch data and non_tensor_batch with extra_info
            return_dict: If True, return dict with reward_tensor and reward_extra_info
            
        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # If rm_scores already exists, use it directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        batch_records = []  # Collect reward breakdown records for this batch
        self.batch_count += 1

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Get extra_info from non_tensor_batch (set by DeepCityAgentLoop)
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            # Get cumulative_reward as reward (new reward function)
            cumulative_reward = extra_info.get("cumulative_reward", 0.0)
            
            # New reward function components
            improvement_count = extra_info.get("improvement_count", 0)
            cumulative_module_count_reward = extra_info.get("cumulative_module_count_reward", 0.0)
            judge_score = extra_info.get("judge_score", 0.0)  # LLM score for entire reasoning trajectory (calculated at FINISH)
            metric_improvement_reward = extra_info.get("metric_improvement_reward", 0.0)
            
            # Legacy fields (for comparison)
            module_count_reward = extra_info.get("module_count_reward", 0.0)
            
            total_modules = extra_info.get("total_modules", 1)
            num_turns = data_item.non_tensor_batch.get("__num_turns__", 0)
            
            # Use cumulative_reward as the reward
            # New formula: (metric_improvement + judge_score) × improvement_count + cumulative_module_count_reward
            reward = float(cumulative_reward)
            
            # Store extra info
            reward_extra_info["cumulative_reward"].append(cumulative_reward)
            reward_extra_info["improvement_count"].append(improvement_count)
            reward_extra_info["cumulative_module_count_reward"].append(cumulative_module_count_reward)
            reward_extra_info["judge_score"].append(judge_score)
            reward_extra_info["metric_improvement_reward"].append(metric_improvement_reward)
            reward_extra_info["module_count_reward"].append(module_count_reward)  # Legacy
            reward_extra_info["total_modules"].append(total_modules)
            reward_extra_info["num_turns"].append(num_turns)
            reward_extra_info["score"].append(reward)
            
            # Collect reward breakdown records
            batch_records.append({
                "timestamp": datetime.now().isoformat(),
                "batch": self.batch_count,
                "is_validate": data.meta_info.get("validate", False),  # Distinguish train/val
                "sample_index": i,
                "sample_id": extra_info.get("sample_id"),
                "batch_index": extra_info.get("batch_index"),  # Corresponds to Master ID
                "total_reward": reward,
                # New reward function fields
                "improvement_count": improvement_count,
                "cumulative_module_count_reward": float(cumulative_module_count_reward),
                "judge_score": float(judge_score),  # Score for entire reasoning trajectory
                "metric_improvement_reward": float(metric_improvement_reward),
                # Legacy fields (for comparison)
                "module_count_reward": float(module_count_reward),
                # Other fields
                "module_rewards": extra_info.get("module_rewards"),  # Per-module metric improvement reward details
                "best_simulation_turn": extra_info.get("best_simulation_turn"),
                "improved_modules_count": extra_info.get("improved_modules_count", 0),
                "total_modules": total_modules,
                "turn_count": extra_info.get("turn_count", num_turns),
                "baseline_metrics": extra_info.get("baseline_metrics"),  # Initial baseline per-module metrics
                "best_metrics": extra_info.get("best_metrics"),  # Final best per-module metrics
            })

            # Get valid response length to place reward at the last token
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            # Place reward at the last valid response token
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            # Debug output
            if self.examined_count < self.num_examine:
                self.examined_count += 1
                
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch["responses"]
                valid_response_ids = response_ids[:valid_response_length]
                
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                print("=" * 60)
                print("[DeepCity Reward Debug]")
                print(f"[cumulative_reward] {cumulative_reward:.2f}")
                print(f"[judge_score] {judge_score:.2f}")
                print(f"[total_modules] {total_modules}")
                print(f"[num_turns] {num_turns}")
                print(f"[reward] {reward:.2f}")
                print(f"[prompt] {prompt_str[:500]}...")
                print(f"[response] {response_str[:500]}...")
                print("=" * 60)

        # Stream write reward breakdown log (append per batch)
        self._write_reward_log(batch_records)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
    
    def _write_reward_log(self, records: list):
        """
        Stream append write reward breakdown log (JSONL format).
        Called once per batch, one line per sample.
        """
        if not records:
            return
        try:
            with open(self.reward_log_path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(records)} reward records to {self.reward_log_path} (batch {self.batch_count})")
        except Exception as e:
            logger.warning(f"Failed to write reward log: {e}")
