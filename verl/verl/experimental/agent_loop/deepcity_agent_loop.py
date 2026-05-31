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
DeepCity Agent Loop for verl framework.

This module implements the agent loop for DeepCity transportation optimization.
It manages the 10-turn dialogue between LLM and SUMO environment:
- LLM generates actions (PLAN/DATA_ANALYSIS/POLICY_PLANNING/FINISH)
- Environment executes actions and returns feedback
- Reward = (metric_improvement + judge_score) × improvement_count + cumulative_module_count_reward
  - metric_improvement: best result vs baseline (mean of all modules' key metric improvements)
  - judge_score: LLM Judge evaluates the entire reasoning trajectory at FINISH
  - improvement_count: number of turns where improved_modules is non-empty
  - cumulative_module_count_reward: sum of (num_improved_modules / total_modules) across improvement turns
"""

import asyncio
import logging
import os
import json
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("deepcity_agent")
class DeepCityAgentLoop(AgentLoopBase):
    """
    DeepCity Agent Loop for transportation optimization.
    
    Architecture (see verl_architecture.md):
    - 2 Masters run in parallel (different scenarios/seeds)
    - Each Master triggers 8 worker samplings when reaching checkpoint
    - 8 workers each do 10 rounds of interaction, results are compared within GRPO group
    
    Dialogue Flow (flexible combination, up to 10 turns):
    - Turn 1: PLAN (required) - LLM generates optimization plan
    - Turn 2: GET_CONTROL_API (required) - LLM requests control module API
    - Turn 3-9: Flexible combination of the following actions:
      * DATA_ANALYSIS - Analyze traffic data, cache results
      * POLICY_PLANNING - Generate policy config, trigger simulation
      * DEBUG - Fix code errors
    - Turn ≤10: FINISH - LLM ends optimization
    
    Note: Each POLICY_PLANNING automatically triggers simulation, not counted as extra turn
    """
    
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """Class-level initialization, called once."""
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level DeepCityAgentLoop initialization")
        
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        
        # DeepCity specific config
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        
        # Initialize DeepCityInteraction (supports both Ray and HTTP modes)
        interaction_config_path = config.actor_rollout_ref.rollout.multi_turn.get("interaction_config_path", None)
        cls.interaction = None
        cls.use_http = False  # Track which mode is being used
        cls.judge_llm_config = {}  #   Default empty config, prevents AttributeError when uninitialized

        if interaction_config_path:
            try:
                import yaml
                with open(interaction_config_path, 'r') as f:
                    interaction_config = yaml.safe_load(f)

                # Find deepcity interaction config
                for item in interaction_config.get("interaction", []):
                    if item.get("name") == "deepcity":
                        item_config = item.get("config", {})

                        # Check if HTTP mode is enabled
                        use_http = item_config.get("use_http", False)
                        cls.use_http = use_http
                        
                        #   Save judge_llm config
                        cls.judge_llm_config = item_config.get("judge_llm", {})

                        #   Use DeepCityInteraction uniformly (creates Ray Master Actor)
                        # In HTTP mode, Master internally calls remote SUMO server via HTTP
                        # This ensures 4 Masters can run main simulation in parallel
                        from verl.interactions.deepcity_interaction import DeepCityInteraction
                        cls.interaction = DeepCityInteraction(item_config)
                        
                        if use_http:
                            print(f"Initialized DeepCityInteraction (HTTP mode) with config: {item_config}")
                            print(f"  SUMO Server URL: {item_config.get('sumo_server_url', 'http://localhost:8000')}")
                            print(f"  4 Ray Masters will call remote SUMO server in parallel")
                        else:
                            print(f"Initialized DeepCityInteraction (Local mode) with config: {item_config}")
                        break
            except Exception as e:
                logger.warning(f"Failed to initialize DeepCityInteraction: {e}")
                import traceback
                traceback.print_exc()

        if cls.interaction is None:
            logger.warning("DeepCityInteraction not initialized. Using fallback mode.")
        else:
            mode = "HTTP" if cls.use_http else "Ray"
            print(f"DeepCityInteraction initialized in {mode} mode with control modules: {getattr(cls.interaction, 'control_modules', 'N/A')}")
    
    @rollout_trace_op
    async def run(
        self, 
        messages: list[dict[str, Any]], 
        sampling_params: dict[str, Any],
        sample_id: str = None,  #   sample_id parameter
        batch_index: int = 0,  #   batch index, used for Master allocation
        is_validate: bool = False,  #   validation mode flag (passed from meta_info by agent_loop.py)
        is_padded: bool = False,  #   padded sample flag (skip SUMO interaction to avoid interfering with Master checkpoint_in_use)
    ) -> AgentLoopOutput:
        """
        Run 10-turn dialogue loop between LLM and SUMO environment.
        
        Args:
            messages: Initial messages (system prompt + task description)
            sampling_params: LLM sampling parameters
            sample_id: Sample ID for shared checkpoint
            
        Returns:
            AgentLoopOutput: Contains prompt_ids, response_ids, response_mask, metrics
        """
        # # === PyCharm remote debugging ===
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('10.7.151.20', port=12345, stdoutToServer=True, stderrToServer=True)
        # # === End debugging code ===

        metrics = {}
        request_id = uuid4().hex
        response_mask = []
        
        #   Padded sample: Skip SUMO interaction, return minimal output
        # Padded samples are duplicate data from padding to num_workers multiple, results will be discarded by unpad
        # Do not register them to Master (would interfere with checkpoint_in_use flag, causing next val round to fail step)
        if is_padded:
            print(f"[PAD_SKIP] batch_index={batch_index}, is_validate={is_validate} - skipping SUMO interaction for padded sample")
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True
                ),
            )
            # Return empty response (will be discarded after unpad)
            eos_id = self.tokenizer.eos_token_id
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[eos_id],
                response_mask=[1],
                num_turns=0,
                metrics=AgentLoopMetrics(),
                extra_info={"is_padded": True, "cumulative_reward": 0.0},
            )
        
        #   Initialize interaction instance
        instance_id = None
        if self.interaction:
            try:
                #   Use global batch_index to generate instance_id
                # batch_index is calculated by ray_trainer.py before repeat, ensuring each sample has correct intra-batch position
                # This way, regardless of num_workers, samples within the same batch are evenly distributed to different Masters
                # e.g.: sample_405 → batch_index=0 → Master_0
                #       sample_289 → batch_index=1 → Master_1
                import time
                import random
                worker_id = f"{int(time.time() * 1000000) % 1000000}_{random.randint(0, 9999)}"
                temp_instance_id = f"batch_{batch_index}_w{worker_id}"
                logger.info(f"Using global batch_index={batch_index} (sample_id='{sample_id}') for instance_id: {temp_instance_id}")
                
                print(f"[VAL_DEBUG] batch_index={batch_index}, is_validate={is_validate}, instance_id={temp_instance_id}, "
                      f"val_master_id={batch_index % 6 if is_validate else 'N/A'}")
                
                #   Start interaction (ignore passed messages, use fixed prompt)
                instance_id = await self.interaction.start_interaction(
                    instance_id=temp_instance_id,
                    initial_messages=messages,  # Passed but will be ignored
                    is_validate=is_validate,
                )
                print(f"[VAL_DEBUG] start_interaction SUCCESS: batch_index={batch_index}, instance_id={instance_id}")
                
                #   Get complete messages from interaction (including system prompt)
                messages = self.interaction.get_messages(instance_id)
                logger.info(f"Got {len(messages)} messages from interaction (including system prompt)")
                
            except Exception as e:
                print(f"[VAL_DEBUG] start_interaction FAILED: batch_index={batch_index}, is_validate={is_validate}, "
                      f"instance_id={temp_instance_id}, error={type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to start interaction: {e}", exc_info=True)
                instance_id = None
        
        # Tokenize initial messages
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )  
        
        user_turns, assistant_turns = 0, 0
        should_terminate = False
        cumulative_reward = 0.0
        metric_improvement_reward = 0.0
        module_count_reward = 0.0
        finish_extra_info = {}  #   Save FINISH details
        baseline_result = None  #   Initial baseline simulation result
        best_result = None  #   Final best simulation result
        
        try:
            while not should_terminate:
                # Check if we have space for more tokens before generating
                if len(response_mask) >= self.response_length:
                    break
                
                # 1. LLM generates response
                with simple_timer("generate_sequences", metrics):
                    response_ids = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                    )
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                assistant_turns += 1
                
                # Check termination conditions
                if len(response_mask) >= self.response_length:
                    break
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    break
                
                # 2. Environment processes LLM output
                if self.interaction and instance_id:
                    # Decode LLM response
                    llm_response = await self.loop.run_in_executor(
                        None,
                        lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
                    )
                    
                    #   Call interaction.generate_response (pass llm_response string)
                    with simple_timer("tool_calls", metrics):
                        should_terminate, env_feedback, turn_reward, turn_extra_info = await self.interaction.generate_response(
                            instance_id, llm_response  #   Pass string instead of messages
                        )
                    
                    # Update cumulative_reward from extra_info
                    if turn_extra_info:
                        cumulative_reward = turn_extra_info.get("cumulative_reward", cumulative_reward)
                        #   Save reward breakdown info at FINISH
                        if should_terminate:
                            metric_improvement_reward = turn_extra_info.get("metric_improvement_reward", 0.0)
                            module_count_reward = turn_extra_info.get("module_count_reward", 0.0)
                            finish_extra_info = turn_extra_info
                            #   Extract baseline and best module_metrics for logging
                            baseline_result = turn_extra_info.get("initial_best_simulation_result")
                            best_result = turn_extra_info.get("best_simulation_result")
                    
                    if should_terminate:
                        # Add final feedback if any
                        if env_feedback:
                            env_response_ids = await self._encode_env_response(env_feedback)
                            if len(response_mask) + len(env_response_ids) < self.response_length:
                                prompt_ids += env_response_ids
                                response_mask += [0] * len(env_response_ids)
                                user_turns += 1
                        break
                    
                    # 3. Append environment feedback to conversation
                    if env_feedback:
                        env_response_ids = await self._encode_env_response(env_feedback)
                        
                        # Check if adding feedback would exceed response length
                        if len(response_mask) + len(env_response_ids) >= self.response_length:
                            break
                        
                        prompt_ids += env_response_ids
                        response_mask += [0] * len(env_response_ids)
                        user_turns += 1
                        
                        #   Get updated messages from interaction
                        messages = self.interaction.get_messages(instance_id)
                else:
                    # Fallback: no interaction, just generate once
                    break
        except Exception as e:
            logger.error(f"Error in agent loop: {e}")
            import traceback
            traceback.print_exc()
        
        #   Step 1: For non-normal FINISH (timeout/token limit/exception), supplement finish_extra_info from interaction state
        # Must execute before Judge LLM and reward calculation to ensure control_modules etc. are available
        if not finish_extra_info and self.interaction and instance_id:
            try:
                state_summary = self.interaction.get_instance_state_summary(instance_id)
                if state_summary:
                    finish_extra_info = state_summary
                    metric_improvement_reward = state_summary.get("metric_improvement_reward", 0.0)
                    module_count_reward = state_summary.get("module_count_reward", 0.0)
                    baseline_result = {"module_metrics": state_summary.get("baseline_metrics")} if state_summary.get("baseline_metrics") else None
                    best_result = {"module_metrics": state_summary.get("best_metrics")} if state_summary.get("best_metrics") else None
                    logger.info(
                        f"Non-normal FINISH fallback: metric_improvement={metric_improvement_reward:.3f}, "
                        f"improvement_count={state_summary.get('improvement_count', 0)}, "
                        f"cumulative_module_count_reward={state_summary.get('cumulative_module_count_reward', 0.0):.3f}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get instance state summary: {e}")
        
        #   Step 2: Judge LLM scoring (score entire reasoning trajectory, called regardless of normal FINISH)
        judge_score = 0.0
        if self.judge_llm_config.get("enabled", False) and messages:
            try:
                control_modules = finish_extra_info.get("control_modules", [])
                judge_score = await self._call_judge_llm(messages, self.judge_llm_config, control_modules)
                logger.info(f"Judge LLM score: {judge_score:.3f}")
            except Exception as e:
                logger.warning(f"Failed to call Judge LLM: {e}")
                judge_score = 0.0
        
        #   Step 3: Calculate final reward uniformly
        # total_reward = (metric_improvement + judge_score) × improvement_count + cumulative_module_count_reward
        improvement_count = finish_extra_info.get("improvement_count", 0)
        cumulative_module_count_reward = finish_extra_info.get("cumulative_module_count_reward", 0.0)
        
        cumulative_reward = (metric_improvement_reward + judge_score) * improvement_count + cumulative_module_count_reward
        
        logger.info(
            f"Final reward: ({metric_improvement_reward:.3f} + {judge_score:.3f}) × {improvement_count} + {cumulative_module_count_reward:.3f} = {cumulative_reward:.3f}"
        )
        
        # Finalize interaction (cleanup after Judge LLM)
        if self.interaction and instance_id:
            try:
                await self.interaction.finalize_interaction(instance_id)
            except Exception as e:
                logger.warning(f"Failed to finalize interaction: {e}")
        
        # Prepare output
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=AgentLoopMetrics(**metrics),
            extra_info={
                "cumulative_reward": cumulative_reward,  # Final reward from new reward function
                "judge_score": judge_score,  # LLM score for entire reasoning trajectory
                "metric_improvement_reward": metric_improvement_reward,  # Metric improvement reward (best vs baseline)
                "improvement_count": improvement_count,  # Number of improvements
                "cumulative_module_count_reward": cumulative_module_count_reward,  # Cumulative module improvement reward
                "module_rewards": finish_extra_info.get("module_rewards"),  # Per-module metric improvement reward details
                "module_count_reward": module_count_reward,  # Legacy module count reward (for comparison)
                "batch_index": batch_index,  # Batch index (corresponds to Master)
                "sample_id": sample_id,  # Sample ID
                "best_simulation_turn": finish_extra_info.get("best_simulation_turn"),  # Best turn
                "improved_modules_count": finish_extra_info.get("improved_modules_count", 0),  # Number of improved modules
                "total_modules": finish_extra_info.get("total_modules", 0),  # Total modules
                "turn_count": finish_extra_info.get("turn_count", user_turns + assistant_turns),  # Total turns
                "baseline_metrics": baseline_result.get("module_metrics") if baseline_result else None,  # Baseline per-module metrics
                "best_metrics": best_result.get("module_metrics") if best_result else None,  # Final best per-module metrics
            },
        )
        return output
    
    async def _encode_env_response(self, content: str) -> list[int]:
        """Encode environment response to token ids."""
        # Truncate if too long
        if len(content) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                content = content[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                content = "(truncated)..." + content[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                content = content[:length] + "...(truncated)..." + content[-length:]
        
        # Encode as user message
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True
            ),
        )
        return response_ids
    
    async def _call_judge_llm(self, messages: list[dict[str, Any]], judge_config: dict, control_modules: list[str] = None) -> float:
        """
        Call Judge LLM to score the overall dialogue quality.
        
        Args:
            messages: Complete dialogue history
            judge_config: Judge LLM configuration
            control_modules: Current environment's control module list
            
        Returns:
            Score (0.0 - 1.0), returns 0.0 on failure
        """
        try:
            import aiohttp
            
            # Build scoring prompt
            conversation_text = self._format_conversation_for_judge(messages)
            
            # Dynamically build scoring criteria based on module count
            modules = control_modules or []
            module_names_str = ", ".join(modules) if modules else "unknown modules"
            is_single_module = len(modules) == 1
            
            if is_single_module:
                # Single module: coordination quality auto full score 5, only evaluate Modeling Effectiveness
                coordination_criteria = f"""1. **Multi-module Coordination Quality (5/5 points - Auto Full Score)**:
   - This environment only has a single control module: **{module_names_str}**.
   - Since there is no multi-module coordination needed, this dimension is automatically scored 5/5.
   - You do NOT need to evaluate this dimension."""
            else:
                # Multi-module: normal scoring
                coordination_criteria = f"""1. **Multi-module Coordination Quality (0-5 points)**:
   - This environment has the following control modules: **{module_names_str}**.
   - Whether these modules are reasonably coordinated with each other
   - Whether the interactions and dependencies between modules are considered
   - Whether conflicts between modules are identified and avoided"""
            
            judge_prompt = f"""You are a professional traffic optimization expert. Please evaluate the following conversation between an LLM Agent and a traffic simulation environment. Score from 0-10 points.

Scoring Criteria (two dimensions, 5 points each, total 10 points):
{coordination_criteria}

2. **Modeling Effectiveness (0-5 points)**:
   - Whether the policy code correctly implements the optimization approach for {module_names_str}
   - Whether appropriate algorithms and parameters are used
   - Whether the data provided by the environment is fully utilized
   - Whether the agent iteratively improves the policy based on simulation feedback

Please read the conversation carefully and provide your evaluation in the following STRICT format:
```
Score: [0-10 integer]
Brief Comment: [Your brief comment in 1-2 sentences]
```

Example output:
```
Score: 7
Brief Comment: The agent demonstrates good optimization strategy with reasonable parameter tuning, but could better utilize the simulation feedback for iterative improvement.
```

Conversation:
{conversation_text}

Your Evaluation:"""

            # Call SiliconFlow API
            # Select API address based on use_proxy config
            use_proxy = judge_config.get("use_proxy", True)
            if use_proxy:
                api_base = judge_config.get("api_base_proxy", "https://localhost:8080/v1")
            else:
                api_base = judge_config.get("api_base_direct", "https://api.siliconflow.cn/v1")
            
            api_key = judge_config.get("api_key", "")
            model = judge_config.get("model", "deepseek-ai/DeepSeek-V3")
            temperature = judge_config.get("temperature", 0.0)
            max_tokens = judge_config.get("max_tokens", 500)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Only disable SSL verification when using reverse proxy (localhost certificate issue)
            if use_proxy:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                connector = aiohttp.TCPConnector(ssl=ssl_context)
            else:
                # Use default SSL verification for direct connection
                connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        score_text = result["choices"][0]["message"]["content"].strip()
                        
                        # Extract numeric score (supports "Score: X" format)
                        match = re.search(r'Score:\s*(\d+)', score_text, re.IGNORECASE)
                        if not match:
                            # If "Score: X" format not found, try extracting first number
                            match = re.search(r'\d+', score_text)
                        
                        if match:
                            score = int(match.group(1) if match.lastindex else match.group())
                            score = max(0, min(10, score))  # Clamp to 0-10 range
                            normalized_score = score / 10.0  # Normalize to 0-1
                            
                            # Extract comment (if any)
                            comment_match = re.search(r'Brief Comment:\s*(.+?)(?:\n|$)', score_text, re.IGNORECASE | re.DOTALL)
                            comment = comment_match.group(1).strip() if comment_match else "N/A"
                            
                            logger.info(f"Judge LLM score: {score}/10 (normalized: {normalized_score:.2f})")
                            logger.info(f"Judge comment: {comment}")
                            return normalized_score
                        else:
                            logger.warning(f"Failed to parse Judge LLM score from: {score_text}")
                            return 0.0
                    else:
                        error_text = await response.text()
                        logger.warning(f"Judge LLM API error {response.status}: {error_text}")
                        return 0.0
                        
        except Exception as e:
            logger.warning(f"Failed to call Judge LLM: {e}")
            return 0.0
    
    def _format_conversation_for_judge(self, messages: list[dict[str, Any]]) -> str:
        """
        Format dialogue history into text suitable for Judge LLM evaluation.
        
        Args:
            messages: Dialogue history
            
        Returns:
            Formatted dialogue text
        """
        formatted_lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Do not truncate content, preserve full dialogue
            formatted_lines.append(f"[{role.upper()}]:\n{content}\n")
        
        return "\n".join(formatted_lines)
