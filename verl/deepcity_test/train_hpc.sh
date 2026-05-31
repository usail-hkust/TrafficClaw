#!/bin/bash
# DeepCity Subway Scheduling Training with test_6lines.sumocfg
# Using: verl AgentLoop + DeepCityInteraction + DeepCityMaster

set -x

# Get project directories first
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEEPCITY_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# GPU Configuration - Use GPU 1,2 (dual GPU training)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add verl source code to PYTHONPATH (to use modified code without reinstalling)
export PYTHONPATH="$DEEPCITY_ROOT/verl:$PYTHONPATH"

# SUMO Configuration - Use compiled SUMO from source
# SUMO is compiled in: /home/apulis-dev/userdata/env/sumo-1_20_0
export SUMO_HOME="/home/apulis-dev/userdata/env/sumo-1_20_0"
export PATH="$SUMO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$SUMO_HOME/build/src:${LD_LIBRARY_PATH:-}"
# Add SUMO Python tools to PYTHONPATH
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Set all cache and temp directories
export RAY_TMPDIR=/home/apulis-dev/tmp/ray
export HF_HOME=/home/apulis-dev/.cache/huggingface
export HF_DATASETS_CACHE=/home/apulis-dev/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/apulis-dev/.cache/huggingface/transformers
export TMPDIR=/home/apulis-dev/tmp
export TEMP=/home/apulis-dev/tmp
export TMP=/home/apulis-dev/tmp

# Disable uvloop to avoid asyncio event loop compatibility issues
export SGLANG_USE_UVLOOP=0

# Set vLLM to use V1 engine for async mode
export VLLM_USE_V1=1

# Set Triton cache
export TRITON_CACHE_DIR=/home/apulis-dev/.cache/triton

# Unset proxy for wandb
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# Create necessary directories
mkdir -p "$DEEPCITY_ROOT/checkpoints/trafficclaw"
mkdir -p $RAY_TMPDIR $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $TMPDIR $TRITON_CACHE_DIR

# Increase file descriptor limit
ulimit -n 65535

# ============================================
# DeepCity Configuration
# ============================================
# Model path
MODEL_PATH="/home/apulis-dev/userdata/models/qwen3-8b-trafficclaw"

function now() {
    date '+%Y%m%d-%H%M%S'
}

EXPERIMENT_NAME="deepcity_20260413"

# ============================================
# Start Training
# ============================================
python3 -m verl.trainer.main_ppo \
    --config-path="$DEEPCITY_ROOT/verl/verl/trainer/config" \
    --config-name='ppo_trainer' \
    +deepcity_config_path="$PROJECT_DIR/deepcity_interaction_config.yaml" \
    algorithm.adv_estimator=grpo \
    +algorithm.reward_std_filter_threshold=0.1 \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=26624 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=30720 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=11 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=12288 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
    +actor_rollout_ref.rollout.multi_turn.agent_loop_class=deepcity_agent \
    +actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/deepcity_interaction_config.yaml" \
    reward_model.reward_manager=deepcity \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='deepcity' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    data.train_files="$PROJECT_DIR/data/deepcity_train.parquet" \
    data.val_files="$PROJECT_DIR/data/deepcity_val.parquet" \
    data.val_batch_size=6 \
    data.validation_shuffle=False \
    $@