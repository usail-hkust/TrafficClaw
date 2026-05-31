# TrafficClaw

```
 ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗
 ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝
    ██║   ██████╔╝███████║█████╗  █████╗  ██║██║     
    ██║   ██╔══██╗██╔══██║██╔══╝  ██╔══╝  ██║██║     
    ██║   ██║  ██║██║  ██║██║     ██║     ██║╚██████╗
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ ╚═════╝
     
   ██████╗██╗      █████╗ ██╗    ██╗
  ██╔════╝██║     ██╔══██╗██║    ██║
  ██║     ██║     ███████║██║ █╗ ██║
  ██║     ██║     ██╔══██║██║███╗██║
  ╚██████╗███████╗██║  ██║╚███╔███╔╝
   ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
```

TrafficClaw is a generalizable LLM agent for unified urban traffic control in a physically grounded SUMO environment. It moves beyond isolated traffic-control tasks by exposing coupled dynamics among traffic signals, freeways, ramps, public transit, and taxi services, then uses executable spatiotemporal reasoning, persistent memory, and agentic reinforcement learning to optimize system-level mobility.

This repository contains the simulation environment, control modules, LLM-agent runtime, and VeRL-based agentic RL integration used in our paper:

> TrafficClaw: A Generalizable LLM Agent in the Unified Physical Environment for Urban Traffic Control.

## Highlights

- Unified physical traffic environment over shared road, lane, station, zone, and mobility-demand states.
- Six control tasks: traffic signal control, highway speed limit control, ramp metering, bus scheduling, subway scheduling, and taxi dispatching.
- Executable spatiotemporal analytics where the agent writes and runs code to model spatial, temporal, and cross-subsystem traffic patterns.
- Feedback-driven policy planning with SUMO rollouts after generated control configurations.
- Episodic Spatiotemporal Context Cache (ESCC) for within-episode analytical reuse.
- Procedural Spatiotemporal Memory (PSM) for cross-episode knowledge about congestion patterns, coordination strategies, and failure modes.
- Multi-stage training: supervised multi-task cold start followed by system-aligned agentic RL with GRPO.
- Interactive TUI for launching and steering runs from the terminal, similar in spirit to OpenClaw's conversational agent workflow.
- Evaluation support for single-module control, joint cross-subsystem control, and VeRL-based training.

## Framework

TrafficClaw is organized around three components.

1. Unified Traffic Environment

   The environment integrates heterogeneous traffic subsystems into a shared physical network. Subsystem actions can affect common infrastructure and mobility demand, so local interventions can propagate to network-wide performance.

2. Generalizable Traffic Control Agent

   The agent follows an analysis-action-feedback loop. It queries traffic-state APIs, generates executable code for analytics, saves reusable intermediate results, writes control configurations, observes rollout feedback, and refines decisions over multiple turns.

3. System-Level Training

   TrafficClaw is first initialized with diverse multi-task trajectories, then optimized with agentic RL under rewards that combine system efficiency and cross-subsystem coordination quality.

```text
Traffic observations + task query
        |
        v
Executable spatiotemporal analytics
  - spatial topology and local congestion
  - temporal trends and peak/off-peak regimes
  - cross-subsystem dependencies
        |
        v
Feedback-grounded policy planning
        |
        v
SUMO rollout and metric feedback
        |
        v
ESCC within episode + PSM across episodes
```

## Supported Control Tasks

| Task | Module | Main action | Typical metrics |
| --- | --- | --- | --- |
| Traffic signal control | `signal_timing` | Adjust signal cycle and phase durations | Throughput, waiting time, travel time |
| Highway speed limit control | `highway_speed_limit` | Adjust freeway speed limits | Travel time, speed |
| Ramp metering | `ramp_metering` | Adjust ramp open/green duration | Travel time, ramp queue length |
| Bus scheduling | `bus_scheduling` | Adjust headway, departure, and dwell-time schedules | Passenger waiting time, fuel consumption |
| Subway scheduling | `subway_scheduling` | Adjust train headway, departure, and dwell-time schedules | Passenger waiting time, electricity usage |
| Taxi dispatching | `taxi_scheduling` | Dispatch and reposition idle taxis | Income, completed trips, utilization |

## Agent Interfaces

The paper environment exposes 98 dynamic traffic-state features, 20 static data resources, and 15 interaction APIs. In this repository, the agent tool definitions live in `agent_tools/`.

Dynamic state APIs include:

- `read_lane_traffic_states`
- `read_highway_traffic_states`
- `read_ramp_lane_traffic_states`
- `read_bus_states`
- `read_subway_states`
- `read_taxi_traffic_states`
- `analyze_zone_traffic`

Planning and helper APIs include:

- `identify_congestion_hotspots`
- `calculate_network_metrics`
- `predict_arima`
- `dispatch_taxi`
- `reposition_taxi`
- `rank_idle_taxis_by_distance`
- `get_zone_infrastructure`
- `get_zones_by_infrastructure`

Static resources include network graphs, infrastructure dictionaries, current control configurations, transit route information, taxi fleet state, pending reservations, and TAZ-level demand/supply statistics.

## Repository Layout

```text
TrafficClaw/
|-- agent_tools/                  # Tool/data schemas exposed to the LLM agent
|-- control_config/               # Generated or default control configurations
|-- control_modules/              # Control modules and registry
|   |-- signal_timing.py
|   |-- highway_speed_limit.py
|   |-- ramp_metering.py
|   |-- bus_scheduling.py
|   |-- subway_scheduling.py
|   `-- taxi_scheduling.py
|-- environment/                  # SUMO environment and physical entities
|-- run_single_control/           # Single-subsystem evaluation runners
|-- run_joint_control/            # Cross-subsystem joint-control runner
|-- tools/                        # Network, OD, and public-transit utilities
|-- utils/                        # LLM agent, sandbox, simulation, logging, prompts
|-- verl/                         # VeRL integration for agentic RL training
|-- trafficclaw_runner.py         # Callable unified simulation entry point
`-- trafficclaw_tui.py            # Interactive terminal UI for launching runs
```

## Installation

### 1. Create a Python environment

```bash
conda create -n trafficclaw python=3.10 -y
conda activate trafficclaw
pip install -r requirements.txt
```

### 2. Install SUMO

TrafficClaw is tested with SUMO 1.20.0. The `requirements.txt` pins `traci==1.20.0` and `sumolib==1.20.0`, and the SUMO binaries (`sumo`, `duarouter`, etc.) must also be available on `PATH`.

Build SUMO 1.20.0 from the official GitHub repository. The SUMO documentation recommends cloning with `--recursive` so submodules are available.

Do not build SUMO inside an active Anaconda/conda environment. Open a separate shell or run `conda deactivate` before compiling, then return to the `trafficclaw` environment when running this project.

Ubuntu/Debian:

```bash
conda deactivate

mkdir -p ~/src
cd ~/src
git clone --recursive --branch v1_20_0 https://github.com/eclipse-sumo/sumo sumo-1.20.0
cd sumo-1.20.0
git fetch origin refs/replace/*:refs/replace/*

export SUMO_HOME="$PWD"
sudo apt-get update
sudo apt-get install -y $(cat build_config/build_req_deb.txt build_config/tools_req_deb.txt)

cmake -B build .
cmake --build build -j"$(nproc)"

export PATH="$SUMO_HOME/bin:$PATH"
```

macOS with Homebrew:

```bash
conda deactivate

xcode-select --install
brew update
brew install cmake
brew install --cask xquartz
brew install xerces-c fox proj gdal gl2ps

mkdir -p ~/src
cd ~/src
git clone --recursive --branch v1_20_0 https://github.com/eclipse-sumo/sumo sumo-1.20.0
cd sumo-1.20.0
git fetch origin refs/replace/*:refs/replace/*

export SUMO_HOME="$PWD"
cmake -B build .
cmake --build build --parallel "$(sysctl -n hw.ncpu)"

export PATH="$SUMO_HOME/bin:$PATH"
```

Persist the environment variables after the build:

```bash
echo 'export SUMO_HOME="$HOME/src/sumo-1.20.0"' >> ~/.zshrc
echo 'export PATH="$SUMO_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

If you use Bash, write the same two lines to `~/.bashrc` instead.

Verify the build:

```bash
sumo --version
duarouter --version
```

The version output should report `Eclipse SUMO sumo Version 1.20.0`.

```bash
python -c "import traci, sumolib; print('traci and sumolib import ok')"
```

### 3. Configure model access

The runners support OpenAI-compatible APIs and SiliconFlow-style model names. Set the credentials required by the model provider you use, for example:

```bash
export OPENAI_API_KEY=...
export SILICONFLOW_API_KEY=...
```

For custom OpenAI-compatible endpoints, pass `--base-url`.

## Environment Data

The SUMO environment data is hosted on Hugging Face:

```text
https://huggingface.co/datasets/TrafficClaw/TrafficClaw-Env-Data
```

The dataset is public and contains the project `Data/` directory. Its main contents are:

- `Data/sumo_config/`: regular SUMO scenarios for Brooklyn, Manhattan, and Queens.
- `Data/sumo_config_highway/`: highway-speed-limit scenarios for Brooklyn, Manhattan, and Queens.
- Each region contains SUMO config files, network XML, OSM/PBF source files, routes, OD matrices, TAZ files, public-transit stops/flows, taxi demand/fleet files, and cached static/runtime network metadata.

Download only the `Data/` tree into the TrafficClaw repository root:

```bash
cd /path/to/TrafficClaw
pip install -U "huggingface_hub[cli]"
hf download TrafficClaw/TrafficClaw-Env-Data \
  --repo-type dataset \
  --include "Data/**" \
  --local-dir .
```

## Quick Start

The recommended way to run TrafficClaw is the built-in terminal UI. You do not need to assemble long CLI commands for everyday experiments.

### TrafficClaw TUI (OpenClaw-style interactive mode)

TrafficClaw can run like [OpenClaw](https://github.com/openclaw/openclaw) in **TUI mode**: one terminal session to configure the agent, start a long-horizon SUMO run, watch checkpoint progress, and send new instructions while the simulation is still running. Instead of editing shell scripts between checkpoints, you steer the traffic-control agent interactively from the same prompt.

**Launch:**

```bash
python trafficclaw_tui.py
```

**What you get:**

- A guided setup flow (checkbox menus when your terminal supports `curses`, with a numbered fallback otherwise).
- Single-module or joint multi-module control without switching scripts.
- A cleaned live dashboard: progress bar, checkpoint metrics, and compact LLM turn previews (noisy SUMO/TraCI logs are filtered out).
- **Runtime queries**: type instructions at any time and press Enter; they are applied at the **next checkpoint**, like chatting with an agent between tool rounds.

**Setup wizard (before the run starts):**

| Step | What you choose |
| --- | --- |
| Modules | One or more of the six control modules (`signal_timing`, `highway_speed_limit`, `ramp_metering`, `bus_scheduling`, `subway_scheduling`, `taxi_scheduling`). Multiple modules automatically use the joint runner. |
| Query | Optional natural-language goal (e.g., reduce bus waiting time during morning peak). Leave blank to use the runner's default task prompt. |
| Duration / checkpoint interval | Simulation length and how often the LLM re-optimizes (defaults match `trafficclaw_runner.py` profiles). |
| LLM model | Provider-prefixed name, e.g. `siliconflow/deepseek-ai/DeepSeek-V4-Flash`. |
| API key | Prompted securely if the provider env var is missing (session-only; export it beforehand for non-interactive use). |
| Zone | Queens (`queens` in the TUI). `sumo_config/Queens/`. |
| Wandb | Enable or disable remote metric logging. |

After the summary screen, confirm to start. The TUI delegates to the same runners as the CLI (`run_single_control/*` or `run_joint_control/run_joint_control.py`), so results are identical to scripted runs.

**During a run:**

1. **Progress** - A bar shows simulated time vs. total duration.
2. **Checkpoints** - At each interval the agent reads traffic state, plans, runs a rollout, and prints module/travel/waiting metrics.
3. **Runtime query line** - Stays active in the background:

   ```text
   User query for next checkpoint (leave blank to execute the current request):
   ```

   Type a new instruction and press Enter anytime; it is queued for the next optimization checkpoint. Examples:

   - `Prioritize reducing passenger waiting time on bus routes with avg wait > 300s.`
   - `Keep signal changes conservative; do not increase cycle times above 90s.`
   - `Coordinate signal timing before bus headway changes.`

4. **Exit** - Press `Ctrl+C` to stop the session.

**Tips:**

- Export API keys before launching for unattended servers, e.g. `export SILICONFLOW_API_KEY=...` or `export OPENAI_API_KEY=...`.
- Use a real terminal (not a non-TTY log pipe) so runtime query input works; the TUI falls back to `/dev/tty` when needed.
- For reproducible batch jobs, use the CLI runners below or `trafficclaw_runner.py`; the TUI is optimized for interactive exploration and demos.

## GRPO Training with VeRL

TrafficClaw includes a VeRL integration for agentic GRPO training. The training flow uses:

- `verl/deepcity_test/create_deepcity_dataset.py` to create trigger datasets.
- `verl/deepcity_test/deepcity_interaction_config.yaml` to configure SUMO environments, control modules, rewards, and validation.
- `verl/deepcity_test/train_hpc.sh` as the reference GRPO launch script.
- `verl/verl/experimental/agent_loop/deepcity_agent_loop.py` and `verl/verl/workers/reward_manager/deepcity.py` for multi-turn interaction and reward computation.

### 1. Install VeRL training dependencies

Use a CUDA machine for training. The example below installs the local VeRL copy with vLLM support:

```bash
conda create -n trafficclaw-verl python=3.10 -y
conda activate trafficclaw-verl

pip install -r requirements.txt
cd verl
pip install -e ".[vllm]"
cd ..
```

Make sure SUMO 1.20.0 is built and exported as described in the installation section:

```bash
export SUMO_HOME="$HOME/src/sumo-1.20.0"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PWD/verl:$PYTHONPATH"
```

### 2. Create the DeepCity trigger dataset

The dataset only triggers the DeepCity agent loop. The live SUMO checkpoint context is generated by `DeepCityMaster` during rollout.

```bash
cd verl/deepcity_test
python create_deepcity_dataset.py \
  --train 450 \
  --val-envs 6 \
  --val-steps 6 \
  --output-dir data
cd ../..
```

This creates:

```text
verl/deepcity_test/data/deepcity_train.parquet
verl/deepcity_test/data/deepcity_val.parquet
```

### 3. Configure training environments

Edit `verl/deepcity_test/deepcity_interaction_config.yaml` before launching:

- Replace every `sumo_config_path` with paths that exist on your machine.
- Set `judge_llm.api_key`, or set `judge_llm.enabled: false` for a no-judge smoke test.
- Keep `num_masters`, `num_val_masters`, `data.train_batch_size`, and `actor_rollout_ref.rollout.agent.num_workers` consistent with your GPU and CPU capacity. Each rollout can start SUMO simulations.

For a small local smoke test, reduce the config first:

```yaml
interaction:
  - name: "deepcity"
    config:
      num_masters: 1
      num_val_masters: 1
      max_concurrent_simulations: 1
      simulation_duration: 3600
      val_simulation_duration: 1800
```

Then regenerate a tiny dataset:

```bash
cd verl/deepcity_test
python create_deepcity_dataset.py --train 4 --val-envs 1 --val-steps 1 --output-dir data
cd ../..
```

### 4. Set model and runtime paths

Open `verl/deepcity_test/train_hpc.sh` and update these values:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SUMO_HOME="$HOME/src/sumo-1.20.0"
MODEL_PATH="/path/to/base-or-sft-model"
EXPERIMENT_NAME="trafficclaw_grpo"
```

The script already enables GRPO with:

```bash
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=8
reward_model.reward_manager=deepcity
+actor_rollout_ref.rollout.multi_turn.agent_loop_class=deepcity_agent
```

### 5. Launch GRPO training

Run the reference script:

```bash
bash verl/deepcity_test/train_hpc.sh
```

For a one-GPU smoke test, pass smaller overrides at the end of the command:

```bash
bash verl/deepcity_test/train_hpc.sh \
  data.train_batch_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  trainer.n_gpus_per_node=1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1
```

Checkpoints and logs are written under:

```text
checkpoints/deepcity/
verl/deepcity_test/
```

TensorBoard logs are enabled by the script:

```bash
tensorboard --logdir checkpoints/deepcity
```

## Evaluation Setup

The paper evaluates TrafficClaw on Manhattan, Queens, and Brooklyn with SUMO-based 24-hour simulations. Manhattan and Queens are used for training/evaluation settings, while Brooklyn is reserved for cross-region transfer.

Training tasks include:

- Signal control
- Highway speed limit control
- Bus scheduling
- Taxi dispatching
- Cooperative settings among these modules

Held-out evaluation tasks include:

- Ramp metering
- Subway scheduling
- Associated cooperative scenarios

Baselines include classic methods, RL-based controllers, traffic LLM agents, general coding agents, and generalist LLMs. See the paper for the full metric table and experimental protocol.

## Outputs and Logs

Runs generate checkpoints, traffic states, control configurations, and metric logs. Common output locations include:

- `records/` for collected traffic states and run artifacts.
- `control_config/` for generated control configurations.
- Weights & Biases logging when `--wandb` is enabled.

Most runners expose common options:

- `--config`: SUMO scenario file.
- `--duration`: simulation duration in seconds.
- `--checkpoint-interval`: interval between optimization checkpoints.
- `--step-seconds`: base simulation step size.
- `--llm-model`: model provider/name.
- `--base-url`: optional OpenAI-compatible API endpoint.
- `--temperature`: decoding temperature.
- `--max-agent-turns`: maximum agent turns per checkpoint.
- `--traffic-state-interval`: data collection interval.
- `--query`: extra user instruction appended to the task prompt.

## License

The project code and post-trained TrafficClaw model are released under the MIT License. Third-party models, datasets, simulators, and baseline systems remain governed by their respective licenses and terms of use.
