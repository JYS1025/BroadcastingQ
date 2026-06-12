# Structural Broadcasting RL

A clean Python reinforcement learning research repository for testing tabular and neural RL algorithms on plug-in applications. The future target method is ANOVA Structural Broadcasting Q-learning, but ANOVA-Q is intentionally left as a TODO placeholder for now.

## Installation

```bash
pip install -e .
```

DQN uses `device: auto` by default, which selects CUDA when PyTorch reports CUDA is available and otherwise falls back to CPU. Set `agent.device` to `cpu`, `cuda`, or another explicit PyTorch device string to override this.

## Run KeyDoor Experiments

```bash
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
pytest tests
```

## Run Additional Applications

```bash
python main.py --config applications/dna_promoter/config_qlearning.yaml
python main.py --config applications/dna_promoter/config_sarsa.yaml
python main.py --config applications/dna_promoter/config_dqn.yaml
python main.py --config applications/lights_out/config_qlearning_3x3.yaml
python main.py --config applications/lights_out/config_qlearning_4x4.yaml
python main.py --config applications/lights_out/config_qlearning_5x5.yaml
python main.py --config applications/lights_out/config_sarsa_3x3.yaml
python main.py --config applications/lights_out/config_dqn_4x4.yaml
python main.py --config applications/lights_out/config_dqn_5x5.yaml
python main.py --config applications/network_routing/config_qlearning.yaml
python main.py --config applications/network_routing/config_dqn.yaml
python main.py --config applications/network_routing/config_sbq.yaml
```

## Applications

Applications live under `applications/`. Each application owns its environment wrapper, complete experiment configs, visualization utilities, and application README. There is no global `configs/` directory.

To add an application:

1. Create `applications/<name>/`.
2. Implement an environment wrapper exposing `reset`, `step`, `render`, `observation_space`, and `action_space`.
3. Return observations as `MultiDiscreteSpace` vectors.
4. Add complete application-local YAML configs.
5. Add a visualizer that saves artifacts into the run output directory.

## KeyDoor

The KeyDoor application wraps MiniGrid DoorKey through `gymnasium` and `minigrid`. The repository does not manually implement movement, pickup, toggle/open, rewards, termination, truncation, or rendering.

MiniGrid internals are converted into a symbolic `MultiDiscreteSpace` observation. The minimal factors are:

```text
[agent_row, agent_col, agent_direction, has_key, door_state]
```

The provided configs also include key and door positions. This avoids severe state aliasing in tabular agents while keeping the Q-table much smaller than including every object position.

Rendered GIFs and PNGs are generated from MiniGrid RGB frames.

## DNA Promoter

The DNA promoter application searches over 6-base sequences using point mutations. Observations are integer-encoded bases in `MultiDiscreteSpace([4, 4, 4, 4, 4, 4])`, and the 18 actions mutate one position to a different base. The default sparse-reward targets are `TATAAT`, `TATAAA`, `TATATT`, and `TAAAAT`.

## Lights Out

The Lights Out application implements the original toggle rule: pressing a cell flips that cell and its vertical/horizontal neighbors. The default environment is 5x5 with solvable scrambled starts and slight action-slip stochasticity. The 4x4 configs are the main tabular/DQN sanity baselines; 3x3 is kept as a fast smoke-test config.

## Network Routing

The Network Routing application simulates packet routing across various complex network topologies (Scale-Free, Multi-Hub, Grid). The agent must navigate to a destination node while avoiding congested queues. The state is extremely compressed into a single 32-bit integer (Destination ID + 3-bit Thermometer Encoding of neighbor queue states). It features dynamic stateful queues with realistic background traffic noise and differential delay penalties, providing a highly challenging bottleneck routing environment.

## Agents

- `RandomAgent`
- `QLearningAgent`
- `SarsaAgent`
- `DQNAgent`
- `AnovaQAgent`, intentionally TODO

Tabular Q-learning and SARSA use sparse `defaultdict`-backed Q-tables, so large factored state spaces do not allocate the full state-action table up front.

## Config Format

Each config contains the complete experiment definition: application, environment, observation, agent, training, visualization, and logging settings.

```yaml
application:
  entrypoint: applications.key_door.env:KeyDoorEnv
  visualizer: applications.key_door.visualize:KeyDoorVisualizer
agent:
  name: q_learning
training:
  total_steps: 200000
logging:
  output_root: outputs
  run_name: keydoor_qlearning_seed0
```

## Outputs

Runs write to:

```text
outputs/<run_name>/
├── config.yaml
├── metrics.csv
├── eval_metrics.csv
├── checkpoints/
└── visualizations/
```
