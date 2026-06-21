# Structural Broadcasting Q-learning: Annealed Kernel Generalization for Tabular Reinforcement Learning

**Authors:** Yoonseong Jeong, Seungjun Kim, Jinwoong Park (School of Computing, KAIST) 

## Overview

This repository contains the official reinforcement learning research framework and application environments for the paper *Structural Broadcasting Q-learning: Annealed Kernel Generalization for Tabular Reinforcement Learning*.

We propose Structural Broadcasting Q-learning (SBQ), a tabular reinforcement learning method that improves sample efficiency by propagating temporal-difference errors across structurally similar states. Standard Q-learning updates only the visited state-action pairs, which is inefficient in large factored state spaces. Our method maintains a global Q-table and an auxiliary kernel estimator that broadcasts TD errors to nearby states according to a similarity kernel. The auxiliary contribution is annealed by visitation count, allowing early generalization while recovering standard Q-learning in the limit.

### Core Formulation

To simultaneously achieve rapid generalization and asymptotic convergence, we utilize a dual-table architecture governed by a combined Q-function:

$$\tilde{Q}_t(s,a)=Q_t^G(s,a)+\beta_t(s,a)Q_t^K(s,a)$$



The spatial similarity between two states in a finite metric space is defined using an exponential kernel:

$$k(x,y)=\exp(-\lambda d(x,y))$$



For factored MDPs, we define the metric $d(x,y)$ as the Hamming distance, which uniformly evaluates categorical mismatches to measure structural proximity for combinatorial state spaces.

---

## Installation

```bash
pip install -e .

```

Deep Q-Networks (DQN) use `device: auto` by default, which selects CUDA when PyTorch reports CUDA is available and otherwise falls back to CPU. Set `agent.device` to `cpu`, `cuda`, or another explicit PyTorch device string within the configuration files to override this behavior.

---

## Implemented Agents

The repository provides modular agents to evaluate tabular and function-approximation baselines against our proposed method:

* `RandomAgent`: Selects actions uniformly at random for baseline comparison.
* `QLearningAgent`: Implements standard Q-learning which updates only the visited state-action pair.
* `SarsaAgent`: Implements the on-policy SARSA algorithm.
* `DQNAgent`: Generalizes through neural function approximation.
* `SBQAgent`: Our proposed Structural Broadcasting Q-learning algorithm that broadcasts TD errors to structurally similar states.

*Note:* Tabular Q-learning and SARSA use sparse `defaultdict`-backed Q-tables, ensuring that large factored state spaces do not allocate the full state-action table up front.

---

## Benchmark Applications

Experiments on structured discrete benchmarks show that SBQ improves early learning speed in several sparse and combinatorial environments while preserving a simple tabular update structure. Applications live under `applications/`. Each application owns its environment wrapper, complete experiment configs, visualization utilities, and application README.

### Environment Details

* **DNA Promoter:** Features a 6-mer DNA sequence resulting in 4,096 states. The agent can mutate one position to another base, creating 18 possible actions. The environment yields a sparse reward, giving +1 only when the sequence exactly matches targets, and 0 otherwise.


* **Lights Out:** The state space consists of 16 binary cells, totaling 65,536 states. Actions involve toggling one cell and its neighbors. It features a sparse reward of +1 only when all lights are off, 0 otherwise. The environment incorporates a 5% action slip.


* **Network Routing:** The state incorporates destination ID and neighboring queue status, yielding about 1 million valid states. The agent must choose one of 8 neighboring nodes. The reward structure gives +100 at the destination, with penalties for congestion or invalid links.


* **SO-101 Reach:** States are modeled as discretized 3D target errors with 1,331 states. The agent executes small Cartesian movements. It provides a dense shaped reward including progress rewards, distance penalties, and success bonuses.


* **Danger Maze & Four-Room Grid:** State spaces are factored into row, col, goal_id, and layout_id. Agents navigate using up, down, left, and right actions. Rewards involve step penalties, invalid move penalties, and goal bonuses.


* **OpenAI Gym Adapters:** Wraps tasks including MiniGrid-DoorKey-5x5-v0, MiniGrid-LavaGapS7-v0, and Taxi-v4. These utilize standard discrete environment actions and standard task rewards.

### Adding a New Application

1. Create `applications/<name>/`.
2. Implement an environment wrapper exposing `reset`, `step`, `render`, `observation_space`, and `action_space`.
3. Return observations as `MultiDiscreteSpace` vectors.
4. Add complete application-local YAML configs.
5. Add a visualizer that saves artifacts into the run output directory.

---

## Running Experiments

### MiniGrid KeyDoor

```bash
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
pytest tests

```

### Additional Applications

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

---

## Configuration & Outputs

Each config contains the complete experiment definition: application, environment, observation, agent, training, visualization, and logging settings. There is no global `configs/` directory; configurations are modularized per application.

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

Experimental outputs are written systematically to the designated root directory:

```text
outputs/<run_name>/
├── config.yaml
├── metrics.csv
├── eval_metrics.csv
├── checkpoints/
└── visualizations/

```
