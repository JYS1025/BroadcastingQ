# gridworld_full_4x5

Native deterministic fallback for the requested Gym-Gridworlds Full-4x5 task. The external Gym-Gridworlds package is not installed in the current `bcrl` environment, so this app uses a local 4x5 fully observed gridworld.

State is `[agent_row, agent_col]` with `MultiDiscreteSpace([4, 5])`. Actions are `0=up`, `1=down`, `2=left`, `3=right`; invalid moves leave the agent in place. Reward is `-1.0` per non-goal step and `+1.0` at the goal. Success means reaching the goal.

SBQ(ours) uses a local Manhattan-distance structural agent with radius 2. Visualization is a PIL-rendered RGB grid with white cells, green goal, black walls, and a blue agent.

Main experiments:

```bash
python main.py --config applications/gridworld_full_4x5/config_qlearning.yaml
python main.py --config applications/gridworld_full_4x5/config_dqn.yaml
python main.py --config applications/gridworld_full_4x5/config_sbq.yaml
```

Hard experiments keep the easy configs intact and add a 10x10 random-corner task. The hard state is `[agent_row, agent_col, goal_id]` with `MultiDiscreteSpace([10, 10, 4])`, so there are 400 state-contexts. Starts and goals are randomized on reset, non-goal moves and invalid moves receive `-1.0`, and the goal receives `+20.0`.

```bash
python main.py --config applications/gridworld_full_4x5/config_qlearning_hard.yaml
python main.py --config applications/gridworld_full_4x5/config_dqn_hard.yaml
python main.py --config applications/gridworld_full_4x5/config_sbq_hard.yaml
```
