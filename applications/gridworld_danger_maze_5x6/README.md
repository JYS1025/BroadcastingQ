# gridworld_danger_maze_5x6

Native fallback for the requested Gym-Gridworlds DangerMaze-5x6 task. The external Gym-Gridworlds package is not installed in the current `bcrl` environment, so this app uses a local fixed 5x6 map with walls and danger cells.

State is `[agent_row, agent_col]` with `MultiDiscreteSpace([5, 6])`. Actions are `0=up`, `1=down`, `2=left`, `3=right`; invalid moves leave the agent in place. Reward is `-1.0` per safe non-goal step, `+1.0` at the goal, and `-10.0` on danger cells. Goal and danger cells terminate; only the goal is success.

SBQ(ours) uses a Manhattan structural agent with radius 2 and excludes walls from neighborhoods. Danger cells remain valid states because entering them is part of the environment semantics. Visualization is a PIL-rendered RGB maze.

Main experiments:

```bash
python main.py --config applications/gridworld_danger_maze_5x6/config_qlearning.yaml
python main.py --config applications/gridworld_danger_maze_5x6/config_dqn.yaml
python main.py --config applications/gridworld_danger_maze_5x6/config_sbq.yaml
```

Hard experiments add an 8x8 randomized-context variant. The hard state is `[agent_row, agent_col, goal_id, layout_id]` with `MultiDiscreteSpace([8, 8, 4, 3])`, so there are 768 state-contexts. The layout is sampled from three fixed maps, the goal is sampled from four corners, starts are randomized, danger gives `-10.0` without terminating, and the goal gives `+20.0`.

```bash
python main.py --config applications/gridworld_danger_maze_5x6/config_qlearning_hard.yaml
python main.py --config applications/gridworld_danger_maze_5x6/config_dqn_hard.yaml
python main.py --config applications/gridworld_danger_maze_5x6/config_sbq_hard.yaml
```
