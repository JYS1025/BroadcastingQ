# MiniGrid DynamicObstacles

Wrapper for the real `MiniGrid-Dynamic-Obstacles-5x5-v0` environment using `render_mode="rgb_array"`. Rewards, obstacle movement, collision behavior, termination, and truncation are passed through from MiniGrid.

## State And Actions

- Observation: `[agent_row, agent_col, agent_direction, goal_row, goal_col, obstacle0_row, obstacle0_col, ...]`
- Default space for the 5x5 config: `MultiDiscreteSpace([5, 5, 4, 5, 5, 6, 6, 6, 6])`
- Actions: navigation-only wrapper actions `0 left`, `1 right`, `2 forward`
- Underlying MiniGrid mapping: identity for three-action envs, or `left/right/forward` subset for seven-action envs.

Obstacles are scanned from MiniGrid objects with type `ball` or `obstacle`, sorted by row/column, and padded with sentinel `[height, width]` if fewer than `max_obstacles` are present.

## Structural SBQ

`minigrid_dynamic_obstacles_sbq` keeps goal and obstacle positions fixed in the neighborhood and broadcasts only across nearby agent positions with the current direction.

## Main Runs

```bash
python main.py --config applications/minigrid_dynamic_obstacles/config_qlearning.yaml
python main.py --config applications/minigrid_dynamic_obstacles/config_dqn.yaml
python main.py --config applications/minigrid_dynamic_obstacles/config_sbq.yaml
```

## Hard Runs

Hard configs switch to `MiniGrid-Dynamic-Obstacles-8x8-v0` with `max_steps_override: 256` and two obstacle slots. The hard state is `[agent_row, agent_col, agent_direction, goal_row, goal_col, obstacle0_row, obstacle0_col, obstacle1_row, obstacle1_col]`, giving `MultiDiscreteSpace([8, 8, 4, 8, 8, 9, 9, 9, 9])`. MiniGrid rewards, obstacle motion, collision behavior, and terminal semantics are passed through unchanged.

```bash
python main.py --config applications/minigrid_dynamic_obstacles/config_qlearning_hard.yaml
python main.py --config applications/minigrid_dynamic_obstacles/config_dqn_hard.yaml
python main.py --config applications/minigrid_dynamic_obstacles/config_sbq_hard.yaml
```
