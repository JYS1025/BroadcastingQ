# SimpleGrid

Small `gym-simplegrid` wrapper with a native fallback. The local environment first tries `gymnasium.make("SimpleGrid-v0", render_mode="rgb_array")`; if `gym_simplegrid` is unavailable, it uses an application-local compatible grid.

## State And Actions

- Observation: `[agent_row, agent_col]`
- Space: `MultiDiscreteSpace([nrow, ncol])`
- Actions: `0 up`, `1 down`, `2 left`, `3 right`
- Default map: 4x5 empty grid, start `[0, 0]`, goal `[3, 4]`

Fallback rewards match the documented SimpleGrid convention: invalid move `-1.0`, normal move `-0.1`, goal `+1.0`. External rewards are passed through unchanged.

## Structural SBQ

`simple_grid_sbq` uses Manhattan distance over the agent position and broadcasts only to valid cells within `search_radius: 2`.

## Main Runs

```bash
python main.py --config applications/simple_grid/config_qlearning.yaml
python main.py --config applications/simple_grid/config_dqn.yaml
python main.py --config applications/simple_grid/config_sbq.yaml
```

## Hard Runs

The hard configs force the native fallback for a 10x10 four-rooms map with randomized corner goals and 10% action slip. The hard state is `[agent_row, agent_col, goal_id]` with `MultiDiscreteSpace([10, 10, 4])`, so there are 400 state-contexts. Normal moves give `-0.1`, invalid moves give `-1.0`, and the goal gives `+10.0`.

```bash
python main.py --config applications/simple_grid/config_qlearning_hard.yaml
python main.py --config applications/simple_grid/config_dqn_hard.yaml
python main.py --config applications/simple_grid/config_sbq_hard.yaml
```
