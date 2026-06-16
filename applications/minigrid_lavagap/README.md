# MiniGrid LavaGap

Wrapper for the real `MiniGrid-LavaGapS5-v0` environment using `render_mode="rgb_array"`. Rewards, transitions, termination, and truncation are passed through from MiniGrid.

## State And Actions

- Observation: `[agent_row, agent_col, agent_direction, goal_row, goal_col, gap_row, gap_col]`
- Space: `MultiDiscreteSpace([height, width, 4, height, width, height + 1, width + 1])`
- Actions: navigation-only wrapper actions `0 left`, `1 right`, `2 forward`
- Underlying MiniGrid mapping: if the env exposes seven actions, wrapper actions map to MiniGrid `left/right/forward`; if it exposes three actions, the mapping is identity.

The gap is inferred as the passable cell in the lava barrier. If inference is uncertain, the gap factors use the sentinel `[height, width]`.

## Structural SBQ

`minigrid_lavagap_sbq` keeps goal and gap fixed, then broadcasts across nearby agent rows/columns and nearby directions with distance `Manhattan(agent position) + direction mismatch`.

## Main Runs

```bash
python main.py --config applications/minigrid_lavagap/config_qlearning.yaml
python main.py --config applications/minigrid_lavagap/config_dqn.yaml
python main.py --config applications/minigrid_lavagap/config_sbq.yaml
```

## Hard Runs

Hard configs switch to `MiniGrid-LavaGapS7-v0` with `max_steps_override: 196`. The hard state remains `[agent_row, agent_col, agent_direction, goal_row, goal_col, gap_row, gap_col]`; for S7 this is `MultiDiscreteSpace([7, 7, 4, 7, 7, 8, 8])`. MiniGrid rewards and terminal semantics are passed through unchanged.

```bash
python main.py --config applications/minigrid_lavagap/config_qlearning_hard.yaml
python main.py --config applications/minigrid_lavagap/config_dqn_hard.yaml
python main.py --config applications/minigrid_lavagap/config_sbq_hard.yaml
```
