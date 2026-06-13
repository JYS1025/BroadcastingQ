# KeyDoor Application

This application wraps MiniGrid DoorKey environments through Gymnasium and `minigrid`.
MiniGrid owns the dynamics, rewards, termination, truncation, and RGB rendering.

The wrapper converts MiniGrid internals into a symbolic `MultiDiscreteSpace` observation. The minimal factors are:

```text
[agent_row, agent_col, agent_direction, has_key, door_state]
```

`door_state` is `0` when no door is found, `1` when a door is closed or locked, and `2` when any door is open.

The bundled experiment configs additionally include key and door positions, and use the MiniGrid actions needed for DoorKey: left, right, forward, pickup, and toggle. This keeps tabular learning from wasting exploration on `drop`/`done` and reduces partial-observability aliasing.

Run examples:

```bash
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
python main.py --config applications/key_door/config_sbq.yaml
```
