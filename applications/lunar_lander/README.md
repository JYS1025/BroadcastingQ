# LunarLander-v3 application

This folder adds a Gymnasium `LunarLander-v3` wrapper compatible with the
existing BroadcastingQ interfaces without modifying `core/`, `agents/`, or
repository dependency files.

## Dependency note

`LunarLander-v3` requires Gymnasium's Box2D extras. If they are not installed,
run:

```bash
pip install "gymnasium[box2d]"
```

## Environment

The wrapper always uses the discrete action version:

```python
gym.make("LunarLander-v3", render_mode="rgb_array", continuous=False)
```

This keeps the application compatible with the existing discrete-action
Q-learning, DQN, and SBQ agents.

| Action index | Meaning |
|---:|---|
| 0 | do nothing |
| 1 | fire left orientation engine |
| 2 | fire main engine |
| 3 | fire right orientation engine |

Optional future configs can turn on wind or turbulence through the application
YAML, but this baseline does not add physics-aware distance logic.

## Observation discretization

The native observation is:

```text
[x, y, x_velocity, y_velocity, angle, angular_velocity, left_contact, right_contact]
```

The first six continuous values are independently and uniformly binned, and the
two leg-contact flags are preserved as binary categorical factors:

```python
MultiDiscreteSpace([7, 7, 7, 7, 9, 7, 2, 2])
```

This baseline discretization is deliberately naive. No physics-aware distance,
ordinal distance, circular angle handling, custom SBQ behavior, or custom agent
is used. Generic SBQ therefore treats a one-bin difference and a far-bin
difference in the same factor as one Hamming mismatch. That is a known baseline
limitation, not an implementation bug.

## Success metric

The wrapper leaves rewards and Gymnasium termination unchanged. The `success`
field in `info` is true when the accumulated episode return reaches the solved
criterion:

```python
episode_return >= 200.0
```

## Run from repository root

```bash
python main.py --config applications/lunar_lander/config_qlearning.yaml
python main.py --config applications/lunar_lander/config_dqn.yaml
python main.py --config applications/lunar_lander/config_sbq.yaml
```

The existing trainer saves metrics, checkpoints, GIF rollouts, and final PNG
frames under the configured `outputs/<run_name>/` directory.
