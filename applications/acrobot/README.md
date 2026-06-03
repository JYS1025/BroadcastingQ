# Acrobot-v1 application

This folder adds a Gymnasium `Acrobot-v1` wrapper compatible with the existing
BroadcastingQ interfaces without modifying `core/` or `agents/`.

## What this application represents

Acrobot is a two-link swing-up control task. It already has a finite discrete
action space:

| Action index | Meaning |
|---:|---|
| 0 | apply -1 torque |
| 1 | apply 0 torque |
| 2 | apply +1 torque |

The native observation is a continuous six-dimensional vector:

```text
[cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]
```

This application converts it to the repository's required factored
`MultiDiscreteSpace` by uniformly binning each coordinate independently:

```python
MultiDiscreteSpace([10, 10, 10, 10, 12, 12])
```

The resulting categorical factors are consumed unchanged by the existing
Q-learning, DQN, and SBQ agents.

## Intentional baseline limitation

This implementation is deliberately naive.

- It does **not** reconstruct angles from sine/cosine values.
- It does **not** implement periodic/circular angle distance.
- It does **not** implement ordinal distance between nearby bins.
- It does **not** override SBQ distance or neighborhood behavior.

Consequently, current generic SBQ uses its existing Hamming distance over
categorical bins. Changing a velocity from bin 3 to bin 4 and changing it from
bin 3 to bin 8 both count as one differing factor to SBQ. This is a baseline
design choice for testing the current method without changing the method
implementation.

## Files

```text
applications/acrobot/
├── __init__.py
├── env.py
├── visualize.py
├── config_qlearning.yaml
├── config_dqn.yaml
├── config_sbq.yaml
├── smoke_check.py
└── README.md
```

## Run from repository root

```bash
python main.py --config applications/acrobot/config_qlearning.yaml
python main.py --config applications/acrobot/config_dqn.yaml
python main.py --config applications/acrobot/config_sbq.yaml
```

The existing trainer saves metrics, checkpoints, GIF rollouts, and final PNG
frames under the configured `outputs/<run_name>/` directory.

## Quick wrapper smoke check

```bash
python -m applications.acrobot.smoke_check
```

## Optional dynamics ablation

Gymnasium's default Acrobot dynamics use the Sutton-and-Barto book equations.
The wrapper allows a separately labelled NIPS-dynamics ablation by setting:

```yaml
env:
  book_or_nips: nips
```

Leave it as `null` for the default experiment.

## Dependency note

This application uses Gymnasium Classic Control rendering. If rendering
dependencies are missing in a fresh environment, install:

```bash
pip install "gymnasium[classic-control]"
```
