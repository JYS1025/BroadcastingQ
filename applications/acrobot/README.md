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

The original baseline converts it to the repository's required factored
`MultiDiscreteSpace` by uniformly binning each coordinate independently:

```python
MultiDiscreteSpace([8, 8, 8, 8, 10, 10])
```

The resulting categorical factors are consumed unchanged by the existing
Q-learning, DQN, and SBQ agents.

## Representation options

The wrapper supports config-selectable observation representations:

- `multidiscrete_binned_continuous`: the backward-compatible naive baseline
  over `[cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]`.
- `theta_binned`: reconstructs `[theta1, theta2, dtheta1, dtheta2]` with
  `np.arctan2` and bins those four ordered scalars as categorical factors.

With `theta_bin_counts: [31, 31, 15, 19]`, the theta representation has four
categorical factors.

## Intentional baseline limitation

The original baseline is deliberately naive.

- It does **not** reconstruct angles from sine/cosine values.
- It does **not** implement periodic/circular angle distance.
- It does **not** implement ordinal distance between nearby bins.
- It does **not** override SBQ distance or neighborhood behavior.

The theta configs keep the same generic SBQ implementation and Hamming
distance, but change the state representation to use explicit angle and
velocity factors rather than raw sine/cosine factors.

## Structural SBQ

`config_sbq_structural.yaml` uses the existing `theta_binned` representation:

```text
[theta1_bin, theta2_bin, dtheta1_bin, dtheta2_bin]
```

The application-local `acrobot_structural_sbq` keeps the base SBQ update logic
but replaces Hamming broadcasting:

- `theta1` and `theta2` use cyclic bin distance, so bin `0` and bin `n-1` are
  adjacent;
- angular velocity bins use ordinary absolute bin distance and do not wrap;
- weights are `[1.0, 1.0, 0.5, 0.5]`;
- neighborhoods are generated with local cyclic/clipped offsets and bounded by
  structural radius `2`.

This makes broadcasting local in the physical angle/velocity representation
without changing Acrobot dynamics, rewards, actions, or baseline configs.

## Files

```text
applications/acrobot/
├── __init__.py
├── env.py
├── visualize.py
├── config_qlearning.yaml
├── config_dqn.yaml
├── config_sbq.yaml
├── config_qlearning_theta.yaml
├── config_dqn_theta.yaml
├── config_sbq_theta.yaml
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
