# PDDLGym Sokoban application

This application wraps PDDLGym `PDDLEnvSokoban-v0` for a fixed problem instance.
It does not vendor PDDLGym or modify repository dependency files. If PDDLGym is
missing, install it manually with `pip install pddlgym`.

## State

The wrapper uses object positions for one fixed problem:

```text
[player_location, stone_0_location, stone_1_location, ...]
```

Locations and stones are ordered deterministically by object name. The
observation space is:

```python
MultiDiscreteSpace([num_locations] * (1 + num_stones))
```

This avoids a full literal-set binary encoding, which would introduce many
invalid symbolic states.

## Actions And Rewards

The repository agents see four fixed actions: up, down, left, and right. These
are mapped to PDDLGym Sokoban move actions with the corresponding direction
objects.

Rewards are passed through exactly from PDDLGym. The default Sokoban reward is
sparse: `1.0` when the goal is reached and `0.0` otherwise. `success` is true
when PDDLGym reports termination.

## SBQ

Sokoban is sparse-reward symbolic planning, and moving a stone can radically
change solvability. The SBQ config therefore uses radius `1`, high
`kernel_lambda`, low `alpha_k`, lower `beta_0`, and faster annealing.
