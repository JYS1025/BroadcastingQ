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

## Info Output

The wrapper keeps per-step `info` lean by default:

```yaml
env:
  verbose_info: false
  include_raw_literals_in_info: false
```

Default `info` includes `symbolic_state`, `episode_steps`, `success`, and
`action_name` after actions. Set `verbose_info: true` to include cached static
object/location names, and set `include_raw_literals_in_info: true` only when
debugging raw PDDL literals.

## SBQ

Sokoban is sparse-reward symbolic planning, and moving a stone can radically
change solvability. The SBQ config therefore uses radius `1`, high
`kernel_lambda`, low `alpha_k`, lower `beta_0`, and faster annealing.

## Structural SBQ

`config_sbq_structural.yaml` uses `sokoban_structural_sbq`, an application-local
SBQ subclass for the same object-position state:

```text
[player_location, stone_0_location, stone_1_location, ...]
```

The structural neighborhood keeps all stone positions fixed and moves only the
player. When PDDLGym exposes `move-dir(from, to, direction)` literals, the
wrapper attaches that location graph to the observation space and neighborhoods
include player locations within graph radius `2`. If a future PDDLGym problem
does not expose robust adjacency literals, the safe fallback is the current
player state only.

Stone position changes have infinite distance in the main structural config.
This is conservative: it shares values across nearby player positions under the
same box configuration, but avoids broad value propagation across box moves.
