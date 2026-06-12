# Atari Freeway application

This application wraps Gymnasium `ALE/Freeway-v5` for the existing
BroadcastingQ interfaces. It uses RGB frames from ALE only as input to an
application-local symbolic extractor; agents never receive raw pixels or RAM.

## Dependency

Atari dependencies and ROMs are not added to repository dependency files. If
the environment is unavailable, install them manually:

```bash
pip install "gymnasium[atari,accept-rom-license]" ale-py
```

## Actions

The wrapper uses Freeway's reduced three-action space:

| Action | Meaning |
|---:|---|
| 0 | NOOP |
| 1 | UP |
| 2 | DOWN |

`full_action_space` must remain `false`.

## Rewards And Success

ALE rewards, termination, and truncation are passed through unchanged. The only
wrapper-level episode cutoff is optional `env.max_steps`, which can return
`truncated=True` without changing reward.

`success` is true when `episode_return > 0.0`, because Freeway awards points
when the controlled chicken crosses successfully.

## Symbolic State

The extractor emits:

```text
[
  chicken_lane,
  chicken_y_bin,
  current_lane_gap_bin,
  next_lane_gap_bin,
  previous_lane_gap_bin,
  current_lane_blocked,
  next_lane_blocked,
  previous_lane_blocked,
]
```

The observation space is:

```python
MultiDiscreteSpace([11, 12, 16, 16, 16, 2, 2, 2])
```

Lane `10` means the chicken is outside the clean traffic-lane bands. Gap bins
and blocked flags describe local hazards around the controlled chicken x-band
for the current lane, the lane above, and the lane below.

## Extractor

`extractor.py` uses named constants for the playfield bounds, lane bands,
controlled chicken x-band, collision margin, and gap quantization. It detects a
bright chicken-like component in the controlled x-band and saturated car pixels
within each lane. In strict mode, extractor failures raise immediately. Fallback
is disabled in the default configs.

These thresholds are empirical assumptions for `ALE/Freeway-v5`. Run the
validation script after installing Atari dependencies:

```bash
python -m applications.atari_freeway.validate_extractor
```

The validation checks multiple seeds, scripted NOOP/UP/DOWN sequences, random
rollouts, and saves overlay artifacts in:

```text
applications/atari_freeway/debug_outputs/
```

## Visualization

Visualization is enabled by default. `render(mode="rgb_array")` returns the ALE
frame with extractor overlays: chicken bbox/center, lane bands, selected
current/next/previous lanes, car segments, collision bands, and the symbolic
state label. `render(mode="raw_rgb_array")` returns the raw ALE frame.

## SBQ

The SBQ config keeps generic Hamming-distance SBQ unchanged. It uses radius
`1.0`, moderate `beta_0`, moderate `kernel_lambda`, and a modest kernel update
because the state is compact and local, but car hazards can change quickly.
