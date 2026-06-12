# BoltCrypt

This wrapper uses the real external `boltcrypt==0.1.2` environment. The package registers `BoltCrypt-v0`, but this wrapper instantiates the source environment directly so it can avoid the package's broken `rgb_array` render path.

Observation factors are a flattened local `10x10` tile grid, the agent local x/y position, and an inventory flag. Q-learning and DQN configs include encoded global room coordinates by default to reduce room aliasing; the SBQ config keeps local-only state for local-pattern ablation. Global coordinates are clipped to `[-global_coord_limit, global_coord_limit]` and shifted into non-negative bins. Actions are `north`, `south`, `east`, and `west`.

The external environment reports `truncated=False`, so the wrapper adds a local `max_episode_steps` time limit and marks `TimeLimit.truncated` when that limit is reached. Tiny `obs_to_rgb()` frames are upscaled before GIF/PNG output.

Q-learning, DQN, and SBQ(ours) configs share seed, 200k-step experiment schedule, evaluation cadence, evaluation episodes, evaluation seed, and visualization settings. SBQ(ours) uses the same SBQ update rule with an app-local Hamming neighborhood implementation, because the flattened local grid creates a symbolic product space too large for the generic `from_index` decoder.

Install into the `bcrl` env with:

```bash
conda run -n bcrl python -m pip install --no-deps boltcrypt==0.1.2
```
