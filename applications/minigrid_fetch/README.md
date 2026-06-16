# MiniGrid Fetch

This wrapper uses the real `MiniGrid-Fetch-5x5-N2-v0` environment by default and exposes a finite symbolic observation for tabular, DQN, and generic SBQ agents.

State factors include the agent row/column/direction, carried object type/color, the target object parsed from the mission, and a small sorted set of object slots. The `done` action is excluded; the default action subset is `left`, `right`, `forward`, and `pickup`.

The mission parser accepts all Fetch mission forms used by MiniGrid: `get`, `go get`, `fetch`, `go fetch`, and `you must fetch`. The wrapper also supports an app-level `episode_step_limit` without changing MiniGrid's reward or `max_steps`.

The configs use 100k-step experiment schedules and identical visualization settings. Q-learning uses an app-local random tie-break variant so all-zero rows do not always choose `left` during diagnostics. SBQ(ours) uses the same SBQ update rule with an app-local Hamming neighborhood implementation, because the symbolic product space is too large for the generic `from_index` decoder.

MiniGrid is installed in the `bcrl` conda environment in this workspace.
