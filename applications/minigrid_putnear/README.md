# MiniGrid PutNear

This wrapper uses the real `MiniGrid-PutNear-6x6-N2-v0` environment by default and converts MiniGrid internals into a finite symbolic state.

State factors include agent row/column/direction, carried object type/color, the object to move, the object it must be placed near, and stable object slots. The `done` action is excluded; the default action subset is `left`, `right`, `forward`, `pickup`, and `drop`.

If the move object is being carried and no longer appears in the grid, the wrapper sets `move_object_carried=1` and represents the carried move object in an object slot with sentinel row/column. This keeps the task object semantically explicit after pickup. The wrapper also supports an app-level `episode_step_limit` without changing MiniGrid's reward or `max_steps`.

Q-learning, DQN, and SBQ(ours) configs share the same seed, 100k-step experiment schedule, evaluation, and visualization settings. Q-learning uses an app-local random tie-break variant. SBQ(ours) uses the same SBQ update rule with an app-local Hamming neighborhood implementation, because the symbolic product space is too large for the generic `from_index` decoder.

MiniGrid is installed in the `bcrl` conda environment in this workspace.
