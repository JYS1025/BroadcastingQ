# MiniGrid BlockedUnlockPickup

This wrapper uses the real `MiniGrid-BlockedUnlockPickup-v0` environment by default and encodes the puzzle into finite symbolic factors.

State factors include agent pose, carried object, key possession, door state and position, key position, blocking ball position, target box position, and target object metadata parsed from the mission. Door state follows MiniGrid's documented encoding: `0=open`, `1=closed`, `2=locked`, and this wrapper uses `3` only as a missing/not-found sentinel. The `done` action is excluded; `toggle` is included because unlocking/opening the door is part of the task.

The wrapper supports an app-level `episode_step_limit` without changing MiniGrid's reward or `max_steps`. All three configs keep seed, 300k-step experiment schedule, evaluation, and visualization settings aligned. Q-learning uses an app-local random tie-break variant. SBQ(ours) uses the same SBQ update rule with an app-local Hamming neighborhood implementation, because the symbolic product space is too large for the generic `from_index` decoder.

MiniGrid is installed in the `bcrl` conda environment in this workspace.
