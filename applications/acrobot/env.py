from __future__ import annotations

import os
from typing import Any

# Acrobot rendering uses Pygame/SDL. On headless SSH servers, SDL may attempt
# to open a non-existent ALSA audio device when rendering evaluation GIFs.
# Use a silent audio backend unless the user explicitly configured one.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class AcrobotEnv(BaseEnv):
    """Gymnasium Acrobot-v1 wrapper with a naively discretized observation.

    BroadcastingQ's existing agents require a finite factored observation
    represented by ``MultiDiscreteSpace`` and a finite discrete action space.
    Acrobot already exposes three discrete torque actions, but its observation
    is a continuous six-vector. This wrapper independently bins those six raw
    coordinates and deliberately leaves all state-similarity behavior to the
    existing agents. Generic SBQ therefore uses its existing Hamming distance
    over these bins; no angle-aware or ordinal distance is introduced here.
    """

    FEATURE_NAMES = [
        "cos_theta1",
        "sin_theta1",
        "cos_theta2",
        "sin_theta2",
        "angular_velocity_theta1",
        "angular_velocity_theta2",
    ]
    ACTION_NAMES = ["torque_-1", "torque_0", "torque_+1"]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        observation_config = dict(config.get("observation", {}))

        env_id = str(env_config.get("gym_env_id", "Acrobot-v1"))
        self.env = gym.make(env_id, render_mode="rgb_array")

        # Gymnasium defaults to the dynamics in Sutton and Barto's book.
        # The documented NIPS alternative is exposed only as an optional,
        # application-local ablation; null preserves the official default.
        dynamics_variant = env_config.get("book_or_nips")
        if dynamics_variant is not None:
            dynamics_variant = str(dynamics_variant).lower()
            if dynamics_variant not in {"book", "nips"}:
                raise ValueError(
                    "env.book_or_nips must be either 'book', 'nips', or null; "
                    f"received {dynamics_variant!r}"
                )
            self.env.unwrapped.book_or_nips = dynamics_variant

        if not hasattr(self.env.action_space, "n"):
            raise TypeError("Acrobot wrapper requires a discrete underlying action space")
        action_count = int(self.env.action_space.n)
        if action_count != len(self.ACTION_NAMES):
            raise ValueError(
                "Unexpected Acrobot action count. Expected 3 discrete torque actions, "
                f"received {action_count}."
            )
        self.action_names = list(self.ACTION_NAMES)
        self.action_space = DiscreteActionSpace(action_count)

        raw_low = np.asarray(self.env.observation_space.low, dtype=np.float64)
        raw_high = np.asarray(self.env.observation_space.high, dtype=np.float64)
        if raw_low.shape != (6,) or raw_high.shape != (6,):
            raise ValueError(
                "Unexpected Acrobot observation space. Expected a continuous vector "
                f"of shape (6,), received low={raw_low.shape}, high={raw_high.shape}."
            )
        if not np.all(np.isfinite(raw_low)) or not np.all(np.isfinite(raw_high)):
            raise ValueError("Acrobot observation bounds must be finite for uniform binning")
        if not np.all(raw_high > raw_low):
            raise ValueError("Every Acrobot observation upper bound must exceed its lower bound")

        bin_counts = observation_config.get("bin_counts", [7, 7, 7, 7, 9, 9])
        self.bin_counts = [int(value) for value in bin_counts]
        if len(self.bin_counts) != 6:
            raise ValueError(
                "observation.bin_counts must specify one integer for each of the "
                f"six Acrobot features; received {self.bin_counts!r}"
            )
        if any(value < 2 for value in self.bin_counts):
            raise ValueError(
                "Every Acrobot bin count must be at least 2 so each feature has "
                "a non-degenerate categorical discretization"
            )

        self.raw_low = raw_low
        self.raw_high = raw_high
        self.observation_space = MultiDiscreteSpace(self.bin_counts)

        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_raw_observation: np.ndarray | None = None
        self.last_binned_observation: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the underlying Acrobot environment and return binned factors."""
        raw_obs, info = self.env.reset(seed=seed)
        obs = self._bin_observation(raw_obs)

        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self.last_binned_observation = obs.copy()

        wrapped_info = dict(info)
        wrapped_info.update(self._build_info(raw_obs, obs, success=False))
        return obs, wrapped_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the underlying environment by one discrete torque action."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; Acrobot action must be an integer in "
                f"[0, {self.action_space.n})"
            )

        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._bin_observation(raw_obs)

        self.episode_return += float(reward)
        self.episode_steps += 1
        self.last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self.last_binned_observation = obs.copy()

        # Acrobot-v1 termination means the free end reached the target height;
        # a time-limit truncation alone is not success.
        success = bool(terminated)
        wrapped_info = dict(info)
        wrapped_info.update(self._build_info(raw_obs, obs, success=success))
        wrapped_info["action_name"] = self.action_names[int(action)]
        return obs, float(reward), bool(terminated), bool(truncated), wrapped_info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Return Gymnasium's RGB rendering for the existing visualizer pipeline."""
        if mode != "rgb_array":
            raise ValueError("AcrobotEnv supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "Gymnasium returned no frame. Ensure the underlying environment "
                "is instantiated with render_mode='rgb_array'."
            )
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[-1] != 3:
            raise RuntimeError(
                f"Expected an RGB frame with shape (H, W, 3), received {frame_array.shape}"
            )
        return frame_array

    def close(self) -> None:
        self.env.close()

    def _bin_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Uniformly quantize all six continuous raw observation coordinates.

        Coordinate i with range [low_i, high_i] and B_i bins is mapped by

            floor(B_i * (clip(x_i) - low_i) / (high_i - low_i)),

        followed by clipping into [0, B_i - 1]. The final clip handles values
        equal to the upper bound exactly.
        """
        raw = np.asarray(raw_obs, dtype=np.float64)
        if raw.shape != (6,):
            raise ValueError(f"Expected raw Acrobot observation shape (6,), received {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Acrobot produced a non-finite observation: {raw!r}")

        clipped = np.clip(raw, self.raw_low, self.raw_high)
        normalized = (clipped - self.raw_low) / (self.raw_high - self.raw_low)
        bin_counts = np.asarray(self.bin_counts, dtype=np.int64)
        binned = np.floor(normalized * bin_counts).astype(np.int64)
        binned = np.clip(binned, 0, bin_counts - 1)

        if not self.observation_space.contains(binned):
            raise RuntimeError(
                "Internal discretization error: converted observation is outside "
                f"MultiDiscreteSpace({self.observation_space.nvec}): {binned!r}"
            )
        return binned

    def _build_info(
        self,
        raw_obs: np.ndarray,
        binned_obs: np.ndarray,
        *,
        success: bool,
    ) -> dict[str, Any]:
        raw = np.asarray(raw_obs, dtype=np.float32)
        return {
            "success": bool(success),
            "raw_observation": raw.tolist(),
            "binned_observation": binned_obs.tolist(),
            "symbolic_state": {
                name: int(value)
                for name, value in zip(self.FEATURE_NAMES, binned_obs)
            },
            "episode_return": float(self.episode_return),
            "episode_steps": int(self.episode_steps),
        }
