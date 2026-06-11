from __future__ import annotations

import os
from typing import Any

# LunarLander rendering uses Pygame/SDL. On headless machines, keep rendering
# quiet unless the user explicitly configured a different SDL audio backend.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class LunarLanderEnv(BaseEnv):
    """Gymnasium LunarLander-v3 wrapper with naively binned observations.

    Generic SBQ still uses its unchanged Hamming distance over the emitted
    ``MultiDiscreteSpace``.
    """

    LOW_CONT = np.array(
        [-2.5, -2.5, -10.0, -10.0, -2.0 * np.pi, -10.0],
        dtype=np.float32,
    )
    HIGH_CONT = np.array(
        [2.5, 2.5, 10.0, 10.0, 2.0 * np.pi, 10.0],
        dtype=np.float32,
    )
    CONTINUOUS_FEATURE_NAMES = [
        "x_coordinate",
        "y_coordinate",
        "x_velocity",
        "y_velocity",
        "angle",
        "angular_velocity",
    ]
    CONTACT_FEATURE_NAMES = [
        "left_leg_contact",
        "right_leg_contact",
    ]
    ACTION_NAMES = [
        "do_nothing",
        "fire_left_orientation_engine",
        "fire_main_engine",
        "fire_right_orientation_engine",
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        observation_config = dict(config.get("observation", {}))

        continuous = env_config.get("continuous", False)
        if isinstance(continuous, str):
            continuous = continuous.strip().lower() in {"1", "true", "yes", "on"}
        if bool(continuous):
            raise ValueError(
                "LunarLanderEnv supports only env.continuous=false because the "
                "existing repository agents require discrete actions."
            )

        env_id = str(env_config.get("gym_env_id", "LunarLander-v3"))
        self.env = gym.make(
            env_id,
            render_mode="rgb_array",
            continuous=False,
            gravity=float(env_config.get("gravity", -10.0)),
            enable_wind=bool(env_config.get("enable_wind", False)),
            wind_power=float(env_config.get("wind_power", 15.0)),
            turbulence_power=float(env_config.get("turbulence_power", 1.5)),
        )

        if not hasattr(self.env.action_space, "n"):
            raise TypeError("LunarLanderEnv requires a discrete underlying action space")
        action_count = int(self.env.action_space.n)
        if action_count != len(self.ACTION_NAMES):
            raise ValueError(
                "Unexpected LunarLander action count. Expected 4 discrete actions, "
                f"received {action_count}."
            )
        self.action_names = list(self.ACTION_NAMES)
        self.action_space = DiscreteActionSpace(action_count)

        self.observation_type = str(observation_config.get("type", "multidiscrete_binned_continuous"))
        if self.observation_type != "multidiscrete_binned_continuous":
            raise ValueError(
                "Unsupported LunarLander observation.type "
                f"{self.observation_type!r}. Expected 'multidiscrete_binned_continuous'."
            )

        continuous_bin_counts = observation_config.get(
            "continuous_bin_counts",
            [7, 7, 7, 7, 9, 7],
        )
        self.continuous_bin_counts = [int(value) for value in continuous_bin_counts]
        if len(self.continuous_bin_counts) != 6:
            raise ValueError(
                "observation.continuous_bin_counts must specify six values for "
                "the continuous LunarLander observation factors; received "
                f"{self.continuous_bin_counts!r}"
            )
        if any(value < 2 for value in self.continuous_bin_counts):
            raise ValueError("Every continuous LunarLander bin count must be at least 2")

        nvec = self.continuous_bin_counts + [2, 2]
        self.feature_names = list(self.CONTINUOUS_FEATURE_NAMES + self.CONTACT_FEATURE_NAMES)
        self.observation_space = MultiDiscreteSpace(nvec)

        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_raw_observation: np.ndarray | None = None
        self.last_binned_observation: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        obs = self._bin_observation(raw_obs)

        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self.last_binned_observation = obs.copy()

        wrapped_info = dict(info)
        wrapped_info.update(
            self._build_info(
                raw_obs,
                obs,
                action_name=None,
                terminated=False,
                reward=0.0,
            )
        )
        return obs, wrapped_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; LunarLander action must be an "
                f"integer in [0, {self.action_space.n})"
            )

        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._bin_observation(raw_obs)

        self.episode_return += float(reward)
        self.episode_steps += 1
        self.last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self.last_binned_observation = obs.copy()

        wrapped_info = dict(info)
        wrapped_info.update(
            self._build_info(
                raw_obs,
                obs,
                action_name=self.action_names[int(action)],
                terminated=bool(terminated),
                reward=float(reward),
            )
        )
        return obs, float(reward), bool(terminated), bool(truncated), wrapped_info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("LunarLanderEnv supports only mode='rgb_array'")
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
        raw = np.asarray(raw_obs, dtype=np.float32)
        if raw.shape != (8,):
            raise ValueError(f"Expected raw LunarLander observation shape (8,), received {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"LunarLander produced a non-finite observation: {raw!r}")

        binned_continuous = self._uniform_bin(
            raw[:6],
            self.LOW_CONT,
            self.HIGH_CONT,
            self.continuous_bin_counts,
        )

        left_contact = int(float(raw[6]) >= 0.5)
        right_contact = int(float(raw[7]) >= 0.5)
        contacts = np.array([left_contact, right_contact], dtype=np.int64)

        obs = np.concatenate([binned_continuous, contacts]).astype(np.int64)

        if not self.observation_space.contains(obs):
            raise RuntimeError(
                "Internal discretization error: converted observation is outside "
                f"MultiDiscreteSpace({self.observation_space.nvec}): {obs!r}"
            )
        return obs

    def _uniform_bin(
        self,
        values: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        bin_counts: list[int],
    ) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=np.float32), low, high)
        normalized = (clipped - low) / (high - low)
        counts = np.asarray(bin_counts, dtype=np.int64)
        binned = np.floor(normalized * counts).astype(np.int64)
        return np.clip(binned, 0, counts - 1).astype(np.int64)

    def _build_info(
        self,
        raw_obs: np.ndarray,
        binned_obs: np.ndarray,
        *,
        action_name: str | None,
        terminated: bool,
        reward: float,
    ) -> dict[str, Any]:
        raw = np.asarray(raw_obs, dtype=np.float32)
        info: dict[str, Any] = {
            "raw_observation": raw.tolist(),
            "binned_observation": binned_obs.tolist(),
            "episode_return": float(self.episode_return),
            "episode_steps": int(self.episode_steps),
            "landed_or_crashed": bool(terminated),
            "success": bool(self.episode_return >= 200.0),
            "symbolic_state": {
                name: int(value)
                for name, value in zip(self.feature_names, binned_obs)
            },
        }
        if action_name is not None:
            info["action_name"] = action_name
        return info
