from __future__ import annotations

import importlib.util
from typing import Any

import gymnasium as gym
import numpy as np

from applications.atari_freeway.extractor import FreewayFeatureExtractor
from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class AtariFreewayEnv(BaseEnv):
    FEATURE_NAMES = [
        "chicken_lane",
        "chicken_y_bin",
        "current_lane_gap_bin",
        "next_lane_gap_bin",
        "previous_lane_gap_bin",
        "current_lane_blocked",
        "next_lane_blocked",
        "previous_lane_blocked",
    ]
    ACTION_NAMES = ["NOOP", "UP", "DOWN"]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        obs_config = dict(config.get("observation", {}))

        if obs_config.get("type", "freeway_symbolic_lane_hazard") != "freeway_symbolic_lane_hazard":
            raise ValueError("AtariFreewayEnv supports only observation.type='freeway_symbolic_lane_hazard'")
        nvec = [int(value) for value in obs_config.get("nvec", [11, 12, 16, 16, 16, 2, 2, 2])]
        self.observation_space = MultiDiscreteSpace(nvec)
        self.action_space = DiscreteActionSpace(3)
        self.action_names = list(self.ACTION_NAMES)

        self.max_steps = env_config.get("max_steps", 1000)
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.render_overlay = bool(env_config.get("render_overlay", True))
        self.strict_extractor = bool(env_config.get("strict_extractor", True))
        self.allow_extractor_fallback = bool(env_config.get("allow_extractor_fallback", False))

        self.extractor = FreewayFeatureExtractor(nvec=tuple(nvec))
        self.raw_frame: np.ndarray | None = None
        self.last_state: np.ndarray | None = None
        self.last_extractor_debug: dict = {}
        self.last_overlay_data: dict = {}
        self.episode_return = 0.0
        self.episode_length = 0

        self.env = self._make_env(env_config)
        action_count = int(getattr(self.env.action_space, "n", 0))
        if action_count != 3:
            self.env.close()
            raise RuntimeError(
                "ALE/Freeway-v5 must expose the reduced 3-action space. "
                f"Received action_space.n={action_count}; do not set full_action_space=True."
            )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        self.episode_return = 0.0
        self.episode_length = 0
        self.extractor.failure_count = 0
        self.extractor.fallback_count = 0
        self.extractor.reset_tracking()
        obs = self._extract(raw_obs)
        wrapped_info = dict(info)
        wrapped_info.update(self._build_info(obs, raw_reward=0.0, action_name=None))
        return obs, wrapped_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Freeway action {action!r}; expected an integer in [0, 3)")
        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.episode_return += float(reward)
        self.episode_length += 1
        obs = self._extract(raw_obs)

        terminated = bool(terminated)
        truncated = bool(truncated)
        if self.max_steps is not None and self.episode_length >= self.max_steps and not terminated:
            truncated = True

        wrapped_info = dict(info)
        wrapped_info.update(
            self._build_info(
                obs,
                raw_reward=float(reward),
                action_name=self.action_names[int(action)],
            )
        )
        return obs, float(reward), terminated, truncated, wrapped_info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if self.raw_frame is None:
            raise RuntimeError("AtariFreewayEnv.render() called before reset()")
        if mode == "raw_rgb_array":
            return np.asarray(self.raw_frame, dtype=np.uint8)
        if mode == "rgb_array":
            if self.render_overlay:
                from applications.atari_freeway.visualize import draw_freeway_overlay

                return draw_freeway_overlay(self.raw_frame, self.last_overlay_data, self.last_extractor_debug)
            return np.asarray(self.raw_frame, dtype=np.uint8)
        raise ValueError("AtariFreewayEnv supports mode='rgb_array' and mode='raw_rgb_array'")

    def close(self) -> None:
        self.env.close()

    def _make_env(self, env_config: dict):
        if importlib.util.find_spec("ale_py") is None:
            raise ImportError(
                "ALE/Atari dependencies are missing. Install them manually with "
                "`pip install \"gymnasium[atari,accept-rom-license]\" ale-py`."
            )
        try:
            import ale_py
        except ImportError as exc:
            raise ImportError(
                "ALE/Atari dependencies are missing. Install them manually with "
                "`pip install \"gymnasium[atari,accept-rom-license]\" ale-py`."
            ) from exc
        gym.register_envs(ale_py)
        try:
            env = gym.make(
                str(env_config.get("gym_env_id", "ALE/Freeway-v5")),
                obs_type=str(env_config.get("obs_type", "rgb")),
                frameskip=int(env_config.get("frameskip", 4)),
                repeat_action_probability=float(env_config.get("repeat_action_probability", 0.0)),
                full_action_space=bool(env_config.get("full_action_space", False)),
                mode=int(env_config.get("mode", 0)),
                difficulty=int(env_config.get("difficulty", 0)),
                render_mode="rgb_array",
            )
        except Exception as exc:
            raise ImportError(
                "Could not create ALE/Freeway-v5. Install Atari extras and ROM license with "
                "`pip install \"gymnasium[atari,accept-rom-license]\" ale-py`."
            ) from exc
        return env

    def _extract(self, raw_obs: np.ndarray) -> np.ndarray:
        self.raw_frame = np.asarray(raw_obs, dtype=np.uint8)
        result = self.extractor.extract(
            self.raw_frame,
            strict=self.strict_extractor,
            allow_fallback=self.allow_extractor_fallback,
        )
        obs = result.state
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Freeway symbolic state outside observation_space: {obs!r}")
        self.last_state = obs.copy()
        self.last_extractor_debug = dict(result.debug)
        self.last_overlay_data = dict(result.overlay_data)
        return obs

    def _build_info(self, obs: np.ndarray, *, raw_reward: float, action_name: str | None) -> dict[str, Any]:
        info = {
            "symbolic_state": {
                name: int(value)
                for name, value in zip(self.FEATURE_NAMES, obs)
            },
            "raw_reward": float(raw_reward),
            "episode_return": float(self.episode_return),
            "episode_length": int(self.episode_length),
            "success": bool(self.episode_return > 0.0),
            "extractor_debug": dict(self.last_extractor_debug),
        }
        if action_name is not None:
            info["action_name"] = action_name
        return info
