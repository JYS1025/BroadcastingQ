from __future__ import annotations

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["boltcrypt_sbq"] = "applications.boltcrypt.agent:BoltCryptSBQ"


class BoltCryptEnv(BaseEnv):
    """Symbolic wrapper for the external BoltCrypt environment."""

    action_names = ["north", "south", "east", "west"]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_config = config.get("env", {})
        self.observation_config = config.get("observation", {})

        try:
            from boltcrypt.envs import BoltCrypt
            from boltcrypt.envs import defines
        except ImportError as exc:
            raise ImportError(
                "BoltCrypt requires boltcrypt==0.1.2. Install it in the bcrl conda env."
            ) from exc

        self.max_room_dim = int(getattr(defines, "MAX_ROOM_DIM", 10))
        self.tile_count = int(self.observation_config.get("tile_count", 11))
        self.include_global_position = bool(self.observation_config.get("include_global_position", False))
        self.global_coord_limit = int(self.observation_config.get("global_coord_limit", 32))
        self.max_episode_steps = int(self.env_config.get("max_episode_steps", 300))
        self.render_scale = int(self.env_config.get("render_scale", 32))
        generator_config = self.env_config.get("generator_config")
        kwargs = dict(
            render_mode="rgb",
            generator_config=generator_config,
            puzzle_bonus=bool(self.env_config.get("puzzle_bonus", True)),
            key_bonus=bool(self.env_config.get("key_bonus", True)),
        )
        self.env = BoltCrypt(**kwargs)
        self.action_space = DiscreteActionSpace(4)
        self.feature_names, nvec = self._build_observation_spec()
        self.observation_space = MultiDiscreteSpace(nvec)
        self.last_raw_obs: dict | None = None
        self.last_obs: np.ndarray | None = None
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        obs = self._convert_obs(raw_obs)
        self.last_raw_obs = raw_obs
        self.last_obs = obs
        self.episode_steps = 0
        self.episode_return = 0.0
        info = dict(info)
        info.update(self._symbolic_info(obs))
        info["success"] = False
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._convert_obs(raw_obs)
        self.last_raw_obs = raw_obs
        self.last_obs = obs
        self.episode_steps += 1
        self.episode_return += float(reward)
        local_truncated = self.episode_steps >= self.max_episode_steps
        truncated = bool(truncated or local_truncated)
        info = dict(info)
        info.update(self._symbolic_info(obs))
        info["action_name"] = self.action_names[int(action)]
        info["episode_steps"] = self.episode_steps
        info["episode_return"] = self.episode_return
        info["TimeLimit.truncated"] = bool(local_truncated and not terminated)
        info["success"] = bool(terminated and float(reward) > 0.0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("BoltCryptEnv supports only mode='rgb_array'")
        frame = None
        if hasattr(self.env, "obs_to_rgb"):
            try:
                frame = self.env.obs_to_rgb()
            except TypeError:
                frame = None
        if frame is None:
            frame = self._render_grid_fallback()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if frame.max(initial=0) <= 1.0:
                frame = frame * 255
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] <= 32 and frame.shape[1] <= 32:
            scale = max(1, self.render_scale)
            frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        return frame

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _convert_obs(self, raw_obs: dict) -> np.ndarray:
        grid = np.asarray(raw_obs["grid"], dtype=np.int64)
        if grid.shape != (self.max_room_dim, self.max_room_dim):
            padded = np.zeros((self.max_room_dim, self.max_room_dim), dtype=np.int64)
            h = min(self.max_room_dim, grid.shape[0])
            w = min(self.max_room_dim, grid.shape[1])
            padded[:h, :w] = grid[:h, :w]
            grid = padded
        grid = np.clip(grid, 0, self.tile_count - 1)
        agent_pos = np.asarray(raw_obs["agent_pos"], dtype=np.int64)
        inventory = int(raw_obs.get("inventory", 0))
        values = list(grid.reshape(-1))
        values.extend([int(np.clip(agent_pos[0], 0, self.max_room_dim - 1)), int(np.clip(agent_pos[1], 0, self.max_room_dim - 1))])
        values.append(int(np.clip(inventory, 0, 1)))
        if self.include_global_position:
            global_pos = np.asarray(raw_obs.get("global_pos", [0, 0]), dtype=np.int64)
            limit = self.global_coord_limit
            gx = int(np.clip(global_pos[0], -limit, limit)) + limit
            gy = int(np.clip(global_pos[1], -limit, limit)) + limit
            values.extend([gx, gy])
        obs = np.asarray(values, dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted BoltCrypt observation is outside {self.observation_space}")
        return obs

    def _build_observation_spec(self) -> tuple[list[str], list[int]]:
        fields = [(f"tile_{idx}", self.tile_count) for idx in range(self.max_room_dim * self.max_room_dim)]
        fields.extend([("agent_local_x", self.max_room_dim), ("agent_local_y", self.max_room_dim), ("inventory", 2)])
        if self.include_global_position:
            fields.extend(
                [
                    ("agent_global_x", 2 * self.global_coord_limit + 1),
                    ("agent_global_y", 2 * self.global_coord_limit + 1),
                ]
            )
        names, nvec = zip(*fields)
        return list(names), list(nvec)

    def _symbolic_info(self, obs: np.ndarray) -> dict:
        values = {name: int(value) for name, value in zip(self.feature_names, obs)}
        agent_local_pos = (values["agent_local_x"], values["agent_local_y"])
        symbolic_state = {
            "agent_local_pos": agent_local_pos,
            "inventory": values["inventory"],
            "action_names": list(self.action_names),
            "agent_local_x": values["agent_local_x"],
            "agent_local_y": values["agent_local_y"],
        }
        if self.include_global_position:
            symbolic_state["agent_global_pos"] = (values["agent_global_x"], values["agent_global_y"])
            symbolic_state["agent_global_x"] = values["agent_global_x"]
            symbolic_state["agent_global_y"] = values["agent_global_y"]
        return {
            "symbolic_state": symbolic_state,
            "agent_local_pos": agent_local_pos,
            "inventory": values["inventory"],
        }

    def _render_grid_fallback(self) -> np.ndarray:
        if self.last_raw_obs is None:
            return np.zeros((320, 320, 3), dtype=np.uint8)
        grid = np.asarray(self.last_raw_obs["grid"], dtype=np.int64)
        colors = np.array(
            [
                [238, 238, 238],
                [40, 40, 40],
                [135, 85, 40],
                [68, 160, 90],
                [246, 203, 92],
                [95, 95, 120],
                [90, 120, 240],
                [150, 150, 150],
                [170, 95, 210],
                [110, 80, 60],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        img = colors[np.clip(grid, 0, len(colors) - 1)]
        agent_x, agent_y = [int(v) for v in np.asarray(self.last_raw_obs["agent_pos"], dtype=np.int64)]
        if 0 <= agent_y < img.shape[0] and 0 <= agent_x < img.shape[1]:
            img[agent_y, agent_x] = np.array([230, 60, 60], dtype=np.uint8)
        return np.kron(img, np.ones((32, 32, 1), dtype=np.uint8))
