from __future__ import annotations

import re
from typing import Any

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
AGENT_REGISTRY[
    "minigrid_blocked_unlock_pickup_sbq"
] = "applications.minigrid_blocked_unlock_pickup.agent:MiniGridBlockedUnlockPickupSBQ"
AGENT_REGISTRY[
    "minigrid_blocked_unlock_pickup_q_learning_tiebreak"
] = "applications.minigrid_blocked_unlock_pickup.agent:MiniGridBlockedUnlockPickupTieBreakQLearning"


class MiniGridBlockedUnlockPickupEnv(BaseEnv):
    """Symbolic wrapper for MiniGrid BlockedUnlockPickup."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_config = config.get("env", {})

        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 - registers MiniGrid env IDs
            from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
        except ImportError as exc:
            raise ImportError(
                "MiniGrid BlockedUnlockPickup requires minigrid. Use the bcrl conda env or install minigrid."
            ) from exc

        self.color_to_idx = dict(COLOR_TO_IDX)
        self.object_to_idx = dict(OBJECT_TO_IDX)
        self.no_color_idx = len(self.color_to_idx)
        self.no_object_idx = self.object_to_idx.get("unseen", 0)

        env_id = self.env_config.get("minigrid_env_id", "MiniGrid-BlockedUnlockPickup-v0")
        self.env = gym.make(env_id, render_mode="rgb_array")
        max_steps_override = self.env_config.get("max_steps_override")
        if max_steps_override is not None:
            self.env.unwrapped.max_steps = int(max_steps_override)
        self.episode_step_limit = self.env_config.get("episode_step_limit")
        self.episode_step_limit = None if self.episode_step_limit is None else int(self.episode_step_limit)

        self.width = int(self.env.unwrapped.width)
        self.height = int(self.env.unwrapped.height)
        self.feature_names, nvec = self._build_observation_spec()
        self.observation_space = MultiDiscreteSpace(nvec)

        self.minigrid_action_count = int(self.env.action_space.n)
        if self.minigrid_action_count != len(MINIGRID_ACTION_NAMES):
            raise ValueError(f"Unexpected MiniGrid action count: {self.minigrid_action_count}")
        default_subset = ["left", "right", "forward", "pickup", "drop", "toggle"]
        self.action_map = self._build_action_map(self.env_config.get("action_subset", default_subset))
        self.action_names = [MINIGRID_ACTION_NAMES[i] for i in self.action_map]
        self.action_space = DiscreteActionSpace(len(self.action_map))
        self.last_obs: np.ndarray | None = None
        self.episode_steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        obs = self._convert_obs(raw_obs)
        self.last_obs = obs
        self.episode_steps = 0
        info = dict(info)
        info.update(self._symbolic_info(obs))
        info["mission"] = str(raw_obs.get("mission", getattr(self.env.unwrapped, "mission", "")))
        info["success"] = False
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        minigrid_action = self.action_map[int(action)]
        raw_obs, reward, terminated, truncated, info = self.env.step(minigrid_action)
        obs = self._convert_obs(raw_obs)
        self.last_obs = obs
        self.episode_steps += 1
        local_truncated = self.episode_step_limit is not None and self.episode_steps >= self.episode_step_limit
        truncated = bool(truncated or local_truncated)
        info = dict(info)
        info.update(self._symbolic_info(obs))
        info["action_name"] = self.action_names[int(action)]
        info["minigrid_action"] = int(minigrid_action)
        info["mission"] = str(raw_obs.get("mission", getattr(self.env.unwrapped, "mission", "")))
        info["episode_steps"] = self.episode_steps
        info["TimeLimit.truncated"] = bool(local_truncated and not terminated)
        info["success"] = bool(terminated and float(reward) > 0.0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("MiniGridBlockedUnlockPickupEnv supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no frame; ensure render_mode='rgb_array'")
        return np.asarray(frame)

    def close(self) -> None:
        self.env.close()

    def _convert_obs(self, raw_obs: dict[str, Any]) -> np.ndarray:
        unwrapped = self.env.unwrapped
        agent_col, agent_row = [int(v) for v in unwrapped.agent_pos]
        carrying = unwrapped.carrying
        summary = self._scan_grid()
        target_type, target_color = self._target_from_mission(str(raw_obs.get("mission", "")))
        values: dict[str, int] = {
            "agent_row": agent_row,
            "agent_col": agent_col,
            "agent_direction": int(unwrapped.agent_dir),
            "carrying_type": self._type_idx(getattr(carrying, "type", None)),
            "carrying_color": self._color_idx(getattr(carrying, "color", None)),
            "has_key": int(carrying is not None and getattr(carrying, "type", None) == "key"),
            "door_state": summary["door_state"],
            "door_row": summary["door_row"],
            "door_col": summary["door_col"],
            "key_row": summary["key_row"],
            "key_col": summary["key_col"],
            "blocking_ball_row": summary["blocking_ball_row"],
            "blocking_ball_col": summary["blocking_ball_col"],
            "target_box_row": summary["target_box_row"],
            "target_box_col": summary["target_box_col"],
            "target_type": target_type,
            "target_color": target_color,
        }
        obs = np.array([values[name] for name in self.feature_names], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted MiniGrid BlockedUnlockPickup observation {obs} is outside {self.observation_space}")
        return obs

    def _scan_grid(self) -> dict[str, int]:
        summary = {
            "door_state": 3,
            "door_row": self.height,
            "door_col": self.width,
            "key_row": self.height,
            "key_col": self.width,
            "blocking_ball_row": self.height,
            "blocking_ball_col": self.width,
            "target_box_row": self.height,
            "target_box_col": self.width,
        }
        grid = self.env.unwrapped.grid
        for x in range(self.width):
            for y in range(self.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = getattr(obj, "type", None)
                if obj_type == "key":
                    summary["key_row"] = int(y)
                    summary["key_col"] = int(x)
                elif obj_type == "ball":
                    summary["blocking_ball_row"] = int(y)
                    summary["blocking_ball_col"] = int(x)
                elif obj_type == "box":
                    summary["target_box_row"] = int(y)
                    summary["target_box_col"] = int(x)
                elif obj_type == "door":
                    summary["door_row"] = int(y)
                    summary["door_col"] = int(x)
                    if bool(getattr(obj, "is_open", False)):
                        summary["door_state"] = 0
                    elif bool(getattr(obj, "is_locked", False)):
                        summary["door_state"] = 2
                    else:
                        summary["door_state"] = 1
        return summary

    def _target_from_mission(self, mission: str) -> tuple[int, int]:
        match = re.search(r"pick up the (?P<color>\w+) (?P<type>\w+)", mission)
        if not match:
            return self.no_object_idx, self.no_color_idx
        return self._type_idx(match.group("type")), self._color_idx(match.group("color"))

    def _build_observation_spec(self) -> tuple[list[str], list[int]]:
        fields: list[tuple[str, int]] = [
            ("agent_row", self.height),
            ("agent_col", self.width),
            ("agent_direction", 4),
            ("carrying_type", len(self.object_to_idx)),
            ("carrying_color", self.no_color_idx + 1),
            ("has_key", 2),
            ("door_state", 4),
            ("door_row", self.height + 1),
            ("door_col", self.width + 1),
            ("key_row", self.height + 1),
            ("key_col", self.width + 1),
            ("blocking_ball_row", self.height + 1),
            ("blocking_ball_col", self.width + 1),
            ("target_box_row", self.height + 1),
            ("target_box_col", self.width + 1),
            ("target_type", len(self.object_to_idx)),
            ("target_color", self.no_color_idx + 1),
        ]
        names, nvec = zip(*fields)
        return list(names), list(nvec)

    def _symbolic_info(self, obs: np.ndarray) -> dict:
        values = {name: int(value) for name, value in zip(self.feature_names, obs)}
        values["action_names"] = list(self.action_names)
        return {
            "symbolic_state": values,
            "agent_pos": (values["agent_row"], values["agent_col"]),
            "agent_dir": values["agent_direction"],
            "has_key": values["has_key"],
            "door_state": values["door_state"],
            "target_box_pos": (values["target_box_row"], values["target_box_col"]),
        }

    def _type_idx(self, value: str | None) -> int:
        if value is None:
            return self.no_object_idx
        return int(self.object_to_idx.get(str(value), self.no_object_idx))

    def _color_idx(self, value: str | None) -> int:
        if value is None:
            return self.no_color_idx
        return int(self.color_to_idx.get(str(value), self.no_color_idx))

    def _build_action_map(self, action_subset) -> list[int]:
        name_to_index = {name: idx for idx, name in enumerate(MINIGRID_ACTION_NAMES)}
        action_map = []
        for action in action_subset:
            if isinstance(action, str):
                if action not in name_to_index:
                    raise ValueError(f"Unknown MiniGrid action name: {action}")
                idx = name_to_index[action]
            else:
                idx = int(action)
            if idx == name_to_index["done"]:
                raise ValueError("The MiniGrid 'done' action is intentionally excluded")
            if idx < 0 or idx >= self.minigrid_action_count:
                raise ValueError(f"MiniGrid action index {idx} is outside the action space")
            action_map.append(idx)
        if not action_map:
            raise ValueError("action_subset must contain at least one action")
        return action_map
