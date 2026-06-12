from __future__ import annotations

import re
from typing import Any

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
AGENT_REGISTRY["minigrid_putnear_sbq"] = "applications.minigrid_putnear.agent:MiniGridPutNearSBQ"
AGENT_REGISTRY[
    "minigrid_putnear_q_learning_tiebreak"
] = "applications.minigrid_putnear.agent:MiniGridPutNearTieBreakQLearning"


class MiniGridPutNearEnv(BaseEnv):
    """Symbolic wrapper for MiniGrid PutNear tasks."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_config = config.get("env", {})
        self.observation_config = config.get("observation", {})

        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 - registers MiniGrid env IDs
            from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
        except ImportError as exc:
            raise ImportError(
                "MiniGrid PutNear requires minigrid. Use the bcrl conda env or install minigrid."
            ) from exc

        self.color_to_idx = dict(COLOR_TO_IDX)
        self.object_to_idx = dict(OBJECT_TO_IDX)
        self.no_color_idx = len(self.color_to_idx)
        self.no_object_idx = self.object_to_idx.get("unseen", 0)

        env_id = self.env_config.get("minigrid_env_id", "MiniGrid-PutNear-6x6-N2-v0")
        self.env = gym.make(env_id, render_mode="rgb_array")
        max_steps_override = self.env_config.get("max_steps_override")
        if max_steps_override is not None:
            self.env.unwrapped.max_steps = int(max_steps_override)
        self.episode_step_limit = self.env_config.get("episode_step_limit")
        self.episode_step_limit = None if self.episode_step_limit is None else int(self.episode_step_limit)

        self.width = int(self.env.unwrapped.width)
        self.height = int(self.env.unwrapped.height)
        self.max_objects = int(self.observation_config.get("max_object_slots", 4))
        self.feature_names, nvec = self._build_observation_spec()
        self.observation_space = MultiDiscreteSpace(nvec)

        self.minigrid_action_count = int(self.env.action_space.n)
        if self.minigrid_action_count != len(MINIGRID_ACTION_NAMES):
            raise ValueError(f"Unexpected MiniGrid action count: {self.minigrid_action_count}")
        default_subset = ["left", "right", "forward", "pickup", "drop"]
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
            raise ValueError("MiniGridPutNearEnv supports only mode='rgb_array'")
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
        carrying_type = self._type_idx(getattr(carrying, "type", None))
        carrying_color = self._color_idx(getattr(carrying, "color", None))
        move_type, move_color, target_type, target_color = self._task_from_mission(str(raw_obs.get("mission", "")))
        move_object_carried = int(carrying_type == move_type and carrying_color == move_color)
        objects = self._scan_objects(move_type, move_color, target_type, target_color, move_object_carried)

        values: dict[str, int] = {
            "agent_row": agent_row,
            "agent_col": agent_col,
            "agent_direction": int(unwrapped.agent_dir),
            "carrying_type": carrying_type,
            "carrying_color": carrying_color,
            "move_type": move_type,
            "move_color": move_color,
            "move_object_carried": move_object_carried,
            "target_type": target_type,
            "target_color": target_color,
        }
        for idx, obj in enumerate(objects):
            prefix = f"object_{idx}"
            values[f"{prefix}_row"] = int(obj["row"])
            values[f"{prefix}_col"] = int(obj["col"])
            values[f"{prefix}_type"] = int(obj["type"])
            values[f"{prefix}_color"] = int(obj["color"])

        obs = np.array([values[name] for name in self.feature_names], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted MiniGrid PutNear observation {obs} is outside {self.observation_space}")
        return obs

    def _scan_objects(
        self,
        move_type: int,
        move_color: int,
        target_type: int,
        target_color: int,
        move_object_carried: int,
    ) -> list[dict[str, int]]:
        rows: list[dict[str, int]] = []
        if move_object_carried:
            rows.append(
                {
                    "row": self.height,
                    "col": self.width,
                    "type": move_type,
                    "color": move_color,
                    "rank": 3,
                }
            )
        grid = self.env.unwrapped.grid
        for x in range(self.width):
            for y in range(self.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = getattr(obj, "type", None)
                if obj_type in {"wall", "floor"}:
                    continue
                type_idx = self._type_idx(obj_type)
                color_idx = self._color_idx(getattr(obj, "color", None))
                is_move = int(type_idx == move_type and color_idx == move_color)
                is_target = int(type_idx == target_type and color_idx == target_color)
                rows.append(
                    {
                        "row": int(y),
                        "col": int(x),
                        "type": type_idx,
                        "color": color_idx,
                        "rank": 2 * is_move + is_target,
                    }
                )
        rows.sort(key=lambda item: (-item["rank"], item["type"], item["color"], item["row"], item["col"]))
        sentinel = {"row": self.height, "col": self.width, "type": self.no_object_idx, "color": self.no_color_idx}
        rows = rows[: self.max_objects]
        while len(rows) < self.max_objects:
            rows.append(dict(sentinel))
        return rows

    def _task_from_mission(self, mission: str) -> tuple[int, int, int, int]:
        mission = str(mission).strip().lower()
        match = re.search(
            r"^put the (?P<move_color>\w+) (?P<move_type>\w+) near the (?P<target_color>\w+) (?P<target_type>\w+)$",
            mission,
        )
        if not match:
            return self.no_object_idx, self.no_color_idx, self.no_object_idx, self.no_color_idx
        return (
            self._type_idx(match.group("move_type")),
            self._color_idx(match.group("move_color")),
            self._type_idx(match.group("target_type")),
            self._color_idx(match.group("target_color")),
        )

    def _build_observation_spec(self) -> tuple[list[str], list[int]]:
        fields: list[tuple[str, int]] = [
            ("agent_row", self.height),
            ("agent_col", self.width),
            ("agent_direction", 4),
            ("carrying_type", len(self.object_to_idx)),
            ("carrying_color", self.no_color_idx + 1),
            ("move_type", len(self.object_to_idx)),
            ("move_color", self.no_color_idx + 1),
            ("move_object_carried", 2),
            ("target_type", len(self.object_to_idx)),
            ("target_color", self.no_color_idx + 1),
        ]
        for idx in range(self.max_objects):
            fields.extend(
                [
                    (f"object_{idx}_row", self.height + 1),
                    (f"object_{idx}_col", self.width + 1),
                    (f"object_{idx}_type", len(self.object_to_idx)),
                    (f"object_{idx}_color", self.no_color_idx + 1),
                ]
            )
        names, nvec = zip(*fields)
        return list(names), list(nvec)

    def _symbolic_info(self, obs: np.ndarray) -> dict:
        values = {name: int(value) for name, value in zip(self.feature_names, obs)}
        values["action_names"] = list(self.action_names)
        return {
            "symbolic_state": values,
            "agent_pos": (values["agent_row"], values["agent_col"]),
            "agent_dir": values["agent_direction"],
            "move_target": (values["move_type"], values["move_color"]),
            "move_object_carried": values["move_object_carried"],
            "near_target": (values["target_type"], values["target_color"]),
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
