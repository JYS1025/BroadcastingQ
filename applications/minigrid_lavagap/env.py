from __future__ import annotations

from collections import Counter

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["minigrid_lavagap_sbq"] = "applications.minigrid_lavagap.agent:LavaGapStructuralSBQ"


class MiniGridLavaGapEnv(BaseEnv):
    """MiniGrid LavaGap wrapper with compact symbolic observations."""

    default_action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    nav_action_names = ["left", "right", "forward"]
    feature_names = [
        "agent_row",
        "agent_col",
        "agent_direction",
        "goal_row",
        "goal_col",
        "gap_row",
        "gap_col",
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        env_id = str(env_config.get("minigrid_env_id", "MiniGrid-LavaGapS5-v0"))
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401
        except ImportError as exc:
            raise ImportError("MiniGrid LavaGap requires gymnasium and minigrid. Run inside the bcrl env or install minigrid.") from exc

        self.env = gym.make(env_id, render_mode="rgb_array")
        max_steps_override = env_config.get("max_steps_override")
        if max_steps_override is not None:
            self.env.unwrapped.max_steps = int(max_steps_override)
        self.width = int(self.env.unwrapped.width)
        self.height = int(self.env.unwrapped.height)
        self.observation_space = MultiDiscreteSpace([self.height, self.width, 4, self.height, self.width, self.height + 1, self.width + 1])
        self._set_valid_agent_positions(None)

        self.minigrid_action_count = int(self.env.action_space.n)
        all_action_names = self._minigrid_action_names()
        self.action_map = self._build_action_map(env_config.get("action_subset", self.nav_action_names), all_action_names)
        self.action_names = [all_action_names[idx] for idx in self.action_map]
        self.action_space = DiscreteActionSpace(len(self.action_map))
        self.last_observation: np.ndarray | None = None
        self.last_action_name: str | None = None
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_action_name = None
        obs = self._convert_obs(raw_obs)
        wrapped = dict(info)
        wrapped.update(self._symbolic_info(obs, action=None, success=False))
        return obs, wrapped

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action = int(action)
        minigrid_action = self.action_map[action]
        raw_obs, reward, terminated, truncated, info = self.env.step(minigrid_action)
        obs = self._convert_obs(raw_obs)
        self.episode_steps += 1
        self.episode_return += float(reward)
        success = bool(terminated and float(reward) > 0.0)
        wrapped = dict(info)
        wrapped.update(self._symbolic_info(obs, action=action, success=success))
        wrapped["minigrid_action"] = int(minigrid_action)
        wrapped["episode_steps"] = self.episode_steps
        wrapped["episode_return"] = self.episode_return
        return obs, float(reward), bool(terminated), bool(truncated), wrapped

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("MiniGridLavaGapEnv supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no frame; ensure render_mode='rgb_array'")
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"Expected MiniGrid RGB frame, got shape {arr.shape}")
        return arr.astype(np.uint8, copy=False)

    def close(self) -> None:
        self.env.close()

    def _convert_obs(self, raw_obs) -> np.ndarray:
        unwrapped = self.env.unwrapped
        agent_col = int(unwrapped.agent_pos[0])
        agent_row = int(unwrapped.agent_pos[1])
        agent_direction = int(unwrapped.agent_dir)
        summary = self._scan_grid()
        self._set_valid_agent_positions(summary["valid_agent_positions"])
        values = {
            "agent_row": agent_row,
            "agent_col": agent_col,
            "agent_direction": agent_direction,
            "goal_row": summary["goal_pos"][0],
            "goal_col": summary["goal_pos"][1],
            "gap_row": summary["gap_pos"][0],
            "gap_col": summary["gap_pos"][1],
        }
        obs = np.array([values[name] for name in self.feature_names], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted LavaGap observation {obs} is outside {self.observation_space}")
        self.last_observation = obs.copy()
        return obs

    def _scan_grid(self) -> dict:
        grid = self.env.unwrapped.grid
        goal_pos: tuple[int, int] | None = None
        lava_positions: list[tuple[int, int]] = []
        valid_agent_positions: set[tuple[int, int]] = set()
        for col in range(self.width):
            for row in range(self.height):
                obj = grid.get(col, row)
                obj_type = getattr(obj, "type", None) if obj is not None else None
                if obj_type == "goal":
                    goal_pos = (row, col)
                elif obj_type == "lava":
                    lava_positions.append((row, col))
                if obj_type not in {"wall", "lava"}:
                    valid_agent_positions.add((row, col))
        if goal_pos is None:
            raise RuntimeError("Could not find MiniGrid goal object for LavaGap state extraction")
        gap_pos = self._infer_gap(grid, lava_positions, goal_pos)
        return {
            "goal_pos": goal_pos,
            "gap_pos": gap_pos,
            "lava_positions": lava_positions,
            "valid_agent_positions": valid_agent_positions,
        }

    def _infer_gap(self, grid, lava_positions: list[tuple[int, int]], goal_pos: tuple[int, int]) -> tuple[int, int]:
        if not lava_positions:
            return (self.height, self.width)
        row_counts = Counter(row for row, _ in lava_positions)
        col_counts = Counter(col for _, col in lava_positions)
        best_row, row_count = row_counts.most_common(1)[0]
        best_col, col_count = col_counts.most_common(1)[0]
        candidates: list[tuple[int, int]] = []
        if col_count >= row_count:
            for row in range(1, self.height - 1):
                if (row, best_col) in lava_positions:
                    continue
                obj = grid.get(best_col, row)
                if obj is None:
                    candidates.append((row, best_col))
        else:
            for col in range(1, self.width - 1):
                if (best_row, col) in lava_positions:
                    continue
                obj = grid.get(col, best_row)
                if obj is None:
                    candidates.append((best_row, col))
        if not candidates:
            return (self.height, self.width)
        return min(candidates, key=lambda rc: abs(rc[0] - goal_pos[0]) + abs(rc[1] - goal_pos[1]))

    def _symbolic_info(self, obs: np.ndarray, action: int | None, success: bool) -> dict:
        values = {name: int(value) for name, value in zip(self.feature_names, obs)}
        return {
            "symbolic_state": values,
            "agent_pos": (values["agent_row"], values["agent_col"]),
            "agent_dir": values["agent_direction"],
            "goal_pos": (values["goal_row"], values["goal_col"]),
            "gap_pos": (values["gap_row"], values["gap_col"]),
            "action_name": None if action is None else self.action_names[int(action)],
            "mission": getattr(self.env.unwrapped, "mission", ""),
            "success": bool(success),
        }

    def _set_valid_agent_positions(self, positions: set[tuple[int, int]] | None) -> None:
        if positions is None:
            positions = {(row, col) for row in range(self.height) for col in range(self.width)}
        object.__setattr__(self.observation_space, "valid_agent_positions", set(positions))

    def _minigrid_action_names(self) -> list[str]:
        if self.minigrid_action_count == 7:
            return list(self.default_action_names)
        if self.minigrid_action_count == 3:
            return list(self.nav_action_names)
        raise ValueError(f"Unexpected MiniGrid action count {self.minigrid_action_count}; update action mapping")

    def _build_action_map(self, action_subset, all_action_names: list[str]) -> list[int]:
        if action_subset is None:
            action_subset = self.nav_action_names
        name_to_index = {name: idx for idx, name in enumerate(all_action_names)}
        action_map = []
        for action in action_subset:
            if isinstance(action, str):
                if action not in name_to_index:
                    raise ValueError(f"Unknown MiniGrid action name {action!r}")
                idx = name_to_index[action]
            else:
                idx = int(action)
            if idx < 0 or idx >= self.minigrid_action_count:
                raise ValueError(f"MiniGrid action index {idx} outside action space")
            action_map.append(idx)
        if not action_map:
            raise ValueError("action_subset must contain at least one action")
        return action_map

