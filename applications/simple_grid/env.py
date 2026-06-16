from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["simple_grid_sbq"] = "applications.simple_grid.agent:SimpleGridStructuralSBQ"


class SimpleGridEnv(BaseEnv):
    """gym-simplegrid wrapper with a native compatible fallback."""

    action_names = ["up", "down", "left", "right"]
    action_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        self.env_id = str(env_config.get("gym_env_id", "SimpleGrid-v0"))
        self.prefer_external = bool(env_config.get("prefer_external", True))
        self.nrow = int(env_config.get("nrow", 4))
        self.ncol = int(env_config.get("ncol", 5))
        self.map_name = str(env_config.get("map_name", "empty"))
        raw_obstacles = env_config.get("obstacle_map", [])
        self.obstacle_positions = self._build_obstacle_positions(raw_obstacles)
        self.start_pos = tuple(int(v) for v in env_config.get("start_loc", [0, 0]))
        self.goal_locations = self._build_goal_locations(env_config)
        self.goal_mode = str(env_config.get("goal_mode", "fixed"))
        self.goal_id = 0
        self.goal_pos = self.goal_locations[self.goal_id]
        self.random_start = bool(env_config.get("random_start", False))
        self.action_slip_probability = float(env_config.get("action_slip_probability", 0.0))
        self.max_steps = int(env_config.get("max_steps", 100))
        self.invalid_move_reward = float(env_config.get("invalid_move_reward", -1.0))
        self.step_reward = float(env_config.get("step_reward", -0.1))
        self.goal_reward = float(env_config.get("goal_reward", 1.0))
        self.include_goal_id = bool(env_config.get("include_goal_id", self.goal_mode != "fixed" or len(self.goal_locations) > 1))
        self.external_env = None
        self.external_available = False
        self.last_raw_observation: int | None = None
        self.last_action_name: str | None = None
        self.rng = np.random.default_rng()

        if self.prefer_external:
            self.external_env = self._try_make_external(env_config)
            self.external_available = self.external_env is not None
            if self.external_available:
                unwrapped = self.external_env.unwrapped
                self.nrow = int(getattr(unwrapped, "nrow", getattr(unwrapped, "rows", self.nrow)))
                self.ncol = int(getattr(unwrapped, "ncol", getattr(unwrapped, "cols", self.ncol)))

        nvec = [self.nrow, self.ncol]
        if self.include_goal_id:
            nvec.append(len(self.goal_locations))
        self.observation_space = MultiDiscreteSpace(nvec)
        self.action_space = DiscreteActionSpace(4)
        self.valid_positions = {
            (row, col)
            for row in range(self.nrow)
            for col in range(self.ncol)
            if (row, col) not in self.obstacle_positions
        }
        object.__setattr__(self.observation_space, "valid_positions", self.valid_positions)
        self.agent_pos = self.start_pos
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_action_name = None
        self.goal_id = self._sample_goal_id()
        self.goal_pos = self.goal_locations[self.goal_id]

        if self.external_available:
            reset_options = {"start_loc": self.start_pos, "goal_loc": self.goal_pos}
            try:
                raw_obs, info = self.external_env.reset(seed=seed, options=reset_options)
            except TypeError:
                raw_obs, info = self.external_env.reset(seed=seed)
            obs = self._decode_raw_observation(raw_obs)
            self.agent_pos = (int(obs[0]), int(obs[1]))
            wrapped = dict(info)
            wrapped.update(self._info(obs, action=None, success=False))
            return obs, wrapped

        self.agent_pos = self._sample_start_pos() if self.random_start else self.start_pos
        obs = self._obs()
        return obs, self._info(obs, action=None, success=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action = int(action)
        self.last_action_name = self.action_names[action]
        intended_action = action
        slipped = False

        if self.external_available:
            raw_obs, reward, terminated, truncated, info = self.external_env.step(action)
            obs = self._decode_raw_observation(raw_obs)
            self.agent_pos = (int(obs[0]), int(obs[1]))
            self.episode_steps += 1
            self.episode_return += float(reward)
            success = bool((terminated and self.agent_pos == self.goal_pos) or (terminated and float(reward) > 0.0))
            wrapped = dict(info)
            wrapped.update(self._info(obs, action=action, success=success))
            wrapped["episode_steps"] = self.episode_steps
            wrapped["episode_return"] = self.episode_return
            return obs, float(reward), bool(terminated), bool(truncated), wrapped

        if self.action_slip_probability > 0.0 and self.rng.random() < self.action_slip_probability:
            action = int(self.rng.integers(self.action_space.n))
            slipped = action != intended_action
        dr, dc = self.action_delta[action]
        row, col = self.agent_pos
        proposed = (row + dr, col + dc)
        invalid = not self._is_valid(proposed)
        if invalid:
            proposed = self.agent_pos
        self.agent_pos = proposed
        self.episode_steps += 1
        terminated = self.agent_pos == self.goal_pos
        truncated = self.episode_steps >= self.max_steps and not terminated
        if invalid:
            reward = self.invalid_move_reward
        elif terminated:
            reward = self.goal_reward
        else:
            reward = self.step_reward
        self.episode_return += float(reward)
        obs = self._obs()
        info = self._info(obs, action=intended_action, success=bool(terminated))
        info["episode_steps"] = self.episode_steps
        info["episode_return"] = self.episode_return
        info["invalid_move"] = bool(invalid)
        info["executed_action_name"] = self.action_names[int(action)]
        info["action_slipped"] = bool(slipped)
        info["TimeLimit.truncated"] = bool(truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("SimpleGridEnv supports only mode='rgb_array'")
        if self.external_available:
            frame = self.external_env.render()
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
        return self._render_frame()

    def close(self) -> None:
        if self.external_env is not None:
            self.external_env.close()

    def _try_make_external(self, env_config: dict):
        try:
            import gymnasium as gym
            import gym_simplegrid  # noqa: F401
        except ImportError:
            return None

        kwargs = {"obstacle_map": self._external_obstacle_map(self.obstacle_positions)}
        try:
            return gym.make(self.env_id, render_mode="rgb_array", **kwargs)
        except TypeError:
            try:
                return gym.make(self.env_id, render_mode="rgb_array")
            except Exception:
                return None
        except Exception:
            return None

    def _external_obstacle_map(self, obstacle_map) -> list[str]:
        grid = [["0" for _ in range(self.ncol)] for _ in range(self.nrow)]
        for item in obstacle_map:
            row, col = [int(v) for v in item]
            if 0 <= row < self.nrow and 0 <= col < self.ncol:
                grid[row][col] = "1"
        return ["".join(row) for row in grid]

    def _decode_raw_observation(self, raw_obs) -> np.ndarray:
        raw_value = int(np.asarray(raw_obs).item())
        self.last_raw_observation = raw_value
        row = raw_value // self.ncol
        col = raw_value % self.ncol
        obs = np.array([row, col], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"SimpleGrid raw observation {raw_value} decoded to invalid state {obs}")
        return obs

    def _obs(self) -> np.ndarray:
        values = [self.agent_pos[0], self.agent_pos[1]]
        if self.include_goal_id:
            values.append(self.goal_id)
        obs = np.array(values, dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Observation {obs} is outside {self.observation_space}")
        return obs

    def _info(self, obs: np.ndarray, action: int | None, success: bool) -> dict:
        row, col = int(obs[0]), int(obs[1])
        return {
            "external_env_available": bool(self.external_available),
            "raw_observation": self.last_raw_observation,
            "symbolic_state": {
                "agent_row": row,
                "agent_col": col,
                "goal_id": self.goal_id,
                "goal_row": self.goal_pos[0],
                "goal_col": self.goal_pos[1],
            },
            "agent_pos": (row, col),
            "goal_pos": self.goal_pos,
            "goal_id": self.goal_id,
            "action_name": None if action is None else self.action_names[int(action)],
            "success": bool(success),
        }

    def _is_valid(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.nrow and 0 <= col < self.ncol and pos not in self.obstacle_positions

    def _build_obstacle_positions(self, obstacle_map) -> set[tuple[int, int]]:
        if obstacle_map and all(isinstance(row, str) for row in obstacle_map):
            return {
                (row_idx, col_idx)
                for row_idx, row in enumerate(obstacle_map)
                for col_idx, value in enumerate(row)
                if value == "1"
            }
        obstacles = {tuple(int(v) for v in pos) for pos in obstacle_map}
        if self.map_name == "four_rooms" and not obstacles:
            obstacles = self._four_rooms_walls()
        return obstacles

    def _four_rooms_walls(self) -> set[tuple[int, int]]:
        wall_row = self.nrow // 2
        wall_col = self.ncol // 2
        row_gaps = {1, self.ncol - 2}
        col_gaps = {1, self.nrow - 2}
        walls = {(wall_row, col) for col in range(self.ncol) if col not in row_gaps}
        walls.update({(row, wall_col) for row in range(self.nrow) if row not in col_gaps})
        return walls

    def _build_goal_locations(self, env_config: dict) -> list[tuple[int, int]]:
        if "goal_locations" in env_config:
            return [tuple(int(v) for v in pos) for pos in env_config["goal_locations"]]
        goal_mode = str(env_config.get("goal_mode", "fixed"))
        if goal_mode == "random_corners":
            return [(0, 0), (0, self.ncol - 1), (self.nrow - 1, 0), (self.nrow - 1, self.ncol - 1)]
        return [tuple(int(v) for v in env_config.get("goal_loc", [self.nrow - 1, self.ncol - 1]))]

    def _sample_goal_id(self) -> int:
        if self.goal_mode == "random_corners" or len(self.goal_locations) > 1:
            return int(self.rng.integers(len(self.goal_locations)))
        return 0

    def _sample_start_pos(self) -> tuple[int, int]:
        candidates = sorted(pos for pos in self.valid_positions if pos != self.goal_pos)
        if not candidates:
            raise ValueError("No valid random start positions remain after excluding the goal")
        return candidates[int(self.rng.integers(len(candidates)))]

    def _render_frame(self) -> np.ndarray:
        cell = 64
        image = Image.new("RGB", (self.ncol * cell, self.nrow * cell), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for row in range(self.nrow):
            for col in range(self.ncol):
                x0, y0 = col * cell, row * cell
                rect = (x0, y0, x0 + cell, y0 + cell)
                fill = (255, 255, 255)
                if (row, col) in self.obstacle_positions:
                    fill = (30, 30, 30)
                elif (row, col) == self.goal_pos:
                    fill = (72, 187, 120)
                draw.rectangle(rect, fill=fill, outline=(180, 180, 180), width=2)
        ar, ac = self.agent_pos
        pad = 14
        draw.ellipse((ac * cell + pad, ar * cell + pad, (ac + 1) * cell - pad, (ar + 1) * cell - pad), fill=(245, 158, 11))
        return np.asarray(image, dtype=np.uint8)
