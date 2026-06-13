from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["gridworld_full_4x5_sbq"] = "applications.gridworld_full_4x5.agent:Full4x5StructuralSBQ"


class GridworldFull4x5Env(BaseEnv):
    action_names = ["up", "down", "left", "right"]
    action_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = config.get("env", {})
        self.height = int(env_config.get("height", 4))
        self.width = int(env_config.get("width", 5))
        self.start_pos = tuple(int(v) for v in env_config.get("start_pos", [self.height - 1, 0]))
        self.goal_locations = self._build_goal_locations(env_config)
        self.goal_mode = str(env_config.get("goal_mode", "fixed"))
        self.goal_id = 0
        self.goal_pos = self.goal_locations[self.goal_id]
        self.random_start = bool(env_config.get("random_start", False))
        self.wall_positions = {tuple(int(v) for v in wall) for wall in env_config.get("walls", [])}
        self.max_steps = int(env_config.get("max_steps", 100))
        self.step_reward = float(env_config.get("step_reward", env_config.get("step_penalty", -1.0)))
        self.invalid_move_reward = float(env_config.get("invalid_move_reward", self.step_reward))
        self.goal_reward = float(env_config.get("goal_reward", 1.0))
        self.include_goal_id = bool(env_config.get("include_goal_id", self.goal_mode != "fixed" or len(self.goal_locations) > 1))
        nvec = [self.height, self.width]
        if self.include_goal_id:
            nvec.append(len(self.goal_locations))
        self.observation_space = MultiDiscreteSpace(nvec)
        self.action_space = DiscreteActionSpace(4)
        self.valid_positions = {
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.wall_positions
        }
        object.__setattr__(self.observation_space, "valid_positions", self.valid_positions)
        self.rng = np.random.default_rng()
        self.agent_pos = self.start_pos
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.goal_id = self._sample_goal_id()
        self.goal_pos = self.goal_locations[self.goal_id]
        self.agent_pos = self._sample_start_pos() if self.random_start else self.start_pos
        self.episode_steps = 0
        self.episode_return = 0.0
        obs = self._obs()
        return obs, self._info(obs, action=None, success=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        dr, dc = self.action_delta[int(action)]
        row, col = self.agent_pos
        next_pos = (row + dr, col + dc)
        invalid = not self._is_valid(next_pos)
        if invalid:
            next_pos = self.agent_pos
        self.agent_pos = next_pos
        self.episode_steps += 1
        terminated = self.agent_pos == self.goal_pos
        truncated = self.episode_steps >= self.max_steps and not terminated
        if terminated:
            reward = self.goal_reward
        elif invalid:
            reward = self.invalid_move_reward
        else:
            reward = self.step_reward
        self.episode_return += float(reward)
        obs = self._obs()
        info = self._info(obs, action=int(action), success=bool(terminated))
        info["episode_steps"] = self.episode_steps
        info["episode_return"] = self.episode_return
        info["invalid_move"] = bool(invalid)
        info["TimeLimit.truncated"] = bool(truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("GridworldFull4x5Env supports only mode='rgb_array'")
        return self._render_frame()

    def close(self) -> None:
        pass

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
        return 0 <= row < self.height and 0 <= col < self.width and pos not in self.wall_positions

    def _build_goal_locations(self, env_config: dict) -> list[tuple[int, int]]:
        if "goal_locations" in env_config:
            return [tuple(int(v) for v in pos) for pos in env_config["goal_locations"]]
        goal_mode = str(env_config.get("goal_mode", "fixed"))
        if goal_mode == "random_corners":
            return [(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)]
        return [tuple(int(v) for v in env_config.get("goal_pos", [0, self.width - 1]))]

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
        image = Image.new("RGB", (self.width * cell, self.height * cell), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for row in range(self.height):
            for col in range(self.width):
                x0, y0 = col * cell, row * cell
                rect = (x0, y0, x0 + cell, y0 + cell)
                fill = (255, 255, 255)
                if (row, col) in self.wall_positions:
                    fill = (30, 30, 30)
                elif (row, col) == self.goal_pos:
                    fill = (68, 180, 93)
                draw.rectangle(rect, fill=fill, outline=(180, 180, 180), width=2)
        ar, ac = self.agent_pos
        pad = 14
        draw.ellipse((ac * cell + pad, ar * cell + pad, (ac + 1) * cell - pad, (ar + 1) * cell - pad), fill=(55, 112, 220))
        return np.asarray(image, dtype=np.uint8)
