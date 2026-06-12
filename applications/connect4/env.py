from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["connect4_sbq"] = "applications.connect4.agent:Connect4SBQ"
AGENT_REGISTRY["connect4_q_learning_masked"] = "applications.connect4.agent:Connect4MaskedQLearningAgent"
AGENT_REGISTRY["connect4_sbq_masked"] = "applications.connect4.agent:Connect4MaskedSBQAgent"


class Connect4Env(BaseEnv):
    """Wrapper around gymnasium-connect-four's ConnectFourEnv."""

    action_names = [f"drop_col_{idx}" for idx in range(7)]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_config = config.get("env", {})

        try:
            from connect_four_gymnasium import ConnectFourEnv as RawConnectFourEnv
            from connect_four_gymnasium import players
        except ImportError as exc:
            raise ImportError(
                "Connect4 requires gymnasium-connect-four==1.3.5. Install it in the bcrl conda env."
            ) from exc

        opponent_name = str(self.env_config.get("opponent", "BabySmarterPlayer"))
        opponent = None
        if opponent_name.lower() not in {"none", "null"}:
            try:
                opponent_cls = getattr(players, opponent_name)
            except AttributeError as exc:
                raise ValueError(f"Unknown connect4 opponent: {opponent_name}") from exc
            opponent = opponent_cls()

        self.env = RawConnectFourEnv(
            opponent=opponent,
            render_mode=self.env_config.get("render_mode", "rgb_array"),
            first_player=int(self.env_config.get("first_player", 1)),
        )
        self.observation_space = MultiDiscreteSpace([3] * 42)
        self.action_space = DiscreteActionSpace(7)
        self.last_board = np.zeros((6, 7), dtype=np.int8)
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(int(seed))
        board, info = self.env.reset(seed=seed)
        self.last_board = np.asarray(board, dtype=np.int8)
        self.episode_steps = 0
        self.episode_return = 0.0
        obs = self._convert_board(self.last_board)
        info = dict(info)
        info.update(self._info(obs))
        info["success"] = False
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        board, reward, terminated, truncated, info = self.env.step(int(action))
        self.last_board = np.asarray(board, dtype=np.int8)
        self.episode_steps += 1
        self.episode_return += float(reward)
        obs = self._convert_board(self.last_board)
        info = dict(info)
        info.update(self._info(obs))
        info["action_name"] = self.action_names[int(action)]
        info["episode_steps"] = self.episode_steps
        info["episode_return"] = self.episode_return
        info["success"] = bool(terminated and float(reward) > 0.0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Connect4Env supports only mode='rgb_array'")
        try:
            frame = self.env.render()
        except Exception:
            frame = None
        if frame is None:
            frame = self._render_board_fallback()
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _convert_board(self, board: np.ndarray) -> np.ndarray:
        arr = np.asarray(board, dtype=np.int64)
        obs = np.zeros_like(arr, dtype=np.int64)
        obs[arr == 1] = 1
        obs[arr == -1] = 2
        obs = obs.reshape(-1)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted Connect4 observation {obs} is outside {self.observation_space}")
        return obs

    def _info(self, obs: np.ndarray) -> dict:
        valid_actions = [int(action) for action in self.env.get_valid_actions()]
        return {
            "symbolic_state": {"board": obs.reshape(6, 7).tolist(), "action_names": list(self.action_names)},
            "valid_actions": valid_actions,
            "board": obs.reshape(6, 7).tolist(),
        }

    def _render_board_fallback(self) -> np.ndarray:
        cell = 64
        width = 7 * cell
        height = 6 * cell + 42
        image = Image.new("RGB", (width, height), (31, 80, 180))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 42), fill=(20, 45, 115))
        draw.text((12, 13), "Connect4: red=current/agent perspective, yellow=opponent", fill=(255, 255, 255))
        colors = {0: (245, 245, 245), 1: (230, 65, 65), -1: (245, 205, 55)}
        for row in range(6):
            for col in range(7):
                x0 = col * cell + 8
                y0 = row * cell + 50
                x1 = (col + 1) * cell - 8
                y1 = (row + 1) * cell + 34
                draw.ellipse((x0, y0, x1, y1), fill=colors[int(self.last_board[row, col])])
        return np.asarray(image, dtype=np.uint8)
