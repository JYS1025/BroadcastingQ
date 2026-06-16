from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class ICUSepsisEnv(BaseEnv):
    """Tabular wrapper for the external ICU-Sepsis environment."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_config = config.get("env", {})
        self.env_id = self.env_config.get("env_id", "Sepsis/ICU-Sepsis-v2")

        try:
            import icu_sepsis  # noqa: F401 - registers environments
        except ImportError as exc:
            raise ImportError(
                "ICU-Sepsis requires icu-sepsis==2.0.1 plus its runtime dependencies in the bcrl env."
            ) from exc

        self.env = self._make_env()
        self.num_states = int(getattr(self.env.observation_space, "n", 716))
        self.num_actions = int(getattr(self.env.action_space, "n", 25))
        self.observation_space = MultiDiscreteSpace([self.num_states])
        self.action_space = DiscreteActionSpace(self.num_actions)
        self.action_names = [self._action_name(action) for action in range(self.num_actions)]
        self.trajectory: list[dict] = []
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_obs = np.array([0], dtype=np.int64)
        self.last_info: dict = {}

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        obs = self._convert_obs(raw_obs)
        self.last_obs = obs
        self.last_info = dict(info)
        self.episode_steps = 0
        self.episode_return = 0.0
        self.trajectory = [{"step": 0, "state_id": int(obs[0]), "reward": 0.0, "action": None, "sofa_score": info.get("sofa_score")}]
        out_info = self._make_info(info, obs, action=None, reward=0.0, terminated=False)
        out_info["success"] = False
        return obs, out_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        result = self.env.step(int(action))
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            raw_obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected ICU-Sepsis step result length: {len(result)}")
        obs = self._convert_obs(raw_obs)
        self.last_obs = obs
        self.last_info = dict(info)
        self.episode_steps += 1
        self.episode_return += float(reward)
        info = self._make_info(info, obs, action=int(action), reward=float(reward), terminated=bool(terminated))
        info["episode_steps"] = self.episode_steps
        info["episode_return"] = self.episode_return
        info["success"] = bool(terminated and float(reward) > 0.0)
        self.trajectory.append(
            {
                "step": self.episode_steps,
                "state_id": int(obs[0]),
                "reward": float(reward),
                "action": int(action),
                "sofa_score": info.get("sofa_score"),
            }
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("ICUSepsisEnv supports only mode='rgb_array'")
        width, height = 900, 420
        image = Image.new("RGB", (width, height), (248, 249, 251))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 56), fill=(38, 70, 120))
        draw.text((24, 18), "ICU-Sepsis trajectory", fill=(255, 255, 255))
        draw.text((24, 76), f"step: {self.episode_steps}", fill=(20, 20, 20))
        draw.text((160, 76), f"state: {int(self.last_obs[0])}", fill=(20, 20, 20))
        draw.text((310, 76), f"return: {self.episode_return:.2f}", fill=(20, 20, 20))
        recent = self.trajectory[-18:]
        x0, y0 = 40, 145
        bar_w, gap = 38, 8
        for idx, item in enumerate(recent):
            x = x0 + idx * (bar_w + gap)
            state_id = int(item["state_id"])
            reward = float(item["reward"])
            color = (95, 155, 220) if reward == 0.0 else ((70, 160, 95) if reward > 0 else (210, 70, 70))
            h = 40 + int((state_id % 100) / 100.0 * 140)
            draw.rectangle((x, y0 + 160 - h, x + bar_w, y0 + 160), fill=color)
            draw.text((x, y0 + 168), str(state_id), fill=(30, 30, 30))
        draw.line((40, 330, 860, 330), fill=(180, 185, 195), width=2)
        if self.trajectory:
            last = self.trajectory[-1]
            action = last.get("action")
            action_label = "none" if action is None else self._action_name(int(action))
            draw.text((40, 350), f"last action: {action_label}", fill=(20, 20, 20))
            draw.text((340, 350), f"last reward: {float(last.get('reward', 0.0)):.2f}", fill=(20, 20, 20))
            sofa = last.get("sofa_score")
            draw.text((540, 350), f"SOFA: {sofa if sofa is not None else 'n/a'}", fill=(20, 20, 20))
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _make_env(self):
        kwargs = dict(self.env_config.get("kwargs", {}))
        try:
            import gymnasium as gym

            return gym.make(self.env_id, **kwargs)
        except Exception:
            import gym

            return gym.make(self.env_id, **kwargs)

    def _convert_obs(self, raw_obs) -> np.ndarray:
        if isinstance(raw_obs, (list, tuple, np.ndarray)):
            state_id = int(np.asarray(raw_obs).reshape(-1)[0])
        else:
            state_id = int(raw_obs)
        state_id = int(np.clip(state_id, 0, self.num_states - 1))
        obs = np.array([state_id], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted ICU-Sepsis observation {obs} is outside {self.observation_space}")
        return obs

    def _make_info(self, raw_info: dict, obs: np.ndarray, action: int | None, reward: float, terminated: bool) -> dict:
        raw_info = dict(raw_info or {})
        admissible_actions = raw_info.get("admissible_actions")
        if admissible_actions is not None:
            admissible_actions = [int(action) for action in np.asarray(admissible_actions).reshape(-1)]
        else:
            admissible_actions = list(range(self.action_space.n))
        action_dose_levels = None if action is None else {"iv_fluid": int(action) // 5, "vasopressor": int(action) % 5}
        terminal_outcome = None
        if terminated:
            state_id = int(obs[0])
            if state_id == 714:
                terminal_outcome = "survival"
            elif state_id == 713:
                terminal_outcome = "death"
            else:
                terminal_outcome = "terminal"
        return {
            "symbolic_state": {"state_id": int(obs[0]), "action_names": list(self.action_names)},
            "state_id": int(obs[0]),
            "original_state_id": int(obs[0]),
            "admissible_actions": admissible_actions,
            "sofa_score": raw_info.get("sofa_score"),
            "terminal_outcome": terminal_outcome,
            "action_name": None if action is None else self.action_names[int(action)],
            "action_dose_levels": action_dose_levels,
        }

    @staticmethod
    def _action_name(action: int) -> str:
        return f"iv_fluid_{int(action) // 5}_vasopressor_{int(action) % 5}"
