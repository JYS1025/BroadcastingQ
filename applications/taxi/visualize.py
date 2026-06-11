from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class TaxiVisualizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def save_episode_gif(self, frames, path, fps: int = 4):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rgb_frames = self._validate_frames(frames)
        imageio.mimsave(out, rgb_frames, duration=1.0 / max(1, int(fps)), loop=0)

    def save_final_frame(self, frame, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._validate_frame(frame)).save(out)

    def render_policy_rollout(self, env, agent, path, max_steps: int, fps: int = 4):
        obs, _ = env.reset()
        frames = [env.render(mode="rgb_array")]
        for _ in range(int(max_steps)):
            action = agent.act(obs, explore=False)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            if terminated or truncated:
                break
        self.save_episode_gif(frames, path, fps=fps)
        return frames

    def _validate_frames(self, frames) -> list[np.ndarray]:
        converted = [self._validate_frame(frame) for frame in frames]
        if not converted:
            raise ValueError("Cannot save an empty Taxi visualization")
        return converted

    @staticmethod
    def _validate_frame(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected an RGB frame shaped (H, W, 3), received {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
