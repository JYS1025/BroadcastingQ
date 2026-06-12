from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class MiniGridBlockedUnlockPickupVisualizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def save_episode_gif(self, frames, path, fps: int = 4):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, [np.asarray(frame) for frame in frames], duration=1.0 / max(1, fps))

    def save_final_frame(self, frame, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(frame)).save(out)

    def render_policy_rollout(self, env, agent, path, max_steps: int, fps: int = 4):
        obs, _ = env.reset()
        frames = [env.render(mode="rgb_array")]
        for _ in range(int(max_steps)):
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            if terminated or truncated:
                break
        self.save_episode_gif(frames, path, fps=fps)
        return frames

