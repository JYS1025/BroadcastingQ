from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_freeway_overlay(frame, overlay_data: dict | None, debug: dict | None = None) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    overlay_data = overlay_data or {}
    debug = debug or {}

    for idx, (y0, y1) in enumerate(overlay_data.get("lane_bands", [])):
        color = (56, 189, 248) if idx == overlay_data.get("selected_lanes", {}).get("current") else (148, 163, 184)
        draw.rectangle([0, y0, image.width - 1, y1], outline=color, width=1)
        draw.text((2, y0 + 1), str(idx), fill=color, font=font)

    for feature in overlay_data.get("lane_features", []):
        for x0, x1, y0, y1 in feature.get("car_segments", []):
            draw.rectangle([x0, y0, x1, y1], outline=(239, 68, 68), width=1)
        if "collision_band" in feature:
            x0, x1, y0, y1 = feature["collision_band"]
            fill = (239, 68, 68) if feature.get("blocked") else (34, 197, 94)
            draw.rectangle([x0, y0, x1, y1], outline=fill, width=1)

    chicken = overlay_data.get("chicken")
    if chicken:
        x0, y0, x1, y1 = chicken["bbox"]
        cx, cy = chicken["center"]
        draw.rectangle([x0, y0, x1, y1], outline=(250, 204, 21), width=2)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(250, 204, 21))

    state = overlay_data.get("state")
    if state is not None:
        draw.rectangle([0, 0, image.width - 1, 16], fill=(15, 23, 42))
        draw.text((2, 2), f"state={list(map(int, state))}", fill=(255, 255, 255), font=font)
    if debug.get("fallback_used"):
        draw.text((2, 18), "FALLBACK", fill=(239, 68, 68), font=font)
    return np.asarray(image, dtype=np.uint8)


class AtariFreewayVisualizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def save_episode_gif(self, frames, path, fps: int = 10):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, self._validate_frames(frames), duration=1.0 / max(1, int(fps)), loop=0)

    def save_final_frame(self, frame, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._validate_frame(frame)).save(out)

    def render_policy_rollout(self, env, agent, path, max_steps: int, fps: int = 10):
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
            raise ValueError("Cannot save an empty Atari Freeway visualization")
        return converted

    @staticmethod
    def _validate_frame(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame shaped (H, W, 3), received {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
