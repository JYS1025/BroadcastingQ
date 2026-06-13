from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class SimpleGridVisualizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def save_episode_gif(self, frames, path, fps: int = 4):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, [self._frame(frame) for frame in frames], duration=1.0 / max(1, int(fps)))

    def save_final_frame(self, frame, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._frame(frame)).save(out)

    @staticmethod
    def _frame(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

