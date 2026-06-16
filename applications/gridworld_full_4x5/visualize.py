from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class GridworldFull4x5Visualizer:
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

