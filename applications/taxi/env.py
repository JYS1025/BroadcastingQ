from __future__ import annotations

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class TaxiEnv(BaseEnv):
    """Gymnasium Taxi-v4 wrapper with the official decoded factored state."""

    FEATURE_NAMES = [
        "taxi_row",
        "taxi_col",
        "passenger_location",
        "destination",
    ]
    ACTION_NAMES = [
        "south",
        "north",
        "east",
        "west",
        "pickup",
        "dropoff",
    ]
    LOCATION_NAMES = ["R", "G", "Y", "B"]
    LOCATION_COORDS = [(0, 0), (0, 4), (4, 0), (4, 3)]
    LOCATION_COLORS = {
        "R": (220, 38, 38),
        "G": (22, 163, 74),
        "Y": (234, 179, 8),
        "B": (37, 99, 235),
    }
    VERTICAL_WALLS = {
        (0, 1),
        (1, 1),
        (3, 0),
        (3, 2),
        (4, 0),
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        observation_config = dict(config.get("observation", {}))
        if observation_config.get("type", "decoded_taxi") != "decoded_taxi":
            raise ValueError("TaxiEnv supports only observation.type='decoded_taxi'")

        env_id = str(env_config.get("gym_env_id", "Taxi-v4"))
        render_mode = env_config.get("render_mode", "ansi")
        kwargs = dict(env_config.get("kwargs", {}))
        self.env = gym.make(env_id, render_mode=render_mode, **kwargs)

        if not hasattr(self.env.action_space, "n"):
            raise TypeError("TaxiEnv requires a discrete underlying action space")
        action_count = int(self.env.action_space.n)
        if action_count != len(self.ACTION_NAMES):
            raise ValueError(f"Expected Taxi-v4 to expose 6 actions, received {action_count}")

        self.action_names = list(self.ACTION_NAMES)
        self.action_space = DiscreteActionSpace(action_count)
        self.observation_space = MultiDiscreteSpace([5, 5, 5, 4])
        self.last_raw_observation: int | None = None
        self.last_observation: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self.env.reset(seed=seed)
        obs = self._decode_taxi_obs(raw_obs)
        self.last_raw_observation = int(raw_obs)
        self.last_observation = obs.copy()
        wrapped_info = dict(info)
        wrapped_info.update(self._build_info(raw_obs, obs, action_name=None, success=False))
        return obs, wrapped_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Taxi action {action!r}; expected an integer in [0, 6)")

        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._decode_taxi_obs(raw_obs)
        self.last_raw_observation = int(raw_obs)
        self.last_observation = obs.copy()
        success = bool(terminated and float(reward) > 0.0)
        wrapped_info = dict(info)
        wrapped_info.update(
            self._build_info(
                raw_obs,
                obs,
                action_name=self.action_names[int(action)],
                success=success,
            )
        )
        return obs, float(reward), bool(terminated), bool(truncated), wrapped_info

    def render(self, mode: str = "rgb_array"):
        if mode == "rgb_array":
            if self.last_observation is None:
                raise RuntimeError("TaxiEnv.render() called before reset()")
            return self._render_rgb(self.last_observation)
        if mode == "ansi":
            return self.env.render()
        raise ValueError("TaxiEnv supports mode='rgb_array' and mode='ansi'")

    def close(self) -> None:
        self.env.close()

    def _decode_taxi_obs(self, raw_obs) -> np.ndarray:
        raw_value = int(raw_obs)
        decoder = getattr(self.env.unwrapped, "decode", None)
        if callable(decoder):
            decoded = tuple(int(value) for value in decoder(raw_value))
        else:
            value = raw_value
            destination = value % 4
            value //= 4
            passenger_location = value % 5
            value //= 5
            taxi_col = value % 5
            value //= 5
            taxi_row = value
            decoded = (taxi_row, taxi_col, passenger_location, destination)

        obs = np.asarray(decoded, dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(
                "Decoded Taxi observation is outside MultiDiscreteSpace([5, 5, 5, 4]): "
                f"raw={raw_value}, decoded={obs!r}"
            )
        return obs

    def _build_info(
        self,
        raw_obs,
        obs: np.ndarray,
        *,
        action_name: str | None,
        success: bool,
    ) -> dict:
        info = {
            "raw_observation": int(raw_obs),
            "symbolic_state": {
                name: int(value)
                for name, value in zip(self.FEATURE_NAMES, obs)
            },
            "success": bool(success),
        }
        if action_name is not None:
            info["action_name"] = action_name
        return info

    def _render_rgb(self, obs: np.ndarray) -> np.ndarray:
        taxi_row, taxi_col, passenger_location, destination = [int(value) for value in obs]
        cell = 64
        margin = 28
        label_height = 34
        width = margin * 2 + cell * 5
        height = margin * 2 + label_height + cell * 5
        image = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        top = margin + label_height
        left = margin
        draw.rectangle(
            [left - 8, top - 8, left + cell * 5 + 8, top + cell * 5 + 8],
            fill=(226, 232, 240),
            outline=(51, 65, 85),
            width=2,
        )

        for row in range(5):
            for col in range(5):
                x0 = left + col * cell
                y0 = top + row * cell
                draw.rectangle(
                    [x0, y0, x0 + cell, y0 + cell],
                    fill=(255, 255, 255),
                    outline=(203, 213, 225),
                    width=1,
                )

        for row, col in self.VERTICAL_WALLS:
            x = left + (col + 1) * cell
            y0 = top + row * cell
            draw.line([x, y0 + 4, x, y0 + cell - 4], fill=(15, 23, 42), width=5)

        for idx, (row, col) in enumerate(self.LOCATION_COORDS):
            name = self.LOCATION_NAMES[idx]
            color = self.LOCATION_COLORS[name]
            x0 = left + col * cell + 8
            y0 = top + row * cell + 8
            x1 = x0 + cell - 16
            y1 = y0 + cell - 16
            draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=color, outline=(15, 23, 42), width=2)
            self._center_text(draw, [x0, y0, x1, y1], name, font, fill=(255, 255, 255))

        if 0 <= passenger_location < len(self.LOCATION_COORDS):
            passenger_row, passenger_col = self.LOCATION_COORDS[passenger_location]
            px = left + passenger_col * cell + cell - 18
            py = top + passenger_row * cell + 18
            draw.ellipse([px - 9, py - 9, px + 9, py + 9], fill=(168, 85, 247), outline=(88, 28, 135), width=2)
            draw.line([px, py + 9, px, py + 25], fill=(88, 28, 135), width=3)
        else:
            draw.text((left, margin + 14), "Passenger onboard", font=font, fill=(88, 28, 135))

        taxi_x0 = left + taxi_col * cell + 13
        taxi_y0 = top + taxi_row * cell + 22
        taxi_x1 = taxi_x0 + 38
        taxi_y1 = taxi_y0 + 26
        draw.rounded_rectangle(
            [taxi_x0, taxi_y0, taxi_x1, taxi_y1],
            radius=6,
            fill=(250, 204, 21),
            outline=(113, 63, 18),
            width=2,
        )
        draw.rectangle([taxi_x0 + 7, taxi_y0 - 8, taxi_x1 - 7, taxi_y0 + 5], fill=(253, 224, 71), outline=(113, 63, 18))
        draw.ellipse([taxi_x0 + 5, taxi_y1 - 4, taxi_x0 + 15, taxi_y1 + 6], fill=(15, 23, 42))
        draw.ellipse([taxi_x1 - 15, taxi_y1 - 4, taxi_x1 - 5, taxi_y1 + 6], fill=(15, 23, 42))

        dest_name = self.LOCATION_NAMES[destination]
        passenger_text = "on taxi" if passenger_location == 4 else self.LOCATION_NAMES[passenger_location]
        title = f"Taxi-v4 | passenger: {passenger_text} | destination: {dest_name}"
        draw.text((left, margin), title, font=font, fill=(15, 23, 42))
        return np.asarray(image, dtype=np.uint8)

    def _center_text(self, draw: ImageDraw.ImageDraw, box: list[int], text: str, font, fill) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = box[0] + ((box[2] - box[0]) - text_width) / 2
        y = box[1] + ((box[3] - box[1]) - text_height) / 2
        draw.text((x, y), text, font=font, fill=fill)
