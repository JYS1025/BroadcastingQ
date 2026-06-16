from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.spaces import MultiDiscreteSpace


@dataclass(frozen=True)
class FeatureBinSpec:
    name: str
    low: float
    high: float
    edges: tuple[float, ...]
    distance_scale: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be smaller than high")
        if self.distance_scale <= 0.0:
            raise ValueError(f"{self.name}: distance_scale must be positive")
        if any(left >= right for left, right in zip(self.edges, self.edges[1:])):
            raise ValueError(f"{self.name}: edges must be strictly increasing")
        if self.edges and (self.edges[0] <= self.low or self.edges[-1] >= self.high):
            raise ValueError(f"{self.name}: edges must be inside (low, high)")

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    @property
    def centers(self) -> np.ndarray:
        bounds = np.asarray((self.low, *self.edges, self.high), dtype=np.float32)
        return (bounds[:-1] + bounds[1:]) / 2.0

    @property
    def distance_centers(self) -> np.ndarray:
        return self.centers / float(self.distance_scale)

    def encode(self, value: float) -> int:
        clipped = float(np.clip(value, self.low, self.high))
        return int(np.digitize(clipped, self.edges, right=False))


class StateDiscretizer:
    def __init__(self, specs: list[FeatureBinSpec]) -> None:
        if not specs:
            raise ValueError("StateDiscretizer needs at least one feature spec")
        self.specs = specs
        self.observation_space = MultiDiscreteSpace([spec.n_bins for spec in specs])
        self.feature_names = [spec.name for spec in specs]

    @classmethod
    def from_config(cls, config: dict[str, Any], joint_names: list[str]) -> "StateDiscretizer":
        obs_cfg = config.get("observation", {})
        xyz_cfg = obs_cfg.get("position_error", {})
        joint_cfg = obs_cfg.get("joint_position", {})

        xyz_edges = tuple(float(v) for v in xyz_cfg.get(
            "edges",
            [-0.20, -0.10, -0.05, -0.025, -0.01, 0.01, 0.025, 0.05, 0.10, 0.20],
        ))
        xyz_low = float(xyz_cfg.get("low", -0.30))
        xyz_high = float(xyz_cfg.get("high", 0.30))
        xyz_scale = _as_len3(xyz_cfg.get("distance_scale", [0.05, 0.05, 0.05]))

        specs = [
            FeatureBinSpec(f"target_error_{axis}", xyz_low, xyz_high, xyz_edges, xyz_scale[i])
            for i, axis in enumerate(("x", "y", "z"))
        ]

        joint_bins = int(joint_cfg.get("bins", 7))
        if joint_bins < 2:
            raise ValueError("joint_position.bins must be at least 2")
        default_limits = {
            "shoulder_pan": (-1.91986, 1.91986),
            "shoulder_lift": (-1.74533, 1.74533),
            "elbow_flex": (-1.69, 1.69),
            "wrist_flex": (-1.65806, 1.65806),
            "wrist_roll": (-2.74385, 2.84121),
        }
        limits = joint_cfg.get("limits", {})
        joint_scale = joint_cfg.get("distance_scale", 0.50)
        for name in joint_names:
            low, high = limits.get(name, default_limits.get(name, (-3.14, 3.14)))
            edges = tuple(np.linspace(float(low), float(high), joint_bins + 1)[1:-1])
            specs.append(FeatureBinSpec(f"joint_{name}", float(low), float(high), edges, float(joint_scale)))

        return cls(specs)

    @property
    def distance_centers(self) -> list[np.ndarray]:
        return [spec.distance_centers for spec in self.specs]

    def encode(self, target_error_xyz, joint_positions) -> np.ndarray:
        target_error_xyz = np.asarray(target_error_xyz, dtype=np.float32).reshape(3)
        joint_positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        values = np.concatenate([target_error_xyz, joint_positions])
        if len(values) != len(self.specs):
            raise ValueError(f"Expected {len(self.specs)} values, got {len(values)}")
        return np.asarray([spec.encode(float(value)) for spec, value in zip(self.specs, values)], dtype=np.int64)

    def describe_bins(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "low": spec.low,
                "high": spec.high,
                "edges": list(spec.edges),
                "centers": spec.centers.tolist(),
                "distance_scale": spec.distance_scale,
            }
            for spec in self.specs
        ]


def _as_len3(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    out = [float(v) for v in value]
    if len(out) != 3:
        raise ValueError("Expected a scalar or three values")
    return out

