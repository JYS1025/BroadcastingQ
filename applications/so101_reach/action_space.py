from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.spaces import DiscreteActionSpace


@dataclass(frozen=True)
class JointStepActionMapper:
    joint_names: list[str]
    step_size: float
    include_noop: bool = True

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ValueError("joint_names must contain at least one joint")
        if float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive")
        object.__setattr__(self, "joint_names", [str(name) for name in self.joint_names])
        object.__setattr__(self, "step_size", float(self.step_size))

    @property
    def action_space(self) -> DiscreteActionSpace:
        return DiscreteActionSpace(len(self._deltas))

    @property
    def action_names(self) -> list[str]:
        names = []
        for delta in self._deltas:
            nz = np.flatnonzero(delta)
            if len(nz) == 0:
                names.append("noop")
            else:
                joint_idx = int(nz[0])
                sign = "+" if delta[joint_idx] > 0.0 else "-"
                names.append(f"{sign}{self.joint_names[joint_idx]}")
        return names

    def to_continuous(self, action: int) -> np.ndarray:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        return self._deltas[int(action)].copy()

    @property
    def _deltas(self) -> list[np.ndarray]:
        deltas: list[np.ndarray] = []
        if self.include_noop:
            deltas.append(np.zeros(len(self.joint_names), dtype=np.float32))
        for joint_idx in range(len(self.joint_names)):
            plus = np.zeros(len(self.joint_names), dtype=np.float32)
            plus[joint_idx] = self.step_size
            minus = np.zeros(len(self.joint_names), dtype=np.float32)
            minus[joint_idx] = -self.step_size
            deltas.extend([plus, minus])
        return deltas



@dataclass(frozen=True)
class CartesianStepActionMapper:
    axes: tuple[str, ...] = ("x", "y", "z")
    step_size: float = 0.02
    include_noop: bool = True

    def __post_init__(self) -> None:
        if tuple(self.axes) != ("x", "y", "z"):
            raise ValueError("CartesianStepActionMapper currently supports axes ('x', 'y', 'z')")
        if float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive")
        object.__setattr__(self, "step_size", float(self.step_size))

    @property
    def action_space(self) -> DiscreteActionSpace:
        return DiscreteActionSpace(len(self._deltas))

    @property
    def action_names(self) -> list[str]:
        names = []
        for delta in self._deltas:
            nz = np.flatnonzero(delta)
            if len(nz) == 0:
                names.append("noop")
            else:
                axis_idx = int(nz[0])
                sign = "+" if delta[axis_idx] > 0.0 else "-"
                names.append(f"{sign}{self.axes[axis_idx]}")
        return names

    def to_continuous(self, action: int) -> np.ndarray:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        return self._deltas[int(action)].copy()

    @property
    def _deltas(self) -> list[np.ndarray]:
        deltas: list[np.ndarray] = []
        if self.include_noop:
            deltas.append(np.zeros(3, dtype=np.float32))
        for axis_idx in range(3):
            plus = np.zeros(3, dtype=np.float32)
            plus[axis_idx] = self.step_size
            minus = np.zeros(3, dtype=np.float32)
            minus[axis_idx] = -self.step_size
            deltas.extend([plus, minus])
        return deltas
