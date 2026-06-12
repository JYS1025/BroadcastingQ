from __future__ import annotations

from itertools import product

import numpy as np

from agents.broadcasting_q import SBQAgent


class AcrobotStructuralSBQ(SBQAgent):
    """Acrobot SBQ with cyclic angle bins and ordinal velocity bins."""

    weights = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float64)

    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        self._validate_theta_space(s1)
        theta1 = self._circular_distance(int(s1[0]), int(s2[0]), self.observation_space.nvec[0])
        theta2 = self._circular_distance(int(s1[1]), int(s2[1]), self.observation_space.nvec[1])
        vel1 = abs(int(s1[2]) - int(s2[2]))
        vel2 = abs(int(s1[3]) - int(s2[3]))
        distances = np.array([theta1, theta2, vel1, vel2], dtype=np.float64)
        return float(np.dot(self.weights, distances))

    def get_neighborhood(self, state: int) -> np.ndarray:
        center = self.observation_space.from_index(state)
        self._validate_theta_space(center)
        radius = float(self.search_radius)
        int_radius = int(np.ceil(radius))
        neighbors: set[int] = set()
        for offsets in product(range(-int_radius, int_radius + 1), repeat=4):
            candidate = center.copy()
            candidate[0] = (int(candidate[0]) + offsets[0]) % self.observation_space.nvec[0]
            candidate[1] = (int(candidate[1]) + offsets[1]) % self.observation_space.nvec[1]
            candidate[2] = np.clip(int(candidate[2]) + offsets[2], 0, self.observation_space.nvec[2] - 1)
            candidate[3] = np.clip(int(candidate[3]) + offsets[3], 0, self.observation_space.nvec[3] - 1)
            if self.get_distance(state, self.observation_space.to_index(candidate)) <= radius + 1e-9:
                neighbors.add(self.observation_space.to_index(candidate))
        return np.array(sorted(neighbors), dtype=np.int64)

    @staticmethod
    def _circular_distance(i: int, j: int, n: int) -> int:
        diff = abs(i - j)
        return int(min(diff, n - diff))

    def _validate_theta_space(self, obs: np.ndarray) -> None:
        if len(self.observation_space.nvec) != 4 or obs.shape != (4,):
            raise ValueError(
                "AcrobotStructuralSBQ requires theta_binned observations shaped "
                "[theta1, theta2, dtheta1, dtheta2]"
            )
