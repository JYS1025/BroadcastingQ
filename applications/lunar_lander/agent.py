from __future__ import annotations

import math
from itertools import product

import numpy as np

from agents.broadcasting_q import SBQAgent


class LunarLanderStructuralSBQ(SBQAgent):
    """LunarLander SBQ with ordinal continuous bins and fixed contact phase."""

    continuous_weights = np.array([0.5, 0.7, 0.5, 0.7, 1.0, 0.5], dtype=np.float64)

    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        self._validate_lunar_space(s1)
        if int(s1[6]) != int(s2[6]) or int(s1[7]) != int(s2[7]):
            return math.inf
        diffs = np.abs(s1[:6].astype(np.int64) - s2[:6].astype(np.int64))
        return float(np.dot(self.continuous_weights, diffs))

    def get_neighborhood(self, state: int) -> np.ndarray:
        center = self.observation_space.from_index(state)
        self._validate_lunar_space(center)
        radius = float(self.search_radius)
        offset_ranges = [
            range(-int(np.floor(radius / weight)), int(np.floor(radius / weight)) + 1)
            for weight in self.continuous_weights
        ]
        neighbors: set[int] = set()
        for offsets in product(*offset_ranges):
            candidate = center.copy()
            for idx, offset in enumerate(offsets):
                candidate[idx] = np.clip(
                    int(candidate[idx]) + int(offset),
                    0,
                    self.observation_space.nvec[idx] - 1,
                )
            candidate_idx = self.observation_space.to_index(candidate)
            if self.get_distance(state, candidate_idx) <= radius + 1e-9:
                neighbors.add(candidate_idx)
        return np.array(sorted(neighbors), dtype=np.int64)

    def _validate_lunar_space(self, obs: np.ndarray) -> None:
        if len(self.observation_space.nvec) != 8 or obs.shape != (8,):
            raise ValueError(
                "LunarLanderStructuralSBQ requires observations shaped "
                "[x, y, vx, vy, angle, angular_velocity, left_contact, right_contact]"
            )
