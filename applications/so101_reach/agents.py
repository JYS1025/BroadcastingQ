from __future__ import annotations

from itertools import product

import numpy as np

from agents.broadcasting_q import SBQAgent


class EuclideanSBQAgent(SBQAgent):
    """SBQ variant for discretized continuous states.

    The base SBQ agent is intentionally left unchanged. This subclass uses
    app-provided bin centers to measure physical closeness between binned states.
    """

    def __init__(
        self,
        *args,
        distance_centers: list[np.ndarray],
        max_neighbor_delta: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.distance_centers = [np.asarray(centers, dtype=np.float32) for centers in distance_centers]
        self.max_neighbor_delta = int(max_neighbor_delta)
        if self.max_neighbor_delta < 0:
            raise ValueError("max_neighbor_delta must be non-negative")
        if len(self.distance_centers) != len(self.observation_space.nvec):
            raise ValueError("distance_centers length must match observation dimensions")
        # Per-axis "zero-error" bin: the bin whose center is closest to 0. The optimal
        # action along an axis flips sign across this bin (positive error -> move negative,
        # negative error -> move positive), so broadcasting must not cross it.
        self.center_bins = [int(np.argmin(np.abs(centers))) for centers in self.distance_centers]

    def get_distance(self, state1: int, state2: int) -> float:
        s1_arr = self.observation_space.from_index(state1)
        s2_arr = self.observation_space.from_index(state2)
        p1 = np.asarray([centers[int(idx)] for centers, idx in zip(self.distance_centers, s1_arr)], dtype=np.float32)
        p2 = np.asarray([centers[int(idx)] for centers, idx in zip(self.distance_centers, s2_arr)], dtype=np.float32)
        return float(np.linalg.norm(p1 - p2))

    def get_neighborhood(self, state: int) -> np.ndarray:
        state_arr = self.observation_space.from_index(state)
        ranges = []
        for axis, (value, n_bins) in enumerate(zip(state_arr, self.observation_space.nvec)):
            center = self.center_bins[axis]
            src_side = int(np.sign(int(value) - center))
            low = max(0, int(value) - self.max_neighbor_delta)
            high = min(int(n_bins) - 1, int(value) + self.max_neighbor_delta)
            # Sign-aware: keep only candidate bins on the same side of the zero-error bin
            # as the source (the center bin itself, side 0, is always allowed). This stops
            # a TD error learned for one sign of the error being broadcast to states where
            # the opposite action is correct.
            allowed = [
                v
                for v in range(low, high + 1)
                if src_side == 0 or int(np.sign(v - center)) in (0, src_side)
            ]
            ranges.append(allowed)

        neighborhood = []
        for candidate in product(*ranges):
            idx = self.observation_space.to_index(np.asarray(candidate, dtype=np.int64))
            if self.get_distance(state, idx) <= self.search_radius:
                neighborhood.append(idx)
        return np.asarray(neighborhood, dtype=np.int64)

