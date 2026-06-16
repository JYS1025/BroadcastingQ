from __future__ import annotations

import math

import numpy as np

from agents.broadcasting_q import SBQAgent


class TaxiStructuralSBQ(SBQAgent):
    """Taxi SBQ with local taxi-position broadcasting under fixed task context."""

    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        if int(s1[2]) != int(s2[2]) or int(s1[3]) != int(s2[3]):
            return math.inf
        return float(abs(int(s1[0]) - int(s2[0])) + abs(int(s1[1]) - int(s2[1])))

    def get_neighborhood(self, state: int) -> np.ndarray:
        center = self.observation_space.from_index(state)
        radius = int(self.search_radius)
        passenger = int(center[2])
        destination = int(center[3])
        neighbors: list[int] = []
        for row in range(self.observation_space.nvec[0]):
            for col in range(self.observation_space.nvec[1]):
                if abs(row - int(center[0])) + abs(col - int(center[1])) <= radius:
                    obs = np.array([row, col, passenger, destination], dtype=np.int64)
                    neighbors.append(self.observation_space.to_index(obs))
        return np.array(sorted(set(neighbors)), dtype=np.int64)
