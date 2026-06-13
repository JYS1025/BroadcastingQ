from __future__ import annotations

import math

import numpy as np

from agents.broadcasting_q import SBQAgent


class Full4x5StructuralSBQ(SBQAgent):
    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        if len(s1) > 2 and not np.array_equal(s1[2:], s2[2:]):
            return math.inf
        return float(abs(int(s1[0]) - int(s2[0])) + abs(int(s1[1]) - int(s2[1])))

    def get_neighborhood(self, state: int):
        center = self.observation_space.from_index(state)
        radius = int(self.search_radius)
        valid_positions = getattr(self.observation_space, "valid_positions", None)
        fixed_context = center[2:].copy()
        neighbors = []
        for row in range(self.observation_space.nvec[0]):
            for col in range(self.observation_space.nvec[1]):
                if valid_positions is not None and (row, col) not in valid_positions:
                    continue
                dist = abs(row - int(center[0])) + abs(col - int(center[1]))
                if dist <= radius:
                    values = [row, col]
                    values.extend(int(v) for v in fixed_context)
                    neighbors.append(self.observation_space.to_index(np.array(values, dtype=np.int64)))
        return np.array(sorted(set(neighbors)), dtype=np.int64)
