from __future__ import annotations

import math

import numpy as np

from agents.broadcasting_q import SBQAgent


class DynamicObstaclesStructuralSBQ(SBQAgent):
    """Broadcast over local agent pose with fixed goal and obstacle context."""

    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        if not np.array_equal(s1[3:], s2[3:]):
            return math.inf
        pos_dist = abs(int(s1[0]) - int(s2[0])) + abs(int(s1[1]) - int(s2[1]))
        dir_dist = 0 if int(s1[2]) == int(s2[2]) else 1
        return float(pos_dist + dir_dist)

    def get_neighborhood(self, state: int) -> np.ndarray:
        center = self.observation_space.from_index(state)
        radius = int(self.search_radius)
        valid_positions = getattr(self.observation_space, "valid_agent_positions", None)
        fixed_context = center[3:].copy()
        neighbors = []
        direction = int(center[2])
        for row in range(self.observation_space.nvec[0]):
            for col in range(self.observation_space.nvec[1]):
                if valid_positions is not None and (row, col) not in valid_positions:
                    continue
                pos_dist = abs(row - int(center[0])) + abs(col - int(center[1]))
                if pos_dist <= radius:
                    obs = np.concatenate(
                        [np.array([row, col, direction], dtype=np.int64), fixed_context.astype(np.int64)]
                    )
                    neighbors.append(self.observation_space.to_index(obs))
        return np.array(sorted(set(neighbors)), dtype=np.int64)

