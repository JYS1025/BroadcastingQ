from __future__ import annotations

import math
from collections import deque

import numpy as np

from agents.broadcasting_q import SBQAgent


class SokobanStructuralSBQ(SBQAgent):
    """Sokoban SBQ that generalizes over nearby player positions only."""

    def get_distance(self, state1: int, state2: int) -> float:
        s1 = self.observation_space.from_index(state1)
        s2 = self.observation_space.from_index(state2)
        if s1.shape[0] < 2:
            raise ValueError("SokobanStructuralSBQ requires player plus at least one stone factor")
        if not np.array_equal(s1[1:], s2[1:]):
            return math.inf
        return float(self._player_graph_distance(int(s1[0]), int(s2[0])))

    def get_neighborhood(self, state: int) -> np.ndarray:
        center = self.observation_space.from_index(state)
        radius = int(self.search_radius)
        player_positions = self._player_positions_within_radius(int(center[0]), radius)
        neighbors: set[int] = set()
        for player_position in player_positions:
            candidate = center.copy()
            candidate[0] = int(player_position)
            if self.observation_space.contains(candidate):
                neighbors.add(self.observation_space.to_index(candidate))
        return np.array(sorted(neighbors), dtype=np.int64)

    def _adjacency(self) -> tuple[tuple[int, ...], ...]:
        return tuple(getattr(self.observation_space, "location_adjacency", ()))

    def _player_positions_within_radius(self, start: int, radius: int) -> set[int]:
        adjacency = self._adjacency()
        if not adjacency or start < 0 or start >= len(adjacency):
            return {int(start)}
        seen = {int(start)}
        queue: deque[tuple[int, int]] = deque([(int(start), 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= radius:
                continue
            for nxt in adjacency[node]:
                if nxt in seen:
                    continue
                seen.add(int(nxt))
                queue.append((int(nxt), dist + 1))
        return seen

    def _player_graph_distance(self, start: int, goal: int) -> float:
        if start == goal:
            return 0.0
        adjacency = self._adjacency()
        if not adjacency or start < 0 or goal < 0 or start >= len(adjacency) or goal >= len(adjacency):
            return math.inf
        seen = {int(start)}
        queue: deque[tuple[int, int]] = deque([(int(start), 0)])
        while queue:
            node, dist = queue.popleft()
            for nxt in adjacency[node]:
                if nxt == goal:
                    return float(dist + 1)
                if nxt not in seen:
                    seen.add(int(nxt))
                    queue.append((int(nxt), dist + 1))
        return math.inf
