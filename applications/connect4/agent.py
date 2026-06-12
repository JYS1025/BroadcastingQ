from __future__ import annotations

import numpy as np

from agents.q_learning import QLearningAgent
from agents.broadcasting_q import SBQAgent
from core.agent_base import Transition


def _valid_actions_from_obs(obs) -> np.ndarray:
    board = np.asarray(obs, dtype=np.int64).reshape(6, 7)
    valid = np.flatnonzero(board[0] == 0)
    if valid.size == 0:
        return np.array([0], dtype=np.int64)
    return valid.astype(np.int64)


class Connect4MaskedQLearningAgent(QLearningAgent):
    """Q-learning that explores and acts only in non-full columns."""

    def act(self, obs, explore: bool = True) -> int:
        valid_actions = _valid_actions_from_obs(obs)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        state_idx = self.observation_space.to_index(obs)
        row = self.q_table.row(state_idx)
        valid_values = row[valid_actions]
        max_value = np.max(valid_values)
        candidates = valid_actions[np.flatnonzero(valid_values == max_value)]
        return int(self.rng.choice(candidates))


class Connect4SBQ(SBQAgent):
    """SBQ with app-local Hamming neighborhoods for very large symbolic spaces."""

    def __init__(self, *args, max_neighbors_per_dim: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_neighbors_per_dim = max_neighbors_per_dim
        self._state_arrays: dict[int, np.ndarray] = {}

    def act(self, obs, explore: bool = True) -> int:
        state = self._remember(obs)
        if explore and self.rng.random() < self.epsilon:
            return self.action_space.sample(self.rng)
        return int(np.argmax(self.get_combined_q(state)))

    def update(self, transition: Transition) -> dict:
        self._remember(transition.obs)
        self._remember(transition.next_obs)
        return super().update(transition)

    def get_distance(self, state1: int, state2: int) -> float:
        arr1 = self._state_arrays.get(int(state1))
        arr2 = self._state_arrays.get(int(state2))
        if arr1 is None or arr2 is None:
            return float("inf")
        return float(np.sum(arr1 != arr2))

    def get_neighborhood(self, state: int) -> np.ndarray:
        arr = self._state_arrays.get(int(state))
        if arr is None:
            return [int(state)]
        neighbors = {int(state)}
        radius = max(0, int(self.search_radius))
        if radius <= 0:
            return [int(state)]
        for dim, radix in enumerate(self.observation_space.nvec):
            original = int(arr[dim])
            candidates = [value for value in range(int(radix)) if value != original]
            if self.max_neighbors_per_dim is not None:
                candidates = candidates[: int(self.max_neighbors_per_dim)]
            for value in candidates:
                candidate = arr.copy()
                candidate[dim] = value
                neighbors.add(self._remember(candidate))
        return sorted(neighbors)

    def _remember(self, obs) -> int:
        arr = np.asarray(obs, dtype=np.int64).copy()
        state = self.observation_space.to_index(arr)
        self._state_arrays[int(state)] = arr
        return int(state)


class Connect4MaskedSBQAgent(Connect4SBQ):
    """SBQ that explores and acts only in non-full columns."""

    def act(self, obs, explore: bool = True) -> int:
        state = self._remember(obs)
        valid_actions = _valid_actions_from_obs(obs)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        row = self.get_combined_q(state)
        valid_values = row[valid_actions]
        max_value = np.max(valid_values)
        candidates = valid_actions[np.flatnonzero(valid_values == max_value)]
        return int(self.rng.choice(candidates))
