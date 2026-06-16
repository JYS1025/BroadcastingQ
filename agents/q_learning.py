from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.q_table import QTable
from core.agent_base import BaseAgent, Transition
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        observation_space: MultiDiscreteSpace,
        action_space: DiscreteActionSpace,
        rng: np.random.Generator,
        gamma: float = 0.99,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        **kwargs,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = rng
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.q_table = QTable(observation_space, action_space)

    def act(self, obs, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.action_space.sample(self.rng)
        state_idx = self.observation_space.to_index(obs)
        return int(np.argmax(self.q_table.row(state_idx)))

    def update(self, transition: Transition) -> dict:
        state = self.observation_space.to_index(transition.obs)
        next_state = self.observation_space.to_index(transition.next_obs)
        current = self.q_table.get(state, transition.action)
        bootstrap = 0.0 if transition.terminated else float(np.max(self.q_table.row(next_state)))
        target = float(transition.reward + self.gamma * bootstrap)
        td_error = target - current
        q_value = self.q_table.add(state, transition.action, self.learning_rate * td_error)
        return {
            "td_error": float(td_error),
            "q_value": q_value,
            "target": target,
        }

    def save(self, path: str) -> None:
        self.q_table.save(Path(path))

    def load(self, path: str) -> None:
        self.q_table.load(Path(path))
