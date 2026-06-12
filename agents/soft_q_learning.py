from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.q_table import QTable
from core.agent_base import BaseAgent, Transition
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class SoftQLearningAgent(BaseAgent):
    def __init__(
        self,
        observation_space: MultiDiscreteSpace,
        action_space: DiscreteActionSpace,
        rng: np.random.Generator,
        gamma: float = 0.99,
        learning_rate: float = 0.1,
        tau: float = 0.1,
        **kwargs,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = rng
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.tau = float(tau)
        self.q_table = QTable(observation_space, action_space)

    def _get_action_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities over actions using log-sum-exp trick for stability."""
        q_shifted = (q_values - np.max(q_values)) / self.tau
        exp_q = np.exp(q_shifted)
        return exp_q / np.sum(exp_q)

    def _get_soft_v(self, q_values: np.ndarray) -> float:
        """Returns the soft value function V(s) = tau * log sum exp(Q(s, a)/tau)"""
        max_q = np.max(q_values)
        q_shifted = (q_values - max_q) / self.tau
        return max_q + self.tau * np.log(np.sum(np.exp(q_shifted)))

    def act(self, obs, explore: bool = True) -> int:
        state_idx = self.observation_space.to_index(obs)
        q_values = self.q_table.values[state_idx]
        
        # Soft Q-learning uses softmax policy for both exploration and evaluation to ensure diversity
        probs = self._get_action_probs(q_values)
        return int(self.rng.choice(self.action_space.n, p=probs))

    def update(self, transition: Transition) -> dict:
        state = self.observation_space.to_index(transition.obs)
        next_state = self.observation_space.to_index(transition.next_obs)
        current = float(self.q_table.values[state, transition.action])
        
        if transition.done:
            bootstrap = 0.0
        else:
            bootstrap = self._get_soft_v(self.q_table.values[next_state])
            
        target = float(transition.reward + self.gamma * bootstrap)
        td_error = target - current
        self.q_table.values[state, transition.action] += self.learning_rate * td_error
        
        return {
            "td_error": float(td_error),
            "q_value": float(self.q_table.values[state, transition.action]),
            "target": target,
        }

    def save(self, path: str) -> None:
        self.q_table.save(Path(path))

    def load(self, path: str) -> None:
        self.q_table.load(Path(path))
