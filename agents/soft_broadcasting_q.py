from __future__ import annotations

from collections import defaultdict
import numpy as np

from core.agent_base import BaseAgent, Transition
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace

class SoftSBQAgent(BaseAgent):
    """
    Soft Spatial Broadcasting Q-learning (Soft SBQ).
    
    This agent propagates TD errors to unvisited, similar states using an exponential
    kernel based on a distance metric, but computes updates and acts using a 
    maximum-entropy (Soft Q-learning) framework to encourage diverse discovery.
    """

    def __init__(
        self,
        observation_space: MultiDiscreteSpace,
        action_space: DiscreteActionSpace,
        rng: np.random.Generator,
        gamma: float = 0.99,
        learning_rate: float = 0.1,  
        tau: float = 0.1,
        # SBQ Hyperparameters
        beta_0: float = 1.0,
        p: float = 0.5,
        kernel_lambda: float = 1.0,
        search_radius: float = 2.0,
        alpha_k: float = 0.1,
        use_local_normalization: bool = True,
        **kwargs,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = rng
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.tau = float(tau)
        
        # SBQ Hyperparameters
        self.beta_0 = float(beta_0)
        self.p = float(p)
        self.kernel_lambda = float(kernel_lambda)
        self.search_radius = float(search_radius)
        self.alpha_k = float(alpha_k)
        self.use_local_normalization = bool(use_local_normalization)
        
        self.action_space_size = action_space.n
        
        # Dual Table Initialization
        self.q_global = defaultdict(lambda: np.zeros(self.action_space_size, dtype=np.float32))
        self.q_kernel = defaultdict(float)
        self.n_visits = defaultdict(lambda: np.zeros(self.action_space_size, dtype=np.float32))
        
        # Caches for performance optimization
        self._kernel_cache = {}

    def get_distance(self, state1: int, state2: int) -> float:
        s1_arr = self.observation_space.from_index(state1)
        s2_arr = self.observation_space.from_index(state2)
        return float(np.sum(s1_arr != s2_arr))

    def get_neighborhood(self, state: int) -> np.ndarray:
        s_arr = self.observation_space.from_index(state)
        nvec = self.observation_space.nvec
        dim = len(nvec)
        r = int(self.search_radius)
        
        neighborhood = set()
        
        def generate_neighbors(current_dim: int, diff_count: int, current_arr: np.ndarray):
            if diff_count > r:
                return
            if current_dim == dim:
                idx = self.observation_space.to_index(current_arr)
                neighborhood.add(idx)
                return
                
            generate_neighbors(current_dim + 1, diff_count, current_arr)
            
            if diff_count < r:
                original_val = current_arr[current_dim]
                for v in range(nvec[current_dim]):
                    if v != original_val:
                        current_arr[current_dim] = v
                        generate_neighbors(current_dim + 1, diff_count + 1, current_arr)
                current_arr[current_dim] = original_val
                
        generate_neighbors(0, 0, s_arr.copy())
        return np.array(list(neighborhood), dtype=np.int64)

    def compute_kernel_weights(self, state: int, neighborhood: np.ndarray) -> np.ndarray:
        distances = np.array([self.get_distance(state, x) for x in neighborhood], dtype=np.float32)
        k_weights = np.exp(-self.kernel_lambda * distances)
        if self.use_local_normalization and len(k_weights) > 0:
            k_weights = k_weights / np.sum(k_weights)
        return k_weights

    def get_neighborhood_weights(self, state: int) -> tuple[np.ndarray, np.ndarray]:
        if state in self._kernel_cache:
            return self._kernel_cache[state]
            
        neighborhood = self.get_neighborhood(state)
        weights = self.compute_kernel_weights(state, neighborhood)
        
        self._kernel_cache[state] = (neighborhood, weights)
        return self._kernel_cache[state]

    def get_annealing_factor(self, state: int) -> np.ndarray:
        n_sa = self.n_visits[state]
        return self.beta_0 / ((1.0 + n_sa) ** self.p)

    def get_combined_q(self, state: int) -> np.ndarray:
        beta_t = self.get_annealing_factor(state)
        q_k_arr = np.array([self.q_kernel[(state, a)] for a in range(self.action_space_size)], dtype=np.float32)
        return self.q_global[state] + beta_t * q_k_arr

    def _get_action_probs(self, q_values: np.ndarray) -> np.ndarray:
        q_shifted = (q_values - np.max(q_values)) / self.tau
        exp_q = np.exp(q_shifted)
        return exp_q / np.sum(exp_q)

    def _get_soft_v(self, q_values: np.ndarray) -> float:
        max_q = np.max(q_values)
        q_shifted = (q_values - max_q) / self.tau
        return max_q + self.tau * np.log(np.sum(np.exp(q_shifted)))

    def act(self, obs, explore: bool = True) -> int:
        state_idx = self.observation_space.to_index(obs)
        combined_q = self.get_combined_q(state_idx)
        
        # Softmax sampling to ensure diverse generations
        probs = self._get_action_probs(combined_q)
        return int(self.rng.choice(self.action_space.n, p=probs))

    def update(self, transition: Transition) -> dict:
        state = self.observation_space.to_index(transition.obs)
        next_state = self.observation_space.to_index(transition.next_obs)
        action = transition.action
        reward = float(transition.reward)
        
        self.n_visits[state][action] += 1.0
        
        current_combined_q = self.get_combined_q(state)[action]
        if transition.done:
            bootstrap = 0.0
        else:
            # Soft Value formulation instead of max Q
            bootstrap = self._get_soft_v(self.get_combined_q(next_state))
            
        target = reward + self.gamma * bootstrap
        td_error = target - current_combined_q
        
        self.q_global[state][action] += self.learning_rate * td_error
        
        neighborhood, weights = self.get_neighborhood_weights(state)
        for x, w in zip(neighborhood, weights):
            self.q_kernel[(x, action)] += self.alpha_k * float(w) * td_error
            
        return {
            "td_error": float(td_error),
            "q_value": float(current_combined_q),
            "target": target,
        }

    def save(self, path: str) -> None:
        data = {
            "q_global": dict(self.q_global),
            "q_kernel": dict(self.q_kernel),
            "n_visits": dict(self.n_visits)
        }
        np.save(path, data, allow_pickle=True)

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True).item()
        self.q_global.update(data["q_global"])
        self.q_kernel.update(data["q_kernel"])
        self.n_visits.update(data["n_visits"])
