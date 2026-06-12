from agents.broadcasting_q import SBQAgent
import numpy as np

class NetworkRoutingSBQ(SBQAgent):
    """
    A lightweight wrapper over SBQAgent that delegates O(1) distance 
    and neighborhood generation to the environment's specialized observation space APIs.
    """
    def get_distance(self, state1: int, state2: int) -> float:
        return self.observation_space.get_distance_api(state1, state2)
        
    def get_neighborhood(self, state: int) -> np.ndarray:
        return self.observation_space.get_neighborhood_api(state, search_radius=int(self.search_radius))
