from __future__ import annotations

import numpy as np
import gymnasium as gym

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace
from core.trainer import AGENT_REGISTRY

# Dynamically register our optimized SBQ agent so we don't need to modify external code
AGENT_REGISTRY["network_routing_sbq"] = "applications.network_routing.agent:NetworkRoutingSBQ"

class NetworkRoutingEnv(BaseEnv):
    """
    Network Routing Environment with State Bit-Compression.
    Optimized natively for O(1) Distance calculations and Neighborhood generation in SBQ.
    """
    def __init__(self, config: dict):
        self.config = config
        env_config = config.get("env", {})
        
        # Total number of nodes in the network
        self.total_nodes = env_config.get("total_nodes", 16)
        
        # Number of neighbors for the current routing decision
        self.N = env_config.get("N", 8)
        
        if self.N > 8:
            raise ValueError("Maximum 8 neighbors supported for 32-bit integer state compression.")
            
        # Network Topology Matrix D (total_nodes x total_nodes)
        D_config = env_config.get("D", None)
        topology_type = env_config.get("topology_type", "scale_free")
        
        if D_config is not None:
            self.D = np.array(D_config, dtype=np.int64)
        else:
            from .topology import generate_topology
            self.D = generate_topology(topology_type, self.total_nodes, self.N)
        
        # Observation Space: Single 32-bit Integer
        self.observation_space = gym.spaces.Discrete(1 << 32)
        self.observation_space.get_distance_api = self.get_distance_api
        self.observation_space.get_neighborhood_api = self.get_neighborhood_api
        self.observation_space.to_index = lambda x: int(x)
        self.observation_space.from_index = lambda x: int(x)
        
        # Monkey-patch flatten and flat_dim for DQNAgent compatibility
        def flatten_api(x):
            arr = np.zeros(32, dtype=np.float32)
            if hasattr(x, "ndim") and x.ndim > 0:
                val = int(x[0])
            else:
                val = int(x.item()) if hasattr(x, "item") else int(x)
            for i in range(32):
                arr[i] = (val >> i) & 1
            return arr
        self.observation_space.flatten = flatten_api
        self.observation_space.flat_dim = 32
        self.observation_space.nvec = [2] * 32  # ReplayBuffer uses len(nvec)
        
        # Action Space: Discrete(N)
        self.action_space = DiscreteActionSpace(self.N)
        
        self.reset()
        
    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        if seed is not None:
            np.random.seed(seed)
            
        self.current_node = np.random.randint(0, self.total_nodes)
        self.destination_id = np.random.randint(0, self.total_nodes)
        while self.destination_id == self.current_node:
            self.destination_id = np.random.randint(0, self.total_nodes)
        self.steps = 0
        self.last_action = None
        self.last_reward = None
        
        # Initialize global stateful queues (0 to 3) for all nodes. Start somewhat light.
        self.global_queues = np.random.randint(0, 2, size=self.total_nodes)
        
        # Update neighbors and encode initial state
        self._update_neighbors()
        state = self._encode_state(self.destination_id, self.queues)
        return state, self._info()
        
    def _update_neighbors(self):
        # Find neighbors
        neighbors = np.where(self.D[self.current_node] == 1)[0]
        
        # Pad or truncate to N neighbors
        self.neighbors = []
        for i in range(self.N):
            if i < len(neighbors):
                self.neighbors.append(neighbors[i])
            else:
                self.neighbors.append(-1) # Disconnected
                
        # Read from stateful global queues
        self.queues = []
        for n in self.neighbors:
            if n == -1:
                self.queues.append(3) # Treat disconnected as saturated to prevent routing
            else:
                self.queues.append(int(self.global_queues[n]))
                
    def _encode_state(self, destination_id: int, queues: list[int]) -> int:
        state = int(destination_id) << 24
        for i in range(self.N):
            q = queues[i]
            if q == 0:
                code = 0b000
            elif q == 1:
                code = 0b100
            elif q == 2:
                code = 0b110
            else:
                code = 0b111
            state |= (code << (i * 3))
        return state
        
    def get_valid_actions(self) -> list[int]:
        valid_actions = []
        for i in range(self.N):
            if self.neighbors[i] != -1:
                valid_actions.append(i)
        return valid_actions
        
    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        action = int(action)
        valid_actions = self.get_valid_actions()
        
        if action not in valid_actions:
            return self._encode_state(self.destination_id, self.queues), -100.0, True, False, self._info()
            
        self.steps += 1
        selected_node = self.neighbors[action]
        selected_queue = self.queues[action]
        
        self.last_action = action
        
        # 0. Simulate global network traffic noise (happens at every time step)
        # Random arrivals: 20% chance an external packet arrives at any node
        arrivals = np.random.random(self.total_nodes) < 0.20
        self.global_queues = np.minimum(3, self.global_queues + arrivals)
        
        # Random departures: 50% chance a node successfully processes and transmits a packet
        departures = np.random.random(self.total_nodes) < 0.50
        self.global_queues = np.maximum(0, self.global_queues - departures)
        
        # 1. Congestion Drop
        if selected_queue >= 3:
            self.last_reward = -50.0
            self._update_neighbors()
            return self._encode_state(self.destination_id, self.queues), -50.0, False, True, self._info()
            
        # Agent's packet successfully enters the queue of selected node
        self.global_queues[selected_node] = min(3, self.global_queues[selected_node] + 1)
            
        # 2. Routing Success
        if selected_node == self.destination_id:
            self.current_node = selected_node
            self._update_neighbors() 
            self.last_reward = 100.0
            return self._encode_state(self.destination_id, self.queues), 100.0, True, False, self._info()
            
        # 3. Hop
        self.current_node = selected_node
        self._update_neighbors()
        new_state = self._encode_state(self.destination_id, self.queues)
        
        # Differential hop cost based on congestion
        if selected_queue == 0:
            hop_reward = -1.0  # Fast
        elif selected_queue == 1:
            hop_reward = -2.0  # Slight delay
        else: # selected_queue == 2
            hop_reward = -5.0  # Heavy delay
            
        self.last_reward = hop_reward
        return new_state, hop_reward, False, False, self._info()
        
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array mode is supported")
        from applications.network_routing.visualize import NetworkRoutingVisualizer
        if not hasattr(self, "visualizer"):
            self.visualizer = NetworkRoutingVisualizer(self.config)
        return self.visualizer.render(self, None, self.last_action, self.last_reward, None, self._info())
        
    def _info(self) -> dict:
        return {
            "current_node": self.current_node,
            "destination_id": self.destination_id,
            "valid_actions": self.get_valid_actions(),
            "step_count": self.steps
        }

    # O(1) SBQ Compatibility APIs
    def get_distance_api(self, s1: int, s2: int) -> float:
        """
        O(1) Distance utilizing Bitwise Operations.
        Mask out top 8 bits for Destination ID. If they differ, distance is inf.
        Otherwise, compute bit count of XOR for the lower 24 bits.
        """
        dest1 = s1 >> 24
        dest2 = s2 >> 24
        if dest1 != dest2:
            return float('inf')
        
        mask = 0xFFFFFF
        diff = (s1 & mask) ^ (s2 & mask)
        return float(diff.bit_count())
        
    def get_neighborhood_api(self, state: int, search_radius: int = 2) -> np.ndarray:
        """
        Generates neighborhood states natively using bit manipulation.
        """
        valid_codes = [0b000, 0b100, 0b110, 0b111]
        
        mask = 0xFFFFFF
        queues_bits = state & mask
        current_codes = []
        for i in range(self.N):
            current_codes.append((queues_bits >> (i * 3)) & 0b111)
            
        dest_prefix = state & ~mask
        neighborhood = set()
        
        def generate(idx, current_dist, accumulated_bits):
            if current_dist > search_radius:
                return
            if idx == self.N:
                neighborhood.add(dest_prefix | accumulated_bits)
                return
                
            orig_code = current_codes[idx]
            for code in valid_codes:
                diff = (orig_code ^ code).bit_count()
                if current_dist + diff <= search_radius:
                    generate(idx + 1, current_dist + diff, accumulated_bits | (code << (idx * 3)))
                    
        generate(0, 0, 0)
        return np.array(list(neighborhood), dtype=np.int64)
