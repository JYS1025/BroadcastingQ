# Network Routing Environment

A complex network simulation environment built to test the dynamic routing performance of Reinforcement Learning agents, particularly Spatial Broadcasting Q-learning (SBQ). It supports various topologies such as Scale-Free, Multi-Hub, and Grid through the modularized `topology.py`.

## Environment Structure (MDP)

### 1. State (Observation)
To achieve extreme memory optimization, the state is compressed into a **single 32-bit integer** using bitwise operations:
- **Upper 8-bits (MSB)**: The target Destination Node ID (supports up to 256 nodes).
- **Lower 24-bits (LSB)**: Real-time queue congestion states of up to 8 neighboring routers connected to the current node.
  - Each neighbor is allocated a 3-bit **Thermometer Encoding**:
  - `0b000` (Free) / `0b100` (Busy) / `0b110` (Congested) / `0b111` (Full or Disconnected)

### 2. Action
- **Discrete(8)**: Select one of the maximum 8 physically connected neighbors to forward the packet to the next hop.
- If the selected action index corresponds to a disconnected edge (no physical link), the packet is immediately dropped.

### 3. Reward
A differential delay penalty system based on network traffic congestion is applied to encourage the agent to find uncongested detour routes:
- **`+100.0`** : Successfully reached the destination (Episode Terminated successfully).
- **`-50.0`** : Packet drop. Occurs when forwarding to a neighbor with a Full queue (`0b111`) or selecting a disconnected edge (Episode Truncated as failure).
- **`-1.0`** : Forwarded to a Free queue (Fastest transmission).
- **`-2.0`** : Forwarded to a Busy queue (Slight traffic delay).
- **`-5.0`** : Forwarded to a Congested queue (Severe bottleneck delay).

### 4. Stateful Queue Dynamics
The traffic state of the entire network is physically updated at every step based on causal relationships:
- **Agent Arrival**: When the agent forwards a packet, the target node's queue is incremented by +1.
- **Background Noise**: Across the entire network, unknown external traffic arrives with a 20% probability (Queue +1), and existing packets are processed and departed with a 50% probability (Queue -1).

## Execution
Various topology settings can be adjusted via `config_*.yaml`.
```bash
# Run SBQ Agent
python main.py --config applications/network_routing/config_sbq.yaml

# Run All Agents Comparison
./run_network_routing_all.sh
```
