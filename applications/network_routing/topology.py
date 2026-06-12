import numpy as np
import networkx as nx

def generate_topology(topology_type: str, total_nodes: int, max_degree: int) -> np.ndarray:
    """
    Generates an adjacency matrix (D) for the given topology type.
    
    Args:
        topology_type (str): Name of the topology (scale_free, multi_hub, grid, etc.)
        total_nodes (int): Total number of nodes in the network
        max_degree (int): Maximum allowed degree per node (for state encoding constraints)
        
    Returns:
        np.ndarray: A (total_nodes x total_nodes) adjacency matrix
    """
    while True:
        if topology_type == "scale_free":
            # m=1 creates a sparse, tree-like scale-free graph
            G = nx.barabasi_albert_graph(total_nodes, 1)
            
        elif topology_type == "multi_hub":
            # Manually construct a 3-hub network to guarantee multiple bottlenecks
            G = nx.Graph()
            G.add_nodes_from(range(total_nodes))
            # Fallback for extremely small networks
            if total_nodes < 16:
                hubs = [0]
                for i in range(1, total_nodes): G.add_edge(0, i)
            else:
                hubs = [0, 5, 10]
                # Connect hubs to form a backbone ring
                G.add_edge(0, 5)
                G.add_edge(5, 10)
                G.add_edge(10, 0)
                
                # Distribute remaining nodes as leaves across the hubs
                for i in range(1, 5): G.add_edge(0, i)
                for i in range(6, 10): G.add_edge(5, i)
                for i in range(11, total_nodes): G.add_edge(10, i)
                
        elif topology_type == "grid":
            side = int(np.sqrt(total_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
            
        else:
            # Default fallback: ring + cross-link
            G = nx.Graph()
            G.add_nodes_from(range(total_nodes))
            for i in range(total_nodes):
                G.add_edge(i, (i+1)%total_nodes)
                G.add_edge(i, (i + total_nodes//2)%total_nodes)
                
        # Validate degree constraint
        max_deg = max(dict(G.degree()).values())
        if max_deg <= max_degree:
            break
            
    return nx.to_numpy_array(G, dtype=np.int64)
