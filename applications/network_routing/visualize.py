import io
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

class NetworkRoutingVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.frames = []
        self.G = None
        self.pos = None
        
    def _init_graph(self, env):
        if self.G is not None:
            return
            
        self.G = nx.Graph() # Undirected for rendering
        self.G.add_nodes_from(range(env.total_nodes))
        
        # Add edges based on D matrix
        for i in range(env.total_nodes):
            for j in range(env.total_nodes):
                if env.D[i, j] == 1:
                    self.G.add_edge(i, j)
                    
        # Spring layout is better for scale-free and small-world networks
        self.pos = nx.spring_layout(self.G, seed=42)
        
    def render(self, env, state, action, reward, next_state, info):
        self._init_graph(env)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Step: {info['step_count']} | Current: {info['current_node']} -> Dest: {info['destination_id']}")
        
        node_colors = []
        node_sizes = []
        
        for i in range(env.total_nodes):
            if i == info['destination_id']:
                node_colors.append('#FF4444') # Red for destination
                node_sizes.append(800)
            elif i == info['current_node']:
                node_colors.append('#4444FF') # Blue for current node
                node_sizes.append(800)
            else:
                node_colors.append('#CCCCCC') # Gray for others
                node_sizes.append(400)
                
        # Draw base graph
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=node_sizes, ax=ax, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_color='white', font_weight='bold')
        
        # Draw neighbor queue states around current node
        for idx, n in enumerate(env.neighbors):
            if n != -1:
                q = env.queues[idx]
                if q == 0:
                    q_color = 'green'
                    q_text = "Free"
                elif q == 1:
                    q_color = 'yellow'
                    q_text = "Busy"
                elif q == 2:
                    q_color = 'orange'
                    q_text = "Cong"
                else:
                    q_color = 'red'
                    q_text = "Full"
                    
                x, y = self.pos[n]
                # Offset text slightly towards the center or outside
                ax.text(x, y + 0.1, f"Q:{q_text}", color=q_color, fontsize=10, 
                        ha='center', va='bottom', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Highlight chosen action if available
        if action is not None and action < len(env.neighbors) and env.neighbors[action] != -1:
            chosen_neighbor = env.neighbors[action]
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[(info['current_node'], chosen_neighbor)], 
                                   edge_color='orange', width=4, ax=ax)
            
        # Optional: Add reward overlay
        if reward is not None:
            ax.text(0, -1.2, f"Reward: {reward}", color='black', fontsize=14, ha='center', va='center')

        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to numpy array for core/trainer.py compatibility
        img = np.array(Image.open(buf))
        self.frames.append(img)
        return img

    def save_episode_gif(self, frames, path, fps=4):
        if not frames:
            return
        imageio.mimsave(path, frames, fps=fps)
        
    def save_final_frame(self, frame, path):
        imageio.imwrite(path, frame)
