# tasks/neighbors_match.py
import torch
from torch_geometric.data import Data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import numpy as np

class NeighborsMatchDataset:
    def __init__(self, depth):
        self.depth = depth
        # Paper uses n=2^depth nodes for fair comparison with tree tasks
        self.num_nodes = 2 ** depth
        self.num_colors = 2 ** depth  # c = n in the paper
        self.edge_prob = 0.5  # p = 0.5 typical for ER graphs
        self.num_graphs = 16000  # Generate many random graphs
        self.criterion = F.cross_entropy
    
    def generate_random_graph(self):
        n = self.num_nodes
        
        # Generate Erdős-Rényi random graph
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < self.edge_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
        
        if len(edges) == 0:
            # Ensure at least one edge
            i, j = np.random.choice(n, 2, replace=False)
            edges = [[i, j], [j, i]]
        
        edge_index = torch.tensor(edges).t()
        
        # Assign random colors to nodes
        colors = torch.randint(0, self.num_colors, (n,))
        
        # Compute labels: for each node, count neighbors with same color
        labels = torch.zeros(n, dtype=torch.long)
        for node in range(n):
            neighbors = edge_index[1][edge_index[0] == node]
            same_color_count = (colors[neighbors] == colors[node]).sum().item()
            labels[node] = same_color_count
        
        # Features: one-hot encoding of color
        x = torch.nn.functional.one_hot(colors, num_classes=self.num_colors).float()
        
        return Data(x=x, edge_index=edge_index, y=labels)
    
    def generate_data(self, train_fraction):
        data_list = [self.generate_random_graph() for _ in range(self.num_graphs)]
        
        # Split train/test
        train_size = int(self.num_graphs * train_fraction)
        X_train = data_list[:train_size]
        X_test = data_list[train_size:]
        
        dim0 = self.num_colors  # Input feature dimension
        max_label = max(max(d.y) for d in data_list).item()
        out_dim = max_label + 1  # Output classes (0 to max neighbor count)
        
        return X_train, X_test, dim0, out_dim, self.criterion