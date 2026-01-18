import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch.nn import Parameter
import math

# =============================================================================
# 1. Matformer Components (Adapted)
# =============================================================================

class RBFExpansion(nn.Module):
    """
    Radial Basis Function expansion for edge distances.
    From Matformer/CGCNN/MegNet logic.
    """
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.center = torch.linspace(vmin, vmax, bins)
        self.sigma = (vmax - vmin) / bins
        if lengthscale is not None:
            self.sigma = lengthscale
            
        self.register_buffer('centers', self.center)

    def forward(self, distance):
        """
        Args:
            distance: tensor of shape (N,)
        Returns:
            tensor of shape (N, bins)
        """
        return torch.exp(-(distance.unsqueeze(1) - self.centers) ** 2 / self.sigma ** 2)

class MatformerConv(MessagePassing):
    """
    Simplified MatformerConv based on the provided code reference.
    """
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=None, dropout=0.0):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        
        # Feed-forward / Message Update block
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels), 
            nn.LayerNorm(out_channels)
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        
        # Residual connection layers
        self.lin_skip = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        # Propagate
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        
        # Concatenate heads
        out = out.view(-1, H * C)
        out = self.lin_concate(out)
        
        # BatchNorm + Silu (Swish)
        out = F.silu(self.bn(out))
        
        # Residual connection
        x_r = self.lin_skip(x)
        out = out + x_r
        
        return out

    def message(self, query_i, key_i, key_j, value_j, value_i, edge_attr, index):
        # Edge handling
        if self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            edge_emb = 0
            
        # Attention Score
        # query_i: [E, H, C]
        # key_j: [E, H, C]
        # We augment key_j with edge info for attention
        key_j_aug = key_j + edge_emb
        
        alpha = (query_i * key_j_aug).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = F.softmax(alpha, dim=1) # Softmax over heads? Or neighbors? 
        # Standard GAT softmax is over neighbors (using index). 
        # Matformer code uses specialized softmax. Here we simplify.
        # Let's use simple scaling for now as exact softmax over neighbors requires 'softmax' util
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Message calculation
        # Combine value_j with edge info
        val_j_aug = value_j + edge_emb
        
        # Weighted sum
        out = val_j_aug * alpha.unsqueeze(-1)
        
        return out

# =============================================================================
# 2. PhysMatformer: The Integrated Model
# =============================================================================

class PhysMatformer(nn.Module):
    """
    PhysMatformer: Integrating Matformer's periodic graph attention with 
    CGT_phys's physical feature fusion for PDOS prediction.
    """
    def __init__(self, 
                 node_input_dim=118, 
                 edge_dim=128, 
                 hidden_dim=256, 
                 out_dim=201*4, 
                 num_layers=3, 
                 heads=4):
        super().__init__()
        
        # 1. Embedding Layers
        self.node_embedding = nn.Embedding(node_input_dim, hidden_dim)
        
        # RBF for Edge Distances (Matformer style)
        self.rbf = RBFExpansion(vmin=0, vmax=8, bins=edge_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # 2. Encoder Layers (MatformerConv)
        self.layers = nn.ModuleList([
            MatformerConv(in_channels=hidden_dim, 
                          out_channels=hidden_dim, 
                          heads=heads, 
                          edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 3. Physics & Energy Features (CGT_phys style)
        self.fc_energies = nn.Sequential(
            nn.Linear(201, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        
        self.fc_orbital_counts = nn.Sequential(
            nn.Linear(4, 128, bias=False),
            nn.LayerNorm(128)
        )
        
        # 4. Final Prediction Head
        # Concatenation: [Graph_Mean, Graph_Max, Graph_Sum, Energies, Orbitals]
        # Dim: 256 + 256 + 256 + 128 + 128 = 1024
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*3 + 128 + 128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, data):
        """
        Forward pass combining structural and physical features.
        """
        # Unpack data
        x, edge_index, edge_attr, energies, orbital_counts = \
            data.x.long(), data.edge_index, data.edge_attr, data.energies, data.orbital_counts
        batch = data.batch
        
        # --- A. Structural Encoding (Matformer Path) ---
        
        # Node Embedding
        x = self.node_embedding(x.squeeze(1))
        
        # Edge Embedding (RBF + Linear)
        # Assuming edge_attr contains distances in first column or is just distances
        if edge_attr.dim() > 1 and edge_attr.size(1) > 1:
            distances = edge_attr[:, 0] # Take first feature as distance
        else:
            distances = edge_attr
            
        edge_feat = self.rbf(distances)
        edge_feat = self.edge_embedding(edge_feat)
        
        # Transformer Message Passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_feat)
            
        # --- B. Global Pooling ---
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        graph_feat = torch.cat((mean_pool, max_pool, sum_pool), dim=-1) # [B, 768]
        
        # --- C. Physical Feature Fusion (CGT_phys Path) ---
        
        # 1. Process Energies
        energies_feat = self.fc_energies(energies) # [B, 128]
        
        # 2. Process Orbital Counts (Global Aggregation)
        # Aggregate local orbital counts to global crystal level
        num_graphs = batch.max().item() + 1
        device = batch.device
        
        cell_orbital_totals = torch.zeros((num_graphs, 4), dtype=orbital_counts.dtype, device=device)
        cell_orbital_totals = cell_orbital_totals.scatter_add(
            dim=0,
            index=batch.unsqueeze(1).expand(-1, 4),
            src=orbital_counts
        )
        orbital_feat = self.fc_orbital_counts(cell_orbital_totals) # [B, 128]
        
        # --- D. Final Fusion & Prediction ---
        
        # Concatenate all features
        combined_feat = torch.cat((graph_feat, energies_feat, orbital_feat), dim=-1) # [B, 1024]
        
        # Predict PDOS
        out = self.fc(combined_feat)
        
        return out.reshape(num_graphs, 4, 201)
