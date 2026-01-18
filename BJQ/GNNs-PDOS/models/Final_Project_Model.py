
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter
import math

# =============================================================================
# PART 1: Core Components (Matformer + Physics + Virtual Nodes)
# =============================================================================

class RBFExpansion(nn.Module):
    """
    [Hierarchical Level 1: Geometric Encoding]
    Radial Basis Function expansion for high-fidelity edge distance encoding.
    Captures the precise local geometry around defects in 2D materials.
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
        return torch.exp(-(distance.unsqueeze(1) - self.centers) ** 2 / self.sigma ** 2)

class MatformerConv(MessagePassing):
    """
    [Hierarchical Level 2: Local Structural Interaction]
    Authentic Matformer Attention Mechanism.
    Uses a unique triplet attention (Query, Key, Edge) to capture atomic interactions.
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
        
        # Matformer specific message update
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels), 
            nn.LayerNorm(out_channels)
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.lin_skip = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        
        out = out.view(-1, H * C)
        out = self.lin_concate(out)
        out = F.silu(self.bn(out))
        out = out + self.lin_skip(x)
        return out

    def message(self, query_i, key_i, key_j, value_j, value_i, edge_attr, index):
        if self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            edge_emb = 0
            
        # Matformer Attention Logic:
        # 1. Augment Query, Key with self/edge info
        query_i_aug = torch.cat((query_i, query_i, query_i), dim=-1) # [E, H, 3C]
        key_j_aug = torch.cat((key_i, key_j, edge_emb), dim=-1)      # [E, H, 3C]
        
        # 2. Compute Attention Score
        alpha = (query_i_aug * key_j_aug).sum(dim=-1) / math.sqrt(self.out_channels * 3)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 3. Compute Value Update
        out = torch.cat((value_i, value_j, edge_emb), dim=-1) # [E, H, 3C]
        
        # 4. Gating Mechanism
        gate = self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 1) * out)) # Simplified gating
        # Note: Original Matformer code logic is complex here, we simplify for stability while keeping the "Triplet" spirit
        
        # Re-implementing exact Matformer logic for "out" mixing
        # out = self.lin_msg_update(out) * self.sigmoid(...)
        # We'll use a simplified weighted sum for robustness in this project:
        out = out * alpha.unsqueeze(-1)
        out = self.lin_msg_update(out) 
        
        # Reduce 3C -> C
        out = self.msg_layer(out) 
        
        return out

class VirtualNodeBlock(nn.Module):
    """
    [Hierarchical Level 3: Virtual Atom (Global Interaction)]
    Implements the "Virtual Atom" concept from the proposal.
    Acts as a global communication hub to capture long-range dependencies in 2D materials.
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp_virtual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.mlp_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, batch, vx):
        # x: Real atoms [N, C]
        # vx: Virtual atoms [Batch, C]
        
        # 1. Real -> Virtual (Aggregation)
        # Sum real atom features to update virtual node
        vx_agg = global_add_pool(x, batch) # [Batch, C]
        vx = vx + vx_agg
        vx = self.mlp_virtual(vx) # Update virtual state
        
        # 2. Virtual -> Real (Broadcast)
        # Broadcast updated virtual state back to real atoms
        vx_broadcast = vx[batch] # [N, C]
        x = x + vx_broadcast
        x = self.mlp_real(x) # Update real state
        
        return x, vx

# =============================================================================
# PART 2: The Hierarchical Hybrid Model
# =============================================================================

class HierarchicalHybridMatformer(nn.Module):
    """
    [Final Project Model: Hierarchical Hybrid Matformer]
    
    Architecture aligned with the Innovation Proposal:
    1. **Local Level**: RBF + MatformerConv (Geometric fidelity).
    2. **Global Level**: Virtual Atom (Long-range interactions for 2D defects).
    3. **Physical Level**: Orbital Electron Fusion (Knowledge injection).
    
    Target: Full-process PDOS prediction for defective 2D materials.
    """
    def __init__(self, 
                 node_input_dim=118, 
                 edge_dim=128, 
                 hidden_dim=256, 
                 out_dim=201*4, 
                 num_layers=3, 
                 heads=4,
                 dropout=0.1):
        super().__init__()
        
        # --- A. Input Embeddings ---
        self.node_embedding = nn.Embedding(node_input_dim, hidden_dim)
        
        # [High-Fidelity Edge] RBF
        self.rbf = RBFExpansion(vmin=0, vmax=8, bins=edge_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # [Virtual Atom] Initial State
        self.virtual_node_embedding = nn.Embedding(1, hidden_dim)
        
        # --- B. Hierarchical Encoder Layers ---
        self.layers = nn.ModuleList()
        self.virtual_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            # Local Interaction (Matformer)
            self.layers.append(
                MatformerConv(in_channels=hidden_dim, 
                              out_channels=hidden_dim, 
                              heads=heads, 
                              edge_dim=hidden_dim,
                              dropout=dropout)
            )
            # Global Interaction (Virtual Atom)
            self.virtual_blocks.append(
                VirtualNodeBlock(hidden_dim, dropout)
            )
            
        # --- C. Physical Feature Fusion Branch ---
        # [Physical Prior] Energy Encoding
        self.fc_energies = nn.Sequential(
            nn.Linear(201, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        
        # [Physical Prior] Orbital Counts (s, p, d, f)
        self.fc_orbital_counts = nn.Sequential(
            nn.Linear(4, 128, bias=False),
            nn.LayerNorm(128)
        )
        
        # --- D. Prediction Head ---
        # Fuses: Graph Features (768) + Energy Features (128) + Orbital Features (128)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*3 + 128 + 128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, data):
        # 1. Unpack Data
        x, edge_index, edge_attr, energies, orbital_counts = \
            data.x.long(), data.edge_index, data.edge_attr, data.energies, data.orbital_counts
        batch = data.batch
        num_graphs = batch.max().item() + 1
        device = batch.device
        
        # 2. Initial Embeddings
        x = self.node_embedding(x.squeeze(1))
        
        # Edge RBF
        if edge_attr.dim() > 1:
            distances = edge_attr[:, 0]
        else:
            distances = edge_attr
        edge_feat = self.rbf(distances)
        edge_feat = self.edge_embedding(edge_feat)
        
        # Virtual Node Initialization [Batch, C]
        vx = self.virtual_node_embedding(torch.zeros(num_graphs, dtype=torch.long, device=device))
        
        # 3. Hierarchical Message Passing
        for layer, v_block in zip(self.layers, self.virtual_blocks):
            # A. Local Matformer Step
            # x_local = x + layer(x, edge_index, edge_feat) # Residual handled in layer
            x = layer(x, edge_index, edge_feat)
            
            # B. Global Virtual Atom Step
            x, vx = v_block(x, batch, vx)
            
        # 4. Global Pooling (Readout)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        graph_feat = torch.cat((mean_pool, max_pool, sum_pool), dim=-1) # [B, 768]
        
        # 5. Physical Fusion
        energies_feat = self.fc_energies(energies) # [B, 128]
        
        # Orbital Counts Aggregation
        cell_orbital_totals = torch.zeros((num_graphs, 4), dtype=orbital_counts.dtype, device=device)
        cell_orbital_totals = cell_orbital_totals.scatter_add(
            dim=0,
            index=batch.unsqueeze(1).expand(-1, 4),
            src=orbital_counts
        )
        orbital_feat = self.fc_orbital_counts(cell_orbital_totals) # [B, 128]
        
        # 6. Final Prediction
        combined_feat = torch.cat((graph_feat, energies_feat, orbital_feat), dim=-1) # [B, 1024]
        out = self.fc(combined_feat)
        
        return out.reshape(num_graphs, 4, 201)
