
import os
import sys
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import numpy as np
from typing import List, Tuple, Optional
import math
import ase.db
from pymatgen.io.ase import AseAtomsAdaptor

# ==============================================================================
# 1. 物理驱动的图构建与虚拟节点处理 (对应项目书 2.2 & 4.3)
# ==============================================================================

class DefectGraphBuilder:
    """
    负责将晶体结构转换为包含缺陷信息的图数据。
    """
    def __init__(self, cutoff: float = 5.0, max_neighbors: int = 12):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.adaptor = AseAtomsAdaptor()
        
    def process_structure(self, structure, defect_info: Optional[dict] = None):
        """
        处理单个晶体结构 (pymatgen Structure 对象)。
        defect_info: 包含缺陷位置和类型的字典。
        """
        # 1. 获取原子特征
        atomic_numbers = [site.specie.Z for site in structure]
        coords = structure.cart_coords
        lattice = structure.lattice.matrix
        
        # 2. 缺陷标记
        is_defect = torch.zeros(len(atomic_numbers), dtype=torch.long)
        is_vacancy = torch.zeros(len(atomic_numbers), dtype=torch.long)
        
        # 如果没有显式缺陷信息，暂且全设为0 (或者后续通过对比 Pristine 结构来推断)
        if defect_info:
            if 'vacancies' in defect_info:
                for vac_idx in defect_info['vacancies']:
                    if vac_idx < len(is_defect):
                        is_defect[vac_idx] = 1
                        is_vacancy[vac_idx] = 1
                        atomic_numbers[vac_idx] = 0 # 虚拟节点
            if 'substitutions' in defect_info:
                for sub_idx in defect_info['substitutions']:
                    if sub_idx < len(is_defect):
                        is_defect[sub_idx] = 1
        
        # 3. 周期性邻居搜索
        all_neighbors = structure.get_all_neighbors(r=self.cutoff)
        
        edge_src = []
        edge_dst = []
        edge_dist = []
        edge_vec = []
        
        for i, neighbors in enumerate(all_neighbors):
            neighbors = sorted(neighbors, key=lambda x: x[1])[:self.max_neighbors]
            for nbr in neighbors:
                j = nbr[2]
                d = nbr[1]
                edge_src.append(i)
                edge_dst.append(j)
                edge_dist.append(d)
                diff_vec = nbr[0].coords - coords[i]
                edge_vec.append(diff_vec)
                
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_dist, dtype=torch.float).unsqueeze(-1)
        edge_vec = torch.tensor(np.array(edge_vec), dtype=torch.float)
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            edge_vec=edge_vec,
            is_defect=is_defect,
            is_vacancy=is_vacancy,
            lattice=torch.tensor(lattice, dtype=torch.float).unsqueeze(0)
        )
        return data

# ==============================================================================
# 2. 基础组件: RBF 与 角度特征
# ==============================================================================

class RBFExpansion(nn.Module):
    def __init__(self, vmin=0, vmax=8.0, bins=40):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.gamma = 10.0

    def forward(self, distance):
        return torch.exp(-self.gamma * (distance - self.centers) ** 2)

# ==============================================================================
# 3. 核心创新: 分层混合注意力模型
# ==============================================================================

class DefectAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.geo_bias_mlp = nn.Sequential(
            nn.Linear(40, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads)
        )
        
        self.defect_bias = nn.Parameter(torch.randn(num_heads, 4)) 

    def forward(self, x, edge_index, edge_attr_rbf, is_defect):
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        src, dst = edge_index
        score = (Q[src] * K[dst]).sum(dim=-1) / math.sqrt(self.head_dim)
        
        geo_bias = self.geo_bias_mlp(edge_attr_rbf)
        score = score + geo_bias
        
        defect_code = is_defect[src] * 2 + is_defect[dst]
        defect_bias_val = self.defect_bias[:, defect_code].t()
        score = score + defect_bias_val
        
        alpha = torch_geometric.utils.softmax(score, dst)
        msg = V[src] * alpha.unsqueeze(-1)
        out = scatter(msg, dst, dim=0, reduce='sum')
        
        out = out.view(-1, self.hidden_dim)
        return self.out_proj(out)

import torch_geometric.utils
from torch_geometric.nn import GATConv

class DefectPredictorModel(nn.Module):
    def __init__(self, node_input_dim=100, hidden_dim=128, output_dim=1, num_layers=3):
        super().__init__()
        self.atom_embedding = nn.Embedding(node_input_dim + 1, hidden_dim)
        self.defect_embedding = nn.Embedding(2, hidden_dim) 
        self.rbf = RBFExpansion(bins=40)
        
        self.local_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=40) 
            for _ in range(num_layers)
        ])
        
        self.global_layers = nn.ModuleList([
            DefectAwareAttention(hidden_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        h_atom = self.atom_embedding(data.x)
        h_defect = self.defect_embedding(data.is_defect)
        h = h_atom + h_defect
        
        edge_feat = self.rbf(data.edge_attr)
        
        for layer in self.local_layers:
            h = h + layer(h, data.edge_index, edge_attr=edge_feat)
            h = F.layer_norm(h, h.shape[1:])
            h = F.silu(h)
            
        for layer in self.global_layers:
            h_global = layer(h, data.edge_index, edge_feat, data.is_defect)
            h = h + h_global
            h = F.layer_norm(h, h.shape[1:])
            
        out = scatter(h, data.batch, dim=0, reduce='mean')
        return self.fc_out(out)

# ==============================================================================
# 4. 数据加载与训练流程
# ==============================================================================

class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def load_data_from_db(db_path, limit=None):
    dataset = []
    builder = DefectGraphBuilder()
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return []

    print(f"Loading data from {db_path}...")
    sys.stdout.flush()
    try:
        db = ase.db.connect(db_path)
        # 如果 limit 为 None，则读取所有数据；为了演示，这里设为 500
        # 修正：limit=None 时 ase.db.select(limit=None) 可能不按预期工作，取决于版本
        # 显式使用一个大数或者不传 limit 参数
        if limit is None:
            selector = db.select(limit=200) # Use 200 for now to be safe
        else:
            selector = db.select(limit=limit)
            
        dataset = []
        for i, row in enumerate(selector):
            if i % 100 == 0:
                print(f"Processing row {i}...")
                sys.stdout.flush()
            if i == 0:
                print(f"First row keys: {list(row.key_value_pairs.keys())}")
                sys.stdout.flush()
            # 提取结构
            atoms = row.toatoms()
            structure = builder.adaptor.get_structure(atoms)
            
            # 尝试提取缺陷信息 (这里假设数据库中有 defect_index 字段，如果没有则跳过显式标记)
            # 实际情况中可能需要根据 formula 或其他 metadata 推断
            defect_info = {}
            if hasattr(row, 'defect_index'):
                 defect_info['vacancies'] = [row.defect_index]
            
            # 提取目标属性 (例如 formation energy 'eform')
            # 如果没有 eform，尝试其他 key
            y_val = row.get('eform', row.get('energy', None))
            
            if y_val is None or math.isnan(y_val):
                continue
                
            data = builder.process_structure(structure, defect_info)
            data.y = torch.tensor([y_val], dtype=torch.float)
            dataset.append(data)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
        
    if len(dataset) > 0:
        return dataset
    else:
        # 如果 dataset 为空，可能是因为 selector 没返回数据
        print("Warning: No data loaded from database.")
        return []

def train_model(dataset, epochs=50, batch_size=16):
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Split Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 2. Compute Normalizer statistics on Train set only
    all_y = torch.cat([d.y for d in train_dataset])
    normalizer = Normalizer(all_y)
    print(f"Target Normalizer: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = DefectPredictorModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # verbose=True deprecated warning fix
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    print(f"Starting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...")
    
    # 记录训练历史
    history_path = r"D:\Github hanjia\whuphy-attention\BJQ\training_history.csv"
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss_mse', 'val_mae'])
    
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        print(f"DEBUG: Starting epoch {epoch}")
        sys.stdout.flush()
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            # Normalize target for loss calculation
            target_norm = normalizer.norm(batch.y)
            
            loss = criterion(pred.squeeze(), target_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            
        avg_train_loss = train_loss / len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_norm = model(batch)
                # Denormalize prediction for MAE
                pred = normalizer.denorm(pred_norm.squeeze())
                val_mae += torch.sum(torch.abs(pred - batch.y)).item()
                
        avg_val_mae = val_mae / len(val_dataset)
        
        # 写入历史数据
        # with open(history_path, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch + 1, avg_train_loss, avg_val_mae])
        
        scheduler.step(avg_val_mae)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f}")
            sys.stdout.flush()
            
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            # torch.save(model.state_dict(), 'best_model.pth')
        
    return model, normalizer

if __name__ == "__main__":
    try:
        db_path = r"D:\Github hanjia\whuphy-attention\imp2d.db"
        
        # 1. Load Data
        dataset = load_data_from_db(db_path, limit=None) # Limit for quick testing
        
        if len(dataset) > 0:
            print(f"Loaded {len(dataset)} samples.")
            sys.stdout.flush()
            
            # 2. Train
            model, normalizer = train_model(dataset, epochs=20)
            
            # 3. Predict / Test (on ALL samples for Figure 4)
            print("Running prediction on full dataset for visualization...")
            model.eval()
            # Use a larger batch size for inference
            full_loader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            all_preds = []
            all_actuals = []
            
            with torch.no_grad():
                for batch in full_loader:
                    pred_norm = model(batch.to('cpu'))
                    pred = normalizer.denorm(pred_norm.squeeze())
                    actual = batch.y
                    
                    all_preds.extend(pred.numpy().flatten())
                    all_actuals.extend(actual.numpy().flatten())
            
            # Save full results to CSV
            import pandas as pd
            df_res = pd.DataFrame({'Actual': all_actuals, 'Predicted': all_preds})
            df_res.to_csv(r"D:\Github hanjia\whuphy-attention\BJQ\predictions_full.csv", index=False)
            print("Full predictions saved to predictions_full.csv")
            
            # Keep the old text file output for backward compatibility (first 5)
            with open(r"D:\Github hanjia\whuphy-attention\BJQ\prediction_results.txt", "w") as f:
                for p, a in zip(all_preds[:5], all_actuals[:5]):
                     f.write(f"Pred: {p:.4f}, Actual: {a:.4f}\n")
        else:
            print("No dataset loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
