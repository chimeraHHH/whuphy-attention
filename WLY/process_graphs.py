import pickle
import numpy as np
import os
from ase import Atoms
from ase.neighborlist import neighbor_list
from tqdm import tqdm

# 配置路径
DATASET_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/cleaned_dataset.pkl'
OUTPUT_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/processed_dataset_with_graphs.pkl'

# 参数设置
CUTOFF_RADIUS = 5.0 # 邻居搜索半径 (Å)

def process_graphs():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'rb') as f:
        raw_data = pickle.load(f)
    
    processed_data = []
    print(f"Processing {len(raw_data)} samples...")
    
    for i, sample in enumerate(tqdm(raw_data)):
        try:
            # 1. 重建 ASE Atoms 对象
            atoms = Atoms(
                numbers=sample['numbers'],
                positions=sample['positions'],
                cell=sample['cell'],
                pbc=sample['pbc']
            )
            
            # 2. 构建邻居表 (Neighbor List)
            # neighbor_list 返回: (i, j, d, D)
            # i, j: 原子索引
            # d: 标量距离
            # D: 矢量距离 (经过 PBC 修正)
            idx_i, idx_j, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=CUTOFF_RADIUS)
            
            # 3. 存储边信息 (Edges)
            # edge_index: (2, E) -> [source, target]
            edge_index = np.vstack([idx_i, idx_j])
            edge_dist = d_ij
            
            # 4. 构建三体组合 (Triplets) & 计算角度
            # 我们需要找到以原子 i 为中心的所有邻居对 (j, k)
            # 这里的 idx_i 已经是源节点列表
            
            triplets = []
            angles = []
            
            # 获取原子总数
            n_atoms = len(atoms)
            
            # 为了加速查找，先构建邻接表： center_atom -> [neighbor_indices_in_edge_list]
            # 注意：这里的 index 是 edge_index 中的列索引 (边的 ID)
            # 例如: atom 0 作为中心，连接了边 5, 8, 12...
            
            # 使用 numpy 排序来分组
            # 先按 idx_i (中心原子) 排序
            sorted_indices = np.argsort(idx_i)
            idx_i_sorted = idx_i[sorted_indices]
            idx_j_sorted = idx_j[sorted_indices]
            D_ij_sorted = D_ij[sorted_indices]
            
            # 找到每个中心原子的切片范围
            # unique_indices: 原子 ID
            # split_indices: 切分点
            unique_center_atoms, split_indices = np.unique(idx_i_sorted, return_index=True)
            
            # 遍历每个作为中心的原子
            # split_indices[k] 到 split_indices[k+1] 是原子 unique_center_atoms[k] 的所有邻居
            split_indices = np.append(split_indices, len(idx_i_sorted))
            
            for k, center_atom in enumerate(unique_center_atoms):
                start = split_indices[k]
                end = split_indices[k+1]
                
                # 当前中心原子的所有邻居边信息
                # neighbors_j: 邻居原子 ID
                # neighbors_D: 中心指向邻居的向量 r_ij
                neighbors_j = idx_j_sorted[start:end]
                neighbors_D = D_ij_sorted[start:end]
                
                num_neighbors = len(neighbors_j)
                if num_neighbors < 2:
                    continue
                
                # 构建两两组合 (j, k)
                # 使用双重循环或矩阵运算
                # 这里为了清晰使用循环，由于邻居数通常较少 (<20)，开销可控
                for m in range(num_neighbors):
                    for n in range(num_neighbors):
                        if m == n:
                            continue
                        
                        # 邻居 m 和 邻居 n
                        # 向量 r_ij (center -> neighbor_m)
                        # 向量 r_ik (center -> neighbor_n)
                        vec1 = neighbors_D[m]
                        vec2 = neighbors_D[n]
                        
                        # 计算角度 theta_jik
                        # cos_theta = (v1 . v2) / (|v1| * |v2|)
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        
                        if norm1 < 1e-6 or norm2 < 1e-6:
                            continue
                            
                        dot_prod = np.dot(vec1, vec2)
                        cos_theta = dot_prod / (norm1 * norm2)
                        
                        # 数值稳定性截断
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        theta = np.arccos(cos_theta) # 弧度制
                        
                        # 记录三体: [neighbor_m, center, neighbor_n] -> [j, i, k]
                        # 这里的索引是原子的全局索引
                        triplets.append([neighbors_j[m], center_atom, neighbors_j[n]])
                        angles.append(theta)
            
            # 转换为 Numpy 数组
            if len(triplets) > 0:
                triplet_index = np.array(triplets)
                angles = np.array(angles)
            else:
                triplet_index = np.zeros((0, 3), dtype=int)
                angles = np.zeros((0,), dtype=float)
            
            # 5. 更新样本字典
            # 保持原有字段，添加新字段
            new_sample = sample.copy()
            new_sample['edge_index'] = edge_index
            new_sample['edge_dist'] = edge_dist
            new_sample['triplet_index'] = triplet_index
            new_sample['angles'] = angles
            
            processed_data.append(new_sample)
            
        except Exception as e:
            print(f"Warning: Failed to process sample {i} (ID: {sample.get('id', 'N/A')}): {e}")
            continue

    # 保存结果
    print("-" * 30)
    print(f"Processed {len(processed_data)} samples successfully.")
    print(f"Saving to {OUTPUT_PATH}...")
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Done!")
    
    # 打印示例
    if len(processed_data) > 0:
        s = processed_data[0]
        print("\n[Sample 0 Structure]")
        print(f"Edge Index shape: {s['edge_index'].shape}")
        print(f"Edge Dist shape: {s['edge_dist'].shape}")
        print(f"Triplet Index shape: {s['triplet_index'].shape}")
        print(f"Angles shape: {s['angles'].shape}")

if __name__ == "__main__":
    process_graphs()
