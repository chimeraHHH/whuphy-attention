import numpy as np
import torch
import json
import os
from mendeleev import element
from pymatgen.core.periodic_table import Element

# 配置输出路径
OUTPUT_DIR = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'atom_features.pth')
JSON_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'atom_features.json')

# 定义需要获取的元素范围 (1 到 100 号元素)
MAX_Z = 100

def get_element_features():
    print(f"正在生成原子特征矩阵 (Z=1 到 {MAX_Z})...")
    
    # 定义特征列表
    features_list = []
    
    # 0号元素占位符 (Z=0, 实际上不存在，全为0)
    features_list.append([0.0] * 9) 

    for z in range(1, MAX_Z + 1):
        try:
            # 使用 mendeleev 获取数据
            el_m = element(z)
            # 使用 pymatgen 获取补充数据
            el_p = Element.from_Z(z)
            
            # 1. 族 (Group): 1-18
            group = el_m.group_id if el_m.group_id is not None else 0
            
            # 2. 周期 (Period): 1-7
            period = el_m.period
            
            # 3. 电负性 (Electronegativity - Pauling)
            en = el_m.electronegativity('pauling')
            if en is None: en = 0.0
            
            # 4. 共价半径 (Covalent Radius) [pm]
            rcov = el_m.covalent_radius
            if rcov is None: rcov = 0.0
            
            # 5. 范德华半径 (Van der Waals Radius) [pm]
            rvdw = el_m.vdw_radius
            if rvdw is None: rvdw = 0.0
            
            # 6. 价电子数 (Valence Electrons)
            # mendeleev 的 nvalence 是方法
            n_valence = 0
            try:
                # 尝试调用方法
                val = el_m.nvalence()
                if val is not None:
                    n_valence = val
            except:
                # 如果调用失败，尝试从 pymatgen 获取或推断
                # 这里简单处理：如果失败则为0
                pass
            
            # 7. 第一电离能 (First Ionization Energy) [eV]
            ion_energy = el_m.ionenergies.get(1, 0.0) if el_m.ionenergies else 0.0
            
            # 8. 电子亲和能 (Electron Affinity) [eV]
            # mendeleev 存储为 electron_affinity
            ea = el_m.electron_affinity
            if ea is None: ea = 0.0
            
            # 9. 原子质量 (Atomic Mass) [u] - 额外添加的一个基础属性
            mass = el_m.atomic_weight
            
            # 构建特征向量
            # [Group, Period, EN, Rcov, Rvdw, Val, Ion1, EA, Mass]
            feat = [
                group,
                period,
                en,
                rcov,
                rvdw,
                n_valence,
                ion_energy,
                ea,
                mass
            ]

            # Convert all to float explicitly to avoid type errors
            feat = [float(x) if x is not None else 0.0 for x in feat]
            
            features_list.append(feat)
            
        except Exception as e:
            print(f"Warning: Error processing Z={z}: {e}")
            features_list.append([0.0] * 9) # 错误时填充0

    # 转换为 Numpy 数组
    features_np = np.array(features_list, dtype=np.float32)
    print(f"原始特征矩阵形状: {features_np.shape}")
    
    # 归一化处理 (Min-Max Normalization)
    # 跳过第0行（padding），只对 Z=1~100 进行统计
    data_rows = features_np[1:]
    
    min_vals = data_rows.min(axis=0)
    max_vals = data_rows.max(axis=0)
    
    # 防止除以零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    normalized_data = (data_rows - min_vals) / range_vals
    
    # 重新组合：第0行保持全0，后面是归一化后的数据
    final_features = np.vstack([features_np[0], normalized_data])
    
    print("特征名称顺序: [Group, Period, Electronegativity, CovalentRadius, VdWRadius, ValenceElectrons, IonizationEnergy, ElectronAffinity, AtomicMass]")
    
    # 保存为 .pth (PyTorch Tensor)
    tensor_data = torch.from_numpy(final_features)
    torch.save(tensor_data, OUTPUT_FILE)
    print(f"已保存 Tensor 至: {OUTPUT_FILE}")
    
    # 保存为 .json (方便查看)
    # 为了 JSON 可读性，保存未归一化的原始数据和归一化后的数据
    output_dict = {
        "feature_names": ["Group", "Period", "Electronegativity", "CovalentRadius", "VdWRadius", "ValenceElectrons", "IonizationEnergy", "ElectronAffinity", "AtomicMass"],
        "normalization_params": {
            "min": min_vals.tolist(),
            "max": max_vals.tolist()
        },
        "data_normalized": final_features.tolist()
    }
    
    with open(JSON_OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"已保存 JSON 至: {JSON_OUTPUT_FILE}")

if __name__ == "__main__":
    get_element_features()
