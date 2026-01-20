import sqlite3
import json
import pickle
import os
import numpy as np

# 配置路径
DB_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/imp2d.db'
OUTPUT_DIR = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'cleaned_dataset.pkl')

def clean_data():
    if not os.path.exists(DB_PATH):
        print(f"错误: 找不到数据库文件 {DB_PATH}")
        return

    print(f"正在处理数据库: {DB_PATH}")
    
    # 连接数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 查询需要的字段
    # 注意: numbers, positions, cell 是二进制 BLOB 数据
    query = """
    SELECT id, unique_id, numbers, positions, cell, pbc, key_value_pairs 
    FROM systems
    """
    
    cursor.execute(query)
    
    cleaned_samples = []
    total_count = 0
    kept_count = 0
    
    print("开始遍历并清洗数据...")
    
    for row in cursor:
        total_count += 1
        row_id, unique_id, numbers_blob, positions_blob, cell_blob, pbc_int, kv_json = row
        
        if not kv_json:
            continue
            
        # 1. 解析元数据
        try:
            kv = json.loads(kv_json)
        except json.JSONDecodeError:
            continue
            
        # 2. 筛选条件: converged == True
        # 兼容 boolean True 或 数字 1
        converged = kv.get('converged')
        if converged is not True and converged != 1:
            continue
            
        # 3. 筛选条件: eform 存在且有效
        eform = kv.get('eform')
        if eform is None:
            continue
        
        try:
            eform_val = float(eform)
            # 过滤掉 NaN (Not a Number)
            if np.isnan(eform_val):
                continue
        except (ValueError, TypeError):
            continue
            
        # 4. 解析几何信息 (BLOB -> Numpy Array)
        try:
            # numbers: int32 (原子序数)
            numbers = np.frombuffer(numbers_blob, dtype=np.int32)
            
            # positions: float64 (N, 3) (原子坐标)
            positions = np.frombuffer(positions_blob, dtype=np.float64).reshape(-1, 3)
            
            # cell: float64 (3, 3) (晶胞矢量)
            cell = np.frombuffer(cell_blob, dtype=np.float64).reshape(3, 3)
            
            # pbc: 解析整数位掩码 -> [bool, bool, bool]
            # ASE pbc 存储逻辑: 通常是一个整数，二进制位表示三个方向的周期性
            pbc = np.array([(pbc_int >> i) & 1 for i in range(3)], dtype=bool)
            
        except Exception as e:
            print(f"ID {row_id} 解析几何信息失败: {e}")
            continue
            
        # 5. 构建样本对象
        # 包含几何信息、基本属性，并将 eform 设为 target
        sample = {
            'id': row_id,
            'unique_id': unique_id,
            'numbers': numbers,       # 原子序数数组
            'positions': positions,   # 原子位置矩阵
            'cell': cell,            # 晶胞矩阵
            'pbc': pbc,              # 周期性边界条件
            'target': eform_val,     # 预测标签: 形成能
            'metadata': {            # 其他元数据
                'formula': kv.get('name', ''), 
                'host': kv.get('host'),
                'dopant': kv.get('dopant'),
                'site': kv.get('site'),
                'defecttype': kv.get('defecttype'),
                'natoms': len(numbers)
            }
        }
        
        cleaned_samples.append(sample)
        kept_count += 1
        
        if kept_count % 2000 == 0:
            print(f"已收集 {kept_count} 个有效样本...")

    conn.close()
    
    # 输出统计信息
    print("-" * 30)
    print(f"原始数据总数: {total_count}")
    print(f"清洗后保留数: {kept_count}")
    print(f"丢弃样本数:   {total_count - kept_count}")
    
    # 6. 保存为 Python 对象 (Pickle)
    if kept_count > 0:
        print(f"正在保存至: {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(cleaned_samples, f)
        print("保存成功!")
        
        # 打印第一个样本以供检查
        print("\n[示例样本结构]:")
        first = cleaned_samples[0]
        print(f"ID: {first['id']}")
        print(f"Target (eform): {first['target']:.4f}")
        print(f"Numbers shape: {first['numbers'].shape}")
        print(f"Positions shape: {first['positions'].shape}")
        print(f"Metadata: {first['metadata']}")
    else:
        print("警告: 没有符合条件的样本被保存。")

if __name__ == "__main__":
    clean_data()
