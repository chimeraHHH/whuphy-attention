#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据检查脚本 - 查看解析后的数据结构
"""

import pickle
import sys
from pathlib import Path

def inspect_data():
    """检查数据文件内容"""
    data_dir = Path("data/processed")
    
    # 检查complete_data.pkl
    complete_file = data_dir / "complete_data.pkl"
    if complete_file.exists():
        print(f"=== 检查 {complete_file} ===")
        try:
            with open(complete_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"总记录数: {len(data)}")
            
            if len(data) > 0:
                # 查看第一条记录的结构
                first_record = data[0]
                print(f"第一条记录的键: {list(first_record.keys())}")
                
                # 查看properties
                if 'properties' in first_record:
                    props = first_record['properties']
                    print(f"Properties键: {list(props.keys())}")
                    for key, value in props.items():
                        print(f"  {key}: {value} (类型: {type(value)})")
                
                # 查看metadata
                if 'metadata' in first_record:
                    meta = first_record['metadata']
                    print(f"Metadata键: {list(meta.keys())}")
                    for key, value in meta.items():
                        print(f"  {key}: {value} (类型: {type(value)})")
                
                # 查看atoms
                if 'atoms' in first_record:
                    atoms = first_record['atoms']
                    print(f"Atoms类型: {type(atoms)}")
                    if hasattr(atoms, 'get_chemical_formula'):
                        print(f"化学式: {atoms.get_chemical_formula()}")
                        print(f"原子数: {len(atoms)}")
                
                # 统计缺失的形成能数据
                missing_eform = sum(1 for d in data if 'eform' not in d.get('properties', {}))
                print(f"缺失形成能的记录数: {missing_eform}")
                
                # 显示一些形成能值
                eform_values = []
                for d in data:
                    props = d.get('properties', {})
                    if 'eform' in props:
                        try:
                            eform_values.append(float(props['eform']))
                        except:
                            pass
                
                if eform_values:
                    print(f"形成能值范围: {min(eform_values):.2f} 到 {max(eform_values):.2f} eV")
                    print(f"形成能值示例: {eform_values[:5]}")
                else:
                    print("没有找到有效的形成能值")
                    
        except Exception as e:
            print(f"读取文件失败: {e}")
    
    # 检查metadata.pkl
    metadata_file = data_dir / "metadata.pkl"
    if metadata_file.exists():
        print(f"\n=== 检查 {metadata_file} ===")
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"元数据键: {list(metadata.keys())}")
            
            if 'stats' in metadata:
                stats = metadata['stats']
                print(f"统计信息: {stats}")
            
            if 'data_info' in metadata:
                data_info = metadata['data_info']
                print(f"数据信息: {data_info}")
                
        except Exception as e:
            print(f"读取文件失败: {e}")

if __name__ == "__main__":
    inspect_data()