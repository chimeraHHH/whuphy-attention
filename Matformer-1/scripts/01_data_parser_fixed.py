#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMP2D数据库数据提取和初步清洗脚本 - 修复版

该脚本用于从IMP2D数据库中提取高质量的数据，包括：
1. 连接数据库并遍历所有条目
2. 提取原子结构、目标性质和元数据
3. 进行收敛性验证和完整性检查
4. 将清洗后的数据保存为pickle文件供后续使用

主要修复:
- 处理形成能中的NaN值
- 修复元数据统计信息
- 改进数据过滤逻辑
- 增强错误处理
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ase.db import connect
from ase import Atoms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts/data_parser_fixed.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class IMP2DDataParser:
    """IMP2D数据库数据解析器"""
    
    def __init__(self, db_path: str, output_dir: str = "data/processed"):
        """
        初始化数据解析器
        
        Args:
            db_path: 数据库文件路径
            output_dir: 输出目录路径
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.db = None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 必需字段定义
        self.required_fields = {
            'structure': ['atoms'],  # ASE Atoms对象
            'properties': ['eform', 'spin'],  # 目标性质
            'metadata': ['host', 'dopant', 'defecttype', 'converged']  # 元数据
        }
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'converged_records': 0,
            'complete_records': 0,
            'filtered_records': 0,
            'error_records': 0,
            'nan_eform_records': 0,
            'nan_spin_records': 0
        }
    
    def connect_database(self) -> bool:
        """
        连接数据库
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info(f"正在连接数据库: {self.db_path}")
            self.db = connect(str(self.db_path))
            
            # 获取总记录数
            self.stats['total_records'] = len(list(self.db.select()))
            logger.info(f"数据库连接成功，总记录数: {self.stats['total_records']}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def extract_atomic_structure(self, row) -> Optional[Atoms]:
        """
        提取原子结构
        
        Args:
            row: 数据库行对象
            
        Returns:
            Optional[Atoms]: ASE Atoms对象或None
        """
        try:
            atoms = row.toatoms()
            if atoms is None or len(atoms) == 0:
                logger.warning(f"Row {row.id}: 原子结构为空")
                return None
            return atoms
            
        except Exception as e:
            logger.error(f"Row {row.id}: 提取原子结构失败 - {e}")
            return None
    
    def extract_properties(self, row) -> Dict[str, float]:
        """
        提取目标性质 - 增强版，处理NaN值
        
        Args:
            row: 数据库行对象
            
        Returns:
            Dict[str, float]: 性质字典
        """
        properties = {}
        
        try:
            # 提取形成能 - 严格检查NaN和无效值
            eform = row.get('eform')
            if eform is None:
                logger.warning(f"Row {row.id}: 缺少形成能数据")
                return {}
            
            # 检查是否为NaN或无效数值
            try:
                eform_float = float(eform)
                if np.isnan(eform_float) or np.isinf(eform_float):
                    logger.warning(f"Row {row.id}: 形成能为NaN或Inf: {eform}")
                    self.stats['nan_eform_records'] += 1
                    return {}
                properties['eform'] = eform_float
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"Row {row.id}: 形成能值无效: {eform}, 错误: {e}")
                return {}
            
            # 提取磁矩 - 同样严格检查
            spin = row.get('spin')
            if spin is None:
                logger.warning(f"Row {row.id}: 缺少磁矩数据")
                return {}
            
            try:
                spin_float = float(spin)
                if np.isnan(spin_float) or np.isinf(spin_float):
                    logger.warning(f"Row {row.id}: 磁矩为NaN或Inf: {spin}")
                    self.stats['nan_spin_records'] += 1
                    return {}
                properties['spin'] = spin_float
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"Row {row.id}: 磁矩值无效: {spin}, 错误: {e}")
                return {}
            
            # 提取其他可选性质
            optional_props = ['en1', 'en2', 'conv1', 'conv2', 'hostenergy', 'dopant_chemical_potential']
            for prop in optional_props:
                value = row.get(prop)
                if value is not None:
                    try:
                        float_val = float(value)
                        if not np.isnan(float_val) and not np.isinf(float_val):
                            properties[prop] = float_val
                    except (ValueError, TypeError, OverflowError):
                        pass  # 忽略无效值
            
            return properties
            
        except Exception as e:
            logger.error(f"Row {row.id}: 提取性质数据失败 - {e}")
            return {}
    
    def extract_metadata(self, row) -> Dict[str, str]:
        """
        提取元数据
        
        Args:
            row: 数据库行对象
            
        Returns:
            Dict[str, str]: 元数据字典
        """
        metadata = {}
        
        try:
            # 必需元数据
            required_meta = ['host', 'dopant', 'defecttype', 'converged']
            for key in required_meta:
                value = row.get(key)
                if value is None:
                    logger.warning(f"Row {row.id}: 缺少必需元数据 {key}")
                    return {}
                
                # 特殊处理converged字段 - 转换为布尔值
                if key == 'converged':
                    if isinstance(value, str):
                        converged_bool = value.lower() in ('true', '1', 'yes', 't', 'y')
                        metadata[key] = converged_bool
                    elif isinstance(value, bool):
                        metadata[key] = value
                    elif isinstance(value, (int, float)):
                        metadata[key] = bool(value)
                    else:
                        logger.warning(f"Row {row.id}: converged字段类型未知: {type(value)}")
                        return {}
                else:
                    metadata[key] = str(value)
            
            # 可选元数据
            optional_meta = ['site', 'depth', 'extension_factor', 'supercell', 'host_spacegroup', 'name']
            for key in optional_meta:
                value = row.get(key)
                if value is not None:
                    metadata[key] = str(value)
            
            # 添加数据库ID
            metadata['db_id'] = str(row.id)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Row {row.id}: 提取元数据失败 - {e}")
            return {}
    
    def validate_convergence(self, row) -> bool:
        """
        验证收敛性
        
        Args:
            row: 数据库行对象
            
        Returns:
            bool: 是否收敛
        """
        converged = row.get('converged', False)
        if not converged:
            logger.debug(f"Row {row.id}: 未收敛，跳过")
            return False
        
        # 额外的收敛性检查
        conv2 = row.get('conv2')
        if conv2 is not None:
            try:
                conv2_float = float(conv2)
                if abs(conv2_float) > 1e-4:
                    logger.warning(f"Row {row.id}: 收敛阈值过大 ({conv2_float})")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"Row {row.id}: 收敛阈值无效")
                return False
        
        return True
    
    def validate_completeness(self, atoms: Atoms, properties: Dict, metadata: Dict) -> bool:
        """
        验证数据完整性
        
        Args:
            atoms: ASE Atoms对象
            properties: 性质字典
            metadata: 元数据字典
            
        Returns:
            bool: 数据是否完整
        """
        # 检查原子结构
        if atoms is None or len(atoms) == 0:
            return False
        
        # 检查必需性质
        required_props = ['eform', 'spin']
        for prop in required_props:
            if prop not in properties:
                return False
        
        # 检查必需元数据
        required_meta = ['host', 'dopant', 'defecttype', 'converged']
        for key in required_meta:
            if key not in metadata:
                return False
        
        return True
    
    def process_single_record(self, row) -> Optional[Dict]:
        """
        处理单条记录
        
        Args:
            row: 数据库行对象
            
        Returns:
            Optional[Dict]: 处理后的数据或None
        """
        try:
            # 收敛性验证
            if not self.validate_convergence(row):
                return None
            
            self.stats['converged_records'] += 1
            
            # 提取数据
            atoms = self.extract_atomic_structure(row)
            properties = self.extract_properties(row)
            metadata = self.extract_metadata(row)
            
            # 完整性检查
            if not self.validate_completeness(atoms, properties, metadata):
                logger.debug(f"Row {row.id}: 数据不完整，跳过")
                return None
            
            # 构建输出数据结构
            data_record = {
                'id': row.id,
                'atoms': atoms,
                'properties': properties,
                'metadata': metadata,
                'formula': atoms.get_chemical_formula(),
                'num_atoms': len(atoms)
            }
            
            self.stats['complete_records'] += 1
            logger.debug(f"Row {row.id}: 处理成功")
            
            return data_record
            
        except Exception as e:
            logger.error(f"Row {row.id}: 处理失败 - {e}")
            self.stats['error_records'] += 1
            return None
    
    def process_all_records(self, batch_size: int = 1000) -> List[Dict]:
        """
        处理所有记录
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            List[Dict]: 清洗后的数据列表
        """
        logger.info("开始处理所有记录...")
        
        filtered_data = []
        batch_data = []
        batch_count = 0
        
        # 用于统计的临时变量
        hosts_set = set()
        dopants_set = set()
        defect_types_set = set()
        
        try:
            # 遍历所有记录
            for idx, row in enumerate(self.db.select(), 1):
                # 处理单条记录
                processed_record = self.process_single_record(row)
                
                if processed_record is not None:
                    batch_data.append(processed_record)
                    filtered_data.append(processed_record)
                    
                    # 收集统计信息
                    metadata = processed_record['metadata']
                    hosts_set.add(metadata['host'])
                    dopants_set.add(metadata['dopant'])
                    defect_types_set.add(metadata['defecttype'])
                
                # 批处理保存
                if len(batch_data) >= batch_size:
                    self._save_batch(batch_data, batch_count)
                    batch_data = []
                    batch_count += 1
                
                # 进度报告
                if idx % 1000 == 0:
                    logger.info(f"已处理 {idx}/{self.stats['total_records']} 条记录")
            
            # 保存最后一批数据
            if batch_data:
                self._save_batch(batch_data, batch_count)
            
            # 更新最终统计信息
            self.stats['filtered_records'] = len(filtered_data)
            
            # 保存完整数据集和元数据
            self._save_complete_data(filtered_data, hosts_set, dopants_set, defect_types_set)
            
            logger.info("数据处理完成")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            raise
    
    def _save_batch(self, batch_data: List[Dict], batch_id: int):
        """
        保存批数据
        
        Args:
            batch_data: 批数据列表
            batch_id: 批ID
        """
        batch_file = self.output_dir / f"batch_{batch_id:04d}.pkl"
        
        try:
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"保存批数据: {batch_file} ({len(batch_data)} 条记录)")
            
        except Exception as e:
            logger.error(f"保存批数据失败: {e}")
            raise
    
    def _save_complete_data(self, complete_data: List[Dict], hosts_set: set, dopants_set: set, defect_types_set: set):
        """
        保存完整数据集
        
        Args:
            complete_data: 完整数据列表
            hosts_set: 宿主材料集合
            dopants_set: 掺杂元素集合
            defect_types_set: 缺陷类型集合
        """
        complete_file = self.output_dir / "complete_data_fixed.pkl"
        metadata_file = self.output_dir / "metadata_fixed.pkl"
        
        try:
            # 保存完整数据
            with open(complete_file, 'wb') as f:
                pickle.dump(complete_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 保存元数据和统计信息
            metadata = {
                'stats': self.stats,
                'required_fields': self.required_fields,
                'data_info': {
                    'total_structures': len(complete_data),
                    'unique_hosts': len(hosts_set),
                    'unique_dopants': len(dopants_set),
                    'defect_types': sorted(list(defect_types_set))
                }
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"保存完整数据: {complete_file} ({len(complete_data)} 条记录)")
            
        except Exception as e:
            logger.error(f"保存完整数据失败: {e}")
            raise
    
    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("数据处理统计信息:")
        logger.info(f"总记录数: {self.stats['total_records']}")
        logger.info(f"收敛记录数: {self.stats['converged_records']}")
        logger.info(f"完整记录数: {self.stats['complete_records']}")
        logger.info(f"筛选后记录数: {self.stats['filtered_records']}")
        logger.info(f"错误记录数: {self.stats['error_records']}")
        logger.info(f"NaN形成能记录数: {self.stats['nan_eform_records']}")
        logger.info(f"NaN磁矩记录数: {self.stats['nan_spin_records']}")
        
        if self.stats['total_records'] > 0:
            convergence_rate = (self.stats['converged_records'] / self.stats['total_records']) * 100
            completion_rate = (self.stats['complete_records'] / self.stats['total_records']) * 100
            final_rate = (self.stats['filtered_records'] / self.stats['total_records']) * 100
            logger.info(f"收敛率: {convergence_rate:.2f}%")
            logger.info(f"完整率: {completion_rate:.2f}%")
            logger.info(f"最终筛选率: {final_rate:.2f}%")
        
        logger.info("=" * 60)


def main():
    """主函数"""
    # 配置路径
    db_path = "data/raw/imp2d.db"  # 数据库路径
    output_dir = "data/processed"  # 输出目录
    
    logger.info("开始IMP2D数据库数据提取和清洗流程 (修复版)")
    
    try:
        # 创建解析器
        parser = IMP2DDataParser(db_path, output_dir)
        
        # 连接数据库
        if not parser.connect_database():
            logger.error("数据库连接失败，程序退出")
            return 1
        
        # 处理所有记录
        filtered_data = parser.process_all_records()
        
        # 打印统计信息
        parser.print_statistics()
        
        logger.info("数据提取和清洗流程完成")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"清洗后数据量: {len(filtered_data)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())