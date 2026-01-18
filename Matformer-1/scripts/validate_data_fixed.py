#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证脚本 - 验证修复后的数据

该脚本用于验证修复后的数据解析脚本的输出结果，包括：
1. 加载complete_data_fixed.pkl和metadata_fixed.pkl文件
2. 验证数据结构和完整性
3. 检查原子结构、性质和元数据的一致性
4. 生成数据质量报告
5. 输出统计信息（宿主材料分布、掺杂元素分布等）

要求：
- 包含完整的错误检查和日志记录
- 提供详细的数据质量分析
- 生成可视化统计图表（如果可能）
- 确保所有提取的数据都符合要求
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from ase import Atoms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts/data_validation_fixed.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        初始化验证器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.complete_data = None
        self.metadata = None
        
        # 验证结果
        self.validation_results = {
            'structure_validation': {'passed': 0, 'failed': 0, 'total': 0},
            'property_validation': {'passed': 0, 'failed': 0, 'total': 0},
            'metadata_validation': {'passed': 0, 'failed': 0, 'total': 0},
            'consistency_checks': {'passed': 0, 'failed': 0, 'total': 0}
        }
        
        # 数据质量统计
        self.quality_stats = {}
        
        # 必需字段定义
        self.required_fields = {
            'structure': ['atoms'],
            'properties': ['eform', 'spin'],
            'metadata': ['host', 'dopant', 'defecttype', 'converged', 'db_id']
        }
    
    def load_data(self) -> bool:
        """
        加载数据文件
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info("开始加载数据文件...")
            
            # 加载完整数据
            complete_data_file = self.data_dir / "complete_data_fixed.pkl"
            if not complete_data_file.exists():
                logger.error(f"完整数据文件不存在: {complete_data_file}")
                return False
            
            with open(complete_data_file, 'rb') as f:
                self.complete_data = pickle.load(f)
            
            logger.info(f"成功加载完整数据: {len(self.complete_data)} 条记录")
            
            # 加载元数据
            metadata_file = self.data_dir / "metadata_fixed.pkl"
            if not metadata_file.exists():
                logger.error(f"元数据文件不存在: {metadata_file}")
                return False
            
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info("成功加载元数据")
            
            return True
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_structure(self, record: Dict) -> bool:
        """
        验证原子结构
        
        Args:
            record: 数据记录
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查必需字段
            if 'atoms' not in record:
                logger.warning(f"Record {record.get('id', 'unknown')}: 缺少atoms字段")
                return False
            
            atoms = record['atoms']
            
            # 检查atoms对象类型
            if not isinstance(atoms, Atoms):
                logger.warning(f"Record {record.get('id', 'unknown')}: atoms不是ASE Atoms对象")
                return False
            
            # 检查原子数量
            if len(atoms) == 0:
                logger.warning(f"Record {record.get('id', 'unknown')}: 原子结构为空")
                return False
            
            # 检查化学公式
            try:
                formula = atoms.get_chemical_formula()
                if not formula or formula.strip() == "":
                    logger.warning(f"Record {record.get('id', 'unknown')}: 化学公式为空")
                    return False
            except Exception as e:
                logger.warning(f"Record {record.get('id', 'unknown')}: 获取化学公式失败 - {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Record {record.get('id', 'unknown')}: 结构验证失败 - {e}")
            return False
    
    def validate_properties(self, record: Dict) -> bool:
        """
        验证性质数据
        
        Args:
            record: 数据记录
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查properties字段
            if 'properties' not in record:
                logger.warning(f"Record {record.get('id', 'unknown')}: 缺少properties字段")
                return False
            
            properties = record['properties']
            
            # 检查必需性质
            required_props = ['eform', 'spin']
            for prop in required_props:
                if prop not in properties:
                    logger.warning(f"Record {record.get('id', 'unknown')}: 缺少性质 {prop}")
                    return False
                
                # 检查数值有效性
                value = properties[prop]
                if not isinstance(value, (int, float)):
                    logger.warning(f"Record {record.get('id', 'unknown')}: 性质 {prop} 不是数值类型")
                    return False
                
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Record {record.get('id', 'unknown')}: 性质 {prop} 为NaN或Inf")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Record {record.get('id', 'unknown')}: 性质验证失败 - {e}")
            return False
    
    def validate_metadata(self, record: Dict) -> bool:
        """
        验证元数据
        
        Args:
            record: 数据记录
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查metadata字段
            if 'metadata' not in record:
                logger.warning(f"Record {record.get('id', 'unknown')}: 缺少metadata字段")
                return False
            
            metadata = record['metadata']
            
            # 检查必需元数据
            required_meta = ['host', 'dopant', 'defecttype', 'converged', 'db_id']
            for key in required_meta:
                if key not in metadata:
                    logger.warning(f"Record {record.get('id', 'unknown')}: 缺少元数据 {key}")
                    return False
                
                # 检查converged字段
                if key == 'converged':
                    converged = metadata[key]
                    if not isinstance(converged, bool):
                        logger.warning(f"Record {record.get('id', 'unknown')}: converged不是布尔值")
                        return False
                    if not converged:
                        logger.warning(f"Record {record.get('id', 'unknown')}: converged为False")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Record {record.get('id', 'unknown')}: 元数据验证失败 - {e}")
            return False
    
    def validate_consistency(self, record: Dict) -> bool:
        """
        验证数据一致性
        
        Args:
            record: 数据记录
            
        Returns:
            bool: 验证是否通过
        """
        try:
            atoms = record['atoms']
            properties = record['properties']
            metadata = record['metadata']
            
            # 验证化学公式一致性
            atoms_formula = atoms.get_chemical_formula()
            record_formula = record.get('formula', '')
            
            if atoms_formula != record_formula:
                logger.warning(f"Record {record.get('id', 'unknown')}: 化学公式不一致 "
                             f"(atoms: {atoms_formula}, record: {record_formula})")
                return False
            
            # 验证原子数量一致性
            atoms_count = len(atoms)
            record_count = record.get('num_atoms', 0)
            
            if atoms_count != record_count:
                logger.warning(f"Record {record.get('id', 'unknown')}: 原子数量不一致 "
                             f"(atoms: {atoms_count}, record: {record_count})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Record {record.get('id', 'unknown')}: 一致性验证失败 - {e}")
            return False
    
    def validate_all_data(self) -> Dict:
        """
        验证所有数据
        
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证所有数据...")
        
        total_records = len(self.complete_data)
        
        # 重置验证结果
        for key in self.validation_results:
            self.validation_results[key] = {'passed': 0, 'failed': 0, 'total': 0}
        
        # 用于统计的变量
        hosts_counter = Counter()
        dopants_counter = Counter()
        defect_types_counter = Counter()
        eform_values = []
        spin_values = []
        
        # 验证每条记录
        for idx, record in enumerate(self.complete_data, 1):
            if idx % 1000 == 0:
                logger.info(f"已验证 {idx}/{total_records} 条记录")
            
            # 结构验证
            self.validation_results['structure_validation']['total'] += 1
            if self.validate_structure(record):
                self.validation_results['structure_validation']['passed'] += 1
            else:
                self.validation_results['structure_validation']['failed'] += 1
            
            # 性质验证
            self.validation_results['property_validation']['total'] += 1
            if self.validate_properties(record):
                self.validation_results['property_validation']['passed'] += 1
                
                # 收集统计信息
                properties = record['properties']
                eform_values.append(properties['eform'])
                spin_values.append(properties['spin'])
            else:
                self.validation_results['property_validation']['failed'] += 1
            
            # 元数据验证
            self.validation_results['metadata_validation']['total'] += 1
            if self.validate_metadata(record):
                self.validation_results['metadata_validation']['passed'] += 1
                
                # 收集统计信息
                metadata = record['metadata']
                hosts_counter[metadata['host']] += 1
                dopants_counter[metadata['dopant']] += 1
                defect_types_counter[metadata['defecttype']] += 1
            else:
                self.validation_results['metadata_validation']['failed'] += 1
            
            # 一致性检查
            self.validation_results['consistency_checks']['total'] += 1
            if self.validate_consistency(record):
                self.validation_results['consistency_checks']['passed'] += 1
            else:
                self.validation_results['consistency_checks']['failed'] += 1
        
        # 计算统计数据
        self.quality_stats = {
            'total_records': total_records,
            'structure_validation_rate': self.validation_results['structure_validation']['passed'] / total_records * 100,
            'property_validation_rate': self.validation_results['property_validation']['passed'] / total_records * 100,
            'metadata_validation_rate': self.validation_results['metadata_validation']['passed'] / total_records * 100,
            'consistency_rate': self.validation_results['consistency_checks']['passed'] / total_records * 100,
            'hosts_distribution': dict(hosts_counter.most_common(10)),
            'dopants_distribution': dict(dopants_counter.most_common(10)),
            'defect_types_distribution': dict(defect_types_counter),
            'eform_statistics': {
                'min': np.min(eform_values) if eform_values else None,
                'max': np.max(eform_values) if eform_values else None,
                'mean': np.mean(eform_values) if eform_values else None,
                'std': np.std(eform_values) if eform_values else None
            },
            'spin_statistics': {
                'min': np.min(spin_values) if spin_values else None,
                'max': np.max(spin_values) if spin_values else None,
                'mean': np.mean(spin_values) if spin_values else None,
                'std': np.std(spin_values) if spin_values else None
            }
        }
        
        logger.info("数据验证完成")
        
        return self.validation_results
    
    def validate_metadata_consistency(self) -> Dict:
        """
        验证元数据一致性
        
        Returns:
            Dict: 元数据一致性验证结果
        """
        logger.info("开始验证元数据一致性...")
        
        results = {
            'parser_stats_consistency': True,
            'record_count_match': True,
            'host_count_match': True,
            'dopant_count_match': True,
            'defect_types_match': True
        }
        
        try:
            # 获取解析器统计信息
            parser_stats = self.metadata.get('stats', {})
            parser_data_info = self.metadata.get('data_info', {})
            
            # 验证记录数
            parser_total = parser_stats.get('total_records', 0)
            actual_total = len(self.complete_data)
            
            if parser_total != actual_total:
                logger.warning(f"记录数不匹配: 解析器={parser_total}, 实际={actual_total}")
                results['record_count_match'] = False
                results['parser_stats_consistency'] = False
            
            # 验证宿主材料数
            parser_hosts = parser_data_info.get('unique_hosts', 0)
            actual_hosts = len(self.quality_stats['hosts_distribution'])
            
            if parser_hosts != actual_hosts:
                logger.warning(f"宿主材料数不匹配: 解析器={parser_hosts}, 实际={actual_hosts}")
                results['host_count_match'] = False
                results['parser_stats_consistency'] = False
            
            # 验证掺杂元素数
            parser_dopants = parser_data_info.get('unique_dopants', 0)
            actual_dopants = len(self.quality_stats['dopants_distribution'])
            
            if parser_dopants != actual_dopants:
                logger.warning(f"掺杂元素数不匹配: 解析器={parser_dopants}, 实际={actual_dopants}")
                results['dopant_count_match'] = False
                results['parser_stats_consistency'] = False
            
            # 验证缺陷类型
            parser_defect_types = set(parser_data_info.get('defect_types', []))
            actual_defect_types = set(self.quality_stats['defect_types_distribution'].keys())
            
            if parser_defect_types != actual_defect_types:
                logger.warning(f"缺陷类型不匹配: 解析器={parser_defect_types}, 实际={actual_defect_types}")
                results['defect_types_match'] = False
                results['parser_stats_consistency'] = False
            
            logger.info("元数据一致性验证完成")
            
        except Exception as e:
            logger.error(f"元数据一致性验证失败: {e}")
            results['parser_stats_consistency'] = False
        
        return results
    
    def generate_report(self, output_file: str = "validation_report_fixed.txt"):
        """
        生成验证报告
        
        Args:
            output_file: 输出文件名
        """
        logger.info("开始生成验证报告...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("IMP2D数据质量验证报告 (修复版)\n")
                f.write("=" * 60 + "\n\n")
                
                # 总体统计
                f.write("1. 总体统计\n")
                f.write("-" * 30 + "\n")
                f.write(f"总记录数: {self.quality_stats['total_records']}\n")
                f.write(f"结构验证通过率: {self.quality_stats['structure_validation_rate']:.1f}%\n")
                f.write(f"性质验证通过率: {self.quality_stats['property_validation_rate']:.1f}%\n")
                f.write(f"元数据验证通过率: {self.quality_stats['metadata_validation_rate']:.1f}%\n")
                f.write(f"一致性检查通过率: {self.quality_stats['consistency_rate']:.1f}%\n\n")
                
                # 详细验证结果
                f.write("2. 详细验证结果\n")
                f.write("-" * 30 + "\n")
                for check_name, results in self.validation_results.items():
                    f.write(f"{check_name}:\n")
                    f.write(f"  总计: {results['total']}\n")
                    f.write(f"  通过: {results['passed']}\n")
                    f.write(f"  失败: {results['failed']}\n")
                    f.write(f"  通过率: {results['passed']/results['total']*100:.1f}%\n\n")
                
                # 统计信息
                f.write("3. 统计信息\n")
                f.write("-" * 30 + "\n")
                
                # 形成能统计
                eform_stats = self.quality_stats['eform_statistics']
                if eform_stats['min'] is not None:
                    f.write("形成能统计:\n")
                    f.write(f"  最小值: {eform_stats['min']:.3f} eV\n")
                    f.write(f"  最大值: {eform_stats['max']:.3f} eV\n")
                    f.write(f"  平均值: {eform_stats['mean']:.3f} eV\n")
                    f.write(f"  标准差: {eform_stats['std']:.3f} eV\n\n")
                
                # 磁矩统计
                spin_stats = self.quality_stats['spin_statistics']
                if spin_stats['min'] is not None:
                    f.write("磁矩统计:\n")
                    f.write(f"  最小值: {spin_stats['min']:.3f}\n")
                    f.write(f"  最大值: {spin_stats['max']:.3f}\n")
                    f.write(f"  平均值: {spin_stats['mean']:.3f}\n")
                    f.write(f"  标准差: {spin_stats['std']:.3f}\n\n")
                
                # 宿主材料分布
                f.write("宿主材料分布 (前10):\n")
                for host, count in self.quality_stats['hosts_distribution'].items():
                    f.write(f"  {host}: {count}\n")
                f.write("\n")
                
                # 掺杂元素分布
                f.write("掺杂元素分布 (前10):\n")
                for dopant, count in self.quality_stats['dopants_distribution'].items():
                    f.write(f"  {dopant}: {count}\n")
                f.write("\n")
                
                # 缺陷类型分布
                f.write("缺陷类型分布:\n")
                for defect_type, count in self.quality_stats['defect_types_distribution'].items():
                    f.write(f"  {defect_type}: {count}\n")
                f.write("\n")
                
                # 数据质量评级
                overall_pass_rate = (
                    self.quality_stats['structure_validation_rate'] +
                    self.quality_stats['property_validation_rate'] +
                    self.quality_stats['metadata_validation_rate'] +
                    self.quality_stats['consistency_rate']
                ) / 4
                
                if overall_pass_rate >= 95:
                    quality_grade = "优秀 (A)"
                elif overall_pass_rate >= 85:
                    quality_grade = "良好 (B)"
                elif overall_pass_rate >= 75:
                    quality_grade = "合格 (C)"
                else:
                    quality_grade = "需要改进 (D)"
                
                f.write("4. 数据质量评级\n")
                f.write("-" * 30 + "\n")
                f.write(f"总体通过率: {overall_pass_rate:.1f}%\n")
                f.write(f"质量评级: {quality_grade}\n")
                
                # 元数据一致性检查
                metadata_consistency = self.validate_metadata_consistency()
                f.write("\n5. 元数据一致性检查\n")
                f.write("-" * 30 + "\n")
                f.write(f"解析器统计一致性: {metadata_consistency['parser_stats_consistency']}\n")
                f.write(f"记录数匹配: {metadata_consistency['record_count_match']}\n")
                f.write(f"宿主材料数匹配: {metadata_consistency['host_count_match']}\n")
                f.write(f"掺杂元素数匹配: {metadata_consistency['dopant_count_match']}\n")
                f.write(f"缺陷类型匹配: {metadata_consistency['defect_types_match']}\n")
            
            logger.info(f"验证报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"生成验证报告失败: {e}")
            raise
    
    def generate_visualizations(self, output_dir: str = "validation_plots"):
        """
        生成可视化图表
        
        Args:
            output_dir: 输出目录
        """
        logger.info("开始生成可视化图表...")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 形成能分布直方图
            if self.quality_stats['eform_statistics']['min'] is not None:
                plt.figure(figsize=(10, 6))
                eform_values = []
                for record in self.complete_data:
                    if 'properties' in record and 'eform' in record['properties']:
                        eform_values.append(record['properties']['eform'])
                
                plt.hist(eform_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('Formation Energy (eV)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Formation Energy')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_path / 'formation_energy_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 磁矩分布直方图
            if self.quality_stats['spin_statistics']['min'] is not None:
                plt.figure(figsize=(10, 6))
                spin_values = []
                for record in self.complete_data:
                    if 'properties' in record and 'spin' in record['properties']:
                        spin_values.append(record['properties']['spin'])
                
                plt.hist(spin_values, bins=50, alpha=0.7, color='green', edgecolor='black')
                plt.xlabel('Magnetic Moment')
                plt.ylabel('Frequency')
                plt.title('Distribution of Magnetic Moment')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_path / 'magnetic_moment_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 宿主材料分布饼图
            if self.quality_stats['hosts_distribution']:
                plt.figure(figsize=(10, 8))
                hosts = list(self.quality_stats['hosts_distribution'].keys())[:10]  # 只显示前10个
                counts = [self.quality_stats['hosts_distribution'][host] for host in hosts]
                
                plt.pie(counts, labels=hosts, autopct='%1.1f%%', startangle=90)
                plt.title('Top 10 Host Materials Distribution')
                plt.axis('equal')
                plt.savefig(output_path / 'host_materials_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 缺陷类型分布条形图
            if self.quality_stats['defect_types_distribution']:
                plt.figure(figsize=(10, 6))
                defect_types = list(self.quality_stats['defect_types_distribution'].keys())
                counts = list(self.quality_stats['defect_types_distribution'].values())
                
                bars = plt.bar(defect_types, counts, color='skyblue', edgecolor='black')
                plt.xlabel('Defect Type')
                plt.ylabel('Count')
                plt.title('Distribution of Defect Types')
                plt.xticks(rotation=45)
                
                # 在条形上添加数值
                for bar, count in zip(bars, counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig(output_path / 'defect_types_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"可视化图表已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
            logger.warning("跳过可视化图表生成")


def main():
    """主函数"""
    logger.info("开始数据验证流程 (修复版)")
    
    try:
        # 创建验证器
        validator = DataValidator()
        
        # 加载数据
        if not validator.load_data():
            logger.error("数据加载失败，程序退出")
            return 1
        
        # 验证所有数据
        validation_results = validator.validate_all_data()
        
        # 生成验证报告
        validator.generate_report()
        
        # 生成可视化图表（可选）
        try:
            validator.generate_visualizations()
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
        
        logger.info("数据验证流程完成")
        
        # 打印简要统计
        logger.info("=" * 60)
        logger.info("数据验证结果摘要:")
        logger.info(f"总记录数: {validator.quality_stats['total_records']}")
        logger.info(f"结构验证通过率: {validator.quality_stats['structure_validation_rate']:.1f}%")
        logger.info(f"性质验证通过率: {validator.quality_stats['property_validation_rate']:.1f}%")
        logger.info(f"元数据验证通过率: {validator.quality_stats['metadata_validation_rate']:.1f}%")
        logger.info(f"一致性检查通过率: {validator.quality_stats['consistency_rate']:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())