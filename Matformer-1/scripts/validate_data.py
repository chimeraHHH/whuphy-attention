#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMP2D数据验证脚本

用于验证01_data_parser.py输出的数据质量和完整性
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from collections import Counter, defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts/data_validation.log', encoding='utf-8')
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
        self.validation_results = {}
        
    def load_data(self) -> bool:
        """
        加载数据文件
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载完整数据
            complete_file = self.data_dir / "complete_data.pkl"
            if not complete_file.exists():
                logger.error(f"完整数据文件不存在: {complete_file}")
                return False
                
            with open(complete_file, 'rb') as f:
                self.complete_data = pickle.load(f)
            logger.info(f"成功加载完整数据: {len(self.complete_data)} 条记录")
            
            # 加载元数据
            metadata_file = self.data_dir / "metadata.pkl"
            if not metadata_file.exists():
                logger.error(f"元数据文件不存在: {metadata_file}")
                return False
                
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info("成功加载元数据")
            
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def validate_data_structure(self) -> Dict[str, Any]:
        """
        验证数据结构
        
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证数据结构...")
        results = {
            'total_records': 0,
            'structure_valid': 0,
            'properties_valid': 0,
            'metadata_valid': 0,
            'structure_errors': [],
            'property_errors': [],
            'metadata_errors': []
        }
        
        if not self.complete_data:
            results['structure_errors'].append("数据未加载")
            return results
            
        results['total_records'] = len(self.complete_data)
        
        for idx, record in enumerate(self.complete_data):
            record_id = record.get('id', f'index_{idx}')
            
            # 验证基本结构
            required_keys = ['id', 'atoms', 'properties', 'metadata', 'formula', 'num_atoms']
            missing_keys = [key for key in required_keys if key not in record]
            if missing_keys:
                results['structure_errors'].append(f"记录 {record_id}: 缺少键 {missing_keys}")
                continue
                
            results['structure_valid'] += 1
            
            # 验证原子结构
            atoms = record.get('atoms')
            if not atoms:
                results['structure_errors'].append(f"记录 {record_id}: 原子结构为空")
                continue
                
            # 验证性质数据
            properties = record.get('properties', {})
            required_props = ['eform', 'spin']
            missing_props = [prop for prop in required_props if prop not in properties]
            if missing_props:
                results['property_errors'].append(f"记录 {record_id}: 缺少性质 {missing_props}")
            else:
                # 验证数值有效性
                try:
                    eform = float(properties.get('eform', np.nan))
                    spin = float(properties.get('spin', np.nan))
                    if np.isnan(eform) or np.isnan(spin):
                        results['property_errors'].append(f"记录 {record_id}: 性质值为NaN")
                    else:
                        results['properties_valid'] += 1
                except (ValueError, TypeError):
                    results['property_errors'].append(f"记录 {record_id}: 性质值类型错误")
            
            # 验证元数据
            metadata = record.get('metadata', {})
            required_meta = ['host', 'dopant', 'defecttype', 'converged', 'db_id']
            missing_meta = [meta for meta in required_meta if meta not in metadata]
            if missing_meta:
                results['metadata_errors'].append(f"记录 {record_id}: 缺少元数据 {missing_meta}")
            else:
                results['metadata_valid'] += 1
        
        # 计算通过率
        results['structure_pass_rate'] = (results['structure_valid'] / results['total_records']) * 100 if results['total_records'] > 0 else 0
        results['properties_pass_rate'] = (results['properties_valid'] / results['total_records']) * 100 if results['total_records'] > 0 else 0
        results['metadata_pass_rate'] = (results['metadata_valid'] / results['total_records']) * 100 if results['total_records'] > 0 else 0
        
        logger.info(f"数据结构验证完成:")
        logger.info(f"  总记录数: {results['total_records']}")
        logger.info(f"  结构验证通过率: {results['structure_pass_rate']:.1f}%")
        logger.info(f"  性质验证通过率: {results['properties_pass_rate']:.1f}%")
        logger.info(f"  元数据验证通过率: {results['metadata_pass_rate']:.1f}%")
        
        return results
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """
        验证数据一致性
        
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证数据一致性...")
        results = {
            'formula_consistency': 0,
            'num_atoms_consistency': 0,
            'convergence_consistency': 0,
            'consistency_errors': []
        }
        
        if not self.complete_data:
            results['consistency_errors'].append("数据未加载")
            return results
        
        for idx, record in enumerate(self.complete_data):
            record_id = record.get('id', f'index_{idx}')
            
            # 验证化学式一致性
            atoms = record.get('atoms')
            formula_from_atoms = atoms.get_chemical_formula() if atoms else None
            formula_from_record = record.get('formula')
            
            if formula_from_atoms and formula_from_record:
                if formula_from_atoms == formula_from_record:
                    results['formula_consistency'] += 1
                else:
                    results['consistency_errors'].append(
                        f"记录 {record_id}: 化学式不一致 (atoms: {formula_from_atoms}, record: {formula_from_record})"
                    )
            
            # 验证原子数一致性
            num_atoms_from_atoms = len(atoms) if atoms else None
            num_atoms_from_record = record.get('num_atoms')
            
            if num_atoms_from_atoms is not None and num_atoms_from_record is not None:
                if num_atoms_from_atoms == num_atoms_from_record:
                    results['num_atoms_consistency'] += 1
                else:
                    results['consistency_errors'].append(
                        f"记录 {record_id}: 原子数不一致 (atoms: {num_atoms_from_atoms}, record: {num_atoms_from_record})"
                    )
            
            # 验证收敛性一致性
            metadata = record.get('metadata', {})
            converged_in_meta = metadata.get('converged')
            
            # 检查是否所有记录都标记为收敛（因为我们只提取了收敛数据）
            if converged_in_meta is True:
                results['convergence_consistency'] += 1
            else:
                results['consistency_errors'].append(f"记录 {record_id}: 收敛性标记不一致")
        
        total_records = len(self.complete_data)
        results['formula_consistency_rate'] = (results['formula_consistency'] / total_records) * 100 if total_records > 0 else 0
        results['num_atoms_consistency_rate'] = (results['num_atoms_consistency'] / total_records) * 100 if total_records > 0 else 0
        results['convergence_consistency_rate'] = (results['convergence_consistency'] / total_records) * 100 if total_records > 0 else 0
        
        logger.info(f"数据一致性验证完成:")
        logger.info(f"  化学式一致性: {results['formula_consistency_rate']:.1f}%")
        logger.info(f"  原子数一致性: {results['num_atoms_consistency_rate']:.1f}%")
        logger.info(f"  收敛性一致性: {results['convergence_consistency_rate']:.1f}%")
        
        return results
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        生成统计信息
        
        Returns:
            Dict: 统计结果
        """
        logger.info("开始生成统计信息...")
        results = {
            'host_distribution': {},
            'dopant_distribution': {},
            'defect_type_distribution': {},
            'formation_energy_stats': {},
            'spin_stats': {},
            'structure_size_stats': {}
        }
        
        if not self.complete_data:
            logger.warning("数据未加载，无法生成统计信息")
            return results
        
        # 基本统计
        hosts = []
        dopants = []
        defect_types = []
        formation_energies = []
        spins = []
        structure_sizes = []
        
        for record in self.complete_data:
            metadata = record.get('metadata', {})
            properties = record.get('properties', {})
            atoms = record.get('atoms')
            
            # 收集基本数据
            hosts.append(metadata.get('host', 'unknown'))
            dopants.append(metadata.get('dopant', 'unknown'))
            defect_types.append(metadata.get('defecttype', 'unknown'))
            
            # 收集性质数据
            eform = properties.get('eform')
            if eform is not None:
                formation_energies.append(float(eform))
            
            spin = properties.get('spin')
            if spin is not None:
                spins.append(float(spin))
            
            # 收集结构数据
            if atoms:
                structure_sizes.append(len(atoms))
        
        # 生成分布统计
        results['host_distribution'] = dict(Counter(hosts).most_common(20))
        results['dopant_distribution'] = dict(Counter(dopants).most_common(20))
        results['defect_type_distribution'] = dict(Counter(defect_types))
        
        # 生成数值统计
        if formation_energies:
            results['formation_energy_stats'] = {
                'count': len(formation_energies),
                'mean': np.mean(formation_energies),
                'std': np.std(formation_energies),
                'min': np.min(formation_energies),
                'max': np.max(formation_energies),
                'median': np.median(formation_energies)
            }
        
        if spins:
            results['spin_stats'] = {
                'count': len(spins),
                'mean': np.mean(spins),
                'std': np.std(spins),
                'min': np.min(spins),
                'max': np.max(spins),
                'median': np.median(spins)
            }
        
        if structure_sizes:
            results['structure_size_stats'] = {
                'count': len(structure_sizes),
                'mean': np.mean(structure_sizes),
                'std': np.std(structure_sizes),
                'min': np.min(structure_sizes),
                'max': np.max(structure_sizes),
                'median': np.median(structure_sizes)
            }
        
        # 打印统计摘要
        logger.info("统计信息生成完成:")
        logger.info(f"  宿主材料种类: {len(results['host_distribution'])}")
        logger.info(f"  掺杂元素种类: {len(results['dopant_distribution'])}")
        logger.info(f"  缺陷类型: {list(results['defect_type_distribution'].keys())}")
        
        if results['formation_energy_stats']:
            stats = results['formation_energy_stats']
            logger.info(f"  形成能范围: {stats['min']:.2f} to {stats['max']:.2f} eV")
            logger.info(f"  形成能均值: {stats['mean']:.2f} ± {stats['std']:.2f} eV")
        
        if results['spin_stats']:
            stats = results['spin_stats']
            logger.info(f"  磁矩范围: {stats['min']:.2f} to {stats['max']:.2f} μB")
            logger.info(f"  磁矩均值: {stats['mean']:.2f} ± {stats['std']:.2f} μB")
        
        return results
    
    def validate_metadata_consistency(self) -> Dict[str, Any]:
        """
        验证元数据一致性
        
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证元数据一致性...")
        results = {
            'parser_stats': {},
            'actual_stats': {},
            'consistency_check': True,
            'inconsistencies': []
        }
        
        if not self.metadata or not self.complete_data:
            results['consistency_check'] = False
            results['inconsistencies'].append("元数据或完整数据未加载")
            return results
        
        # 获取解析器统计信息
        results['parser_stats'] = self.metadata.get('stats', {})
        
        # 计算实际统计信息
        actual_stats = {
            'total_structures': len(self.complete_data),
            'unique_hosts': len(set(d['metadata']['host'] for d in self.complete_data)),
            'unique_dopants': len(set(d['metadata']['dopant'] for d in self.complete_data)),
            'defect_types': list(set(d['metadata']['defecttype'] for d in self.complete_data))
        }
        results['actual_stats'] = actual_stats
        
        # 验证一致性
        parser_total = results['parser_stats'].get('filtered_records', 0)
        actual_total = actual_stats['total_structures']
        
        if parser_total != actual_total:
            results['consistency_check'] = False
            results['inconsistencies'].append(
                f"记录数量不一致 (parser: {parser_total}, actual: {actual_total})"
            )
        
        parser_hosts = results['parser_stats'].get('data_info', {}).get('unique_hosts', 0)
        actual_hosts = actual_stats['unique_hosts']
        
        if parser_hosts != actual_hosts:
            results['consistency_check'] = False
            results['inconsistencies'].append(
                f"宿主材料数量不一致 (parser: {parser_hosts}, actual: {actual_hosts})"
            )
        
        logger.info(f"元数据一致性验证:")
        logger.info(f"  记录数量一致性: {'✓' if results['consistency_check'] else '✗'}")
        
        if not results['consistency_check']:
            for inconsistency in results['inconsistencies']:
                logger.warning(f"  不一致: {inconsistency}")
        
        return results
    
    def generate_validation_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 验证报告
        """
        logger.info("生成验证报告...")
        
        # 执行所有验证
        structure_validation = self.validate_data_structure()
        consistency_validation = self.validate_data_consistency()
        statistics = self.generate_statistics()
        metadata_validation = self.validate_metadata_consistency()
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("IMP2D数据验证报告")
        report.append("=" * 80)
        report.append("")
        
        # 数据概览
        report.append("数据概览:")
        report.append(f"  总记录数: {structure_validation['total_records']}")
        report.append(f"  结构验证通过率: {structure_validation['structure_pass_rate']:.1f}%")
        report.append(f"  性质验证通过率: {structure_validation['properties_pass_rate']:.1f}%")
        report.append(f"  元数据验证通过率: {structure_validation['metadata_pass_rate']:.1f}%")
        report.append("")
        
        # 一致性检查
        report.append("一致性检查:")
        report.append(f"  化学式一致性: {consistency_validation['formula_consistency_rate']:.1f}%")
        report.append(f"  原子数一致性: {consistency_validation['num_atoms_consistency_rate']:.1f}%")
        report.append(f"  元数据一致性: {'✓' if metadata_validation['consistency_check'] else '✗'}")
        report.append("")
        
        # 统计信息
        report.append("统计信息:")
        report.append(f"  宿主材料种类: {len(statistics['host_distribution'])}")
        report.append(f"  掺杂元素种类: {len(statistics['dopant_distribution'])}")
        
        if statistics['formation_energy_stats']:
            fe_stats = statistics['formation_energy_stats']
            report.append(f"  形成能: {fe_stats['mean']:.2f} ± {fe_stats['std']:.2f} eV (范围: {fe_stats['min']:.2f} to {fe_stats['max']:.2f})")
        
        if statistics['spin_stats']:
            spin_stats = statistics['spin_stats']
            report.append(f"  磁矩: {spin_stats['mean']:.2f} ± {spin_stats['std']:.2f} μB (范围: {spin_stats['min']:.2f} to {spin_stats['max']:.2f})")
        
        report.append("")
        
        # 错误汇总
        total_errors = (len(structure_validation['structure_errors']) + 
                       len(structure_validation['property_errors']) + 
                       len(structure_validation['metadata_errors']) +
                       len(consistency_validation['consistency_errors']))
        
        if total_errors > 0:
            report.append("错误汇总:")
            report.append(f"  结构错误: {len(structure_validation['structure_errors'])}")
            report.append(f"  性质错误: {len(structure_validation['property_errors'])}")
            report.append(f"  元数据错误: {len(structure_validation['metadata_errors'])}")
            report.append(f"  一致性错误: {len(consistency_validation['consistency_errors'])}")
            report.append("")
        
        # 质量评级
        overall_quality = self._assess_overall_quality(
            structure_validation, consistency_validation, metadata_validation
        )
        report.append(f"数据质量评级: {overall_quality}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _assess_overall_quality(self, structure_val: Dict, consistency_val: Dict, metadata_val: Dict) -> str:
        """
        评估整体数据质量
        
        Returns:
            str: 质量评级
        """
        scores = []
        
        # 结构质量分数
        structure_score = (
            structure_val['structure_pass_rate'] * 0.4 +
            structure_val['properties_pass_rate'] * 0.3 +
            structure_val['metadata_pass_rate'] * 0.3
        )
        scores.append(structure_score)
        
        # 一致性分数
        consistency_score = (
            consistency_val['formula_consistency_rate'] * 0.5 +
            consistency_val['num_atoms_consistency_rate'] * 0.5
        )
        scores.append(consistency_score)
        
        # 元数据一致性分数
        metadata_score = 100.0 if metadata_val['consistency_check'] else 0.0
        scores.append(metadata_score)
        
        # 计算平均分
        average_score = np.mean(scores)
        
        if average_score >= 95:
            return "优秀 (A+)"
        elif average_score >= 90:
            return "良好 (A)"
        elif average_score >= 80:
            return "中等 (B)"
        elif average_score >= 70:
            return "及格 (C)"
        else:
            return "需要改进 (D)"
    
    def save_report(self, report: str, filename: str = "validation_report.txt"):
        """
        保存验证报告
        
        Args:
            report: 验证报告内容
            filename: 文件名
        """
        try:
            report_file = self.data_dir / filename
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"验证报告已保存: {report_file}")
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")


def main():
    """主函数"""
    logger.info("开始IMP2D数据验证流程")
    
    try:
        # 创建验证器
        validator = DataValidator()
        
        # 加载数据
        if not validator.load_data():
            logger.error("数据加载失败，程序退出")
            return 1
        
        # 生成验证报告
        report = validator.generate_validation_report()
        
        # 打印报告
        print("\n" + report)
        
        # 保存报告
        validator.save_report(report)
        
        logger.info("数据验证流程完成")
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())