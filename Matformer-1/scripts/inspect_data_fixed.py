#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据检查脚本 - 检查修复后的数据结构

该脚本用于检查修复后的数据处理结果，包括：
1. 检查complete_data_fixed.pkl的结构
2. 检查metadata_fixed.pkl的内容
3. 分析元数据验证失败的原因
4. 提供修复建议
"""

import pickle
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_complete_data():
    """检查完整数据文件"""
    logger.info("=== 检查 complete_data_fixed.pkl ===")
    
    try:
        with open('data/processed/complete_data_fixed.pkl', 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"总记录数: {len(data)}")
        
        if len(data) > 0:
            # 检查第一条记录
            first_record = data[0]
            logger.info(f"第一条记录的键: {list(first_record.keys())}")
            
            # 检查结构
            for key, value in first_record.items():
                logger.info(f"  {key}: {type(value)}")
                if key == 'properties':
                    logger.info(f"    properties键: {list(value.keys())}")
                elif key == 'metadata':
                    logger.info(f"    metadata键: {list(value.keys())}")
                elif key == 'atoms':
                    logger.info(f"    atoms类型: {type(value)}")
                    logger.info(f"    atoms化学式: {value.get_chemical_formula()}")
                    logger.info(f"    atoms原子数: {len(value)}")
            
            # 检查元数据字段
            if 'metadata' in first_record:
                metadata = first_record['metadata']
                logger.info("\n元数据字段详情:")
                for key, value in metadata.items():
                    logger.info(f"  {key}: {value} ({type(value)})")
        
        # 检查收敛字段
        converged_count = 0
        for record in data:
            if 'metadata' in record and 'converged' in record['metadata']:
                if record['metadata']['converged']:
                    converged_count += 1
        
        logger.info(f"\n收敛记录数: {converged_count}/{len(data)}")
        
    except Exception as e:
        logger.error(f"检查complete_data_fixed.pkl失败: {e}")
        import traceback
        traceback.print_exc()


def check_metadata():
    """检查元数据文件"""
    logger.info("\n=== 检查 metadata_fixed.pkl ===")
    
    try:
        with open('data/processed/metadata_fixed.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"元数据文件键: {list(metadata.keys())}")
        
        # 检查stats
        if 'stats' in metadata:
            stats = metadata['stats']
            logger.info("统计信息:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        # 检查data_info
        if 'data_info' in metadata:
            data_info = metadata['data_info']
            logger.info("\n数据信息:")
            for key, value in data_info.items():
                logger.info(f"  {key}: {value}")
        
        # 检查required_fields
        if 'required_fields' in metadata:
            required_fields = metadata['required_fields']
            logger.info("\n必需字段:")
            for category, fields in required_fields.items():
                logger.info(f"  {category}: {fields}")
        
    except Exception as e:
        logger.error(f"检查metadata_fixed.pkl失败: {e}")
        import traceback
        traceback.print_exc()


def check_specific_issues():
    """检查具体问题"""
    logger.info("\n=== 检查具体问题 ===")
    
    try:
        # 加载数据
        with open('data/processed/complete_data_fixed.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # 检查元数据验证失败的原因
        metadata_failures = []
        
        for i, record in enumerate(data[:10]):  # 检查前10条记录
            issues = []
            
            if 'metadata' not in record:
                issues.append("缺少metadata字段")
            else:
                metadata = record['metadata']
                
                # 检查必需字段
                required_fields = ['host', 'dopant', 'defecttype', 'converged', 'db_id']
                for field in required_fields:
                    if field not in metadata:
                        issues.append(f"缺少{field}字段")
                    elif field == 'converged':
                        converged = metadata[field]
                        if not isinstance(converged, bool):
                            issues.append(f"converged不是布尔值: {type(converged)} = {converged}")
                        elif not converged:
                            issues.append(f"converged为False")
            
            if issues:
                metadata_failures.append(f"记录{i+1}: {'; '.join(issues)}")
        
        if metadata_failures:
            logger.info("元数据验证失败示例:")
            for failure in metadata_failures:
                logger.info(f"  {failure}")
        else:
            logger.info("前10条记录没有明显的元数据问题")
        
        # 检查converged字段的具体值
        converged_values = []
        for record in data[:100]:
            if 'metadata' in record and 'converged' in record['metadata']:
                converged_values.append(record['metadata']['converged'])
        
        logger.info(f"\n前100条记录的converged值分布:")
        value_counts = {}
        for value in converged_values:
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1
        
        for value, count in value_counts.items():
            logger.info(f"  converged={value}: {count}条记录")
        
    except Exception as e:
        logger.error(f"检查具体问题失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    logger.info("开始检查修复后的数据结构")
    
    try:
        # 检查完整数据
        check_complete_data()
        
        # 检查元数据
        check_metadata()
        
        # 检查具体问题
        check_specific_issues()
        
        logger.info("\n数据检查完成")
        
    except Exception as e:
        logger.error(f"数据检查失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())