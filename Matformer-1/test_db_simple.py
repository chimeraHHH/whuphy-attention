#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的数据解析测试脚本"""

import logging
from ase.db import connect
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database():
    """测试数据库连接和基本数据提取"""
    db_path = "data/raw/imp2d.db"
    
    logger.info(f"正在连接数据库: {db_path}")
    
    try:
        # 连接数据库
        db = connect(db_path)
        logger.info("数据库连接成功")
        
        # 获取总记录数
        total_records = len(list(db.select()))
        logger.info(f"总记录数: {total_records}")
        
        # 测试单行数据提取
        logger.info("测试单行数据提取...")
        row = list(db.select(limit=1))[0]
        
        logger.info(f"Row ID: {row.id}")
        logger.info(f"Available keys: {row._keys}")
        logger.info(f"Converged: {row.get('converged', 'not found')}")
        logger.info(f"Eform: {row.get('eform', 'not found')}")
        logger.info(f"Spin: {row.get('spin', 'not found')}")
        
        # 测试原子结构转换
        atoms = row.toatoms()
        logger.info(f"Formula: {atoms.get_chemical_formula()}")
        logger.info(f"Number of atoms: {len(atoms)}")
        
        # 测试收敛性筛选
        logger.info("测试收敛性筛选...")
        converged_count = 0
        for row in db.select(limit=100):  # 测试前100条记录
            if row.get('converged', False):
                converged_count += 1
        
        logger.info(f"前100条记录中收敛的记录数: {converged_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database()
    if success:
        logger.info("数据库测试完成！")
    else:
        logger.error("数据库测试失败！")