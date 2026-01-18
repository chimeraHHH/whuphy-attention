#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMP2D数据库数据提取与清洗（宽松收敛阈值版本）

功能：
- 提供可调节的 conv2 收敛阈值（如 1e-4、1e-3 或禁用）
- 保持 converged=True 的基本要求不变
- 对比扩大阈值前后的数据量与分布变化
- 生成详细的对比统计报告和中间格式输出（pickle）

运行示例：
python scripts/01_data_parser_relaxed.py --db-path data/raw/imp2d.db \
    --thresholds 1e-4 1e-3 none \
    --output-dir data/processed
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase.db import connect
from ase import Atoms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts/data_parser_relaxed.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("IMP2DRelaxedParser")


def find_db_path(db_path_arg: Optional[str]) -> Optional[Path]:
    """带回退逻辑查找数据库路径。"""
    candidates: List[Path] = []
    if db_path_arg:
        candidates.append(Path(db_path_arg))
    candidates.extend([
        Path("data/raw/imp2d.db"),
        Path("imp2d.db"),
    ])
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def validate_convergence(row: Any, conv2_threshold: Optional[float]) -> bool:
    """
    收敛性验证：
    - 必须 row['converged'] 为 True
    - 若提供 conv2_threshold，则要求 |conv2| <= conv2_threshold
    - 若 conv2_threshold 为 None，则不检查 conv2
    """
    converged = row.get('converged', False)
    if not converged:
        return False
    conv2 = row.get('conv2')
    if conv2_threshold is not None and conv2 is not None:
        try:
            if abs(float(conv2)) > float(conv2_threshold):
                return False
        except Exception:
            # 无法解析 conv2 则保守跳过该额外检查
            return False
    return True


def extract_atoms(row: Any) -> Optional[Atoms]:
    try:
        atoms = row.toatoms()
        if atoms is None or len(atoms) == 0:
            return None
        return atoms
    except Exception:
        return None


def extract_properties(row: Any) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    # 必需性质
    eform = row.get('eform')
    spin = row.get('spin')
    if eform is None or spin is None:
        return {}
    try:
        props['eform'] = float(eform)
    except Exception:
        return {}
    try:
        props['spin'] = float(spin)
    except Exception:
        return {}
    # 可选性质
    for key in ['en1', 'en2', 'conv1', 'conv2', 'hostenergy', 'dopant_chemical_potential']:
        val = row.get(key)
        if val is not None:
            props[key] = val
    return props


def extract_metadata(row: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in ['host', 'dopant', 'defecttype', 'converged']:
        val = row.get(key)
        if val is None:
            return {}
        meta[key] = val
    for key in ['site', 'depth', 'extension_factor', 'supercell', 'host_spacegroup', 'name']:
        val = row.get(key)
        if val is not None:
            meta[key] = val
    meta['db_id'] = str(row.id)
    return meta


def is_complete(atoms: Optional[Atoms], props: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    if atoms is None or len(atoms) == 0:
        return False
    for k in ['eform', 'spin']:
        if k not in props:
            return False
    for k in ['host', 'dopant', 'defecttype', 'converged']:
        if k not in meta:
            return False
    return True


def process_with_threshold(db, conv2_threshold: Optional[float], output_dir: Path, tag: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    在给定阈值下处理所有记录，返回数据与统计信息，并保存中间输出。
    tag 用于区分输出文件名（例如 strict_t1e-4 / relaxed_t1e-3 / relaxed_noconv2）。
    """
    stats = {
        'total_records': 0,
        'converged_records': 0,
        'passed_conv2_records': 0,
        'complete_records': 0,
        'error_records': 0,
    }
    data: List[Dict[str, Any]] = []

    # 分布统计
    eforms: List[float] = []
    spins: List[float] = []
    defecttypes: Counter = Counter()
    hosts: Counter = Counter()

    for row in db.select():
        stats['total_records'] += 1
        try:
            if not validate_convergence(row, conv2_threshold):
                continue
            stats['converged_records'] += 1
            if conv2_threshold is None:
                stats['passed_conv2_records'] += 1  # 视为全部通过
            else:
                conv2 = row.get('conv2')
                if conv2 is not None and abs(float(conv2)) <= float(conv2_threshold):
                    stats['passed_conv2_records'] += 1

            atoms = extract_atoms(row)
            props = extract_properties(row)
            meta = extract_metadata(row)
            if not is_complete(atoms, props, meta):
                continue

            record = {
                'id': row.id,
                'atoms': atoms,
                'properties': props,
                'metadata': meta,
                'formula': atoms.get_chemical_formula() if atoms is not None else None,
                'num_atoms': len(atoms) if atoms is not None else None,
            }
            data.append(record)
            stats['complete_records'] += 1

            # 收集分布
            eforms.append(props['eform'])
            spins.append(props['spin'])
            defecttypes.update([meta['defecttype']])
            hosts.update([meta['host']])
        except Exception:
            stats['error_records'] += 1
            continue

    # 保存数据与统计
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"complete_data_{tag}.pkl", 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        'stats': stats,
        'distributions': {
            'eform': {
                'count': len(eforms),
                'mean': float(np.mean(eforms)) if eforms else None,
                'std': float(np.std(eforms)) if eforms else None,
                'min': float(np.min(eforms)) if eforms else None,
                'max': float(np.max(eforms)) if eforms else None,
            },
            'spin': {
                'count': len(spins),
                'mean': float(np.mean(spins)) if spins else None,
                'std': float(np.std(spins)) if spins else None,
                'min': float(np.min(spins)) if spins else None,
                'max': float(np.max(spins)) if spins else None,
            },
            'defecttype_counts': dict(defecttypes),
            'host_counts': dict(hosts),
        },
        'tag': tag,
        'conv2_threshold': None if conv2_threshold is None else float(conv2_threshold),
    }

    with open(output_dir / f"metadata_{tag}.pkl", 'wb') as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data, summary


def render_report(reports: List[Dict[str, Any]], output_path: Path):
    lines: List[str] = []
    lines.append("# IMP2D 收敛阈值对比报告\n")
    for rep in reports:
        tag = rep.get('tag')
        thr = rep.get('conv2_threshold')
        stats = rep['stats']
        dist = rep['distributions']
        lines.append(f"## 方案: {tag} (conv2阈值: {thr})\n")
        lines.append("- 记录总数: " + str(stats['total_records']))
        lines.append("- 收敛记录数(converged=True): " + str(stats['converged_records']))
        lines.append("- 通过conv2阈值记录数: " + str(stats['passed_conv2_records']))
        lines.append("- 完整记录数(Atoms/eform/spin齐全): " + str(stats['complete_records']))
        lines.append("- 处理错误记录数: " + str(stats['error_records']))
        lines.append("")
        lines.append("- eform分布: count={} mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
            dist['eform']['count'],
            dist['eform']['mean'] if dist['eform']['mean'] is not None else float('nan'),
            dist['eform']['std'] if dist['eform']['std'] is not None else float('nan'),
            dist['eform']['min'] if dist['eform']['min'] is not None else float('nan'),
            dist['eform']['max'] if dist['eform']['max'] is not None else float('nan'),
        ))
        lines.append("- spin分布: count={} mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
            dist['spin']['count'],
            dist['spin']['mean'] if dist['spin']['mean'] is not None else float('nan'),
            dist['spin']['std'] if dist['spin']['std'] is not None else float('nan'),
            dist['spin']['min'] if dist['spin']['min'] is not None else float('nan'),
            dist['spin']['max'] if dist['spin']['max'] is not None else float('nan'),
        ))
        lines.append("- 缺陷类型计数(Top10): ")
        top_defects = sorted(dist['defecttype_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append("  " + ", ".join([f"{k}:{v}" for k, v in top_defects]))
        lines.append("- 基体材料计数(Top10): ")
        top_hosts = sorted(dist['host_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append("  " + ", ".join([f"{k}:{v}" for k, v in top_hosts]))
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')


def parse_thresholds(raw_list: List[str]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for s in raw_list:
        s_low = s.strip().lower()
        if s_low in ("none", "disable", "off"):
            out.append(None)
        else:
            out.append(float(s_low))
    return out


def main():
    parser = argparse.ArgumentParser(description="IMP2D 数据解析（可调收敛阈值）")
    parser.add_argument('--db-path', type=str, default=None, help='数据库文件路径（可选）')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='输出目录')
    parser.add_argument('--thresholds', nargs='+', default=['1e-4', '1e-3', 'none'], help='conv2阈值列表，如 1e-4 1e-3 none')
    args = parser.parse_args()

    db_path = find_db_path(args.db_path)
    if not db_path:
        logger.error('找不到 imp2d.db 数据库文件，请将 imp2d.db 放置在 data/raw 或项目根目录下')
        sys.exit(1)

    logger.info(f"正在连接数据库: {db_path}")
    db = connect(str(db_path))

    output_dir = Path(args.output_dir)
    thresholds = parse_thresholds(args.thresholds)

    reports: List[Dict[str, Any]] = []
    for thr in thresholds:
        tag = (
            f"strict_t{thr:.0e}" if thr is not None and np.isfinite(thr) and thr == 1e-4 else
            f"relaxed_t{thr:.0e}" if thr is not None and np.isfinite(thr) else
            "relaxed_noconv2"
        )
        logger.info(f"开始处理方案: {tag}")
        data, summary = process_with_threshold(db, thr, output_dir, tag)
        reports.append(summary)
        logger.info(f"完成方案: {tag} | 完整记录数: {summary['stats']['complete_records']}")

    report_path = Path('scripts/relaxed_comparison_report.md')
    render_report(reports, report_path)
    logger.info(f"对比报告已生成: {report_path}")