import sqlite3
import json
import os
import sys

def inspect_ase_db(db_path):
    if not os.path.exists(db_path):
        print(f"错误: 找不到数据库文件 {db_path}")
        return

    print(f"正在探测数据库: {os.path.basename(db_path)}")
    print("-" * 50)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 检查标准 ASE 字段 (Standard Columns)
        print("\n[1] 标准 ASE 字段 (表 'systems'):")
        cursor.execute("PRAGMA table_info(systems)")
        columns = cursor.fetchall()
        
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            
            # 检查该列是否有非空数据
            cursor.execute(f"SELECT 1 FROM systems WHERE {col_name} IS NOT NULL LIMIT 1")
            has_data = cursor.fetchone() is not None
            
            status = "✓ 已填充" if has_data else "○ 空/全Null"
            print(f"  - {col_name:<25} ({col_type}): {status}")

        # 2. 检查元数据键 (Key-Value Pairs)
        print("\n[2] 元数据键 (自定义 Key-Value Pairs):")
        metadata_keys = set()
        
        # 从辅助表中获取 (速度快)
        for table, dtype in [('text_key_values', 'TEXT'), ('number_key_values', 'NUMBER')]:
            try:
                cursor.execute(f"SELECT DISTINCT key FROM {table}")
                keys = [row[0] for row in cursor.fetchall()]
                for k in keys: metadata_keys.add((k, dtype))
            except sqlite3.OperationalError:
                pass 

        # 如果辅助表为空，尝试从 systems 表采样 (备用方案)
        if not metadata_keys:
             cursor.execute("SELECT key_value_pairs FROM systems LIMIT 100")
             rows = cursor.fetchall()
             for row in rows:
                if row[0]:
                    data = json.loads(row[0])
                    for k, v in data.items():
                        dtype = 'NUMBER' if isinstance(v, (int, float)) else 'TEXT'
                        metadata_keys.add((k, dtype))

        # 打印结果
        if metadata_keys:
            for key, dtype in sorted(list(metadata_keys)):
                print(f"  - {key:<25} (类型: {dtype})")
        else:
            print("  未发现元数据键")

        # 3. 统计总行数
        cursor.execute("SELECT count(*) FROM systems")
        total_rows = cursor.fetchone()[0]
        print("-" * 50)
        print(f"总行数: {total_rows}")
        print(f"元数据键总数: {len(metadata_keys)}")

        conn.close()

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 使用命令行参数或默认路径
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
    else:
        # 默认路径
        db_file = "/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/imp2d.db"
    
    inspect_ase_db(db_file)
