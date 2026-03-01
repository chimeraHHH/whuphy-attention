# 基准模型下一步方案（按《确立AI科研基准模型指南》）

## 1. 主线与边界（先统一工程，再做对比）

唯一主线：
- `mainline/` 负责统一入口与统一配置。
- `WLY/` 负责数据流水线与主干训练链路。
- `Matformer-1/` 作为主线可切换训练目标之一。

实验分支：
- `BJQ/`、`HYM/` 仅做探索和消融，不进入主基准结论。

硬约束：
1. 所有正式实验都从 `mainline/run.py` 启动。
2. 所有正式实验都从 `mainline/config.toml` 读取参数。
3. 不允许在脚本内写死本机路径、临时文件名、私有数据目录。

## 2. 基准阵列（梯队化）

按指南要求建立三层对照：
1. `CGCNN`：下限基线（局部距离）。
2. `ALIGNN`：强基线（局部距离+键角，Line Graph）。
3. `Ours(mainline)`：你们主线模型（后续引入全局注意力/虚拟节点等创新）。

## 3. 公平性铁律（不满足则结果作废）

1. 数据划分锁死：`80/10/10`，固定 `seed`，导出固定 ID 文件。
2. 标签与过滤一致：目标定义、清洗逻辑、异常值截断规则完全一致。
3. 指标与脚本一致：同一套 MAE/RMSE/R2 计算脚本。
4. 基线不混创新：跑 `CGCNN/ALIGNN` 时不注入“空位虚拟节点”等新机制。

## 4. 时间表（从 2026-03-02 开始）

### Sprint 1（2026-03-02 至 2026-03-08）：可复现实验底座

目标：
- 打通“同一入口、同一配置、同一 split”的可复现实验流程。

任务：
1. 生成并冻结 `split v1`：
   - `train_ids.json` / `val_ids.json` / `test_ids.json`
2. 建立 `toy=200` 样本冒烟测试集。
3. 主线先跑通 1 个模型的 toy 训练，验证 loss 正常下降。
4. 固化结果记录模板（含 commit/config/seed/hardware）。

验收：
- 同一 seed 连续两次运行，测试指标误差在可接受浮动内。
- 实验日志可追溯到唯一配置与唯一 split 文件。

### Sprint 2（2026-03-09 至 2026-03-15）：完整基线跑数

目标：
- 在同一 test IDs 上完成三模型可比结果。

任务：
1. 跑 `CGCNN` 全量训练与评估。
2. 在独立 conda 环境跑 `ALIGNN`（避免 DGL/PyG 冲突）。
3. 跑 `mainline` 当前模型。
4. 汇总单表：
   - MAE、RMSE、R2
   - Params
   - Max VRAM
   - 单样本推理时延（ms/sample）

验收：
- 三模型结果全部来自同一版本 split 和同一评估脚本。

### Sprint 3（2026-03-16 至 2026-03-22）：瓶颈定位与创新靶点

目标：
- 给“下一版模型创新”提供可量化靶点。

任务：
1. 做大超胞（100+ 原子）压力测试。
2. 记录 ALIGNN 的显存膨胀/OOM 边界。
3. 产出误差 Top-N 样本分析（按缺陷类型分桶）。

验收：
- 至少形成 2 条定量优化目标（如“显存下降 X%”“时延下降 Y%”）。

## 5. 与当前仓库对齐的执行命令

统一入口命令（已存在）：

```bash
python mainline/run.py --stage preprocess --target wly
python mainline/run.py --stage train --target wly
python mainline/run.py --stage train --target matformer
python mainline/run.py --stage all --target both
```

建议新增（本周内）：
1. `--stage split`：专门生成并锁定 split artifacts。
2. `--stage evaluate`：统一指标计算与表格导出。
3. `--baseline {cgcnn,alignn,ours}`：统一基线编排接口。

## 6. 交付物清单（你们结题/论文可直接用）

1. `mainline/baseline/splits/v1/*.json`
2. `mainline/baseline/manifests/data_manifest_v1.json`
3. `mainline/baseline/reports/toy_smoke_report.md`
4. `mainline/baseline/tables/benchmark_table_v1.csv`
5. `mainline/baseline/reports/stress_test.md`
6. `mainline/baseline/reports/error_analysis.md`

## 7. 48小时动作（马上执行）

1. 冻结 `seed + split + filter` 三件套，产出 `v1`。
2. 跑通 toy（200）并保存首个可复现实验记录。
3. 建立基准结果表头，先填运行资源指标（VRAM/时延），再填精度。
4. 拉起 ALIGNN 独立环境，先做 1 次 toy 验证，不直接上全量。
