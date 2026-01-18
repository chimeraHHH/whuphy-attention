# 组会汇报：EDOS/pDOS 复现进度与结果（Wu et al., 2025）

## 背景与目标
- 目标：复现论文中针对能量分辨的总态密度（EDOS）与投影态密度（pDOS）预测，并生成 Figure 4 风格的“目标 vs 预测”谱形对比图。
- 框架：使用本地 `Matformer-1` 项目（周期图 Transformer，集成元素属性、距离 RBF、线图角特征）。

## 数据与环境
- 数据集：`edos_pdos`（JARVIS 数据源，通过项目脚本自动加载）。
- 设备：CPU（本次快速重训），后续建议使用 GPU 加速。
- 工作目录：`E:\AShengChang\CODE2\Matformer-1`

## 训练设置
- 模型：`matformer`，PyG 输入、线图启用，`conv_layers=5`，`node_features=128`。
- 关键配置：
  - 链接函数：先验尝试 `log`，重训改为 `identity`，避免与逐 bin 标准化的目标域不匹配。
  - 拆分比例（重训）：`train≈10% / val≈5% / test≈5%`。
  - 轮次与批量（重训）：`epochs=10`、`batch_size=16`、`lr=1e-3`。

## 改动与诊断
- 问题定位：`link='log'` 将网络输出强制非负，与逐 bin 标准化后的标签（含负值）不匹配，导致收敛到近似常数低幅输出，谱形拟合极差。
- 解决策略：
  - 改用 `link='identity'`，先保证输出域与目标一致；后续如需非负展示，在绘图端做裁剪处理。
  - 修复训练历史写出中的 JSON 序列化问题，确保验证曲线与指标稳定输出。

## 结果指标（按能量 bin 平均 MAE）
- EDOS（300 维）：`best_val_mae ≈ 4.598 @ epoch 9`（路径：`./runs_identity/edos_up_identity`）
- pDOS（200 维）：`best_val_mae ≈ 0.778 @ epoch 7`（路径：`./runs_identity/pdos_elast_identity`）
- 注：CPU 短训与较小子集限制了最终精度；与快速试训（`link=log`）相比已明显改善，但仍有优化空间。

## 图像展示（Figure 4 风格）
- 生成脚本：`matformer/scripts/plot_figure4.py`
- 本次输出：`./runs_identity/figure4.png`

![Figure 4 复现](./runs_identity/figure4.png)

## 拟合问题与改进方向
- 训练强度不足：多输出谱拟合需要更长训练与更大数据覆盖，建议 `epochs=100–300`、`batch_size=64`（GPU）。
- 损失权重：在收敛后再启用费米加权（`weighted_loss=True`、`fermi_sigma≈0.05` 或按维度比例），突出关键能量区拟合。
- 邻域与角特征：保持线图角特征，适度调参 `cutoff=6–8`、`max_neighbors=12–20`，平衡几何信息与计算量。
- 非负展示：如需与论文图一致的非负谱，可在后处理绘图阶段对预测做 `clip(pred, 0, None)`。

## 下一步计划（建议 GPU）
- 长训（EDOS）：
  - `python matformer/scripts/run_edos_pdos.py --task edos_up --epochs 200 --batch_size 64 --lr 0.001 --link identity --output_root .\runs_strong`
- 长训（pDOS）：
  - `python matformer/scripts/run_edos_pdos.py --task pdos_elast --epochs 200 --batch_size 64 --lr 0.001 --link identity --output_root .\runs_strong`
- 训练完成后：用 `plot_figure4.py` 重新生成图像，对齐论文中的关键材料与能量窗口，并报告费米附近窗口 MAE。

## 本次复现实验命令记录
- 快速试训（`log`，失败表现，用于诊断）：
  - `python matformer\scripts\run_edos_pdos.py --task both --epochs 5 --batch_size 8 --lr 0.001 --link log --weighted_loss --fermi_sigma 0.1 --num_workers 0 --train_ratio 0.02 --val_ratio 0.01 --test_ratio 0.01 --output_root .\runs_quick`
- 重训（`identity`，改进表现）：
  - `python matformer\scripts\run_edos_pdos.py --task both --epochs 10 --batch_size 16 --lr 0.001 --link identity --num_workers 0 --train_ratio 0.10 --val_ratio 0.05 --test_ratio 0.05 --output_root .\runs_identity`
- 指标统计与图：
  - `python matformer\scripts\plot_figure4.py --edos_dir .\runs_identity\edos_up_identity --pdos_dir .\runs_identity\pdos_elast_identity --output .\runs_identity\figure4.png`

## 结论
- 使用 `identity` 链接后谱形拟合显著改善，但受限于 CPU 短训与子集规模，EDOS 与 pDOS 的绝对指标仍偏高。
- 建议转入 GPU、延长训练、引入费米加权与更合理邻域配置，以进一步贴近论文结果并稳定复现 Figure 4 的峰位与形状。