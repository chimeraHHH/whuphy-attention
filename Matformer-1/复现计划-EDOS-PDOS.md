# Graph Transformer 复现计划（EDOS/pDOS）

基于论文《Graph Transformer Model Integrating Physical Features for Projected Electronic Density of States（Wu et al., 2025）》与现有 Matformer 项目，制定一份可执行复现方案，用于训练与评估能量分辨的总态密度（EDOS）与投影态密度（pDOS）向量。

## 论文要点
- 目标：预测固定能量网格上的 DOS 向量（多输出）。
- 模型：图 Transformer（节点/边注意力），强化物理特征（元素属性、距离 RBF、角信息/线图）。
- 输出：多维向量，建议使用非负约束（如 log 链接）。
- 训练：使用 MAE/MSE 等回归损失；关注费米能级附近的拟合质量。

## 复现目标
- 使用本项目的 `edos_pdos` 数据集与多输出能力，训练出 EDOS/pDOS 的能量分辨预测模型。
- 与论文方法对齐：图 Transformer、物理特征、线图、多输出回归、可选非负链接函数。

## 环境准备
- Python `>=3.9`；可用 CUDA 则建议开启。
- 安装依赖：
  - 在项目根目录执行：`pip install -e .`
  - 或：`pip install torch torch-geometric scikit-learn ignite jarvis-tools`
- 工作目录：`e:\AShengChang\CODE2\Matformer-1`

## 数据准备
- 数据集：`edos_pdos`（JARVIS 来源），项目已内置加载逻辑。
- 输出维度：
  - `prop='edos_up'` → `output_features=300`
  - `prop='pdos_elast'` → `output_features=200`
- 训练/验证/测试拆分与 ID 字段由 `matformer/data.py` & `matformer/train.py` 自动处理。

## 模型配置要点
- 模型名必须为 `matformer`（训练函数仅注册该名）。
- 图输入：`pyg_input=True`；默认启用线图（`line_graph=True`）。
- 多输出维度：通过 `output_features` 与能量网格长度对齐。
- 链接函数：支持 `link`（identity/log/logit）。DOS 建议 `log`（非负）。

## 训练流程（示例）
- 训练 EDOS（300 维）：
  - `python -c "from matformer.train_props import train_prop_model; train_prop_model(dataset='edos_pdos', prop='edos_up', name='matformer', pyg_input=True, n_epochs=500, batch_size=64, learning_rate=0.001, use_lattice=True, use_angle=False, output_dir='./runs/edos_up_300')"`
- 训练 pDOS（200 维）：
  - `python -c "from matformer.train_props import train_prop_model; train_prop_model(dataset='edos_pdos', prop='pdos_elast', name='matformer', pyg_input=True, n_epochs=500, batch_size=64, learning_rate=0.001, use_lattice=True, use_angle=False, output_dir='./runs/pdos_elast_200')"`
- 可调超参：`cutoff`、`max_neighbors`、`weight_decay`、`scheduler` 等通过 `train_prop_model` 参数传入。

## 评估与产出
- 输出目录（`output_dir`）下生成：
  - `history_train.json` 与 `history_val.json`：各 epoch 的 `loss` 与 `mae`。
  - 多输出任务：`multi_out_predictions.json`（含测试集每样本的 `id`、`target`、`predictions`）。
- 早停趋势检查：
  - `python e:\AShengChang\CODE2\Matformer-1\matformer\scripts\early_stopping_checker.py`

## 关键开关建议（对齐论文）
- 非负约束：将 `model.link` 设为 `log`。目前需在配置构造处显式传入（`train_props.py` 未直接暴露该参数）。
- 特征增强：
  - 距离 RBF 展开与线图批处理已在 `graphs.py` / `data.py` 实现。
  - `use_angle` 参数在模型内使用有限，但线图接口已具备，可后续增强角度特征。
- 标准化：现有 `standard_scalar_and_pca` 主要用于推理输出的写出时变换，若需训练期标准化/降维需在训练管线中插入相应步骤。

## 可选增强改动（贴近论文）
- 费米能级加权损失：在 `matformer/train.py` 自定义 criterion，对能量索引加权（费米附近权重更高）。
- 链接函数暴露：在 `train_props.py` 增加 `model.link` 参数透传（如 `log`）。
- 训练期标准化：为每个能量 bin 做逐维标准化，并在评估时反变换。

## 时间线
- 第 0–1 天：环境与数据连通性；小规模试训（`n_epochs=50`）。
- 第 2–4 天：完整训练（`n_epochs=500`）跑通 `edos_up` 与 `pdos_elast`；收集指标与预测。
- 第 5–6 天：与论文指标对齐、超参网格搜索（`link=log`、`cutoff/max_neighbors`）。
- 第 7–9 天：可选增强（加权损失、训练期标准化）对比改动前后表现。

## 完成标志
- 正确生成 `history_val.json` 与 `multi_out_predictions.json`，指标稳定且非负约束有效。
- 费米能级附近的形状/峰位拟合合理；引入物理特征后优于基线。

## 注意点
- `zero_inflated` 未在训练管线使用，若需零膨胀处理需二次开发。
- 单输出分支旧逻辑与 DOS 训练无关，推荐仅使用多输出流程。

---
如需，我可以提供一个最小启动脚本（同时设置 `model.link='log'`）以一键跑通 `edos_up` 与 `pdos_elast`。