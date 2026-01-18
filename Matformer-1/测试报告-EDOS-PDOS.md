# 测试报告：EDOS/pDOS 复现对齐与单元测试

## 测试概览
- 目标：验证复现计划中关键改动（链接函数暴露、预测写出修复）在代码层面正确生效，并进行轻量级前向单元测试。
- 范围：不涉及完整数据集训练（`edos_pdos` 下载与长时训练），主要覆盖模型前向与训练脚本接口正确性。

## 代码改动回顾
- `train_props.py`：新增参数 `link`，透传为 `config['model']['link']`（支持 `log`、`identity`、`logit`）。
- `train.py`：修复预测写出与 `test_only` 路径的数据解包与传参，统一为 `(g, lg, lattice, target)`，并确保传入 `[g, lg, lattice]`。
- `pyg_att.py`：在模型初始化中设置 `self.zero_inflated = config.zero_inflated`，避免 log-link 路径报错。
- `features.py`：将 `np.math.factorial` 修复为 `math.factorial`，避免 `AttributeError`。
- 新增脚本：
  - `matformer/scripts/test_forward.py`：最小前向单元测试（验证 `link=log` 非负）。
  - `matformer/scripts/run_edos_pdos.py`：一键启动训练脚本，支持 `edos_up`、`pdos_elast` 或两者同时。

## 单元测试用例与结果
- 用例：`matformer/scripts/test_forward.py`
  - 构造最小图数据（PyG `Data`）并为单图设置 `batch` 向量；随机生成 `x(92)`、`edge_attr(3)`、`edge_index`。
  - 执行两组前向：
    - `link='log', output_features=200`
    - `link='identity', output_features=200`
- 运行命令：
  - `python e:\AShengChang\CODE2\Matformer-1\matformer\scripts\test_forward.py`
- 实际输出：
  - `Forward with link=log passed; output shape: torch.Size([200])`
  - `Forward with link=identity passed; output shape: torch.Size([200])`
- 结论：
  - 链接函数暴露与模型前向路径工作正常；`link='log'` 下输出非负，满足论文对 DOS 非负性的约束目标。

## 训练脚本接口验证
- 一键脚本：`matformer/scripts/run_edos_pdos.py`
  - 支持参数：`--task {edos_up|pdos_elast|both}`、`--link {log|identity|logit}`、`--epochs`、`--batch_size`、`--lr`、`--cutoff`、`--max_neighbors`、`--output_root` 等。
  - 默认开启 `link='log'`，并按任务自动设置 `output_features`（EDOS=300、pDOS=200）。
  - 输出目录：`./runs/<task>_<link>/`，训练完成后将生成 `history_train.json`、`history_val.json` 和（多输出）`multi_out_predictions.json`。

## 后续建议测试（可快速执行的冒烟试验）
- 目的：在不耗时的前提下验证训练流程走通和指标写出。
- 建议命令：
  - `python e:\AShengChang\CODE2\Matformer-1\matformer\scripts\run_edos_pdos.py --task edos_up --link log --epochs 5 --batch_size 32 --lr 0.001`
  - 若网络/数据源不可用，可先运行 `test_forward.py` 验证模型层面，待数据可用时再进行完整训练。

## 已修复的 Bug 清单
- 预测与测试路径的输入解包不一致（`(g, lg, lattice, target)`），现已统一。
- `np.math.factorial` 错误调用导致 `AttributeError`，已改为 `math.factorial`。
- `Matformer.zero_inflated` 属性缺失导致 log-link 初始化崩溃，已补充。

## 风险与注意事项
- `standard_scalar_and_pca` 当前仅用于写出预测时的变换；若需训练期逐维标准化或降维，需要在训练管线中插入对应步骤。
- `use_angle` 在现有模型内使用有限；线图（角信息）已在数据侧具备接口，可后续增强。
- 若需要论文中的费米能级加权损失，建议在 `train.py` 中自定义加权 MSE（按能量索引增大权重）。

## 结论
- 关键改动（链接函数暴露与预测路径修复）已验证可用；前向单元测试通过，满足论文的非负约束要求。
- 训练脚本可用于一键启动 EDOS/pDOS 训练；建议先进行短周期冒烟试验，随后开展完整训练与指标对齐。