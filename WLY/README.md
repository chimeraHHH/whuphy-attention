# WLY（CrystalTransformer）运行说明（Windows/超算）

## 目录里已有的数据
- `dataset_imp2d/*.pt`：训练目标是 `y=eform`（formation energy），可直接作为数据源
- `atom_features.pth`：原子特征查表（训练默认使用）

## 依赖
- 训练/推理：`torch` `numpy` `tqdm`
- 数据预处理（构图/角度）：`ase`
- 从 `dataset_imp2d/*.pt` 读取 Data：`torch_geometric`
- 评估画图：`matplotlib`

## WHU 超算基础操作（SSH/传输/Slurm）
连接与注意事项（登录节点/传输节点、VPN、禁止 sudo、不要在登录节点跑计算）可参考官方指南：<https://docs.hpc.whu.edu.cn/>

### 连接登录节点（SSH）
登录节点：
- `swarm01.whu.edu.cn`
- `swarm02.whu.edu.cn`
- `swarm03.whu.edu.cn`

示例：
```bash
ssh 你的用户名@swarm01.whu.edu.cn
```

### 文件传输（建议走传输节点）
传输节点：
- `swarm-xfe.whu.edu.cn`

示例（上传）：
```bash
scp -r whuphy-attention 你的用户名@swarm-xfe.whu.edu.cn:~/
```

### Slurm 作业提交（必须 sbatch）
- 提交：`sbatch yourjob.sbatch`
- 查看队列：`squeue`
- 取消作业：`scancel JobID`
- Windows 写的 sbatch 脚本上传后建议 `dos2unix yourjob.sbatch`，避免 CRLF 换行导致脚本格式不符

## 数据预处理（.pt → .pkl → 构图 → 过滤）
在 `whuphy-attention/WLY` 目录执行：

```powershell
python pt_to_pickle.py --input_dir dataset_imp2d --output cleaned_dataset.pkl --limit 2000
python process_graphs.py --input cleaned_dataset.pkl --output processed_dataset_with_graphs.pkl --cutoff 5.0
python filter_dataset.py --input processed_dataset_with_graphs.pkl --output final_dataset.pkl --min -20 --max 20
```

说明：
- `--limit` 用于先快速跑通流程；去掉 `--limit` 会处理全部 `dataset_imp2d`（数量较多，构图会明显耗时）。

## 训练
```powershell
python train.py --data final_dataset.pkl --features atom_features.pth --output_dir checkpoints --epochs 25 --batch_size 16
```

断点续训（默认自动使用 `checkpoints/latest_model.pth`，存在即续训）：
```powershell
python train.py --data final_dataset.pkl --features atom_features.pth --output_dir checkpoints
```

每个 epoch 结束后退出（配合外部脚本重启）：
```powershell
python train.py --data final_dataset.pkl --features atom_features.pth --output_dir checkpoints --restart_each_epoch
```

常用性能参数（多核/多进程加载，GPU 推荐开 pin_memory）：
```powershell
python train.py --data final_dataset.pkl --features atom_features.pth --output_dir checkpoints --num_workers 16 --pin_memory
```

两卡训练（torchrun/DDP，超算单节点 2 卡 A100 示例）：
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --distributed --backend nccl \
  --data final_dataset.pkl --features atom_features.pth --output_dir checkpoints \
  --num_workers 16 --pin_memory
```

## 推理（简单验证前 N 个样本）
```powershell
python predict.py --ckpt checkpoints/latest_model.pth --data final_dataset.pkl --features atom_features.pth --num 5
```

## 评估（test split + parity plot + 最大误差样本）
```powershell
python tets_all.py --ckpt checkpoints/latest_model.pth --plot_path test_result_plot.png
```

## 超算（SLURM）提交方式
脚本在 `hpc/` 目录：
- `preprocess_cpu.slurm`：CPU 预处理（默认 `-p 9a14a`；如需改用 `hpib/pub/sd530/h240/xh321` 等分区，修改 `#SBATCH -p ...` 或提交时用 `sbatch -p 分区名` 覆盖）
- `train_gpu_a100x2.slurm`：GPU 两卡训练（默认 `-p gpu/a100x4`；资源请求按集群规则可改 `--gres=gpu:2`）
- `eval_gpu.slurm`：单卡评估（生成 parity plot）

提交示例：
```bash
cd /path/to/whuphy-attention/WLY
sbatch hpc/preprocess_cpu.slurm
sbatch hpc/train_gpu_a100x2.slurm
sbatch hpc/eval_gpu.slurm
```
