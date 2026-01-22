#!/bin/bash
#SBATCH -J wly_train
#SBATCH -p gpu/a100x4
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG=warn

# 按你的超算环境加载 Python/CUDA（示例）
# module purge
# module load cuda/12.1
# source activate your_env

python -V
python -c "import torch; print('cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count());"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train.py \
  --distributed \
  --backend nccl \
  --data final_dataset.pkl \
  --features atom_features.pth \
  --output_dir checkpoints \
  --batch_size 16 \
  --epochs 25 \
  --num_workers 16 \
  --pin_memory
