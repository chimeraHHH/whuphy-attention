#!/bin/bash
#SBATCH -J wly_eval
#SBATCH -p gpu/a100x4
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# module purge
# module load cuda/12.1
# source activate your_env

python -V
python -c "import torch; print('cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count());"

python tets_all.py --ckpt checkpoints/latest_model.pth --plot_path test_result_plot.png --num_workers 8 --pin_memory
