#!/bin/bash
#SBATCH -J wly_preprocess
#SBATCH -p 9a14a
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# 按你的超算环境加载 Python（示例）
# module purge
# module load cuda/12.1
# source activate your_env

python -V

python pt_to_pickle.py --input_dir dataset_imp2d --output cleaned_dataset.pkl
python process_graphs.py --input cleaned_dataset.pkl --output processed_dataset_with_graphs.pkl --cutoff 5.0
python filter_dataset.py --input processed_dataset_with_graphs.pkl --output final_dataset.pkl --min -20 --max 20
