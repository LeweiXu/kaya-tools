#!/bin/bash --login
# A hand-written job script. `kaya submit example.sbatch -- some args` runs it from the
# project's remote folder, so relative paths below are relative to that. Flags given to
# kaya (--time, --gres, ...) override the #SBATCH lines; nothing else from [slurm] is added.
#SBATCH --job-name=example
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load Anaconda3/2024.06
module load cuda/12.6.3

echo "host $(hostname), args: $*"
nvidia-smi
