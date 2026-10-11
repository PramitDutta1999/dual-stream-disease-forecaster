#!/bin/bash
#SBATCH --account=def-erangauk-ab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=smoke-%N-%j.out

# 1. Compute Canada Environment Setup
module load python/3.11 gcc arrow/23.0.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

# 2. Weights & Biases Offline Mode
export WANDB_API_KEY="${WANDB_API_KEY}"
wandb offline

# 3. Execution (Smoke Test)
echo "Testing Clinical Baseline..."
python scripts/train.py --mode clinical --max-batches 10

echo "Testing Multimodal Baseline..."
python scripts/train.py --mode multimodal --max-batches 10

echo "Smoke Test Complete."