#!/bin/bash
#SBATCH --account=def-erangauk-ab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%N-%j.out

# 1. Compute Canada Environment Setup
module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

# 2. Weights & Biases Offline Mode
export WANDB_API_KEY="${WANDB_API_KEY}"
wandb offline

# 3. Execution
echo "Starting Clinical Baseline Training..."
python scripts/train.py --mode clinical

echo "Starting Multimodal Baseline Training..."
python scripts/train.py --mode multimodal

echo "Job Complete."

