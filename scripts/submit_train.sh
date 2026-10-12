#!/bin/bash
#SBATCH --job-name=ehr-train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load environment modules
module load python/3.11 gcc arrow/23.0.1

# Set up virtual environment in fast local scratch storage ($SLURM_TMPDIR)
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install project dependencies from Compute Canada wheel cache
pip install --no-index -r requirements.txt

# Weights & Biases configuration (strictly offline for compute nodes)
export WANDB_API_KEY="${WANDB_API_KEY}"
wandb offline

# Ensure logs directory exists
mkdir -p logs

# Run complete training pipeline
python scripts/train.py

