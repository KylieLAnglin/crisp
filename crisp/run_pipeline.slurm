#!/bin/bash
#SBATCH --job-name=llama_pipeline
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kylie.anglin@uconn.edu

set -e  # <-- Stop immediately if any command fails

# Load Python and activate environment
module load python/3.12.2
source /scratch/kla21002/kla21002/crisp/ollama_env_2025_05_22/bin/activate
export PYTHONPATH=/home/kla21002/crisp:$PYTHONPATH

cd /home/kla21002/crisp

# Run pipeline
echo "Starting baseline training..."
python3 crisp/1_baseline_prompt/01_baseline_train.py

echo "Starting dev evaluation..."
python3 crisp/1_baseline_prompt/02_baseline_dev.py

echo "Starting APE training..."
python3 crisp/1_baseline_prompt/03_ape_training.py

echo "Pipeline complete."