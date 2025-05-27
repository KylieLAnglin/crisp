#!/bin/bash
#SBATCH --job-name=llama_pipeline_05_26_2025
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kylie.anglin@uconn.edu

set -e  # Exit if any command fails
set -o pipefail  # Catch failures in pipelines

echo "===== 🧠 Starting pipeline on: $(hostname) at $(date) ====="
