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

# ------------------ ENV SETUP ------------------
echo -e "\n🔧 [1/5] Loading Python and activating environment"
module load python/3.12.2
source /scratch/kla21002/kla21002/ollama_env_2025_05_22/bin/activate
export PYTHONPATH=/scratch/kla21002/kla21002/crisp:$PYTHONPATH
cd /scratch/kla21002/kla21002/crisp

# ------------------ START OLLAMA ------------------
echo -e "\n🚀 [2/5] Starting Ollama instance and server..."
apptainer instance start --nv /scratch/kla21002/kla21002/ollama ollama_instance
apptainer exec instance://ollama_instance ollama serve &

echo "🕒 Waiting for Ollama server to become ready..."
sleep 10
apptainer exec instance://ollama_instance ollama run llama3 "Is the following a negative core belief? Respond with only Yes or No. Text: I am worthless."
