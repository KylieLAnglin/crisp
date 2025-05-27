#!/bin/bash
#SBATCH --job-name=llama_pipeline_05_26_2025
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kylie.anglin@uconn.edu

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Capture pipeline errors

echo "===== 🧠 Starting pipeline on: $(hostname) at $(date) ====="

# ------------------ ENV SETUP ------------------
echo -e "\n🔧 [1/5] Loading Python and activating environment"
module load python/3.12.2
source /scratch/kla21002/kla21002/ollama_env_2025_05_22/bin/activate
export PYTHONPATH=/scratch/kla21002/kla21002/crisp:$PYTHONPATH
cd /scratch/kla21002/kla21002/crisp
mkdir -p logs

echo -e "\n🧠 Memory and GPU Info:"
free -h
nvidia-smi

# ------------------ START OLLAMA ------------------
echo -e "\n🚀 [2/5] Starting Ollama instance and server..."
apptainer instance start --nv /scratch/kla21002/kla21002/ollama ollama_instance
apptainer exec instance://ollama_instance ollama serve > ollama_server_$SLURM_JOB_ID.log 2>&1 &

echo "🕒 Waiting for Ollama server to become ready..."
sleep 10

echo "🔍 Checking if Ollama server is reachable..."
curl --silent http://localhost:11434 | grep -q "Ollama" \
    && echo "✅ Ollama server is running." \
    || { echo "❌ Ollama server did not start properly. Exiting."; exit 1; }

# ------------------ RUN PIPELINE ------------------
run_step() {
    SCRIPT=$1
    echo -e "\n📄 Running: $SCRIPT at $(date)\n------------------------------------------------------------"
    python3 "$SCRIPT" || { echo "❌ Script failed: $SCRIPT" >&2; exit 1; }
    echo -e "\n✅ Completed: $SCRIPT at $(date)"
}

echo -e "\n🏃 [3/5] Running classification pipeline...\n"
run_step crisp/1_baseline_prompt/01_baseline_train.py
# Uncomment steps as needed
# run_step crisp/1_baseline_prompt/02_baseline_dev.py
# run_step crisp/1_baseline_prompt/03_fewshot_train.py
# run_step crisp/1_baseline_prompt/04_fewshot_dev.py
# run_step crisp/2_APE/01_ape_train.py
# run_step crisp/2_APE/02_ape_dev.py
# run_step crisp/2_APE/03_APE_fewshot_train.py
# run_step crisp/2_APE/04_APE_fewshot_dev.py
# run_step crisp/3_persona/01_persona_train.py
# run_step crisp/3_persona/02_persona_dev.py
# run_step crisp/3_persona/03_persona_fewshot_dev.py
# run_step crisp/4_cot/01_cot_zero_dev.py
# run_step crisp/4_cot/02_cot_fewshot_train.py
# run_step crisp/4_cot/03_cot_fewshot_dev.py
# run_step crisp/explanation/01_explanation_fewshot_train.py
# run_step crisp/explanation/02_explanation_fewshot_dev.py

# ------------------ END ------------------
echo -e "\n✅ [5/5] Pipeline complete at $(date)"